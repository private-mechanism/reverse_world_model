import re, sys, os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from wam.models.hub_mixin import CompatiblePyTorchModelHubMixin
from wam.samples.noise_scheduler import NoiseTimestepSampler
from wam.models.mvwam.world_stack_registry import (
    VIDEO_BASE_MODEL_PATHS,
    build_world_stack,
    resolve_model_name,
    resolve_video_base_model_path,
)
from wam.runners.runner_video import (
    _video_inference_timestep_tensor,
    _video_train_diffusion_prep,
)

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler, DDPMScheduler, DPMSolverMultistepScheduler
from typing import Optional, Tuple

# 获取当前文件的绝对路径（包含文件名）
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
# ``wam/runners/`` → 上两级为 WAM 仓库根，与原先 ``models/`` 下 ``../demos`` 一致
_WAM_ROOT = Path(__file__).resolve().parents[2]
_DEMO_UNCOND_PT = str(_WAM_ROOT / "data" / "uncond.pt")


def _paths_equal_canonical(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return os.path.realpath(os.path.expanduser(a)) == os.path.realpath(os.path.expanduser(b))


def _is_empty_pretrained_path(value) -> bool:
    if value is None:
        return True
    s = str(value).strip()
    return s == "" or s.lower() in ("none", "null", "nil")


def _pick_pretrained_path(kwargs_value, config, attr_name: str):
    """kwargs 非空优先，否则读 ``config`` 上的同名字段（OmegaConf / dict）。"""
    if not _is_empty_pretrained_path(kwargs_value):
        return kwargs_value
    if config is None:
        return None
    if hasattr(config, "get"):
        cfg_v = config.get(attr_name, None)
    else:
        cfg_v = getattr(config, attr_name, None)
    if _is_empty_pretrained_path(cfg_v):
        return None
    return cfg_v


class FMPRunner(nn.Module,
                CompatiblePyTorchModelHubMixin,):

    def __init__(self,
                 *,
                 pred_horizon,
                 config,
                 img_cond_len,
                 img_pos_embed_config,
                 sample_fps=1,
                 pretrained_video_expert_path="none", #"/mnt/dataset/projs/pretrained_models/WoW-1-Wan-1.3B-2M-Diffusers",
                 pretrained_action_expert_path="none",
                 pretrained_wam_path=None,
                 dtype=torch.bfloat16,
                 model_name=None,
                 model_type=None,
                 video_base_model=None,
                 video_variant=None):
        """``model_name`` 优先（如 ``world_policy_causal_value``）；未给时用 ``model_type`` 做旧版映射。

        预训练路径：``pretrained_wam_path`` 与 ``pretrained_*_expert_path`` 可由 kwargs 或 ``config`` 提供；
        kwargs 非空时覆盖 config。若 ``pretrained_wam_path`` 有效则整网加载，否则分别加载两个 expert 微调权重。
        """
        mn = resolve_model_name(model_name, model_type)
        super(FMPRunner, self).__init__()
        vid = _pick_pretrained_path(pretrained_video_expert_path, config, "pretrained_video_expert_path")
        act = _pick_pretrained_path(pretrained_action_expert_path, config, "pretrained_action_expert_path")
        wam = _pick_pretrained_path(pretrained_wam_path, config, "pretrained_wam_path")
        vb_raw = _pick_pretrained_path(video_base_model, config, "VIDEO_BASE_MODEL")
        self.pretrained_video_expert_base_path = resolve_video_base_model_path(vb_raw)
        self.model = build_world_stack(
            mn,
            config=config,
            img_cond_len=img_cond_len,
            img_pos_embed_config=img_pos_embed_config,
            pretrained_video_expert_path=vid if vid is not None else "none",
            pretrained_action_expert_path=act if act is not None else "none",
            pretrained_wam_path=wam,
            dtype=dtype,
            pretrained_video_expert_base_path=self.pretrained_video_expert_base_path,
        )
        if video_variant is not None:
            self._video_expert_is_wan22 = (video_variant == "wan22")
        else:
            _wan22_root = VIDEO_BASE_MODEL_PATHS.get("Wan2.2-TI2V-5B-Diffusers")
            self._video_expert_is_wan22 = _wan22_root is not None and _paths_equal_canonical(
                self.pretrained_video_expert_base_path, _wan22_root
            )
        self._noise_video_channels = 48 if self._video_expert_is_wan22 else 16
        self.use_value = config.action_expert.get("use_value", False)
        self.dtype = dtype
        # Create the noise scheduler
        noise_scheduler_config = config['noise_scheduler']
        # default: beta distribution
        if noise_scheduler_config['type'] != 'ddpm':
            self.noise_scheduler_action = NoiseTimestepSampler(
                num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
                distribution_type=noise_scheduler_config['distribution_type'],
                alpha=noise_scheduler_config['alpha'],
                beta=noise_scheduler_config['beta'],
                noise_s=noise_scheduler_config['noise_s'],
            )
            self.noise_scheduler_sample_action = NoiseTimestepSampler(
                num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
                distribution_type=noise_scheduler_config['distribution_type'],
                alpha=noise_scheduler_config['alpha'],
                beta=noise_scheduler_config['beta'],
                noise_s=noise_scheduler_config['noise_s'],
            )
        elif noise_scheduler_config['type'] == 'ddpm':
            self.noise_scheduler_action = DDPMScheduler(
                num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
                beta_schedule=noise_scheduler_config['beta_schedule'],
                prediction_type=noise_scheduler_config['prediction_type'],
                clip_sample=noise_scheduler_config['clip_sample'],
            )
            self.noise_scheduler_sample_action = DPMSolverMultistepScheduler(
                num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
                beta_schedule=noise_scheduler_config['beta_schedule'],
                prediction_type=noise_scheduler_config['prediction_type'],
            )

        self.noise_scheduler_video = NoiseTimestepSampler(
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
            distribution_type="normal",
            shift=5.0
        )
        self.noise_scheduler_sample_video = UniPCMultistepScheduler.from_pretrained(
            self.pretrained_video_expert_base_path,
            subfolder='scheduler',
            torch_dtype=dtype,
            local_files_only=True
        )

        self.num_train_timesteps = noise_scheduler_config['num_train_timesteps']
        self.num_inference_timesteps = noise_scheduler_config['num_inference_timesteps']
        self.prediction_type = noise_scheduler_config['prediction_type']
        self.noise_scheduler_type = noise_scheduler_config['type']

        self.pred_horizon = pred_horizon
        self.action_dim = config['action_expert']['out_action_dim']
        self.state_token_dim = config['state_token_dim']
        self.video_pred_horizon = pred_horizon // sample_fps
        self.stage2_enable_reverse_ar_action = bool(config.get("stage2_enable_reverse_ar_action", False))
        self.stage2_key_action_loss_weight = float(config.get("stage2_key_action_loss_weight", 0.0))
        self.stage2_reverse_ar_action_loss_weight = float(
            config.get("stage2_reverse_ar_action_loss_weight", 0.0)
        )
        self.last_stage2_metrics = {}
        self.generation_mode = str(config.get("generation_mode", "diffusion")).strip().lower()
        if self.generation_mode not in ("diffusion", "reverse_ar"):
            raise ValueError(
                f"Unsupported generation_mode={self.generation_mode!r}; "
                "expected 'diffusion' or 'reverse_ar'."
            )
        self.stage3_predict_keyframe = bool(config.get("stage3_predict_keyframe", False))
        self.stage3_predict_key_action = bool(config.get("stage3_predict_key_action", False))
        self.stage3_replace_video_diffusion = bool(config.get("stage3_replace_video_diffusion", False))
        self.stage3_replace_action_diffusion = bool(config.get("stage3_replace_action_diffusion", False))
        self.stage3_video_ar_causal_attention = bool(config.get("stage3_video_ar_causal_attention", False))
        self.stage3_action_ar_causal_attention = bool(config.get("stage3_action_ar_causal_attention", True))
        self.stage3_keyframe_loss_weight = float(config.get("stage3_keyframe_loss_weight", 1.0))
        self.stage3_video_ar_loss_weight = float(config.get("stage3_video_ar_loss_weight", 1.0))
        self.stage3_key_action_loss_weight = float(config.get("stage3_key_action_loss_weight", 1.0))
        self.stage3_action_ar_loss_weight = float(config.get("stage3_action_ar_loss_weight", 1.0))
        self.stage3_use_coa_action_loss = bool(config.get("stage3_use_coa_action_loss", False))
        self.stage3_action_latent_loss_weight = float(config.get("stage3_action_latent_loss_weight", 0.1))
        self.stage3_action_latent_detach_target = bool(
            config.get("stage3_action_latent_detach_target", True)
        )
        self.stage3_use_scheduled_sampling = bool(config.get("stage3_use_scheduled_sampling", False))
        self.stage3_key_teacher_forcing_prob_start = float(
            config.get("stage3_key_teacher_forcing_prob_start", 1.0)
        )
        self.stage3_key_teacher_forcing_prob_end = float(
            config.get("stage3_key_teacher_forcing_prob_end", 0.0)
        )
        self.stage3_key_teacher_forcing_decay_steps = int(
            config.get("stage3_key_teacher_forcing_decay_steps", 20000)
        )
        self.last_stage3_metrics = {}
        self.stage4_full_coa_ar = bool(config.get("stage4_full_coa_ar", False))
        self.stage4_replace_video_diffusion = bool(config.get("stage4_replace_video_diffusion", False))
        self.stage4_replace_action_diffusion = bool(config.get("stage4_replace_action_diffusion", False))
        self.stage4_keyframe_loss_weight = float(config.get("stage4_keyframe_loss_weight", 1.0))
        self.stage4_video_ar_loss_weight = float(config.get("stage4_video_ar_loss_weight", 1.0))
        self.stage4_key_action_loss_weight = float(config.get("stage4_key_action_loss_weight", 1.0))
        self.stage4_action_ar_loss_weight = float(config.get("stage4_action_ar_loss_weight", 1.0))
        self.stage4_action_latent_loss_weight = float(config.get("stage4_action_latent_loss_weight", 0.1))
        self.stage4_action_latent_detach_target = bool(config.get("stage4_action_latent_detach_target", True))
        self.last_stage4_metrics = {}
        self.goal_conditioned_wam = bool(config.get("goal_conditioned_wam", False))
        self.predict_key_video = bool(config.get("predict_key_video", True))
        self.predict_key_action = bool(config.get("predict_key_action", True))
        self.key_action_chunk_size = int(config.get("key_action_chunk_size", 4))
        self.use_predicted_key_condition = bool(config.get("use_predicted_key_condition", True))
        self.detach_predicted_key_condition = bool(config.get("detach_predicted_key_condition", True))
        self.reverse_video_key_condition_type = str(
            config.get("reverse_video_key_condition_type", "latent_inpainting")
        ).strip().lower()
        self.reverse_action_key_condition_type = str(
            config.get("reverse_action_key_condition_type", "prefix")
        ).strip().lower()
        self.key_video_loss_weight = float(config.get("key_video_loss_weight", 1.0))
        self.key_action_loss_weight = float(config.get("key_action_loss_weight", 1.0))
        self.reverse_video_loss_weight = float(config.get("reverse_video_loss_weight", 1.0))
        self.reverse_action_loss_weight = float(config.get("reverse_action_loss_weight", 1.0))
        self.reverse_world_order = bool(config.get("reverse_world_order", False))
        self.goal_conditioned_multistep_inference = bool(
            config.get("goal_conditioned_multistep_inference", False)
        )
        self.goal_conditioned_return_forward_actions = bool(
            config.get("goal_conditioned_return_forward_actions", True)
        )
        self.goal_conditioned_video_guidance_scale = float(
            config.get("goal_conditioned_video_guidance_scale", 5.0)
        )
        self.goal_joint_key_diffusion = bool(
            config.get("goal_joint_key_diffusion", False)
        )
        self.goal_joint_reverse_diffusion = bool(
            config.get("goal_joint_reverse_diffusion", False)
        )
        if (
            self.goal_conditioned_wam
            and self.goal_joint_key_diffusion
            and (not self.predict_key_video or not self.predict_key_action)
        ):
            raise ValueError(
                "goal_joint_key_diffusion=true requires both "
                "predict_key_video=true and predict_key_action=true."
            )
        self.last_goal_conditioned_metrics = {}
        lang_token_dim = int(config.get("lang_token_dim", config.action_expert.get("text_dim", 4096)))
        self.stage4_video_lang_proj = nn.Linear(lang_token_dim, self._noise_video_channels)
        self.stage4_video_state_proj = nn.Linear(self.state_token_dim, self._noise_video_channels)
        self.stage4_video_sos = nn.Parameter(torch.zeros(1, self._noise_video_channels, 1, 1, 1))
        self.stage4_video_ar_head = nn.Sequential(
            nn.Conv3d(self._noise_video_channels, self._noise_video_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(self._noise_video_channels, self._noise_video_channels, kernel_size=1),
        )
        if self.generation_mode == "reverse_ar" and self.stage3_action_ar_causal_attention:
            action_expert = getattr(self.model, "action_expert", None)
            if action_expert is not None and not getattr(action_expert, "frame_causal_self_attention", False):
                raise ValueError(
                    "generation_mode=reverse_ar requires action_expert.frame_causal_self_attention=true. "
                    "Set architecture.model.action_expert.frame_causal_self_attn: true."
                )

        print("FMPRunner params: %e" %
              sum([p.numel() for p in self.model.parameters()]))

    def _timestep_video_tensor(
        self, t_video_scalar, batch_size: int, device: torch.device, noisy_video: torch.Tensor
    ) -> torch.Tensor:
        if not self._video_expert_is_wan22:
            return (torch.ones(batch_size, device=device) * t_video_scalar).long()
        latent_spatial = (
            noisy_video.shape[2],
            noisy_video.shape[3] // 2,
            noisy_video.shape[4] // 2,
        )
        return _video_inference_timestep_tensor(
            t_video_scalar,
            device,
            batch_size,
            wan22=True,
            latent_spatial_shape=latent_spatial,
        )

    def _goal_action_scheduler_step(
        self,
        *,
        model_pred: torch.Tensor,
        initial_noise: torch.Tensor,
        timestep,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """Advance one action diffusion step using the configured prediction parameterization."""
        if self.noise_scheduler_type == "ddpm":
            return self.noise_scheduler_sample_action.step(
                model_pred, timestep, sample
            ).prev_sample
        if self.prediction_type == "sample":
            scheduler_output = initial_noise - model_pred
        elif self.prediction_type == "noise":
            scheduler_output = sample - model_pred
        else:
            scheduler_output = model_pred
        return self.noise_scheduler_sample_action.step(
            scheduler_output, timestep, sample
        )

    @torch.no_grad()
    def _sample_goal_joint_video_action(
        self,
        *,
        lang_tokens: torch.Tensor,
        img_tokens: torch.Tensor,
        state_tokens: torch.Tensor,
        action_mask: torch.Tensor,
        video_template: torch.Tensor,
        condition_video_latents: torch.Tensor,
        anchor_video_latent: torch.Tensor,
        action_horizon: int,
        seed: Optional[int],
        num_inference_timesteps: Optional[int],
        clamp_video_anchor: bool,
        fixed_action_prefix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Jointly denoise video/action so action queries can attend causal video KV."""
        if self._video_expert_is_wan22:
            raise NotImplementedError(
                "Goal-conditioned joint video/action sampling currently requires "
                "WoW-style condition channels."
            )
        if condition_video_latents is None:
            raise ValueError(
                "condition_video_latents is required for joint goal-conditioned sampling."
            )

        device = video_template.device
        video_dtype = video_template.dtype
        action_dtype = state_tokens.dtype
        batch_size, _channels, frames, height, width = video_template.shape
        base_seed = 42 if seed is None else int(seed)
        video_generator = torch.Generator(device=device)
        video_generator.manual_seed(base_seed)
        action_generator = torch.Generator(device=device)
        action_generator.manual_seed(base_seed + 1)

        noisy_video = torch.randn(
            batch_size,
            self._noise_video_channels,
            frames,
            height,
            width,
            device=device,
            dtype=video_dtype,
            generator=video_generator,
        )
        noisy_action = torch.randn(
            batch_size,
            action_horizon,
            self.state_token_dim,
            device=device,
            dtype=action_dtype,
            generator=action_generator,
        )
        initial_action_noise = noisy_action.clone()
        anchor = anchor_video_latent.detach().to(device=device, dtype=video_dtype)
        condition = self._video_condition_with_anchor(
            condition_video_latents, anchor
        )
        action_prefix = (
            fixed_action_prefix.detach().to(device=device, dtype=action_dtype)
            if fixed_action_prefix is not None
            else None
        )
        prefix_len = (
            min(action_prefix.shape[1], action_horizon)
            if action_prefix is not None
            else 0
        )
        action_mask_expanded = action_mask.to(
            device=device, dtype=action_dtype
        ).expand(-1, action_horizon, -1)

        num_steps = (
            int(num_inference_timesteps)
            if num_inference_timesteps is not None
            else self.num_inference_timesteps
        )
        timesteps_action = self.noise_scheduler_sample_action.set_timesteps(num_steps)
        self.noise_scheduler_sample_video.set_timesteps(num_steps, device=device)
        timesteps_video = self.noise_scheduler_sample_video.timesteps
        if len(timesteps_action) != len(timesteps_video):
            raise RuntimeError(
                "Joint goal-conditioned sampling requires action/video schedulers "
                "to expose the same number of inference timesteps."
            )

        guidance_scale = self.goal_conditioned_video_guidance_scale
        for timestep_action_scalar, timestep_video_scalar in zip(
            timesteps_action, timesteps_video
        ):
            if clamp_video_anchor:
                noisy_video[:, :, :1] = anchor[:, :, :1]
            if prefix_len:
                noisy_action[:, :prefix_len] = action_prefix[:, :prefix_len]

            hidden_states_video = torch.cat(
                [noisy_video, condition], dim=1
            ).to(self.dtype)
            timestep_action = (
                torch.ones(batch_size, device=device) * timestep_action_scalar
            ).long()
            timestep_video = self._timestep_video_tensor(
                timestep_video_scalar, batch_size, device, noisy_video
            )
            pred_video, pred_action = self.model(
                hidden_states_video=hidden_states_video,
                hidden_states_action=torch.cat(
                    [noisy_action, action_mask_expanded], dim=2
                ).to(self.dtype),
                hidden_states_robostate=torch.cat(
                    [state_tokens, action_mask], dim=2
                ).to(self.dtype),
                hidden_states_visual=img_tokens.to(self.dtype),
                timestep_action=timestep_action,
                timestep_video=timestep_video,
                encoder_hidden_states=lang_tokens.to(self.dtype),
                value=None,
                video_kv_mask_mode="off",
                return_dict=False,
            )
            if guidance_scale > 1.0:
                uncond_video, _uncond_action = self.model(
                    hidden_states_video=hidden_states_video,
                    hidden_states_action=torch.cat(
                        [noisy_action, action_mask_expanded], dim=2
                    ).to(self.dtype),
                    hidden_states_robostate=torch.cat(
                        [state_tokens, action_mask], dim=2
                    ).to(self.dtype),
                    hidden_states_visual=img_tokens.to(self.dtype),
                    timestep_action=timestep_action,
                    timestep_video=timestep_video,
                    encoder_hidden_states=torch.zeros_like(lang_tokens).to(self.dtype),
                    value=None,
                    video_kv_mask_mode="off",
                    return_dict=False,
                )
                pred_video = uncond_video + guidance_scale * (
                    pred_video - uncond_video
                )

            noisy_video = self.noise_scheduler_sample_video.step(
                pred_video,
                timestep_video_scalar,
                noisy_video,
                return_dict=False,
            )[0].to(video_dtype)
            noisy_action = self._goal_action_scheduler_step(
                model_pred=pred_action[:, :action_horizon].to(action_dtype),
                initial_noise=initial_action_noise,
                timestep=timestep_action_scalar,
                sample=noisy_action,
            ).to(action_dtype)
            if clamp_video_anchor:
                noisy_video[:, :, :1] = anchor[:, :, :1]
            if prefix_len:
                noisy_action[:, :prefix_len] = action_prefix[:, :prefix_len]

        return noisy_video, noisy_action.mul(action_mask_expanded)

    @torch.no_grad()
    def _sample_goal_action_chunk(
        self,
        *,
        lang_tokens: torch.Tensor,
        img_tokens: torch.Tensor,
        state_tokens: torch.Tensor,
        action_mask: torch.Tensor,
        horizon: int,
        seed: Optional[int],
        num_inference_timesteps: Optional[int],
        fixed_prefix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample an action chunk from pure noise, optionally clamping a clean prefix."""
        device = lang_tokens.device
        dtype = state_tokens.dtype
        batch_size = state_tokens.shape[0]
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
        noisy_action = torch.randn(
            batch_size,
            horizon,
            self.state_token_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        initial_noise = noisy_action.clone()
        prefix = fixed_prefix.detach().to(device=device, dtype=dtype) if fixed_prefix is not None else None
        prefix_len = min(prefix.shape[1], horizon) if prefix is not None else 0
        action_mask_expanded = action_mask.to(device=device, dtype=dtype).expand(-1, horizon, -1)
        num_steps = (
            int(num_inference_timesteps)
            if num_inference_timesteps is not None
            else self.num_inference_timesteps
        )
        timesteps = self.noise_scheduler_sample_action.set_timesteps(num_steps)

        for timestep in timesteps:
            if prefix_len:
                noisy_action[:, :prefix_len] = prefix[:, :prefix_len]
            timestep_action = (
                torch.ones(batch_size, device=device) * timestep
            ).long()
            _pred_video, pred_action = self.model(
                hidden_states_video=None,
                hidden_states_action=torch.cat(
                    [noisy_action, action_mask_expanded], dim=2
                ).to(self.dtype),
                hidden_states_robostate=torch.cat(
                    [state_tokens, action_mask], dim=2
                ).to(self.dtype),
                hidden_states_visual=img_tokens.to(self.dtype),
                timestep_action=timestep_action,
                timestep_video=None,
                encoder_hidden_states=lang_tokens.to(self.dtype),
                value=None,
                video_kv_mask_mode="off",
                return_dict=False,
            )
            del _pred_video
            noisy_action = self._goal_action_scheduler_step(
                model_pred=pred_action[:, :horizon].to(dtype),
                initial_noise=initial_noise,
                timestep=timestep,
                sample=noisy_action,
            ).to(dtype)
            if prefix_len:
                noisy_action[:, :prefix_len] = prefix[:, :prefix_len]

        return noisy_action.mul(action_mask_expanded)

    @torch.no_grad()
    def _sample_goal_video(
        self,
        *,
        lang_tokens: torch.Tensor,
        video_latents: torch.Tensor,
        condition_video_latents: torch.Tensor,
        anchor_video_latent: torch.Tensor,
        seed: int,
        num_inference_timesteps: Optional[int],
        clamp_anchor: bool,
    ) -> torch.Tensor:
        """Sample a video from pure noise with a current-frame or predicted-key anchor."""
        if self._video_expert_is_wan22:
            raise NotImplementedError(
                "Goal-conditioned key-video sampling currently requires WoW-style "
                "condition channels; Wan2.2 needs a separate non-leaking key condition."
            )
        if condition_video_latents is None:
            raise ValueError("condition_video_latents is required for goal-conditioned video sampling.")

        device = video_latents.device
        dtype = video_latents.dtype
        batch_size, _channels, frames, height, width = video_latents.shape
        anchor = anchor_video_latent.detach().to(device=device, dtype=dtype)
        condition = self._video_condition_with_anchor(
            condition_video_latents, anchor
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        noisy_video = torch.randn(
            batch_size,
            self._noise_video_channels,
            frames,
            height,
            width,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        num_steps = (
            int(num_inference_timesteps)
            if num_inference_timesteps is not None
            else self.num_inference_timesteps
        )
        self.noise_scheduler_sample_video.set_timesteps(num_steps, device=device)
        timesteps = self.noise_scheduler_sample_video.timesteps
        guidance_scale = self.goal_conditioned_video_guidance_scale

        for timestep in timesteps:
            if clamp_anchor:
                noisy_video[:, :, :1] = anchor[:, :, :1]
            hidden_states_video = torch.cat(
                [noisy_video, condition], dim=1
            ).to(self.dtype)
            timestep_video = self._timestep_video_tensor(
                timestep, batch_size, device, noisy_video
            )
            pred_video, _pred_action = self.model(
                hidden_states_video=hidden_states_video,
                hidden_states_action=None,
                hidden_states_robostate=None,
                hidden_states_visual=None,
                timestep_action=None,
                timestep_video=timestep_video,
                encoder_hidden_states=lang_tokens.to(self.dtype),
                value=None,
                video_kv_mask_mode="off",
                return_dict=False,
            )
            del _pred_action
            if guidance_scale > 1.0:
                uncond_video, _uncond_action = self.model(
                    hidden_states_video=hidden_states_video,
                    hidden_states_action=None,
                    hidden_states_robostate=None,
                    hidden_states_visual=None,
                    timestep_action=None,
                    timestep_video=timestep_video,
                    encoder_hidden_states=torch.zeros_like(lang_tokens).to(self.dtype),
                    value=None,
                    video_kv_mask_mode="off",
                    return_dict=False,
                )
                del _uncond_action
                pred_video = uncond_video + guidance_scale * (
                    pred_video - uncond_video
                )
            noisy_video = self.noise_scheduler_sample_video.step(
                pred_video, timestep, noisy_video, return_dict=False
            )[0].to(dtype)
            if clamp_anchor:
                noisy_video[:, :, :1] = anchor[:, :, :1]

        return noisy_video

    @torch.no_grad()
    def conditional_sample_goal_conditioned_reverse_diffusion(
        self,
        *,
        lang_tokens: torch.Tensor,
        lang_attn_mask: torch.Tensor,
        img_tokens: torch.Tensor,
        state_tokens: torch.Tensor,
        action_mask: torch.Tensor,
        ctrl_freqs: torch.Tensor,
        video_latents: Optional[torch.Tensor],
        condition_video_latents: Optional[torch.Tensor],
        video_only: bool,
        action_only: bool,
        seed: Optional[int],
        num_inference_timesteps: Optional[int],
    ):
        """Two-stage inference: sample keys first, then reverse trajectories."""
        del lang_attn_mask, ctrl_freqs
        if not self.reverse_world_order:
            raise ValueError(
                "Goal-conditioned reverse diffusion inference requires reverse_world_order=true."
            )

        sample_video = not action_only
        sample_action = not video_only
        pred_key_video = None
        pred_reverse_video = None
        pred_key_action = None
        pred_reverse_action = None
        base_seed = 42 if seed is None else int(seed)

        if sample_video:
            if not self.predict_key_video:
                raise ValueError("predict_key_video must be enabled for multi-step key-video inference.")
            if video_latents is None or condition_video_latents is None:
                raise ValueError(
                    "video_latents and condition_video_latents are required when video sampling is enabled."
                )
        if sample_action and not self.predict_key_action:
            raise ValueError(
                "predict_key_action must be enabled for multi-step key-action inference."
            )

        use_joint_key = (
            self.goal_joint_key_diffusion and sample_video and sample_action
        )
        use_joint_reverse = (
            self.goal_joint_reverse_diffusion and sample_video and sample_action
        )
        key_len = max(1, min(self.key_action_chunk_size, self.pred_horizon))

        if use_joint_key:
            current_video_anchor = video_latents[:, :, -1:].detach()
            key_video_candidate, key_action_candidate = (
                self._sample_goal_joint_video_action(
                    lang_tokens=lang_tokens,
                    img_tokens=img_tokens,
                    state_tokens=state_tokens,
                    action_mask=action_mask,
                    video_template=video_latents[:, :, :1],
                    condition_video_latents=condition_video_latents[:, :, :1],
                    anchor_video_latent=current_video_anchor,
                    action_horizon=key_len,
                    seed=base_seed,
                    num_inference_timesteps=num_inference_timesteps,
                    clamp_video_anchor=False,
                )
            )
            pred_key_video = key_video_candidate[:, :, :1].detach()
            pred_key_action = key_action_candidate[:, :key_len].detach()
        else:
            if sample_video:
                current_video_anchor = video_latents[:, :, -1:].detach()
                key_video_candidate = self._sample_goal_video(
                    lang_tokens=lang_tokens,
                    video_latents=video_latents,
                    condition_video_latents=condition_video_latents,
                    anchor_video_latent=current_video_anchor,
                    seed=base_seed,
                    num_inference_timesteps=num_inference_timesteps,
                    clamp_anchor=False,
                )
                pred_key_video = key_video_candidate[:, :, :1].detach()
            if sample_action:
                pred_key_action = self._sample_goal_action_chunk(
                    lang_tokens=lang_tokens,
                    img_tokens=img_tokens,
                    state_tokens=state_tokens,
                    action_mask=action_mask,
                    horizon=key_len,
                    seed=base_seed + 1,
                    num_inference_timesteps=num_inference_timesteps,
                ).detach()

        if use_joint_reverse:
            pred_reverse_video, pred_reverse_action = (
                self._sample_goal_joint_video_action(
                    lang_tokens=lang_tokens,
                    img_tokens=img_tokens,
                    state_tokens=state_tokens,
                    action_mask=action_mask,
                    video_template=video_latents,
                    condition_video_latents=condition_video_latents,
                    anchor_video_latent=pred_key_video,
                    action_horizon=self.pred_horizon,
                    seed=base_seed + 2,
                    num_inference_timesteps=num_inference_timesteps,
                    clamp_video_anchor=True,
                    fixed_action_prefix=pred_key_action,
                )
            )
        else:
            if sample_video:
                pred_reverse_video = self._sample_goal_video(
                    lang_tokens=lang_tokens,
                    video_latents=video_latents,
                    condition_video_latents=condition_video_latents,
                    anchor_video_latent=pred_key_video,
                    seed=base_seed + 2,
                    num_inference_timesteps=num_inference_timesteps,
                    clamp_anchor=True,
                )
            if sample_action:
                pred_reverse_action = self._sample_goal_action_chunk(
                    lang_tokens=lang_tokens,
                    img_tokens=img_tokens,
                    state_tokens=state_tokens,
                    action_mask=action_mask,
                    horizon=self.pred_horizon,
                    seed=base_seed + 3,
                    num_inference_timesteps=num_inference_timesteps,
                    fixed_prefix=pred_key_action,
                )

        if pred_reverse_action is not None and self.goal_conditioned_return_forward_actions:
            pred_trajectory = torch.flip(pred_reverse_action, dims=[1])
            trajectory_order = "forward"
        else:
            pred_trajectory = pred_reverse_action
            trajectory_order = "reverse"

        return {
            "pred_trajectory": pred_trajectory,
            "pred_reverse_trajectory": pred_reverse_action,
            "pred_video": pred_reverse_video,
            "pred_key_video": pred_key_video,
            "pred_key_action": pred_key_action,
            "pred_value": None,
            "trajectory_order": trajectory_order,
        }

    def conditional_sample(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens, action_mask,
                            ctrl_freqs, video_latents, condition_video_latents,
                            seed: Optional[int] = 42, guidance_scale: float = 1.0, use_mean_flow=False,
                            num_inference_timesteps: Optional[int] = None):
        '''
        lang_cond: language conditional data, (batch_size, lang_len, hidden_size).
        lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
            which should be True-False bool tensor.
        img_cond: image conditional data, (batch_size, img_len, hidden_size).
        state_traj: (batch_size, 1, hidden_size), state trajectory.
        action_mask: (batch_size, 1, action_dim), a 0-1 **float** tensor
            indicating the valid action dimensions.
        ctrl_freqs: (batch_size,), control frequency for each sample.

        return: (batch_size, horizon, action_dim)
        '''
        device = lang_tokens.device
        dtype = lang_tokens.dtype
        state_tokens = state_tokens.to(device, dtype=dtype)
        batch_size = state_tokens.shape[0]
        num_inference_timesteps = (
            num_inference_timesteps if num_inference_timesteps is not None else self.num_inference_timesteps
        )
        generator: Optional[torch.Generator] = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
        # 1. 初始化2个随机噪声: 动作和视频
        noisy_action = torch.randn(
            size=(state_tokens.shape[0], self.pred_horizon + 1 if self.use_value else self.pred_horizon, self.state_token_dim),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        noise_action = noisy_action.clone()
        if self.use_value:
            noisy_value = noisy_action[:, -1:]
            noisy_action = noisy_action[:, :-1]
            noise_value = noise_action[:, -1:]
            noise_action = noise_action[:, :-1]
        else:
            noisy_value = None
            noise_value = None
        batch_size, c, f, h, w = video_latents.shape
        noisy_video = torch.randn(
            size=(batch_size, self._noise_video_channels, f, h, w),
            dtype=video_latents.dtype,
            device=device,
            generator=generator,
        )
        # 2. 设置扩散步数(step values) (i.e, 调度器)
        timesteps_action = self.noise_scheduler_sample_action.set_timesteps(num_inference_timesteps)
        # timesteps_video = self.noise_scheduler_sample_video.set_timesteps(num_inference_timesteps)
        self.noise_scheduler_sample_video.set_timesteps(num_inference_timesteps, device=device)
        timesteps_video = self.noise_scheduler_sample_video.timesteps
        uncond_lang_tokens = torch.load(_DEMO_UNCOND_PT, map_location="cpu").to(device,torch.bfloat16)

        # 3. DDPM联合去噪循环
        # Pre-expand action_mask to match action horizon (mirrors official RDT behaviour)
        action_mask_expanded = action_mask.expand(-1, self.pred_horizon, -1)
        video_stop_time = 5
        dt = 1000.0 / num_inference_timesteps
        for idx, (t_action, t_video) in enumerate(zip(timesteps_action, timesteps_video)):
            # Prepare state-action trajectory
            # Predict the model output
            if idx >= video_stop_time:
                t_video = timesteps_video[video_stop_time]
            timestep_action = (torch.ones(batch_size, device=device) * t_action).long()
            timestep_video = self._timestep_video_tensor(t_video, batch_size, device, noisy_video)
            if self._video_expert_is_wan22:
                noisy_video[:, :, :1] = video_latents[:, :, :1] # first frame is clean
                hidden_states_video = noisy_video
            else:
                mask = torch.zeros(batch_size, c, 1, h, w, device=device)
                mask[:, :, 0, :, :] = 1
                hidden_states_video = torch.concat([noisy_video, condition_video_latents], dim=1)
            # condition_video_latents 是1个20dim的，前4dim是0-1的值用来表示是否被mask 掉了 => 第1 frame的4dim 是可以见到的所以是全1，第2frame的4dim 是不可以见的所以是全0
            # 用1个 latent 表示4frames rgb 图像，所以我们用 4dim 不能用1dim 的 mask
            # 剩下 16dim 就是表示 视频的内容了，前4dim是可以见到的所以是有意义的，后16dim 是不可以见的所以是全0的
            # 同时预测视频和动作的速度
            # noisy_action = noisy_action * action_mask_expanded
            video_velocity_pred, action_velocity_pred = self.model(
                hidden_states_action=torch.cat([noisy_action, action_mask_expanded], dim=2),
                hidden_states_robostate=torch.cat([state_tokens, action_mask], dim=2),
                hidden_states_video=hidden_states_video,
                hidden_states_visual=img_tokens,         # [1, 4374, 1152] (这个是专门给AE的, 当下t, t-1 时刻的渲染出来的3个视角的图片)
                timestep_action=timestep_action,
                timestep_video=timestep_video,
                encoder_hidden_states=lang_tokens,       # [1, 512, 4096]
                value=torch.cat([noisy_value, action_mask], dim=2) if self.use_value else noisy_value,
                return_dict=False,
            )
            # Classifier-free guidance (可选)
            if idx < video_stop_time and guidance_scale > 1.0:
                # 无条件预测
                uncond_video_velocity_pred, uncond_action_velocity_pred = self.model(
                    hidden_states_action=torch.cat([noisy_action, action_mask_expanded], dim=2),
                    hidden_states_robostate=torch.cat([state_tokens, action_mask], dim=2),
                    hidden_states_video=hidden_states_video,
                    hidden_states_visual=img_tokens,
                    timestep_action=timestep_action,
                    timestep_video=timestep_video,
                    encoder_hidden_states=uncond_lang_tokens,
                    value=torch.cat([noisy_value, action_mask], dim=2) if self.use_value else noisy_value,
                    return_dict=False,
                )
                # 增强条件信号
                video_velocity_pred = uncond_video_velocity_pred + guidance_scale * (video_velocity_pred - uncond_video_velocity_pred)
            # Compute previous actions: x_t -> x_t-1
            if self.noise_scheduler_type != "ddpm":
                if self.prediction_type == 'sample':
                    if self.use_value:
                        value_velocity_pred = action_velocity_pred[:, -1:]
                        action_velocity_pred = action_velocity_pred[:, :-1]
                        noisy_value = self.noise_scheduler_sample_action.step(noise_value - value_velocity_pred, t_action, noisy_value)
                    noisy_action = self.noise_scheduler_sample_action.step((noise_action - action_velocity_pred) / (t_action * 0.001) if use_mean_flow else (noise_action - action_velocity_pred), t_action, noisy_action)
                    # noise = torch.randn_like(noisy_action)
                    # _timestep_action = timesteps_video[idx+1] if idx+1 < len(timesteps_video) else 0
                    # _timestep_action = torch.ones(batch_size, device=device) * _timestep_action
                    # noisy_action = self.noise_scheduler_sample_action.add_noise(action_velocity_pred, noise, timestep_action)
                elif self.prediction_type == 'noise':
                    if self.use_value:
                        value_velocity_pred = action_velocity_pred[:, -1:]
                        action_velocity_pred = action_velocity_pred[:, :-1]
                        noisy_value = self.noise_scheduler_sample_action.step((value_velocity_pred - noisy_value) / (1-(t_action-dt) * 0.001), t_action, noisy_value)
                    noisy_action = self.noise_scheduler_sample_action.step((action_velocity_pred - noisy_action) / (1-(t_action-dt)*0.001), t_action, noisy_action)
                else:
                    if self.use_value:
                        value_velocity_pred = action_velocity_pred[:, -1:]
                        action_velocity_pred = action_velocity_pred[:, :-1]
                        noisy_value = self.noise_scheduler_sample_action.step(value_velocity_pred, t_action, noisy_value)
                    noisy_action = self.noise_scheduler_sample_action.step(action_velocity_pred, t_action, noisy_action)
            else:
                noisy_action = self.noise_scheduler_sample_action.step(action_velocity_pred, t_action, noisy_action).prev_sample
            if idx < video_stop_time:
                noisy_video = self.noise_scheduler_sample_video.step(video_velocity_pred, t_video, noisy_video, return_dict=False)[0]
            # 一步去噪: x_t -> x_{t-1}
            noisy_action = noisy_action.to(state_tokens.dtype)
            noisy_video = noisy_video.to(video_latents.dtype)
            noisy_value = noisy_value.to(state_tokens.dtype) if self.use_value else None
        # Apply action mask to zero out invalid action dimensions (mirrors official RDT)
        noisy_action = noisy_action * action_mask_expanded
        if self._video_expert_is_wan22:
            noisy_video[:, :, :1] = video_latents[:, :, :1]
        out = {
            "pred_trajectory": noisy_action,
            "pred_video": noisy_video,
            "pred_value": noisy_value,
        }
        return out

    def conditional_sample_action_only(
        self,
        lang_tokens,
        lang_attn_mask,
        img_tokens,
        state_tokens,
        action_mask,
        ctrl_freqs,
        guidance_scale: float = 5.0,
        use_cfg: bool = False,
        seed: Optional[int] = 42,
        num_inference_timesteps: Optional[int] = None,
    ):
        """
        仅对动作分支做扩散采样（无视频 latent），接口与 conditional_sample 对齐，但不传 video。

        Args:
            lang_tokens: (B, lang_len, dim)
            lang_attn_mask: (B, lang_len) 保留与 conditional_sample 一致，当前未传入 WorldPolicyModel
            img_tokens: (B, img_len, dim) 与训练时 hidden_states_visual 一致
            state_tokens: (B, 1, state_dim) 机器人状态
            action_mask / ctrl_freqs: 与 conditional_sample 一致，便于外部统一调用
            guidance_scale: CFG 强度（use_cfg 为 True 时）
            use_cfg: 是否做语言 classifier-free guidance（需 demos/uncond.pt）

        Returns:
            dict: pred_trajectory (B, pred_horizon, action_dim)，若 use_value 则含 pred_value
        """
        del lang_attn_mask, ctrl_freqs  # 与 conditional_sample 同签名，WorldPolicyModel 未用

        device = lang_tokens.device
        dtype = lang_tokens.dtype
        state_tokens = state_tokens.to(device, dtype=dtype)
        batch_size = state_tokens.shape[0]
        generator: Optional[torch.Generator] = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
        noisy_action = torch.randn(
            size=(batch_size, self.pred_horizon, self.state_token_dim),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        noise_action = noisy_action.clone()
        noisy_value = (
            torch.randn(size=(batch_size, 1, self.state_token_dim), dtype=dtype, device=device)
            if self.use_value
            else None
        )
        noise_value = noisy_value.clone() if self.use_value else None

        action_mask_expanded = action_mask.expand(-1, self.pred_horizon, -1)
        num_inference_timesteps = (
            num_inference_timesteps if num_inference_timesteps is not None else self.num_inference_timesteps
        )
        timesteps = self.noise_scheduler_sample_action.set_timesteps(num_inference_timesteps)
        uncond_lang_tokens = torch.load(
            _DEMO_UNCOND_PT, map_location="cpu"
        ).to(device, torch.bfloat16)

        for idx, t_action in enumerate(timesteps):
            timestep_action = (torch.ones(batch_size, device=device) * t_action).long()
            # noisy_action = noisy_action * action_mask_expanded
            _, action_velocity_pred = self.model(
                hidden_states_action=torch.cat([noisy_action, action_mask_expanded], dim=2),
                hidden_states_robostate=torch.cat([state_tokens, action_mask], dim=2),
                hidden_states_video=None,
                hidden_states_visual=img_tokens,
                timestep_action=timestep_action,
                timestep_video=None,
                encoder_hidden_states=lang_tokens,
                value=torch.cat([noisy_value, action_mask], dim=2) if self.use_value else noisy_value,
                video_kv_mask_mode="off",
                return_dict=False,
            )
            if use_cfg:
                _, uncond_action_velocity_pred = self.model(
                    hidden_states_action=torch.cat([noisy_action, action_mask_expanded], dim=2),
                    hidden_states_robostate=torch.cat([state_tokens, action_mask], dim=2),
                    hidden_states_video=None,
                    hidden_states_visual=img_tokens,
                    timestep_action=timestep_action,
                    timestep_video=None,
                    encoder_hidden_states=uncond_lang_tokens,
                    value=torch.cat([noisy_value, action_mask], dim=2) if self.use_value else noisy_value,
                    video_kv_mask_mode="off",
                    return_dict=False,
                )
                action_velocity_pred = uncond_action_velocity_pred + guidance_scale * (
                    action_velocity_pred - uncond_action_velocity_pred
                )

            if self.noise_scheduler_type != "ddpm":
                if self.prediction_type == "sample":
                    if self.use_value:
                        value_velocity_pred = action_velocity_pred[:, -1:]
                        action_velocity_pred = action_velocity_pred[:, :-1]
                        noisy_value = self.noise_scheduler_sample_action.step(
                            noise_value - value_velocity_pred, t_action, noisy_value
                        )
                    noisy_action = self.noise_scheduler_sample_action.step(
                        noise_action - action_velocity_pred, t_action, noisy_action
                    )
                elif self.prediction_type == 'noise':
                    if self.use_value:
                        value_velocity_pred = action_velocity_pred[:, -1:]
                        action_velocity_pred = action_velocity_pred[:, :-1]
                        noisy_value = self.noise_scheduler_sample_action.step(
                            noisy_value - value_velocity_pred, t_action, noisy_value
                        )
                    noisy_action = self.noise_scheduler_sample_action.step(
                        noisy_action - action_velocity_pred, t_action, noisy_action
                    )
                else:
                    if self.use_value:
                        value_velocity_pred = action_velocity_pred[:, -1:]
                        action_velocity_pred = action_velocity_pred[:, :-1]
                        noisy_value = self.noise_scheduler_sample_action.step(
                            value_velocity_pred, t_action, noisy_value
                        )
                    noisy_action = self.noise_scheduler_sample_action.step(
                        action_velocity_pred, t_action, noisy_action
                    )
            else:
                noisy_action = self.noise_scheduler_sample_action.step(
                    action_velocity_pred, t_action, noisy_action
                ).prev_sample

            noisy_action = noisy_action.to(state_tokens.dtype)
            noisy_value = noisy_value.to(state_tokens.dtype) if self.use_value else None

        # Apply action mask to zero out invalid action dimensions (mirrors official RDT)
        noisy_action = noisy_action * action_mask_expanded
        out = {
            "pred_trajectory": noisy_action,
            "pred_video": None,
            "pred_value": noisy_value if self.use_value else None,
        }
        return out

    def conditional_sample_video_only(
        self,
        lang_tokens,
        video_latents,
        condition_video_latents,
        denoise_steps: Optional[int] = None,
        seed: int = 42,
        num_inference_timesteps: Optional[int] = None,
    ):
        """
        仅对视频 latent 做扩散去噪。

        Args:
            lang_tokens: 语言条件 (B, lang_len, dim)。
            video_latents: 形状参考 (B, C, F, H, W)，此处主要用设备与 spatial 维以初始化噪声。
            condition_video_latents: 与 noisy 视频在通道维拼接的条件。
            denoise_steps: 实际执行的去噪步数；``None`` 表示用 scheduler 的全部步数；
                若小于总步数则提前停止；若为 0 则直接返回初始高斯噪声。
            seed: 初始 ``noisy_video`` 的随机种子。
            num_inference_timesteps: 传给调度器的推理步数；``None`` 表示使用 ``self.num_inference_timesteps``。

        Returns:
            ``(noisy_video, t_last)``：``noisy_video`` 为 (B, C_vid, F, H, W)，``C_vid`` 为 16（WoW）或 48（Wan2.2）；
            ``t_last`` 在 ``denoise_steps is None`` 时为 ``0``，否则为 ``timesteps_video[n_run]``（供二阶段构造 ``timestep_video``）。
        """
        device = video_latents.device
        batch_size = video_latents.shape[0]
        batch_size, _c, f, h, w = video_latents.shape
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        noisy_video = torch.randn(
            size=(batch_size, self._noise_video_channels, f, h, w),
            dtype=video_latents.dtype,
            device=device,
            generator=generator,
        )
        hidden_states_video = (
            noisy_video
            if self._video_expert_is_wan22
            else torch.concat([noisy_video, condition_video_latents], dim=1)
        ).to(self.dtype)
        num_inf = num_inference_timesteps if num_inference_timesteps is not None else self.num_inference_timesteps
        self.noise_scheduler_sample_video.set_timesteps(num_inf, device=device)
        timesteps_video = self.noise_scheduler_sample_video.timesteps
        n_total = len(timesteps_video)
        if denoise_steps is None:
            n_run = n_total
        else:
            n_run = max(0, min(int(denoise_steps), n_total))

        for idx in range(n_run):
            t_video = timesteps_video[idx]
            if self._video_expert_is_wan22:
                noisy_video[:, :, :1] = video_latents[:, :, :1]
            timestep_video = self._timestep_video_tensor(t_video, batch_size, device, noisy_video)
            video_velocity_pred, _action_velocity_pred = self.model(
                hidden_states_video=hidden_states_video,
                timestep_video=timestep_video,
                encoder_hidden_states=lang_tokens,
                return_dict=False,
            )
            uncond_video_velocity_pred, _uncond_action_velocity_pred = self.model(
                hidden_states_video=hidden_states_video,
                timestep_video=timestep_video,
                encoder_hidden_states=torch.zeros_like(lang_tokens),
                return_dict=False,
            )
            video_velocity_pred = uncond_video_velocity_pred + 5.0 * (
                video_velocity_pred - uncond_video_velocity_pred
            )
            noisy_video = self.noise_scheduler_sample_video.step(
                video_velocity_pred, t_video, noisy_video, return_dict=False
            )[0]
            noisy_video = noisy_video.to(video_latents.dtype)
            hidden_states_video = (
                noisy_video
                if self._video_expert_is_wan22
                else torch.concat([noisy_video, condition_video_latents], dim=1)
            )

        if self._video_expert_is_wan22:
            noisy_video[:, :, :1] = video_latents[:, :, :1]
        return noisy_video, 0 if denoise_steps is None else timesteps_video[n_run]

    def conditional_sample_video_then_action(
        self,
        lang_tokens,
        lang_attn_mask,
        img_tokens,
        state_tokens,
        action_mask,
        ctrl_freqs,
        video_latents,
        condition_video_latents,
        seed: Optional[int] = 42,
        guidance_scale_action: float = 1.0,
        use_cfg_action: bool = False,
        video_denoise_steps: Optional[int] = None,
        num_inference_timesteps: Optional[int] = None,
        use_mean_flow: bool = False,
    ):
        """
        两阶段采样：先仅对视频分支做完整去噪得到 ``noisy_video``（此处为去噪后的视频 latent），
        再固定该视频与 ``condition_video_latents`` 拼接后的 ``hidden_states_video``，
        仅对 action（及 ``use_value`` 时的 value）做扩散去噪；视频分支不再更新。

        video_denoise_steps: 第一阶段视频去噪步数上限，``None`` 表示与 ``num_inference_timesteps`` 一致。

        Returns:
            dict: ``pred_trajectory``, ``pred_video``（第一阶段输出）, ``pred_value``（可选）
        """
        del lang_attn_mask, ctrl_freqs

        device = lang_tokens.device
        dtype = lang_tokens.dtype
        state_tokens = state_tokens.to(device, dtype=dtype)
        batch_size = state_tokens.shape[0]

        # ---------- 1) 视频先去噪 ----------
        noisy_video, t_video = self.conditional_sample_video_only(
            lang_tokens,
            video_latents,
            condition_video_latents,
            denoise_steps=video_denoise_steps,
            seed=42 if seed is None else int(seed),
            num_inference_timesteps=num_inference_timesteps,
        )
        # 拼接条件视频通道；去噪结束后视频条件用 scheduler 的最末时刻（最小噪声）
        # self.noise_scheduler_sample_video.set_timesteps(self.num_inference_timesteps, device=device)
        # timesteps_video = self.noise_scheduler_sample_video.timesteps
        # t_video_clean = timesteps_video[-1]
        hidden_states_video = (
            noisy_video
            if self._video_expert_is_wan22
            else torch.concat([noisy_video, condition_video_latents], dim=1)
        )
        timestep_video = self._timestep_video_tensor(t_video, batch_size, device, noisy_video)

        # ---------- 2) 再对 action 去噪 ----------
        generator: Optional[torch.Generator] = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))

        noisy_action = torch.randn(
            size=(
                batch_size,
                self.pred_horizon + 1 if self.use_value else self.pred_horizon,
                self.state_token_dim,
            ),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        noise_action = noisy_action.clone()
        if self.use_value:
            noisy_value = noisy_action[:, -1:]
            noisy_action = noisy_action[:, :-1]
            noise_value = noise_action[:, -1:]
            noise_action = noise_action[:, :-1]
        else:
            noisy_value = None
            noise_value = None

        action_mask_expanded = action_mask.expand(-1, self.pred_horizon, -1)
        num_inf = num_inference_timesteps if num_inference_timesteps is not None else self.num_inference_timesteps
        timesteps = self.noise_scheduler_sample_action.set_timesteps(num_inf)
        uncond_lang_tokens = torch.load(
            _DEMO_UNCOND_PT, map_location="cpu"
        ).to(device, torch.bfloat16)

        for idx, t_action in enumerate(timesteps):
            timestep_action = (torch.ones(batch_size, device=device) * t_action).long()
            _, action_velocity_pred = self.model(
                hidden_states_action=torch.cat([noisy_action, action_mask_expanded], dim=2),
                hidden_states_robostate=torch.cat([state_tokens, action_mask], dim=2),
                hidden_states_video=hidden_states_video,
                hidden_states_visual=img_tokens,
                timestep_action=timestep_action,
                timestep_video=timestep_video,
                encoder_hidden_states=lang_tokens,
                value=torch.cat([noisy_value, action_mask], dim=2) if self.use_value else noisy_value,
                video_kv_mask_mode='off',
                return_dict=False,
            )
            if use_cfg_action:
                _, uncond_action_velocity_pred = self.model(
                    hidden_states_action=torch.cat([noisy_action, action_mask_expanded], dim=2),
                    hidden_states_robostate=torch.cat([state_tokens, action_mask], dim=2),
                    hidden_states_video=hidden_states_video,
                    hidden_states_visual=img_tokens,
                    timestep_action=timestep_action,
                    timestep_video=timestep_video,
                    encoder_hidden_states=uncond_lang_tokens,
                    value=torch.cat([noisy_value, action_mask], dim=2) if self.use_value else noisy_value,
                    video_kv_mask_mode='off',
                    return_dict=False,
                )
                action_velocity_pred = uncond_action_velocity_pred + guidance_scale_action * (
                    action_velocity_pred - uncond_action_velocity_pred
                )

            if self.noise_scheduler_type != "ddpm":
                if self.prediction_type == "sample":
                    if self.use_value:
                        value_velocity_pred = action_velocity_pred[:, -1:]
                        action_velocity_pred = action_velocity_pred[:, :-1]
                        noisy_value = self.noise_scheduler_sample_action.step(
                            noise_value - value_velocity_pred, t_action, noisy_value
                        )
                    noisy_action = self.noise_scheduler_sample_action.step(
                        (noise_action - action_velocity_pred) / (t_action * 0.001)
                        if use_mean_flow
                        else (noise_action - action_velocity_pred),
                        t_action,
                        noisy_action,
                    )
                else:
                    noisy_action = self.noise_scheduler_sample_action.step(
                        action_velocity_pred, t_action, noisy_action
                    )
            else:
                noisy_action = self.noise_scheduler_sample_action.step(
                    action_velocity_pred, t_action, noisy_action
                ).prev_sample

            noisy_action = noisy_action.to(state_tokens.dtype)
            noisy_value = noisy_value.to(state_tokens.dtype) if self.use_value else None

        # Apply action mask to zero out invalid action dimensions (mirrors official RDT)
        noisy_action = noisy_action * action_mask_expanded
        return {
            "pred_trajectory": noisy_action,
            "pred_video": noisy_video,
            "pred_value": noisy_value if self.use_value else None,
        }

    def _masked_mse(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if valid_mask is None:
            return F.mse_loss(pred, target)
        valid = valid_mask.to(device=pred.device, dtype=pred.dtype).unsqueeze(-1)
        loss_elem = F.mse_loss(pred, target, reduction="none")
        denom = valid.expand_as(loss_elem).sum().clamp_min(1.0)
        return (loss_elem * valid).sum() / denom

    def _compute_stage2_reverse_ar_action_loss(
        self,
        *,
        lang_tokens: torch.Tensor,
        img_tokens: torch.Tensor,
        state_tokens: torch.Tensor,
        action_gt: torch.Tensor,
        action_mask: torch.Tensor,
        action_valid_mask: Optional[torch.Tensor],
        video_kv_mask_mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced CoA-style reverse action loss.

        ``action_gt`` is already in the training view. With ``reverse_world_order=true``,
        token 0 is the semantic key action. The decoder input is shifted right with an
        all-zero SOS action token; causal action attention can then learn
        ``SOS -> key action -> previous action -> ...``.
        """
        pred_action_ar, target_action = self._predict_reverse_ar_action_teacher_forced(
            lang_tokens=lang_tokens,
            img_tokens=img_tokens,
            state_tokens=state_tokens,
            action_gt=action_gt,
            action_mask=action_mask,
            video_kv_mask_mode=video_kv_mask_mode,
        )
        reverse_ar_loss = self._masked_mse(pred_action_ar, target_action, action_valid_mask)
        key_valid_mask = (
            action_valid_mask[:, :1]
            if action_valid_mask is not None
            else None
        )
        key_action_loss = self._masked_mse(pred_action_ar[:, :1], target_action[:, :1], key_valid_mask)
        return key_action_loss, reverse_ar_loss

    def _predict_reverse_ar_action_teacher_forced(
        self,
        *,
        lang_tokens: torch.Tensor,
        img_tokens: torch.Tensor,
        state_tokens: torch.Tensor,
        action_gt: torch.Tensor,
        action_mask: torch.Tensor,
        video_kv_mask_mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, horizon, _ = action_gt.shape
        sos = torch.zeros_like(action_gt[:, :1])
        shifted_action = torch.cat([sos, action_gt[:, :-1]], dim=1)
        action_mask_expanded = action_mask.expand(-1, horizon, -1)
        timestep_action = torch.zeros(batch_size, device=action_gt.device, dtype=torch.long)
        ar_value = None
        if self.use_value:
            zero_value = torch.zeros_like(action_gt[:, :1])
            ar_value = torch.cat([zero_value, action_mask], dim=2)

        pred_video_unused, pred_action_ar = self.model(
            hidden_states_video=None,
            hidden_states_action=torch.cat([shifted_action, action_mask_expanded], dim=2).to(self.dtype),
            hidden_states_robostate=torch.cat([state_tokens, action_mask], dim=2).to(self.dtype),
            hidden_states_visual=img_tokens.to(self.dtype),
            timestep_action=timestep_action,
            timestep_video=None,
            encoder_hidden_states=lang_tokens.to(self.dtype),
            value=ar_value.to(self.dtype) if ar_value is not None else None,
            video_kv_mask_mode=video_kv_mask_mode,
            return_dict=False,
        )
        del pred_video_unused
        pred_action_ar = pred_action_ar[:, :horizon].to(action_gt.dtype)
        target_action = action_gt.to(pred_action_ar.dtype)
        return pred_action_ar, target_action

    def _compute_video_diffusion_loss(
        self,
        *,
        lang_tokens: torch.Tensor,
        video_latents: torch.Tensor,
        condition_video_latents: torch.Tensor,
        video_kv_mask_mode: str,
    ) -> torch.Tensor:
        (
            noise_for_video,
            noisy_video_hidden,
            timesteps_for_video,
            per_element_loss_mask,
        ) = _video_train_diffusion_prep(
            self.noise_scheduler_video,
            video_latents,
            condition_video_latents,
            self.dtype,
            wan22=self._video_expert_is_wan22,
        )
        pred_video, pred_action_unused = self.model(
            hidden_states_video=noisy_video_hidden.to(self.dtype),
            hidden_states_action=None,
            hidden_states_robostate=None,
            hidden_states_visual=None,
            timestep_action=None,
            timestep_video=timesteps_for_video,
            encoder_hidden_states=lang_tokens.to(self.dtype),
            value=None,
            video_kv_mask_mode=video_kv_mask_mode,
            return_dict=False,
        )
        del pred_action_unused
        target_video = (noise_for_video - video_latents).to(self.dtype)
        if per_element_loss_mask is None:
            return F.mse_loss(pred_video, target_video)
        return (
            F.mse_loss(pred_video, target_video, reduction="none") * per_element_loss_mask
        ).sum() / per_element_loss_mask.sum()

    def _stage4_video_condition(
        self,
        *,
        lang_tokens: torch.Tensor,
        state_tokens: Optional[torch.Tensor],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        lang_cond = self.stage4_video_lang_proj(lang_tokens.mean(dim=1).to(self.stage4_video_lang_proj.weight.dtype))
        if state_tokens is not None:
            state_cond = self.stage4_video_state_proj(
                state_tokens[:, 0].to(self.stage4_video_state_proj.weight.dtype)
            )
            lang_cond = lang_cond + state_cond
        return lang_cond.to(dtype).view(lang_tokens.shape[0], self._noise_video_channels, 1, 1, 1)

    def _predict_stage4_video_ar_teacher_forced(
        self,
        *,
        lang_tokens: torch.Tensor,
        state_tokens: Optional[torch.Tensor],
        video_latents: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_video = video_latents[:, : self._noise_video_channels].to(self.dtype)
        batch_size, channels, frames, height, width = target_video.shape
        sos = self.stage4_video_sos.to(device=target_video.device, dtype=target_video.dtype)
        sos = sos.expand(batch_size, channels, 1, height, width)
        shifted_video = torch.cat([sos, target_video[:, :, :-1]], dim=2)
        cond = self._stage4_video_condition(
            lang_tokens=lang_tokens,
            state_tokens=state_tokens,
            dtype=target_video.dtype,
        )
        head_dtype = next(self.stage4_video_ar_head.parameters()).dtype
        pred_video = self.stage4_video_ar_head(shifted_video.to(head_dtype)).to(target_video.dtype) + cond
        return pred_video.to(video_latents.dtype), target_video.to(video_latents.dtype)

    def _compute_stage4_video_ar_loss(
        self,
        *,
        lang_tokens: torch.Tensor,
        state_tokens: Optional[torch.Tensor],
        video_latents: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_video, target_video = self._predict_stage4_video_ar_teacher_forced(
            lang_tokens=lang_tokens,
            state_tokens=state_tokens,
            video_latents=video_latents,
        )
        video_ar_loss = F.mse_loss(pred_video, target_video)
        keyframe_loss = F.mse_loss(pred_video[:, :, :1], target_video[:, :, :1])
        loss_video = (
            self.stage4_keyframe_loss_weight * keyframe_loss
            + self.stage4_video_ar_loss_weight * video_ar_loss
        )
        return loss_video, keyframe_loss, video_ar_loss

    def _compute_action_diffusion_loss(
        self,
        *,
        lang_tokens: torch.Tensor,
        img_tokens: torch.Tensor,
        state_tokens: torch.Tensor,
        action_gt: torch.Tensor,
        action_mask: torch.Tensor,
        value: Optional[torch.Tensor],
        action_valid_mask: Optional[torch.Tensor],
        video_kv_mask_mode: str,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device
        noise_for_action = torch.randn(action_gt.shape, dtype=action_gt.dtype, device=device)
        noise_for_value = torch.randn(value.shape, dtype=value.dtype, device=device) if self.use_value else None
        _sigmas_for_action, timesteps_for_action = self.noise_scheduler_action.sample(batch_size, device=device)
        noisy_action = self.noise_scheduler_action.add_noise(action_gt, noise_for_action, timesteps_for_action)
        noisy_value = (
            self.noise_scheduler_action.add_noise(value, noise_for_value, timesteps_for_action)
            if self.use_value
            else None
        )
        action_mask_expanded = action_mask.expand(-1, noisy_action.shape[1], -1)
        _pred_video_unused, pred_action = self.model(
            hidden_states_video=None,
            hidden_states_action=torch.cat([noisy_action, action_mask_expanded], dim=2).to(self.dtype),
            hidden_states_robostate=torch.cat([state_tokens, action_mask], dim=2).to(self.dtype),
            hidden_states_visual=img_tokens.to(self.dtype),
            timestep_action=timesteps_for_action,
            timestep_video=None,
            encoder_hidden_states=lang_tokens.to(self.dtype),
            value=torch.cat([noisy_value, action_mask], dim=2).to(self.dtype) if self.use_value else None,
            video_kv_mask_mode=video_kv_mask_mode,
            return_dict=False,
        )
        if self.use_value and value is not None:
            pred_value = pred_action[:, -1:]
            pred_action = pred_action[:, :-1]
            target_value = (noise_for_value - value).to(self.dtype)
        else:
            pred_value = None
            target_value = None
        target_action = (noise_for_action - action_gt).to(self.dtype)
        if self.noise_scheduler_type != "ddpm":
            if self.prediction_type == "velocity":
                v_pred = pred_action.to(self.dtype)
                v_pred_value = pred_value.detach().to(self.dtype) if pred_value is not None else None
            elif self.prediction_type == "sample":
                v_pred = (noise_for_action - pred_action).to(self.dtype)
                v_pred_value = (
                    (noise_for_value.detach() - pred_value.detach()).to(self.dtype)
                    if pred_value is not None
                    else None
                )
            elif self.prediction_type == "noise":
                v_pred = (pred_action - action_gt).to(self.dtype)
                v_pred_value = (
                    (pred_value.detach() - value).to(self.dtype)
                    if pred_value is not None
                    else None
                )
            else:
                v_pred = pred_action.to(self.dtype)
                v_pred_value = pred_value.detach().to(self.dtype) if pred_value is not None else None
            loss_action = self._masked_mse(v_pred, target_action, action_valid_mask)
            loss_value = (
                F.mse_loss(v_pred_value, target_value.detach())
                if self.use_value and v_pred_value is not None and target_value is not None
                else None
            )
        else:
            loss_action = self._masked_mse(pred_action.to(self.dtype), target_action, action_valid_mask)
            loss_value = None
        return loss_action, loss_value

    def _video_condition_with_anchor(
        self,
        condition_video_latents: torch.Tensor,
        anchor_video_latent: torch.Tensor,
    ) -> torch.Tensor:
        """Use ``anchor_video_latent`` as the visible frame in WoW-style video condition channels."""
        if condition_video_latents is None or self._video_expert_is_wan22:
            return condition_video_latents
        cond = condition_video_latents.clone()
        cond_channels = cond.shape[1]
        content_channels = min(self._noise_video_channels, anchor_video_latent.shape[1])
        mask_channels = cond_channels - content_channels
        if mask_channels <= 0:
            cond[:, :content_channels, :1] = anchor_video_latent[:, :content_channels, :1]
            return cond
        cond[:, :mask_channels] = 0
        cond[:, :mask_channels, :1] = 1
        cond[:, mask_channels : mask_channels + content_channels] = 0
        cond[:, mask_channels : mask_channels + content_channels, :1] = anchor_video_latent[
            :, :content_channels, :1
        ]
        return cond

    def _clean_action_from_diffusion_pred(
        self,
        *,
        pred_action: torch.Tensor,
        noise_for_action: torch.Tensor,
        noisy_action: torch.Tensor,
    ) -> torch.Tensor:
        if self.prediction_type in ("velocity", "v_prediction"):
            return noise_for_action - pred_action
        if self.prediction_type == "sample":
            return pred_action
        if self.prediction_type == "noise":
            return noisy_action - pred_action
        return noise_for_action - pred_action

    def _compute_goal_joint_key_diffusion_loss(
        self,
        *,
        lang_tokens: torch.Tensor,
        img_tokens: torch.Tensor,
        state_tokens: torch.Tensor,
        action_gt: torch.Tensor,
        action_mask: torch.Tensor,
        action_valid_mask: Optional[torch.Tensor],
        video_latents: torch.Tensor,
        condition_video_latents: torch.Tensor,
        video_kv_mask_mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict horizon key-video and key-action in one coupled diffusion forward."""
        if self._video_expert_is_wan22:
            raise NotImplementedError(
                "Joint key diffusion currently requires WoW-style video conditions."
            )

        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device
        key_len = max(1, min(self.key_action_chunk_size, action_gt.shape[1]))
        key_action_gt = action_gt[:, :key_len]
        key_valid_mask = (
            action_valid_mask[:, :key_len]
            if action_valid_mask is not None
            else None
        )
        key_video_gt = video_latents[:, :, :1]
        current_anchor = video_latents[:, :, -1:]
        current_condition = self._video_condition_with_anchor(
            condition_video_latents[:, :, :1], current_anchor
        )

        (
            noise_for_video,
            noisy_video_hidden,
            timesteps_for_video,
            per_element_loss_mask,
        ) = _video_train_diffusion_prep(
            self.noise_scheduler_video,
            key_video_gt,
            current_condition,
            self.dtype,
            wan22=False,
        )
        noise_for_action = torch.randn(
            key_action_gt.shape,
            dtype=key_action_gt.dtype,
            device=device,
        )
        _sigmas_action, timesteps_for_action = self.noise_scheduler_action.sample(
            batch_size, device=device
        )
        noisy_action = self.noise_scheduler_action.add_noise(
            key_action_gt, noise_for_action, timesteps_for_action
        )
        action_mask_expanded = action_mask.expand(-1, key_len, -1)

        pred_video, pred_action = self.model(
            hidden_states_video=noisy_video_hidden.to(self.dtype),
            hidden_states_action=torch.cat(
                [noisy_action, action_mask_expanded], dim=2
            ).to(self.dtype),
            hidden_states_robostate=torch.cat(
                [state_tokens, action_mask], dim=2
            ).to(self.dtype),
            hidden_states_visual=img_tokens.to(self.dtype),
            timestep_action=timesteps_for_action,
            timestep_video=timesteps_for_video,
            encoder_hidden_states=lang_tokens.to(self.dtype),
            value=None,
            video_kv_mask_mode=video_kv_mask_mode,
            return_dict=False,
        )
        pred_action = pred_action[:, :key_len]

        target_video = (noise_for_video - key_video_gt).to(self.dtype)
        video_loss_elem = F.mse_loss(
            pred_video, target_video, reduction="none"
        )
        if per_element_loss_mask is None:
            loss_key_video = video_loss_elem.mean()
        else:
            loss_key_video = (
                video_loss_elem * per_element_loss_mask
            ).sum() / per_element_loss_mask.sum().clamp_min(1.0)

        target_action = (noise_for_action - key_action_gt).to(self.dtype)
        if self.prediction_type == "velocity":
            action_loss_pred = pred_action.to(self.dtype)
        elif self.prediction_type == "sample":
            action_loss_pred = (noise_for_action - pred_action).to(self.dtype)
        elif self.prediction_type == "noise":
            action_loss_pred = (pred_action - key_action_gt).to(self.dtype)
        else:
            action_loss_pred = pred_action.to(self.dtype)
        loss_key_action = self._masked_mse(
            action_loss_pred, target_action, key_valid_mask
        )

        pred_key_video = (
            noise_for_video - pred_video.to(noise_for_video.dtype)
        )
        pred_key_action = self._clean_action_from_diffusion_pred(
            pred_action=pred_action.to(key_action_gt.dtype),
            noise_for_action=noise_for_action,
            noisy_action=noisy_action,
        )
        return (
            loss_key_video,
            loss_key_action,
            pred_key_video.to(video_latents.dtype),
            pred_key_action,
        )

    def _compute_goal_joint_reverse_diffusion_loss(
        self,
        *,
        lang_tokens: torch.Tensor,
        img_tokens: torch.Tensor,
        state_tokens: torch.Tensor,
        action_gt: torch.Tensor,
        action_mask: torch.Tensor,
        value: Optional[torch.Tensor],
        action_valid_mask: Optional[torch.Tensor],
        video_latents: torch.Tensor,
        condition_video_latents: torch.Tensor,
        pred_key_video: Optional[torch.Tensor],
        pred_key_action: Optional[torch.Tensor],
        video_kv_mask_mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Denoise complete reverse video/action trajectories in one model forward."""
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device
        reverse_condition = condition_video_latents
        if (
            pred_key_video is not None
            and self.use_predicted_key_condition
            and self.reverse_video_key_condition_type
            in ("latent_inpainting", "first_frame", "anchor")
        ):
            reverse_condition = self._video_condition_with_anchor(
                condition_video_latents, pred_key_video
            )

        (
            noise_for_video,
            noisy_video_hidden,
            timesteps_for_video,
            per_element_loss_mask,
        ) = _video_train_diffusion_prep(
            self.noise_scheduler_video,
            video_latents,
            reverse_condition,
            self.dtype,
            wan22=self._video_expert_is_wan22,
        )
        fixed_video_key = (
            pred_key_video is not None
            and self.use_predicted_key_condition
            and self.reverse_video_key_condition_type
            in ("latent_inpainting", "first_frame", "anchor")
        )
        if fixed_video_key and not self._video_expert_is_wan22:
            noisy_video_hidden = noisy_video_hidden.clone()
            noisy_video_hidden[
                :, : self._noise_video_channels, :1
            ] = pred_key_video[:, : self._noise_video_channels, :1].to(
                noisy_video_hidden.dtype
            )
        noise_for_action = torch.randn(
            action_gt.shape, dtype=action_gt.dtype, device=device
        )
        noise_for_value = (
            torch.randn(value.shape, dtype=value.dtype, device=device)
            if self.use_value
            else None
        )
        _sigmas_action, timesteps_for_action = self.noise_scheduler_action.sample(
            batch_size, device=device
        )
        noisy_action = self.noise_scheduler_action.add_noise(
            action_gt, noise_for_action, timesteps_for_action
        )
        fixed_action_key_len = 0
        if (
            pred_key_action is not None
            and self.use_predicted_key_condition
            and self.reverse_action_key_condition_type
            in ("prefix", "replace_first")
        ):
            fixed_action_key_len = min(
                pred_key_action.shape[1], noisy_action.shape[1]
            )
            noisy_action = noisy_action.clone()
            noisy_action[:, :fixed_action_key_len] = pred_key_action[
                :, :fixed_action_key_len
            ].to(noisy_action.dtype)
        noisy_value = (
            self.noise_scheduler_action.add_noise(
                value, noise_for_value, timesteps_for_action
            )
            if self.use_value
            else None
        )
        action_mask_expanded = action_mask.expand(-1, noisy_action.shape[1], -1)

        pred_video, pred_action = self.model(
            hidden_states_video=noisy_video_hidden.to(self.dtype),
            hidden_states_action=torch.cat(
                [noisy_action, action_mask_expanded], dim=2
            ).to(self.dtype),
            hidden_states_robostate=torch.cat(
                [state_tokens, action_mask], dim=2
            ).to(self.dtype),
            hidden_states_visual=img_tokens.to(self.dtype),
            timestep_action=timesteps_for_action,
            timestep_video=timesteps_for_video,
            encoder_hidden_states=lang_tokens.to(self.dtype),
            value=(
                torch.cat([noisy_value, action_mask], dim=2).to(self.dtype)
                if self.use_value
                else None
            ),
            video_kv_mask_mode=video_kv_mask_mode,
            return_dict=False,
        )

        if self.use_value and value is not None:
            pred_value = pred_action[:, -1:]
            pred_action = pred_action[:, :-1]
            target_value = (noise_for_value - value).to(self.dtype)
        else:
            pred_value = None
            target_value = None

        target_action = (noise_for_action - action_gt).to(self.dtype)
        if self.prediction_type == "velocity":
            action_loss_pred = pred_action.to(self.dtype)
            value_loss_pred = (
                pred_value.detach().to(self.dtype)
                if pred_value is not None
                else None
            )
        elif self.prediction_type == "sample":
            action_loss_pred = (noise_for_action - pred_action).to(self.dtype)
            value_loss_pred = (
                (noise_for_value.detach() - pred_value.detach()).to(self.dtype)
                if pred_value is not None
                else None
            )
        elif self.prediction_type == "noise":
            action_loss_pred = (pred_action - action_gt).to(self.dtype)
            value_loss_pred = (
                (pred_value.detach() - value).to(self.dtype)
                if pred_value is not None
                else None
            )
        else:
            action_loss_pred = pred_action.to(self.dtype)
            value_loss_pred = (
                pred_value.detach().to(self.dtype)
                if pred_value is not None
                else None
            )
        reverse_action_valid_mask = action_valid_mask
        if fixed_action_key_len:
            if reverse_action_valid_mask is None:
                reverse_action_valid_mask = action_gt.new_ones(
                    action_gt.shape[0], action_gt.shape[1]
                )
            else:
                reverse_action_valid_mask = reverse_action_valid_mask.clone()
            reverse_action_valid_mask[:, :fixed_action_key_len] = 0
        loss_reverse_action = self._masked_mse(
            action_loss_pred, target_action, reverse_action_valid_mask
        )
        loss_value = (
            F.mse_loss(value_loss_pred, target_value.detach())
            if value_loss_pred is not None and target_value is not None
            else None
        )

        target_video = (noise_for_video - video_latents).to(self.dtype)
        video_loss_elem = F.mse_loss(
            pred_video, target_video, reduction="none"
        )
        if fixed_video_key:
            if per_element_loss_mask is None:
                per_element_loss_mask = torch.ones_like(
                    video_loss_elem, dtype=self.dtype
                )
            else:
                per_element_loss_mask = per_element_loss_mask.clone()
            per_element_loss_mask[:, :, :1] = 0
        if per_element_loss_mask is None:
            loss_reverse_video = video_loss_elem.mean()
        else:
            loss_reverse_video = (
                video_loss_elem * per_element_loss_mask
            ).sum() / per_element_loss_mask.sum().clamp_min(1.0)
        return loss_reverse_action, loss_reverse_video, loss_value

    def _compute_goal_key_action_diffusion_loss(
        self,
        *,
        lang_tokens: torch.Tensor,
        img_tokens: torch.Tensor,
        state_tokens: torch.Tensor,
        action_gt: torch.Tensor,
        action_mask: torch.Tensor,
        action_valid_mask: Optional[torch.Tensor],
        video_kv_mask_mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key_len = max(1, min(self.key_action_chunk_size, action_gt.shape[1]))
        key_action_gt = action_gt[:, :key_len]
        key_valid_mask = action_valid_mask[:, :key_len] if action_valid_mask is not None else None
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device
        noise_for_action = torch.randn(key_action_gt.shape, dtype=key_action_gt.dtype, device=device)
        _sigmas, timesteps = self.noise_scheduler_action.sample(batch_size, device=device)
        noisy_action = self.noise_scheduler_action.add_noise(key_action_gt, noise_for_action, timesteps)
        action_mask_expanded = action_mask.expand(-1, key_len, -1)
        _pred_video_unused, pred_action = self.model(
            hidden_states_video=None,
            hidden_states_action=torch.cat([noisy_action, action_mask_expanded], dim=2).to(self.dtype),
            hidden_states_robostate=torch.cat([state_tokens, action_mask], dim=2).to(self.dtype),
            hidden_states_visual=img_tokens.to(self.dtype),
            timestep_action=timesteps,
            timestep_video=None,
            encoder_hidden_states=lang_tokens.to(self.dtype),
            value=None,
            video_kv_mask_mode=video_kv_mask_mode,
            return_dict=False,
        )
        pred_action = pred_action[:, :key_len]
        target_action = (noise_for_action - key_action_gt).to(self.dtype)
        if self.prediction_type == "velocity":
            v_pred = pred_action.to(self.dtype)
        elif self.prediction_type == "sample":
            v_pred = (noise_for_action - pred_action).to(self.dtype)
        elif self.prediction_type == "noise":
            v_pred = (pred_action - key_action_gt).to(self.dtype)
        else:
            v_pred = pred_action.to(self.dtype)
        loss_key_action = self._masked_mse(v_pred, target_action, key_valid_mask)
        pred_key_action = self._clean_action_from_diffusion_pred(
            pred_action=pred_action.to(key_action_gt.dtype),
            noise_for_action=noise_for_action,
            noisy_action=noisy_action,
        )
        return loss_key_action, pred_key_action

    def _compute_goal_key_video_diffusion_loss(
        self,
        *,
        lang_tokens: torch.Tensor,
        video_latents: torch.Tensor,
        condition_video_latents: torch.Tensor,
        video_kv_mask_mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        current_anchor = video_latents[:, :, -1:]
        current_condition = self._video_condition_with_anchor(condition_video_latents, current_anchor)
        (
            noise_for_video,
            noisy_video_hidden,
            timesteps_for_video,
            per_element_loss_mask,
        ) = _video_train_diffusion_prep(
            self.noise_scheduler_video,
            video_latents,
            current_condition,
            self.dtype,
            wan22=self._video_expert_is_wan22,
        )
        pred_video, pred_action_unused = self.model(
            hidden_states_video=noisy_video_hidden.to(self.dtype),
            hidden_states_action=None,
            hidden_states_robostate=None,
            hidden_states_visual=None,
            timestep_action=None,
            timestep_video=timesteps_for_video,
            encoder_hidden_states=lang_tokens.to(self.dtype),
            value=None,
            video_kv_mask_mode=video_kv_mask_mode,
            return_dict=False,
        )
        del pred_action_unused
        target_video = (noise_for_video - video_latents).to(self.dtype)
        key_loss_elem = F.mse_loss(pred_video[:, :, :1], target_video[:, :, :1], reduction="none")
        if per_element_loss_mask is not None:
            key_mask = per_element_loss_mask[:, :, :1]
            loss_key_video = (key_loss_elem * key_mask).sum() / key_mask.sum().clamp_min(1.0)
        else:
            loss_key_video = key_loss_elem.mean()
        pred_key_video = (noise_for_video[:, :, :1] - pred_video[:, :, :1].to(noise_for_video.dtype))
        return loss_key_video, pred_key_video.to(video_latents.dtype)

    def _compute_goal_reverse_action_diffusion_loss(
        self,
        *,
        lang_tokens: torch.Tensor,
        img_tokens: torch.Tensor,
        state_tokens: torch.Tensor,
        action_gt: torch.Tensor,
        action_mask: torch.Tensor,
        value: Optional[torch.Tensor],
        action_valid_mask: Optional[torch.Tensor],
        pred_key_action: Optional[torch.Tensor],
        video_kv_mask_mode: str,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device
        noise_for_action = torch.randn(action_gt.shape, dtype=action_gt.dtype, device=device)
        noise_for_value = torch.randn(value.shape, dtype=value.dtype, device=device) if self.use_value else None
        _sigmas, timesteps = self.noise_scheduler_action.sample(batch_size, device=device)
        noisy_action = self.noise_scheduler_action.add_noise(action_gt, noise_for_action, timesteps)
        if (
            pred_key_action is not None
            and self.use_predicted_key_condition
            and self.reverse_action_key_condition_type in ("prefix", "replace_first")
        ):
            key_len = min(pred_key_action.shape[1], noisy_action.shape[1])
            noisy_action = noisy_action.clone()
            noisy_action[:, :key_len] = pred_key_action[:, :key_len].to(noisy_action.dtype)
        noisy_value = (
            self.noise_scheduler_action.add_noise(value, noise_for_value, timesteps)
            if self.use_value
            else None
        )
        action_mask_expanded = action_mask.expand(-1, noisy_action.shape[1], -1)
        _pred_video_unused, pred_action = self.model(
            hidden_states_video=None,
            hidden_states_action=torch.cat([noisy_action, action_mask_expanded], dim=2).to(self.dtype),
            hidden_states_robostate=torch.cat([state_tokens, action_mask], dim=2).to(self.dtype),
            hidden_states_visual=img_tokens.to(self.dtype),
            timestep_action=timesteps,
            timestep_video=None,
            encoder_hidden_states=lang_tokens.to(self.dtype),
            value=torch.cat([noisy_value, action_mask], dim=2).to(self.dtype) if self.use_value else None,
            video_kv_mask_mode=video_kv_mask_mode,
            return_dict=False,
        )
        if self.use_value and value is not None:
            pred_value = pred_action[:, -1:]
            pred_action = pred_action[:, :-1]
            target_value = (noise_for_value - value).to(self.dtype)
        else:
            pred_value = None
            target_value = None
        target_action = (noise_for_action - action_gt).to(self.dtype)
        if self.prediction_type == "velocity":
            v_pred = pred_action.to(self.dtype)
            v_pred_value = pred_value.detach().to(self.dtype) if pred_value is not None else None
        elif self.prediction_type == "sample":
            v_pred = (noise_for_action - pred_action).to(self.dtype)
            v_pred_value = (
                (noise_for_value.detach() - pred_value.detach()).to(self.dtype)
                if pred_value is not None
                else None
            )
        elif self.prediction_type == "noise":
            v_pred = (pred_action - action_gt).to(self.dtype)
            v_pred_value = (
                (pred_value.detach() - value).to(self.dtype)
                if pred_value is not None
                else None
            )
        else:
            v_pred = pred_action.to(self.dtype)
            v_pred_value = pred_value.detach().to(self.dtype) if pred_value is not None else None
        loss_action = self._masked_mse(v_pred, target_action, action_valid_mask)
        loss_value = (
            F.mse_loss(v_pred_value, target_value.detach())
            if self.use_value and v_pred_value is not None and target_value is not None
            else None
        )
        return loss_action, loss_value

    def _compute_goal_conditioned_diffusion_loss(
        self,
        *,
        lang_tokens: torch.Tensor,
        img_tokens: torch.Tensor,
        state_tokens: torch.Tensor,
        action_gt: torch.Tensor,
        action_mask: torch.Tensor,
        value: Optional[torch.Tensor],
        video_latents: torch.Tensor,
        condition_video_latents: torch.Tensor,
        action_valid_mask: Optional[torch.Tensor],
        video_kv_mask_mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        zero = action_gt.new_zeros(())
        if self.goal_joint_key_diffusion:
            (
                loss_key_video,
                loss_key_action,
                pred_key_video,
                pred_key_action,
            ) = self._compute_goal_joint_key_diffusion_loss(
                lang_tokens=lang_tokens,
                img_tokens=img_tokens,
                state_tokens=state_tokens,
                action_gt=action_gt,
                action_mask=action_mask,
                action_valid_mask=action_valid_mask,
                video_latents=video_latents,
                condition_video_latents=condition_video_latents,
                video_kv_mask_mode=video_kv_mask_mode,
            )
        elif self.predict_key_video:
            loss_key_video, pred_key_video = self._compute_goal_key_video_diffusion_loss(
                lang_tokens=lang_tokens,
                video_latents=video_latents,
                condition_video_latents=condition_video_latents,
                video_kv_mask_mode=video_kv_mask_mode,
            )
        else:
            loss_key_video = zero
            pred_key_video = video_latents[:, :, :1]
        if not self.goal_joint_key_diffusion:
            if self.predict_key_action:
                loss_key_action, pred_key_action = self._compute_goal_key_action_diffusion_loss(
                    lang_tokens=lang_tokens,
                    img_tokens=img_tokens,
                    state_tokens=state_tokens,
                    action_gt=action_gt,
                    action_mask=action_mask,
                    action_valid_mask=action_valid_mask,
                    video_kv_mask_mode=video_kv_mask_mode,
                )
            else:
                loss_key_action = zero
                key_len = max(1, min(self.key_action_chunk_size, action_gt.shape[1]))
                pred_key_action = action_gt[:, :key_len]
        if self.detach_predicted_key_condition:
            pred_key_video = pred_key_video.detach()
            pred_key_action = pred_key_action.detach()
        if self.goal_joint_reverse_diffusion:
            (
                loss_reverse_action,
                loss_reverse_video,
                loss_value,
            ) = self._compute_goal_joint_reverse_diffusion_loss(
                lang_tokens=lang_tokens,
                img_tokens=img_tokens,
                state_tokens=state_tokens,
                action_gt=action_gt,
                action_mask=action_mask,
                value=value,
                action_valid_mask=action_valid_mask,
                video_latents=video_latents,
                condition_video_latents=condition_video_latents,
                pred_key_video=pred_key_video,
                pred_key_action=pred_key_action,
                video_kv_mask_mode=video_kv_mask_mode,
            )
        else:
            reverse_condition = condition_video_latents
            if self.use_predicted_key_condition and self.reverse_video_key_condition_type in (
                "latent_inpainting",
                "first_frame",
                "anchor",
            ):
                reverse_condition = self._video_condition_with_anchor(
                    condition_video_latents, pred_key_video
                )
            loss_reverse_video = self._compute_video_diffusion_loss(
                lang_tokens=lang_tokens,
                video_latents=video_latents,
                condition_video_latents=reverse_condition,
                video_kv_mask_mode=video_kv_mask_mode,
            )
            loss_reverse_action, loss_value = self._compute_goal_reverse_action_diffusion_loss(
                lang_tokens=lang_tokens,
                img_tokens=img_tokens,
                state_tokens=state_tokens,
                action_gt=action_gt,
                action_mask=action_mask,
                value=value,
                action_valid_mask=action_valid_mask,
                pred_key_action=pred_key_action,
                video_kv_mask_mode=video_kv_mask_mode,
            )
        loss_action = (
            self.key_action_loss_weight * loss_key_action
            + self.reverse_action_loss_weight * loss_reverse_action
        )
        loss_video = (
            self.key_video_loss_weight * loss_key_video
            + self.reverse_video_loss_weight * loss_reverse_video
        )
        self.last_goal_conditioned_metrics = {
            "loss_gc_key_video": loss_key_video.detach(),
            "loss_gc_key_action": loss_key_action.detach(),
            "loss_gc_reverse_video": loss_reverse_video.detach(),
            "loss_gc_reverse_action": loss_reverse_action.detach(),
            "loss_gc_video_main": loss_video.detach(),
            "loss_gc_action_main": loss_action.detach(),
        }
        return loss_action, loss_video, loss_value

    def _compute_stage3_coa_action_latent_loss(
        self,
        *,
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
        action_mask: torch.Tensor,
        action_valid_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        horizon = target_action.shape[1]
        action_mask_expanded = action_mask.expand(-1, horizon, -1)
        pred_encoder_input = torch.cat(
            [pred_action, action_mask_expanded.to(pred_action.dtype)], dim=2
        ).to(self.dtype)
        target_encoder_input = torch.cat(
            [target_action, action_mask_expanded.to(target_action.dtype)], dim=2
        ).to(self.dtype)
        action_encoder = self.model.action_expert.action_encoder
        pred_latent = action_encoder(pred_encoder_input)
        target_latent = action_encoder(target_encoder_input)
        if self.stage3_action_latent_detach_target:
            target_latent = target_latent.detach()
        return self._masked_mse(pred_latent, target_latent, action_valid_mask)

    def _compute_stage3_reverse_ar_action_main_loss(
        self,
        *,
        lang_tokens: torch.Tensor,
        img_tokens: torch.Tensor,
        state_tokens: torch.Tensor,
        action_gt: torch.Tensor,
        action_mask: torch.Tensor,
        action_valid_mask: Optional[torch.Tensor],
        video_kv_mask_mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_action_ar, target_action = self._predict_reverse_ar_action_teacher_forced(
            lang_tokens=lang_tokens,
            img_tokens=img_tokens,
            state_tokens=state_tokens,
            action_gt=action_gt,
            action_mask=action_mask,
            video_kv_mask_mode=video_kv_mask_mode,
        )
        reverse_ar_action_loss = self._masked_mse(pred_action_ar, target_action, action_valid_mask)
        key_valid_mask = (
            action_valid_mask[:, :1]
            if action_valid_mask is not None
            else None
        )
        key_action_loss = self._masked_mse(pred_action_ar[:, :1], target_action[:, :1], key_valid_mask)
        if self.stage3_use_coa_action_loss and self.stage3_action_latent_loss_weight > 0.0:
            action_latent_loss = self._compute_stage3_coa_action_latent_loss(
                pred_action=pred_action_ar,
                target_action=target_action,
                action_mask=action_mask,
                action_valid_mask=action_valid_mask,
            )
        else:
            action_latent_loss = target_action.new_zeros(())
        loss_action = (
            self.stage3_key_action_loss_weight * key_action_loss
            + self.stage3_action_ar_loss_weight * reverse_ar_action_loss
            + self.stage3_action_latent_loss_weight * action_latent_loss
        )
        return loss_action, key_action_loss, reverse_ar_action_loss, action_latent_loss

    def _compute_stage4_action_ar_loss(
        self,
        *,
        lang_tokens: torch.Tensor,
        img_tokens: torch.Tensor,
        state_tokens: torch.Tensor,
        action_gt: torch.Tensor,
        action_mask: torch.Tensor,
        action_valid_mask: Optional[torch.Tensor],
        video_kv_mask_mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_action_ar, target_action = self._predict_reverse_ar_action_teacher_forced(
            lang_tokens=lang_tokens,
            img_tokens=img_tokens,
            state_tokens=state_tokens,
            action_gt=action_gt,
            action_mask=action_mask,
            video_kv_mask_mode=video_kv_mask_mode,
        )
        reverse_ar_action_loss = self._masked_mse(pred_action_ar, target_action, action_valid_mask)
        key_valid_mask = action_valid_mask[:, :1] if action_valid_mask is not None else None
        key_action_loss = self._masked_mse(pred_action_ar[:, :1], target_action[:, :1], key_valid_mask)
        prev_detach = self.stage3_action_latent_detach_target
        self.stage3_action_latent_detach_target = self.stage4_action_latent_detach_target
        try:
            action_latent_loss = self._compute_stage3_coa_action_latent_loss(
                pred_action=pred_action_ar,
                target_action=target_action,
                action_mask=action_mask,
                action_valid_mask=action_valid_mask,
            )
        finally:
            self.stage3_action_latent_detach_target = prev_detach
        loss_action = (
            self.stage4_key_action_loss_weight * key_action_loss
            + self.stage4_action_ar_loss_weight * reverse_ar_action_loss
            + self.stage4_action_latent_loss_weight * action_latent_loss
        )
        return loss_action, key_action_loss, reverse_ar_action_loss, action_latent_loss

    # ========= Train  ============
    def compute_loss(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens, action_gt, action_mask,
                     ctrl_freqs, video_latents, condition_video_latents, value=None,
                     action_valid_mask: Optional[torch.Tensor] = None,
                     video_kv_mask_mode: str = "random") -> torch.Tensor:
        '''
        lang_tokens: (batch_size, lang_len, lang_token_dim)
        lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
            which should be True-False bool tensor.
        img_tokens: (batch_size, img_len, img_token_dim)
        state_tokens: (batch_size, 1, state_token_dim)
        action_gt: (batch_size, horizon, state_token_dim), ground-truth actions for supervision
        action_mask: (batch_size, 1, state_token_dim), a 0-1 **float** tensor.
        ctrl_freqs: (batch_size,), control frequency for each sample.
        
        return: loss_value, a scalar tensor
        '''
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device
        self.last_stage2_metrics = {}
        self.last_stage3_metrics = {}
        self.last_stage4_metrics = {}
        self.last_goal_conditioned_metrics = {}
        stage2_action_gt = action_gt
        stage2_action_valid_mask = action_valid_mask

        if self.goal_conditioned_wam:
            return self._compute_goal_conditioned_diffusion_loss(
                lang_tokens=lang_tokens,
                img_tokens=img_tokens,
                state_tokens=state_tokens,
                action_gt=action_gt,
                action_mask=action_mask,
                value=value,
                video_latents=video_latents,
                condition_video_latents=condition_video_latents,
                action_valid_mask=action_valid_mask,
                video_kv_mask_mode=video_kv_mask_mode,
            )

        if self.stage4_full_coa_ar:
            if self.stage4_replace_action_diffusion:
                loss_action, key_action_loss, reverse_ar_action_loss, action_latent_loss = (
                    self._compute_stage4_action_ar_loss(
                        lang_tokens=lang_tokens,
                        img_tokens=img_tokens,
                        state_tokens=state_tokens,
                        action_gt=action_gt,
                        action_mask=action_mask,
                        action_valid_mask=action_valid_mask,
                        video_kv_mask_mode=video_kv_mask_mode,
                    )
                )
                action_metrics = {
                    "loss_stage4_key_action": key_action_loss.detach(),
                    "loss_stage4_action_ar": reverse_ar_action_loss.detach(),
                    "loss_stage4_action_latent": action_latent_loss.detach(),
                    "loss_stage4_action_main": loss_action.detach(),
                }
                loss_value = None
            else:
                loss_action, loss_value = self._compute_action_diffusion_loss(
                    lang_tokens=lang_tokens,
                    img_tokens=img_tokens,
                    state_tokens=state_tokens,
                    action_gt=action_gt,
                    action_mask=action_mask,
                    value=value,
                    action_valid_mask=action_valid_mask,
                    video_kv_mask_mode=video_kv_mask_mode,
                )
                action_metrics = {
                    "loss_stage4_action_diffusion_fallback": loss_action.detach(),
                }

            if self.stage4_replace_video_diffusion:
                loss_video, keyframe_loss, video_ar_loss = self._compute_stage4_video_ar_loss(
                    lang_tokens=lang_tokens,
                    state_tokens=state_tokens,
                    video_latents=video_latents,
                )
                video_metrics = {
                    "loss_stage4_keyframe": keyframe_loss.detach(),
                    "loss_stage4_video_ar": video_ar_loss.detach(),
                    "loss_stage4_video_main": loss_video.detach(),
                }
            else:
                loss_video = self._compute_video_diffusion_loss(
                    lang_tokens=lang_tokens,
                    video_latents=video_latents,
                    condition_video_latents=condition_video_latents,
                    video_kv_mask_mode=video_kv_mask_mode,
                )
                video_metrics = {
                    "loss_stage4_video_diffusion_fallback": loss_video.detach(),
                }

            self.last_stage4_metrics = {}
            self.last_stage4_metrics.update(action_metrics)
            self.last_stage4_metrics.update(video_metrics)
            return loss_action, loss_video, loss_value

        if self.generation_mode == "reverse_ar" and self.stage3_replace_video_diffusion:
            raise NotImplementedError(
                "Stage 3 video reverse-AR replacement is not implemented yet. "
                "Keep stage3_replace_video_diffusion=false to use video diffusion "
                "as an explicit fallback while developing the video AR decoder."
            )

        if self.generation_mode == "reverse_ar" and self.stage3_replace_action_diffusion:
            loss_action, key_action_loss, reverse_ar_action_loss, action_latent_loss = (
                self._compute_stage3_reverse_ar_action_main_loss(
                    lang_tokens=lang_tokens,
                    img_tokens=img_tokens,
                    state_tokens=state_tokens,
                    action_gt=action_gt,
                    action_mask=action_mask,
                    action_valid_mask=action_valid_mask,
                    video_kv_mask_mode=video_kv_mask_mode,
                )
            )
            loss_video = self._compute_video_diffusion_loss(
                lang_tokens=lang_tokens,
                video_latents=video_latents,
                condition_video_latents=condition_video_latents,
                video_kv_mask_mode=video_kv_mask_mode,
            )
            self.last_stage3_metrics = {
                "loss_stage3_key_action": key_action_loss.detach(),
                "loss_stage3_reverse_ar_action": reverse_ar_action_loss.detach(),
                "loss_stage3_action_latent": action_latent_loss.detach(),
                "loss_stage3_action_main": loss_action.detach(),
                "loss_stage3_video_diffusion_fallback": loss_video.detach(),
            }
            return loss_action, loss_video, None

        # Sample noise that we'll add to the actions
        noise_for_action = torch.randn(action_gt.shape, dtype=action_gt.dtype, device=device)
        noise_for_value = torch.randn(value.shape, dtype=value.dtype, device=device) if self.use_value else None
        # Sample random diffusion timesteps
        sigmas_for_action, timesteps_for_action = self.noise_scheduler_action.sample(batch_size, device=device)
        # Add noise to the clean actions according to the noise magnitude at each timestep
        noisy_action = self.noise_scheduler_action.add_noise(action_gt, noise_for_action, timesteps_for_action)
        noisy_value = self.noise_scheduler_action.add_noise(value, noise_for_value, timesteps_for_action) if self.use_value else None
        (
            noise_for_video,
            noisy_video_hidden,
            timesteps_for_video,
            per_element_loss_mask,
        ) = _video_train_diffusion_prep(
            self.noise_scheduler_video,
            video_latents,
            condition_video_latents,
            self.dtype,
            wan22=self._video_expert_is_wan22,
        )
        # WoW: _video_train_diffusion_prep 已拼接 condition → noisy_video_hidden = [B, 36, F, H, W]
        # Wan2.2: _video_train_diffusion_prep 返回纯 noisy → noisy_video_hidden = [B, 48, F, H, W]


        action_mask_expanded = action_mask.expand(-1, noisy_action.shape[1], -1)
        pred_video, pred_action = self.model(
            hidden_states_video=noisy_video_hidden.to(self.dtype),
            hidden_states_action=torch.cat([noisy_action, action_mask_expanded], dim=2).to(self.dtype),
            hidden_states_robostate=torch.cat([state_tokens, action_mask], dim=2).to(self.dtype),
            hidden_states_visual=img_tokens.to(self.dtype),
            timestep_action=timesteps_for_action,
            timestep_video=timesteps_for_video,
            encoder_hidden_states=lang_tokens.to(self.dtype),
            value=torch.cat([noisy_value, action_mask], dim=2).to(self.dtype) if self.use_value else None,
            video_kv_mask_mode=video_kv_mask_mode,
            return_dict=False,
        )

        if self.use_value and value is not None:
            pred_value = pred_action[:, -1:]
            target_value = (noise_for_value - value).to(self.dtype)
            noise_for_action = torch.cat([noise_for_action, noise_for_value], dim=1)
            action_gt = torch.cat([action_gt, value], dim=1)
            if action_valid_mask is not None:
                value_valid_mask = action_valid_mask.new_ones(action_valid_mask.shape[0], 1)
                action_valid_mask = torch.cat([action_valid_mask, value_valid_mask], dim=1)
        target_action = (noise_for_action - action_gt).to(self.dtype) #/ sigmas_for_action.unsqueeze(-1).unsqueeze(-1)).to(self.dtype)
        target_video = (noise_for_video - video_latents).to(self.dtype)
        if self.noise_scheduler_type != "ddpm":
            # target = (noise - action_gt).to(self.dtype)
            # import pdb; pdb.set_trace()
            if self.prediction_type == 'velocity':
                v_pred = pred_action.to(self.dtype)
                if self.use_value:
                    v_pred_value = pred_value.detach().to(self.dtype)
            elif self.prediction_type == 'sample':
                v_pred = (noise_for_action - pred_action).to(self.dtype) #/ sigmas_for_action.unsqueeze(-1).unsqueeze(-1)).to(self.dtype)
                if self.use_value:
                    v_pred_value = (noise_for_value.detach() - pred_value.detach()).to(self.dtype)
            elif self.prediction_type == 'noise':
                v_pred = (pred_action - action_gt).to(self.dtype)
                if self.use_value:
                    v_pred_value = (pred_value.detach() - value).to(self.dtype)
            if action_valid_mask is not None:
                loss_action = self._masked_mse(v_pred, target_action, action_valid_mask)
            else:
                loss_action = F.mse_loss(v_pred, target_action)
            loss_value = F.mse_loss(v_pred_value, target_value.detach()) if self.use_value else None
        if per_element_loss_mask is None:
            loss_video = F.mse_loss(pred_video, target_video)
        else:
            loss_video = (
                F.mse_loss(pred_video, target_video, reduction="none") * per_element_loss_mask
            ).sum() / per_element_loss_mask.sum()
        if (
            self.stage2_enable_reverse_ar_action
            and (
                self.stage2_key_action_loss_weight > 0.0
                or self.stage2_reverse_ar_action_loss_weight > 0.0
            )
        ):
            key_action_loss, reverse_ar_action_loss = self._compute_stage2_reverse_ar_action_loss(
                lang_tokens=lang_tokens,
                img_tokens=img_tokens,
                state_tokens=state_tokens,
                action_gt=stage2_action_gt,
                action_mask=action_mask,
                action_valid_mask=stage2_action_valid_mask,
                video_kv_mask_mode=video_kv_mask_mode,
            )
            loss_action = (
                loss_action
                + self.stage2_key_action_loss_weight * key_action_loss
                + self.stage2_reverse_ar_action_loss_weight * reverse_ar_action_loss
            )
            self.last_stage2_metrics = {
                "loss_stage2_key_action": key_action_loss.detach(),
                "loss_stage2_reverse_ar_action": reverse_ar_action_loss.detach(),
            }
        # loss = loss_action + loss_video
        return loss_action, loss_video, loss_value

    # ========= Inference  ============
    @torch.no_grad()
    def conditional_sample_reverse_ar_action_only(
        self,
        lang_tokens: torch.Tensor,
        lang_attn_mask: torch.Tensor,
        img_tokens: torch.Tensor,
        state_tokens: torch.Tensor,
        action_mask: torch.Tensor,
        ctrl_freqs: torch.Tensor,
    ):
        """Autoregressively sample the reverse-time action sequence.

        This is the Stage 3 action-only path. It reuses the existing action
        expert with causal self-attention instead of the action diffusion
        scheduler. The returned trajectory stays in the training view; callers
        that execute forward in time should reverse it when
        ``reverse_world_order=true``.
        """
        del lang_attn_mask, ctrl_freqs
        device = lang_tokens.device
        dtype = lang_tokens.dtype
        state_tokens = state_tokens.to(device, dtype=dtype)
        action_mask = action_mask.to(device, dtype=dtype)
        img_tokens = img_tokens.to(device, dtype=dtype)
        batch_size = state_tokens.shape[0]
        generated = torch.zeros(
            batch_size,
            self.pred_horizon,
            self.state_token_dim,
            device=device,
            dtype=dtype,
        )
        action_mask_expanded = action_mask.expand(-1, self.pred_horizon, -1)
        timestep_action = torch.zeros(batch_size, device=device, dtype=torch.long)
        ar_value = None
        if self.use_value:
            zero_value = torch.zeros(batch_size, 1, self.state_token_dim, device=device, dtype=dtype)
            ar_value = torch.cat([zero_value, action_mask], dim=2)

        for step_idx in range(self.pred_horizon):
            pred_video_unused, pred_action_ar = self.model(
                hidden_states_video=None,
                hidden_states_action=torch.cat([generated, action_mask_expanded], dim=2).to(self.dtype),
                hidden_states_robostate=torch.cat([state_tokens, action_mask], dim=2).to(self.dtype),
                hidden_states_visual=img_tokens.to(self.dtype),
                timestep_action=timestep_action,
                timestep_video=None,
                encoder_hidden_states=lang_tokens.to(self.dtype),
                value=ar_value.to(self.dtype) if ar_value is not None else None,
                video_kv_mask_mode="off",
                return_dict=False,
            )
            del pred_video_unused
            next_action = pred_action_ar[:, step_idx : step_idx + 1].to(dtype)
            generated[:, step_idx : step_idx + 1] = next_action * action_mask

        return {
            "pred_trajectory": generated * action_mask_expanded,
            "pred_video": None,
            "pred_value": None,
        }

    @torch.no_grad()
    def conditional_sample_stage4_video_ar(
        self,
        *,
        lang_tokens: torch.Tensor,
        state_tokens: Optional[torch.Tensor],
        video_latents: torch.Tensor,
    ) -> torch.Tensor:
        device = video_latents.device
        dtype = video_latents.dtype
        batch_size, _channels, frames, height, width = video_latents.shape
        generated = torch.zeros(
            batch_size,
            self._noise_video_channels,
            frames,
            height,
            width,
            device=device,
            dtype=dtype,
        )
        cond = self._stage4_video_condition(
            lang_tokens=lang_tokens,
            state_tokens=state_tokens,
            dtype=dtype,
        )
        sos = self.stage4_video_sos.to(device=device, dtype=dtype).expand(
            batch_size, self._noise_video_channels, 1, height, width
        )
        for step_idx in range(frames):
            shifted = torch.cat([sos, generated[:, :, :-1]], dim=2)
            head_dtype = next(self.stage4_video_ar_head.parameters()).dtype
            pred_video = self.stage4_video_ar_head(shifted.to(head_dtype)).to(dtype) + cond
            generated[:, :, step_idx : step_idx + 1] = pred_video[:, :, step_idx : step_idx + 1]
        return generated

    @torch.no_grad()
    def conditional_sample_stage4_full_coa_ar(
        self,
        lang_tokens: torch.Tensor,
        lang_attn_mask: torch.Tensor,
        img_tokens: torch.Tensor,
        state_tokens: torch.Tensor,
        action_mask: torch.Tensor,
        ctrl_freqs: torch.Tensor,
        video_latents: torch.Tensor,
        condition_video_latents: torch.Tensor,
        seed: Optional[int] = 42,
        num_inference_timesteps: Optional[int] = None,
    ):
        del seed
        if self.stage4_replace_video_diffusion:
            pred_video = self.conditional_sample_stage4_video_ar(
                lang_tokens=lang_tokens,
                state_tokens=state_tokens,
                video_latents=video_latents,
            )
        else:
            pred_video, _t_last = self.conditional_sample_video_only(
                lang_tokens=lang_tokens,
                video_latents=video_latents,
                condition_video_latents=condition_video_latents,
            )

        if self.stage4_replace_action_diffusion:
            action_out = self.conditional_sample_reverse_ar_action_only(
                lang_tokens=lang_tokens,
                lang_attn_mask=lang_attn_mask,
                img_tokens=img_tokens,
                state_tokens=state_tokens,
                action_mask=action_mask,
                ctrl_freqs=ctrl_freqs,
            )
        else:
            action_out = self.conditional_sample_action_only(
                lang_tokens=lang_tokens,
                lang_attn_mask=lang_attn_mask,
                img_tokens=img_tokens,
                state_tokens=state_tokens,
                action_mask=action_mask,
                ctrl_freqs=ctrl_freqs,
            )
        action_out["pred_video"] = pred_video
        return action_out

    def predict_action(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens, action_mask, ctrl_freqs, 
                       video_latents, condition_video_latents, video_only=False, action_only=False, 
                       sample_batch_size=1, video_denoise_steps: Optional[int] = None,
                       video_then_action: bool = False, seed: Optional[int] = 42,
                       video_then_action_guidance_scale: float = 1.0, video_then_action_use_cfg: bool = False,
                       use_mean_flow: bool = False, num_inference_timesteps: Optional[int] = None):
        '''
        lang_tokens: (batch_size, lang_len, lang_token_dim)
        lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
            which should be True-False bool tensor.
        img_tokens: (batch_size, img_len, img_token_dim)
        state_tokens: (batch_size, 1, state_token_dim)
        action_mask: (batch_size, 1, action_dim),
            which should be a 0-1 **float** tensor.
        ctrl_freqs: (batch_size,), control frequency for each sample.
        video_denoise_steps: ``video_only`` / ``video_then_action`` 时传给视频去噪阶段的步数上限。
        video_then_action: 为 True 时调用 ``conditional_sample_video_then_action``（先视频去噪再动作去噪），
            与 ``video_only`` / ``action_only`` 互斥；若同时为 True，以 ``video_then_action`` 为准。
        seed: ``video_then_action`` 时传给两阶段采样的随机种子。
        video_then_action_guidance_scale / video_then_action_use_cfg: 第二阶段动作分支 CFG 参数。
        num_inference_timesteps: 推理扩散步数；``None`` 表示使用配置里的 ``self.num_inference_timesteps``。

        return: (batch_size, horizon, action_dim), predicted action sequence
        '''
        # Prepare the state and conditions
        # state_tokens = torch.cat([state_tokens, action_mask], dim=2)
        # device = state_tokens.device
        # batch_size = state_tokens.shape[0]
        if state_tokens.shape[0] < sample_batch_size:
            # expand the state_tokens to the sample_batch_size
            state_tokens = state_tokens.repeat(sample_batch_size, 1, 1)
            lang_tokens = lang_tokens.repeat(sample_batch_size, 1, 1)
            img_tokens = img_tokens.repeat(sample_batch_size, 1, 1)
            action_mask = action_mask.repeat(sample_batch_size, 1, 1)
            ctrl_freqs = ctrl_freqs.repeat(sample_batch_size)
            if video_latents is not None:
                video_latents = video_latents.repeat(sample_batch_size, 1, 1, 1, 1)
            if condition_video_latents is not None:
                condition_video_latents = condition_video_latents.repeat(sample_batch_size, 1, 1, 1, 1)

        if self.goal_conditioned_wam and self.goal_conditioned_multistep_inference:
            return self.conditional_sample_goal_conditioned_reverse_diffusion(
                lang_tokens=lang_tokens,
                lang_attn_mask=lang_attn_mask,
                img_tokens=img_tokens,
                state_tokens=state_tokens,
                action_mask=action_mask,
                ctrl_freqs=ctrl_freqs,
                video_latents=video_latents,
                condition_video_latents=condition_video_latents,
                video_only=video_only,
                action_only=action_only,
                seed=seed,
                num_inference_timesteps=num_inference_timesteps,
            )

        # (1) 准备输入
        # (2) 调用扩散采样
        if self.stage4_full_coa_ar and not video_then_action:
            if video_only and not action_only:
                if self.stage4_replace_video_diffusion:
                    video_pred = self.conditional_sample_stage4_video_ar(
                        lang_tokens=lang_tokens,
                        state_tokens=state_tokens,
                        video_latents=video_latents,
                    )
                else:
                    video_pred, _t_last = self.conditional_sample_video_only(
                        lang_tokens,
                        video_latents,
                        condition_video_latents,
                        denoise_steps=video_denoise_steps,
                        num_inference_timesteps=num_inference_timesteps,
                    )
                return {
                    "pred_trajectory": None,
                    "pred_video": video_pred,
                    "pred_value": None,
                }
            if action_only and not video_only:
                if self.stage4_replace_action_diffusion:
                    return self.conditional_sample_reverse_ar_action_only(
                        lang_tokens=lang_tokens,
                        lang_attn_mask=lang_attn_mask,
                        img_tokens=img_tokens,
                        state_tokens=state_tokens,
                        action_mask=action_mask,
                        ctrl_freqs=ctrl_freqs,
                    )
                return self.conditional_sample_action_only(
                    lang_tokens,
                    lang_attn_mask,
                    img_tokens,
                    state_tokens,
                    action_mask,
                    ctrl_freqs,
                    num_inference_timesteps=num_inference_timesteps,
                )
            return self.conditional_sample_stage4_full_coa_ar(
                lang_tokens=lang_tokens,
                lang_attn_mask=lang_attn_mask,
                img_tokens=img_tokens,
                state_tokens=state_tokens,
                action_mask=action_mask,
                ctrl_freqs=ctrl_freqs,
                video_latents=video_latents,
                condition_video_latents=condition_video_latents,
                seed=seed,
                num_inference_timesteps=num_inference_timesteps,
            )
        if self.generation_mode == "reverse_ar" and action_only and not video_only:
            return self.conditional_sample_reverse_ar_action_only(
                lang_tokens=lang_tokens,
                lang_attn_mask=lang_attn_mask,
                img_tokens=img_tokens,
                state_tokens=state_tokens,
                action_mask=action_mask,
                ctrl_freqs=ctrl_freqs,
            )
        if video_then_action:
            out = self.conditional_sample_video_then_action(
                lang_tokens=lang_tokens,
                lang_attn_mask=lang_attn_mask,
                img_tokens=img_tokens,
                state_tokens=state_tokens,
                action_mask=action_mask,
                ctrl_freqs=ctrl_freqs,
                video_latents=video_latents,
                condition_video_latents=condition_video_latents,
                seed=seed,
                guidance_scale_action=video_then_action_guidance_scale,
                use_cfg_action=video_then_action_use_cfg,
                video_denoise_steps=video_denoise_steps,
                num_inference_timesteps=num_inference_timesteps,
                use_mean_flow=use_mean_flow,
            )
        elif video_only and not action_only:
            video_pred, _t_last = self.conditional_sample_video_only(
                lang_tokens,
                video_latents,
                condition_video_latents,
                denoise_steps=video_denoise_steps,
                num_inference_timesteps=num_inference_timesteps,
            )
            out = {
                "pred_trajectory": None,
                "pred_video": video_pred,
                "pred_value": None,
            }
        elif action_only and not video_only:
            out = self.conditional_sample_action_only(
                lang_tokens,
                lang_attn_mask,
                img_tokens,
                state_tokens,
                action_mask,
                ctrl_freqs,
                num_inference_timesteps=num_inference_timesteps,
            )
        else:
            out = self.conditional_sample(
            lang_tokens=lang_tokens,       # [1, 512, 4096]
            lang_attn_mask=lang_attn_mask, # [1, 512]
            img_tokens=img_tokens,         # [1, 4374, 1152]
            state_tokens=state_tokens,     # [1, 1, 14]
            action_mask=action_mask,       # torch.Size([1, 1, 14])
            ctrl_freqs=ctrl_freqs,         # tensor([25], device='cuda:0')
            video_latents=video_latents,   # torch.Size([1, 20, 13, 30, 40])
            condition_video_latents=condition_video_latents, # torch.Size([1, 20, 13, 30, 40])
            use_mean_flow=use_mean_flow,
            num_inference_timesteps=num_inference_timesteps,
        )

        return out


    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.compute_loss(*args, **kwargs)
