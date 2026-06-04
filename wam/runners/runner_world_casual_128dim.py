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
        reverse_ar_loss = self._masked_mse(pred_action_ar, target_action, action_valid_mask)
        key_valid_mask = (
            action_valid_mask[:, :1]
            if action_valid_mask is not None
            else None
        )
        key_action_loss = self._masked_mse(pred_action_ar[:, :1], target_action[:, :1], key_valid_mask)
        return key_action_loss, reverse_ar_loss

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
        stage2_action_gt = action_gt
        stage2_action_valid_mask = action_valid_mask

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
            if video_latents is not None:
                video_latents = video_latents.repeat(sample_batch_size, 1, 1, 1, 1)
            if condition_video_latents is not None:
                condition_video_latents = condition_video_latents.repeat(sample_batch_size, 1, 1, 1, 1)

        # (1) 准备输入
        # (2) 调用扩散采样
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
