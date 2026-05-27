# coding=utf-8
"""仅视频 Wan Transformer Runner。

- ``video_variant='wow'``：WoW-1.3B 风格（16 通道 latent、标量 timestep）。
- ``video_variant='wan22'``：Wan2.2 TI2V 5B（48 通道、时空 timestep 与首帧 mask）。

``runner_video_wan2_2`` 模块仅 re-export 默认 ``video_variant='wan22'`` 的子类；旧名 ``rdt_runner_*`` 仍可由 ``runner_registry`` 解析。
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from wam.models.hub_mixin import CompatiblePyTorchModelHubMixin
from wam.models.mvwam.video_expert import WanTransformer3DModel
from wam.models.mvwam.world_stack_registry import (
    pick_video_base_model_str,
    resolve_video_base_model_path,
)
from wam.samples.noise_scheduler import NoiseTimestepSampler

from diffusers.schedulers import UniPCMultistepScheduler


VideoVariant = Literal["wow", "wan22"]

_WAM_ROOT = Path(__file__).resolve().parents[2]
_DEMO_UNCOND_PT = str(_WAM_ROOT / "data" / "uncond.pt")



def _video_train_diffusion_prep(
    noise_scheduler,
    video_latents: torch.Tensor,
    condition_video_latents: torch.Tensor,
    weight_dtype: torch.dtype,
    *,
    wan22: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """采样视频扩散训练用的时间步并加噪。


    返回 ``(noise_for_video, noisy_hidden_states, diffusion_timesteps, per_element_loss_mask)``。
    WoW：``noisy_hidden_states`` 为 noisy 视频 latent 与 ``condition_video_latents`` 在通道维拼接；
    ``diffusion_timesteps`` 为 ``(B,)``；``per_element_loss_mask`` 为 ``None``。
    Wan2.2：``noisy_hidden_states`` 仅为加噪后的视频 latent；``diffusion_timesteps`` 为时空展开并 flatten；
    ``per_element_loss_mask`` 用于首帧不参与 loss。
    """
    batch_size = video_latents.shape[0]
    device = video_latents.device
    _, sampled_timesteps_per_batch = noise_scheduler.sample(batch_size, device=device)
    noise_for_video = torch.randn(video_latents.shape, dtype=video_latents.dtype, device=device)
    if not wan22:
        noisy_video_latents = noise_scheduler.add_noise(
            video_latents, noise_for_video, sampled_timesteps_per_batch
        )
        noisy_hidden_states = torch.concat([noisy_video_latents, condition_video_latents], dim=1)
        return noise_for_video, noisy_hidden_states, sampled_timesteps_per_batch, None

    timesteps_for_mask = sampled_timesteps_per_batch.clone()
    timesteps_broadcast = sampled_timesteps_per_batch[:, None, None, None, None].expand_as(video_latents)
    per_element_loss_mask = torch.ones_like(video_latents)
    per_element_loss_mask[:, :, :1] = 0
    timesteps_broadcast = timesteps_broadcast * per_element_loss_mask
    noisy_hidden_states = noise_scheduler.add_noise(
        video_latents, noise_for_video, timesteps_broadcast
    )
    per_element_loss_mask = per_element_loss_mask.to(weight_dtype)
    latent_num_frames = video_latents.shape[2]
    latent_patch_height = video_latents.shape[3] // 2
    latent_patch_width = video_latents.shape[4] // 2
    diffusion_timesteps = (
        timesteps_for_mask[:, None, None, None]
        .repeat(1, latent_num_frames, latent_patch_height, latent_patch_width)
        * per_element_loss_mask[:, 0, :, :1, :1]
    )
    diffusion_timesteps = diffusion_timesteps.flatten(1)
    return noise_for_video, noisy_hidden_states, diffusion_timesteps, per_element_loss_mask


def _video_inference_timestep_tensor(
    timestep_scalar,
    device: torch.device,
    batch_size: int,
    *,
    wan22: bool,
    latent_spatial_shape: tuple[int, int, int] | None,
) -> torch.Tensor:
    """单步采样用 ``timestep``：WoW 为 ``[t]`` long；Wan2.2 为时空网格 flatten。"""
    if not wan22:
        return torch.tensor([timestep_scalar], dtype=torch.long, device=device)
    assert latent_spatial_shape is not None
    latent_num_frames, latent_patch_height, latent_patch_width = latent_spatial_shape
    diffusion_timestep_grid = torch.ones(batch_size, device=device) * timestep_scalar
    diffusion_timestep_grid = diffusion_timestep_grid[:, None, None, None].repeat(
        1, latent_num_frames, latent_patch_height, latent_patch_width
    )
    diffusion_timestep_grid[:, :1] = 0
    return diffusion_timestep_grid.flatten(1)


def _resolve_video_runner_pretrained_base(
    video_base_model,
    config,
    *,
    variant: VideoVariant,
) -> str:
    """``VIDEO_BASE_MODEL`` / kwargs 优先，再环境变量，最后按 variant 使用注册表默认键。"""
    selected_video_base_model_key = pick_video_base_model_str(video_base_model, config)
    if selected_video_base_model_key is not None:
        return resolve_video_base_model_path(selected_video_base_model_key)
    if variant == "wan22":
        wan22_override_root = os.environ.get("WAM_WAN22_VIDEO_BASE", "").strip()
        if wan22_override_root:
            return wan22_override_root
        return resolve_video_base_model_path("Wan2.2-TI2V-5B-Diffusers")
    wow_override_root = os.environ.get("WAM_WOW_VIDEO_BASE", "").strip()
    if wow_override_root:
        return wow_override_root
    return resolve_video_base_model_path(None)


def _config_bool(config, key: str, default: bool = False) -> bool:
    if config is None:
        return default
    if hasattr(config, "get"):
        value = config.get(key, default)
    else:
        value = getattr(config, key, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


class VideoRunner(nn.Module, CompatiblePyTorchModelHubMixin):
    def __init__(
        self,
        *,
        pred_horizon,
        config,
        img_cond_len,
        img_pos_embed_config,
        sample_fps=1,
        pretrained_video_expert_path="none",
        dtype=torch.bfloat16,
        video_base_model=None,
        video_variant: VideoVariant = "wow",
        uncond_lang_embed_path: str | None = None,
    ):
        super().__init__()
        self._uncond_lang_embed_path = uncond_lang_embed_path or _DEMO_UNCOND_PT
        self._video_variant: VideoVariant = "wan22" if str(video_variant).lower() in ("wan22", "wan2.2", "wan2_2") else "wow"
        pretrained_diffusers_root = _resolve_video_runner_pretrained_base(
            video_base_model, config, variant=self._video_variant
        )
        self.model = WanTransformer3DModel.from_pretrained(
            pretrained_diffusers_root,
            subfolder="transformer",
            torch_dtype=dtype,
            local_files_only=True,
        )
        video_frame_causal = _config_bool(config, "video_frame_causal_self_attn", False) or _config_bool(
            config, "frame_causal_self_attention", False
        )
        self.model.frame_causal_self_attention = video_frame_causal
        if hasattr(self.model, "register_to_config"):
            self.model.register_to_config(frame_causal_self_attention=video_frame_causal)
        _should_load_ckpt = (
            pretrained_video_expert_path != "none"
            and os.path.exists(pretrained_video_expert_path)
        )
        if self._video_variant == "wan22" and _should_load_ckpt:
            if "Wan2.2-TI2V-5B-Diffusers" in pretrained_video_expert_path:
                _should_load_ckpt = False
        if _should_load_ckpt:
            pt = os.path.join(pretrained_video_expert_path, "pytorch_model/mp_rank_00_model_states.pt")
            ckpt = torch.load(pt, map_location="cpu")
            state_dict = {}
            for key, value in ckpt["module"].items():
                if "model." in key:
                    state_dict[key.replace("model.", "")] = value
            r = self.model.load_state_dict(state_dict, strict=False)
            assert len(r.missing_keys) == 0, f"Missing keys: {r.missing_keys}"

        self.dtype = dtype
        noise_scheduler_config = config["noise_scheduler"]

        self.noise_scheduler_video = NoiseTimestepSampler(
            num_train_timesteps=1000,
            distribution_type="normal",
            shift=5.0,
        )
        self.noise_scheduler_sample_video = UniPCMultistepScheduler.from_pretrained(
            pretrained_diffusers_root,
            subfolder="scheduler",
            torch_dtype=dtype,
            local_files_only=True,
        )

        self.num_train_timesteps = noise_scheduler_config["num_train_timesteps"]
        self.num_inference_timesteps = noise_scheduler_config["num_inference_timesteps"]
        self.prediction_type = noise_scheduler_config["prediction_type"]
        self.noise_scheduler_type = noise_scheduler_config["type"]

        self.pred_horizon = pred_horizon
        self.video_pred_horizon = pred_horizon // sample_fps

        tag = "Wan2.2" if self._video_variant == "wan22" else "WoW"
        print("VideoRunner(%s) params: %e" % (tag, sum(p.numel() for p in self.model.parameters())))

    def conditional_sample(
        self,
        lang_tokens,
        video_latents,
        condition_video_latents,
        guidance: float = 5.0,
    ):
        """视频扩散采样。

        ``guidance`` 与 1 在浮点容差内相等时**关闭** CFG（仅一次条件速度预测）；否则对无条件分支再前向一次，
        并按 ``v = v_uncond + guidance * (v_cond - v_uncond)`` 合成（与原先固定 ``5.0`` 的语义一致，默认 ``guidance=5.0``）。
        """
        device = lang_tokens.device
        batch_size, _channels, num_frames, height, width = video_latents.shape
        wan22 = self._video_variant == "wan22"
        use_classifier_free_guidance = not math.isclose(float(guidance), 1.0, rel_tol=0.0, abs_tol=1e-5)
        noise_latent_channels = 48 if wan22 else 16
        noisy_video = torch.randn(
            (batch_size, noise_latent_channels, num_frames, height, width),
            dtype=video_latents.dtype,
            device=device,
        )
        self.noise_scheduler_sample_video.set_timesteps(self.num_inference_timesteps, device=device)
        inference_timesteps = self.noise_scheduler_sample_video.timesteps
        unconditional_language_embeds = None
        if use_classifier_free_guidance:
            unconditional_language_embeds = torch.load(
                self._uncond_lang_embed_path, map_location="cpu"
            ).to(device, torch.bfloat16)

        latent_spatial_shape_for_timestep = None
        if wan22:
            latent_spatial_shape_for_timestep = (
                noisy_video.shape[2],
                noisy_video.shape[3] // 2,
                noisy_video.shape[4] // 2,
            )

        for _, current_timestep in enumerate(inference_timesteps):
            if wan22:
                noisy_video[:, :, :1] = video_latents[:, :, :1]
                noisy_hidden_states = noisy_video
            else:
                noisy_hidden_states = torch.concat([noisy_video, condition_video_latents], dim=1)
            diffusion_timestep = _video_inference_timestep_tensor(
                current_timestep,
                device,
                batch_size,
                wan22=wan22,
                latent_spatial_shape=latent_spatial_shape_for_timestep,
            )
            noisy_hidden_states = noisy_hidden_states.to(self.dtype)
            language_hidden_states = lang_tokens.to(self.dtype)
            video_velocity_cond = self.model(
                hidden_states=noisy_hidden_states,
                timestep=diffusion_timestep,
                encoder_hidden_states=language_hidden_states,
                return_dict=False,
            )[0]
            if use_classifier_free_guidance:
                assert unconditional_language_embeds is not None
                video_velocity_uncond = self.model(
                    hidden_states=noisy_hidden_states,
                    timestep=diffusion_timestep,
                    encoder_hidden_states=unconditional_language_embeds.to(self.dtype),
                    return_dict=False,
                )[0]
                guidance_scale = float(guidance)
                video_velocity_pred = video_velocity_uncond + guidance_scale * (
                    video_velocity_cond - video_velocity_uncond
                )
            else:
                video_velocity_pred = video_velocity_cond
            noisy_video = self.noise_scheduler_sample_video.step(
                video_velocity_pred, current_timestep, noisy_video, return_dict=False
            )[0]
            noisy_video = noisy_video.to(video_latents.dtype)

        if wan22:
            noisy_video[:, :, :1] = video_latents[:, :, :1]
        return {"pred_video": noisy_video, "pred_value": None}

    def compute_loss(self, lang_tokens, video_latents, condition_video_latents, value=None):
        del value
        wan22 = self._video_variant == "wan22"
        noise_for_video, noisy_hidden_states, diffusion_timesteps, per_element_loss_mask = (
            _video_train_diffusion_prep(
                self.noise_scheduler_video,
                video_latents,
                condition_video_latents,
                self.dtype,
                wan22=wan22,
            )
        )
        noisy_hidden_states = noisy_hidden_states.to(self.dtype)
        language_hidden_states = lang_tokens.to(self.dtype)
        pred_video = self.model(
            hidden_states=noisy_hidden_states,
            timestep=diffusion_timesteps,
            encoder_hidden_states=language_hidden_states,
            return_dict=False,
        )[0]
        target_video = (noise_for_video - video_latents).to(self.dtype)
        if per_element_loss_mask is None:
            loss_video = F.mse_loss(pred_video, target_video)
        else:
            loss_video = (
                F.mse_loss(pred_video, target_video, reduction="none") * per_element_loss_mask
            ).sum() / per_element_loss_mask.sum()
        return loss_video, None

    def predict_video(
        self,
        lang_tokens,
        video_latents,
        condition_video_latents,
        guidance: float = 5.0,
    ):
        return self.conditional_sample(
            lang_tokens,
            video_latents,
            condition_video_latents,
            guidance=guidance,
        )

    def forward(self, *args, **kwargs):
        return self.compute_loss(*args, **kwargs)
