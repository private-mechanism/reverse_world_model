"""RoboTwin online success-rate deployment wrapper.

This module intentionally stays outside the training/sampling code.  RoboTwin
can import it from ``policy/<PolicyName>/deploy_policy.py`` and delegate model
loading plus per-step action generation to ``WVWAMRoboTwinPolicy``.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Mapping, Optional, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

from configs.model_config_loader import load_model_config_dict
from data.helpers import expand2square_pil
from data.helpers import (
    preprocess_video_chw_for_training,
    video_metas_nhwc_to_chw_float_tensor,
)
from data.hdf5_dataset_128dim import fallback_separate_statistics_path
from wam.models.multimodal_encoder.siglip2_encoder import SiglipVisionTower
from wam.models.multimodal_encoder.umt5_encoder import umT5Embedder
from wam.models.mvwam.build_model import compute_img_cond_len_and_pos_embed_config
from wam.runners.runner_registry import import_train_runner_class


DEFAULT_CAMERA_KEYS = ("head_camera", "right_camera", "left_camera")


@dataclass
class RoboTwinDeployConfig:
    model_config_path: str
    checkpoint_path: str
    statistics_path: Optional[str] = None
    device: str = "cuda"
    dtype: str = "bf16"
    action_only: bool = True
    reverse_world_order: Optional[bool] = None
    action_dim: int = 14
    action_type: str = "qpos"
    control_freq: int = 15
    replan_interval: int = 8
    eval_steps_per_call: int = 1
    num_inference_timesteps: Optional[int] = 5
    camera_keys: Sequence[str] = DEFAULT_CAMERA_KEYS
    video_camera_key: str = "head_camera"
    video_num_frames: Optional[int] = None
    instruction: Optional[str] = None

    @classmethod
    def from_args(cls, args: Any) -> "RoboTwinDeployConfig":
        """Build from a RoboTwin ``usr_args`` namespace or a plain mapping."""
        getter = args.get if isinstance(args, Mapping) else lambda key, default=None: getattr(args, key, default)
        camera_keys = getter("camera_keys", DEFAULT_CAMERA_KEYS)
        if isinstance(camera_keys, str):
            camera_keys = tuple(x.strip() for x in camera_keys.split(",") if x.strip())
        return cls(
            model_config_path=str(getter("model_config_path")),
            checkpoint_path=str(getter("checkpoint_path")),
            statistics_path=getter("statistics_path", None),
            device=str(getter("device", "cuda")),
            dtype=str(getter("dtype", "bf16")),
            action_only=_as_bool(getter("action_only", True), default=True),
            reverse_world_order=(
                None
                if getter("reverse_world_order", None) is None
                else _as_bool(getter("reverse_world_order", None))
            ),
            action_dim=int(getter("action_dim", 14)),
            action_type=str(getter("action_type", "qpos")),
            control_freq=int(getter("control_freq", 15)),
            replan_interval=int(getter("replan_interval", 8)),
            eval_steps_per_call=int(getter("eval_steps_per_call", 1)),
            num_inference_timesteps=_optional_int(getter("num_inference_timesteps", 5)),
            camera_keys=tuple(camera_keys),
            video_camera_key=str(getter("video_camera_key", "head_camera")),
            video_num_frames=_optional_int(getter("video_num_frames", None)),
            instruction=getter("instruction", None),
        )


def _torch_dtype(name: str) -> torch.dtype:
    lowered = str(name).lower()
    if lowered in ("bf16", "bfloat16"):
        return torch.bfloat16
    if lowered in ("fp16", "float16", "half"):
        return torch.float16
    if lowered in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = str(value).strip().lower()
    if lowered in ("1", "true", "yes", "y", "on"):
        return True
    if lowered in ("0", "false", "no", "n", "off", "none", ""):
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("", "none", "null"):
        return None
    return int(value)


def _resolve_checkpoint_path(path: str) -> str:
    checkpoint = Path(path).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = Path.cwd() / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint_path does not exist: {checkpoint}")
    return str(checkpoint)


def _load_statistics(path: Optional[str], model_config: dict, state_dim: int) -> Optional[dict]:
    candidates = []
    if path:
        candidates.append(Path(path).expanduser())
    data_paths = model_config.get("data_paths") or []
    if isinstance(data_paths, str):
        data_paths = [data_paths]
    for data_path in data_paths:
        candidates.append(Path(data_path) / f"separate_statistics-state_dim{state_dim}.json")
        candidates.append(Path(fallback_separate_statistics_path(str(data_path), state_dim)))

    for candidate in candidates:
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as handle:
                return json.load(handle)
    return None


def _normalize_minmax(value: np.ndarray, statics: Optional[dict]) -> np.ndarray:
    if statics is None:
        return value
    q01 = np.asarray(statics["q01"], dtype=np.float32)
    q99 = np.asarray(statics["q99"], dtype=np.float32)
    denom = q99 - q01
    safe_denom = np.where(denom == 0, 1.0, denom)
    out = (value - q01) / safe_denom * 2.0 - 1.0
    out = np.clip(out, -1.0, 1.0)
    return np.where(denom == 0, 0.0, out).astype(np.float32)


def _denormalize_minmax(value: np.ndarray, statics: Optional[dict]) -> np.ndarray:
    if statics is None:
        return value
    q01 = np.asarray(statics["q01"], dtype=np.float32)
    q99 = np.asarray(statics["q99"], dtype=np.float32)
    denom = q99 - q01
    out = (value + 1.0) * 0.5 * denom + q01
    return np.where(denom == 0, 0.0, out).astype(np.float32)


def _extract_observation_root(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    return observation.get("observation", observation)


def _extract_rgb(observation: Mapping[str, Any], camera_key: str) -> np.ndarray:
    root = _extract_observation_root(observation)
    camera = root.get(camera_key)
    if camera is None:
        raise KeyError(f"RoboTwin observation is missing camera key: {camera_key}")
    image = camera.get("rgb", camera) if isinstance(camera, Mapping) else camera
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image for {camera_key}, got shape {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _extract_joint_vector(observation: Mapping[str, Any]) -> np.ndarray:
    candidates = [
        ("joint_action", "vector"),
        ("agent", "qpos"),
        ("robot", "qpos"),
        ("qpos",),
        ("state",),
    ]
    root = _extract_observation_root(observation)
    for path in candidates:
        value: Any = observation if path[0] == "joint_action" else root
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                value = None
                break
            value = value[key]
        if value is not None:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size > 0:
                return arr
    raise KeyError("Could not find qpos/state vector in RoboTwin observation.")


class WVWAMRoboTwinPolicy:
    """Stateful RoboTwin policy adapter for WVWAM checkpoints."""

    def __init__(self, config: RoboTwinDeployConfig):
        self.deploy_config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.weight_dtype = _torch_dtype(config.dtype)
        self.model_config = load_model_config_dict(config.model_config_path)
        self.model_structure = self.model_config["model_structure"]
        self.common = self.model_structure["common"]
        self.dataset_cfg = self.model_structure["dataset"]
        self.state_dim = int(self.common["state_dim"])
        self.action_chunk_size = int(self.common["action_chunk_size"])
        self.reverse_world_order = (
            bool(self.model_config.get("reverse_world_order", False))
            if config.reverse_world_order is None
            else bool(config.reverse_world_order)
        )
        self.norm_minmax = bool(self.common.get("norm_minmax", False))
        self.image_size = self.dataset_cfg.get("video_size") or self.dataset_cfg.get("image_size")
        self.image_aspect_ratio = self.dataset_cfg.get("image_aspect_ratio", "pad")
        self.pred_video_num_cameras = int(self.dataset_cfg.get("num_cameras", 1))
        self.action_queue: Deque[np.ndarray] = deque()
        self.steps_since_replan = 0

        checkpoint_path = _resolve_checkpoint_path(config.checkpoint_path)
        self.model_config["pretrained_wam_path"] = checkpoint_path
        self.model_structure["model"]["pretrained_wam_path"] = checkpoint_path
        omega_model_config = OmegaConf.create({"model": self.model_structure["model"]})

        self.text_embedder = umT5Embedder(
            device=self.device,
            from_pretrained=self.model_config["pretrained_text_encoder_name_or_path"],
            model_max_length=int(self.dataset_cfg["tokenizer_max_length"]),
            torch_dtype=self.weight_dtype,
        )
        self.vision_encoder = SiglipVisionTower(
            vision_tower=self.model_config["pretrained_vision_encoder_name_or_path"],
            args=None,
        )
        self.vision_encoder.vision_tower.to(self.device, dtype=self.weight_dtype).eval()

        runner_cls = import_train_runner_class(
            self.model_config.get("train_runner_module"),
            self.model_config.get("train_runner_class"),
        )
        img_cond_len, img_pos_embed_config = compute_img_cond_len_and_pos_embed_config(
            self.common, self.vision_encoder.num_patches
        )
        self.model = runner_cls(
            pred_horizon=self.action_chunk_size,
            config=omega_model_config.model,
            img_cond_len=img_cond_len,
            img_pos_embed_config=img_pos_embed_config,
            pretrained_wam_path=checkpoint_path,
            pretrained_video_expert_path=self.model_config.get("pretrained_video_expert_path"),
            pretrained_action_expert_path=self.model_config.get("pretrained_action_expert_path"),
            model_name=self.model_config.get("model_name"),
            model_type=self.model_config.get("model_type"),
            video_base_model=self.model_config.get("VIDEO_BASE_MODEL"),
            video_variant=self.model_config.get("video_variant"),
        ).to(self.device, dtype=self.weight_dtype)
        self.model.eval()

        self.statistics = _load_statistics(config.statistics_path, self.model_config, self.state_dim)
        if self.norm_minmax and self.statistics is None:
            raise FileNotFoundError(
                "norm_minmax=true but no statistics file was found. Pass statistics_path "
                "or keep separate_statistics-state_dim128.json beside the RoboTwin HDF5 data."
            )
        self.state_stats = self.statistics.get("state") if self.statistics else None
        self.action_stats = self.statistics.get("action") if self.statistics else None
        self.vae = None
        if not self.deploy_config.action_only:
            from wam.models.multimodal_encoder.vae_encoder import VAEEncoder

            self.vae = VAEEncoder(self.model_config["VIDEO_BASE_MODEL"])
            self.vae.model = self.vae.model.to(
                self.device, dtype=self.weight_dtype
            )

    def reset(self) -> None:
        self.action_queue.clear()
        self.steps_since_replan = 0

    def _preprocess_one_image(self, image_array: np.ndarray) -> torch.Tensor:
        image = Image.fromarray(image_array)
        if self.image_size is not None:
            image = transforms.Resize(tuple(self.image_size))(image)
        if self.image_aspect_ratio == "pad":
            processor = self.vision_encoder.image_processor
            image = expand2square_pil(
                image, tuple(int(channel * 255) for channel in processor.image_mean)
            )
        return self.vision_encoder.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

    def _build_image_tokens(self, observation: Mapping[str, Any]) -> torch.Tensor:
        images = [
            self._preprocess_one_image(_extract_rgb(observation, camera_key))
            for camera_key in self.deploy_config.camera_keys
        ]
        image_tensor = torch.stack(images, dim=0).to(self.device, dtype=self.weight_dtype)
        with torch.no_grad():
            embeds = self.vision_encoder(image_tensor).detach()
        return embeds.reshape(1, -1, self.vision_encoder.hidden_size)

    def _build_state_tokens(self, observation: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        qpos = _extract_joint_vector(observation)
        valid_dim = min(qpos.size, self.state_dim)
        state = np.zeros(self.state_dim, dtype=np.float32)
        mask = np.zeros(self.state_dim, dtype=np.float32)
        state[:valid_dim] = qpos[:valid_dim]
        mask[:valid_dim] = 1.0
        if self.norm_minmax:
            state = _normalize_minmax(state, self.state_stats)
        state_tokens = torch.from_numpy(state).to(self.device, dtype=self.weight_dtype).view(1, 1, -1)
        action_mask = torch.from_numpy(mask).to(self.device, dtype=self.weight_dtype).view(1, 1, -1)
        return state_tokens, action_mask

    def _build_video_condition(
        self, observation: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the current camera frame and build a reverse-view latent template."""
        if self.vae is None:
            raise RuntimeError("Video conditioning requires action_only=false.")
        if self.pred_video_num_cameras == 3:
            camera_keys = tuple(self.deploy_config.camera_keys)
            if len(camera_keys) < 3:
                raise ValueError(
                    "Three-camera video conditioning requires three camera_keys."
                )
            # Training stitches [high, left wrist, right wrist].
            video_metas = [
                _extract_rgb(observation, camera_keys[0])[None],
                _extract_rgb(observation, camera_keys[2])[None],
                _extract_rgb(observation, camera_keys[1])[None],
            ]
        elif self.pred_video_num_cameras == 1:
            video_metas = [
                _extract_rgb(
                    observation, self.deploy_config.video_camera_key
                )[None]
            ]
        else:
            raise ValueError(
                f"Unsupported video camera count: {self.pred_video_num_cameras}"
            )
        video_frames = video_metas_nhwc_to_chw_float_tensor(video_metas)
        processed = preprocess_video_chw_for_training(
            video_frames,
            self.vision_encoder.image_processor,
            tuple(self.dataset_cfg["video_size"]),
            image_size=self.dataset_cfg.get("image_size"),
            image_aug=False,
            auto_adjust_image_brightness=False,
            sample_image_aug_type_fn=lambda: "mixed",
        )
        current_frame = (
            processed.permute(1, 0, 2, 3)
            .unsqueeze(0)
            .to(self.device, dtype=self.weight_dtype)
        )
        current_latent = self.vae.encode_to_latents(current_frame)
        num_video_frames = (
            int(self.deploy_config.video_num_frames)
            if self.deploy_config.video_num_frames is not None
            else self.action_chunk_size + 1
        )
        latent_frames = 1 + max(0, num_video_frames - 1) // 4
        batch_size, channels, _one, height, width = current_latent.shape
        video_latents = current_latent.new_zeros(
            batch_size, channels, latent_frames, height, width
        )
        # Goal-conditioned reverse inference reads the current frame from the
        # final temporal slot and predicts the horizon key at slot zero.
        video_latents[:, :, -1:] = current_latent[:, :, :1]
        mask_channels = int(self.vae.model.config.scale_factor_temporal)
        condition_video_latents = current_latent.new_zeros(
            batch_size,
            mask_channels + channels,
            latent_frames,
            height,
            width,
        )
        return video_latents, condition_video_latents

    def _instruction_from_env(
        self,
        task_env: Optional[Any],
        observation: Mapping[str, Any],
        instruction: Optional[str],
    ) -> str:
        if instruction is not None:
            return instruction
        if self.deploy_config.instruction is not None:
            return self.deploy_config.instruction
        for key in ("instruction", "task_instruction", "prompt"):
            if key in observation:
                return str(observation[key])
        for attr in ("get_instruction", "get_task_instruction"):
            if task_env is not None and hasattr(task_env, attr):
                value = getattr(task_env, attr)()
                if value:
                    return str(value)
        return ""

    @torch.no_grad()
    def plan_action_chunk(
        self,
        observation: Mapping[str, Any],
        *,
        task_env: Optional[Any] = None,
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        text = self._instruction_from_env(task_env, observation, instruction)
        lang_tokens, lang_attn_mask = self.text_embedder.get_text_embeddings([text])
        lang_tokens = lang_tokens.to(self.device, dtype=self.weight_dtype)
        lang_attn_mask = lang_attn_mask.to(self.device)
        img_tokens = self._build_image_tokens(observation)
        state_tokens, action_mask = self._build_state_tokens(observation)
        ctrl_freqs = torch.tensor([self.deploy_config.control_freq], device=self.device, dtype=torch.long)
        if self.deploy_config.action_only:
            video_latents = None
            condition_video_latents = None
        else:
            video_latents, condition_video_latents = self._build_video_condition(
                observation
            )

        out = self.model.predict_action(
            lang_tokens=lang_tokens,
            lang_attn_mask=lang_attn_mask,
            img_tokens=img_tokens,
            state_tokens=state_tokens,
            action_mask=action_mask,
            ctrl_freqs=ctrl_freqs,
            video_latents=video_latents,
            condition_video_latents=condition_video_latents,
            action_only=self.deploy_config.action_only,
            video_only=False,
            sample_batch_size=1,
            num_inference_timesteps=self.deploy_config.num_inference_timesteps,
        )
        pred = out["pred_trajectory"].detach().float().cpu().numpy()[0]
        if self.reverse_world_order and out.get("trajectory_order") != "forward":
            pred = pred[::-1].copy()
        if self.norm_minmax:
            pred = _denormalize_minmax(pred, self.action_stats)
        return pred[:, : self.deploy_config.action_dim]

    def act(
        self,
        observation: Mapping[str, Any],
        *,
        task_env: Optional[Any] = None,
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        should_replan = (
            not self.action_queue
            or self.steps_since_replan >= max(1, int(self.deploy_config.replan_interval))
        )
        if should_replan:
            chunk = self.plan_action_chunk(observation, task_env=task_env, instruction=instruction)
            self.action_queue.clear()
            self.action_queue.extend(chunk)
            self.steps_since_replan = 0
        self.steps_since_replan += 1
        return self.action_queue.popleft()


def take_action_with_robotwin(
    task_env: Any,
    policy: WVWAMRoboTwinPolicy,
    observation: Mapping[str, Any],
    *,
    action_type: Optional[str] = None,
    steps: Optional[int] = None,
) -> list[np.ndarray]:
    """RoboTwin ``deploy_policy.eval`` helper."""
    executed = []
    n_steps = int(steps if steps is not None else policy.deploy_config.eval_steps_per_call)
    env_action_type = action_type or policy.deploy_config.action_type
    for _ in range(max(1, n_steps)):
        action = policy.act(observation, task_env=task_env)
        task_env.take_action(action, action_type=env_action_type)
        executed.append(action)
        if hasattr(task_env, "get_obs"):
            observation = task_env.get_obs()
        if bool(getattr(task_env, "suc", False)):
            break
    return executed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WVWAM RoboTwin deployment helper")
    parser.add_argument("--model_config_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--statistics_path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--action_dim", type=int, default=14)
    parser.add_argument("--action_type", default="qpos")
    parser.add_argument("--control_freq", type=int, default=15)
    parser.add_argument("--replan_interval", type=int, default=8)
    parser.add_argument("--eval_steps_per_call", type=int, default=1)
    parser.add_argument("--num_inference_timesteps", type=int, default=5)
    parser.add_argument("--camera_keys", default="head_camera,right_camera,left_camera")
    parser.add_argument("--video_camera_key", default="head_camera")
    parser.add_argument("--video_num_frames", type=int, default=None)
    parser.add_argument("--instruction", default=None)
    return parser


def load_policy_from_args(args: Any) -> WVWAMRoboTwinPolicy:
    return WVWAMRoboTwinPolicy(RoboTwinDeployConfig.from_args(args))


__all__ = [
    "RoboTwinDeployConfig",
    "WVWAMRoboTwinPolicy",
    "load_policy_from_args",
    "take_action_with_robotwin",
]
