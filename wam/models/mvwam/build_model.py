# coding=utf-8
"""MVWAM 模型装配：``build_action_expert`` / ``build_video_expert`` / ``build_world_stack``。"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Tuple

import torch

from wam.models.mvwam.world_stack_registry import resolve_video_base_model_path

from .action_expert import ActionExpertModel
from .video_expert import WanTransformer3DModel

logger = logging.getLogger(__name__)


def _common_get(common, key: str, default=None):
    if isinstance(common, dict):
        return common.get(key, default)
    v = getattr(common, key, None)
    return default if v is None else v


def resolve_vision_num_patches_from_common(common) -> int:
    """从 ``common`` 解析每图 patch 数（与 ``SiglipVisionTower.num_patches`` 一致）。

    优先 ``common.num_vision_patches``；否则用 ``common.vision_tower`` 或环境变量
    ``PRETRAINED_VISION_ENCODER_NAME_OR_PATH``（默认 Siglip2 路径）仅拉取 ``AutoConfig``（``delay_load``）。
    """
    np = _common_get(common, "num_vision_patches", None)
    if np is not None:
        return int(np)
    from wam.models.multimodal_encoder.siglip2_encoder import SiglipVisionTower

    vt = _common_get(common, "vision_tower", None) or os.environ.get(
        "PRETRAINED_VISION_ENCODER_NAME_OR_PATH",
        "/mnt/dataset/ckpt/pretrained_models/siglip2-so400m-patch14-384",
    )
    enc = SiglipVisionTower(str(vt), None, delay_load=True)
    return int(enc.num_patches)


def compute_img_cond_len_and_pos_embed_config(
    common, vision_num_patches: Optional[int] = None
) -> Tuple[int, list]:
    """由 ``model_structure['common']`` 与视觉 patch 数得到 ``img_cond_len``、``img_pos_embed_config``（与 ``train_world`` 一致）。"""
    if vision_num_patches is None:
        vision_num_patches = resolve_vision_num_patches_from_common(common)
    ihs = int(_common_get(common, "img_history_size"))
    nc = int(_common_get(common, "num_cameras"))
    vp = int(vision_num_patches)
    img_cond_len = ihs * nc * vp
    img_pos_embed_config = [("image", (ihs, nc, -vp))]
    return img_cond_len, img_pos_embed_config


def _is_empty_pretrained_path(value) -> bool:
    if value is None:
        return True
    s = str(value).strip()
    return s == "" or s.lower() in ("none", "null", "nil")


def _pick_pretrained_path(kwargs_value, config, attr_name: str):
    """kwargs 非空优先，否则读 ``config`` 上的同名属性（OmegaConf / dict）。"""
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


def _normalize_expert_path(value) -> str:
    if value is None:
        return "none"
    return value


def build_action_expert(
    action_expert_config: Any,
    *,
    img_cond_len: int,
    img_pos_embed_config,
    wan_transformer_for_init: Optional[WanTransformer3DModel] = None,
    load_weights_from_transformer: bool = False,
    pretrained_action_expert_path: str = "none",
) -> ActionExpertModel:
    """从 ``action_expert`` 配置构造 ``ActionExpertModel``，可选加载微调 checkpoint。"""
    pretrained_action_expert_path = _normalize_expert_path(pretrained_action_expert_path)
    action_expert = ActionExpertModel.from_custom_config(
        action_expert_config,
        wan_transformer_for_init,
        load_weights_from_transformer,
        img_cond_len=img_cond_len,
        img_pos_embed_config=img_pos_embed_config,
    )
    if pretrained_action_expert_path != "none" and os.path.exists(pretrained_action_expert_path):
        pt_path = os.path.join(pretrained_action_expert_path, "pytorch_model/mp_rank_00_model_states.pt")
        ckpt = torch.load(pt_path, map_location="cpu")
        state_dict = {}
        for key, value in ckpt["module"].items():
            state_dict[key.replace("model.", "")] = value
        action_expert.load_state_dict(state_dict)
    return action_expert


def build_video_expert(
    *,
    pretrained_video_expert_path: str = "none",
    dtype=torch.bfloat16,
    pretrained_video_expert_base_path: str | None = None,
) -> WanTransformer3DModel:
    """构造 ``WanTransformer3DModel``：底座来自 ``pretrained_video_expert_base_path``，可按路径叠加微调权重。"""
    pretrained_video_expert_path = _normalize_expert_path(pretrained_video_expert_path)
    base_path = pretrained_video_expert_base_path or resolve_video_base_model_path(None)

    if pretrained_video_expert_path != "none" and "WoW" in pretrained_video_expert_path:
        return WanTransformer3DModel.from_pretrained(
            pretrained_video_expert_path,
            subfolder="transformer",
            torch_dtype=dtype,
            local_files_only=True,
        )
    video_expert = WanTransformer3DModel.from_pretrained(
        base_path,
        subfolder="transformer",
        torch_dtype=dtype,
        local_files_only=True,
    )
    if pretrained_video_expert_path != "none" and os.path.exists(pretrained_video_expert_path):
        pretrained_video_expert_path_pt = os.path.join(
            pretrained_video_expert_path, "pytorch_model/mp_rank_00_model_states.pt"
        )
        if not os.path.exists(pretrained_video_expert_path_pt):
            safepath = os.path.join(pretrained_video_expert_path, "model.safetensors")
            from safetensors.torch import load_file

            ckpt = load_file(safepath)
            state_dict = {}
            for key, value in ckpt.items():
                if "model." in key:
                    state_dict[key.replace("model.", "")] = value
            video_expert.load_state_dict(state_dict)
        else:
            ckpt = torch.load(pretrained_video_expert_path_pt, map_location="cpu")
            state_dict = {}
            for key, value in ckpt["module"].items():
                if "model." in key:
                    state_dict[key.replace("model.", "")] = value
            video_expert.load_state_dict(state_dict)
    return video_expert


def _strip_to_world_policy_keys(state_dict: dict) -> dict:
    """将 FMPRunner / DeepSpeed 等前缀剥到 ``WorldPolicyModel`` 的 ``video_expert.*`` / ``action_expert.*``。"""
    out = {}
    for k, v in state_dict.items():
        nk = k
        while True:
            if nk.startswith("module.") or nk.startswith("model."):
                nk = nk.split(".", 1)[1]
                continue
            break
        if nk.startswith("video_expert.") or nk.startswith("action_expert."):
            out[nk] = v
    return out


def load_wam_checkpoint_into_world_policy(world, ckpt_root: str) -> None:
    """从训练保存目录或 HF 风格目录加载整网权重到 ``WorldPolicyModel``。"""
    root = os.path.expanduser(str(ckpt_root))
    pt_path = os.path.join(root, "pytorch_model", "mp_rank_00_model_states.pt")
    st_path = os.path.join(root, "model.safetensors")
    raw: dict
    if os.path.isfile(pt_path):
        ckpt = torch.load(pt_path, map_location="cpu")
        raw = ckpt["module"] if isinstance(ckpt, dict) and "module" in ckpt else ckpt
    elif os.path.isfile(st_path):
        from safetensors.torch import load_file

        raw = load_file(st_path)
    else:
        raise ValueError(
            f"pretrained_wam_path={root!r} 下未找到 pytorch_model/mp_rank_00_model_states.pt 或 model.safetensors"
        )
    if not isinstance(raw, dict):
        raise ValueError(f"checkpoint 格式异常: 期望 state_dict dict，得到 {type(raw)}")
    filtered = _strip_to_world_policy_keys(raw)
    inc = world.load_state_dict(filtered, strict=False)
    missing = getattr(inc, "missing_keys", None)
    if missing is None and isinstance(inc, tuple):
        missing = inc[0]
    if missing:
        logger.warning("WAM 整网加载 missing_keys (前 20 个): %s", list(missing)[:20])


def build_world_stack(
    *,
    config,
    img_cond_len: int,
    img_pos_embed_config,
    pretrained_video_expert_path: str = "none",
    pretrained_action_expert_path: str = "none",
    pretrained_wam_path=None,
    dtype=torch.bfloat16,
    pretrained_video_expert_base_path: str | None = None,
):
    """由 ``model_name=world_policy_causal_value`` 经 registry 调用；装配 Expert 与 ``WorldPolicyModel``。

    ``pretrained_wam_path`` 若有效则**仅**从此加载整网，不再应用分 expert 的微调加载。
    """
    from .world_policy_causal_value import WorldPolicyModel

    pretrained_video_expert_path = _normalize_expert_path(pretrained_video_expert_path)
    pretrained_action_expert_path = _normalize_expert_path(pretrained_action_expert_path)

    base_path = pretrained_video_expert_base_path or resolve_video_base_model_path(None)

    wam_path = _pick_pretrained_path(pretrained_wam_path, config, "pretrained_wam_path")
    if not _is_empty_pretrained_path(wam_path):
        root = os.path.expanduser(str(wam_path))
        if not os.path.isdir(root):
            raise ValueError(
                f"pretrained_wam_path={root!r} 必须是含 pytorch_model/mp_rank_00_model_states.pt 或 model.safetensors 的目录"
            )
        use_full_wam = True
    else:
        use_full_wam = False

    if use_full_wam:
        action_expert = build_action_expert(
            config.action_expert,
            img_cond_len=img_cond_len,
            img_pos_embed_config=img_pos_embed_config,
            wan_transformer_for_init=None,
            load_weights_from_transformer=False,
            pretrained_action_expert_path="none",
        )
        video_expert = build_video_expert(
            pretrained_video_expert_path="none",
            dtype=dtype,
            pretrained_video_expert_base_path=base_path,
        )
        world = WorldPolicyModel(video_expert, action_expert)
        load_wam_checkpoint_into_world_policy(world, str(wam_path))
        return world

    action_expert = build_action_expert(
        config.action_expert,
        img_cond_len=img_cond_len,
        img_pos_embed_config=img_pos_embed_config,
        wan_transformer_for_init=None,
        load_weights_from_transformer=False,
        pretrained_action_expert_path=pretrained_action_expert_path,
    )
    video_expert = build_video_expert(
        pretrained_video_expert_path=pretrained_video_expert_path,
        dtype=dtype,
        pretrained_video_expert_base_path=base_path,
    )
    return WorldPolicyModel(video_expert, action_expert)


__all__ = [
    "build_action_expert",
    "build_video_expert",
    "build_world_stack",
    "compute_img_cond_len_and_pos_embed_config",
    "load_wam_checkpoint_into_world_policy",
    "resolve_vision_num_patches_from_common",
]
