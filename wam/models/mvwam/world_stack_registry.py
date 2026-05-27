# coding=utf-8
"""按 ``model_name`` 动态加载 ``wam.models.mvwam.<model_name>.build_world_stack``。

Runner（``FMPRunner``）不再 import 具体 Expert 类，只调用本模块。
"""

from __future__ import annotations

import importlib
import os
import re
from typing import Any, Dict, Optional

# ``model_config`` 中 ``VIDEO_BASE_MODEL`` / ``weights.video_base_model`` 的合法取值（键）→ Diffusers 根目录
VIDEO_BASE_MODEL_PATHS: Dict[str, str] = {
    "WoW-1-Wan-1.3B-2M-Diffusers": "/mnt/dataset/datasets/cjt_personal/pretrained_models/WoW-1-Wan-1.3B-2M-Diffusers",
    "Wan2.2-TI2V-5B-Diffusers": "/mnt/dataset/projs/projects/checkpoints/Wan2.2-TI2V-5B-Diffusers",
}

DEFAULT_VIDEO_BASE_MODEL = "WoW-1-Wan-1.3B-2M-Diffusers"

# 与历史代码兼容：默认 Wan 底座（等价于 ``resolve_video_base_model_path(None)``）
PRETRAINED_VIDEO_EXPERT_PATH = VIDEO_BASE_MODEL_PATHS[DEFAULT_VIDEO_BASE_MODEL]


def resolve_video_base_model_path(video_base_model: Optional[str] = None) -> str:
    """将 ``VIDEO_BASE_MODEL`` 解析为含 ``transformer/``、``scheduler/`` 的 Diffusers 根目录。

    - ``None`` / 空 / ``none``：使用 ``DEFAULT_VIDEO_BASE_MODEL``。
    - 与 ``VIDEO_BASE_MODEL_PATHS`` 的键一致：返回对应绝对路径。
    - 以 ``/`` 或 ``~`` 开头的路径：``expanduser`` 后作为自定义目录（须自行保证目录结构正确）。
    """
    if video_base_model is None:
        return VIDEO_BASE_MODEL_PATHS[DEFAULT_VIDEO_BASE_MODEL]
    v = str(video_base_model).strip()
    if v == "" or v.lower() in ("none", "null", "nil"):
        return VIDEO_BASE_MODEL_PATHS[DEFAULT_VIDEO_BASE_MODEL]
    if v in VIDEO_BASE_MODEL_PATHS:
        return VIDEO_BASE_MODEL_PATHS[v]
    exp = os.path.expanduser(v)
    if os.path.isabs(exp):
        return exp
    raise ValueError(
        f"未知的 VIDEO_BASE_MODEL={v!r}。请使用下列键之一，或传入绝对路径: {sorted(VIDEO_BASE_MODEL_PATHS.keys())}"
    )


def pick_video_base_model_str(video_base_model: Optional[str], config: Any) -> Optional[str]:
    """kwargs ``video_base_model`` 非空优先，否则读 ``config`` 上的 ``VIDEO_BASE_MODEL``（OmegaConf / dict）。"""
    if video_base_model is not None:
        s = str(video_base_model).strip()
        if s and s.lower() not in ("none", "null", "nil"):
            return s
    if config is None:
        return None
    cfg_v = config.get("VIDEO_BASE_MODEL", None) if hasattr(config, "get") else getattr(config, "VIDEO_BASE_MODEL", None)
    if cfg_v is None:
        return None
    s = str(cfg_v).strip()
    if not s or s.lower() in ("none", "null", "nil"):
        return None
    return s


# 旧 yml 字段 ``model_type``（world_casual*）→ 新 ``model_name``（须与 ``wam/models/mvwam/`` 下模块名一致）
LEGACY_MODEL_TYPE_TO_MODEL_NAME: Dict[str, str] = {
    "world_casual": "world_policy_causal",
    "world_casual_value": "world_policy_causal_value",
    "world_casual_VA": "world_policy_causal_VA",
}

_MODEL_NAME_SAFE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def resolve_model_name(
    model_name: Optional[str],
    model_type: Optional[str],
) -> str:
    """优先 ``model_name``；否则将旧 ``model_type`` 映射为 ``model_name``。"""
    if model_name is not None and str(model_name).strip() != "":
        return str(model_name)
    if model_type is not None and str(model_type) in LEGACY_MODEL_TYPE_TO_MODEL_NAME:
        return LEGACY_MODEL_TYPE_TO_MODEL_NAME[str(model_type)]
    if model_type is not None and str(model_type).strip() != "":
        return str(model_type)
    return "world_policy_causal_value"


def build_world_stack(model_name: str, **kwargs: Any):
    """``import wam.models.mvwam.<model_name>`` 并调用其 ``build_world_stack``。"""
    mn = str(model_name)
    if not _MODEL_NAME_SAFE.match(mn):
        raise ValueError(f"model_name={mn!r} 非法（仅允许 Python 标识符字符）。")
    mod_path = f"wam.models.mvwam.{mn}"
    try:
        mod = importlib.import_module(mod_path)
    except ModuleNotFoundError as e:
        raise ValueError(
            f"找不到 model_name={mn!r} 对应的模块 {mod_path}。"
            f"请在 wam/models/mvwam/ 下添加 {mn}.py 并实现 build_world_stack(**kwargs)。"
        ) from e
    _build = getattr(mod, "build_world_stack", None)
    if _build is None or not callable(_build):
        raise ValueError(f"模块 {mod_path} 未定义可调用 build_world_stack。")
    return _build(**kwargs)
