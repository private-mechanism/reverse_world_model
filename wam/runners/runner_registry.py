# coding=utf-8
"""按配置动态加载训练用 Runner 类，避免 ``train_world`` 硬编码 ``wam.runners.<某文件>``。"""

from __future__ import annotations

import importlib
import re
from typing import Any, Optional, Type

_MODULE_SAFE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_CLASS_SAFE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# 历史 yml 曾写 ``rdt_runner_*``；``normalize_train_runner_module_name`` 会剥去 ``rdt_`` 前缀。
LEGACY_RUNNER_MODULE_PREFIX = "rdt_"

DEFAULT_TRAIN_RUNNER_MODULE = "runner_world_casual_128dim"
DEFAULT_TRAIN_RUNNER_CLASS = "FMPRunner"


def normalize_train_runner_module_name(name: str) -> str:
    """将 ``rdt_runner_*`` 规范为 ``runner_*``，其它字符串原样返回。"""
    n = (name or "").strip()
    if n.startswith(LEGACY_RUNNER_MODULE_PREFIX):
        return n[len(LEGACY_RUNNER_MODULE_PREFIX) :]
    return n


def import_train_runner_class(
    runner_module: Optional[str] = None,
    runner_class: Optional[str] = None,
) -> Type[Any]:
    """``import wam.runners.<runner_module>`` 并返回 ``runner_class`` 指向的类型。

    扁平 ``model_config`` 可选键：``train_runner_module``、``train_runner_class``（缺省为当前因果 128dim Runner）。
    仍接受旧模块名 ``rdt_runner_*``（自动去掉 ``rdt_`` 前缀）。
    """
    raw_mod = (runner_module or DEFAULT_TRAIN_RUNNER_MODULE).strip()
    mod_name = normalize_train_runner_module_name(raw_mod)
    cls_name = (runner_class or DEFAULT_TRAIN_RUNNER_CLASS).strip()
    if not _MODULE_SAFE.match(mod_name) or not _CLASS_SAFE.match(cls_name):
        raise ValueError(f"非法 train_runner 标识: module={mod_name!r}, class={cls_name!r}")
    path = f"wam.runners.{mod_name}"
    try:
        mod = importlib.import_module(path)
    except ModuleNotFoundError as e:
        raise ValueError(
            f"找不到 train_runner 模块 {path!r}。请在 wam/runners/ 下添加 {mod_name}.py。"
        ) from e
    cls = getattr(mod, cls_name, None)
    if cls is None:
        raise ValueError(f"模块 {path} 中不存在类 {cls_name!r}。")
    return cls
