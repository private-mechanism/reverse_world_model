# coding=utf-8
"""兼容旧 checkpoint（model_type=world_casual）使用的 ``build_world_stack`` 入口。

与 ``world_policy_causal_value`` 共用同一装配逻辑；value 相关分支由
``action_expert.use_value`` 配置控制，不在本模块层面区分。
"""

from .build_model import build_world_stack  # noqa: F401
