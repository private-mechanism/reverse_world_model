# coding=utf-8
"""训练用学习率调度：自定义 warmup+cosine / constant-then-decay，以及 diffusers 内置调度器。

多卡时 scheduler 步数需乘 ``num_processes``，与 Accelerate 在每次优化步上的对齐方式一致
（见 HuggingFace DreamBooth / diffusers PR #8312）。
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from diffusers.optimization import get_scheduler
from torch.optim.lr_scheduler import LambdaLR

if TYPE_CHECKING:
    from accelerate import Accelerator


def build_custom_cosine_scheduler(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
):
    """Warmup 后 **单次** 半余弦从峰值降到 0（等价于 diffusers ``get_cosine_schedule_with_warmup(..., num_cycles=0.5)``）。

    与 ``cosine`` / ``lr_num_cycles`` 解耦，避免部分 diffusers 版本对 ``num_cycles`` 传递不一致导致的多周期现象。
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        denom = max(1, num_training_steps - num_warmup_steps)
        progress = float(current_step - num_warmup_steps) / float(denom)
        progress = min(1.0, max(0.0, progress))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def build_custom_constant_then_decay_scheduler(
    optimizer,
    num_training_steps: int,
    hold_ratio: float = 0.2,
    last_epoch: int = -1,
):
    """前 ``hold_ratio`` 比例的 scheduler 步保持 lr 倍数 1.0，之后线性降到 0。

    不使用 ``lr_warmup_steps``；总步数与 ``max_train_steps * num_processes`` 一致。
    """
    hold_ratio = float(min(1.0, max(0.0, hold_ratio)))
    hold_steps = int(round(num_training_steps * hold_ratio))
    hold_steps = min(max(hold_steps, 0), num_training_steps)

    def lr_lambda(current_step: int):
        if current_step < hold_steps:
            return 1.0
        if current_step >= num_training_steps:
            return 0.0
        return max(
            0.0,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - hold_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def build_training_lr_scheduler(
    optimizer,
    args,
    accelerator: Accelerator,
    logger: logging.Logger,
):
    """按 ``args.lr_scheduler`` 构建调度器（未 ``accelerator.prepare``）。

    ``num_scheduler_steps`` / warmup 步数 = ``max_train_steps`` / ``lr_warmup_steps`` × ``num_processes``。
    """
    num_processes = accelerator.num_processes
    num_scheduler_steps = args.max_train_steps * num_processes
    num_warmup_scheduler_steps = args.lr_warmup_steps * num_processes
    lr_hold_ratio = float(getattr(args, "lr_hold_ratio", 0.2))

    if args.lr_scheduler == "custom_cosine":
        lr_scheduler = build_custom_cosine_scheduler(
            optimizer,
            num_warmup_steps=num_warmup_scheduler_steps,
            num_training_steps=num_scheduler_steps,
        )
        logger.info(
            "LR scheduler=custom_cosine: warmup_sched_steps=%s, total_sched_steps=%s (×num_processes=%s)",
            num_warmup_scheduler_steps,
            num_scheduler_steps,
            num_processes,
        )
        return lr_scheduler

    if args.lr_scheduler == "custom_constant_then_decay":
        lr_scheduler = build_custom_constant_then_decay_scheduler(
            optimizer,
            num_training_steps=num_scheduler_steps,
            hold_ratio=lr_hold_ratio,
        )
        logger.info(
            "LR scheduler=custom_constant_then_decay: hold_ratio=%s, hold_sched_steps≈%s, total_sched_steps=%s (×num_processes=%s)",
            lr_hold_ratio,
            int(round(num_scheduler_steps * lr_hold_ratio)),
            num_scheduler_steps,
            num_processes,
        )
        return lr_scheduler

    return get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_scheduler_steps,
        num_training_steps=num_scheduler_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
