# coding=utf-8
"""训练 checkpoint 目录维护（与 ``checkpoint-{global_step}`` 布局兼容）。"""

from __future__ import annotations

import logging
import os
import re
import shutil
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")


def list_checkpoint_dirs(output_dir: str) -> List[Tuple[int, str]]:
    """列出 ``output_dir`` 下 ``checkpoint-<step>`` 目录，按 step 升序。"""
    if not output_dir or not os.path.isdir(output_dir):
        return []
    out: List[Tuple[int, str]] = []
    for name in os.listdir(output_dir):
        m = _CHECKPOINT_DIR_RE.match(name)
        if not m:
            continue
        path = os.path.join(output_dir, name)
        if os.path.isdir(path):
            out.append((int(m.group(1)), path))
    out.sort(key=lambda x: x[0])
    return out


def prune_old_checkpoints(
    output_dir: str,
    total_limit: Optional[int],
    *,
    log: Optional[logging.Logger] = None,
) -> int:
    """仅保留 step 最大的 ``total_limit`` 个 ``checkpoint-*`` 目录；返回删除个数。"""
    if total_limit is None or int(total_limit) <= 0:
        return 0
    limit = int(total_limit)
    checkpoints = list_checkpoint_dirs(output_dir)
    excess = len(checkpoints) - limit
    if excess <= 0:
        return 0
    log = log or logger
    removed = 0
    for step, path in checkpoints[:excess]:
        log.info(
            "Pruning checkpoint %s (step=%s); keeping latest %s under %s",
            path,
            step,
            limit,
            output_dir,
        )
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
    return removed
