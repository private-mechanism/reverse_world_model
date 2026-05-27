#!/usr/bin/env python3
"""Generate and run the two WorldPolicy frame-causal ablation experiments.

The script derives two configs from ``model_config/wam_wow1.3b_demo.yml``:

1. all three causal switches enabled
2. all three causal switches disabled

Both configs enable the lightweight sample metrics added for action/value/latent/video
evaluation and use the same WandB project.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = REPO_ROOT / "model_config" / "wam_wow1.3b_demo.yml"
DEFAULT_GENERATED_DIR = REPO_ROOT / "model_config" / "generated" / "frame_causal_worldpolicy"
WANDB_PROJECT = "frame-causal-WorldPolicy"


EXPERIMENTS = (
    ("causal_all_true", True),
    ("causal_all_false", False),
)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _find_block(lines: List[str], path: Sequence[str]) -> tuple[int, int]:
    search_start = 0
    search_end = len(lines)
    for depth, key in enumerate(path):
        indent = depth * 2
        block_start = -1
        prefix = " " * indent + f"{key}:"
        for idx in range(search_start, search_end):
            if lines[idx].startswith(prefix) and _indent(lines[idx]) == indent:
                block_start = idx
                break
        if block_start < 0:
            raise KeyError(f"Cannot find YAML block: {'.'.join(path[: depth + 1])}")

        block_end = len(lines)
        for idx in range(block_start + 1, search_end):
            stripped = lines[idx].strip()
            if stripped and _indent(lines[idx]) <= indent:
                block_end = idx
                break
        search_start = block_start + 1
        search_end = block_end
    return block_start, search_end


def _yaml_scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _set_scalar(lines: List[str], path: Sequence[str], key: str, value) -> None:
    _, block_end = _find_block(lines, path)
    child_indent = len(path) * 2
    prefix = " " * child_indent + f"{key}:"
    new_line = " " * child_indent + f"{key}: {_yaml_scalar(value)}\n"

    search_start = _find_block(lines, path)[0] + 1
    for idx in range(search_start, block_end):
        if lines[idx].startswith(prefix) and _indent(lines[idx]) == child_indent:
            lines[idx] = new_line
            return
    lines.insert(block_end, new_line)


def _get_scalar(lines: List[str], path: Sequence[str], key: str, default: str) -> str:
    _, block_end = _find_block(lines, path)
    child_indent = len(path) * 2
    prefix = " " * child_indent + f"{key}:"
    search_start = _find_block(lines, path)[0] + 1
    for idx in range(search_start, block_end):
        if lines[idx].startswith(prefix) and _indent(lines[idx]) == child_indent:
            value = lines[idx].split(":", 1)[1].strip()
            return value.split("#", 1)[0].strip() or default
    return default


def _build_experiment_config_text(base_text: str, *, exp_name: str, enabled: bool) -> str:
    lines = base_text.splitlines(keepends=True)

    base_id = _get_scalar(lines, ("experiment",), "id", "wam-wow1.3b")
    base_run_name = _get_scalar(lines, ("checkpoints",), "run_name", base_id)

    _set_scalar(lines, ("experiment",), "id", f"{base_id}-{exp_name}")
    _set_scalar(lines, ("experiment",), "config_name", exp_name)
    _set_scalar(lines, ("experiment",), "wandb_project", WANDB_PROJECT)
    _set_scalar(lines, ("checkpoints",), "run_name", f"{base_run_name}-{exp_name}")

    _set_scalar(lines, ("architecture", "model"), "video_frame_causal_self_attn", enabled)
    _set_scalar(lines, ("architecture", "model"), "action_video_frame_causal_kv", enabled)
    _set_scalar(lines, ("architecture", "model", "action_expert"), "frame_causal_self_attn", enabled)

    _set_scalar(lines, ("training",), "sample_compute_video_metrics", True)
    _set_scalar(lines, ("training",), "sample_joint_only", True)
    _set_scalar(lines, ("training",), "sample_light", False)
    _set_scalar(lines, ("training",), "sample_save_video", False)
    _set_scalar(lines, ("logging",), "report_to", "wandb")

    return "".join(lines)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _launch_args(args: argparse.Namespace) -> List[str]:
    launch_args: List[str] = []
    num_processes = args.num_processes or os.environ.get("TRAIN_WAM_NUM_PROCESSES")
    if num_processes:
        launch_args.extend(["--num_processes", str(num_processes)])
        try:
            if int(num_processes) >= 2:
                launch_args.append("--multi_gpu")
        except ValueError:
            pass

    extra = args.launch_extra or os.environ.get("TRAIN_WAM_LAUNCH_EXTRA", "")
    if extra:
        launch_args.extend(shlex.split(extra))
    return launch_args


def _run_command(cmd: List[str], *, dry_run: bool) -> int:
    print("+ " + shlex.join(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--generated-dir", type=Path, default=DEFAULT_GENERATED_DIR)
    parser.add_argument(
        "--only",
        choices=("causal_all_true", "causal_all_false"),
        help="Run only one experiment from the two-experiment matrix.",
    )
    parser.add_argument(
        "--num-processes",
        help="Passed to accelerate launch as --num_processes. Defaults to TRAIN_WAM_NUM_PROCESSES.",
    )
    parser.add_argument(
        "--launch-extra",
        help="Extra accelerate launch args. Defaults to TRAIN_WAM_LAUNCH_EXTRA.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print generated commands without running training.")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue to the next experiment if one experiment fails.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_config = args.base_config.expanduser().resolve()
    generated_dir = args.generated_dir.expanduser().resolve()

    if not base_config.is_file():
        raise FileNotFoundError(f"Base config not found: {base_config}")

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "8")
    Path("/tmp/deepspeed_nvme").mkdir(parents=True, exist_ok=True)

    base_text = base_config.read_text(encoding="utf-8")
    selected = [item for item in EXPERIMENTS if args.only in (None, item[0])]
    launch_args = _launch_args(args)

    exit_code = 0
    for exp_name, enabled in selected:
        cfg_text = _build_experiment_config_text(base_text, exp_name=exp_name, enabled=enabled)
        cfg_path = generated_dir / f"wam_wow1.3b_demo__{exp_name}.yml"
        _write_text(cfg_path, cfg_text)

        print(
            f"\n=== {exp_name}: "
            f"video_frame_causal_self_attn={enabled}, "
            f"action_expert.frame_causal_self_attn={enabled}, "
            f"action_video_frame_causal_kv={enabled} ===",
            flush=True,
        )
        cmd = [
            "accelerate",
            "launch",
            *launch_args,
            "main_world.py",
            f"--model_config_path={cfg_path}",
        ]
        code = _run_command(cmd, dry_run=args.dry_run)
        if code != 0:
            exit_code = code
            print(f"Experiment {exp_name} failed with exit code {code}.", file=sys.stderr, flush=True)
            if not args.keep_going:
                break

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
