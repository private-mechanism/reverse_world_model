#!/usr/bin/env python3
# coding=utf-8
"""监控训练采样目录中的 ``sample_latents_*.pt``，发现新文件后立刻用 Wan VAE 解码为 mp4。

与 ``wam.samples.sample_world._save_joint_sample_latents`` 配套：训练时 ``sample_save_video=false``
只落盘 latent，本脚本在另一进程/机器上离线解码，避免训练期 VAE OOM。

示例::

    python scripts/watch_sample_latents_decode.py \\
        --watch-dir checkpoints/demo-128dim-value/sample \\
        --model-config model_config/wam2.2_demo.yml

    # 或显式指定 Wan Diffusers 根目录（内含 vae/ 子目录）
    python scripts/watch_sample_latents_decode.py \\
        --watch-dir checkpoints/demo-128dim-value/sample \\
        --vae-path /path/to/Wan2.2-TI2V-5B-Diffusers
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Set

import imageio
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from configs.model_config_loader import load_model_config_dict
from wam.models.multimodal_encoder.vae_encoder import VAEEncoder

LATENT_NAME_RE = re.compile(r"^sample_latents_rank(\d+)_case(\d+)\.pt$")
logger = logging.getLogger("watch_sample_latents_decode")


def _resolve_vae_path(*, vae_path: Optional[str], model_config: Optional[str]) -> str:
    if vae_path:
        return str(Path(vae_path).expanduser().resolve())
    if not model_config:
        raise ValueError("请指定 --vae-path 或 --model-config")
    cfg = load_model_config_dict(str(Path(model_config).expanduser().resolve()))
    path = cfg.get("pretrained_text_encoder_name_or_path") or cfg.get("video_base_model")
    if not path:
        raise ValueError(f"{model_config} 中未找到 weights.text_encoder / video_base_model")
    return str(Path(path).expanduser().resolve())


def mp4_path_for_latent(latent_path: Path) -> Path:
    m = LATENT_NAME_RE.match(latent_path.name)
    if m is None:
        return latent_path.with_name(f"{latent_path.stem}_decoded.mp4")
    rank, case = m.group(1), m.group(2)
    return latent_path.with_name(f"output_video_rank{rank}_case{case}.mp4")


def marker_path(latent_path: Path) -> Path:
    return latent_path.with_name(latent_path.name + ".decoded")


def _load_latent_blob(latent_path: Path) -> dict:
    try:
        return torch.load(latent_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(latent_path, map_location="cpu")


def wait_file_stable(path: Path, *, poll: float, stable_rounds: int) -> bool:
    """等待写入结束（大小连续不变）。返回 False 表示文件已不存在。"""
    last_size = -1
    unchanged = 0
    while unchanged < stable_rounds:
        if not path.is_file():
            return False
        size = path.stat().st_size
        if size > 0 and size == last_size:
            unchanged += 1
        else:
            unchanged = 0
        last_size = size
        time.sleep(poll)
    return path.is_file()


def iter_latent_files(watch_dir: Path, *, recursive: bool) -> Iterable[Path]:
    pattern = "**/sample_latents_*.pt" if recursive else "sample_latents_*.pt"
    yield from sorted(watch_dir.glob(pattern))


def decode_latent_file(
    vae: VAEEncoder,
    latent_path: Path,
    *,
    vae_mini_batch: int,
    overwrite: bool,
) -> Optional[Path]:
    out_mp4 = mp4_path_for_latent(latent_path)
    done_marker = marker_path(latent_path)
    if not overwrite and (out_mp4.is_file() or done_marker.is_file()):
        return None

    blob = _load_latent_blob(latent_path)
    gt = blob.get("video_latents_gt")
    pred = blob.get("video_latents_pred")
    if gt is None or pred is None:
        raise KeyError(f"{latent_path} 缺少 video_latents_gt / video_latents_pred")

    video_orig = vae.decode_to_video(gt, vae_mini_batch=vae_mini_batch, to_save=True)[0]
    video_pred = vae.decode_to_video(pred, vae_mini_batch=vae_mini_batch, to_save=True)[0]
    n_frames = min(video_orig.shape[0], video_pred.shape[0])
    video_orig = video_orig[:n_frames]
    video_pred = video_pred[:n_frames]
    video_concat = np.concatenate([video_orig, video_pred], axis=2)

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(
        str(out_mp4),
        video_concat,
        fps=max(1, n_frames // 5),
        codec="libx264",
    )
    done_marker.write_text(f"mp4={out_mp4.name}\n", encoding="utf-8")
    return out_mp4


def process_pending(
    vae: VAEEncoder,
    watch_dir: Path,
    *,
    recursive: bool,
    poll: float,
    stable_rounds: int,
    vae_mini_batch: int,
    overwrite: bool,
    seen: Set[str],
) -> int:
    n_done = 0
    for latent_path in iter_latent_files(watch_dir, recursive=recursive):
        key = str(latent_path.resolve())
        out_mp4 = mp4_path_for_latent(latent_path)
        if not overwrite and (out_mp4.is_file() or marker_path(latent_path).is_file()):
            seen.add(key)
            continue
        if key in seen:
            continue
        if not wait_file_stable(latent_path, poll=poll, stable_rounds=stable_rounds):
            continue
        try:
            saved = decode_latent_file(
                vae, latent_path, vae_mini_batch=vae_mini_batch, overwrite=overwrite
            )
        except Exception:
            logger.exception("解码失败: %s", latent_path)
            continue
        seen.add(key)
        if saved is not None:
            n_done += 1
            logger.info("已解码 %s -> %s", latent_path, saved)
    return n_done


def main() -> None:
    parser = argparse.ArgumentParser(description="监控 sample_latents_*.pt 并解码为 mp4")
    parser.add_argument(
        "--watch-dir",
        type=str,
        required=True,
        help="训练 sample 根目录，如 checkpoints/<run>/sample（会递归子目录 step-*）",
    )
    parser.add_argument("--model-config", type=str, default=None, help="从 weights.text_encoder 读 VAE 根路径")
    parser.add_argument("--vae-path", type=str, default=None, help="Wan Diffusers 根目录（含 vae/）")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="扫描间隔（秒）")
    parser.add_argument(
        "--stable-rounds",
        type=int,
        default=2,
        help="文件大小连续不变多少次轮询后认为写入完成",
    )
    parser.add_argument("--vae-mini-batch", type=int, default=1, help="VAE decode mini-batch")
    parser.add_argument("--device", type=str, default="cuda", help="cuda 或 cpu")
    parser.add_argument("--no-recursive", action="store_true", help="不递归子目录")
    parser.add_argument("--once", action="store_true", help="只处理已有文件后退出")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有 mp4")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    watch_dir = Path(args.watch_dir).expanduser().resolve()
    if not watch_dir.is_dir():
        raise SystemExit(f"watch-dir 不存在: {watch_dir}")

    vae_root = _resolve_vae_path(vae_path=args.vae_path, model_config=args.model_config)
    logger.info("加载 VAE from %s", vae_root)
    vae = VAEEncoder(vae_root)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    vae.model.to(device)

    seen: Set[str] = set()
    recursive = not args.no_recursive
    logger.info("监控目录 %s (recursive=%s, poll=%ss)", watch_dir, recursive, args.poll_interval)

    while True:
        n = process_pending(
            vae,
            watch_dir,
            recursive=recursive,
            poll=max(0.2, args.poll_interval / max(1, args.stable_rounds)),
            stable_rounds=args.stable_rounds,
            vae_mini_batch=args.vae_mini_batch,
            overwrite=args.overwrite,
            seen=seen,
        )
        if args.once:
            logger.info("完成，本轮新解码 %s 个文件", n)
            break
        if n == 0:
            time.sleep(args.poll_interval)
        # 有产出时立即进入下一轮扫描（尽快处理连续写入的 latent）


if __name__ == "__main__":
    main()
