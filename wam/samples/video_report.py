# coding=utf-8
"""Shared video sample metrics and visualization helpers."""

from __future__ import annotations

import os
from typing import Dict, Optional

import imageio
import numpy as np
import torch

from wam.samples.quick_metrics import compute_video_l1_psnr_ssim


VIDEO_METRIC_KEYS = (
    "overall_avg_sample_video_l1",
    "overall_avg_sample_video_psnr",
    "overall_avg_sample_video_ssim",
)


def maybe_reverse_video_batch(video: torch.Tensor, reverse_video_order: bool) -> torch.Tensor:
    """Reverse ``[B,T,C,H,W]`` video order when running backward-video experiments."""
    if reverse_video_order:
        return torch.flip(video, dims=[1])
    return video


def align_decoded_videos(gt_video: np.ndarray, pred_video: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_frames = min(gt_video.shape[0], pred_video.shape[0])
    return gt_video[:n_frames], pred_video[:n_frames]


def _save_concat_video(
    gt_video: np.ndarray,
    pred_video: np.ndarray,
    *,
    path: str,
    fps: Optional[int] = None,
) -> None:
    gt_video, pred_video = align_decoded_videos(gt_video, pred_video)
    n_frames = max(1, gt_video.shape[0])
    video_concat = np.concatenate([gt_video, pred_video], axis=2)
    imageio.mimsave(
        path,
        video_concat,
        fps=fps or max(1, n_frames // 5),
        codec="libx264",
    )


def save_video_pair_report(
    gt_video: np.ndarray,
    pred_video: np.ndarray,
    *,
    sample_save_path: str,
    filename_prefix: str,
    reverse_video_order: bool = False,
    save_forward_view: bool = True,
    fps: Optional[int] = None,
) -> None:
    os.makedirs(sample_save_path, exist_ok=True)
    if reverse_video_order:
        _save_concat_video(
            gt_video,
            pred_video,
            path=os.path.join(sample_save_path, f"{filename_prefix}_reverse_compare.mp4"),
            fps=fps,
        )
        if save_forward_view:
            _save_concat_video(
                gt_video[::-1],
                pred_video[::-1],
                path=os.path.join(sample_save_path, f"{filename_prefix}_forward_view.mp4"),
                fps=fps,
            )
    else:
        _save_concat_video(
            gt_video,
            pred_video,
            path=os.path.join(sample_save_path, f"{filename_prefix}.mp4"),
            fps=fps,
        )


def report_decoded_video_pair(
    *,
    gt_video: np.ndarray,
    pred_video: np.ndarray,
    sample_save_path: str,
    filename_prefix: str,
    compute_metrics: bool = True,
    save_video: bool = False,
    reverse_video_order: bool = False,
    save_forward_view: bool = True,
    fps: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    gt_video, pred_video = align_decoded_videos(gt_video, pred_video)
    metrics = compute_video_l1_psnr_ssim(pred_video, gt_video) if compute_metrics else {}
    if save_video:
        save_video_pair_report(
            gt_video,
            pred_video,
            sample_save_path=sample_save_path,
            filename_prefix=filename_prefix,
            reverse_video_order=reverse_video_order,
            save_forward_view=save_forward_view,
            fps=fps,
        )
    return metrics


def finalize_video_metrics(loss_for_log, divisor: int, *, prefixes=("",)) -> None:
    denom = max(1, int(divisor))
    for prefix in prefixes:
        for key in VIDEO_METRIC_KEYS:
            full_key = prefix + key
            if full_key in loss_for_log:
                loss_for_log[full_key] = round(loss_for_log[full_key] / denom, 4)
