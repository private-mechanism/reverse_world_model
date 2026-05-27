# coding=utf-8
"""Qwen2.5-VL + Video 训练：batch 内视频帧转 PIL、语言条件编码与 Wan text_dim 对齐。"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from wam.models.multimodal_encoder.qwen25_encoder import Qwen25Embedder


def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    return model


def chw_float_to_pil(frame: torch.Tensor) -> Image.Image:
    """``[C,H,W]`` float（约 0–1 或 0–255）→ RGB PIL。"""
    t = frame.detach().float().cpu()
    if t.ndim != 3:
        raise ValueError(f"期望 3D CHW，得到 shape={tuple(t.shape)}")
    if t.max() <= 1.0 + 1e-3:
        t = (t * 255.0).clamp(0, 255)
    arr = t.permute(1, 2, 0).numpy().astype(np.uint8)
    return Image.fromarray(arr)


def video_tensor_to_pil_frame_lists(
    video: torch.Tensor,
    *,
    num_frames: int = 4,
) -> List[List[Image.Image]]:
    """``video`` 为 ``[B, T, C, H, W]``（与 ``VideoDataset`` / collator 一致）。"""
    if video.ndim != 5:
        raise ValueError(f"期望 video [B,T,C,H,W]，得到 {tuple(video.shape)}")
    B, T, _, _, _ = video.shape
    n = max(1, min(int(num_frames), T))
    out: List[List[Image.Image]] = []
    for b in range(B):
        if n == 1:
            idx = [0]
        else:
            idx = [int(round(i * (T - 1) / (n - 1))) for i in range(n)]
        out.append([chw_float_to_pil(video[b, i]) for i in idx])
    return out


class VideoRunnerWithLangAdapter(nn.Module):
    """可选 ``Linear(qwen_dim → wan_text_dim)``，其余委托给 ``VideoRunner``。"""

    def __init__(self, runner: nn.Module, lang_adapter: Optional[nn.Linear] = None):
        super().__init__()
        self.runner = runner
        self.lang_adapter = lang_adapter

    def _map_lang(self, lang_tokens: torch.Tensor) -> torch.Tensor:
        if self.lang_adapter is not None:
            return self.lang_adapter(lang_tokens)
        return lang_tokens

    def forward(self, lang_tokens, video_latents, condition_video_latents, value=None):
        return self.runner(
            lang_tokens=self._map_lang(lang_tokens),
            video_latents=video_latents,
            condition_video_latents=condition_video_latents,
            value=value,
        )

    def predict_video(self, lang_tokens, video_latents, condition_video_latents, guidance: float = 5.0):
        return self.runner.predict_video(
            lang_tokens=self._map_lang(lang_tokens),
            video_latents=video_latents,
            condition_video_latents=condition_video_latents,
            guidance=guidance,
        )

    def __getattr__(self, name):
        if name in ("runner", "lang_adapter"):
            raise AttributeError(name)
        return getattr(self.runner, name)


@torch.no_grad()
def encode_lang_tokens_for_video_batch(
    qwen_encoder: Qwen25Embedder,
    batch: dict,
    *,
    use_precomp: bool,
    num_cond_frames: int,
    max_images: Optional[int],
    weight_dtype: torch.dtype,
    lang_adapter: Optional[nn.Linear] = None,
) -> torch.Tensor:
    if use_precomp:
        lang = batch["lang_embeds"].to(dtype=weight_dtype)
    else:
        texts = batch.get("instruction_text")
        if not texts:
            raise KeyError("在线 Qwen 编码需要 batch['instruction_text']，请使用更新后的 VideoDataset / DataCollator")
        video = batch["video"]
        frame_lists = video_tensor_to_pil_frame_lists(video, num_frames=num_cond_frames)
        samples: Sequence[Tuple[str, Optional[List[Image.Image]]]] = [
            (str(t), frames) for t, frames in zip(texts, frame_lists)
        ]
        lang, _ = qwen_encoder.get_vlm_embeddings_batch(samples, max_images=max_images)
        lang = lang.to(dtype=weight_dtype)

    if lang_adapter is not None:
        lang = lang_adapter(lang)
    return lang
