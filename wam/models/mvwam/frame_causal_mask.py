# coding=utf-8
"""Frame-level causal attention masks shared by video/action/world paths."""

from __future__ import annotations

from typing import Optional

import torch


def bool_to_additive_mask(allowed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert a bool allowed mask to additive attention bias."""
    mask = torch.zeros(allowed.shape, device=allowed.device, dtype=dtype)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask


def merge_attention_bias(
    base_mask: Optional[torch.Tensor],
    additive_mask: Optional[torch.Tensor],
    *,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Merge an existing bool/additive mask with an additive mask."""
    if additive_mask is None:
        return base_mask
    if base_mask is None:
        return additive_mask
    if base_mask.dtype == torch.bool:
        base_mask = bool_to_additive_mask(base_mask.to(additive_mask.device), dtype)
    else:
        base_mask = base_mask.to(device=additive_mask.device, dtype=dtype)
    return base_mask + additive_mask


def build_video_frame_causal_bool_mask(
    num_frames: int,
    tokens_per_frame: int,
    device: torch.device,
) -> torch.Tensor:
    """Return ``[T*P, T*P]`` where each frame can see itself and earlier frames."""
    if num_frames <= 0 or tokens_per_frame <= 0:
        raise ValueError(
            f"num_frames and tokens_per_frame must be positive, got {num_frames=}, {tokens_per_frame=}"
        )
    frame_ids = torch.arange(num_frames, device=device).repeat_interleave(tokens_per_frame)
    return frame_ids[:, None] >= frame_ids[None, :]


def build_video_frame_causal_additive_mask(
    num_frames: int,
    tokens_per_frame: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return bool_to_additive_mask(
        build_video_frame_causal_bool_mask(num_frames, tokens_per_frame, device),
        dtype,
    )


def map_action_steps_to_video_frames(
    num_action_steps: int,
    num_video_frames: int,
    device: torch.device,
) -> torch.Tensor:
    """Map action step index to visible video cutoff using end-of-bin alignment.

    For action step ``a`` in ``[0, A)``, cutoff is
    ``ceil((a + 1) * F / A) - 1``. Integer form avoids floating point drift:
    ``((a + 1) * F - 1) // A``.
    """
    if num_action_steps <= 0 or num_video_frames <= 0:
        raise ValueError(
            f"num_action_steps and num_video_frames must be positive, got {num_action_steps=}, {num_video_frames=}"
        )
    action_ids = torch.arange(num_action_steps, device=device, dtype=torch.long)
    return ((action_ids + 1) * num_video_frames - 1) // num_action_steps


def build_action_self_causal_bool_mask(
    *,
    prefix_len: int,
    action_len: int,
    value_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Return action-expert self-attention visibility over ``[prefix, action, value]``.

    Prefix queries see only prefix tokens; action query ``t`` sees prefix and
    actions ``<= t``; value queries see the complete sequence.
    """
    if prefix_len < 0 or action_len < 0 or value_len < 0:
        raise ValueError(
            f"Token lengths must be non-negative, got {prefix_len=}, {action_len=}, {value_len=}"
        )
    total_len = prefix_len + action_len + value_len
    allowed = torch.zeros(total_len, total_len, device=device, dtype=torch.bool)
    if prefix_len > 0:
        allowed[:prefix_len, :prefix_len] = True
    for action_idx in range(action_len):
        query_idx = prefix_len + action_idx
        allowed[query_idx, : prefix_len + action_idx + 1] = True
    if value_len > 0:
        allowed[prefix_len + action_len :, :] = True
    return allowed


def build_action_self_causal_additive_mask(
    *,
    prefix_len: int,
    action_len: int,
    value_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return bool_to_additive_mask(
        build_action_self_causal_bool_mask(
            prefix_len=prefix_len,
            action_len=action_len,
            value_len=value_len,
            device=device,
        ),
        dtype,
    )


def build_action_to_video_frame_causal_bool_mask(
    *,
    query_len: int,
    prefix_len: int,
    action_len: int,
    value_len: int,
    num_video_frames: int,
    tokens_per_video_frame: int,
    device: torch.device,
) -> torch.Tensor:
    """Return ``[query_len, video_tokens]`` visibility for action queries over video K/V."""
    if query_len != prefix_len + action_len + value_len:
        raise ValueError(
            "query_len must equal prefix_len + action_len + value_len, got "
            f"{query_len=} {prefix_len=} {action_len=} {value_len=}"
        )
    video_frame_ids = torch.arange(num_video_frames, device=device).repeat_interleave(tokens_per_video_frame)
    allowed = torch.zeros(query_len, video_frame_ids.numel(), device=device, dtype=torch.bool)

    if prefix_len > 0:
        allowed[:prefix_len] = video_frame_ids[None, :] <= 0

    if action_len > 0:
        cutoffs = map_action_steps_to_video_frames(action_len, num_video_frames, device)
        allowed[prefix_len : prefix_len + action_len] = video_frame_ids[None, :] <= cutoffs[:, None]

    if value_len > 0:
        allowed[prefix_len + action_len :] = True

    return allowed
