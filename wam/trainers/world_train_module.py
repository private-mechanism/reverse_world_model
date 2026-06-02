# coding=utf-8
"""World 训练一步：batch → 编码器 → model forward。

`trainers.train_world` 只负责 Accelerate 循环与优化器；具体前向与 batch 拆包集中在此，便于替换 model 实现
而少改 Trainer。参见 ``docs/refactor_trainer_runner_model.md``。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


def maybe_reverse_world_batch(video: torch.Tensor, actions: torch.Tensor, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reverse video/action targets for oracle reverse WVWAM experiments."""
    if not bool(getattr(args, "reverse_world_order", False)):
        return video, actions
    return torch.flip(video, dims=[1]), torch.flip(actions, dims=[1])


class TrainModule:
    """持有可训练 model（当前为 ``FMPRunner``），提供 ``training_step``。"""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def training_step(
        self,
        batch: Dict[str, Any],
        *,
        vae: Any,
        text_encoder: Optional[nn.Module],
        vision_encoder: Any,
        weight_dtype: torch.dtype,
        args: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            ``loss`` (``loss_a + loss_v``，标量 tensor，需 grad)、``loss_a``、``loss_v``、``loss_value``（可能为 None）。
        """
        video = batch["video"]
        images = batch["images"]
        states = batch["states"][:, -1:, :]
        actions = batch["actions"]
        video, actions = maybe_reverse_world_batch(video, actions, args)
        value = batch.get("value", None)
        state_elem_mask = batch["state_elem_mask"].to(dtype=weight_dtype)
        ctrl_freqs = batch["ctrl_freqs"]

        vae_mini_batch = int(getattr(args, "vae_mini_batch", 1))

        with torch.no_grad():
            video = video.transpose(1, 2).to(dtype=weight_dtype)
            video_latents = vae.encode_to_latents(video, vae_mini_batch=vae_mini_batch)
            condition_video_latents = vae.get_condition(
                video, video_latents=video_latents, vae_mini_batch=vae_mini_batch
            )

            batch_size, _, c, h, w = images.shape
            image_embeds = vision_encoder(images.reshape(-1, c, h, w)).detach()
            image_embeds = image_embeds.reshape((batch_size, -1, vision_encoder.hidden_size))

            lang_attn_mask = batch["lang_attn_mask"]
            seq_lens = lang_attn_mask.gt(0).sum(dim=1).long()
            text_embeds = (
                batch["lang_embeds"].to(dtype=weight_dtype)
                if args.precomp_lang_embed
                else text_encoder(
                    input_ids=batch["input_ids"], attention_mask=lang_attn_mask
                )["last_hidden_state"].detach()
            )
            text_embeds = [u[:v] for u, v in zip(text_embeds, seq_lens)]
            text_embeds = torch.stack(
                [torch.cat([u, u.new_zeros(512 - u.size(0), u.size(1))]) for u in text_embeds], dim=0
            )

        state_elem_mask = state_elem_mask.unsqueeze(1)
        loss_a, loss_v, loss_value = self.model(
            lang_tokens=text_embeds,
            lang_attn_mask=lang_attn_mask,
            img_tokens=image_embeds,
            state_tokens=states.float(),
            action_gt=actions.float(),
            action_mask=state_elem_mask,
            ctrl_freqs=ctrl_freqs,
            video_latents=video_latents,
            condition_video_latents=condition_video_latents,
            value=value,
        )
        loss = loss_a + loss_v
        return loss, loss_a, loss_v, loss_value
