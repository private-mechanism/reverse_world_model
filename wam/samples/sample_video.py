# coding=utf-8
"""仅视频 Runner 训练期采样与指标（与 RoboTwin ``train/sample_video.py`` 对齐，适配 WAM 路径）。"""

from collections import defaultdict

import imageio
import numpy as np
import os
import torch
import torch.nn.functional as F


def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    return model


@torch.no_grad()
def log_sample_res(
    vae,
    text_encoder,
    model,
    args,
    accelerator,
    weight_dtype,
    dataloader,
    logger,
    sample_save_path,
):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logger.info(f"Running sampling for {args.num_sample_batches} batches...")

        model.eval()

        loss_for_log = defaultdict(float)
        log_key_prefix = ""

        for step, batch in enumerate(dataloader):
            if step >= args.num_sample_batches:
                break

            video = batch["video"].to(dtype=weight_dtype)
            video = video.transpose(1, 2)
            video_latents = vae.encode_to_latents(video)
            condition_video_latents = vae.get_condition(video)

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

            sample_guidance = float(getattr(args, "video_sample_guidance", 5.0))
            out = unwrap_model(model).predict_video(
                lang_tokens=text_embeds,
                video_latents=video_latents,
                condition_video_latents=condition_video_latents,
                guidance=sample_guidance,
            )

            pred_video = out["pred_video"]
            pred_value = out["pred_value"]
            if pred_value is not None:
                value_gt = batch["value"].to(pred_value)
                l1_value_loss = F.l1_loss(pred_value, value_gt).mean()
                l1_value_loss_scaler = accelerator.gather(l1_value_loss).mean().item()
                loss_for_log[log_key_prefix + "overall_avg_sample_l1_value_loss"] += l1_value_loss_scaler

            video_orig = vae.decode_to_video(video_latents, to_save=True)[0]
            video_pred = vae.decode_to_video(pred_video, to_save=True)[0]
            n_frames = min(video_orig.shape[0], video_pred.shape[0])

            rank_id = getattr(accelerator, "process_index", getattr(accelerator, "local_process_index", 0))
            if rank_id < 16:
                video_orig = video_orig[:n_frames]
                video_pred = video_pred[:n_frames]
                video_concat = np.concatenate([video_orig, video_pred], axis=2)
                video_name = f"output_video_rank{rank_id}_case{step}.mp4"
                imageio.mimsave(
                    os.path.join(sample_save_path, video_name),
                    video_concat,
                    fps=max(1, n_frames // 5),
                    codec="libx264",
                )

        value_loss_keys = ["overall_avg_sample_l1_value_loss"]
        for vk in value_loss_keys:
            if vk in loss_for_log:
                loss_for_log[vk] = round(loss_for_log[vk] / (args.num_sample_batches), 4)

        model.train()
        torch.cuda.empty_cache()

        return dict(loss_for_log)
