# coding=utf-8
"""仅视频 Runner + Qwen2.5-VL 条件：训练期采样。"""

from collections import defaultdict

import imageio
import numpy as np
import os
import torch
import torch.nn.functional as F

from wam.samples.quick_metrics import compute_video_l1_psnr_ssim, latent_l1
from wam.trainers.video_qwen_helpers import encode_lang_tokens_for_video_batch, unwrap_model


def _reduce_metric(accelerator, value: torch.Tensor) -> float:
    value = value.detach().float().to(accelerator.device)
    if accelerator.num_processes <= 1:
        return value.mean().item()
    return accelerator.gather(value).mean().item()


@torch.no_grad()
def log_sample_res(
    vae,
    qwen_encoder,
    model,
    args,
    accelerator,
    weight_dtype,
    dataloader,
    logger,
    sample_save_path,
    *,
    lang_adapter=None,
    num_cond_frames: int = 4,
    max_images=None,
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

            text_embeds = encode_lang_tokens_for_video_batch(
                qwen_encoder,
                batch,
                use_precomp=args.precomp_lang_embed,
                num_cond_frames=num_cond_frames,
                max_images=max_images,
                weight_dtype=weight_dtype,
                lang_adapter=lang_adapter,
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
            latent_l1_loss = latent_l1(pred_video, video_latents)
            loss_for_log[log_key_prefix + "overall_avg_sample_latent_l1"] += _reduce_metric(
                accelerator, latent_l1_loss
            )
            if pred_value is not None:
                value_gt = batch["value"].to(pred_value)
                l1_value_loss = F.l1_loss(pred_value, value_gt).mean()
                l1_value_loss_scaler = _reduce_metric(accelerator, l1_value_loss)
                loss_for_log[log_key_prefix + "overall_avg_sample_l1_value_loss"] += l1_value_loss_scaler

            video_orig = vae.decode_to_video(video_latents, to_save=True)[0]
            video_pred = vae.decode_to_video(pred_video, to_save=True)[0]
            n_frames = min(video_orig.shape[0], video_pred.shape[0])
            video_metrics = compute_video_l1_psnr_ssim(video_pred[:n_frames], video_orig[:n_frames])
            for metric_name, metric_value in video_metrics.items():
                loss_for_log[log_key_prefix + "overall_avg_sample_" + metric_name] += _reduce_metric(
                    accelerator, metric_value
                )

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

        metric_keys = [
            "overall_avg_sample_l1_value_loss",
            "overall_avg_sample_latent_l1",
            "overall_avg_sample_video_l1",
            "overall_avg_sample_video_psnr",
            "overall_avg_sample_video_ssim",
        ]
        for vk in metric_keys:
            if vk in loss_for_log:
                loss_for_log[vk] = round(loss_for_log[vk] / (args.num_sample_batches), 4)

        model.train()
        torch.cuda.empty_cache()
        return dict(loss_for_log)
