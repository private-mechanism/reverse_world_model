# coding=utf-8
"""Video-only Runner + Qwen2.5-VL training-time sampling."""

from collections import defaultdict

import torch
import torch.nn.functional as F

from wam.samples.quick_metrics import latent_l1
from wam.samples.video_report import (
    finalize_video_metrics,
    maybe_reverse_video_batch,
    report_decoded_video_pair,
)
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
    compute_video_metrics: bool = True,
    save_video: bool = False,
):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logger.info(f"Running sampling for {args.num_sample_batches} batches...")
        model.eval()
        loss_for_log = defaultdict(float)
        log_key_prefix = ""

        reverse_video_order = bool(getattr(args, "reverse_video_order", False))
        for step, batch in enumerate(dataloader):
            if step >= args.num_sample_batches:
                break

            video = batch["video"].to(dtype=weight_dtype)
            video = maybe_reverse_video_batch(video, reverse_video_order)
            batch_for_lang = dict(batch)
            batch_for_lang["video"] = video
            video = video.transpose(1, 2)
            video_latents = vae.encode_to_latents(video)
            condition_video_latents = vae.get_condition(video)

            text_embeds = encode_lang_tokens_for_video_batch(
                qwen_encoder,
                batch_for_lang,
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

            if compute_video_metrics or save_video:
                video_orig = vae.decode_to_video(video_latents, to_save=True)[0]
                video_pred = vae.decode_to_video(pred_video, to_save=True)[0]
                n_frames = min(video_orig.shape[0], video_pred.shape[0])
                rank_id = getattr(accelerator, "process_index", getattr(accelerator, "local_process_index", 0))
                video_metrics = report_decoded_video_pair(
                    gt_video=video_orig[:n_frames],
                    pred_video=video_pred[:n_frames],
                    sample_save_path=sample_save_path,
                    filename_prefix=f"output_video_rank{rank_id}_case{step}",
                    compute_metrics=compute_video_metrics,
                    save_video=save_video and rank_id < int(getattr(args, "sample_video_max_rank", 16)),
                    reverse_video_order=reverse_video_order,
                    save_forward_view=bool(getattr(args, "sample_save_forward_view", True)),
                    fps=max(1, n_frames // 5),
                )
                for metric_name, metric_value in video_metrics.items():
                    loss_for_log[log_key_prefix + "overall_avg_sample_" + metric_name] += _reduce_metric(
                        accelerator, metric_value
                    )

        metric_keys = [
            "overall_avg_sample_l1_value_loss",
            "overall_avg_sample_latent_l1",
        ]
        for vk in metric_keys:
            if vk in loss_for_log:
                loss_for_log[vk] = round(loss_for_log[vk] / (args.num_sample_batches), 4)
        finalize_video_metrics(loss_for_log, args.num_sample_batches)

        model.train()
        torch.cuda.empty_cache()
        return dict(loss_for_log)
