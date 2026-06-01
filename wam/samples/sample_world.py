import gc
from collections import defaultdict
from typing import Any, Dict, Iterator, Optional

import torch
import torch.nn.functional as F
import os
from accelerate.utils import broadcast_object_list

from wam.samples.quick_metrics import latent_l1
from wam.samples.video_report import finalize_video_metrics, report_decoded_video_pair


def value_to_value_frame(value: torch.Tensor, noisy_video: torch.Tensor) -> torch.Tensor:
    """将 value 扩展为与 noisy_video 单帧相同的空间尺寸 (b, c, 1, h, w)。"""
    b, c, f, h, w = noisy_video.shape
    if value.dim() == 2:
        assert value.shape == (b, 1), f"value (b,1) 期望 {(b, 1)}, 得到 {value.shape}"
        return value.view(b, 1, 1, 1, 1).expand(b, c, 1, h, w).contiguous()
    if value.dim() == 4:
        assert value.shape == (b, c, h, w), f"value (b,c,h,w) 期望 {(b, c, h, w)}, 得到 {value.shape}"
        return value.unsqueeze(2)
    if value.dim() == 5:
        assert value.shape == (b, c, 1, h, w), f"value (b,c,1,h,w) 期望 {(b, c, 1, h, w)}, 得到 {value.shape}"
        return value
    if value.dim() == 3:
        value = value[:, :1, 0]
        return value.view(b, 1, 1, 1, 1).expand(b, c, 1, h, w).contiguous()
    raise ValueError(f"value 维度应为 2/3/4/5，得到 dim={value.dim()}, shape={value.shape}")


def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    return model


def _broadcast_sample_batch(batch: Dict[str, Any], accelerator) -> Dict[str, Any]:
    """ZeRO-3 集体推理：rank0 的 batch 广播到各 rank，保证各卡同一输入、同步进入 forward。"""
    if accelerator.num_processes <= 1:
        return batch
    import torch.distributed as dist

    device = accelerator.device
    if accelerator.is_main_process:
        meta = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                meta[key] = ("tensor", tuple(value.shape), value.dtype)
            else:
                meta[key] = ("other", value)
    else:
        meta = None
    meta_holder = [meta]
    broadcast_object_list(meta_holder, from_process=0)
    meta = meta_holder[0]

    synced: Dict[str, Any] = {}
    for key, spec in meta.items():
        if spec[0] == "tensor":
            _, shape, dtype = spec
            if accelerator.is_main_process:
                tensor = batch[key].to(device=device, non_blocking=True).contiguous()
            else:
                tensor = torch.empty(shape, dtype=dtype, device=device)
            dist.broadcast(tensor, src=0)
            synced[key] = tensor
        else:
            synced[key] = spec[1]
    return synced


def _predict_action_sharded(model, accelerator, **kwargs):
    """各 rank 同步调用；ZeRO-3 下 ``unwrap_model`` 内参数仍分片，``predict_action`` 内 forward 逐层 all-gather。"""
    _ = accelerator
    return unwrap_model(model).predict_action(**kwargs)


def _should_export_joint_video(
    *,
    sample_light: bool,
    sample_joint_only: bool,
    sample_sharded_inference: bool,
    accelerator,
    rank_id: int,
    pa_flag: bool,
    pred_video,
) -> bool:
    if pa_flag or pred_video is None:
        return False
    if sample_light and not sample_joint_only:
        return False
    if sample_sharded_inference and not accelerator.is_main_process:
        return False
    return rank_id < 16


def _save_joint_sample_latents(
    sample_save_path: str,
    *,
    step: int,
    rank_id: int,
    video_latents_gt: torch.Tensor,
    video_latents_pred: torch.Tensor,
) -> None:
    """保存 GT / 预测 video latent，供离线 ``vae.decode_to_video``。"""
    os.makedirs(sample_save_path, exist_ok=True)
    path = os.path.join(sample_save_path, f"sample_latents_rank{rank_id}_case{step}.pt")
    torch.save(
        {
            "video_latents_gt": video_latents_gt.detach().float().cpu(),
            "video_latents_pred": video_latents_pred.detach().float().cpu(),
            "step": step,
            "rank": rank_id,
        },
        path,
    )


def _reduce_metric(accelerator, value: torch.Tensor, *, distributed_reduce: bool) -> float:
    value = value.detach().float()
    if not distributed_reduce or accelerator.num_processes == 1:
        return value.mean().item()
    return accelerator.gather(value).mean().item()


@torch.no_grad()
def log_sample_res(
    vae,
    text_encoder,
    vision_encoder,
    model,
    args,
    accelerator,
    weight_dtype,
    dataset_id2name,
    dataloader,
    logger,
    sample_save_path,
    action_only=False,
    distributed_reduce: bool = True,
    sample_light: bool = False,
    sample_sharded_inference: bool = False,
    compute_video_metrics: Optional[bool] = None,
    save_video: Optional[bool] = None,
):
    """
    Evaluate model performance on validation set and log metrics during training.
    
    This function performs inference on a subset of validation data to:
    1. Calculate action prediction errors (overall MSE / L1 等)
    2. Generate and save predicted videos for visual inspection
    3. Log 仅 overall 指标（不再按数据集名写 ``agilex_*`` 等 key）
    
    Args:
        vae: Variational autoencoder for video encoding/decoding
        text_encoder: Language encoder (CLIP/T5) for instruction embedding
        vision_encoder: Vision encoder (ResNet/ViT) for image feature extraction
        model: Main world model being trained (RDT-World)
        dataloader: Validation DataLoader (NOT training data)
        dataset_id2name: 与 ``train_world`` 调用兼容保留，本函数不再使用其写 log。
        logger: Logger instance for info messages
        sample_save_path: Directory path for saving output videos
        sample_sharded_inference: 为 True 时各 rank 对 **同一 batch** 做 ZeRO 分片集体
            ``predict_action``（与训练 forward 一致）；为 False 时各 rank 用 dataloader 分片各跑各的 batch。
        action_only: 为 True 时，每个 batch 会 **调用两次** ``predict_action``：
        先 ``action_only=True``（指标 key 带前缀 ``AE_``），再 ``action_only=False``（联合 world，无前缀）。
        为 False 时只调用一次 ``action_only=False``。
    
    Returns:
        dict: 仅 overall 类指标，例如 ``overall_avg_sample_mse``、``overall_avg_sample_l1``、
        ``overall_avg_sample_l1_value_loss``；若 ``action_only=True`` 另含 ``AE_*`` 前缀的对应项。
        不再包含 ``agilex_sample_mse`` 等按数据集名拼出来的 key。
    
    Side Effects:
        - Saves ``output_video_rank{R}_case{S}.mp4`` (ground truth left, prediction right)
        - ``sample_sharded_inference=True``：ZeRO 分片权重 + 各 rank 同一 batch（非每卡各扛一整份 6B）
        - ``distributed_reduce=True`` 且非 sharded 时，指标跨 rank 聚合不同 batch
        - Temporarily sets model to eval mode, restores train mode after
    """
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _ = dataset_id2name  # 与 train_world 调用签名兼容，本函数不再按数据集名打 log
        if accelerator.is_main_process:
            mode = "sharded (ZeRO collective)" if sample_sharded_inference else "data-parallel batches"
            logger.info(
                "Running sampling: %s global batch step(s), mode=%s, distributed_reduce=%s",
                args.num_sample_batches,
                mode,
                distributed_reduce,
            )

        model.eval()

        # 训练期轻量采样：只做 action-only 一次，避免 causal 下 AE+联合 两次 6B 推理 + VAE 解码 OOM。
        save_sample_video = bool(getattr(args, "sample_save_video", False)) if save_video is None else bool(save_video)
        sample_compute_video_metrics = (
            bool(getattr(args, "sample_compute_video_metrics", False))
            if compute_video_metrics is None
            else bool(compute_video_metrics)
        )
        sample_joint_only = bool(getattr(args, "sample_joint_only", False))
        if sample_joint_only:
            predict_runs = [(False, "")]
        elif sample_light:
            predict_runs = [(True, "AE_")] if action_only else [(False, "")]
        else:
            predict_runs = [(True, "AE_"), (False, "")] if action_only else [(False, "")]
        loss_for_log = defaultdict(float)
        if sample_light and not sample_joint_only and accelerator.is_main_process:
            logger.info("sample_light=true: skip joint world predict and video mp4 export.")
        if sample_joint_only and accelerator.is_main_process:
            logger.info("sample_joint_only=true: run joint world sampling only; skip action-only sampling.")
        if not save_sample_video and accelerator.is_main_process:
            logger.info(
                "sample_save_video=false: skip VAE decode/mp4; joint samples save "
                "sample_latents_rank*_case*.pt for offline decode."
            )
        if sample_compute_video_metrics and accelerator.is_main_process:
            logger.info("sample_compute_video_metrics=true: decode sample batches for video L1/PSNR/SSIM.")
        sync_each_batch = accelerator.num_processes > 1
        import torch.distributed as dist

        dl_iter: Iterator = iter(dataloader)
        for step in range(int(args.num_sample_batches)):
            if sample_sharded_inference:
                batch = None
                if accelerator.is_main_process:
                    try:
                        batch = next(dl_iter)
                    except StopIteration:
                        batch = None
                stop_flag = torch.tensor(
                    [1 if (accelerator.is_main_process and batch is None) else 0],
                    device=accelerator.device,
                    dtype=torch.int32,
                )
                dist.broadcast(stop_flag, src=0)
                if stop_flag.item() == 1:
                    break
                batch = _broadcast_sample_batch(batch, accelerator)
            else:
                try:
                    batch = next(dl_iter)
                except StopIteration:
                    break

            ctrl_freqs = batch["ctrl_freqs"]
            state_norm = batch["state_norm"].to(dtype=weight_dtype)
            images = batch["images"].to(dtype=weight_dtype)
            states = batch["states"].to(dtype=weight_dtype)
            needs_video_latents = any(not pa_flag for pa_flag, _ in predict_runs)
            video = batch["video"].to(dtype=weight_dtype) if needs_video_latents else None
            # We only use the last state as input
            states = states[:, -1:, :]
            actions = batch["actions"].to(dtype=weight_dtype)
            state_elem_mask = batch["state_elem_mask"].to(dtype=weight_dtype)

            vae_mini_batch = int(getattr(args, "vae_mini_batch", 1))

            if needs_video_latents:
                # encode with vae
                video = video.transpose(1, 2)
                video_latents = vae.encode_to_latents(video, vae_mini_batch=vae_mini_batch)
                condition_video_latents = vae.get_condition(
                    video, video_latents=video_latents, vae_mini_batch=int(getattr(args, "vae_mini_batch", 1))
                )
            else:
                video_latents = None
                condition_video_latents = None

            batch_size, _, C, H, W = images.shape
            image_embeds = vision_encoder(images.reshape(-1, C, H, W)).detach()
            image_embeds = image_embeds.reshape((batch_size, -1, vision_encoder.hidden_size))

            lang_attn_mask = batch["lang_attn_mask"]
            seq_lens = lang_attn_mask.gt(0).sum(dim=1).long()
            if args.precomp_lang_embed:
                text_embeds = batch["lang_embeds"].to(dtype=weight_dtype)
            else:
                text_embeds = text_encoder(
                    input_ids=batch["input_ids"], attention_mask=lang_attn_mask
                )["last_hidden_state"].detach()
            text_embeds = [u[:v] for u, v in zip(text_embeds, seq_lens)]
            text_embeds = torch.stack(
                [torch.cat([u, u.new_zeros(512 - u.size(0), u.size(1))]) for u in text_embeds], dim=0
            )

            rank_id = getattr(accelerator, "process_index", getattr(accelerator, "local_process_index", 0))
            for pa_flag, log_key_prefix in predict_runs:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                out = _predict_action_sharded(
                    model,
                    accelerator,
                    lang_tokens=text_embeds,
                    lang_attn_mask=lang_attn_mask,
                    img_tokens=image_embeds,
                    state_tokens=states,
                    action_mask=state_elem_mask.unsqueeze(1),
                    ctrl_freqs=ctrl_freqs,
                    video_latents=video_latents,
                    condition_video_latents=condition_video_latents,
                    action_only=pa_flag,
                    video_only=False,
                )
                pred_actions = out["pred_trajectory"]
                pred_video = out["pred_video"]
                pred_value = out["pred_value"]
                if pred_video is not None and video_latents is not None:
                    latent_l1_loss = latent_l1(pred_video, video_latents)
                    latent_l1_scaler = _reduce_metric(
                        accelerator, latent_l1_loss, distributed_reduce=distributed_reduce
                    )
                    loss_for_log[log_key_prefix + "overall_avg_sample_latent_l1"] += latent_l1_scaler
                if pred_value is not None:
                    value_gt = batch["value"].to(pred_value)
                    if pred_value.ndim == 5:
                        # 与 pred_video 空间维对齐得到 (B,C,1,H,W)，再与 pred_value 算 L1
                        ref = pred_video if pred_video is not None else pred_value
                        value_for_l1 = value_to_value_frame(value_gt, ref)
                        l1_value_loss = F.l1_loss(pred_value, value_for_l1).mean()
                    else:
                        l1_value_loss = F.l1_loss(pred_value, value_gt).mean()
                    l1_value_loss_scaler = _reduce_metric(
                        accelerator, l1_value_loss, distributed_reduce=distributed_reduce
                    )
                    loss_for_log[log_key_prefix + "overall_avg_sample_l1_value_loss"] += l1_value_loss_scaler

                # 联合路径：写 mp4 或仅落盘 latent（离线 VAE 解码）
                export_joint = _should_export_joint_video(
                    sample_light=sample_light,
                    sample_joint_only=sample_joint_only,
                    sample_sharded_inference=sample_sharded_inference,
                    accelerator=accelerator,
                    rank_id=rank_id,
                    pa_flag=pa_flag,
                    pred_video=pred_video,
                )
                can_decode_video_metrics = (
                    sample_compute_video_metrics
                    and pred_video is not None
                    and video_latents is not None
                    and (not sample_sharded_inference or accelerator.is_main_process)
                )
                decoded_for_export = export_joint and save_sample_video
                if can_decode_video_metrics or decoded_for_export:
                    video_orig = vae.decode_to_video(video_latents, to_save=True)[0]
                    video_pred = vae.decode_to_video(pred_video, to_save=True)[0]
                    n_frames = min(video_orig.shape[0], video_pred.shape[0])
                    video_orig = video_orig[:n_frames]
                    video_pred = video_pred[:n_frames]
                    metric_reduce = distributed_reduce and not sample_sharded_inference
                    video_metrics = report_decoded_video_pair(
                        gt_video=video_orig,
                        pred_video=video_pred,
                        sample_save_path=sample_save_path,
                        filename_prefix=f"output_video_rank{rank_id}_case{step}",
                        compute_metrics=can_decode_video_metrics,
                        save_video=(
                            decoded_for_export
                            and rank_id < int(getattr(args, "sample_video_max_rank", 16))
                        ),
                        reverse_video_order=bool(getattr(args, "reverse_video_order", False)),
                        save_forward_view=bool(getattr(args, "sample_save_forward_view", True)),
                        fps=max(1, n_frames // 5),
                    )
                    for metric_name, metric_value in video_metrics.items():
                        metric_scaler = _reduce_metric(
                            accelerator,
                            metric_value.to(accelerator.device),
                            distributed_reduce=metric_reduce,
                        )
                        loss_for_log[
                            log_key_prefix + "overall_avg_sample_" + metric_name
                        ] += metric_scaler
                elif export_joint and not save_sample_video:
                    _save_joint_sample_latents(
                        sample_save_path,
                        step=step,
                        rank_id=rank_id,
                        video_latents_gt=video_latents,
                        video_latents_pred=pred_video,
                    )

                num_steps = pred_actions.shape[1]
                expanded_state_elem_mask = (state_elem_mask.unsqueeze(1).tile((1, num_steps, 1)).float())

                loss = F.mse_loss(pred_actions, actions, reduction="none").float()
                l1_elem = (pred_actions - actions).abs().float()

                mse_loss = (loss * expanded_state_elem_mask).sum() / expanded_state_elem_mask.sum()
                mse_loss_scaler = _reduce_metric(
                    accelerator, mse_loss, distributed_reduce=distributed_reduce
                )
                loss_for_log[log_key_prefix + "overall_avg_sample_mse"] += mse_loss_scaler
                loss_for_log[log_key_prefix + "overall_avg_sample_action_mse"] += mse_loss_scaler

                l1_loss = (l1_elem * expanded_state_elem_mask).sum() / expanded_state_elem_mask.sum()
                l1_loss_scaler = _reduce_metric(
                    accelerator, l1_loss, distributed_reduce=distributed_reduce
                )
                loss_for_log[log_key_prefix + "overall_avg_sample_l1"] += l1_loss_scaler
                loss_for_log[log_key_prefix + "overall_avg_sample_action_l1"] += l1_loss_scaler
                del out, pred_actions, pred_video, pred_value, loss, l1_elem

            if sync_each_batch:
                accelerator.wait_for_everyone()
            del batch, state_norm, images, states, actions, state_elem_mask
            del image_embeds, text_embeds, lang_attn_mask, seq_lens
            if needs_video_latents:
                del video, video_latents, condition_video_latents
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        overall_mse_l1_keys = [
            "overall_avg_sample_mse",
            "overall_avg_sample_l1",
            "overall_avg_sample_action_mse",
            "overall_avg_sample_action_l1",
            "overall_avg_sample_latent_l1",
            "overall_avg_sample_video_l1",
            "overall_avg_sample_video_psnr",
            "overall_avg_sample_video_ssim",
        ]
        if action_only:
            overall_mse_l1_keys = [
                "AE_overall_avg_sample_mse",
                "AE_overall_avg_sample_l1",
                "AE_overall_avg_sample_action_mse",
                "AE_overall_avg_sample_action_l1",
                "AE_overall_avg_sample_latent_l1",
                "AE_overall_avg_sample_video_l1",
                "AE_overall_avg_sample_video_psnr",
                "AE_overall_avg_sample_video_ssim",
            ] + overall_mse_l1_keys
        value_loss_keys = ["overall_avg_sample_l1_value_loss"]
        if action_only:
            value_loss_keys = ["AE_overall_avg_sample_l1_value_loss"] + value_loss_keys

        for name in list(loss_for_log.keys()):
            if name in overall_mse_l1_keys:
                loss_for_log[name] = round(loss_for_log[name] / (args.num_sample_batches), 4)
            elif name in value_loss_keys:
                continue
        for vk in value_loss_keys:
            if vk in loss_for_log:
                loss_for_log[vk] = round(loss_for_log[vk] / (args.num_sample_batches), 4)
        prefixes = ("AE_", "") if action_only else ("",)
        finalize_video_metrics(loss_for_log, args.num_sample_batches, prefixes=prefixes)


        model.train()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        return dict(loss_for_log)
