#!/usr/bin/env python
# coding=utf-8
"""仅训练 Video Expert（Wan），语言/视觉条件由 **Qwen2.5-VL** 提供（不再使用 SigLIP + umT5）。

在线：从 batch 视频帧 + ``instruction_text`` 经 Qwen2.5-VL 编码；可选 ``lang_adapter`` 将 hidden 对齐 Wan ``text_dim``。
离线：``precomp_lang_embed=true`` 且 ``precomp_lang_embed_prefix: qwen25``（见 ``data/preprocess/batch_lang_embed.py``）。

``train_runner_module: runner_video``；入口 ``main_video_qwen.py``。
"""

import copy
import logging
import math
import os
import shutil
from pathlib import Path

import diffusers
import torch
import transformers
import yaml
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration, set_seed
from diffusers.utils import is_wandb_available
from huggingface_hub import create_repo, upload_folder
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from torch.utils.data import Subset

from configs.model_config_loader import load_model_config_dict, load_model_structure_dict
from data.dataset import DataCollatorForVLAConsumerDataset
from data.video_dataset import VideoDataset
from wam.models.ema_model import EMAModel
import torch.nn as nn

from wam.models.multimodal_encoder.qwen25_encoder import Qwen25Embedder
from wam.models.multimodal_encoder.vae_encoder import VAEEncoder
from wam.runners.runner_registry import import_train_runner_class, normalize_train_runner_module_name
from wam.samples.sample_video_qwen import log_sample_res
from wam.trainers.video_qwen_helpers import (
    VideoRunnerWithLangAdapter,
    encode_lang_tokens_for_video_batch,
    unwrap_model,
)
from wam.trainers.checkpoint_utils import prune_old_checkpoints
from wam.trainers.lr_schedulers import build_training_lr_scheduler
from wam.trainers.train_world import apply_model_config_to_training_args
from wam.models.mvwam.build_model import compute_img_cond_len_and_pos_embed_config
from wam.models.mvwam.world_stack_registry import (
    VIDEO_BASE_MODEL_PATHS,
    resolve_video_base_model_path,
)

if is_wandb_available():
    import wandb


def _resolve_sample_period(args, attr_name: str) -> int:
    period = int(getattr(args, attr_name, -1))
    if period < 0:
        period = int(getattr(args, "sample_period", -1))
    return period


def _should_run_periodic(global_step: int, period: int) -> bool:
    return period > 0 and global_step % period == 0


def _should_save_video_visual(args, global_step: int, period: int) -> bool:
    first_sample = bool(getattr(args, "sample_video_visual_on_step0", True)) and global_step == 1
    return first_sample or _should_run_periodic(global_step, period)


def _maybe_reverse_video(video: torch.Tensor, args) -> torch.Tensor:
    if bool(getattr(args, "reverse_video_order", False)):
        return torch.flip(video, dims=[1])
    return video


def save_model_card(repo_id: str, base_model: str, repo_folder=None):
    yaml_front = f"""
---
license: mit
base_model: {base_model}
language:
- en
pipeline_tag: robotics
library_name: transformers
tags:
- robotics
- pytorch
- multimodal
- pretraining
- vla
- diffusion
- rdt
---
    """
    model_card = f"""
# RDT - {repo_id}

This is a RDT model derived from {base_model}. The weights were trained using [RDT](https://rdt-robotics.github.io/rdt-robotics/).
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml_front + model_card)


_WAM_ROOT = Path(__file__).resolve().parents[2]


def _repo_empty_lang_dim(model_config: dict) -> int:
    p = model_config.get("empty_lang_embed_path") or str(_WAM_ROOT / "data" / "empty_lang_embed.pt")
    t = torch.load(p, map_location="cpu")
    return int(t.shape[-1])


def _ensure_uncond_embed_for_cfg(
    accelerator,
    model_config: dict,
    qwen_encoder: Qwen25Embedder | None,
    lang_adapter: nn.Linear | None,
    lang_max_length: int,
    wan_text_dim: int,
    weight_dtype: torch.dtype,
    logger,
) -> str | None:
    """生成/复用 CFG 用无条件语言向量（形状须为 ``[1, L, wan_text_dim]``）。"""
    out_dir = Path(model_config.get("checkpoint_root_dir", "checkpoints")) / model_config.get(
        "checkpoint_path", "video_qwen"
    ).replace("checkpoints/", "")
    uncond_path = model_config.get("uncond_lang_embed_path") or str(out_dir / "uncond_qwen25_wan.pt")
    if os.path.isfile(uncond_path):
        return uncond_path
    if not accelerator.is_main_process:
        accelerator.wait_for_everyone()
        return uncond_path if os.path.isfile(uncond_path) else None
    if qwen_encoder is None:
        logger.warning("precomp 模式且不存在 %s：采样 CFG 可能维度不匹配，请预生成 uncond", uncond_path)
        accelerator.wait_for_everyone()
        return uncond_path if os.path.isfile(uncond_path) else None
    os.makedirs(os.path.dirname(uncond_path) or ".", exist_ok=True)
    emb, _ = qwen_encoder.get_vlm_embeddings("", images=None)
    emb = emb.to(dtype=weight_dtype)
    if lang_adapter is not None:
        emb = lang_adapter(emb)
    if emb.shape[-1] != wan_text_dim:
        raise ValueError(f"uncond embed dim {emb.shape[-1]} != Wan text_dim {wan_text_dim}")
    if emb.shape[1] != lang_max_length:
        logger.info("uncond seq_len %s（lang_max_length=%s）", emb.shape[1], lang_max_length)
    torch.save(emb[0:1].cpu(), uncond_path)
    logger.info("Wrote CFG uncond embed: %s", uncond_path)
    accelerator.wait_for_everyone()
    return uncond_path


def _video_base_model_implies_wan22(model_config: dict) -> bool:
    """``VIDEO_BASE_MODEL`` 为 Wan2.2 注册键或其绝对路径时，应使用 ``runner_video_wan2_2``。"""
    v = model_config.get("VIDEO_BASE_MODEL")
    if v is None:
        return False
    s = str(v).strip()
    if not s or s.lower() in ("none", "null", "nil"):
        return False
    wan22_path = VIDEO_BASE_MODEL_PATHS.get("Wan2.2-TI2V-5B-Diffusers")
    if s == "Wan2.2-TI2V-5B-Diffusers":
        return True
    if wan22_path and os.path.expanduser(s) == os.path.expanduser(wan22_path):
        return True
    try:
        resolved = resolve_video_base_model_path(s)
    except ValueError:
        return False
    return resolved == wan22_path


def train(args, logger):
    model_config = load_model_config_dict(args.model_config_path)
    train_mod = model_config.get("train_runner_module")
    if train_mod not in (None, ""):
        model_config["train_runner_module"] = normalize_train_runner_module_name(str(train_mod))
    apply_model_config_to_training_args(model_config, args)

    # 确定 video_variant：yml 显式指定 > VIDEO_BASE_MODEL 路径自动检测 > 默认 wow
    video_variant = str(model_config.get("video_variant", "")).strip().lower()
    if not video_variant or video_variant in ("none", "null"):
        video_variant = "wan22" if _video_base_model_implies_wan22(model_config) else "wow"
    else:
        video_variant = "wan22" if video_variant in ("wan22", "wan2.2", "wan2_2") else "wow"

    runner_mod = str(model_config.get("train_runner_module", "")).lower()
    if "runner_video" not in runner_mod:
        raise ValueError(
            "wam/trainers/train_video_qwen 仅用于仅视频 Runner：请在 model_config 中设置 "
            "``train_runner_module: runner_video``（旧名 ``rdt_runner_*`` 亦可），"
            "``train_runner_class: VideoRunner``。"
            f" 当前 train_runner_module={model_config.get('train_runner_module')!r}"
        )

    config = load_model_structure_dict(model_config)
    args.reverse_video_order = bool(config.get("dataset", {}).get("reverse_video_order", False))
    vbm = model_config.get("VIDEO_BASE_MODEL")
    if vbm is not None and str(vbm).strip() != "" and str(vbm).strip().lower() not in ("none", "null"):
        m = config.setdefault("model", {})
        if isinstance(m, dict):
            m["VIDEO_BASE_MODEL"] = vbm
    config_ = OmegaConf.create(config)

    args.output_dir = model_config["checkpoint_path"].replace("checkpoints/", "")
    args.output_dir = os.path.join(args.checkpoint_root_dir, args.output_dir)
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
    accelerator = Accelerator(
        deepspeed_plugin=(DeepSpeedPlugin(hf_ds_config=args.deepspeed) if args.deepspeed is not None else None),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb" and not is_wandb_available():
        raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None or model_config.get("seed", None) is not None:
        set_seed(model_config.get("seed", args.seed))

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            model_cfg_dir = os.path.join(args.output_dir, "configs", "model_config")
            model_structure_cfg_dir = os.path.join(args.output_dir, "configs", "model_structure")
            os.makedirs(model_cfg_dir, exist_ok=True)
            os.makedirs(model_structure_cfg_dir, exist_ok=True)
            model_src = Path(args.model_config_path).expanduser().resolve()
            shutil.copy2(model_src, os.path.join(model_cfg_dir, model_src.name))
            with open(os.path.join(model_structure_cfg_dir, "model_structure.yaml"), "w", encoding="utf-8") as fp:
                yaml.safe_dump(config, fp, allow_unicode=True, sort_keys=False)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    qwen_path = model_config.get("pretrained_qwen25_embed_path")
    if not qwen_path:
        raise ValueError(
            "train_video_qwen 需要在 model_config weights.qwen25_embed（或 pretrained_qwen25_embed_path）中指定 Qwen2.5-VL 路径"
        )
    lang_max_length = int(
        model_config.get("qwen_model_max_length", config["dataset"].get("tokenizer_max_length", 512))
    )
    qwen_num_cond_frames = int(model_config.get("qwen_num_cond_frames", 4))
    qwen_max_images = model_config.get("qwen_max_images")
    if qwen_max_images is not None:
        qwen_max_images = int(qwen_max_images)

    qwen_encoder = None
    if not args.precomp_lang_embed:
        qwen_encoder = Qwen25Embedder(
            device=accelerator.device,
            from_pretrained=qwen_path,
            model_max_length=lang_max_length,
            torch_dtype=weight_dtype,
        )

    vae_ckpt = model_config.get("pretrained_text_encoder_name_or_path") or args.pretrained_text_encoder_name_or_path
    if not vae_ckpt:
        vae_ckpt = model_config.get("VIDEO_BASE_MODEL") or args.pretrained_text_encoder_name_or_path
    vae = VAEEncoder(vae_ckpt)

    args.pretrained_model_name_or_path = model_config.get(
        "pretrained_model_name_or_path", args.pretrained_model_name_or_path
    )

    runner_cls = import_train_runner_class(
        model_config.get("train_runner_module"),
        model_config.get("train_runner_class"),
    )

    if args.pretrained_model_name_or_path is not None and not os.path.isfile(args.pretrained_model_name_or_path):
        logger.info("Constructing VideoRunner from pretrained checkpoint.")
        video_model = runner_cls.from_pretrained(
            args.pretrained_model_name_or_path,
            video_variant=video_variant,
        )
    else:
        logger.info("Constructing VideoRunner from provided config.")
        img_cond_len, img_pos_embed_config = compute_img_cond_len_and_pos_embed_config(config["common"], 1)
        video_model = runner_cls(
            pred_horizon=config["common"]["action_chunk_size"],
            config=config_.model,
            img_cond_len=img_cond_len,
            img_pos_embed_config=img_pos_embed_config,
            pretrained_video_expert_path=args.pretrained_video_expert_path,
            video_base_model=model_config.get("VIDEO_BASE_MODEL"),
            video_variant=video_variant,
        )

    wan_text_dim = int(getattr(video_model.model.config, "text_dim", 4096))
    qwen_dim = int(model_config.get("qwen_hidden_size", 0)) or (
        qwen_encoder.hidden_size if qwen_encoder is not None else int(model_config.get("lang_token_dim", 0))
    )
    if not qwen_dim and args.precomp_lang_embed:
        sample_empty = _repo_empty_lang_dim(model_config)
        qwen_dim = sample_empty
    lang_adapter = None
    if qwen_dim and qwen_dim != wan_text_dim:
        lang_adapter = nn.Linear(qwen_dim, wan_text_dim, dtype=weight_dtype)
        logger.info("lang_adapter: %s -> %s (Wan text_dim)", qwen_dim, wan_text_dim)

    uncond_path = _ensure_uncond_embed_for_cfg(
        accelerator,
        model_config,
        qwen_encoder,
        lang_adapter,
        lang_max_length,
        wan_text_dim,
        weight_dtype,
        logger,
    )
    if uncond_path:
        video_model._uncond_lang_embed_path = uncond_path

    video_model = VideoRunnerWithLangAdapter(video_model, lang_adapter)

    ema_shadow = copy.deepcopy(video_model)
    ema_cfg = config["model"].get("ema") or {}
    ema_model = EMAModel(
        ema_shadow,
        update_after_step=ema_cfg.get("update_after_step", 0),
        inv_gamma=ema_cfg.get("inv_gamma", 1.0),
        power=ema_cfg.get("power", 2 / 3),
        min_value=ema_cfg.get("min_value", 0.0),
        max_value=ema_cfg.get("max_value", 0.9999),
    )

    _trainable_cls = type(accelerator.unwrap_model(video_model))

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for wrapped in models:
                to_save = wrapped.module if hasattr(wrapped, "module") else wrapped
                if isinstance(to_save, _trainable_cls):
                    inner = unwrap_model(to_save)
                    if isinstance(inner, VideoRunnerWithLangAdapter):
                        inner.runner.save_pretrained(output_dir)
                        if inner.lang_adapter is not None:
                            torch.save(
                                inner.lang_adapter.state_dict(),
                                os.path.join(output_dir, "lang_adapter.pt"),
                            )
                    else:
                        to_save.save_pretrained(output_dir)
        while weights:
            weights.pop()

    accelerator.register_save_state_pre_hook(save_model_hook)

    if args.gradient_checkpointing:
        _inner = unwrap_model(video_model)
        _runner = _inner.runner if isinstance(_inner, VideoRunnerWithLangAdapter) else _inner
        fn = getattr(_runner.model, "enable_gradient_checkpointing", None)
        if callable(fn):
            fn()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, install bitsandbytes: `pip install bitsandbytes`."
            ) from None
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    train_params = list(video_model.parameters())
    optimizer = optimizer_class(
        train_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lang_prefix = model_config.get("precomp_lang_embed_prefix", "qwen25")
    train_dataset = VideoDataset(
        model_config_path=args.model_config_path,
        config=config["dataset"],
        tokenizer=None,
        image_processor=None,
        lang_embed_prefix_override=lang_prefix if args.precomp_lang_embed else None,
        num_cameras=config["common"]["num_cameras"],
        img_history_size=config["common"]["img_history_size"],
        video_size=tuple(config["dataset"]["video_size"]),
        dataset_type=args.dataset_type,
        image_aug=args.image_aug,
        image_aug_type=model_config.get("image_aug_type", "mixed"),
        cond_mask_prob=args.cond_mask_prob,
        cam_ext_mask_prob=args.cam_ext_mask_prob,
        state_noise_snr=args.state_noise_snr,
        use_precomp_lang_embed=args.precomp_lang_embed,
    )
    total_indices = range(len(train_dataset))
    sample_dataset = Subset(train_dataset, total_indices)
    train_dataset = Subset(train_dataset, total_indices)
    data_collator = DataCollatorForVLAConsumerDataset(None)
    _nw = max(0, int(args.dataloader_num_workers))
    _persistent = _nw > 0

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=_nw,
        pin_memory=True,
        persistent_workers=_persistent,
    )
    sample_dataloader = torch.utils.data.DataLoader(
        sample_dataset,
        batch_size=args.sample_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=_nw,
        pin_memory=True,
        persistent_workers=_persistent,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    video_model, optimizer, train_dataloader, sample_dataloader = accelerator.prepare(
        video_model, optimizer, train_dataloader, sample_dataloader
    )

    ema_model.averaged_model.to(accelerator.device, dtype=weight_dtype)

    if qwen_encoder is not None:
        qwen_encoder.model.to(accelerator.device, dtype=weight_dtype)
    if vae is not None:
        vae.model.to(accelerator.device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = build_training_lr_scheduler(optimizer, args, accelerator, logger)
    lr_scheduler = accelerator.prepare(lr_scheduler)

    if accelerator.is_main_process:
        accelerator.init_trackers(
            model_config.get("WANB_PROJECT_NAME", "VLA-Video"),
            config=vars(args),
            init_kwargs={"wandb": {
                "name": f"{args.CONFIG_NAME}",
            }},
        )
        if args.report_to == "wandb":
            import wandb as _wandb

            _wandb.run.log_code(
                root=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                include_fn=lambda path: path.endswith((".py", ".json", ".yaml", ".yml")),
            )

    vae_mini_batch = int(model_config.get("vae_mini_batch", 16))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running video-only training (Qwen2.5-VL cond) *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %s",
        total_batch_size,
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    progress_bar = tqdm(
        total=args.max_train_steps,
        initial=global_step,
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        video_model.train()
        for batch in train_dataloader:
            with accelerator.accumulate(video_model):
                video = _maybe_reverse_video(batch["video"], args)
                with torch.no_grad():
                    batch_for_lang = dict(batch)
                    batch_for_lang["video"] = video
                    video = video.transpose(1, 2).to(dtype=weight_dtype)
                    video_latents = vae.encode_to_latents(video, vae_mini_batch=vae_mini_batch)
                    condition_video_latents = vae.get_condition(video)
                    text_embeds = encode_lang_tokens_for_video_batch(
                        qwen_encoder,
                        batch_for_lang,
                        use_precomp=args.precomp_lang_embed,
                        num_cond_frames=qwen_num_cond_frames,
                        max_images=qwen_max_images,
                        weight_dtype=weight_dtype,
                        lang_adapter=unwrap_model(video_model).lang_adapter,
                    )

                loss, _loss_val = video_model(
                    lang_tokens=text_embeds,
                    video_latents=video_latents,
                    condition_video_latents=condition_video_latents,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(video_model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                    ema_model.step(accelerator.unwrap_model(video_model))

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_period == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    ema_save_path = os.path.join(save_path, "ema")
                    accelerator.save_model(ema_model.averaged_model, ema_save_path)
                    if accelerator.is_main_process:
                        logger.info(f"Saved state to {save_path}")
                        prune_old_checkpoints(
                            args.output_dir, args.checkpoints_total_limit, log=logger
                        )
                    accelerator.wait_for_everyone()

                metric_period = _resolve_sample_period(args, "sample_video_metric_period")
                visual_period = _resolve_sample_period(args, "sample_video_visual_period")
                run_video_metrics = (
                    bool(getattr(args, "sample_compute_video_metrics", True))
                    and _should_run_periodic(global_step, metric_period)
                )
                run_video_visual = (
                    bool(getattr(args, "sample_save_video", False))
                    and _should_save_video_visual(args, global_step, visual_period)
                )
                if run_video_metrics or run_video_visual:
                    sample_save_path = os.path.join(args.output_dir, "sample", f"step-{global_step}")
                    os.makedirs(sample_save_path, exist_ok=True)
                    sample_loss_for_log = log_sample_res(
                        vae,
                        qwen_encoder,
                        video_model,
                        args,
                        accelerator,
                        weight_dtype,
                        sample_dataloader,
                        logger,
                        sample_save_path,
                        lang_adapter=unwrap_model(video_model).lang_adapter,
                        num_cond_frames=qwen_num_cond_frames,
                        max_images=qwen_max_images,
                        compute_video_metrics=run_video_metrics,
                        save_video=run_video_visual,
                    )
                    logger.info(sample_loss_for_log)
                    accelerator.log(sample_loss_for_log, step=global_step)

                logs = {"loss_v": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final = unwrap_model(accelerator.unwrap_model(video_model))
        if isinstance(final, VideoRunnerWithLangAdapter):
            final.runner.save_pretrained(args.output_dir)
            if final.lang_adapter is not None:
                torch.save(final.lang_adapter.state_dict(), os.path.join(args.output_dir, "lang_adapter.pt"))
        else:
            final.save_pretrained(args.output_dir)
        accelerator.save_model(ema_model.averaged_model, os.path.join(args.output_dir, "ema"))
        logger.info(f"Saved Model to {args.output_dir}")
        if args.push_to_hub:
            save_model_card(
                repo_id,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of video training",
                token=args.hub_token,
                allow_patterns=["pytorch_model.bin", "*.json", "*.md"],
            )

    accelerator.end_training()
