#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import copy
import gc
import logging
import math
import os
import shutil
from pathlib import Path
from typing import List, Optional

import diffusers
import torch
import torch.nn as nn
import transformers
import yaml
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration, set_seed
from diffusers.utils import is_wandb_available
from huggingface_hub import create_repo, upload_folder
from omegaconf import OmegaConf
from torch.utils.data import Subset
from tqdm.auto import tqdm

from configs.model_config_loader import load_model_config_dict, load_model_structure_dict
from data.dataset import DataCollatorForVLAConsumerDataset, VLAConsumerDataset
from wam.models.ema_model import EMAModel
from wam.models.multimodal_encoder.siglip2_encoder import SiglipVisionTower
from wam.models.multimodal_encoder.umt5_encoder import umT5Embedder
from wam.models.multimodal_encoder.vae_encoder import VAEEncoder
from wam.models.mvwam.build_model import compute_img_cond_len_and_pos_embed_config
from wam.runners.runner_registry import import_train_runner_class, normalize_train_runner_module_name
from wam.samples.sample_world import log_sample_res
from wam.trainers.checkpoint_utils import prune_old_checkpoints
from wam.trainers.lr_schedulers import build_training_lr_scheduler
from wam.trainers.world_train_module import TrainModule

if is_wandb_available():
    import wandb


# ---------------------------------------------------------------------------
# model_config yml → training args
# ---------------------------------------------------------------------------


def resolve_deepspeed_config_path(config_value, working_dir: str) -> Optional[str]:
    """将 yml 中的 deepspeed 字段解析为 DeepSpeedPlugin 的 json 路径；None 表示关闭。"""
    if config_value is None:
        return None
    path_str = str(config_value).strip()
    lowered = path_str.lower()
    if lowered in ("", "off", "false", "no", "none"):
        return None
    if lowered in ("zero0", "zero1", "zero2", "zero3"):
        return os.path.join(working_dir, "configs", f"{lowered}.json")
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(working_dir, path_str)


def coerce_scalar_config_value(value):
    """将 yml 扁平标量转为与 argparse 一致的 Python 类型。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return str(value)


def apply_model_config_to_training_args(model_config: dict, args) -> None:
    """训练相关参数以 model_config yml 为准写入 ``args``（在 ``parse_args`` 之后调用）。

    凡 ``model_config`` 中与 ``args`` 同名的**扁平标量**键，按值的类型自动写入；嵌套 dict/list 跳过。
    ``consumer_dataset_type`` 写入 ``args.dataset_type``（finetune / pretrain 分支），
    与 yml 里 HDF5 布局字段 ``dataset_type``（如 ``128dim_with_value``）区分。
    """
    working_dir = os.getcwd()
    keys_skip_auto_apply = frozenset(
        {
            "dataset_type",
            "consumer_dataset_type",
            "deepspeed",
            "CONFIG_NAME",
        }
    )

    for key, value in model_config.items():
        if value is None or key in keys_skip_auto_apply:
            continue
        if not hasattr(args, key):
            continue
        if isinstance(value, (dict, list)):
            continue
        setattr(args, key, coerce_scalar_config_value(value))

    if "consumer_dataset_type" in model_config and model_config["consumer_dataset_type"] is not None:
        args.dataset_type = str(model_config["consumer_dataset_type"])
    else:
        args.dataset_type = "finetune"

    if "deepspeed" in model_config:
        args.deepspeed = resolve_deepspeed_config_path(model_config["deepspeed"], working_dir)

    if "CONFIG_NAME" in model_config and model_config["CONFIG_NAME"] not in (None, "", "null", "Null"):
        args.CONFIG_NAME = str(model_config["CONFIG_NAME"])
    config_name = getattr(args, "CONFIG_NAME", None)
    if config_name is None or str(config_name).strip() == "" or str(config_name).lower() in ("null", "none"):
        args.CONFIG_NAME = Path(args.model_config_path).stem


# ---------------------------------------------------------------------------
# 冻结推理编码器
# ---------------------------------------------------------------------------


def freeze_module_for_inference(module: Optional[nn.Module], module_name: str, logger=None) -> None:
    """eval + requires_grad_(False)，避免编码器进入优化器或意外建图。"""
    if module is None:
        return
    log = logger if logger is not None else logging.getLogger(__name__)
    module.eval()
    frozen_element_count = 0
    for parameter in module.parameters():
        if parameter.requires_grad:
            parameter.requires_grad_(False)
            frozen_element_count += parameter.numel()
    log.info("Frozen inference module %s (%s parameter elements).", module_name, frozen_element_count)


def freeze_world_inference_encoders(
    *,
    text_encoder: Optional[nn.Module],
    vision_encoder: Optional[SiglipVisionTower],
    vae: Optional[VAEEncoder],
    logger,
) -> None:
    if text_encoder is not None:
        freeze_module_for_inference(text_encoder, "text_encoder (umT5)", logger)
    if vision_encoder is not None:
        vision_encoder.eval()
        if getattr(vision_encoder, "is_loaded", False):
            freeze_module_for_inference(vision_encoder.vision_tower, "vision_encoder (SigLIP)", logger)
    if vae is not None and getattr(vae, "model", None) is not None:
        freeze_module_for_inference(vae.model, "vae", logger)


def iter_trainable_parameters(module: nn.Module) -> List[torch.nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


# ---------------------------------------------------------------------------
# 训练期采样
# ---------------------------------------------------------------------------


def resolve_sample_sharded_inference(args, *, use_ema_for_sampling: bool) -> bool:
    """DeepSpeed 下默认 ZeRO 分片集体推理；EMA 采样为完整权重副本，自动退回非分片路径。"""
    if getattr(args, "sample_sharded_inference", None) is not None:
        sharded = bool(args.sample_sharded_inference)
    else:
        sharded = args.deepspeed is not None
    if sharded and use_ema_for_sampling:
        return False
    return sharded


def resolve_sample_light_mode(args) -> bool:
    """训练期采样是否用轻量模式（默认：开 DeepSpeed 时为 true）。"""
    if getattr(args, "sample_light", None) is not None:
        return bool(args.sample_light)
    return args.deepspeed is not None


def release_memory_after_sampling(
    accelerator: Accelerator,
    vae,
    text_encoder,
    vision_encoder,
    *,
    weight_dtype: torch.dtype,
) -> None:
    """采样后回收 GPU/主机内存，并确保 VAE/编码器仍在训练 device 上。"""
    encoder_modules = frozen_encoder_modules_for_checkpoint(vae, text_encoder, vision_encoder)
    if encoder_modules:
        reload_modules_to_device(encoder_modules, accelerator.device)
        for module in encoder_modules:
            if weight_dtype is not None and hasattr(module, "to"):
                module.to(dtype=weight_dtype)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    accelerator.wait_for_everyone()


def run_training_sample_logging(
    *,
    enabled: bool,
    vae,
    text_encoder,
    vision_encoder,
    model,
    args,
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
    dataset_id2name,
    sample_dataloader,
    logger,
    sample_save_path: str,
    action_only: bool,
    sample_light: bool = False,
    sample_sharded_inference: bool = False,
) -> dict:
    """训练期采样。``sample_sharded_inference=True`` 时与 ZeRO 训练一致：各 rank 同一 batch、分片权重集体
    ``predict_action``；否则各 rank 用 dataloader 分片各跑各的 batch（数据并行采样）。"""
    if not enabled:
        accelerator.wait_for_everyone()
        return {}
    accelerator.wait_for_everyone()
    num_sample_batches = int(getattr(args, "num_sample_batches", 1))
    if sample_sharded_inference:
        distributed_reduce = False
    else:
        distributed_reduce = accelerator.num_processes > 1
    if accelerator.is_main_process:
        os.makedirs(sample_save_path, exist_ok=True)
        if sample_sharded_inference:
            logger.info(
                "ZeRO sharded sampling: %s rank(s), %s collective batch step(s) "
                "(same batch all ranks, model params sharded like training; "
                "causal_world_training=%s, sample_light=%s).",
                accelerator.num_processes,
                num_sample_batches,
                action_only,
                sample_light,
            )
        else:
            logger.info(
                "Data-parallel sampling: %s rank(s), %s batch(es)/rank, "
                "~%s steps (causal_world_training=%s, sample_light=%s).",
                accelerator.num_processes,
                num_sample_batches,
                num_sample_batches * accelerator.num_processes,
                action_only,
                sample_light,
            )
    accelerator.wait_for_everyone()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    metrics = log_sample_res(
        vae,
        text_encoder,
        vision_encoder,
        model,
        args,
        accelerator,
        weight_dtype,
        dataset_id2name,
        sample_dataloader,
        logger,
        sample_save_path,
        action_only=action_only,
        distributed_reduce=distributed_reduce,
        sample_light=sample_light,
        sample_sharded_inference=sample_sharded_inference,
    )
    release_memory_after_sampling(
        accelerator,
        vae,
        text_encoder,
        vision_encoder,
        weight_dtype=weight_dtype,
    )
    return metrics


# ---------------------------------------------------------------------------
# checkpoint / 内存
# ---------------------------------------------------------------------------


def release_gpu_memory_before_checkpoint(accelerator: Accelerator, model: nn.Module) -> bool:
    """存盘前尽量腾出 GPU 峰值（SIGKILL/-9 多为 OOM）。返回原先是否 train 模式。"""
    accelerator.wait_for_everyone()
    was_training = model.training
    model.eval()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    accelerator.wait_for_everyone()
    return was_training


def frozen_encoder_modules_for_checkpoint(vae, text_encoder, vision_encoder) -> List[nn.Module]:
    """训练期常驻 GPU 的冻结编码器，存盘前可暂卸到 CPU 以降低 rank0 峰值。"""
    modules: List[nn.Module] = []
    if vae is not None and getattr(vae, "model", None) is not None:
        modules.append(vae.model)
    if text_encoder is not None:
        modules.append(text_encoder)
    if vision_encoder is not None and getattr(vision_encoder, "vision_tower", None) is not None:
        modules.append(vision_encoder.vision_tower)
    return modules


def offload_modules_to_cpu(modules: List[nn.Module]) -> None:
    for module in modules:
        module.cpu()


def reload_modules_to_device(modules: List[nn.Module], device: torch.device) -> None:
    for module in modules:
        module.to(device)


def save_training_checkpoint(
    accelerator: Accelerator,
    save_path: str,
    logger: logging.Logger,
    *,
    model: nn.Module,
    use_deepspeed: bool,
    ema_model: Optional[EMAModel],
    aux_modules: Optional[List[nn.Module]] = None,
    output_dir: Optional[str] = None,
    checkpoints_total_limit: Optional[int] = None,
) -> None:
    """周期/结束存盘；DeepSpeed 仅写分片 ckpt（见 zero3 ``stage3_gather_16bit_weights_on_model_save``）。"""
    encoder_modules = list(aux_modules or [])
    if encoder_modules:
        offload_modules_to_cpu(encoder_modules)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    was_training = release_gpu_memory_before_checkpoint(accelerator, model)
    if accelerator.is_main_process:
        logger.info("Saving checkpoint to %s (DeepSpeed=%s)...", save_path, use_deepspeed)
    try:
        accelerator.save_state(save_path)
    finally:
        if encoder_modules:
            reload_modules_to_device(encoder_modules, accelerator.device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        if was_training:
            model.train()
    accelerator.wait_for_everyone()
    if ema_model is not None:
        ema_save_path = os.path.join(save_path, "ema")
        accelerator.save_model(ema_model.averaged_model, ema_save_path)
    if accelerator.is_main_process:
        logger.info("Saved state to %s", save_path)
        prune_root = output_dir or os.path.dirname(save_path)
        num_pruned = prune_old_checkpoints(prune_root, checkpoints_total_limit, log=logger)
        if num_pruned:
            logger.info(
                "checkpoints_total_limit=%s: removed %s old checkpoint dir(s) under %s",
                checkpoints_total_limit,
                num_pruned,
                prune_root,
            )
    accelerator.wait_for_everyone()


def cap_dataloader_workers_for_ddp(
    num_workers: int,
    *,
    num_processes: int,
    use_deepspeed: bool,
    logger: logging.Logger,
    max_per_process: int = 2,
) -> int:
    """多卡 DDP 且无 DeepSpeed 时限制 worker，兼顾吞吐与采样后主机内存。"""
    num_workers = max(0, int(num_workers))
    if use_deepspeed or num_processes <= 1:
        return num_workers
    cap = max(0, int(max_per_process))
    if num_workers > cap:
        logger.warning(
            "Multi-GPU DDP（无 DeepSpeed）：dataloader_num_workers %s -> %s（%s 进程，"
            "上限 %s；全 0 时 HDF5 解码易成瓶颈）",
            num_workers,
            cap,
            num_processes,
            cap,
        )
        return cap
    return num_workers


def is_ema_enabled(ema_config: dict) -> bool:
    """``training.ema.enabled``；缺省 True，与历史「始终开启 EMA」行为一致。"""
    if not ema_config or "enabled" not in ema_config:
        return True
    enabled_value = ema_config["enabled"]
    if isinstance(enabled_value, bool):
        return enabled_value
    if isinstance(enabled_value, str):
        return enabled_value.strip().lower() not in ("false", "0", "no", "off", "none")
    return bool(enabled_value)


def save_model_card(repo_id: str, base_model: str, repo_folder=None):
    card_yaml = f"""
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
    with open(os.path.join(repo_folder, "README.md"), "w") as file_handle:
        file_handle.write(card_yaml + model_card)


# ---------------------------------------------------------------------------
# 主训练入口
# ---------------------------------------------------------------------------


def train(args, logger):
    model_config = load_model_config_dict(args.model_config_path)
    runner_module_name = model_config.get("train_runner_module")
    if runner_module_name not in (None, ""):
        model_config["train_runner_module"] = normalize_train_runner_module_name(str(runner_module_name))

    apply_model_config_to_training_args(model_config, args)

    model_structure = load_model_structure_dict(model_config)
    video_base_model_name = model_config.get("VIDEO_BASE_MODEL")
    if (
        video_base_model_name is not None
        and str(video_base_model_name).strip() != ""
        and str(video_base_model_name).strip().lower() not in ("none", "null")
    ):
        model_section = model_structure.setdefault("model", {})
        if isinstance(model_section, dict):
            model_section["VIDEO_BASE_MODEL"] = video_base_model_name
    omega_model_config = OmegaConf.create(model_structure)

    causal_world_training = model_config.get("causal_world_training", False)
    is_128dim_dataset = "128dim" in model_config.get("dataset_type", "")

    if not (causal_world_training and is_128dim_dataset):
        raise ValueError(
            "wam/trainers/train_world 仅支持 model_config 中 causal_world_training=true 且 dataset_type 含 128dim；"
            f"当前 causal_world_training={causal_world_training!r}, dataset_type={model_config.get('dataset_type', '')!r}"
        )

    args.output_dir = model_config["checkpoint_path"].replace("checkpoints/", "")
    args.output_dir = os.path.join(args.checkpoint_root_dir, args.output_dir)
    logging_dir = Path(args.output_dir, args.logging_dir)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 冻结编码器须在 DeepSpeed Accelerator 之前加载完整权重；ZeRO-3 初始化后再 from_pretrained 会把参数切成碎片，
    # 导致 SigLIP 等出现 ``conv2d: weight should have at least three dimensions``。
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    encoder_device = torch.device("cuda", local_rank)

    if args.precomp_lang_embed:
        tokenizer, text_encoder = None, None
    else:
        text_embedder = umT5Embedder(
            device=encoder_device,
            from_pretrained=args.pretrained_text_encoder_name_or_path,
            model_max_length=model_structure["dataset"]["tokenizer_max_length"],
            torch_dtype=weight_dtype,
        )
        tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model

    vision_encoder = SiglipVisionTower(vision_tower=args.pretrained_vision_encoder_name_or_path, args=None)
    image_processor = vision_encoder.image_processor
    vae = VAEEncoder(args.pretrained_text_encoder_name_or_path)
    if text_encoder is not None:
        text_encoder.to(encoder_device, dtype=weight_dtype)
    vision_encoder.vision_tower.to(encoder_device, dtype=weight_dtype)
    vae.model.to(encoder_device, dtype=weight_dtype)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
    accelerator = Accelerator(
        deepspeed_plugin=(DeepSpeedPlugin(hf_ds_config=args.deepspeed) if args.deepspeed is not None else None),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    freeze_world_inference_encoders(
        text_encoder=text_encoder,
        vision_encoder=vision_encoder,
        vae=vae,
        logger=logger,
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
                yaml.safe_dump(model_structure, fp, allow_unicode=True, sort_keys=False)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id

    runner_cls = import_train_runner_class(
        model_config.get("train_runner_module"),
        model_config.get("train_runner_class"),
    )

    if args.pretrained_model_name_or_path is not None and not os.path.isfile(args.pretrained_model_name_or_path):
        logger.info("Constructing model from pretrained checkpoint.")
        img_cond_len, img_pos_embed_config = compute_img_cond_len_and_pos_embed_config(
            model_structure["common"], vision_encoder.num_patches
        )
        model = runner_cls.from_pretrained(
            args.pretrained_model_name_or_path,
            pred_horizon=model_structure["common"]["action_chunk_size"],
            config=omega_model_config.model,
            img_cond_len=img_cond_len,
            img_pos_embed_config=img_pos_embed_config,
            model_name=model_config.get("model_name"),
            model_type=model_config.get("model_type"),
            video_base_model=model_config.get("VIDEO_BASE_MODEL"),
            video_variant=model_config.get("video_variant"),
        )
    else:
        logger.info("Constructing model from provided config.")
        img_cond_len, img_pos_embed_config = compute_img_cond_len_and_pos_embed_config(
            model_structure["common"], vision_encoder.num_patches
        )
        model = runner_cls(
            pred_horizon=model_structure["common"]["action_chunk_size"],
            config=omega_model_config.model,
            img_cond_len=img_cond_len,
            img_pos_embed_config=img_pos_embed_config,
            pretrained_wam_path=getattr(args, "pretrained_wam_path", None),
            pretrained_video_expert_path=args.pretrained_video_expert_path,
            pretrained_action_expert_path=args.pretrained_action_expert_path,
            model_name=model_config.get("model_name"),
            model_type=model_config.get("model_type"),
            video_base_model=model_config.get("VIDEO_BASE_MODEL"),
            video_variant=model_config.get("video_variant"),
        )

    train_module = TrainModule(model=model)

    ema_config = model_structure["model"].get("ema") or {}
    if args.deepspeed is not None and (not ema_config or "enabled" not in ema_config):
        ema_config = dict(ema_config)
        ema_config["enabled"] = False
        logger.warning(
            "DeepSpeed is enabled and training.ema.enabled is not set; disabling EMA by default. "
            "Set training.ema.enabled=true explicitly if you have memory for a full EMA replica per rank."
        )
    ema_enabled = is_ema_enabled(ema_config)
    ema_model = None
    use_ema_for_sampling = False
    if ema_enabled:
        ema_shadow = copy.deepcopy(model)
        ema_model = EMAModel(
            ema_shadow,
            update_after_step=ema_config.get("update_after_step", 0),
            inv_gamma=ema_config.get("inv_gamma", 1.0),
            power=ema_config.get("power", 2 / 3),
            min_value=ema_config.get("min_value", 0.0),
            max_value=ema_config.get("max_value", 0.9999),
        )
        use_ema_for_sampling = bool(ema_config.get("use_ema_for_sampling", True))
        logger.info("EMA enabled (use_ema_for_sampling=%s).", use_ema_for_sampling)
    else:
        logger.info("EMA disabled (training.ema.enabled=false); sampling and checkpoints use trainable weights only.")

    def unwrap_wrapped_model(wrapped_model):
        return wrapped_model.module if hasattr(wrapped_model, "module") else wrapped_model

    def sync_ema_weights_from_model() -> None:
        """将 EMA 影子权重与当前可训练模型对齐（加载 ckpt / resume 后必须调用）。"""
        if ema_model is None:
            return
        ema_model.averaged_model.load_state_dict(unwrap_wrapped_model(model).state_dict())

    def get_sampling_model():
        """验证/中间采样用：EMA 开启且 ``use_ema_for_sampling`` 时用 EMA 权重，否则用当前训练权重。"""
        if ema_model is not None and use_ema_for_sampling:
            return ema_model.averaged_model
        return model

    def try_load_ema_weights_from_checkpoint(checkpoint_dir: str) -> bool:
        """若目录下存在 ``ema/model.safetensors`` 或 ``ema/pytorch_model.bin``，则加载到 EMA。"""
        if ema_model is None:
            return False
        ema_dir = os.path.join(checkpoint_dir, "ema")
        safetensors_path = os.path.join(ema_dir, "model.safetensors")
        pytorch_bin_path = os.path.join(ema_dir, "pytorch_model.bin")
        try:
            if os.path.isfile(safetensors_path):
                from safetensors.torch import load_file

                ema_model.averaged_model.load_state_dict(load_file(safetensors_path), strict=False)
                return True
            if os.path.isfile(pytorch_bin_path):
                try:
                    state_dict = torch.load(pytorch_bin_path, map_location="cpu", weights_only=True)
                except TypeError:
                    state_dict = torch.load(pytorch_bin_path, map_location="cpu")
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                ema_model.averaged_model.load_state_dict(state_dict, strict=False)
                return True
        except Exception as error:
            logger.warning("加载 EMA 权重失败（将用当前 trainable 同步）: %s", error)
        return False

    trainable_model_class = type(unwrap_wrapped_model(model))
    use_deepspeed = args.deepspeed is not None

    def save_model_hook(models, weights, output_dir):
        if use_deepspeed:
            while weights:
                weights.pop()
            return
        if accelerator.is_main_process:
            for wrapped_model in models:
                model_to_save = wrapped_model.module if hasattr(wrapped_model, "module") else wrapped_model  # type: ignore
                if isinstance(model_to_save, trainable_model_class):
                    model_to_save.save_pretrained(output_dir)
        while weights:
            weights.pop()

    accelerator.register_save_state_pre_hook(save_model_hook)
    if not use_deepspeed and accelerator.is_main_process:
        logger.info(
            "DDP 存盘：save_pretrained 后已 pop weights，避免与 save_state 重复写模型；"
            "若仍 SIGKILL(-9) 请增大 checkpointing_period 或启用 DeepSpeed ZeRO-3。"
        )
    if use_deepspeed and accelerator.is_main_process:
        logger.info(
            "DeepSpeed 已启用：周期 checkpoint 仅保存 DeepSpeed 分片（不额外写 HuggingFace save_pretrained）。"
            "恢复训练用 ``accelerator.load_state``；合并 fp32 权重见 DeepSpeed ``zero_to_fp32``。"
            "zero3 已 offload optimizer 到 CPU；若存盘仍 OOM 请增大 checkpointing_period。"
        )

    if args.gradient_checkpointing:
        world_policy_model = getattr(model, "model", None)
        if world_policy_model is None or not hasattr(world_policy_model, "gradient_checkpointing"):
            raise ValueError(
                "gradient_checkpointing=true 但 Runner 内无 ``model.gradient_checkpointing``（期望 WorldPolicyModel）。"
            )
        world_policy_model.gradient_checkpointing = True
        logger.info("Gradient checkpointing enabled on WorldPolicyModel (per-layer forward_joint).")

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
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    trainable_params = iter_trainable_parameters(model)
    if not trainable_params:
        raise ValueError("没有 requires_grad=True 的参数，请检查 Runner 是否正确加载。")
    logger.info("Optimizer trainable elements: %.3e", float(sum(parameter.numel() for parameter in trainable_params)))

    optimizer = optimizer_class(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = VLAConsumerDataset(
        model_config_path=args.model_config_path,
        config=model_structure["dataset"],
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_cameras=model_structure["common"]["num_cameras"],
        img_history_size=model_structure["common"]["img_history_size"],
        video_size=model_structure["dataset"]["video_size"],
        dataset_type=args.dataset_type,
        image_aug=args.image_aug,
        image_aug_type=model_config.get("image_aug_type", "mixed"),
        cond_mask_prob=args.cond_mask_prob,
        cam_ext_mask_prob=args.cam_ext_mask_prob,
        state_noise_snr=args.state_noise_snr,
        use_hdf5=args.load_from_hdf5,
        use_precomp_lang_embed=args.precomp_lang_embed,
        precomp_lang_embed_prefix=getattr(args, "precomp_lang_embed_prefix", None),
    )
    all_indices = range(len(train_dataset))
    sample_dataset = Subset(train_dataset, all_indices)
    train_dataset = Subset(train_dataset, all_indices)
    data_collator = DataCollatorForVLAConsumerDataset(tokenizer)
    dataloader_num_workers = cap_dataloader_workers_for_ddp(
        args.dataloader_num_workers,
        num_processes=accelerator.num_processes,
        use_deepspeed=use_deepspeed,
        logger=logger,
    )
    persistent_workers = dataloader_num_workers > 0

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=dataloader_num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    sample_dataloader = torch.utils.data.DataLoader(
        sample_dataset,
        batch_size=args.sample_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=dataloader_num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    model, optimizer, train_dataloader, sample_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, sample_dataloader
    )
    train_module.model = model
    params_to_clip = iter_trainable_parameters(accelerator.unwrap_model(model))
    if not params_to_clip:
        params_to_clip = [parameter for group in optimizer.param_groups for parameter in group["params"]]

    if ema_model is not None:
        ema_model.averaged_model.to(accelerator.device, dtype=weight_dtype)

    if text_encoder is not None:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    if vision_encoder is not None:
        vision_encoder.vision_tower.to(accelerator.device, dtype=weight_dtype)

    if vae is not None:
        vae.model.to(accelerator.device, dtype=weight_dtype)

    freeze_world_inference_encoders(
        text_encoder=text_encoder,
        vision_encoder=vision_encoder,
        vae=vae,
        logger=logger,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = build_training_lr_scheduler(optimizer, args, accelerator, logger)
    lr_scheduler = accelerator.prepare(lr_scheduler)

    if accelerator.is_main_process:
        accelerator.init_trackers(
            model_config.get("WANB_PROJECT_NAME", "VLA"),
            config=vars(args),
            init_kwargs={"wandb": {
                "name": f"{args.CONFIG_NAME}",
            }},
        )
        if args.report_to == "wandb":
            import wandb

            wandb.run.log_code(
                root=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                include_fn=lambda path: path.endswith((".py", ".json", ".yaml", ".yml")),
            )

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    args.pretrained_model_name_or_path = model_config.get(
        "pretrained_model_name_or_path", args.pretrained_model_name_or_path
    )
    if args.pretrained_model_name_or_path is not None and os.path.exists(args.pretrained_model_name_or_path):
        logger.info("Loading from a pretrained checkpoint.")
        deepspeed_state_path = os.path.join(
            args.pretrained_model_name_or_path, "pytorch_model", "mp_rank_00_model_states.pt"
        )
        safetensors_path = os.path.join(args.pretrained_model_name_or_path, "model.safetensors")
        if os.path.exists(deepspeed_state_path):
            checkpoint = torch.load(deepspeed_state_path)
            model.load_state_dict(checkpoint["module"])
        elif os.path.exists(safetensors_path):
            from safetensors.torch import load_file

            checkpoint = load_file(safetensors_path)
            if hasattr(model, "module"):
                model.module.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
        else:
            raise ValueError(f"Unknown checkpoint format: {args.pretrained_model_name_or_path}")
        if ema_model is not None:
            sync_ema_weights_from_model()
            ema_model.optimization_step = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            checkpoint_name = os.path.basename(args.resume_from_checkpoint)
        else:
            checkpoint_dirs = os.listdir(args.output_dir)
            checkpoint_dirs = [name for name in checkpoint_dirs if name.startswith("checkpoint")]
            checkpoint_dirs = sorted(checkpoint_dirs, key=lambda name: int(name.split("-")[1]))
            checkpoint_name = checkpoint_dirs[-1] if len(checkpoint_dirs) > 0 else None

        if checkpoint_name is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {checkpoint_name}")
            try:
                accelerator.load_state(os.path.join(args.output_dir, checkpoint_name))
            except Exception:
                logger.info("Resuming training state failed. Attempting to only load from model checkpoint.")
                checkpoint = torch.load(
                    os.path.join(
                        args.output_dir,
                        checkpoint_name,
                        "pytorch_model",
                        "mp_rank_00_model_states.pt",
                    )
                )
                missing_keys, unexpected_keys = unwrap_wrapped_model(model).load_state_dict(
                    checkpoint["module"], strict=False
                )
                assert len(missing_keys) == 0, f"Missing keys: {missing_keys}"

            checkpoint_dir = os.path.join(args.output_dir, checkpoint_name)
            global_step = int(checkpoint_name.split("-")[1])
            if ema_model is not None:
                if not try_load_ema_weights_from_checkpoint(checkpoint_dir):
                    sync_ema_weights_from_model()
                ema_model.optimization_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        total=args.max_train_steps,
        initial=global_step,
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    sample_enabled = int(getattr(args, "sample_period", -1)) > 0
    sample_light = resolve_sample_light_mode(args)
    sample_sharded_inference = resolve_sample_sharded_inference(
        args, use_ema_for_sampling=use_ema_for_sampling
    )
    if sample_enabled and accelerator.is_main_process:
        logger.info(
            "Training sample_light=%s, sample_sharded_inference=%s "
            "(ZeRO: same batch + sharded weights; EMA sampling forces non-sharded).",
            sample_light,
            sample_sharded_inference,
        )

    extra_log_metrics = {}
    for epoch in range(first_epoch, args.num_train_epochs):
        model.train()

        for batch in train_dataloader:
            with accelerator.accumulate(model):
                loss, loss_action, loss_video, loss_value = train_module.training_step(
                    batch,
                    vae=vae,
                    text_encoder=text_encoder,
                    vision_encoder=vision_encoder,
                    weight_dtype=weight_dtype,
                    args=args,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                    if ema_model is not None:
                        ema_model.step(accelerator.unwrap_model(model))

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_period == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    save_training_checkpoint(
                        accelerator,
                        save_path,
                        logger,
                        model=model,
                        use_deepspeed=use_deepspeed,
                        ema_model=ema_model,
                        aux_modules=frozen_encoder_modules_for_checkpoint(vae, text_encoder, vision_encoder),
                        output_dir=args.output_dir,
                        checkpoints_total_limit=args.checkpoints_total_limit,
                    )

                if sample_enabled and global_step % args.sample_period == 0:
                    sample_model = model if sample_sharded_inference else get_sampling_model()
                    sample_metrics = run_training_sample_logging(
                        enabled=True,
                        vae=vae,
                        text_encoder=text_encoder,
                        vision_encoder=vision_encoder,
                        model=sample_model,
                        args=args,
                        accelerator=accelerator,
                        weight_dtype=weight_dtype,
                        dataset_id2name=sample_dataset.dataset.get_dataset_id2name(),
                        sample_dataloader=sample_dataloader,
                        logger=logger,
                        sample_save_path=os.path.join(args.output_dir, "sample", f"step-{global_step}"),
                        action_only=causal_world_training,
                        sample_light=sample_light,
                        sample_sharded_inference=sample_sharded_inference,
                    )
                    if sample_metrics and accelerator.is_main_process:
                        logger.info(sample_metrics)
                        accelerator.log(sample_metrics, step=global_step)

            if accelerator.sync_gradients:
                logs = {
                    "loss_a": loss_action.detach().item(),
                    "loss_v": loss_video.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                if loss_value is not None:
                    logs.update({"loss_value": loss_value.detach().item()})
                progress_bar.set_postfix(**logs)
                logs.update(extra_log_metrics)
                accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if use_deepspeed:
        final_checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        save_training_checkpoint(
            accelerator,
            final_checkpoint_path,
            logger,
            model=model,
            use_deepspeed=True,
            ema_model=ema_model,
            aux_modules=frozen_encoder_modules_for_checkpoint(vae, text_encoder, vision_encoder),
            output_dir=args.output_dir,
            checkpoints_total_limit=args.checkpoints_total_limit,
        )
        if accelerator.is_main_process:
            logger.info(
                "Training finished. DeepSpeed checkpoint at %s (resume: load_state). "
                "Merge to fp32 with DeepSpeed zero_to_fp32 if needed.",
                final_checkpoint_path,
            )
    elif accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(args.output_dir)
        logger.info("Saved model to %s", args.output_dir)

    if ema_model is not None and accelerator.is_main_process:
        ema_save_path = os.path.join(args.output_dir, "ema")
        accelerator.save_model(ema_model.averaged_model, ema_save_path)

        if args.push_to_hub:
            save_model_card(
                repo_id,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                token=args.hub_token,
                allow_patterns=["pytorch_model.bin", "*.json", "*.md"],
            )

    accelerator.end_training()
