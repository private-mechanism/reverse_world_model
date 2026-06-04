"""
model_config yml 加载与 schema_version 2 分组格式 → 扁平 dict 的兼容层。

训练与数据代码仍使用「一层 dict + 历史键名」；分组 yml 仅在加载时展开。
结构配置（``common`` / ``dataset`` / ``model``）在扁平结果中置于 ``model_structure``，避免与 ``experiment.id`` → 扁平键 ``model``（字符串实验 id）冲突。
"""

from __future__ import annotations

import copy
from typing import Any, Dict

import yaml


def _ensure_model_token_dims(struct: Dict[str, Any]) -> None:
    """若 yml 省略 ``lang_token_dim`` / ``img_token_dim`` / ``state_token_dim``，从 ``action_expert`` 与 ``common`` 推断。"""
    model = struct.get("model")
    if not isinstance(model, dict):
        return
    ae = model.get("action_expert")
    ae = ae if isinstance(ae, dict) else {}
    common = struct.get("common")
    common = common if isinstance(common, dict) else {}
    if model.get("state_token_dim") is None:
        sd = common.get("state_dim")
        if sd is None:
            sd = ae.get("out_action_dim")
        if sd is not None:
            model["state_token_dim"] = sd
    if model.get("lang_token_dim") is None and ae.get("text_dim") is not None:
        model["lang_token_dim"] = ae["text_dim"]
    if model.get("img_token_dim") is None and ae.get("in_visual_dim") is not None:
        model["img_token_dim"] = ae["in_visual_dim"]


def _normalize_model_structure_keys(struct: Dict[str, Any]) -> None:
    """``model.action_noise_scheduler`` → ``model.noise_scheduler``（FMPRunner 仍读后者）。"""
    model = struct.get("model")
    if not isinstance(model, dict):
        return
    if "action_noise_scheduler" in model:
        if "noise_scheduler" not in model:
            model["noise_scheduler"] = model.pop("action_noise_scheduler")
        else:
            del model["action_noise_scheduler"]


def assemble_model_structure(raw: Dict[str, Any], arch_partial: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 ``architecture`` 内联块（旧式整块或仅 ``model``）与 ``data.common`` / ``data.dataset`` / ``training.ema``
    合并为 ``{common, dataset, model}``。
    """
    out = copy.deepcopy(arch_partial)
    data = raw.get("data") or {}
    tr = raw.get("training") or {}

    if isinstance(data.get("common"), dict):
        out["common"] = copy.deepcopy(data["common"])
    if isinstance(data.get("dataset"), dict):
        out["dataset"] = copy.deepcopy(data["dataset"])

    if not isinstance(out.get("model"), dict):
        out["model"] = {}
    _normalize_model_structure_keys(out)

    if isinstance(tr.get("ema"), dict):
        out["model"]["ema"] = copy.deepcopy(tr["ema"])

    _ensure_model_token_dims(out)
    return out


def _model_structure_nonempty(w: Dict[str, Any]) -> bool:
    return bool(w.get("model")) or bool(w.get("common")) or bool(w.get("dataset"))


_V2_SECTION_KEYS = frozenset(
    {
        "schema_version",
        "experiment",
        "data",
        "architecture",
        "checkpoints",
        "weights",
        "training",
        "optimizer",
        "logging",
        "runtime",
        "distributed",
        "inference",
    }
)


def flatten_model_config_v2(raw: Dict[str, Any]) -> Dict[str, Any]:
    """将 schema_version>=2 的分组配置展开为与 v1 扁平 yml 等价的 dict。"""
    out: Dict[str, Any] = {}

    exp = raw.get("experiment") or {}
    data = raw.get("data") or {}
    data_hdf5 = data.get("hdf5") or {}
    arch = raw.get("architecture") or {}
    ckpt = raw.get("checkpoints") or {}
    w = raw.get("weights") or {}
    tr = raw.get("training") or {}
    tr_batch = tr.get("batch") or {}
    opt = raw.get("optimizer") or {}
    lr = opt.get("lr") or {}
    log = raw.get("logging") or {}
    rt = raw.get("runtime") or {}
    dist = raw.get("distributed") or {}
    inf = raw.get("inference") or {}

    if exp.get("id") is not None:
        out["model"] = exp["id"]
    if exp.get("wandb_project") is not None:
        out["WANB_PROJECT_NAME"] = exp["wandb_project"]
    if exp.get("cuda_visible_devices") is not None:
        out["cuda_visible_device"] = exp["cuda_visible_devices"]
    if exp.get("config_name") is not None:
        out["CONFIG_NAME"] = exp["config_name"]

    paths = data.get("paths")
    if paths is not None:
        out["data_paths"] = paths
    if data.get("single_path") is not None:
        out["data_path"] = data["single_path"]
    if data.get("consumer_dataset_type") is not None:
        out["consumer_dataset_type"] = data["consumer_dataset_type"]

    for k in ("use_prompt_template", "global_stats_path", "control_freqs_path", "target_control_freq"):
        if k in data_hdf5 and data_hdf5[k] is not None:
            out[k] = data_hdf5[k]

    if arch.get("model") is not None and isinstance(arch["model"], dict):
        out["model_structure"] = assemble_model_structure(raw, {"model": arch["model"]})

    if arch.get("model_type") is not None:
        out["model_type"] = arch["model_type"]
    if arch.get("model_name") is not None:
        out["model_name"] = str(arch["model_name"])
    if arch.get("causal_world_training") is not None:
        out["causal_world_training"] = arch["causal_world_training"]
    if arch.get("dataset_type") is not None:
        out["dataset_type"] = arch["dataset_type"]
    if arch.get("image_aug_type") is not None:
        out["image_aug_type"] = arch["image_aug_type"]
    if arch.get("video_variant") is not None:
        out["video_variant"] = arch["video_variant"]

    if ckpt.get("run_name") is not None:
        out["checkpoint_path"] = ckpt["run_name"]
    if ckpt.get("root_dir") is not None:
        out["checkpoint_root_dir"] = ckpt["root_dir"]

    if w.get("pretrained_model_name_or_path") is not None:
        out["pretrained_model_name_or_path"] = w["pretrained_model_name_or_path"]
    if w.get("text_encoder") is not None:
        out["pretrained_text_encoder_name_or_path"] = w["text_encoder"]
    if w.get("qwen3_embed") is not None:
        out["pretrained_qwen3_embed_path"] = w["qwen3_embed"]
    if w.get("qwen25_embed") is not None:
        out["pretrained_qwen25_embed_path"] = w["qwen25_embed"]
    if w.get("vision_encoder") is not None:
        out["pretrained_vision_encoder_name_or_path"] = w["vision_encoder"]
    if w.get("wam") is not None:
        out["pretrained_wam_path"] = w["wam"]
    if w.get("action_expert") is not None:
        out["pretrained_action_expert_path"] = w["action_expert"]
    if w.get("video_expert") is not None:
        out["pretrained_video_expert_path"] = w["video_expert"]
    if w.get("video_base_model") is not None:
        out["VIDEO_BASE_MODEL"] = w["video_base_model"]

    # 将 weights 中的预训练路径写入 ``model_structure.model``，Runner 可从 ``config_.model`` 读取
    if isinstance(out.get("model_structure"), dict):
        mroot = out["model_structure"].get("model")
        if isinstance(mroot, dict):
            if w.get("wam") is not None:
                mroot["pretrained_wam_path"] = w["wam"]
            if w.get("action_expert") is not None:
                mroot["pretrained_action_expert_path"] = w["action_expert"]
            if w.get("video_expert") is not None:
                mroot["pretrained_video_expert_path"] = w["video_expert"]
            if w.get("video_base_model") is not None:
                mroot["VIDEO_BASE_MODEL"] = w["video_base_model"]
            for k in (
                "stage2_enable_reverse_ar_action",
                "stage2_key_action_loss_weight",
                "stage2_reverse_ar_action_loss_weight",
                "generation_mode",
                "stage3_predict_keyframe",
                "stage3_predict_key_action",
                "stage3_replace_video_diffusion",
                "stage3_replace_action_diffusion",
                "stage3_video_ar_causal_attention",
                "stage3_action_ar_causal_attention",
                "stage3_keyframe_loss_weight",
                "stage3_video_ar_loss_weight",
                "stage3_key_action_loss_weight",
                "stage3_action_ar_loss_weight",
                "stage3_use_scheduled_sampling",
                "stage3_key_teacher_forcing_prob_start",
                "stage3_key_teacher_forcing_prob_end",
                "stage3_key_teacher_forcing_decay_steps",
            ):
                if tr.get(k) is not None:
                    mroot[k] = tr[k]

    if tr_batch.get("train") is not None:
        out["train_batch_size"] = tr_batch["train"]
    if tr_batch.get("sample") is not None:
        out["sample_batch_size"] = tr_batch["sample"]
    if tr.get("vae_mini_batch") is not None:
        out["vae_mini_batch"] = tr["vae_mini_batch"]
    for k_flat, k_nested in (
        ("max_train_steps", "max_train_steps"),
        ("checkpointing_period", "checkpointing_period"),
        ("sample_period", "sample_period"),
        ("sample_light", "sample_light"),
        ("sample_joint_only", "sample_joint_only"),
        ("sample_save_video", "sample_save_video"),
        ("sample_compute_video_metrics", "sample_compute_video_metrics"),
        ("sample_sharded_inference", "sample_sharded_inference"),
        ("reverse_world_order", "reverse_world_order"),
        ("stage2_enable_reverse_ar_action", "stage2_enable_reverse_ar_action"),
        ("stage2_key_action_loss_weight", "stage2_key_action_loss_weight"),
        ("stage2_reverse_ar_action_loss_weight", "stage2_reverse_ar_action_loss_weight"),
        ("generation_mode", "generation_mode"),
        ("stage3_predict_keyframe", "stage3_predict_keyframe"),
        ("stage3_predict_key_action", "stage3_predict_key_action"),
        ("stage3_replace_video_diffusion", "stage3_replace_video_diffusion"),
        ("stage3_replace_action_diffusion", "stage3_replace_action_diffusion"),
        ("stage3_video_ar_causal_attention", "stage3_video_ar_causal_attention"),
        ("stage3_action_ar_causal_attention", "stage3_action_ar_causal_attention"),
        ("stage3_keyframe_loss_weight", "stage3_keyframe_loss_weight"),
        ("stage3_video_ar_loss_weight", "stage3_video_ar_loss_weight"),
        ("stage3_key_action_loss_weight", "stage3_key_action_loss_weight"),
        ("stage3_action_ar_loss_weight", "stage3_action_ar_loss_weight"),
        ("stage3_use_scheduled_sampling", "stage3_use_scheduled_sampling"),
        ("stage3_key_teacher_forcing_prob_start", "stage3_key_teacher_forcing_prob_start"),
        ("stage3_key_teacher_forcing_prob_end", "stage3_key_teacher_forcing_prob_end"),
        ("stage3_key_teacher_forcing_decay_steps", "stage3_key_teacher_forcing_decay_steps"),
        ("checkpoints_total_limit", "checkpoints_total_limit"),
        ("dataloader_num_workers", "dataloader_num_workers"),
        ("gradient_accumulation_steps", "gradient_accumulation_steps"),
        ("state_noise_snr", "state_noise_snr"),
        ("cond_mask_prob", "cond_mask_prob"),
        ("cam_ext_mask_prob", "cam_ext_mask_prob"),
        ("num_train_epochs", "num_train_epochs"),
        ("seed", "seed"),
    ):
        v = tr.get(k_nested)
        if v is not None:
            out[k_flat] = v

    if opt.get("learning_rate") is not None:
        out["learning_rate"] = opt["learning_rate"]
    if lr.get("scheduler") is not None:
        out["lr_scheduler"] = lr["scheduler"]
    if lr.get("warmup_steps") is not None:
        out["lr_warmup_steps"] = lr["warmup_steps"]
    if lr.get("num_cycles") is not None:
        out["lr_num_cycles"] = lr["num_cycles"]
    if lr.get("power") is not None:
        out["lr_power"] = lr["power"]
    if lr.get("hold_ratio") is not None:
        out["lr_hold_ratio"] = lr["hold_ratio"]

    for k_flat, k_nested in (
        ("adam_beta1", "adam_beta1"),
        ("adam_beta2", "adam_beta2"),
        ("adam_weight_decay", "adam_weight_decay"),
        ("adam_epsilon", "adam_epsilon"),
        ("max_grad_norm", "max_grad_norm"),
        ("alpha", "alpha"),
    ):
        v = opt.get(k_nested)
        if v is not None:
            out[k_flat] = v

    if log.get("report_to") is not None:
        out["report_to"] = log["report_to"]
    if log.get("logging_dir") is not None:
        out["logging_dir"] = log["logging_dir"]
    if log.get("resume_from_checkpoint") is not None:
        out["resume_from_checkpoint"] = log["resume_from_checkpoint"]

    for k in (
        "load_from_hdf5",
        "gradient_checkpointing",
        "image_aug",
        "precomp_lang_embed",
        "precomp_lang_embed_prefix",
        "scale_lr",
        "allow_tf32",
        "use_8bit_adam",
        "set_grads_to_none",
        "push_to_hub",
    ):
        if k in rt and rt[k] is not None:
            out[k] = rt[k]

    if rt.get("mixed_precision") is not None:
        out["mixed_precision"] = rt["mixed_precision"]
    if rt.get("train_runner_module") is not None:
        out["train_runner_module"] = rt["train_runner_module"]
    if rt.get("train_runner_class") is not None:
        out["train_runner_class"] = rt["train_runner_class"]
    if rt.get("video_sample_guidance") is not None:
        out["video_sample_guidance"] = rt["video_sample_guidance"]

    if dist.get("deepspeed") is not None:
        out["deepspeed"] = dist["deepspeed"]

    for k, v in (inf or {}).items():
        if v is not None:
            out[f"inference_{k}"] = v

    return out


def load_model_config_dict(path: str) -> Dict[str, Any]:
    """
    读取 model_config yml，若为 schema_version>=2 则先展开再返回。

    顶层若存在 v2 分组之外的「补充扁平键」，会覆盖同名的展开结果（便于局部覆盖）。
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raw = {}

    version = raw.get("schema_version", 1)
    try:
        version_int = int(version)
    except (TypeError, ValueError):
        version_int = 1

    if version_int < 2:
        return dict(raw)

    flat = flatten_model_config_v2(raw)
    for k, v in raw.items():
        if k in _V2_SECTION_KEYS:
            continue
        if v is not None:
            flat[k] = v
    return flat


def load_model_structure_dict(model_config: Dict[str, Any]) -> Dict[str, Any]:
    """返回 ``{common, dataset, model}``：来自扁平键 ``model_structure``（由 ``architecture.model`` 与 ``data``/``training`` 展开得到）。"""
    ms = model_config.get("model_structure")
    if isinstance(ms, dict) and _model_structure_nonempty(ms):
        out = copy.deepcopy(ms)
        _normalize_model_structure_keys(out)
        _ensure_model_token_dims(out)
        return out
    raise ValueError(
        "model_config 缺少结构配置：请在 v2 的 ``architecture.model`` 与 ``data``/``training`` 中提供配置，"
        "以便展开为扁平键 ``model_structure``（含 ``common``/``dataset``/``model``）。"
    )
