# Model Config 规范（`schema_version: 2`）

WAM 中 `model_config/*.yml` 用于同时描述**数据**、**权重路径**、**训练超参**与**日志/分布式**等。为便于维护，采用 **分组嵌套** 书写；运行时通过 `configs/model_config_loader.load_model_config_dict()` **展开为与历史 v1 兼容的一层 dict**，`train_world`、`dataset`、`HDF5Dataset` 等无需改键名即可使用。

结构配置（`common` / `dataset` / `model`）在扁平 dict 中置于 **`model_structure`**，由 `load_model_structure_dict()` 读取；**不再使用**扁平键 `base_config`。内容仅来自 v2 分组内的 **`data`** / **`architecture.model`** / **`training.ema`** 的合并，**不支持**再指向外部 yaml 路径。

---

## 1. 版本与加载行为

| 字段 | 说明 |
|------|------|
| `schema_version` | **整数**。缺省或 `1`：整份 yml 视为 v1 **扁平**格式，不做展开。`>=2`：按本文分组规则展开后再合并顶层「非分组」补充键。 |

**顶层补充键（v2）**：与分组键并列的扁平键（如临时调试开关）会在展开后 **覆盖** 同名的展开结果，便于局部覆盖而不改深层结构。

**实现位置**：`configs/model_config_loader.py`（`load_model_config_dict`、`flatten_model_config_v2`）。

---

## 2. 分组总览

| 分组 | 用途 |
|------|------|
| `experiment` | 实验 id、W&B 工程名、可见 GPU 列表占位、可选 run 名 |
| `data` | HDF5 目录列表、`consumer_dataset_type`、HDF5 读取相关可选项 |
| `architecture` | `model`、World 模型类型、因果开关、**HDF5 布局** `dataset_type`、增强类型 |
| `checkpoints` | 本次 run 目录名、checkpoint 根目录 |
| `weights` | RDT / 文本+VAEViT 路径 / SigLIP / action & video expert |
| `training` | batch、步数、周期、workers、噪声与 mask 等 |
| `optimizer` | 学习率及 `lr` 子块（调度器参数） |
| `logging` | `report_to`、`resume_from_checkpoint`、`logging_dir` |
| `runtime` | 混合精度、HDF5、梯度检查点、图像增强等写入 `args` 的布尔/字符串 |
| `distributed` | DeepSpeed 策略或 json 路径 |
| `inference` | **预留**：展开为扁平键 `inference_<子键>`，供后续推理/评测脚本使用，当前训练主流程不读 |

---

## 3. 各分组字段说明

### 3.1 `experiment`

| 键 | 展开为（扁平键） | 说明 |
|----|------------------|------|
| `id` | `model` | 实验/模型标识字符串 |
| `wandb_project` | `WANB_PROJECT_NAME` | `accelerator.init_trackers` 使用的工程名 |
| `cuda_visible_devices` | `cuda_visible_device` | 记录在 yml 中；是否 `export CUDA_VISIBLE_DEVICES` 仍由启动脚本决定 |
| `config_name` | `CONFIG_NAME` | 可选；不设则由 `train_world` 用 **yml 文件名（无扩展名）** 填充 |

### 3.2 `data`

| 键 | 展开为 | 说明 |
|----|--------|------|
| `paths` | `data_paths` | HDF5 根目录列表 |
| `single_path` | `data_path` | 单目录（与 `paths` 二选一） |
| `consumer_dataset_type` | `consumer_dataset_type` | 写入 `args.dataset_type`：`finetune` / `pretrain`，对应 `finetune_datasets.json` 等。**勿与** `architecture.dataset_type`（如 `128dim_with_value`）混淆 |
| `hdf5.use_prompt_template` | `use_prompt_template` | 布尔，可选 |
| `hdf5.global_stats_path` | `global_stats_path` | 可选 |
| `hdf5.control_freqs_path` | `control_freqs_path` | 可选 |
| `hdf5.target_control_freq` | `target_control_freq` | 可选，默认由数据集代码处理 |
| `common` | （并入 `model_structure`） | **可选**：chunk、相机数、state 维等；展开后进入 `model_structure["common"]` |
| `dataset` | （并入 `model_structure`） | **可选**：buffer、分辨率、`tokenizer_max_length` 等；进入 `model_structure["dataset"]` |

若 v2 中未写 `consumer_dataset_type`，`train_world._apply_training_args_from_model_config` 仍将 `args.dataset_type` 默认设为 `finetune`。

展开时由 `assemble_model_structure` 合并：`data.common` / `data.dataset` / `training.ema` / `architecture.model` → 扁平 **`model_structure`**（`{common, dataset, model}`）。**勿与** `experiment.id` 展开得到的扁平键 `model`（**字符串**实验 id）混淆。

### 3.3 `architecture`

| 键 | 展开为 | 说明 |
|----|--------|------|
| `model` | `model_structure` | **可选**：`action_expert`、`action_noise_scheduler` 等；与 `data` / `training.ema` 合并为扁平 **`model_structure`**。`architecture.model.action_noise_scheduler` → 运行时的 `model_structure["model"]["noise_scheduler"]`。省略 `lang_token_dim` / `img_token_dim` / `state_token_dim` 时由 `action_expert` 与 `data.common` 推断 |
| `model_name` | `model_name` | **推荐**：与 `wam/models/mot/<model_name>.py` 模块名一致，该模块需提供 `build_world_stack`；由 `world_stack_registry` 动态加载 |
| `model_type` | `model_type` | **兼容旧版**：如 `world_casual_value`；未写 `model_name` 时会映射到 `model_name` |
| `causal_world_training` | `causal_world_training` | 布尔 |
| `dataset_type` | `dataset_type` | **HDF5 管线**类型，如 `128dim_with_value` |
| `image_aug_type` | `image_aug_type` | `mixed` / `color_only` / `corrput_only` / `both` |

**保存 checkpoint**：将合并后的结构写入 `output_dir/configs/model_structure/model_structure.yaml`。

### 3.4 `checkpoints`

| 键 | 展开为 | 说明 |
|----|--------|------|
| `run_name` | `checkpoint_path` | 输出子目录名（可与旧字段 `checkpoint_path` 同义） |
| `root_dir` | `checkpoint_root_dir` | 与 `run_name` 拼接得到完整 `output_dir` |

### 3.5 `weights`

| 键 | 展开为 | 说明 |
|----|--------|------|
| `pretrained_model_name_or_path` | `pretrained_model_name_or_path` | 整 Runner 初始化：本地目录、单文件权重，或 Hugging Face 模型 id |
| `text_encoder` | `pretrained_text_encoder_name_or_path` | 一般为 WoW 目录（含 tokenizer + VAE 子目录约定） |
| `vision_encoder` | `pretrained_vision_encoder_name_or_path` | SigLIP2 本地目录 |
| `action_expert` | `pretrained_action_expert_path` | 可为字符串 `none` |
| `video_expert` | `pretrained_video_expert_path` | Video expert 微调 checkpoint 等 |
| `video_base_model` | `VIDEO_BASE_MODEL` | 因果 World 中 Wan 底座 Diffusers 根目录：键名见 `wam.models.mvwam.world_stack_registry.VIDEO_BASE_MODEL_PATHS`；亦可写绝对路径。写入扁平 dict 并合并进 `model_structure.model`，供 `FMPRunner` 解析 |

### 3.6 `training`

| 键 | 展开为 | 说明 |
|----|--------|------|
| `batch.train` | `train_batch_size` | 每卡训练 batch |
| `batch.sample` | `sample_batch_size` | 采样 batch |
| `vae_mini_batch` | `vae_mini_batch` | 可选；VAE 编码微批 |
| `max_train_steps` | `max_train_steps` | 总优化步数 |
| `checkpointing_period` | `checkpointing_period` | 存盘周期（步） |
| `sample_period` | `sample_period` | 采样日志周期 |
| `checkpoints_total_limit` | `checkpoints_total_limit` | 最多保留 checkpoint 个数 |
| `dataloader_num_workers` | `dataloader_num_workers` | DataLoader workers |
| `gradient_accumulation_steps` | `gradient_accumulation_steps` | 梯度累积 |
| `state_noise_snr` / `cond_mask_prob` / `cam_ext_mask_prob` | 同名 | 与训练管线一致 |
| `num_train_epochs` / `seed` | 同名 | 可选 |
| `ema` | （不单独展开） | **可选**：`EMAModel` 参数；展开合并进结构配置 `model.ema`（与旧 base yaml 中 `model.ema` 等价） |

### 3.7 `optimizer`

| 键 | 展开为 | 说明 |
|----|--------|------|
| `learning_rate` | `learning_rate` | 初始学习率 |
| `lr.scheduler` | `lr_scheduler` | 含 `custom_constant_then_decay` 等 |
| `lr.warmup_steps` | `lr_warmup_steps` | |
| `lr.num_cycles` | `lr_num_cycles` | |
| `lr.power` | `lr_power` | |
| `lr.hold_ratio` | `lr_hold_ratio` | `custom_constant_then_decay` 用 |
| `adam_*`、`max_grad_norm`、`alpha` | 同名 | 可选，与 `train_world._apply_training_args_from_model_config` 一致 |

### 3.8 `logging`

| 键 | 展开为 | 说明 |
|----|--------|------|
| `report_to` | `report_to` | `wandb` / `tensorboard` 等 |
| `resume_from_checkpoint` | `resume_from_checkpoint` | 如 `latest` |
| `logging_dir` | `logging_dir` | 可选 |

### 3.9 `runtime`

布尔与字符串会写入 `args`（见 `train_world._apply_training_args_from_model_config`），常见键：

| 键 | 展开为 |
|----|--------|
| `mixed_precision` | `mixed_precision` |
| `train_runner_module` / `train_runner_class` | 同名 | 仅视频训练等动态 import Runner |
| `video_sample_guidance` | `video_sample_guidance` | 仅视频中间采样：`VideoRunner` CFG 强度；≈`1.0` 关闭 CFG |
| `load_from_hdf5` | `load_from_hdf5` |
| `gradient_checkpointing` | `gradient_checkpointing` |
| `image_aug` | `image_aug` |
| `precomp_lang_embed`、`scale_lr`、`allow_tf32`、`use_8bit_adam`、`set_grads_to_none`、`push_to_hub` | 同名 |

### 3.10 `distributed`

| 键 | 展开为 | 说明 |
|----|--------|------|
| `deepspeed` | `deepspeed` | `off` / `zero0` / `zero1` / `zero2` 或 json 路径；`train_world._resolve_deepspeed_plugin_path` 解析 |

### 3.11 `inference`（预留）

子键 `k: v` 展开为 **`inference_k: v`**，便于与训练扁平键区分。可在后续 eval/export 脚本中读取，**当前 `train_world` 未使用**。

---

## 4. 与 v1 扁平格式的关系

- 旧 yml **不写** `schema_version` 或写 `schema_version: 1`：行为与从前一致，仍为单层键。
- 新 yml 使用 **`schema_version: 2`** 并按上表分组；团队只需维护一份「可读」结构，由 loader 保证下游兼容。

---

## 5. 示例文件

见仓库内：`model_config/0414-causal-world-d768-n30-c32-s4-value-nobad--128dims.yml`。

---

## 6. 校验建议

```bash
cd /path/to/WAM
conda activate videogen  # 或你的训练环境
python -c "
from configs.model_config_loader import load_model_config_dict, load_model_structure_dict
mc = load_model_config_dict('model_config/0414-causal-world-d768-n30-c32-s4-value-nobad--128dims.yml')
assert mc['train_batch_size'] == 16
assert mc['dataset_type'] == '128dim_with_value'
assert mc['consumer_dataset_type'] == 'finetune'
_ = load_model_structure_dict(mc)
print('ok', mc['checkpoint_path'])
"
```

---

## 7. 与 `scripts/read_yaml.py` 的关系

`scripts/read_yaml.py` 按**顶层键**读取 yml。对 **`schema_version: 2`** 的分组文件，其中多数训练字段在嵌套块内，**不能**再用该脚本读 `train_batch_size` 等键；请使用 Python：

`from configs.model_config_loader import load_model_config_dict`。

---

*维护者：修改分组时务必同步更新 `configs/model_config_loader.flatten_model_config_v2`、`assemble_model_structure` 与本页映射表。*
