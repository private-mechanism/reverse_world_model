# 训练配置说明（Train Config Guide）

本文面向当前 `WAM` 工程，说明如何编写 `model_config/*.yml` 来启动：

- World 训练（`train_world`）
- Video-only 训练（`train_video`，`VideoRunner`）
- DeepSpeed 开关与路径写法

---

## 1. 基本原则

- 推荐使用 `schema_version: 2` 分组格式（可读性更好）。
- 启动时以 `--model_config_path` 指向 yml；训练参数会在代码里由 `model_config` 覆盖到 `args`。
- 结构参数通过 `data.common` / `data.dataset` / `architecture.model` / `training.ema` 合并为 `model_structure`。
- `consumer_dataset_type` 与 `dataset_type` 含义不同：
  - `consumer_dataset_type`: `finetune` / `pretrain`（数据消费分支）
  - `dataset_type`: HDF5 管线类型（如 `128dim_with_value` 使用 `hdf5_dataset_128dim`；其它取值使用通用 `HDF5Dataset`）

更多字段映射可见：`docs/model_config_schema.md`。

---

## 2. 最小可用配置模板（World）

```yaml
schema_version: 2

experiment:
  id: wam-world-exp
  wandb_project: WAM
  config_name: wam-world-exp

data:
  paths:
    - /path/to/hdf5_root
  consumer_dataset_type: finetune
  common:
    num_cameras: 3
    img_history_size: 2
    action_chunk_size: 30
    state_dim: 128
  dataset:
    tokenizer_max_length: 512
    video_size: [256, 448]

architecture:
  model_name: world_policy_causal_value
  causal_world_training: true
  dataset_type: 128dim_with_value
  train_runner_module: runner_world_casual_128dim
  train_runner_class: FMPRunner
  model:
    action_expert:
      in_action_dim: 128
      out_action_dim: 128
      action_dim: 128
      in_visual_dim: 1024
      text_dim: 1024
      hidden_size: 768
    action_noise_scheduler:
      num_train_timesteps: 1000
      num_inference_timesteps: 100
      prediction_type: sample
      type: normal

weights:
  text_encoder: /path/to/WoW-1-Wan-1.3B-2M-Diffusers
  vision_encoder: /path/to/siglip
  action_expert: none
  video_expert: /path/to/video_expert_ckpt_or_none
  # pretrained_model_name_or_path: /path/to/checkpoint  # 可选，恢复整模型

checkpoints:
  root_dir: checkpoints
  run_name: checkpoints/wam-world-exp

training:
  batch:
    train: 4
    sample: 4
  max_train_steps: 20000
  checkpointing_period: 500
  sample_period: 500
  dataloader_num_workers: 4
  gradient_accumulation_steps: 1
  ema:
    update_after_step: 0
    inv_gamma: 1.0
    power: 0.6666667
    min_value: 0.0
    max_value: 0.9999

optimizer:
  learning_rate: 5.0e-6
  lr:
    scheduler: constant
    warmup_steps: 100

runtime:
  mixed_precision: bf16
  load_from_hdf5: true
  gradient_checkpointing: false
  image_aug: true

logging:
  report_to: wandb

distributed:
  deepspeed: off
```

启动：

```bash
cd /mnt/dataset/projs/projects/WAM
accelerate launch main_world.py --model_config_path model_config/your_world.yml
```

---

## 3. 最小可用配置模板（Video-only）

与 World 的差异重点：

- `train_runner_module` 需为：
  - `runner_video`（WoW/Wan1.3B 逻辑）
  - 或 `runner_video_wan2_2`（Wan2.2 逻辑）
- 若 `weights.video_base_model`（扁平键 `VIDEO_BASE_MODEL`）为 **Wan2.2**（注册键或与其相同的绝对路径），`train_video` 会**自动**将 `train_runner_module` 设为 `runner_video_wan2_2`，无需手改 `runtime`（仍会打一条 info 日志）。
- `train_runner_class` 固定 `VideoRunner`
- `main_video.py` 作为入口
- **仅视频训练不要求** `dataset_type: 128dim_with_value`：`128dim` / `128dim_with_value` 会走 `hdf5_dataset_128dim`，其它取值走通用 `HDF5Dataset`（示例：`model_config/smoke-video-wow.yml` 使用 `robotwin`）。
- v2 下 `train_runner_module` / `train_runner_class` 写在 **`runtime`** 块（由 loader 展开到扁平 `model_config`）。
- 中间采样 CFG：在 **`runtime.video_sample_guidance`**（扁平键 `video_sample_guidance`）或命令行 **`--video_sample_guidance`** 指定；与 `1.0` 接近时关闭 CFG，默认 `5.0`（与历史硬编码一致）。

```yaml
schema_version: 2

experiment:
  id: wam-video-exp
  wandb_project: WAM-Video
  config_name: wam-video-exp

data:
  paths:
    - /path/to/hdf5_root
  consumer_dataset_type: finetune
  common:
    num_cameras: 3
    img_history_size: 2
    action_chunk_size: 30
    state_dim: 128
  dataset:
    tokenizer_max_length: 512
    video_size: [256, 448]

architecture:
  dataset_type: robotwin
  model:
    noise_scheduler:
      num_train_timesteps: 1000
      num_inference_timesteps: 50
      prediction_type: sample
      type: normal

weights:
  text_encoder: /path/to/WoW-1-Wan-1.3B-2M-Diffusers
  vision_encoder: /path/to/siglip
  video_expert: /path/to/video_expert_checkpoint_or_none

checkpoints:
  root_dir: checkpoints
  run_name: checkpoints/wam-video-exp

training:
  batch:
    train: 2
    sample: 2
  max_train_steps: 10000
  checkpointing_period: 500
  sample_period: 500
  dataloader_num_workers: 4
  gradient_accumulation_steps: 1

optimizer:
  learning_rate: 5.0e-6
  lr:
    scheduler: constant
    warmup_steps: 100

runtime:
  mixed_precision: bf16
  load_from_hdf5: true
  image_aug: true
  train_runner_module: runner_video
  train_runner_class: VideoRunner

logging:
  report_to: wandb

distributed:
  deepspeed: off
```

启动：

```bash
cd /mnt/dataset/projs/projects/WAM
accelerate launch main_video.py --model_config_path model_config/your_video.yml
```

---

## 4. DeepSpeed 配置方式

`distributed.deepspeed` 支持以下值：

- `off` / `false` / `none` / 空：关闭
- `zero0` / `zero1` / `zero2`：自动映射到 `configs/zero0.json` / `configs/zero1.json` / `configs/zero2.json`
- 自定义 json 路径：如 `configs/my_ds.json` 或绝对路径

示例：

```yaml
distributed:
  deepspeed: zero2
```

或：

```yaml
distributed:
  deepspeed: configs/my_ds.json
```

> 注意：如果写 `zero2`，需要仓库内存在 `configs/zero2.json`。

---

## 5. 常见问题

- `train_video` 报 runner 错误  
  检查 `train_runner_module` 是否包含 `runner_video`，且 `train_runner_class: VideoRunner`。

- 设置了 `--deepspeed` 但不生效  
  当前训练逻辑会按 `model_config` 再覆盖参数；建议把 deepspeed 配置写进 yml 的 `distributed.deepspeed`。

- `consumer_dataset_type` 和 `dataset_type` 混淆  
  前者是 `finetune/pretrain` 分支，后者是 HDF5 管线类型（如 `128dim_with_value` 走 value_128dim；其它字符串如 `robotwin` 走通用 `HDF5Dataset`，与「仅测 video」无必然绑定）。

- Video runner 基座路径不对  
  可通过环境变量覆盖：
  - `WAM_WOW_VIDEO_BASE`（`runner_video`）
  - `WAM_WAN22_VIDEO_BASE`（`runner_video_wan2_2`）

