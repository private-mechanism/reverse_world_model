# WAM 精简训练目录说明

本目录是从 `RoboTwin/policy/RDT` 按「World 因果 + 128 维状态 + value」训练链路精简拷贝的**最小可运行子集**（工作目录应设为 **`WAM` 根目录**，与原先 `policy/RDT` 下相对路径约定一致）。

---

## 1. 目录结构

```
WAM/
├── train_act_world_aug.sh        # 与 RDT 对齐的启动脚本（--image_aug，需多机环境变量）
├── scripts/
│   └── read_yaml.py              # 供 shell 从 yml 读字段
├── main_world.py                 # 训练入口（命令行参数）
├── model_config/
│   └── 0414-causal-world-d768-n30-c32-s4-value-nobad--128dims.yml
├── configs/
│   ├── base_world_flow_sample_d768_n30_c32_s4_rope--128dim--value.yaml
│   ├── dataset_control_freq.json
│   ├── dataset_stat.json
│   ├── finetune_datasets.json
│   └── state_vec.py              # 状态向量维度与 HDF5 字段映射
├── wam/                          # Python 包根：models / trainers / runners / samples
│   ├── trainers/                 # Accelerate、TrainModule、Dataset、增广
│   ├── runners/                  # FMPRunner
│   ├── samples/                  # 训练期采样与指标、noise_scheduler
│   └── models/
│       ├── hub_mixin.py
│       ├── ema_model.py
│       ├── action_expert/
│       │   └── action_expert.py
│       ├── video_expert/
│       │   └── video_expert.py
│       ├── mot/
│       │   ├── world_policy_causal_value.py
│       │   └── world_stack_registry.py
│       └── multimodal_encoder/
│           ├── siglip2_encoder.py
│           ├── umt5_encoder.py
│           ├── t5_encoder.py
│           ├── vae_encoder.py
│           └── debug_visual/
│               └── vae_utils.py
├── data/
│   ├── filelock.py
│   ├── dataset.py
│   ├── image_corrupt.py
│   ├── video_augment.py
│   ├── hdf5_dataset.py
│   ├── hdf5_dataset_128dim.py
│   └── utils/
│       ├── configs.py
│       └── utils_datasets.py
└── docs/
    └── README.md
```

---

## 2. 各 Python 文件用途

| 路径 | 用途 |
|------|------|
| `main_world.py` | 解析 `accelerate launch` / 命令行参数，调用 `wam.trainers.train_world.train` 启动训练。 |
| `configs/state_vec.py` | 定义 `STATE_VEC_IDX_MAPPING`：状态/动作各维与语义名称对应，供 HDF5 解析与 mask 使用。 |
| `wam/trainers/train_world.py` | World 训练主逻辑：读 yml、构建编码器与 `FMPRunner`、DataLoader、优化器与调度器；训练步调用 `TrainModule.training_step`；反向与 checkpoint / 采样日志。 |
| `wam/trainers/world_train_module.py` | `TrainModule`：单步内完成 batch 拆包、VAE/SigLIP/文本编码、`FMPRunner` forward 与 `loss_a+loss_v`。 |
| `data/dataset.py` | `VLAConsumerDataset`：支持 buffer 消费或 HDF5；`DataCollatorForVLAConsumerDataset` 组 batch；读取 `configs/*.json` 与 `model_config` 控制数据集名、控制频率等。 |
| `wam/samples/sample_world.py` | `log_sample_res`：固定间隔在样本 batch 上做推理，算误差类指标并可选保存对比视频；内含 `value_to_value_frame` 供 value 与 `pred_video` 空间对齐后算 L1。 |
| `wam/samples/noise_scheduler.py` | `NoiseTimestepSampler`：统一 Beta/正态/均匀等训练时间步分布，供 `FMPRunner` 训练加噪使用。 |
| `data/image_corrupt.py` | 基于 imgaug 的噪声、模糊等 corrupt，供 `data.dataset` 在开启图像增强时使用。 |
| `data/video_augment.py` | 对整段视频 clip 做时间一致的 ColorJitter 或弱 corrupt，避免逐帧闪烁。 |
| `data/filelock.py` | 基于 `fcntl` 的文件锁，用于多进程安全读写 buffer 分块中的 `json`/`npz`/`dirty_bit`。 |
| `data/hdf5_dataset.py` | 从 yml 的 `data_paths` 收集 HDF5  episode、归一化与采样逻辑的**基类**；由 `data.dataset` import，与具体 128dim 子类并存。 |
| `data/hdf5_dataset_128dim.py` | 针对 **128 维状态/动作 + 标量 value** 的 HDF5 数据集实现（与 `dataset_type: 128dim_with_value` 配合）。 |
| `data/utils/configs.py` | `DatasetConfig` 与各数据源（如 robotwin、agibot）的 HDF5 键、臂维度、相机键等静态配置表。 |
| `data/utils/utils_datasets.py` | `detect_dataset_type`、`get_action_array`、`get_instruction` 等：按路径/类型从 HDF5 取动作、相机、指令。 |
| `wam/models/hub_mixin.py` | 为 runner 提供与 Transformers/Diffusers 类似的 `save_pretrained`、配置序列化接口。 |
| `wam/models/ema_model.py` | 对可训练模块做 EMA 更新（训练中 `ema_model.step`）。 |
| `wam/runners/runner_world_casual_128dim.py` | **Runner**：因果 World 的 `FMPRunner`——组装 `ActionExpert`、预训练 `WanTransformer3D`、`WorldPolicyModel`、噪声采样器与 `predict`/`forward` 训练损失。 |
| `wam/models/mot/world_policy_causal_value.py` | 因果注意力下的 World 策略：融合语言、图像、状态、动作与视频 latent，并支持 value 相关分支。 |
| `wam/models/action_expert/action_expert.py` | 动作 chunk 上的 DiT/Transformer 结构（与 diffusers 组件集成）。 |
| `wam/models/video_expert/video_expert.py` | Wan 3D Transformer：视频 latent 扩散主干（与预训练目录 `from_pretrained` 加载）。 |
| `wam/models/multimodal_encoder/siglip2_encoder.py` | `SiglipVisionTower`：加载 SigLIP2，输出 patch 级视觉 token。 |
| `wam/models/multimodal_encoder/umt5_encoder.py` | `umT5Embedder`：UMT5 tokenizer + encoder，输出语言 hidden states。 |
| `wam/models/multimodal_encoder/t5_encoder.py` | 标准 T5 封装；`train_world.py` 仍 import `T5Embedder`，保留以满足 import，训练主路径使用 umT5。 |
| `wam/models/multimodal_encoder/vae_encoder.py` | `VAEEncoder`：封装 `AutoencoderKLWan`，`encode_to_latents` / `get_condition` 供训练视频条件。 |
| `wam/models/multimodal_encoder/debug_visual/vae_utils.py` | Latent 可视化函数；被 `vae_encoder.py` 在模块加载时 import（不调用也依赖 matplotlib 等）。 |

---

## 3. 非 Python 配置文件（简要）

| 路径 | 用途 |
|------|------|
| `model_config/*.yml` | 建议使用 **`schema_version: 2`** 分组格式（见 **`docs/model_config_schema.md`**）；加载时由 `configs/model_config_loader.py` 展开为扁平 dict。 |
| `configs/base_world_flow_sample_*.yaml` | 模型维度、相机数、chunk、噪声与 EMA 等**结构级**超参。 |
| `configs/dataset_control_freq.json` | 各数据集名称 → 控制频率（Hz），写入 batch 供模型使用。 |
| `configs/finetune_datasets.json` | 微调阶段数据集名称列表，与 `dataset_name` / `data_idx` 对齐。 |
| `configs/dataset_stat.json` | 各数据集归一化统计（mean/std 等），与 HDF5 中样本字段一致才可稳定训练。 |

---

## 4. 环境与运行提示

- 推荐使用 **`conda activate videogen`** 等与原 RDT 训练一致的环境（需 `torch`、`diffusers`、`accelerate`、`transformers`、`omegaconf`、`h5py`、`imgaug`、`yaml` 等）。
- 在 **`WAM` 根目录** 下执行：`python -c "from wam.trainers.train_world import train; print('import ok')"` 可做快速校验。
- 分布式训练：`./train_act_world_aug.sh <CONFIG_NAME>`。脚本只传 `--model_config_path`；**batch、学习率、编码器路径、`--image_aug` / `--load_from_hdf5` / `mixed_precision` / `report_to` / DeepSpeed 等**均在 **`model_config` 的 yml** 中由 `train_world._apply_training_args_from_model_config` 写入 `args`（见示例 `model_config/0414-causal-world-...128dims.yml` 底部注释块）。需已配置 `accelerate`，多机时由调度系统提供 `WORLD_SIZE`、`NPROC_PER_NODE`、`RANK`、`MASTER_ADDR`、`MASTER_PORT`。
- **`consumer_dataset_type`**：写入 `args.dataset_type`，决定使用 `finetune_datasets.json` 还是 `pretrain_datasets.json`；**勿与** yml 里的 **`dataset_type`**（如 `128dim_with_value`，给 HDF5 子类用）混淆。若 yml 未写 `consumer_dataset_type`，默认 **`finetune`**。
- 权重、HDF5 数据、SigLIP/ WoW 等路径主要在 **model_config yml** 中配置，本目录不包含大文件。
- 训练配置速查与可复制模板见：`docs/train_config_guide.md`（含 world/video/deepspeed）。
- conda 环境准备与自检见：`docs/env.md`。

---

### model_config 与训练 `args`

- **v2 分组 yml**：字段含义与展开映射见 **`docs/model_config_schema.md`**。
- **扁平键 → `args`**：展开后的键仍由 `wam/trainers/train_world.py` 中 `_apply_training_args_from_model_config` 写入 `args`（与旧版扁平 yml 相同）。

---

*文档随 `WAM` 精简拷贝内容整理；若上游 `RDT` 有变更，请同步核对 import 与配置字段。*
