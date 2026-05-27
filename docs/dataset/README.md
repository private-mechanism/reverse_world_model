# WAM 数据管道说明

本文档说明 WAM 中的数据 pipeline：从 HDF5 文件到训练 batch 的完整链路。

```
docs/
├── README.md
├── dataset/
│   └── README.md          ← 本文
├── train/
└── deploy/
```

---

## 1. 数据流总览

```
HDF5 文件（episode 粒度）
    │
    ▼
HDF5Dataset / HDF5Dataset128dim               ← 解析单条 episode，返回 dict
    │
    ▼
VLAConsumerDataset / VideoDataset             ← 封装图像预处理、文本 tokenize、增强
    │
    ▼
DataCollatorForVLAConsumerDataset             ← 组 batch、padding
    │
    ▼
Trainer（train_world / train_video）
```

- **World 训练**：`HDF5Dataset128dim` → `VLAConsumerDataset` → `DataCollatorForVLAConsumerDataset`
- **Video-only 训练**：`HDF5Dataset` → `VideoDataset` → `DataCollatorForVLAConsumerDataset`（video 侧也使用同一个 collator）

---

## 2. HDF5 文件结构

每个 `.hdf5` 文件对应一条 episode，文件结构约定如下（以 real_world / robotwin 数据为例）：

```
episode_000000.hdf5
├── action                          # (T, action_dim) 动作轨迹
├── prompts                         # 语言指令（多选时为字节串列表，单选为标量）
├── observations/
│   ├── qpos                        # (T, 关节数) 关节位置轨迹
│   ├── left_arm_dim                # 左臂关节数（标量）
│   ├── right_arm_dim               # 右臂关节数（标量）
│   └── images/
│       ├── cam_high                # (T,) 外相机编码帧
│       ├── cam_left_wrist          # (T,) 左腕相机编码帧
│       └── cam_right_wrist         # (T,) 右腕相机编码帧
└── (可选) instructions/
    ├── embed_xxx.pt                # 预计算语言嵌入
    └── ...
```

图像在 HDF5 中以 **JPEG 编码字节** 存储，读取时由 `cv2.imdecode` 解码。

---

## 3. HDF5 数据集实现

### 3.1 `HDF5Dataset`（`data/hdf5_dataset.py`）

通用 HDF5 数据集基类。用于 video-only 训练和非 128dim 数据。

| 方法 | 功能 |
|------|------|
| `__init__(model_config_path)` | 从 model_config 读取 `data_paths`，遍历收集所有 `.hdf5` 文件路径，加载 episode 长度用于加权采样 |
| `get_item(index=None)` | 随机选一条 episode + 随机时间步，解析状态/动作/图像/指令，返回完整 dict |
| `compute_statistics(save_path)` | 遍历全部 episode 计算 state/action 的 min、max、q01、q99、mean、std |

### 3.2 `HDF5Dataset128dim`（`data/hdf5_dataset_128dim.py`）

128 维状态/动作 + value 专用子类。特性：

- 读取 `cam_left` / `cam_right` 而非 `cam_left_wrist` / `cam_right_wrist`（通过 `get_cam_wrist_keys` 按数据集类型动态选择）
- 解析 `action_joint` 键（通过 `get_action_array` 按数据集类型动态选择）
- 将多源数据映射到统一的 128 维状态空间（通过 `STATE_VEC_IDX_MAPPING`）
- 支持 `norm_minmax` 归一化和 per-episode `separate_statistics`
- 返回 value 标签（`action/value` 键）
- 支持 `skip_still_steps` 跳过起始静止帧

### 3.3 `VideoDataset`（`data/video_dataset.py`）

仅视频训练用 Dataset。`train_video` 入口下替代 `VLAConsumerDataset`。

| 差异 | 说明 |
|------|------|
| 数据读取 | 只读 `observations/images` 前向帧 + `prompts`/`instructions`，**不读** `qpos`/`action`（状态/动作置为哑值） |
| 图像解码 | 使用 `detect_dataset_type` 自动检测数据类型，决定相机键名和指令读取方式 |
| 输出 | 输出图像序列 + 语言指令，state/action 用零填充占位 |
| 构造参数 | 与 `VLAConsumerDataset` 对齐（不含 `use_hdf5`/`train_type` 参数） |

---

## 4. 统一 Dataset 封装

### 4.1 `VLAConsumerDataset`（`data/dataset.py`）

World 训练的主 Dataset 类，内部委托给具体的 `HDF5Dataset` 实现。

**构造参数**（来自 model_config + 训练脚本）：

```python
VLAConsumerDataset(
    model_config_path,           # yml 路径
    config,                      # model_structure["dataset"]
    tokenizer,                   # 文本 tokenizer
    image_processor,             # 图像 processor（SigLIP 等）
    num_cameras=3,
    img_history_size=1,
    image_size=None,
    video_size=(384, 320),
    auto_adjust_image_brightness=False,
    image_aug=False,
    image_aug_type="mixed",
    dataset_type="finetune",     # consumer_dataset_type
    cond_mask_prob=0.0,          # 条件 mask 概率
    cam_ext_mask_prob=-1.0,
    state_noise_snr=None,
    use_hdf5=True,
    use_precomp_lang_embed=False,
    train_type=None,
)
```

**HDF5 布局选择**（`dataset_type` vs `consumer_dataset_type`）：

```
model_config 中的 dataset_type（如 128dim_with_value）
    → 决定使用哪个 HDF5 实现类

model_config 中的 consumer_dataset_type（finetune / pretrain）
    → 决定读取 finetune_datasets.json 还是 pretrain_datasets.json
    → 写入 args.dataset_type，控制数据消费分支
```

**__getitem__ 输出**：`data_dict` 包含以下字段：

| 字段 | 来源 | 说明 |
|------|------|------|
| `dataset_name` | HDF5 元数据 | 数据集名称（如 `agilex`） |
| `data_idx` | 名称 → id 映射 | 用于 dataset id 条件 |
| `ctrl_freq` | `dataset_control_freq.json` | 控制频率；cond_mask 时可被 mask 为 0 |
| `states` | HDF5 qpos | (1, state_dim)；可加噪声或 mask 为均值 |
| `actions` | HDF5 action | (action_chunk_size, state_dim) |
| `value` | HDF5 value（可选） | 仅 `use_value=True` 时 |
| `state_elem_mask` | 状态维度有效性 | (state_dim,) |
| `state_norm` | episode 统计 | (state_dim,) |
| `images` | HDF5 图像 | 经 resize/pad/增强/预处理后的 tensor 列表 |
| `video` | 前向预测帧（可选） | 仅 `pred_video=True` 时 |
| `input_ids` / `lang_embeds` | 指令 tokenize 或预计算 | 二选一 |
| `lang_attn_mask` | attention mask | 由 DataCollator 填充 |

### 4.2 `DataCollatorForVLAConsumerDataset`（`data/dataset.py`）

将 `VLAConsumerDataset` 的样本整理为训练 batch：

- `states`、`actions`、`state_elem_mask`、`state_norm`、`images`、`video`、`value` → `torch.stack`
- `input_ids` → `pad_sequence`（pad 到 batch 内最大长度）
- `lang_embeds` → `pad_sequence` + 生成 `lang_attn_mask`
- 禁止同一 batch 混用 `input_ids` 和 `lang_embeds`

---

## 5. 数据集类型系统

每个 HDF5 目录通过其路径中的标识 fragment 自动匹配数据集类型，决定：

- 左右臂关节数（`left_arm_dim` / `right_arm_dim`）
- 动作键名（`action` / `action_joint`）
- 腕部相机键名（`cam_left_wrist` / `cam_left`）
- 指令读取方式（`prompts` 多选列表 / 单标量）

### 5.1 注册的数据集类型（`data/utils/configs.py`）

| 类型 | 臂关节数(L/R) | 动作键 | 腕相机键(L/R) | prompts 模式 | 数据来源 |
|------|---------------|--------|---------------|--------------|----------|
| `robotwin` | auto(从 HDF5 读) | `action` | `cam_left_wrist` / `cam_right_wrist` | 多选列表 | RoboTwin 处理数据 |
| `agibot` | 7/7 | `action_joint` | `cam_left` / `cam_right` | 单标量 | 灵巧手数据 |
| `robocoin` | 7/7 | `action_joint` | `cam_left` / `cam_right` | 单标量 | robocoin |
| `agilex` | 6/6 | `action_joint` | `cam_left` / `cam_right` | 单标量 | agilex 底盘 |
| `robomind` | 6/6 | `action_joint` | `cam_left` / `cam_right` | 单标量 | robomind |
| `tianyi` | 7/7 | `action_joint` | `cam_left` / `cam_right` | 单标量 | 天依数据 |
| `tienkung` | 7/7 | `action_joint` | `cam_left` / `cam_right` | 单标量 | 天工真机数据 |
| `songling` | 6/6 | `action_joint` | `cam_left` / `cam_right` | 单标量 | 松灵数据 |

### 5.2 路径 → 类型映射（`data/utils/configs.py: _DIR_TO_DATASET_TYPE`）

通过数据路径中包含的目录名 fragment 自动检测，例如：

| 路径片段 | 类型 |
|----------|------|
| `robotwin_clean_5k_2` | `robotwin` |
| `tianyi_data` | `tianyi` |
| `tienkung` | `tienkung` |
| `agilex` | `agilex` |
| `robomind` | `robomind` |

检测函数：`detect_dataset_type(file_path)` → 类型字符串。

### 5.3 辅助函数（`data/utils/utils_datasets.py`）

| 函数 | 功能 |
|------|------|
| `detect_dataset_type(file_path)` | 按路径片段检测数据类型 |
| `get_config(dataset_type)` | 获取类型对应的 `DatasetConfig` |
| `get_arm_dims(f, dataset_type)` | 从 HDF5 或配置读左右臂维数 |
| `get_action_array(f, dataset_type)` | 按类型读取动作数组 |
| `get_cam_wrist_keys(dataset_type)` | 获取左右腕相机键名 |
| `get_instruction(f, dataset_type)` | 获取语言指令（单/多选） |
| `has_usable_qpos(dataset_type)` | qpos 是否有运动检测意义 |

---

## 6. model_config 中的数据相关配置

以 `model_config/real_world/0509-i3-tienkung-close-door.yml` 为例：

```yaml
schema_version: 2

data:
  paths:
    - /mnt/dataset/datasets/RoboTwin2_0_processed/tianyi_data/...  # HDF5 数据目录（必填）
  consumer_dataset_type: finetune     # finetune / pretrain
  hdf5:
    use_prompt_template: false
    control_freqs_path: configs/real_world_control_freqs.json      # 控制频率表（real_world 专用）
    target_control_freq: 15
  common:                             # 合并到 model_structure["common"]
    img_history_size: 1
    action_chunk_size: 32
    num_cameras: 3
    num_vision_patches: 729
    state_dim: 128
    norm_minmax: true
    statistics_mode: separate
  dataset:                            # 合并到 model_structure["dataset"]
    buf_path: /path/to/buffer         # 缓冲路径（必填，需按本机修改）
    buf_num_chunks: 512
    buf_chunk_size: 512
    epsd_len_thresh_low: 32
    epsd_len_thresh_high: 2048
    image_aspect_ratio: pad
    tokenizer_max_length: 1024
    pred_video: true
    sample_video_fps: 1
    video_size: [384, 320]
    num_cameras: 1

architecture:
  dataset_type: 128dim_with_value        # HDF5 管线类型（决定走哪个 HDF5 实现类）
```

### 关键区分

| 键 | 作用域 | 用途 |
|----|--------|------|
| `data.consumer_dataset_type` | 数据消费 | `finetune` / `pretrain`，决定 `finetune_datasets.json` 或 `pretrain_datasets.json` |
| `architecture.dataset_type` | HDF5 布局 | 如 `128dim_with_value`，决定 HDF5 实现类（`hdf5_dataset_128dim` 或通用 `HDF5Dataset`） |
| `data.common.statistics_mode` | 归一化 | `separate` = 逐 episode 单独统计；`global` = 全数据集统一统计 |

### `data.hdf5` 子块

| 键 | 说明 |
|----|------|
| `control_freqs_path` | 控制频率 json 路径（相对仓库根） |
| `target_control_freq` | 目标控制频率（Hz） |
| `use_prompt_template` | 是否使用 prompt 模板 |
| `global_stats_path` | 全局统计路径（可选） |

---

## 7. 配置文件

| 路径 | 用途 |
|------|------|
| `configs/finetune_datasets.json` | 微调阶段数据集名称列表（`consumer_dataset_type: finetune` 时使用） |
| `configs/pretrain_datasets.json` | 预训练阶段数据集名称列表 |
| `configs/dataset_control_freq.json` | 数据集名称 → 控制频率（Hz）映射 |
| `configs/dataset_stat.json` | 数据集名称 → state/action 统计量（mean、std 等） |
| `configs/real_world_control_freqs.json` | real_world 任务专用控制频率表（与原 RDT `data/control_freqs.json` 副本） |
| `configs/state_vec.py` | 128 维状态空间中各维度的语义映射表（`STATE_VEC_IDX_MAPPING`） |

---

## 8. 图像预处理流程

```
HDF5 JPEG 字节
    → cv2.imdecode → BGR ndarray
    → Image.fromarray → PIL
    → Resize（可选）
    → auto_adjust_brightness（可选，亮度 < 0.15 时提亮）
    → 50% 概率 ColorJitter（可为 color_only / corrput_only / both）
    → 50% 概率 image_corrupt（噪声、模糊等）
    → expand2square_pad
    → image_processor.preprocess → pixel_values
```

视频帧额外处理：
```
视频帧 NHWC uint8
    → video_metas_nhwc_to_chw_float_tensor → CHW float
    → preprocess_video_chw_for_training（resize + 亮度调整 + 增强 + normalize）
```
