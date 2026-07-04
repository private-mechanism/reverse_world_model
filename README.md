# WAM

<div align="center">

**世界模型 / 视频专家（Wan）训练子集** — 自 [`RoboTwin/policy/RDT`](https://rdt-robotics.github.io/rdt-robotics/) 精简，保留因果 World 策略（含 value 分支）与仅训练 Video Expert 两条链路。

<!-- 发布前替换下方占位：仓库 URL、LICENSE、Python 版本、论文 arXiv 等 -->
<!--
[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://example.com/your-project-page)
[![License](https://img.shields.io/badge/License-TBD-lightgrey.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
-->

[论文（arXiv）](#引用) · [项目主页](#链接速览) · [环境安装](#环境) · [快速开始](#快速开始) · [文档](#文档索引)

</div>

---

## 链接速览

| 类型 | 链接 |
|------|------|
| **论文（arXiv）** | `https://arxiv.org/abs/XXXX.XXXXX` <!-- TODO: 替换为正式 arXiv ID --> |
| **项目主页 / Demo** | `https://example.com/wam` <!-- TODO: 项目页、Gradio/在线 Demo 等 --> |
| **补充材料** | `https://example.com/wam/supplement` <!-- TODO: 可选 --> |
| **上游 RDT** | [RDT 官网](https://rdt-robotics.github.io/rdt-robotics/) |

---

## 简介

本仓库是从 RDT 中拆分出的**最小可运行训练子集**：多模态世界策略在因果注意力设定下训练（SigLIP / UMT5 / VAE + Action & Video Expert），并支持 **HDF5 / buffer** 数据管线；另提供 **Video-only** 路径，单独训练 Video Expert（`VideoDataset`），与全量 `VLAConsumerDataset` 解耦。配置风格与 RDT 对齐：`model_config` v2、`accelerate launch`。

**工作目录**：请在仓库 **`WAM/` 根目录** 下执行脚本与 Python，相对路径与 `policy/RDT` 约定一致。

---

## 功能概览

- **World 训练**：只保留 Original、Independent Goal Reverse、Joint Goal Reverse 三种 diffusion variant。
- **Video-only 训练**：单独训练 Video Expert（Wan），数据使用 `VideoDataset`。
- **配置**：`model_config/*.yml`（schema v2）+ `configs/` 下结构 YAML 与数据集 JSON。

---

## 目录结构（简）

```
WAM/
├── main_world.py              # World 训练入口
├── main_video.py              # Video Expert 训练入口
├── train_act_world_aug.sh     # World 分布式启动（示例）
├── train_wam_local.sh         # World 单机/本地启动（见脚本内注释）
├── train_video.sh             # Video 多节点集群启动
├── train_video_local.sh       # Video 单机/本地启动
├── model_config/              # 训练 YAML（数据路径、runner、超参等）
├── configs/                   # 模型结构、数据集统计与控制频率等
├── wam/                       # 核心包：models / trainers / runners / samples
├── data/                      # 数据集、collator、HDF5 与增广
├── scripts/                   # 辅助脚本（如从 yml 读字段）
└── docs/                      # 详细说明与配置速查
```

---

## 环境

依赖与 Conda 步骤见 **[`docs/env.md`](docs/env.md)**（PyTorch、Diffusers、Accelerate、h5py、imgaug 等）。

```bash
conda activate wam   # 或你自建的环境名
accelerate config    # 首次建议配置分布式默认值
```

---

## 快速开始

在 **`WAM/`** 根目录执行：

```bash
# Original WVWAM
accelerate launch main_world.py --model_config_path validation_tmp/original_wvwam.yml

# Independent Goal-Conditioned Reverse Diffusion
accelerate launch main_world.py --model_config_path validation_tmp/goal_reverse_independent.yml

# Joint Goal-Conditioned Reverse Diffusion
accelerate launch main_world.py --model_config_path validation_tmp/goal_reverse_joint.yml

# 仅 Video Expert
accelerate launch main_video.py --model_config_path model_config/<你的配置>.yml
```

集群与多机脚本：`./train_act_world_aug.sh`、`./train_video.sh`、`./train_video_local.sh`（用法见各脚本内注释）。

### 自检

```bash
PYTHONPATH=. python scripts/smoke_three_world_variants.py
```

---

## 预训练权重与数据

<!-- TODO: 公开 checkpoint 时在此列出 HuggingFace / Google Drive 链接与文件名；若不开源权重，保留说明即可 -->

本仓库**不包含**大体积预训练权重与 HDF5 数据；路径请在本地 `model_config` 中配置。若后续发布权重，建议在此增加表格：`名称 | 说明 | 下载链接`。

---

## 文档索引

| 文档 | 内容 |
|------|------|
| [`docs/README.md`](docs/README.md) | 模块与文件职责、配置表、运行提示 |
| [`docs/env.md`](docs/env.md) | Conda / pip 依赖与启动前检查 |
| [`docs/model_config_schema.md`](docs/model_config_schema.md) | `model_config` v2 字段说明 |
| [`docs/train_config_guide.md`](docs/train_config_guide.md) | World / Video / DeepSpeed 配置速查与模板 |

---

## 引用

若本工作伴随正式论文发表，请在发表后替换下方 BibTeX（并同步更新 [链接速览](#链接速览) 中的 arXiv 地址）。

```bibtex
@article{yourname2026wam,
  title   = {TODO: Paper Title},
  author  = {TODO},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

**arXiv（占位）**：`https://arxiv.org/abs/XXXX.XXXXX`

---

## 贡献

欢迎 Issue 与 Pull Request。<!-- TODO: 若有 CONTRIBUTING.md，改为：详见 [CONTRIBUTING.md](CONTRIBUTING.md) -->

---

## 许可证与致谢

<!-- TODO: 与上游 RDT 仓库许可证对齐后，在仓库根目录添加 LICENSE 并取消下方注释中的链接 -->

训练管线基于 RDT 相关代码精简与扩展；使用前请确认与上游许可证兼容。

---

## 其他占位（可选后续补充）

- **常见问题（FAQ）**：可新增 `docs/FAQ.md` 并在本节链接。
- **更新日志**：可新增 `CHANGELOG.md` 遵循 [Keep a Changelog](https://keepachangelog.com/)。
