# WAM Conda 环境说明

本文给出一套在 `WAM` 里可复现训练/调试的 conda 环境准备步骤，包含：

- 基础 conda 环境创建
- 训练依赖安装（PyTorch / Diffusers / Accelerate 等）
- 可选 DeepSpeed 安装
- 启动前自检

---

## 1. 创建 conda 环境

推荐 Python 版本：`3.10`（与当前主流 `torch` / `diffusers` 兼容性较好）。

```bash
conda create -n wam python=3.10 -y
conda activate wam
python -V
```

---

## 2. 安装核心依赖

> 请根据你的 CUDA 版本选择对应的 PyTorch 安装命令。下面示例使用 CUDA 12.1。

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate omegaconf safetensors
pip install h5py imageio imageio-ffmpeg imgaug pyyaml tqdm einops
pip install huggingface_hub wandb matplotlib
```

如果你使用的是 CUDA 11.8，可将 torch 源替换为：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 3. 可选：安装 DeepSpeed

仅在你需要 `distributed.deepspeed` 时安装。

```bash
pip install deepspeed
```

安装后快速检查：

```bash
python -c "import deepspeed; print('deepspeed ok')"
```

---

## 4. Accelerate 初始化

首次使用建议执行一次：

```bash
accelerate config
```

常见训练启动命令：

```bash
# world
accelerate launch main_world.py --model_config_path model_config/your_world.yml

# video-only
accelerate launch main_video.py --model_config_path model_config/your_video.yml
```

---

## 5. 启动前自检

在 `WAM` 根目录执行以下检查：

```bash
cd /mnt/dataset/projs/projects/WAM

# 代码可导入
python -c "from wam.trainers.train_world import train; print('train_world import ok')"
python -c "from wam.trainers.train_video import train; print('train_video import ok')"

# CUDA 可用性
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

如果你配置了 video runner，也建议检查底座路径环境变量（按需）：

```bash
echo $WAM_WOW_VIDEO_BASE
echo $WAM_WAN22_VIDEO_BASE
```

---

## 6. 常见问题

- `ModuleNotFoundError: No module named xxx`  
  先确认已激活 conda 环境（`conda activate wam`），再补装对应 pip 包。

- `CUDA out of memory`  
  优先降低 `train_batch_size`，或开启 `gradient_accumulation_steps`，必要时启用 DeepSpeed ZeRO。

- `deepspeed` 配置不生效  
  建议在 `model_config` 里设置 `distributed.deepspeed`，避免仅靠命令行参数导致被配置覆盖。

- 训练卡在 DataLoader  
  可先将 `dataloader_num_workers` 设为 `0` 排查，再逐步增大。

