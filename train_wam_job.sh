#!/bin/bash
# World 训练启动（与原先 aug 版等价超参改由 model_config yml 提供，见 trainers.train_world._apply_training_args_from_model_config）。
# 用法：./train_act_world_aug.sh <CONFIG_NAME>   （对应 model_config/<CONFIG_NAME>.yml）
# 多机：由调度系统注入 WORLD_SIZE、NPROC_PER_NODE、RANK、MASTER_ADDR、MASTER_PORT。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_NAME="${1:?用法: $0 <CONFIG_NAME>（不含 .yml）}"
CONFIG_FILE="model_config/${CONFIG_NAME}.yml"

if [ ! -f "$CONFIG_FILE" ]; then
  echo "找不到配置: $CONFIG_FILE"
  exit 1
fi

echo "CONFIG_FILE_PATH: $CONFIG_FILE"

export TORCH_DISTRIBUTED_BACKEND=nccl
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=180000
export NCCL_P2P_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_IB_HCA=

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_DEVICE_MAX_CONNECTIONS=8
export CUDA_LAUNCH_BLOCKING=0
mkdir -p /tmp/deepspeed_nvme

num_processes=$((WORLD_SIZE * NPROC_PER_NODE))
echo "num_processes=$num_processes"

accelerate launch --same_network --multi_gpu --gpu_ids all \
  --num_machines "${WORLD_SIZE}" --num_processes "${num_processes}" \
  --machine_rank "${RANK}" --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  main_world.py \
  --model_config_path="$CONFIG_FILE"
