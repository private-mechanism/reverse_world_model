#!/bin/bash
# World（WAM 联合）本地 / 单机训练；不依赖调度系统注入的 WORLD_SIZE、NPROC_PER_NODE、MASTER_*。
# 用法：./train_wam_local.sh <CONFIG_NAME>   （对应 model_config/<CONFIG_NAME>.yml）
# 超参由 yml 与 trainers.train_world 等提供（与 train_act_world_aug.sh 相同入口 main_world.py）。
# 进程与 GPU 由当前环境的 accelerate 默认配置决定；也可覆盖：
#   export TRAIN_WAM_NUM_PROCESSES=1   # 单进程单卡
#   export TRAIN_WAM_NUM_PROCESSES=4   # 多卡时 num_processes>=2 会自动加 --multi_gpu
#   export TRAIN_WAM_LAUNCH_EXTRA="--mixed_precision bf16"   # 追加传给 accelerate launch（可选）
# 多节点集群请用：./train_act_world_aug.sh <CONFIG_NAME>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_NAME="${1:?用法: $0 <CONFIG_NAME>（不含 .yml）}"
CONFIG_FILE="model_config/${CONFIG_NAME}.yml"

if [ ! -f "$CONFIG_FILE" ]; then
  echo "找不到配置: $CONFIG_FILE"
  exit 1
fi

echo "CONFIG_FILE_PATH: $CONFIG_FILE"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-8}"
mkdir -p /tmp/deepspeed_nvme

LAUNCH_ARGS=()
if [ -n "${TRAIN_WAM_NUM_PROCESSES:-}" ]; then
  LAUNCH_ARGS+=(--num_processes "${TRAIN_WAM_NUM_PROCESSES}")
  if [ "${TRAIN_WAM_NUM_PROCESSES}" -ge 2 ] 2>/dev/null; then
    LAUNCH_ARGS+=(--multi_gpu)
  fi
fi

# shellcheck disable=SC2206
EXTRA=( ${TRAIN_WAM_LAUNCH_EXTRA:-} )

accelerate launch "${LAUNCH_ARGS[@]}" "${EXTRA[@]}" \
  main_world.py \
  --model_config_path="$CONFIG_FILE"
