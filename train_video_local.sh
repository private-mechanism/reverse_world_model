#!/bin/bash
# 自动激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate wam
# Video Expert 本地 / 单机训练（不依赖百度集群的 WORLD_SIZE、NPROC_PER_NODE、MASTER_* 等变量）。
# 用法：./train_video_local.sh <CONFIG_NAME>   （对应 model_config/<CONFIG_NAME>.yml）
# 进程与 GPU 由当前环境的 accelerate 默认配置决定；也可覆盖：
#   export TRAIN_VIDEO_NUM_PROCESSES=1   # 单进程单卡（不加 --multi_gpu，避免 accelerate 报错）
#   export TRAIN_VIDEO_NUM_PROCESSES=4   # 多卡时加 --num_processes 4 --multi_gpu（需 >=2 才会带 --multi_gpu）
#   export TRAIN_VIDEO_LAUNCH_EXTRA="--mixed_precision bf16"   # 追加传给 accelerate launch 的参数（可选）
# 多节点请继续用 train_video.sh。

#todo: 目前直接sh train_video_local.sh命令无法执行，

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

LAUNCH_ARGS=()
if [ -n "${TRAIN_VIDEO_NUM_PROCESSES:-}" ]; then
  LAUNCH_ARGS+=(--num_processes "${TRAIN_VIDEO_NUM_PROCESSES}")
  # accelerate 要求 --multi_gpu 时 num_processes>=2；单卡只传 --num_processes 1
  if [ "${TRAIN_VIDEO_NUM_PROCESSES}" -ge 2 ] 2>/dev/null; then
    LAUNCH_ARGS+=(--multi_gpu)
  fi
fi

# shellcheck disable=SC2206
EXTRA=( ${TRAIN_VIDEO_LAUNCH_EXTRA:-} )

shift  # 去掉 CONFIG_NAME，剩余参数传给 main_video.py
accelerate launch "${LAUNCH_ARGS[@]}" "${EXTRA[@]}" \
  main_video.py \
  --model_config_path="$CONFIG_FILE" \
  "$@"
