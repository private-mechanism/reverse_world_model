#!/bin/bash
# Video Expert + Qwen2.5-VL 本地训练。用法：./train_video_qwen_local.sh <CONFIG_NAME>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_NAME="${1:?用法: $0 <CONFIG_NAME>（不含 .yml）}"
CONFIG_FILE="model_config/${CONFIG_NAME}.yml"

if [ ! -f "$CONFIG_FILE" ]; then
  echo "找不到配置: $CONFIG_FILE"
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-8}"

LAUNCH_ARGS=()
if [ -n "${TRAIN_VIDEO_NUM_PROCESSES:-}" ]; then
  LAUNCH_ARGS+=(--num_processes "${TRAIN_VIDEO_NUM_PROCESSES}")
  if [ "${TRAIN_VIDEO_NUM_PROCESSES}" -ge 2 ] 2>/dev/null; then
    LAUNCH_ARGS+=(--multi_gpu)
  fi
fi

EXTRA=( ${TRAIN_VIDEO_LAUNCH_EXTRA:-} )

accelerate launch "${LAUNCH_ARGS[@]}" "${EXTRA[@]}" \
  main_video_qwen.py \
  --model_config_path="$CONFIG_FILE"
