#!/usr/bin/env bash
# 多卡预计算 umT5 / Qwen3 语言 embedding（Accelerate，与训练启动方式一致）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_CONFIG="${MODEL_CONFIG:-model_config/wam2.2_demo.yml}"
NUM_PROCESSES="${PREPROCESS_NUM_PROCESSES:-${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}}"
BACKENDS="${BACKENDS:-umt5}"

LAUNCH_ARGS=()
if [ -n "${NUM_PROCESSES}" ] && [ "${NUM_PROCESSES}" -ge 1 ] 2>/dev/null; then
  LAUNCH_ARGS+=(--num_processes "${NUM_PROCESSES}")
  if [ "${NUM_PROCESSES}" -ge 2 ] 2>/dev/null; then
    LAUNCH_ARGS+=(--multi_gpu)
  fi
fi

# shellcheck disable=SC2206
EXTRA=( ${PREPROCESS_LAUNCH_EXTRA:-} )

accelerate launch "${LAUNCH_ARGS[@]}" "${EXTRA[@]}" \
  -m data.preprocess.batch_lang_embed \
  --model-config "$MODEL_CONFIG" \
  --backends $BACKENDS \
  "$@"
