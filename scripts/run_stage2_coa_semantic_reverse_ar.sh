#!/usr/bin/env bash
set -eo pipefail

cd /mnt/world_foundational_model/fyzhao/reverse_world_model
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate wam

export WANDB_MODE=offline

accelerate launch --num_processes 1 main_world.py \
  --model_config_path=validation_tmp/stage2_coa_semantic_reverse_ar.yml \
  2>&1 | tee logs/stage2_coa_semantic_reverse_ar.log
