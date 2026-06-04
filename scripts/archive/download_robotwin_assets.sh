#!/usr/bin/env bash
set -eo pipefail

cd /mnt/world_foundational_model/pd_data/RoboTwin

eval "$(conda shell.bash hook)"
conda activate wam

export TMPDIR=/mnt/world_foundational_model/pd_data/pip_tmp
export HF_HOME=/mnt/world_foundational_model/pd_data/hf_cache

bash script/_download_assets.sh

