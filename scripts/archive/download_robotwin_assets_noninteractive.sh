#!/usr/bin/env bash
set -eo pipefail

ROBOTWIN_ROOT=/mnt/world_foundational_model/pd_data/RoboTwin
ASSET_DIR="$ROBOTWIN_ROOT/assets"

cd "$ASSET_DIR"

eval "$(conda shell.bash hook)"
conda activate wam

export TMPDIR=/mnt/world_foundational_model/pd_data/pip_tmp
export HF_HOME=/mnt/world_foundational_model/pd_data/hf_cache

python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="TianxingChen/RoboTwin2.0",
    allow_patterns=["background_texture.zip", "embodiments.zip", "objects.zip"],
    local_dir=".",
    repo_type="dataset",
)
PY

for archive in background_texture.zip embodiments.zip objects.zip; do
  test -f "$archive"
done

unzip -o background_texture.zip
unzip -o embodiments.zip
unzip -o objects.zip

cd "$ROBOTWIN_ROOT"
python ./script/update_embodiment_config_path.py

