#!/usr/bin/env bash
set -eo pipefail

usage() {
  cat <<'EOF'
Run WVWAM Stage 2 checkpoint in RoboTwin online success-rate evaluation.

Usage:
  scripts/run_robotwin_stage2_success_rate.sh \
    --robotwin_root /path/to/RoboTwin \
    --config /path/to/robotwin_eval_config.yml \
    [extra eval_policy.py args...]

The script:
  1. Symlinks this repo's RoboTwin adapter to <RoboTwin>/policy/WVWAM.
  2. Exports WAM_REPO.
  3. Runs RoboTwin's script/eval_policy.py with the provided config.

The RoboTwin config should set:
  policy_name: WVWAM.deploy_policy

and include the WVWAM policy args from:
  robotwin_success_rate/WVWAM/wvwam_policy_args.yml
EOF
}

ROBOTWIN_ROOT=""
CONFIG_PATH=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --robotwin_root)
      ROBOTWIN_ROOT="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$ROBOTWIN_ROOT" || -z "$CONFIG_PATH" ]]; then
  usage
  exit 2
fi

WAM_REPO="/mnt/world_foundational_model/fyzhao/reverse_world_model"
ROBOTWIN_ROOT="$(cd "$ROBOTWIN_ROOT" && pwd)"

if [[ ! -f "$ROBOTWIN_ROOT/script/eval_policy.py" ]]; then
  echo "RoboTwin eval script not found: $ROBOTWIN_ROOT/script/eval_policy.py" >&2
  exit 1
fi

mkdir -p "$ROBOTWIN_ROOT/policy"
ln -sfn "$WAM_REPO/robotwin_success_rate/WVWAM" "$ROBOTWIN_ROOT/policy/WVWAM"

export WAM_REPO
export PYTHONPATH="$WAM_REPO:$ROBOTWIN_ROOT:$ROBOTWIN_ROOT/policy:${PYTHONPATH:-}"

cd "$ROBOTWIN_ROOT"
python script/eval_policy.py --config "$CONFIG_PATH" "${EXTRA_ARGS[@]}"
