#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/sync_checkpoints_to_hdfs.sh --hdfs-root HDFS_PATH [options]

Options:
  --hdfs-root PATH       Destination root on HDFS, e.g.
                         hdfs:///user/$USER/reverse_world_model/checkpoints
  --local-root PATH      Local checkpoint root. Default: checkpoints
  --run-name NAME        Upload only one run directory under local-root.
                         Example: validation-stage3-1-coa-action-loss
  --checkpoint NAME      Upload only one checkpoint directory under run-name.
                         Example: checkpoint-1000
  --dry-run             Print what would be uploaded without writing to HDFS.
  -h, --help            Show this help.

Examples:
  scripts/sync_checkpoints_to_hdfs.sh \
    --hdfs-root hdfs:///user/$USER/reverse_world_model/checkpoints \
    --dry-run

  scripts/sync_checkpoints_to_hdfs.sh \
    --hdfs-root hdfs:///user/$USER/reverse_world_model/checkpoints \
    --run-name validation-stage3-1-coa-action-loss

  scripts/sync_checkpoints_to_hdfs.sh \
    --hdfs-root hdfs:///user/$USER/reverse_world_model/checkpoints \
    --run-name validation-stage3-1-coa-action-loss \
    --checkpoint checkpoint-1000
EOF
}

local_root="checkpoints"
hdfs_root="${HDFS_CHECKPOINT_ROOT:-}"
run_name=""
checkpoint_name=""
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hdfs-root)
      hdfs_root="$2"
      shift 2
      ;;
    --local-root)
      local_root="$2"
      shift 2
      ;;
    --run-name)
      run_name="$2"
      shift 2
      ;;
    --checkpoint)
      checkpoint_name="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$hdfs_root" ]]; then
  echo "ERROR: --hdfs-root is required, or set HDFS_CHECKPOINT_ROOT." >&2
  exit 2
fi

if [[ ! -d "$local_root" ]]; then
  echo "ERROR: local checkpoint root does not exist: $local_root" >&2
  exit 2
fi

fs_cmd=()
if command -v hdfs >/dev/null 2>&1; then
  fs_cmd=(hdfs dfs)
elif command -v hadoop >/dev/null 2>&1; then
  fs_cmd=(hadoop fs)
elif [[ "$dry_run" -eq 0 ]]; then
  echo "ERROR: neither 'hdfs' nor 'hadoop' was found in PATH." >&2
  echo "Activate the environment/module that provides the HDFS client, then rerun this script." >&2
  exit 127
fi

targets=()
if [[ -n "$run_name" && -n "$checkpoint_name" ]]; then
  targets+=("$local_root/$run_name/$checkpoint_name")
elif [[ -n "$run_name" ]]; then
  targets+=("$local_root/$run_name")
else
  while IFS= read -r path; do
    targets+=("$path")
  done < <(find "$local_root" -mindepth 1 -maxdepth 1 -type d | sort)
fi

if [[ ${#targets[@]} -eq 0 ]]; then
  echo "ERROR: no checkpoint targets found under $local_root." >&2
  exit 2
fi

for target in "${targets[@]}"; do
  if [[ ! -e "$target" ]]; then
    echo "ERROR: target does not exist: $target" >&2
    exit 2
  fi
done

echo "Local root: $local_root"
echo "HDFS root:  $hdfs_root"
echo "Targets:"
for target in "${targets[@]}"; do
  du -sh "$target"
done

if [[ "$dry_run" -eq 1 ]]; then
  echo "Dry run only; no files uploaded."
  exit 0
fi

"${fs_cmd[@]}" -mkdir -p "$hdfs_root"

manifest_dir="$(mktemp -d)"
trap 'rm -rf "$manifest_dir"' EXIT

for target in "${targets[@]}"; do
  base="$(basename "$target")"
  manifest="$manifest_dir/${base}.manifest.tsv"
  find "$target" -type f -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS\t%p\n' | sort > "$manifest"

  echo "Uploading $target -> $hdfs_root/"
  "${fs_cmd[@]}" -put -f "$target" "$hdfs_root/"
  echo "Uploading manifest $manifest -> $hdfs_root/${base}.manifest.tsv"
  "${fs_cmd[@]}" -put -f "$manifest" "$hdfs_root/${base}.manifest.tsv"
done

echo "Done. HDFS listing:"
"${fs_cmd[@]}" -ls -h "$hdfs_root"
