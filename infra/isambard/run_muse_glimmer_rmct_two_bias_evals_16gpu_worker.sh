#!/usr/bin/env bash
# One rank in a three-phase, two-condition Muse Glimmer two-bias campaign.

set -euo pipefail
umask 077

if (( $# != 9 )); then
    echo "usage: $0 <launcher.py> <python> <early|late> <phase> <campaign-root> <training-repository> <source-manifest> <artifact-root> <cache-parent>" >&2
    exit 2
fi

launcher=$1
python_bin=$2
group=$3
phase=$4
campaign_root=$5
training_repository=$6
source_manifest=$7
artifact_root=$8
cache_parent=$9

if [[ ! -f "$launcher" || -L "$launcher" || ! -x "$python_bin" ]]; then
    echo "ERROR: Muse evaluator launcher or Python is unavailable" >&2
    exit 2
fi
if [[ "$group" != early && "$group" != late ]]; then
    echo "ERROR: Muse evaluation group must be early or late" >&2
    exit 2
fi
if [[ ! "$phase" =~ ^[123]$ ]]; then
    echo "ERROR: Muse evaluation phase must be 1, 2, or 3" >&2
    exit 2
fi
if [[ ! "$cache_parent" = /* || ! -d "$cache_parent" ]]; then
    echo "ERROR: worker cache parent must be an existing absolute directory" >&2
    exit 2
fi

rank=${SLURM_PROCID:-}
if [[ ! "$rank" =~ ^[0-9]+$ ]] || (( 10#$rank > 15 )); then
    echo "ERROR: Muse evaluation worker requires a Slurm rank in [0, 15]" >&2
    exit 2
fi
if (( 10#$rank >= 14 )); then
    exit 0
fi

if [[ "$group" == early ]]; then
    first_condition=base
    second_condition=step016
else
    first_condition=step064
    second_condition=final
fi

condition=''
task_index=''
case "$phase" in
    1)
        condition=$first_condition
        task_index=$((10#$rank + 1))
        ;;
    2)
        if (( 10#$rank < 7 )); then
            condition=$first_condition
            task_index=$((10#$rank + 15))
        else
            condition=$second_condition
            task_index=$((10#$rank - 6))
        fi
        ;;
    3)
        condition=$second_condition
        task_index=$((10#$rank + 8))
        ;;
esac

tokens=${CUDA_VISIBLE_DEVICES:-}
"$python_bin" - "$tokens" <<'PY'
import sys

tokens = [value.strip() for value in sys.argv[1].split(",") if value.strip()]
if len(tokens) != 1 or tokens[0] in {"-1", "NoDevFiles"} or any(character.isspace() for character in tokens[0]):
    raise SystemExit("error: every Muse evaluation rank must receive exactly one Slurm-visible GPU")
PY

worker_tmp=$(mktemp -d "$cache_parent/ctm-muse-two-bias-${group}-p${phase}-j${SLURM_JOB_ID:-manual}-r${rank}-XXXXXX")
mkdir -p "$worker_tmp/tmp" "$worker_tmp/xdg-cache" "$worker_tmp/torchinductor" "$worker_tmp/triton" "$worker_tmp/cuda-cache"
chmod 700 "$worker_tmp" "$worker_tmp/tmp" "$worker_tmp/xdg-cache" "$worker_tmp/torchinductor" "$worker_tmp/triton" "$worker_tmp/cuda-cache"
export TMPDIR="$worker_tmp/tmp"
export TMP="$worker_tmp/tmp"
export TEMP="$worker_tmp/tmp"
export XDG_CACHE_HOME="$worker_tmp/xdg-cache"
export TORCHINDUCTOR_CACHE_DIR="$worker_tmp/torchinductor"
export TRITON_CACHE_DIR="$worker_tmp/triton"
export CUDA_CACHE_PATH="$worker_tmp/cuda-cache"

exec "$python_bin" "$launcher" --group "$group" worker \
    --campaign-root "$campaign_root" \
    --training-repository "$training_repository" \
    --source-stage2-manifest "$source_manifest" \
    --stage2-artifact-root "$artifact_root" \
    --condition "$condition" \
    --task-index "$task_index" \
    --python "$python_bin"
