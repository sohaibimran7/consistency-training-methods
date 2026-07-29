#!/usr/bin/env bash
# Resumably populate the shared Hugging Face cache for Figure 6.
#
# Required environment:
#   REPO_DIR                         absolute repository checkout
#   HF_HOME                          shared project/scratch Hugging Face cache
#   FIGURE6_CACHE_CAPACITY_GB        verified usable cache allocation (>=1300)
#   HF_TOKEN                         required when the gated Llama checkpoint is selected
#
# With no arguments all seven models are fetched.  Model keys may be supplied
# after the script name to fetch a subset.  Re-submission reuses cached shards.
#SBATCH --job-name=f6-prefetch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=72
#SBATCH --time=12:00:00
#SBATCH --signal=TERM@300
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

require_env() {
    local name=$1
    if [[ -z "${!name:-}" ]]; then
        echo "ERROR: required environment variable $name is not set." >&2
        exit 2
    fi
}

require_env REPO_DIR
require_env HF_HOME
require_env FIGURE6_CACHE_CAPACITY_GB

if [[ "$REPO_DIR" != /* || "$HF_HOME" != /* ]]; then
    echo "ERROR: REPO_DIR and HF_HOME must be absolute paths." >&2
    exit 2
fi
if [[ ! -f "$REPO_DIR/experiments/eval_awareness/figure6/models.yaml" ]]; then
    echo "ERROR: REPO_DIR does not contain the Figure 6 model registry." >&2
    exit 2
fi

shared_cache=false
if [[ -n "${PROJECTDIR:-}" && "$HF_HOME" == "$PROJECTDIR"/* ]]; then
    shared_cache=true
fi
if [[ -n "${SCRATCHDIR:-}" && "$HF_HOME" == "$SCRATCHDIR"/* ]]; then
    shared_cache=true
fi
if [[ "$shared_cache" != true ]]; then
    echo "ERROR: HF_HOME must be inside the shared project or scratch filesystem." >&2
    exit 2
fi

cd "$REPO_DIR"
mkdir -p "$HF_HOME"
export HF_HOME

# Fail before installation when the declared cache capacity or gated-model
# credential is missing.
"$REPO_DIR/.venv/bin/python" - "$FIGURE6_CACHE_CAPACITY_GB" <<'PY'
import math
import sys

try:
    capacity_gb = float(sys.argv[1])
except ValueError as exc:
    raise SystemExit("FIGURE6_CACHE_CAPACITY_GB must be numeric") from exc
if not math.isfinite(capacity_gb) or capacity_gb < 1300:
    raise SystemExit("Figure 6 needs a verified shared cache allocation of at least 1300 GB")
print(f"Verified cache allocation: {capacity_gb:.1f} GB (minimum 1300 GB)")
PY

gated_selected=false
if [[ $# -eq 0 ]]; then
    gated_selected=true
else
    for key in "$@"; do
        case "$key" in
            qwen36|qwen32|qwen_mo_mid|qwen_mo_post|llama_mo_mid|llama_mo_post) ;;
            llama33) gated_selected=true ;;
            *)
                echo "ERROR: unknown Figure 6 model key $key." >&2
                exit 2
                ;;
        esac
    done
fi
if [[ "$gated_selected" == true && -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is required to prefetch the selected gated checkpoint." >&2
    exit 2
fi

# Reuse this already-allocated GPU to install and smoke-check the exact serving
# stack, avoiding a second sequential scheduler request before prefetch.
bash infra/isambard/setup_gpu_env.sh
source .venv/bin/activate

MODELS_CONFIG="$REPO_DIR/experiments/eval_awareness/figure6/models.yaml"

# snapshot_download uses resumable cache blobs.  Each checkpoint is processed
# in registry order while up to four shards download concurrently.
python - "$MODELS_CONFIG" "$@" <<'PY'
import os
import sys

import yaml
from huggingface_hub import snapshot_download

config_path, *requested = sys.argv[1:]
with open(config_path, encoding="utf-8") as handle:
    registry = yaml.safe_load(handle)["models"]

keys = requested or list(registry)
unknown = sorted(set(keys) - set(registry))
if unknown:
    raise SystemExit(f"unknown Figure 6 model key(s): {unknown}")
if any(registry[key].get("gated", False) for key in keys) and not os.environ.get("HF_TOKEN"):
    raise SystemExit("HF_TOKEN is required to prefetch the selected gated checkpoint")

try:
    max_workers = int(os.environ.get("FIGURE6_PREFETCH_WORKERS", "4"))
except ValueError as exc:
    raise SystemExit("FIGURE6_PREFETCH_WORKERS must be an integer") from exc
if not 1 <= max_workers <= 8:
    raise SystemExit("FIGURE6_PREFETCH_WORKERS must be between 1 and 8")

for key in keys:
    model = registry[key]
    print(f"Prefetching {key}: {model['model_id']} at {model['revision']}", flush=True)
    path = snapshot_download(
        repo_id=model["model_id"],
        revision=model["revision"],
        token=os.environ.get("HF_TOKEN") or None,
        ignore_patterns=model.get("prefetch_ignore_patterns"),
        max_workers=max_workers,
    )
    print(f"Cached {key} at {path}", flush=True)
PY
