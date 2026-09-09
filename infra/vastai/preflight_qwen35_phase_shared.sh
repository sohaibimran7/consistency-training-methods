#!/usr/bin/env bash
# Non-production wrapper for the generic phase-shared Qwen3.5 GPU preflight.
#
# It intentionally does not create an output directory, copy credentials, or
# select GPUs.  The Python harness refuses a non-fresh evidence path and reads
# only the caller's explicit CUDA_VISIBLE_DEVICES allocation.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == "-1" || "${CUDA_VISIBLE_DEVICES}" == "NoDevFiles" ]]; then
    echo "ERROR: set CUDA_VISIBLE_DEVICES explicitly (for example 0,1 or GPU-uuid-a,GPU-uuid-b)." >&2
    exit 2
fi

python_bin="${CTM_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
    if [[ -x "$repo_root/.venv/bin/python" ]]; then
        python_bin="$repo_root/.venv/bin/python"
    else
        python_bin="python3"
    fi
fi
if ! "$python_bin" -c 'import sys' >/dev/null 2>&1; then
    echo "ERROR: CTM_PYTHON is not an executable Python interpreter: $python_bin" >&2
    exit 2
fi

# Run this in a deliberately short-lived process before any model is loaded.
# A GPU reported by NVIDIA as requiring a reset can appear in
# CUDA_VISIBLE_DEVICES yet fail only on its first CUDA allocation.  Touching
# every logical device here makes that condition fail before this harness
# starts HF, vLLM, multiprocessing, or an expensive download.  The subprocess
# exit releases all of these probe contexts before the main Python harness
# begins.
"$python_bin" - <<'PY'
import os
import sys

try:
    import torch
except Exception as exc:
    raise SystemExit(f"ERROR: phase-shared GPU health probe cannot import torch: {type(exc).__name__}: {exc}")

raw_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
requested = [token.strip() for token in raw_visible.split(",")]
if not requested or any(not token for token in requested):
    raise SystemExit("ERROR: phase-shared GPU health probe received an invalid CUDA_VISIBLE_DEVICES allocation")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: phase-shared GPU health probe requires CUDA, but torch.cuda.is_available() is false")
visible_count = torch.cuda.device_count()
if visible_count != len(requested):
    raise SystemExit(
        "ERROR: phase-shared GPU health probe sees a different number of logical GPUs than CUDA_VISIBLE_DEVICES "
        f"(torch={visible_count}, requested={len(requested)})"
    )

for logical_index in range(visible_count):
    torch.cuda.set_device(logical_index)
    probe = torch.empty((1,), device=f"cuda:{logical_index}", dtype=torch.uint8)
    probe.fill_(1)
    torch.cuda.synchronize(logical_index)
    del probe

print(f"CTM_PHASE_SHARED_GPU_HEALTH=passed visible_gpu_count={visible_count}")
PY

cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" infra/vastai/preflight_qwen35_phase_shared.py "$@"
