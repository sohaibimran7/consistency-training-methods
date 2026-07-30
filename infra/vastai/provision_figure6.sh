#!/usr/bin/env bash
# Build the minimal Figure 6 client venv without replacing the image runtime.
set -euo pipefail
set +x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
REPO_DIR=$(cd "$SCRIPT_DIR/../.." && pwd -P)
FIGURE6_VENV=${FIGURE6_VENV:-$REPO_DIR/.venv-figure6}
BASE_PYTHON=${FIGURE6_BASE_PYTHON:-python3}

if [[ "$FIGURE6_VENV" != /* ]]; then
    echo "ERROR: FIGURE6_VENV must be an absolute path." >&2
    exit 2
fi
if ! command -v "$BASE_PYTHON" >/dev/null 2>&1; then
    echo "ERROR: base Python executable not found: $BASE_PYTHON" >&2
    exit 2
fi

if [[ ! -x "$FIGURE6_VENV/bin/python" ]]; then
    "$BASE_PYTHON" -m venv --system-site-packages "$FIGURE6_VENV"
fi
source "$FIGURE6_VENV/bin/activate"

python -m pip install --disable-pip-version-check -r "$SCRIPT_DIR/figure6-requirements.txt"
python -m pip install --disable-pip-version-check --no-deps -e "$REPO_DIR"
python -m pip check

python - "$FIGURE6_VENV" <<'PY'
import pathlib
import platform
import sys

import openai
import torch
import vllm
import yaml

venv = pathlib.Path(sys.argv[1]).resolve()
assert platform.machine().lower() in {"x86_64", "amd64"}, platform.machine()
assert vllm.__version__ == "0.26.0", f"expected image vLLM 0.26.0, found {vllm.__version__}"
assert torch.cuda.is_available(), "CUDA is not available"
assert torch.cuda.device_count() > 0, "no CUDA GPUs are visible"
assert torch.cuda.is_bf16_supported(), "visible CUDA GPU does not support bfloat16"
for module in (torch, vllm):
    module_path = pathlib.Path(module.__file__).resolve()
    if module_path.is_relative_to(venv):
        raise AssertionError(f"{module.__name__} must come from the vLLM image, not {venv}")
print(
    "Figure 6 runtime verified:",
    f"python={platform.python_version()}",
    f"torch={torch.__version__}",
    f"vllm={vllm.__version__}",
    f"gpus={torch.cuda.device_count()}",
    f"openai={openai.__version__}",
    f"pyyaml={yaml.__version__}",
)
PY

if ! command -v vllm >/dev/null 2>&1; then
    echo "ERROR: the vLLM image's vllm console command is not on PATH." >&2
    exit 2
fi

echo "Figure 6 environment ready at $FIGURE6_VENV"
