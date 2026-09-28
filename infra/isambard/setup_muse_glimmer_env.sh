#!/usr/bin/env bash
# Build the isolated Muse Glimmer runtime from one exact AArch64 vLLM wheel.
# Must run inside a GPU allocation so uv resolves the CUDA-enabled torch stack.
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "ERROR: setup_muse_glimmer_env.sh must run inside a Slurm GPU allocation." >&2
    exit 2
fi

cd "$(dirname "$0")/../.."

ctm_muse_python_version="3.12"
ctm_muse_venv=".venv-muse"
ctm_muse_vllm_commit="8c2bbe00d58a930c6c09a80495728b26b79d9200"
ctm_muse_vllm_version="0.26.1rc1.dev1136+g8c2bbe00d"
ctm_muse_vllm_url="https://wheels.vllm.ai/${ctm_muse_vllm_commit}/vllm-0.26.1rc1.dev1136%2Bg8c2bbe00d-cp38-abi3-manylinux_2_28_aarch64.whl"
ctm_muse_vllm_sha256="9b26de5f7bf0f2b7c7b722d6368d7fe18a211744ccad4598c468dfe86fb15e02"
ctm_muse_transformers="5.15.1"
ctm_muse_cookbook="0.4.3"
ctm_muse_mcq_bias_revision="1df2ea1ed8a1eeaf6ec5088c066db7c8c1049119"

if [[ -n "${SCRATCHDIR:-}" ]]; then
    export HF_HOME="${HF_HOME:-$SCRATCHDIR/ctm/huggingface}"
    export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCHDIR/ctm/uv-cache-muse}"
    export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
    mkdir -p "$HF_HOME" "$UV_CACHE_DIR"
fi
export UV_CONCURRENT_BUILDS="${UV_CONCURRENT_BUILDS:-1}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"
export UV_CONCURRENT_INSTALLS="${UV_CONCURRENT_INSTALLS:-1}"

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is required; run infra/isambard/setup_env.sh first." >&2
    exit 2
fi
if [[ ! -x "$ctm_muse_venv/bin/python" ]]; then
    uv venv "$ctm_muse_venv" --python "$ctm_muse_python_version"
fi

ctm_muse_python="$(pwd -P)/$ctm_muse_venv/bin/python"
ctm_muse_vllm_requirement="${ctm_muse_vllm_url}#sha256=${ctm_muse_vllm_sha256}"

# The commit wheel pins torch/kernel dependencies and contains native AArch64
# CUDA binaries. The FlashInfer index is an official dependency source named
# by this vLLM revision's CUDA requirements.  vLLM declares InstantTensor as a
# base requirement even though it is imported only for the explicit
# ``load_format=instanttensor`` path.  InstantTensor publishes no AArch64 wheel
# and the bare Isambard compute image intentionally has no compiler toolchain.
# Install every requirement embedded in the exact vLLM wheel except that one
# unused loader, then fail closed on any other missing/incompatible dependency
# below.  The production preflight exercises the default safetensors loader.
uv pip install --python "$ctm_muse_python" \
    "$ctm_muse_vllm_requirement" \
    --no-deps \
    --torch-backend=auto \
    --extra-index-url https://flashinfer.ai/whl/
mapfile -t ctm_muse_vllm_dependencies < <(
    "$ctm_muse_python" - <<'PY'
import importlib.metadata
import re

for raw in importlib.metadata.requires("vllm") or ():
    name = re.split(r"[ (<>=!~;\[]", raw, maxsplit=1)[0].lower().replace("_", "-")
    if name != "instanttensor":
        print(raw)
PY
)
if [[ "${#ctm_muse_vllm_dependencies[@]}" -eq 0 ]]; then
    echo "ERROR: exact vLLM wheel exposed no dependency metadata." >&2
    exit 2
fi
uv pip install --python "$ctm_muse_python" \
    "${ctm_muse_vllm_dependencies[@]}" \
    --torch-backend=auto \
    --extra-index-url https://flashinfer.ai/whl/
uv pip install --python "$ctm_muse_python" \
    -r infra/isambard/muse-runtime-requirements.txt \
    --torch-backend=auto \
    --extra-index-url https://flashinfer.ai/whl/

# CTM uses a small stable subset of cookbook data containers/math. Cookbook
# 0.4.3 conservatively declares Transformers <=5.5.4, before Muse support
# shipped. Install its exact already-validated source without allowing that
# metadata cap to downgrade the pinned Muse implementation; the exhaustive
# dependency audit below permits exactly this one declared mismatch and the
# import/preflight gates exercise the used API surface.
uv pip install --python "$ctm_muse_python" --no-deps "tinker-cookbook==${ctm_muse_cookbook}"
uv pip install --python "$ctm_muse_python" --no-deps \
    "mcq-bias @ git+https://github.com/sohaibimran7/mcq-bias@${ctm_muse_mcq_bias_revision}"
uv pip install --python "$ctm_muse_python" --no-deps -e .

# The exact commit wheel links one stable extension against the vendored CUDA
# 13 runtime.  Isambard does not automatically add Python package libraries to
# the ELF search path, so every Muse job sources the same fail-closed helper.
source infra/isambard/muse_glimmer_runtime_env.sh

CTM_MUSE_EXPECTED_TRANSFORMERS="$ctm_muse_transformers" \
CTM_MUSE_EXPECTED_VLLM_VERSION="$ctm_muse_vllm_version" \
"$ctm_muse_python" - <<'PY'
import importlib.metadata
import os

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

allowed = []
errors = []
installed = {
    canonicalize_name(distribution.metadata["Name"]): distribution.version
    for distribution in importlib.metadata.distributions()
    if distribution.metadata.get("Name")
}
for distribution in importlib.metadata.distributions():
    source = canonicalize_name(distribution.metadata.get("Name", ""))
    for raw_requirement in distribution.requires or []:
        requirement = Requirement(raw_requirement)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        target = canonicalize_name(requirement.name)
        version = installed.get(target)
        if version is not None and (not requirement.specifier or requirement.specifier.contains(version, prereleases=True)):
            continue
        expected_exception = (
            source == "tinker-cookbook"
            and distribution.version == "0.4.3"
            and target == "transformers"
            and version == os.environ["CTM_MUSE_EXPECTED_TRANSFORMERS"]
            and "<=5.5.4" in str(requirement.specifier)
        )
        expected_unused_loader = (
            source == "vllm"
            and distribution.version == os.environ["CTM_MUSE_EXPECTED_VLLM_VERSION"]
            and target == "instanttensor"
            and version is None
            and ">=0.1.9" in str(requirement.specifier)
        )
        record = f"{source} {distribution.version}: {raw_requirement!r}; installed={version!r}"
        if expected_exception or expected_unused_loader:
            allowed.append(record)
        else:
            errors.append(record)
if len(allowed) != 2 or errors:
    raise SystemExit("Muse dependency audit failed:\n" + "\n".join([*allowed, *errors]))

import torch
import transformers
import vllm
from tinker import types  # noqa: F401
from tinker_cookbook.completers import TokensWithLogprobs  # noqa: F401
from tinker_cookbook.rl.data_processing import trajectory_to_data  # noqa: F401
from tinker_cookbook.rl.metrics import discounted_future_sum_vectorized  # noqa: F401
from vllm import LLM, SamplingParams  # noqa: F401
from vllm.model_executor.models.muse_glimmer import MuseGlimmerForCausalLM

assert torch.cuda.is_available(), "CUDA unavailable inside the Muse setup allocation"
assert transformers.__version__ == os.environ["CTM_MUSE_EXPECTED_TRANSFORMERS"]
assert vllm.__version__ == os.environ["CTM_MUSE_EXPECTED_VLLM_VERSION"]
mapping = MuseGlimmerForCausalLM.get_mm_mapping(object.__new__(MuseGlimmerForCausalLM))
assert mapping.language_model == ["model"], mapping
print(
    "Muse runtime imports passed:",
    "torch", torch.__version__,
    "transformers", transformers.__version__,
    "vllm", vllm.__version__,
    "device", torch.cuda.get_device_name(0),
)
print("Accepted exhaustive dependency exceptions:", *allowed, sep="\n- ")
PY

ctm_muse_receipt_dir="artifacts/muse-glimmer-runtime-20260824"
mkdir -p "$ctm_muse_receipt_dir"
CTM_MUSE_RECEIPT_DIR="$ctm_muse_receipt_dir" \
CTM_MUSE_VLLM_COMMIT="$ctm_muse_vllm_commit" \
CTM_MUSE_VLLM_VERSION="$ctm_muse_vllm_version" \
CTM_MUSE_VLLM_URL="$ctm_muse_vllm_url" \
CTM_MUSE_VLLM_SHA256="$ctm_muse_vllm_sha256" \
CTM_MUSE_EXPECTED_VLLM_VERSION="$ctm_muse_vllm_version" \
CTM_MUSE_TRANSFORMERS="$ctm_muse_transformers" \
"$ctm_muse_python" - <<'PY'
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

root = Path(os.environ["CTM_MUSE_RECEIPT_DIR"]).resolve()
freeze = subprocess.run(
    ["uv", "pip", "freeze", "--python", os.sys.executable],
    check=True,
    capture_output=True,
    text=True,
).stdout
freeze_path = root / "pip-freeze.txt"
if freeze_path.exists() and freeze_path.read_text(encoding="utf-8") != freeze:
    raise SystemExit(f"refusing to overwrite a different Muse freeze: {freeze_path}")
freeze_path.write_text(freeze, encoding="utf-8")
document = {
    "schema": "muse-glimmer-isambard-runtime-v1",
    "platform": platform.platform(),
    "python": platform.python_version(),
    "vllm": {
        "version": importlib.metadata.version("vllm"),
        "commit": os.environ["CTM_MUSE_VLLM_COMMIT"],
        "wheel_url": os.environ["CTM_MUSE_VLLM_URL"],
        "wheel_sha256": os.environ["CTM_MUSE_VLLM_SHA256"],
    },
    "transformers": importlib.metadata.version("transformers"),
    "expected_transformers": os.environ["CTM_MUSE_TRANSFORMERS"],
    "torch": importlib.metadata.version("torch"),
    "pip_freeze": {
        "path": str(freeze_path),
        "sha256": hashlib.sha256(freeze.encode()).hexdigest(),
    },
    "cookbook_metadata_exception": {
        "package": "tinker-cookbook==0.4.3",
        "declared_transformers_upper_bound": "5.5.4",
        "reason": "Muse requires Transformers 5.15.1; CTM-used cookbook imports passed",
    },
    "omitted_optional_loader_dependency": {
        "package": "instanttensor>=0.1.9",
        "declared_by": os.environ["CTM_MUSE_EXPECTED_VLLM_VERSION"],
        "reason": (
            "No AArch64 wheel; vLLM imports it only for explicit "
            "load_format=instanttensor. Production uses and preflights the "
            "default safetensors loader."
        ),
    },
}
payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
receipt = root / "runtime.json"
if receipt.exists() and receipt.read_bytes() != payload:
    raise SystemExit(f"refusing to overwrite a different Muse runtime receipt: {receipt}")
receipt.write_bytes(payload)
print(f"CTM_MUSE_RUNTIME_RECEIPT={receipt}")
PY
