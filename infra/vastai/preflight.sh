#!/usr/bin/env bash
# Validate a Vast GPU host and the text-only Qwen model-loading stack.
# Downloads model config metadata only; it does not download weights or run an
# experiment. Override EXPECTED_GPUS for a deliberately smaller smoke host.
set -euo pipefail

EXPECTED_GPUS="${EXPECTED_GPUS:-8}"
MIN_GPU_MEMORY_GB="${MIN_GPU_MEMORY_GB:-80}"

case "$EXPECTED_GPUS" in
    ''|*[!0-9]*) echo "ERROR: EXPECTED_GPUS must be a positive integer." >&2; exit 2 ;;
esac
if [[ "$EXPECTED_GPUS" -lt 1 ]]; then
    echo "ERROR: EXPECTED_GPUS must be at least 1." >&2
    exit 2
fi

export EXPECTED_GPUS MIN_GPU_MEMORY_GB

for required_tool in git nvidia-smi; do
    if ! command -v "$required_tool" >/dev/null 2>&1; then
        echo "ERROR: preflight requires '$required_tool'." >&2
        exit 2
    fi
done
printf 'repository.commit='
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git rev-parse HEAD
elif [[ -n "${CTM_SOURCE_REVISION:-}" ]]; then
    printf '%s (synced checkout)\n' "$CTM_SOURCE_REVISION"
else
    echo "ERROR: synced checkouts must set CTM_SOURCE_REVISION to the source Git commit." >&2
    exit 2
fi
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

python - <<'PY'
from __future__ import annotations

import importlib
import json
import os
from collections import Counter
from importlib.metadata import PackageNotFoundError, distribution, version


def package_details(name: str) -> str:
    try:
        dist = distribution(name)
    except PackageNotFoundError as exc:
        raise SystemExit(f"ERROR: required distribution {name!r} is not installed") from exc
    details = f"{name}=={dist.version}"
    direct_url = dist.read_text("direct_url.json")
    if direct_url:
        metadata = json.loads(direct_url)
        commit = metadata.get("vcs_info", {}).get("commit_id")
        if commit:
            details += f" commit={commit}"
    return details


for package in ("torch", "transformers", "vllm", "peft"):
    print(package_details(package))

torch = importlib.import_module("torch")
importlib.import_module("transformers")
importlib.import_module("vllm")
importlib.import_module("peft")

print(f"torch.version.cuda={torch.version.cuda}")
print(f"cudnn.version={torch.backends.cudnn.version()}")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: torch.cuda.is_available() is false")

expected_gpus = int(os.environ["EXPECTED_GPUS"])
minimum_memory_gb = float(os.environ["MIN_GPU_MEMORY_GB"])
visible_gpus = torch.cuda.device_count()
print(f"visible_gpus={visible_gpus}")
if visible_gpus < expected_gpus:
    raise SystemExit(f"ERROR: expected at least {expected_gpus} visible GPUs, found {visible_gpus}")
for index in range(visible_gpus):
    props = torch.cuda.get_device_properties(index)
    memory_gb = props.total_memory / 1_000_000_000
    print(f"gpu[{index}]={props.name!r} memory={memory_gb:.1f} GB")
    if memory_gb < minimum_memory_gb:
        raise SystemExit(
            f"ERROR: gpu[{index}] has {memory_gb:.1f} GB; require at least {minimum_memory_gb:.1f} GB"
        )

# Transformers checks both importability and its minimum FLA version (>=0.2.2).
from transformers.utils.import_utils import (
    is_causal_conv1d_available,
    is_flash_linear_attention_available,
)

kernel_checks = (
    ("flash-linear-attention", "fla", is_flash_linear_attention_available()),
    ("causal-conv1d", "causal_conv1d", is_causal_conv1d_available()),
)
kernel_errors: list[str] = []
for package, module, transformers_available in kernel_checks:
    try:
        package_version = version(package)
        importlib.import_module(module)
        import_ok = True
    except (ImportError, OSError, PackageNotFoundError) as exc:
        package_version = "missing/unusable"
        import_ok = False
        kernel_errors.append(f"{package} ({module}): {exc}")
    print(
        f"{package}=={package_version} import={module} ok={import_ok} "
        f"transformers_available={transformers_available}"
    )
    if not transformers_available and import_ok:
        kernel_errors.append(f"Transformers does not accept the installed {package} version/CUDA state")

if kernel_errors:
    print("ERROR: Qwen3.5 would use the slow Torch DeltaNet fallback:")
    for error in kernel_errors:
        print(f"  - {error}")
    raise SystemExit(1)

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    from fla.modules import FusedRMSNormGated
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
except (ImportError, OSError) as exc:
    raise SystemExit(f"ERROR: a Qwen3.5 fast-path symbol is not importable: {exc}") from exc

fast_symbols = (
    causal_conv1d_fn,
    causal_conv1d_update,
    FusedRMSNormGated,
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)
if not all(fast_symbols):
    raise SystemExit("ERROR: a Qwen3.5 fast-path symbol resolved to None")

qwen35_modeling = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
print(f"qwen3_5.is_fast_path_available={qwen35_modeling.is_fast_path_available}")
if not qwen35_modeling.is_fast_path_available:
    raise SystemExit("ERROR: Transformers reports that the Qwen3.5 fast path is unavailable")

from transformers import AutoConfig, AutoModelForCausalLM

models = {
    "Qwen/Qwen3.5-9B": ("qwen3_5", "Qwen3_5ForCausalLM"),
    "Qwen/Qwen3-8B": ("qwen3", "Qwen3ForCausalLM"),
}
for model_id, (expected_model_type, expected_class) in models.items():
    config = AutoConfig.from_pretrained(model_id)
    mapped_class = AutoModelForCausalLM._model_mapping[type(config)]
    text_config = getattr(config, "text_config", config)
    print(f"model={model_id}")
    print(
        f"  config={type(config).__name__} model_type={config.model_type} "
        f"declared_architectures={getattr(config, 'architectures', None)}"
    )
    print(
        f"  AutoModelForCausalLM={mapped_class.__name__} "
        f"layers={getattr(text_config, 'num_hidden_layers', None)}"
    )
    if config.model_type != expected_model_type or mapped_class.__name__ != expected_class:
        raise SystemExit(
            f"ERROR: unexpected text-only compatibility mapping for {model_id}: "
            f"{config.model_type=} {mapped_class.__name__=}"
        )
    layer_types = getattr(text_config, "layer_types", None)
    if model_id == "Qwen/Qwen3.5-9B":
        counts = Counter(layer_types or ())
        print(f"  layer_types={dict(counts)}")
        if counts != {"linear_attention": 24, "full_attention": 8}:
            raise SystemExit(f"ERROR: unexpected Qwen3.5 hybrid layer layout: {dict(counts)}")

print("Preflight passed: stack, GPUs, fast kernels, and text-only model mappings are compatible.")
PY
