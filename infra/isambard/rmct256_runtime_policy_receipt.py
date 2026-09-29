"""Fail-closed runtime-policy evidence for the RMCT-256 Isambard chain.

This module deliberately has a very small scope.  The segment contract owns
data, checkpoints, and the static target graph; the rollout-worker preflight
owns the non-zero LoRA transport proof.  Here we bind the *runtime policy*
which turns an RMCT rollout request into sampled tokens and an AdamW update:

* the actual :class:`vllm.SamplingParams` object made from the same arguments
  as ``VLLMSampler.sample_batch`` (including all of vLLM's retained defaults);
* the pinned Qwen tokenizer's stop-token IDs and bfloat16 worker resolution;
* the imported runtime stack, versions, attention choice, and Torch numerical
  state; and
* the AdamW flags which CTM intentionally leaves to the installed Torch
  release.

The preflight writes an immutable, secret-free sidecar.  A production segment
re-runs ``validate`` in its Slurm allocation before its irreversible marker,
so a package upgrade, a modified source file, a changed vLLM default, or a
different numerical mode fails closed rather than silently changing the
condition.  ``capture --reference-receipt`` additionally binds a per-segment
sidecar to the real four-GH200 preflight baseline.

No model weights are loaded here.  Loading a tokenizer/config is sufficient to
resolve the Qwen stop IDs, model dtype, and Transformers attention dispatcher;
the separate worker preflight remains the actual engine/LoRA transport test.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import enum
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from infra.isambard import rmct256_convergence_segment_contract as contract

SCHEMA = "rmct256-convergence-isambard-runtime-policy-receipt-v1"
SOURCE_MANIFEST_SCHEMA = "rmct256-convergence-isambard-runtime-source-manifest-v1"
CONDITION = contract.CONDITION
MODEL_REPO = contract.MODEL_REPO
MODEL_REVISION = contract.MODEL_REVISION

# These are the exact keyword arguments in VLLMSampler.sample_batch.  Do not
# add a sampling processor here without also changing the sampler itself: the
# source manifest then guarantees that the two cannot drift unnoticed.
RMCT_SAMPLING_CONSTRUCTOR = {
    "n": 96,
    "max_tokens": 20_480,
    "temperature": 1.0,
    "ignore_eos": False,
    "logprobs": 0,
}

# These are the vLLM 0.21.0 defaults that matter directly to the behaviour
# distribution.  Other public SamplingParams state (including all default
# constraints newly added by vLLM) is still recorded in full and compared on
# every validation.  Keeping the numerical defaults explicit here prevents a
# locally patched package from silently changing the distribution while
# retaining its package-version label.
VLLM_021_DISTRIBUTIONAL_DEFAULTS = {
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
}

# Values which must remain vLLM defaults for RMCT's behaviour distribution.
# They receive an immediate semantic assertion at preflight rather than
# relying solely on a later byte comparison.
REQUIRED_DEFAULT_SAMPLING_FIELDS = (
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
)
REQUIRED_EFFECTIVE_SAMPLING_FIELDS = (
    "n",
    "max_tokens",
    "temperature",
    "stop_token_ids",
    "ignore_eos",
    "logprobs",
    *REQUIRED_DEFAULT_SAMPLING_FIELDS,
)

# This is intentionally a curated *method-defining* stack, not an inventory of
# every transitive dependency.  Every listed module is imported during capture;
# its imported origin must be the expected repository source file, and its
# bytes are hashed.  Thus PYTHONPATH shadowing and edits between preflight and
# production are both caught.
METHOD_SOURCE_MODULES: tuple[tuple[str, str], ...] = (
    ("scripts.run_experiment", "scripts/run_experiment.py"),
    ("scripts.train_rlct", "scripts/train_rlct.py"),
    ("ctm.core.config", "ctm/core/config.py"),
    ("ctm.settings.runtime", "ctm/settings/runtime.py"),
    ("ctm.training.rl", "ctm/training/rl.py"),
    ("ctm.training.resume_state", "ctm/training/resume_state.py"),
    ("ctm.backends.cli", "ctm/backends/cli.py"),
    ("ctm.backends.renderers", "ctm/backends/renderers.py"),
    ("ctm.backends.local.losses", "ctm/backends/local/losses.py"),
    ("ctm.backends.local.engine", "ctm/backends/local/engine.py"),
    ("ctm.backends.local.vllm_sampler", "ctm/backends/local/vllm_sampler.py"),
    ("ctm.backends.local.rollout_workers", "ctm/backends/local/rollout_workers.py"),
    ("ctm.backends.local.replicated", "ctm/backends/local/replicated.py"),
    ("ctm.backends.local.qwen35_vllm_compat", "ctm/backends/local/qwen35_vllm_compat.py"),
    ("ctm_data.adapters.mcq_bias.data", "ctm_data/adapters/mcq_bias/data.py"),
    ("ctm_data.adapters.mcq_bias.setting", "ctm_data/adapters/mcq_bias/setting.py"),
    ("experiments.rmct_256_convergence.plan", "experiments/rmct_256_convergence/plan.py"),
    (
        "experiments.rmct_paper_vast_dense_models.stage1.onpolicy_target_attestation",
        "experiments/rmct_paper_vast_dense_models/stage1/onpolicy_target_attestation.py",
    ),
    ("infra.isambard.rmct256_convergence_segment_contract", "infra/isambard/rmct256_convergence_segment_contract.py"),
    ("infra.isambard.rmct256_runtime_policy_receipt", "infra/isambard/rmct256_runtime_policy_receipt.py"),
)

_FORBIDDEN_KEY = re.compile(r"(?:api[_-]?key|secret|token|password|credential)", re.IGNORECASE)


class RuntimePolicyError(contract.ContractError):
    """The runtime policy cannot be proven to be the preflight condition."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular_file(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimePolicyError(f"{label} must be a regular file: {path}")


def _root(root: str | Path) -> Path:
    path = Path(root).resolve()
    if path.is_symlink() or not path.is_dir():
        raise RuntimePolicyError(f"repository root must be a regular directory: {path}")
    return path


def _under_root(root: Path, path: str | Path) -> Path:
    raw = Path(path)
    resolved = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimePolicyError(f"path escapes repository root: {resolved}") from exc
    return resolved


def _file_identity(path: Path, *, label: str) -> dict[str, Any]:
    _require_regular_file(path, label=label)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _file_sha256(path)}


def _canonical_value(value: Any, *, label: str) -> Any:
    """Make runtime state JSON-safe without falling back to unstable repr()."""

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimePolicyError(f"{label} is not finite")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, enum.Enum):
        return {
            "enum": f"{type(value).__module__}.{type(value).__qualname__}",
            "name": value.name,
            "value": _canonical_value(value.value, label=f"{label}.value"),
        }
    # torch.dtype, torch.device, and similar stable value objects are not
    # Enums on every supported release.  Their textual spelling is part of the
    # public framework contract; reject arbitrary object reprs with addresses.
    module = type(value).__module__
    if module.startswith("torch") and type(value).__name__ in {"dtype", "device", "layout"}:
        return str(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise RuntimePolicyError(f"{label} has a non-string mapping key")
            result[key] = _canonical_value(value[key], label=f"{label}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item, label=f"{label}[]") for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_canonical_value(item, label=f"{label}[]") for item in value]
        return sorted(normalized, key=lambda item: _canonical_json_bytes(item))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _canonical_value(dataclasses.asdict(value), label=label)
    raise RuntimePolicyError(f"{label} has unsupported runtime type {type(value).__module__}.{type(value).__qualname__}")


def _assert_no_secret_keys(value: Any, *, label: str = "runtime policy receipt") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str) and _FORBIDDEN_KEY.search(key):
                raise RuntimePolicyError(f"{label} contains a credential-like key: {key}")
            _assert_no_secret_keys(nested, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _assert_no_secret_keys(nested, label=f"{label}[{index}]")


def _sampling_attributes(params: Any) -> dict[str, Any]:
    """Read every stored SamplingParams field across dataclass/slots releases."""

    raw: dict[str, Any] = {}
    try:
        raw.update(vars(params))
    except TypeError:
        pass
    if not raw and dataclasses.is_dataclass(params):
        raw.update({field.name: getattr(params, field.name) for field in dataclasses.fields(params)})
    if not raw:
        for cls in type(params).mro():
            slots = getattr(cls, "__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for name in slots:
                if isinstance(name, str) and name not in {"__weakref__", "__dict__"} and hasattr(params, name):
                    raw[name] = getattr(params, name)
    if not raw:
        raise RuntimePolicyError("could not enumerate public vLLM SamplingParams state")
    result: dict[str, Any] = {}
    for name in sorted(raw):
        if not isinstance(name, str) or name.startswith("__"):
            continue
        result[name] = _canonical_value(raw[name], label=f"SamplingParams.{name}")
    if not result:
        raise RuntimePolicyError("vLLM SamplingParams has no attested fields")
    return result


def _token_ids(tokenizer: Any) -> list[int]:
    value = getattr(tokenizer, "eos_token_id", None)
    if isinstance(value, int) and not isinstance(value, bool):
        result = [value]
    elif isinstance(value, (list, tuple)):
        result = [item for item in value if isinstance(item, int) and not isinstance(item, bool)]
        if len(result) != len(value):
            raise RuntimePolicyError("tokenizer eos_token_id contains a non-integer entry")
    elif value is None:
        result = []
    else:
        raise RuntimePolicyError(f"tokenizer eos_token_id has unsupported type {type(value).__name__}")
    if len(set(result)) != len(result):
        raise RuntimePolicyError("tokenizer eos_token_id contains duplicate stop IDs")
    if any(item < 0 for item in result):
        raise RuntimePolicyError("tokenizer eos_token_id contains a negative stop ID")
    return result


def _sampling_params_record(sampling_params_cls: type[Any], *, stop_token_ids: Sequence[int]) -> dict[str, Any]:
    """Build and prove the exact RMCT behaviour-distribution parameters."""

    baseline_params = sampling_params_cls()
    effective_params = sampling_params_cls(
        **RMCT_SAMPLING_CONSTRUCTOR,
        stop_token_ids=list(stop_token_ids) or None,
    )
    baseline = _sampling_attributes(baseline_params)
    effective = _sampling_attributes(effective_params)
    if set(baseline) != set(effective):
        raise RuntimePolicyError("vLLM SamplingParams changed its stored field set between default and RMCT construction")
    for field in REQUIRED_EFFECTIVE_SAMPLING_FIELDS:
        if field not in effective:
            raise RuntimePolicyError(f"vLLM SamplingParams is missing required field {field!r}")
    expected = {
        **RMCT_SAMPLING_CONSTRUCTOR,
        "stop_token_ids": list(stop_token_ids) or None,
    }
    for field, required in expected.items():
        actual = effective[field]
        if actual != required:
            raise RuntimePolicyError(f"effective vLLM SamplingParams.{field} differs from RMCT request: expected {required!r}, got {actual!r}")
    for field in REQUIRED_DEFAULT_SAMPLING_FIELDS:
        expected_default = VLLM_021_DISTRIBUTIONAL_DEFAULTS[field]
        if baseline[field] != expected_default:
            raise RuntimePolicyError(f"vLLM 0.21 SamplingParams.{field} default must be {expected_default!r}, got {baseline[field]!r}")
        if effective[field] != baseline[field]:
            raise RuntimePolicyError(f"RMCT must retain vLLM's default SamplingParams.{field}, got default={baseline[field]!r}, effective={effective[field]!r}")
    changed = sorted(name for name in baseline if baseline[name] != effective[name])
    return {
        "constructor": {**RMCT_SAMPLING_CONSTRUCTOR, "stop_token_ids": list(stop_token_ids) or None},
        "renderer_stop_token_ids": list(stop_token_ids),
        "required_default_fields": list(REQUIRED_DEFAULT_SAMPLING_FIELDS),
        "baseline_defaults": baseline,
        "effective": effective,
        "changed_from_default": changed,
        "effective_sha256": _sha256(effective),
    }


def _distribution_version(distribution: str, module: ModuleType) -> dict[str, str | None]:
    try:
        installed = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimePolicyError(f"required distribution is not installed: {distribution}") from exc
    module_version = getattr(module, "__version__", None)
    if module_version is not None and not isinstance(module_version, str):
        raise RuntimePolicyError(f"{distribution} module __version__ is not a string")
    return {"distribution": installed, "module": module_version}


def _release_component(version: str | None, *, label: str) -> str | None:
    """Return a package's upstream release while retaining its full label.

    The official Isambard Arm vLLM wheel is allowed to expose a local build
    suffix such as ``0.21.0+cu129``.  The scientific condition pins the
    upstream vLLM release (``0.21.0``), while the full distribution/module
    labels remain in the receipt and therefore still have to match exactly on
    every production segment.
    """

    if version is None:
        return None
    if not version:
        raise RuntimePolicyError(f"{label} version is empty")
    return version.split("+", 1)[0]


def _runtime_modules() -> dict[str, ModuleType]:
    try:
        return {
            "vllm": importlib.import_module("vllm"),
            "torch": importlib.import_module("torch"),
            "transformers": importlib.import_module("transformers"),
            "peft": importlib.import_module("peft"),
        }
    except ImportError as exc:
        raise RuntimePolicyError(f"required RMCT runtime module is unavailable: {exc.name}") from exc


def _runtime_versions(modules: Mapping[str, ModuleType], *, runtime: Mapping[str, Any]) -> dict[str, Any]:
    vllm = _distribution_version("vllm", modules["vllm"])
    transformers = _distribution_version("transformers", modules["transformers"])
    torch = _distribution_version("torch", modules["torch"])
    peft = _distribution_version("peft", modules["peft"])
    expected_vllm = runtime.get("vllm_version")
    expected_transformers = runtime.get("transformers_version")
    expected_cuda = runtime.get("vllm_cuda")
    if not isinstance(expected_vllm, str) or not isinstance(expected_transformers, str) or not isinstance(expected_cuda, str):
        raise RuntimePolicyError("compiled plan has malformed Isambard runtime versions")
    vllm_distribution_release = _release_component(vllm["distribution"], label="vLLM distribution")
    vllm_module_release = _release_component(vllm["module"], label="vLLM module")
    if vllm_distribution_release != expected_vllm or vllm_module_release not in {None, expected_vllm}:
        raise RuntimePolicyError(f"vLLM version must be {expected_vllm}, got distribution={vllm['distribution']!r}, module={vllm['module']!r}")
    if transformers["distribution"] != expected_transformers or transformers["module"] not in {None, expected_transformers}:
        raise RuntimePolicyError(f"Transformers version must be {expected_transformers}, got distribution={transformers['distribution']!r}, module={transformers['module']!r}")
    torch_module = modules["torch"]
    cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
    if not isinstance(cuda_version, str) or cuda_version != expected_cuda:
        raise RuntimePolicyError(f"Torch CUDA build must be {expected_cuda!r}, got {cuda_version!r}")
    return {
        "vllm": vllm,
        "torch": torch,
        "transformers": transformers,
        "peft": peft,
        "torch_cuda_build": cuda_version,
        "vllm_distribution_release": vllm_distribution_release,
        "vllm_module_release": vllm_module_release,
    }


def _dtype_name(value: Any) -> str:
    spelling = str(value).strip().lower().replace("torch.", "")
    if spelling == "bfloat16":
        return spelling
    return spelling


def _load_qwen_config_and_tokenizer(transformers: ModuleType) -> tuple[Any, Any]:
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimePolicyError("runtime-policy capture requires HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1")
    auto_config = getattr(transformers, "AutoConfig", None)
    auto_tokenizer = getattr(transformers, "AutoTokenizer", None)
    if auto_config is None or auto_tokenizer is None:
        raise RuntimePolicyError("Transformers lacks AutoConfig or AutoTokenizer")
    common = {"revision": MODEL_REVISION, "local_files_only": True, "trust_remote_code": False}
    try:
        config = auto_config.from_pretrained(MODEL_REPO, **common)
        tokenizer = auto_tokenizer.from_pretrained(MODEL_REPO, **common)
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimePolicyError(f"could not resolve the pinned local Qwen config/tokenizer: {exc}") from exc
    return config, tokenizer


def _attention_value(config: Any) -> str | None:
    for name in ("_attn_implementation", "attn_implementation", "_attn_implementation_internal"):
        value = getattr(config, name, None)
        if isinstance(value, str) and value and value.lower() not in {"auto", "none"}:
            return value
    return None


def _resolve_attention_implementation(transformers: ModuleType, torch_module: ModuleType, config: Any) -> dict[str, Any]:
    """Resolve Transformers' attention dispatcher without allocating 9B weights."""

    initial = _attention_value(config)
    if initial is not None:
        return {"implementation": initial, "resolution": "AutoConfig"}

    auto_model = getattr(transformers, "AutoModelForCausalLM", None)
    mapping = getattr(auto_model, "_model_mapping", None)
    if mapping is None:
        raise RuntimePolicyError("cannot resolve attention implementation: Transformers exposes no AutoModel mapping")
    try:
        model_cls = mapping[type(config)]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimePolicyError(f"cannot resolve attention implementation for config class {type(config).__name__}") from exc
    resolver = getattr(model_cls, "_autoset_attn_implementation", None)
    if not callable(resolver):
        raise RuntimePolicyError(f"{model_cls.__name__} has no attention auto-resolution hook")
    kwargs: dict[str, Any] = {}
    try:
        signature = inspect.signature(resolver)
        if "torch_dtype" in signature.parameters:
            kwargs["torch_dtype"] = torch_module.bfloat16
        if "check_device_map" in signature.parameters:
            kwargs["check_device_map"] = False
        if "hard_check_only" in signature.parameters:
            kwargs["hard_check_only"] = False
    except (TypeError, ValueError):
        # A C-extension/described callable is unusual here.  Calling it with
        # the minimum model-config argument is safer than inventing kwargs.
        pass
    candidate = copy.deepcopy(config)
    try:
        returned = resolver(candidate, **kwargs)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise RuntimePolicyError(f"Transformers attention auto-resolution failed: {type(exc).__name__}: {exc}") from exc
    resolved_config = candidate if returned is None else returned
    resolved = _attention_value(resolved_config)
    if resolved is None:
        raise RuntimePolicyError("Transformers attention auto-resolution left the implementation unspecified")
    return {
        "implementation": resolved,
        "resolution": "AutoModelForCausalLM._autoset_attn_implementation",
        "model_class": f"{model_cls.__module__}.{model_cls.__qualname__}",
    }


def _worker_dtype_and_attention(*, transformers: ModuleType, torch_module: ModuleType, config: Any, tokenizer: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    declared_dtype = _dtype_name(getattr(config, "torch_dtype", None))
    if declared_dtype != "bfloat16":
        raise RuntimePolicyError(f"pinned Qwen config must resolve to bfloat16, got {declared_dtype!r}")
    stop_ids = _token_ids(tokenizer)
    attention = _resolve_attention_implementation(transformers, torch_module, config)
    worker_dtype = {
        "model_config_torch_dtype": declared_dtype,
        # ctm.backends.cli._vllm_options now sends this explicit value to both
        # the coordinator and every dedicated rollout worker.  Its source hash
        # is in the manifest below, so neither a future `auto` default nor a
        # worker-only dtype omission can alter the rollout distribution.
        "vllm_dtype_argument": "bfloat16",
        "resolved_worker_dtype": "bfloat16",
        "assertion": "explicit_local_dtype_bfloat16",
        "renderer_stop_token_ids": stop_ids,
    }
    return worker_dtype, attention


def _cuda_devices(torch_module: ModuleType) -> list[dict[str, Any]]:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not callable(getattr(cuda, "is_available", None)) or not cuda.is_available():
        raise RuntimePolicyError("Torch CUDA is unavailable in the runtime-policy allocation")
    count = cuda.device_count()
    if count != 4:
        raise RuntimePolicyError(f"runtime-policy attestation requires exactly four visible GH200 GPUs, got {count}")
    devices: list[dict[str, Any]] = []
    for index in range(count):
        try:
            properties = cuda.get_device_properties(index)
            name = str(getattr(properties, "name", ""))
            capability = cuda.get_device_capability(index)
            total_memory = int(getattr(properties, "total_memory", 0))
        except (AttributeError, RuntimeError) as exc:
            raise RuntimePolicyError(f"could not inspect CUDA device {index}: {exc}") from exc
        if "GH200" not in name.upper():
            raise RuntimePolicyError(f"runtime-policy attestation requires a GH200 allocation; device {index} is {name!r}")
        if not isinstance(capability, tuple) or len(capability) != 2 or any(not isinstance(item, int) for item in capability):
            raise RuntimePolicyError(f"CUDA device {index} has invalid compute capability {capability!r}")
        if total_memory <= 0:
            raise RuntimePolicyError(f"CUDA device {index} has invalid total memory {total_memory!r}")
        devices.append(
            {
                "logical_index": index,
                "name": name,
                "compute_capability": list(capability),
                "total_memory_bytes": total_memory,
            }
        )
    return devices


def _numerics_record(torch_module: ModuleType) -> dict[str, Any]:
    backends = getattr(torch_module, "backends", None)
    cuda_backend = getattr(backends, "cuda", None)
    matmul = getattr(cuda_backend, "matmul", None)
    cudnn = getattr(backends, "cudnn", None)
    required = {
        "cuda_matmul_allow_tf32": getattr(matmul, "allow_tf32", None),
        "cuda_matmul_allow_fp16_reduced_precision_reduction": getattr(matmul, "allow_fp16_reduced_precision_reduction", None),
        "cuda_matmul_allow_bf16_reduced_precision_reduction": getattr(matmul, "allow_bf16_reduced_precision_reduction", None),
        "cudnn_allow_tf32": getattr(cudnn, "allow_tf32", None),
        "cudnn_benchmark": getattr(cudnn, "benchmark", None),
        "float32_matmul_precision": (torch_module.get_float32_matmul_precision() if callable(getattr(torch_module, "get_float32_matmul_precision", None)) else None),
        "deterministic_algorithms": (torch_module.are_deterministic_algorithms_enabled() if callable(getattr(torch_module, "are_deterministic_algorithms_enabled", None)) else None),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise RuntimePolicyError(f"Torch does not expose required numerical runtime state: {missing}")
    return _canonical_value(required, label="torch numerical state")


def _adamw_record(torch_module: ModuleType, args: Mapping[str, Any]) -> dict[str, Any]:
    required = {"lr", "beta1", "beta2", "eps", "weight_decay", "grad_clip_norm", "gradient_accumulation_steps"}
    missing = sorted(required - set(args))
    if missing:
        raise RuntimePolicyError(f"compiled target lacks AdamW argument(s): {missing}")
    parameter = torch_module.nn.Parameter(torch_module.zeros(1))
    optimizer = torch_module.optim.AdamW(
        [parameter],
        lr=args["lr"],
        betas=(args["beta1"], args["beta2"]),
        eps=args["eps"],
        weight_decay=args["weight_decay"],
    )
    defaults = _canonical_value(dict(optimizer.defaults), label="AdamW.defaults")
    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(groups[0], Mapping):
        raise RuntimePolicyError("toy AdamW has no single effective parameter group")
    group = {key: value for key, value in dict(groups[0]).items() if key != "params"}
    effective_group = _canonical_value(group, label="AdamW.param_group")
    expected = {
        "lr": args["lr"],
        "betas": [args["beta1"], args["beta2"]],
        "eps": args["eps"],
        "weight_decay": args["weight_decay"],
    }
    for name, value in expected.items():
        if defaults.get(name) != value or effective_group.get(name) != value:
            raise RuntimePolicyError(f"effective AdamW {name} differs from compiled target")
    remaining_flags = ("amsgrad", "maximize", "foreach", "capturable", "differentiable", "fused")
    missing_flags = [name for name in remaining_flags if name not in defaults or name not in effective_group]
    if missing_flags:
        raise RuntimePolicyError(f"Torch AdamW does not expose required effective flag(s): {missing_flags}")
    if args["gradient_accumulation_steps"] != 1:
        raise RuntimePolicyError("RMCT256 runtime policy requires exactly one gradient accumulation step")
    return {
        "constructor": {
            "lr": args["lr"],
            "betas": [args["beta1"], args["beta2"]],
            "eps": args["eps"],
            "weight_decay": args["weight_decay"],
        },
        "effective_defaults": defaults,
        "effective_param_group": effective_group,
        "remaining_default_flags": {name: effective_group[name] for name in remaining_flags},
        "gradient_accumulation_steps": args["gradient_accumulation_steps"],
        "gradient_clip_norm": args["grad_clip_norm"],
        "gradient_clip_implementation": "torch.nn.utils.clip_grad_norm_",
        "zero_grad_set_to_none": True,
    }


def _method_source_manifest(root: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for module_name, relative_path in METHOD_SOURCE_MODULES:
        expected = _under_root(root, relative_path)
        _require_regular_file(expected, label=f"method-defining source {module_name}")
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise RuntimePolicyError(f"could not import method-defining source module {module_name}: {exc}") from exc
        origin_raw = getattr(module, "__file__", None)
        if not isinstance(origin_raw, str) or not origin_raw:
            raise RuntimePolicyError(f"method-defining module {module_name} has no source-file origin")
        origin = Path(origin_raw).resolve()
        if origin != expected:
            raise RuntimePolicyError(f"method-defining module {module_name} was imported from {origin}, expected {expected}; refusing PYTHONPATH shadowing")
        entries.append(
            {
                "module": module_name,
                "relative_path": relative_path,
                "size_bytes": expected.stat().st_size,
                "sha256": _file_sha256(expected),
            }
        )
    manifest = {"schema": SOURCE_MANIFEST_SCHEMA, "modules": entries}
    return {**manifest, "sha256": _sha256(manifest)}


def _expected_target(root: Path, plan: Path, segment: contract.Segment) -> tuple[dict[str, Any], Mapping[str, Any]]:
    validated = contract.validate_segment_plan(root, plan, segment)
    args = validated.get("args")
    metadata = validated.get("metadata")
    if not isinstance(args, Mapping) or not isinstance(metadata, Mapping):
        raise RuntimePolicyError("segment contract did not return compiled arguments and metadata")
    expected = {
        "model": MODEL_REPO,
        "local_dtype": "bfloat16",
        "local_sampler": "vllm",
        "n_ref_rollouts": 96,
        "n_train_rollouts": 96,
        "n_consistency_rollouts": 96,
        "n_anchor_rollouts": 96,
        "temperature": 1.0,
        "max_new_tokens": 20_480,
        "batch_size": 4,
        "gradient_accumulation_steps": 1,
        "loss_fn": "ppo",
        "local_vllm_gdn_prefill_backend": "triton",
    }
    for key, value in expected.items():
        if args.get(key) != value:
            raise RuntimePolicyError(f"compiled target {key} must be {value!r}, got {args.get(key)!r}")
    runtime = metadata.get("isambard_runtime")
    if not isinstance(runtime, Mapping):
        raise RuntimePolicyError("compiled segment metadata has no isambard_runtime")
    return dict(args), runtime


def _scope(root: Path, plan: Path, segment: contract.Segment, args: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "condition": CONDITION,
        "segment": {
            "global_segment_index": segment.global_index,
            "target": segment.target,
            "run_name": segment.run_name,
            "worker_seed_base": segment.worker_seed_base,
        },
        "plan": _file_identity(plan, label="authored RMCT256 convergence plan"),
        "compiled_target_args_sha256": _sha256(_canonical_value(dict(args), label="compiled target args")),
    }


def build_runtime_policy_document(
    root: Path,
    plan: Path,
    segment: contract.Segment,
    *,
    reference_receipt: Path | None = None,
) -> dict[str, Any]:
    """Collect all deterministic runtime-policy evidence for one allocation."""

    args, runtime = _expected_target(root, plan, segment)
    modules = _runtime_modules()
    config, tokenizer = _load_qwen_config_and_tokenizer(modules["transformers"])
    worker_dtype, attention = _worker_dtype_and_attention(transformers=modules["transformers"], torch_module=modules["torch"], config=config, tokenizer=tokenizer)
    sampling_cls = getattr(modules["vllm"], "SamplingParams", None)
    if not isinstance(sampling_cls, type):
        raise RuntimePolicyError("vLLM exposes no SamplingParams class")
    sampling = _sampling_params_record(sampling_cls, stop_token_ids=worker_dtype["renderer_stop_token_ids"])
    devices = _cuda_devices(modules["torch"])
    payload = {
        "runtime_versions": _runtime_versions(modules, runtime=runtime),
        "cuda_devices": devices,
        "worker_dtype": worker_dtype,
        "attention": attention,
        "torch_numerics": _numerics_record(modules["torch"]),
        "sampling_params": sampling,
        "adamw": _adamw_record(modules["torch"], args),
        "method_source_manifest": _method_source_manifest(root),
    }
    _assert_no_secret_keys(payload)
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "scope": _scope(root, plan, segment, args),
        "runtime_policy": payload,
        "runtime_policy_sha256": _sha256(payload),
    }
    if reference_receipt is not None:
        reference = validate_runtime_policy_receipt(root, reference_receipt)
        reference_hash = reference.get("runtime_policy_sha256")
        if not isinstance(reference_hash, str) or len(reference_hash) != 64:
            raise RuntimePolicyError("reference runtime-policy receipt has no valid payload hash")
        if reference.get("runtime_policy") != payload:
            raise RuntimePolicyError("current runtime policy differs from the immutable four-GH200 preflight receipt")
        document["reference_preflight_receipt"] = _file_identity(reference_receipt, label="reference runtime-policy receipt")
    _assert_no_secret_keys(document)
    return document


def _read_document(path: Path) -> dict[str, Any]:
    _require_regular_file(path, label="runtime-policy receipt")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimePolicyError(f"runtime-policy receipt is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimePolicyError("runtime-policy receipt must be a JSON object")
    _assert_no_secret_keys(value)
    if value.get("schema") != SCHEMA:
        raise RuntimePolicyError("runtime-policy receipt has an unexpected schema")
    return value


def _plan_from_scope(root: Path, scope: Mapping[str, Any]) -> Path:
    plan_record = scope.get("plan")
    if not isinstance(plan_record, Mapping) or not isinstance(plan_record.get("path"), str):
        raise RuntimePolicyError("runtime-policy receipt has no plan identity")
    plan = _under_root(root, plan_record["path"])
    current = _file_identity(plan, label="runtime-policy receipt plan")
    if current != dict(plan_record):
        raise RuntimePolicyError("runtime-policy receipt plan changed after capture")
    return plan


def _segment_from_scope(scope: Mapping[str, Any]) -> contract.Segment:
    value = scope.get("segment")
    if not isinstance(value, Mapping):
        raise RuntimePolicyError("runtime-policy receipt has no segment scope")
    index = value.get("global_segment_index")
    if isinstance(index, bool) or not isinstance(index, int):
        raise RuntimePolicyError("runtime-policy receipt has invalid global segment index")
    segment = contract.segment_for_index(index)
    expected = {
        "global_segment_index": segment.global_index,
        "target": segment.target,
        "run_name": segment.run_name,
        "worker_seed_base": segment.worker_seed_base,
    }
    if dict(value) != expected:
        raise RuntimePolicyError("runtime-policy receipt segment scope does not match the static contract")
    return segment


def validate_runtime_policy_receipt(
    root: str | Path,
    receipt: str | Path,
    *,
    expected_segment: contract.Segment | None = None,
    _seen: set[Path] | None = None,
) -> dict[str, Any]:
    """Recompute and compare a receipt in the current Slurm allocation."""

    repository = _root(root)
    path = _under_root(repository, receipt)
    resolved = path.resolve()
    seen = set() if _seen is None else _seen
    if resolved in seen:
        raise RuntimePolicyError(f"runtime-policy reference cycle: {resolved}")
    seen.add(resolved)
    document = _read_document(resolved)
    scope = document.get("scope")
    if not isinstance(scope, Mapping) or scope.get("condition") != CONDITION:
        raise RuntimePolicyError("runtime-policy receipt scope has the wrong condition")
    plan = _plan_from_scope(repository, scope)
    segment = _segment_from_scope(scope)
    if expected_segment is not None and segment != expected_segment:
        raise RuntimePolicyError("runtime-policy receipt belongs to another static segment")
    reference = document.get("reference_preflight_receipt")
    reference_path: Path | None = None
    if reference is not None:
        if not isinstance(reference, Mapping) or not isinstance(reference.get("path"), str):
            raise RuntimePolicyError("runtime-policy reference has no file identity")
        reference_path = _under_root(repository, reference["path"])
        if _file_identity(reference_path, label="runtime-policy reference receipt") != dict(reference):
            raise RuntimePolicyError("runtime-policy reference receipt changed after capture")
        reference_document = validate_runtime_policy_receipt(repository, reference_path, _seen=seen)
        if reference_document.get("runtime_policy") != document.get("runtime_policy"):
            raise RuntimePolicyError("runtime-policy receipt differs from its preflight reference")
    current = build_runtime_policy_document(repository, plan, segment, reference_receipt=reference_path)
    if current != document:
        raise RuntimePolicyError("runtime-policy receipt no longer matches the effective runtime")
    if document.get("runtime_policy_sha256") != _sha256(document.get("runtime_policy")):
        raise RuntimePolicyError("runtime-policy payload hash is invalid")
    return document


def capture_runtime_policy_receipt(
    root: str | Path,
    plan: str | Path,
    segment: contract.Segment,
    *,
    output: str | Path,
    reference_receipt: str | Path | None = None,
) -> dict[str, Any]:
    """Write exactly one canonical runtime-policy receipt, or validate it."""

    repository = _root(root)
    plan_path = _under_root(repository, plan)
    _require_regular_file(plan_path, label="authored RMCT256 convergence plan")
    output_path = _under_root(repository, output)
    reference_path = None if reference_receipt is None else _under_root(repository, reference_receipt)
    if output_path.exists() or output_path.is_symlink():
        document = validate_runtime_policy_receipt(repository, output_path, expected_segment=segment)
        return {
            "status": "resumed",
            "receipt": str(output_path),
            "runtime_policy_sha256": document["runtime_policy_sha256"],
        }
    document = build_runtime_policy_document(repository, plan_path, segment, reference_receipt=reference_path)
    payload = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.parent.is_symlink():
        raise RuntimePolicyError(f"runtime-policy receipt parent must not be a symlink: {output_path.parent}")
    try:
        with output_path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        document = validate_runtime_policy_receipt(repository, output_path, expected_segment=segment)
        status = "resumed"
    else:
        status = "written"
    return {"status": status, "receipt": str(output_path), "runtime_policy_sha256": document["runtime_policy_sha256"]}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture", help="write one immutable runtime-policy receipt")
    capture.add_argument("--repo-root", required=True)
    capture.add_argument("--plan", required=True)
    capture.add_argument("--segment-index", required=True, type=int)
    capture.add_argument("--output", required=True)
    capture.add_argument(
        "--reference-receipt",
        help="optional immutable four-GH200 preflight receipt which this segment must match exactly",
    )
    validate = commands.add_parser("validate", help="recompute and validate a receipt")
    validate.add_argument("--repo-root", required=True)
    validate.add_argument("--receipt", required=True)
    validate.add_argument("--segment-index", type=int, help="optional expected static segment index")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "capture":
            segment = contract.segment_for_index(args.segment_index)
            result = capture_runtime_policy_receipt(
                args.repo_root,
                args.plan,
                segment,
                output=args.output,
                reference_receipt=args.reference_receipt,
            )
        else:
            expected = None if args.segment_index is None else contract.segment_for_index(args.segment_index)
            document = validate_runtime_policy_receipt(args.repo_root, args.receipt, expected_segment=expected)
            result = {
                "status": "validated",
                "receipt": str(_under_root(_root(args.repo_root), args.receipt)),
                "runtime_policy_sha256": document["runtime_policy_sha256"],
            }
    except (OSError, RuntimePolicyError, contract.ContractError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())


__all__ = [
    "CONDITION",
    "METHOD_SOURCE_MODULES",
    "RMCT_SAMPLING_CONSTRUCTOR",
    "SCHEMA",
    "SOURCE_MANIFEST_SCHEMA",
    "RuntimePolicyError",
    "build_runtime_policy_document",
    "capture_runtime_policy_receipt",
    "validate_runtime_policy_receipt",
]
