"""Probe HF/PEFT versus vLLM runtime application of a Qwen3.5 LoRA adapter.

The Stage-1 Qwen3.5 runs train through Transformers/PEFT but evaluate through
vLLM.  Qwen3.5 is hybrid: most layers are Gated DeltaNet layers and the
remaining layers use ordinary attention.  A successful LoRA registration in a
vLLM log is useful, but is not enough to establish that both projection
families change inference numerically.

This diagnostic creates three disposable versions of one adapter:

``full``
    The supplied adapter unchanged.
``linear_only``
    The ordinary-attention LoRA-B matrices are zeroed.  Only DeltaNet / linear
    attention LoRA can affect the output.
``self_attn_only``
    The DeltaNet / linear-attention LoRA-B matrices are zeroed.  Only ordinary
    self-attention LoRA can affect the output.

For a small fixed set of frozen prompts it compares base-to-adapter changes in
relative next-token scores.  HF records raw logits; vLLM records logprobs.  By
subtracting a per-prompt reference-token score, both become logit differences
and are directly comparable.  The vLLM request uses the identical pretokenized
prompt IDs that HF receives, so this tests the evaluation boundary rather than
chat-template differences.

The output directory is deliberately single-use.  It contains the disposable
adapter copies, the vLLM server log, and an immutable JSON report.  It never
modifies the supplied checkpoint.

Example (on the ACT gate Vast host)::

    CUDA_VISIBLE_DEVICES=0 python -m experiments.act_repair_gate.runtime_parity \
      --model Qwen/Qwen3.5-9B \
      --adapter logs/.../canonical-train-n200-4000steps/checkpoints/... \
      --data artifacts/act-repair-gate-20260731/data/canonical-train-eval-n200.jsonl \
      --output-dir artifacts/act-repair-gate-20260731/runtime-parity-20260801
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "qwen35-lora-runtime-parity-v1"
VARIANTS = ("full", "linear_only", "self_attn_only")
# Inspect's local-checkpoint route selects the runtime adapter by its resolved
# checkpoint directory, rather than by a short disposable alias.  Probe that
# exact identity too: it distinguishes an OpenAI-server routing problem from a
# lower-level Qwen3.5/vLLM LoRA application problem.
EVALUATOR_PATH_VARIANT = "evaluator_path"
RESULT_VARIANTS = (*VARIANTS, EVALUATOR_PATH_VARIANT)
DEFAULT_LABELS = ("A", "B", "C", "D")
DEFAULT_PROBE_SAMPLES = 8
DEFAULT_TOP_TOKEN_COUNT = 16
DEFAULT_HF_BATCH_SIZE = 2
DEFAULT_VLLM_PORT = 8789
DEFAULT_MAX_MODEL_LEN = 32768
DEFAULT_VLLM_MEMORY_UTILIZATION = 0.90
# The sealed Qwen3.5 parity request contains up to the frozen top-16 IDs plus
# thirteen unique A/B/C/D surface-form IDs (the newline form is shared): 29
# scores at most.  Configure
# vLLM with that exact admission cap rather than its default 20.  This changes
# only request admission; every requested token ID and numerical comparison
# stays unchanged.
DEFAULT_MAX_LOGPROBS = 29
# Request processed (rather than raw) logprobs, then make every operation
# which could otherwise change those scores explicit and neutral.  Scores are
# collected only from the per-request ``allowed_token_ids`` support.  The
# resulting restricted-softmax normalizer cancels when the diagnostic takes
# relative token differences, while the restriction prevents vLLM from
# silently returning a different top-k set than HF selected.
VLLM_LOGPROBS_MODE = "processed_logprobs"
VLLM_SCORE_TRANSPORT = "allowed_token_ids_restricted_softmax"
VLLM_PARITY_SAMPLING = {
    "max_tokens": 1,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
    "min_tokens": 0,
    "ignore_eos": True,
}
# vLLM 0.26.0 exposes precisely these CLI choices.  ``auto`` is its implicit
# default, but is deliberately not an accepted explicit command-line value.
GDN_PREFILL_BACKENDS = ("flashinfer", "triton", "cutedsl")
PARALLEL_VLLM_CACHE_ENVIRONMENT = (
    ("XDG_CACHE_HOME", "xdg-cache"),
    ("TRITON_CACHE_DIR", "triton-cache"),
    ("TORCHINDUCTOR_CACHE_DIR", "torchinductor-cache"),
)


def _parallel_isolated_vllm_plan(
    *,
    requested_result_variants: Sequence[str],
    device_tokens: Sequence[str] | None,
    vllm_port: int,
    isolate_vllm_variants: bool,
    parallel_isolated_vllm_variants: bool,
    enforce_eager: bool,
    max_logprobs: int = DEFAULT_MAX_LOGPROBS,
) -> dict[str, dict[str, int | str]] | None:
    """Validate and deterministically assign the strict four-server fast path.

    Parallel isolation is deliberately an all-or-nothing attestation mode.  A
    subset, repeated device, shared port, or accidental fallback to the normal
    multi-LoRA server would otherwise make a fast-looking report weaker than
    the original sequential diagnostic.  ``CUDA_VISIBLE_DEVICES`` accepts
    logical indices and GPU UUIDs, so device values remain opaque tokens here;
    a comma is rejected because it would expose more than one GPU to one
    supposedly isolated server.
    """

    max_logprobs = _validate_max_logprobs(max_logprobs)
    if not parallel_isolated_vllm_variants:
        if device_tokens is not None:
            raise ValueError(
                "vllm_device_tokens requires parallel_isolated_vllm_variants=True; "
                "refusing to run a requested parallel attestation sequentially"
            )
        return None
    if not isolate_vllm_variants:
        raise ValueError(
            "parallel_isolated_vllm_variants requires isolate_vllm_variants=True; "
            "refusing to weaken the one-server-per-variant diagnostic"
        )
    if not enforce_eager:
        raise ValueError(
            "parallel_isolated_vllm_variants requires enforce_eager=True; "
            "refusing a faster topology with a less controlled numerical comparison"
        )
    if tuple(requested_result_variants) != RESULT_VARIANTS:
        raise ValueError(
            "parallel isolated vLLM parity requires the complete attestation set "
            f"in canonical order: {list(RESULT_VARIANTS)!r}"
        )
    if device_tokens is None or isinstance(device_tokens, (str, bytes)):
        raise ValueError(
            "parallel isolated vLLM parity requires exactly four explicit, unique vLLM device tokens"
        )
    tokens = tuple(device_tokens)
    if len(tokens) != len(RESULT_VARIANTS):
        raise ValueError(
            "parallel isolated vLLM parity requires exactly four explicit, unique vLLM device tokens"
        )
    if any(
        not isinstance(token, str)
        or not token
        or token != token.strip()
        or any(character.isspace() for character in token)
        or "," in token
        for token in tokens
    ):
        raise ValueError(
            "each vLLM device token must be one non-empty CUDA_VISIBLE_DEVICES token without whitespace or commas"
        )
    if len(set(tokens)) != len(tokens):
        raise ValueError("parallel isolated vLLM parity requires four unique device tokens")
    if isinstance(vllm_port, bool) or not isinstance(vllm_port, int) or not 1 <= vllm_port <= 65532:
        raise ValueError("parallel isolated vLLM parity needs a base vLLM port in the inclusive range 1..65532")

    # The explicit lists make the report's execution topology independently
    # auditable, while dict insertion order preserves the canonical attestation
    # order before JSON serialization sorts the final document's keys.
    return {
        variant: {
            "device_token": token,
            "port": vllm_port + index,
            "server_log": f"vllm-server-{variant}.log",
            "max_logprobs": max_logprobs,
            "logprobs_mode": VLLM_LOGPROBS_MODE,
        }
        for index, (variant, token) in enumerate(zip(RESULT_VARIANTS, tokens, strict=True))
    }


def _parallel_isolated_vllm_report_metadata(
    *, root: Path, plan: Mapping[str, Mapping[str, int | str]]
) -> dict[str, dict[str, Any]]:
    """Render the auditable, stable topology record for a parallel report."""

    metadata: dict[str, dict[str, Any]] = {}
    for variant in RESULT_VARIANTS:
        configuration = plan[variant]
        port = configuration["port"]
        device_token = configuration["device_token"]
        log_name = configuration["server_log"]
        max_logprobs = configuration["max_logprobs"]
        logprobs_mode = configuration["logprobs_mode"]
        if (
            not isinstance(port, int)
            or not isinstance(device_token, str)
            or not isinstance(log_name, str)
            or isinstance(max_logprobs, bool)
            or not isinstance(max_logprobs, int)
            or logprobs_mode != VLLM_LOGPROBS_MODE
        ):
            raise TypeError(f"invalid parallel vLLM configuration for {variant!r}")
        max_logprobs = _validate_max_logprobs(max_logprobs)
        metadata[variant] = {
            "device_token": device_token,
            "port": port,
            "server_log": str(root / log_name),
            "max_loras": 1,
            "max_logprobs": max_logprobs,
            "logprobs_mode": logprobs_mode,
            "cache_directories": {
                name: str(path)
                for name, path in _parallel_isolated_vllm_cache_paths(root=root, variant=variant).items()
            },
        }
    return metadata


def _parallel_isolated_vllm_cache_paths(*, root: Path, variant: str) -> dict[str, Path]:
    """Return the three cache locations exclusively assigned to one server."""

    if variant not in RESULT_VARIANTS:
        raise ValueError(f"unknown parallel isolated vLLM variant: {variant!r}")
    variant_root = root / "vllm-server-caches" / variant
    return {
        environment_name: variant_root / directory_name
        for environment_name, directory_name in PARALLEL_VLLM_CACHE_ENVIRONMENT
    }


def _prepare_parallel_isolated_vllm_cache_paths(*, root: Path, variant: str) -> dict[str, Path]:
    """Create regular, per-server cache directories under a fresh output root."""

    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"parallel vLLM cache root is not a regular directory: {root}")
    paths = _parallel_isolated_vllm_cache_paths(root=root, variant=variant)
    variant_root = next(iter(paths.values())).parent
    if variant_root.exists() or variant_root.is_symlink():
        raise FileExistsError(f"refusing to reuse or follow a parallel vLLM cache directory: {variant_root}")
    variant_root.mkdir(parents=True)
    root_resolved = root.resolve()
    for path in paths.values():
        path.mkdir()
        if path.is_symlink() or not path.is_dir() or not path.resolve().is_relative_to(root_resolved):
            raise RuntimeError(f"parallel vLLM cache path is not a regular directory beneath output root: {path}")
    return paths


@dataclass(frozen=True, slots=True)
class Prompt:
    """One exact frozen message list and its external identity."""

    question_id: str
    messages: tuple[dict[str, str], ...]


def _validate_gdn_prefill_backend(backend: str | None) -> str | None:
    """Match the installed vLLM CLI's explicit GDN backend choices."""

    if backend is not None and backend not in GDN_PREFILL_BACKENDS:
        raise ValueError(
            "gdn_prefill_backend must be one of "
            f"{list(GDN_PREFILL_BACKENDS)!r} or None, got {backend!r}"
        )
    return backend


def _validate_max_logprobs(value: int) -> int:
    """Keep parity's server admission cap exactly equal to its protocol bound."""

    if isinstance(value, bool) or not isinstance(value, int) or value != DEFAULT_MAX_LOGPROBS:
        raise ValueError(
            "max_logprobs must be exactly "
            f"{DEFAULT_MAX_LOGPROBS} for the frozen parity token protocol, got {value!r}"
        )
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_message_list(value: Any, *, location: str) -> tuple[dict[str, str], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location}: messages must be a non-empty array")
    messages: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"{location}: message {index} is not an object")
        role, content = item.get("role"), item.get("content")
        if not isinstance(role, str) or not role or not isinstance(content, str) or not content:
            raise ValueError(f"{location}: message {index} must have non-empty role/content strings")
        messages.append({"role": role, "content": content})
    return tuple(messages)


def load_prompts(path: str | Path, *, limit: int, message_field: str = "biased_messages") -> list[Prompt]:
    """Load a deterministic, unique prefix of a frozen canonical JSONL file."""

    source = Path(path).resolve()
    if limit < 1:
        raise ValueError("limit must be positive")
    found: list[Prompt] = []
    seen: set[str] = set()
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"{source}:{line_number}: row is not an object")
            question_id = row.get("question_id")
            if not isinstance(question_id, str) or not question_id:
                raise ValueError(f"{source}:{line_number}: missing question_id")
            if question_id in seen:
                raise ValueError(f"{source}:{line_number}: duplicate question_id {question_id!r}")
            seen.add(question_id)
            found.append(
                Prompt(
                    question_id=question_id,
                    messages=_require_message_list(row.get(message_field), location=f"{source}:{line_number}"),
                )
            )
            if len(found) == limit:
                break
    if len(found) != limit:
        raise ValueError(f"{source}: asked for {limit} prompts but found only {len(found)}")
    return found


def _lora_b_family(key: str) -> str | None:
    """Return the Qwen3.5 projection family for one PEFT LoRA-B tensor."""

    # PEFT has used both ``lora_B.default.weight`` and ``lora_B.weight``
    # spellings.  A is deliberately not touched: B=0 exactly removes a LoRA
    # update while retaining a structurally valid adapter for vLLM packing.
    if ".lora_B." not in key and not key.endswith(".lora_B.weight"):
        return None
    if "linear_attn" in key:
        return "linear_attn"
    if "self_attn" in key:
        return "self_attn"
    return None


def _variant_should_zero(key: str, variant: str) -> bool:
    family = _lora_b_family(key)
    if variant == "full":
        return False
    if variant == "linear_only":
        return family == "self_attn"
    if variant == "self_attn_only":
        return family == "linear_attn"
    raise ValueError(f"unknown adapter variant {variant!r}")


def _copy_adapter_variant(source: Path, destination: Path, *, variant: str) -> dict[str, Any]:
    """Create a disposable structurally-identical adapter with one family zeroed."""

    try:
        from safetensors.torch import load_file, save_file
    except ImportError as exc:  # pragma: no cover - exercised only on GPU host
        raise RuntimeError("runtime parity needs safetensors") from exc

    config = source / "adapter_config.json"
    weights = source / "adapter_model.safetensors"
    if not config.is_file() or not weights.is_file():
        raise FileNotFoundError(f"adapter must contain adapter_config.json and adapter_model.safetensors: {source}")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite adapter variant: {destination}")
    destination.mkdir(parents=True)
    shutil.copy2(config, destination / config.name)
    tensors = load_file(str(weights), device="cpu")
    copied: dict[str, Any] = {}
    zeroed: list[str] = []
    family_counts = {"linear_attn": 0, "self_attn": 0}
    for key, tensor in tensors.items():
        family = _lora_b_family(key)
        if family is not None:
            family_counts[family] += 1
        if _variant_should_zero(key, variant):
            copied[key] = tensor.new_zeros(tensor.shape)
            zeroed.append(key)
        else:
            copied[key] = tensor
    if variant != "full" and not zeroed:
        raise ValueError(f"{source}: no LoRA-B tensors matched variant {variant!r}")
    if not all(family_counts.values()):
        raise ValueError(f"{source}: expected both Qwen3.5 LoRA families, got {family_counts}")
    save_file(copied, str(destination / weights.name), metadata={"format": "pt"})
    return {
        "variant": variant,
        "path": str(destination),
        "adapter_sha256": _sha256(destination / weights.name),
        "zeroed_tensors": len(zeroed),
        "zeroed_tensor_names_sha256": _json_sha256(zeroed),
        "family_lora_b_tensor_counts": family_counts,
    }


def prepare_adapter_variants(
    adapter: str | Path,
    output_dir: str | Path,
    *,
    directory_name: str = "adapter_variants",
) -> dict[str, dict[str, Any]]:
    """Materialize all three variants under an otherwise empty output directory."""

    source = Path(adapter).resolve()
    if Path(directory_name).name != directory_name:
        raise ValueError(f"adapter-variant directory name must be a single path component: {directory_name!r}")
    root = Path(output_dir).resolve() / directory_name
    if root.exists():
        raise FileExistsError(f"runtime parity adapter directory already exists: {root}")
    root.mkdir(parents=True)
    return {variant: _copy_adapter_variant(source, root / variant, variant=variant) for variant in VARIANTS}


def _chat_prompt_ids(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> list[int]:
    """Render the same generation boundary the normal evaluator uses."""

    rendered = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    if isinstance(rendered, Mapping):
        rendered = rendered.get("input_ids")
    if hasattr(rendered, "ids"):
        rendered = rendered.ids
    if not isinstance(rendered, Sequence) or isinstance(rendered, (str, bytes)):
        raise TypeError(f"tokenizer returned invalid chat token sequence: {type(rendered).__name__}")
    ids = [int(token) for token in rendered]
    if not ids or min(ids) < 0:
        raise ValueError("tokenizer returned an empty or invalid chat token sequence")
    return ids


def _letter_candidate_token_ids(tokenizer: Any, labels: Sequence[str] = DEFAULT_LABELS) -> dict[str, list[int]]:
    """Mirror the paper's robust one-token surface-form candidates."""

    values: dict[str, list[int]] = {}
    for label in labels:
        candidates = (label, f" {label}", f"\n{label}", f"({label})")
        ids = []
        for candidate in candidates:
            encoded = tokenizer.encode(candidate, add_special_tokens=False)
            if encoded:
                ids.append(int(encoded[0]))
        unique = list(dict.fromkeys(ids))
        if not unique:
            raise ValueError(f"tokenizer cannot encode any surface form for answer label {label!r}")
        values[label] = unique
    return values


def _relative_scores(values: Mapping[int, float], *, reference_token_id: int) -> dict[int, float]:
    if reference_token_id not in values:
        raise ValueError(f"reference token {reference_token_id} is missing")
    reference = float(values[reference_token_id])
    return {int(token_id): float(value) - reference for token_id, value in values.items()}


def _effect_vector(
    base: Mapping[int, float],
    variant: Mapping[int, float],
    *,
    reference_token_id: int,
    token_ids: Sequence[int],
) -> list[float]:
    base_relative = _relative_scores(base, reference_token_id=reference_token_id)
    variant_relative = _relative_scores(variant, reference_token_id=reference_token_id)
    try:
        return [variant_relative[token_id] - base_relative[token_id] for token_id in token_ids]
    except KeyError as exc:
        raise ValueError(f"probe score is missing token {exc.args[0]}") from exc


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(first * second for first, second in zip(left, right, strict=True))


def compare_effects(hf: Sequence[float], vllm: Sequence[float]) -> dict[str, float | None]:
    """Summarize two same-token effect vectors without presuming a threshold."""

    if not hf or len(hf) != len(vllm):
        raise ValueError("HF and vLLM effect vectors must be non-empty and equal length")
    hf_norm = math.sqrt(_dot(hf, hf))
    vllm_norm = math.sqrt(_dot(vllm, vllm))
    cosine = _dot(hf, vllm) / (hf_norm * vllm_norm) if hf_norm and vllm_norm else None
    absolute_errors = [abs(first - second) for first, second in zip(hf, vllm, strict=True)]
    return {
        "hf_l2": hf_norm,
        "vllm_l2": vllm_norm,
        "hf_max_abs": max(abs(value) for value in hf),
        "vllm_max_abs": max(abs(value) for value in vllm),
        "mean_abs_error": sum(absolute_errors) / len(absolute_errors),
        "max_abs_error": max(absolute_errors),
        "cosine_similarity": cosine,
    }


def _hf_scores(
    *,
    model: Any,
    tokenizer: Any,
    prompt_token_ids: Sequence[Sequence[int]],
    adapter_name: str | None,
    batch_size: int,
) -> list[dict[int, float]]:
    """Return next-token raw logits for a supplied token set later selected by caller."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only on GPU host
        raise RuntimeError("runtime parity needs PyTorch") from exc
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        context = model.disable_adapter() if adapter_name is None else contextlib.nullcontext()
        with context:
            if adapter_name is not None:
                model.set_adapter(adapter_name)
            results: list[dict[int, float]] = []
            for start in range(0, len(prompt_token_ids), batch_size):
                batch = list(prompt_token_ids[start : start + batch_size])
                encoded = tokenizer.pad({"input_ids": batch}, padding=True, return_tensors="pt")
                device = next(model.parameters()).device
                inputs = {name: value.to(device) for name, value in encoded.items()}
                with torch.inference_mode():
                    output = model(**inputs, use_cache=False, logits_to_keep=1)
                logits = output.logits[:, -1, :].float().cpu()
                results.extend({index: float(value) for index, value in enumerate(row.tolist())} for row in logits)
            return results
    finally:
        tokenizer.padding_side = old_padding_side


def _select_scores(full_scores: Sequence[Mapping[int, float]], token_ids: Sequence[Sequence[int]]) -> list[dict[int, float]]:
    if len(full_scores) != len(token_ids):
        raise ValueError("score and token-id sequence counts differ")
    selected: list[dict[int, float]] = []
    for scores, wanted in zip(full_scores, token_ids, strict=True):
        selected.append({int(token_id): float(scores[int(token_id)]) for token_id in wanted})
    return selected


def _top_token_ids(full_scores: Sequence[Mapping[int, float]], *, count: int) -> list[list[int]]:
    if count < 1:
        raise ValueError("top token count must be positive")
    return [
        [token_id for token_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:count]]
        for scores in full_scores
    ]


def _http_json(method: str, url: str, payload: Mapping[str, Any] | None = None, *, timeout: float = 60.0) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            decoded = response.read().decode("utf-8")
            # The vLLM runtime-LoRA endpoint in 0.26 returns a bare
            # ``Success: ...`` text response, while model/completion endpoints
            # return JSON.  Retain either exact response shape for the caller.
            try:
                return json.loads(decoded)
            except json.JSONDecodeError:
                return decoded
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} returned HTTP {exc.code}: {detail[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc}") from exc


def _vllm_server_command(
    *,
    executable: str,
    model: str,
    port: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_loras: int,
    enforce_eager: bool,
    gdn_prefill_backend: str | None,
    max_logprobs: int = DEFAULT_MAX_LOGPROBS,
) -> list[str]:
    """Build the vLLM command for a reproducible parity-server launch."""

    gdn_prefill_backend = _validate_gdn_prefill_backend(gdn_prefill_backend)
    max_logprobs = _validate_max_logprobs(max_logprobs)
    command = [
        executable,
        "serve",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--enable-lora",
        "--max-lora-rank",
        "8",
        "--max-loras",
        str(max_loras),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(max_model_len),
        "--max-logprobs",
        str(max_logprobs),
        "--logprobs-mode",
        VLLM_LOGPROBS_MODE,
        "--language-model-only",
    ]
    if gdn_prefill_backend is not None:
        command.extend(("--gdn-prefill-backend", gdn_prefill_backend))
    if enforce_eager:
        # This is useful for a diagnostic where the LoRA signal is small: it
        # removes CUDA-graph / adapter-slot specialization as a possible
        # source of numerical drift. It is deliberately opt-in because the
        # normal parity gate should exercise the evaluator's production mode.
        command.append("--enforce-eager")
    return command


def _start_vllm(
    *,
    model: str,
    port: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_loras: int,
    log_path: Path,
    enforce_eager: bool = False,
    gdn_prefill_backend: str | None = None,
    max_logprobs: int = DEFAULT_MAX_LOGPROBS,
    device_token: str | None = None,
    cache_directories: Mapping[str, Path] | None = None,
) -> subprocess.Popen[bytes]:
    executable = shutil.which("vllm")
    if executable is None:
        raise FileNotFoundError("vllm executable is not on PATH")
    if not 0 < gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    if max_loras < 1:
        raise ValueError("max_loras must be positive")
    log_handle = log_path.open("wb")
    environment = dict(os.environ)
    environment["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
    if device_token is not None:
        # A parallel isolated attestation starts one process per physical
        # allocation.  Each process must see exactly one device, otherwise
        # vLLM could form a tensor-parallel group and invalidate the intended
        # one-fresh-server-per-variant comparison.
        environment["CUDA_VISIBLE_DEVICES"] = device_token
    if cache_directories is not None:
        expected_names = {name for name, _ in PARALLEL_VLLM_CACHE_ENVIRONMENT}
        if set(cache_directories) != expected_names:
            raise ValueError(
                "parallel vLLM cache directories must supply exactly "
                f"{sorted(expected_names)!r}"
            )
        for name, path in cache_directories.items():
            if path.is_symlink() or not path.is_dir():
                raise FileNotFoundError(f"parallel vLLM cache path is not a regular directory: {path}")
            environment[name] = str(path)
    command = _vllm_server_command(
        executable=executable,
        model=model,
        port=port,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_loras=max_loras,
        enforce_eager=enforce_eager,
        gdn_prefill_backend=gdn_prefill_backend,
        max_logprobs=max_logprobs,
    )
    # Store the handle on the process so it remains open until the child exits.
    try:
        process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, env=environment)
    except BaseException:
        log_handle.close()
        raise
    setattr(process, "_ctm_log_handle", log_handle)
    return process


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
    handle = getattr(process, "_ctm_log_handle", None)
    if handle is not None:
        handle.close()


def _await_server(process: subprocess.Popen[bytes], base_url: str, log_path: Path) -> str:
    deadline = time.monotonic() + 300
    last_error = "server did not respond"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = log_path.read_text(errors="replace")[-4000:] if log_path.exists() else ""
            raise RuntimeError(f"vLLM exited during startup ({process.returncode}):\n{tail}")
        try:
            models = _http_json("GET", f"{base_url}/models", timeout=5.0)
            entries = models.get("data") if isinstance(models, Mapping) else None
            if isinstance(entries, list) and entries and isinstance(entries[0], Mapping):
                model_id = entries[0].get("id")
                if isinstance(model_id, str) and model_id:
                    return model_id
            last_error = f"unexpected /models response: {models!r}"
        except RuntimeError as exc:
            last_error = str(exc)
        time.sleep(1)
    raise TimeoutError(f"vLLM did not become ready within 300 seconds: {last_error}")


def _load_vllm_adapters(base_url: str, adapters: Mapping[str, str]) -> None:
    """Load exact runtime names, including the name Inspect uses for a checkpoint."""

    for name, path in adapters.items():
        response = _http_json(
            "POST",
            f"{base_url}/load_lora_adapter",
            {"lora_name": name, "lora_path": path},
        )
        # vLLM 0.26 returns a JSON string here (``"Success: ..."``), while
        # other builds return a small object.  Accept both documented success
        # forms, but never silently continue after an unrecognised response.
        if isinstance(response, str):
            if response.lower().startswith("success:"):
                continue
        elif isinstance(response, Mapping) and response.get("status") in {"success", "ok", None}:
            continue
        raise RuntimeError(f"vLLM did not confirm loading {name!r}: {response!r}")


def _vllm_model_ids(base_url: str) -> list[str]:
    """Return the exact OpenAI model identities currently exposed by vLLM."""

    response = _http_json("GET", f"{base_url}/models")
    entries = response.get("data") if isinstance(response, Mapping) else None
    if not isinstance(entries, list):
        raise RuntimeError(f"vLLM /models has no data array: {response!r}")
    ids = [entry.get("id") for entry in entries if isinstance(entry, Mapping) and isinstance(entry.get("id"), str)]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"vLLM /models contains duplicate model identities: {ids!r}")
    if not ids:
        raise RuntimeError("vLLM /models returned no model identities")
    return ids


def _runtime_adapter_loads(
    runtime_adapter_names: Mapping[str, str], runtime_adapter_paths: Mapping[str, str]
) -> dict[str, str]:
    """Map vLLM's public adapter identities to their on-disk directories.

    The evaluator-path probe deliberately uses the resolved checkpoint path as
    its *public* model identity, whereas its bookkeeping key remains the short
    ``evaluator_path`` label.  Passing the bookkeeping mapping directly to
    vLLM would register the latter instead, then make the following `/models`
    identity check fail before any scores were collected.
    """

    if set(runtime_adapter_names) != set(runtime_adapter_paths):
        raise ValueError("runtime adapter names and paths must have identical variants")
    loads = {runtime_adapter_names[variant]: runtime_adapter_paths[variant] for variant in runtime_adapter_names}
    if len(loads) != len(runtime_adapter_names):
        raise ValueError("runtime adapter public identities must be unique")
    return loads


def _vllm_scores(
    *,
    base_url: str,
    model_name: str,
    prompt_token_ids: Sequence[Sequence[int]],
    token_ids: Sequence[Sequence[int]],
) -> list[dict[int, float]]:
    if len(prompt_token_ids) != len(token_ids):
        raise ValueError("vLLM prompts and requested-token rows differ")
    results: list[dict[int, float]] = []
    for prompt, wanted in zip(prompt_token_ids, token_ids, strict=True):
        unique = list(dict.fromkeys(int(value) for value in wanted))
        if not unique or len(unique) > DEFAULT_MAX_LOGPROBS:
            raise ValueError(
                "vLLM requested token set must contain "
                f"1..{DEFAULT_MAX_LOGPROBS} values, got {len(unique)}"
            )
        response = _http_json(
            "POST",
            f"{base_url}/completions",
            {
                "model": model_name,
                "prompt": list(prompt),
                **VLLM_PARITY_SAMPLING,
                "logprobs": len(unique),
                "allowed_token_ids": unique,
                "return_tokens_as_token_ids": True,
                "return_token_ids": True,
            },
        )
        try:
            top_logprobs = response["choices"][0]["logprobs"]["top_logprobs"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"vLLM completion lacks one-step token logprobs: {response!r}") from exc
        if not isinstance(top_logprobs, Mapping):
            raise RuntimeError(f"vLLM token logprobs are not an object: {top_logprobs!r}")
        decoded: dict[int, float] = {}
        for key, value in top_logprobs.items():
            if not isinstance(key, str) or not key.startswith("token_id:"):
                raise RuntimeError(f"vLLM did not return token IDs despite return_tokens_as_token_ids: {key!r}")
            token_id = int(key.removeprefix("token_id:"))
            decoded[token_id] = float(value)
        expected_ids = set(unique)
        actual_ids = set(decoded)
        if actual_ids != expected_ids:
            missing = sorted(expected_ids - actual_ids)
            unexpected = sorted(actual_ids - expected_ids)
            raise RuntimeError(
                "vLLM did not return exactly the requested logprobs for token IDs: "
                f"missing={missing}, unexpected={unexpected}"
            )
        results.append({token_id: decoded[token_id] for token_id in unique})
    return results


def _probe_isolated_vllm_server(
    *,
    process: subprocess.Popen[bytes],
    variant: str,
    base_url: str,
    log_path: Path,
    runtime_adapter_name: str,
    runtime_adapter_path: str,
    prompt_token_ids: Sequence[Sequence[int]],
    token_ids: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Probe one already-launched, one-slot vLLM server and always tear it down."""

    try:
        base_name = _await_server(process, base_url, log_path)
        _load_vllm_adapters(base_url, {runtime_adapter_name: runtime_adapter_path})
        model_ids = _vllm_model_ids(base_url)
        if runtime_adapter_name not in model_ids:
            raise RuntimeError(
                "vLLM acknowledged adapter loading but /models does not expose the exact request identity: "
                f"{runtime_adapter_name!r}; available={model_ids!r}"
            )
        return {
            "base_name": base_name,
            "model_ids": model_ids,
            "base_scores": _vllm_scores(
                base_url=base_url,
                model_name=base_name,
                prompt_token_ids=prompt_token_ids,
                token_ids=token_ids,
            ),
            "scores": _vllm_scores(
                base_url=base_url,
                model_name=runtime_adapter_name,
                prompt_token_ids=prompt_token_ids,
                token_ids=token_ids,
            ),
            "variant": variant,
        }
    finally:
        _stop_process(process)


def _signal_process_termination(process: subprocess.Popen[bytes]) -> None:
    """Prompt a sibling worker to exit without racing its log-handle cleanup."""

    if process.poll() is None:
        with contextlib.suppress(OSError):
            process.terminate()


def _run_parallel_isolated_vllm_variants(
    *,
    model_name: str,
    root: Path,
    plan: Mapping[str, Mapping[str, int | str]],
    runtime_adapter_names: Mapping[str, str],
    runtime_adapter_paths: Mapping[str, str],
    prompt_token_ids: Sequence[Sequence[int]],
    token_ids: Sequence[Sequence[int]],
    max_model_len: int,
    vllm_memory_utilization: float,
    enforce_eager: bool,
    gdn_prefill_backend: str | None,
    max_logprobs: int = DEFAULT_MAX_LOGPROBS,
) -> dict[str, dict[str, Any]]:
    """Run all four strict isolated probes concurrently, failing closed as one unit.

    Starting the four subprocesses first lets model loading overlap.  The
    worker futures then independently await, load, score, and clean up their
    own one-slot server.  If any one fails, the remaining servers receive a
    termination signal immediately; their workers retain their own ``finally``
    cleanup, so logs and file handles are never raced by the coordinator.
    """

    processes: dict[str, subprocess.Popen[bytes]] = {}
    try:
        for variant in RESULT_VARIANTS:
            configuration = plan[variant]
            port = configuration["port"]
            device_token = configuration["device_token"]
            log_name = configuration["server_log"]
            server_max_logprobs = configuration["max_logprobs"]
            server_logprobs_mode = configuration["logprobs_mode"]
            if (
                not isinstance(port, int)
                or not isinstance(device_token, str)
                or not isinstance(log_name, str)
                or isinstance(server_max_logprobs, bool)
                or not isinstance(server_max_logprobs, int)
                or server_logprobs_mode != VLLM_LOGPROBS_MODE
            ):
                raise TypeError(f"invalid parallel vLLM configuration for {variant!r}")
            if _validate_max_logprobs(server_max_logprobs) != _validate_max_logprobs(max_logprobs):
                raise ValueError(f"parallel vLLM server cap differs from requested max_logprobs for {variant!r}")
            cache_directories = _prepare_parallel_isolated_vllm_cache_paths(root=root, variant=variant)
            processes[variant] = _start_vllm(
                model=model_name,
                port=port,
                max_model_len=max_model_len,
                gpu_memory_utilization=vllm_memory_utilization,
                max_loras=1,
                log_path=root / log_name,
                enforce_eager=enforce_eager,
                gdn_prefill_backend=gdn_prefill_backend,
                max_logprobs=max_logprobs,
                device_token=device_token,
                cache_directories=cache_directories,
            )
    except BaseException:
        for process in processes.values():
            _stop_process(process)
        raise

    outcomes: dict[str, dict[str, Any]] = {}
    futures: dict[Any, str] = {}
    submitted_variants: set[str] = set()
    try:
        with ThreadPoolExecutor(max_workers=len(RESULT_VARIANTS), thread_name_prefix="ctm-parity-vllm") as executor:
            try:
                for variant in RESULT_VARIANTS:
                    configuration = plan[variant]
                    port = configuration["port"]
                    log_name = configuration["server_log"]
                    if not isinstance(port, int) or not isinstance(log_name, str):
                        raise TypeError(f"invalid parallel vLLM configuration for {variant!r}")
                    future = executor.submit(
                        _probe_isolated_vllm_server,
                        process=processes[variant],
                        variant=variant,
                        base_url=f"http://127.0.0.1:{port}/v1",
                        log_path=root / log_name,
                        runtime_adapter_name=runtime_adapter_names[variant],
                        runtime_adapter_path=runtime_adapter_paths[variant],
                        prompt_token_ids=prompt_token_ids,
                        token_ids=token_ids,
                    )
                    futures[future] = variant
                    submitted_variants.add(variant)
                for future in as_completed(futures):
                    variant = futures[future]
                    outcomes[variant] = future.result()
            except BaseException:
                # Do not let an unrelated variant keep consuming a GPU while
                # the all-variants attestation is already known to be invalid.
                for process in processes.values():
                    _signal_process_termination(process)
                raise
    except BaseException:
        # A failure while submitting futures leaves no worker responsible for
        # any never-submitted process.  Submitted workers own normal cleanup.
        for variant, process in processes.items():
            if variant not in submitted_variants:
                _stop_process(process)
        raise

    # Future completion order is intentionally irrelevant to the report.
    return {variant: outcomes[variant] for variant in RESULT_VARIANTS}


def _variant_verdict(summary: Mapping[str, float | None]) -> str:
    """Use conservative diagnostic categories rather than declare behavioural success."""

    hf = float(summary["hf_max_abs"])
    vllm = float(summary["vllm_max_abs"])
    cosine = summary["cosine_similarity"]
    if hf < 1e-5:
        return "no_measurable_hf_effect"
    if vllm < 1e-5:
        return "hf_effect_missing_in_vllm"
    if cosine is None or float(cosine) < 0.90:
        return "nonzero_but_delta_mismatch"
    return "hf_vllm_effects_agree"


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    payload = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite parity report: {path}")
    path.write_bytes(payload)


def run_probe(
    *,
    model_name: str,
    adapter: str | Path,
    hf_adapter: str | Path | None = None,
    data: str | Path,
    output_dir: str | Path,
    samples: int = DEFAULT_PROBE_SAMPLES,
    top_token_count: int = DEFAULT_TOP_TOKEN_COUNT,
    hf_batch_size: int = DEFAULT_HF_BATCH_SIZE,
    vllm_port: int = DEFAULT_VLLM_PORT,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    vllm_memory_utilization: float = DEFAULT_VLLM_MEMORY_UTILIZATION,
    max_logprobs: int = DEFAULT_MAX_LOGPROBS,
    result_variants: Sequence[str] = RESULT_VARIANTS,
    enforce_eager: bool = False,
    isolate_vllm_variants: bool = False,
    parallel_isolated_vllm_variants: bool = False,
    vllm_device_tokens: Sequence[str] | None = None,
    gdn_prefill_backend: str | None = None,
) -> dict[str, Any]:
    """Execute the complete HF/vLLM parity probe.

    The default and sequential-isolation modes use one visible GPU.  The
    explicit parallel-isolation mode consumes four visible GPUs only after HF
    scoring completes, then runs the complete four-variant attestation at once.
    """

    gdn_prefill_backend = _validate_gdn_prefill_backend(gdn_prefill_backend)
    max_logprobs = _validate_max_logprobs(max_logprobs)
    requested_result_variants = tuple(result_variants)
    if not requested_result_variants:
        raise ValueError("at least one result variant is required")
    if len(set(requested_result_variants)) != len(requested_result_variants):
        raise ValueError("result variants must be unique")
    unknown_variants = sorted(set(requested_result_variants) - set(RESULT_VARIANTS))
    if unknown_variants:
        raise ValueError(f"unknown result variants: {unknown_variants!r}")
    parallel_isolated_plan = _parallel_isolated_vllm_plan(
        requested_result_variants=requested_result_variants,
        device_tokens=vllm_device_tokens,
        vllm_port=vllm_port,
        isolate_vllm_variants=isolate_vllm_variants,
        parallel_isolated_vllm_variants=parallel_isolated_vllm_variants,
        enforce_eager=enforce_eager,
        max_logprobs=max_logprobs,
    )
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised only on GPU host
        raise RuntimeError("runtime parity needs transformers, peft, and torch") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("runtime parity requires one visible CUDA GPU")
    root = Path(output_dir).resolve()
    if root.exists():
        raise FileExistsError(f"runtime parity output directory already exists: {root}")
    root.mkdir(parents=True)
    prompts = load_prompts(data, limit=samples)
    vllm_adapter = Path(adapter).resolve()
    hf_source_adapter = Path(hf_adapter).resolve() if hf_adapter is not None else vllm_adapter
    # Normally both runtimes read the same adapter.  A Qwen3.5 compatibility
    # copy is the one deliberate exception: its key spelling is for vLLM's VL
    # wrapper and must be compared with HF using the untouched PEFT adapter.
    variants = prepare_adapter_variants(
        vllm_adapter,
        root,
        directory_name="vllm_adapter_variants",
    )
    hf_variants = (
        variants
        if hf_source_adapter == vllm_adapter
        else prepare_adapter_variants(
            hf_source_adapter,
            root,
            directory_name="hf_adapter_variants",
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt_ids = [_chat_prompt_ids(tokenizer, prompt.messages) for prompt in prompts]
    if max(len(ids) for ids in prompt_ids) > max_model_len:
        raise ValueError("a probe prompt exceeds max_model_len")

    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda")
    base_model.eval()
    peft_model = PeftModel.from_pretrained(
        base_model,
        hf_variants["full"]["path"],
        adapter_name="full",
        is_trainable=False,
    )
    for variant in ("linear_only", "self_attn_only"):
        peft_model.load_adapter(hf_variants[variant]["path"], adapter_name=variant, is_trainable=False)
    peft_model.eval()

    # We deliberately retrieve dense last-token logits only in this tiny
    # (<=8-prompt) probe.  Qwen3.5's model forward uses logits_to_keep=1, so
    # no sequence-length-times-vocabulary tensor is materialized.
    hf_base_dense = _hf_scores(
        model=peft_model,
        tokenizer=tokenizer,
        prompt_token_ids=prompt_ids,
        adapter_name=None,
        batch_size=hf_batch_size,
    )
    candidate_ids = {token_id for values in _letter_candidate_token_ids(tokenizer).values() for token_id in values}
    top_ids = _top_token_ids(hf_base_dense, count=top_token_count)
    requested_ids = [list(dict.fromkeys([*top, *sorted(candidate_ids)])) for top in top_ids]
    requested_logprobs = max(len(token_ids) for token_ids in requested_ids)
    if requested_logprobs > max_logprobs:
        # Keep the failure before any vLLM process starts.  This should be
        # unreachable for the sealed 29-token protocol cap, but makes a
        # future request-set expansion an explicit transport change.
        raise ValueError(
            "parity requested more next-token logprobs than the server admits: "
            f"requested={requested_logprobs}, max_logprobs={max_logprobs}"
        )
    references = [top[0] for top in top_ids]
    hf_scores = {"base": _select_scores(hf_base_dense, requested_ids)}
    hf_required_variants = tuple(
        dict.fromkeys("full" if variant == EVALUATOR_PATH_VARIANT else variant for variant in requested_result_variants)
    )
    for variant in hf_required_variants:
        dense = _hf_scores(
            model=peft_model,
            tokenizer=tokenizer,
            prompt_token_ids=prompt_ids,
            adapter_name=variant,
            batch_size=hf_batch_size,
        )
        hf_scores[variant] = _select_scores(dense, requested_ids)

    # Free the training-stack model before vLLM reserves its KV cache.
    del peft_model, base_model, hf_base_dense
    torch.cuda.empty_cache()

    base_url = f"http://127.0.0.1:{vllm_port}/v1"
    server_log = root / "vllm-server.log"
    # The final entry exactly matches the resolved checkpoint name selected by
    # ``local_checkpoint_model(..., provider='vllm')``.  The other names are
    # deliberately short aliases that isolate individual projection families.
    all_runtime_adapter_names = {
        "full": "full",
        "linear_only": "linear_only",
        "self_attn_only": "self_attn_only",
        EVALUATOR_PATH_VARIANT: str(vllm_adapter),
    }
    all_runtime_adapter_paths = {
        "full": str(variants["full"]["path"]),
        "linear_only": str(variants["linear_only"]["path"]),
        "self_attn_only": str(variants["self_attn_only"]["path"]),
        EVALUATOR_PATH_VARIANT: str(vllm_adapter),
    }
    runtime_adapter_names = {
        variant: all_runtime_adapter_names[variant] for variant in requested_result_variants
    }
    runtime_adapter_paths = {
        variant: all_runtime_adapter_paths[variant] for variant in requested_result_variants
    }
    # In isolated mode each score pair is produced by a fresh one-adapter
    # server. This is a stricter transport diagnostic for weak LoRA families:
    # it eliminates interactions between concurrently resident adapter slots.
    vllm_scores: dict[str, list[dict[int, float]]] = {}
    vllm_base_scores: dict[str, list[dict[int, float]]] = {}
    base_vllm_names: dict[str, str] = {}
    model_ids_after_load: dict[str, list[str]] | list[str]
    if parallel_isolated_plan is not None:
        parallel_outcomes = _run_parallel_isolated_vllm_variants(
            model_name=model_name,
            root=root,
            plan=parallel_isolated_plan,
            runtime_adapter_names=runtime_adapter_names,
            runtime_adapter_paths=runtime_adapter_paths,
            prompt_token_ids=prompt_ids,
            token_ids=requested_ids,
            max_model_len=max_model_len,
            vllm_memory_utilization=vllm_memory_utilization,
            enforce_eager=enforce_eager,
            gdn_prefill_backend=gdn_prefill_backend,
            max_logprobs=max_logprobs,
        )
        for variant in requested_result_variants:
            outcome = parallel_outcomes[variant]
            base_vllm_names[variant] = outcome["base_name"]
            vllm_base_scores[variant] = outcome["base_scores"]
            vllm_scores[variant] = outcome["scores"]
        model_ids_after_load = {
            variant: parallel_outcomes[variant]["model_ids"] for variant in requested_result_variants
        }
    elif isolate_vllm_variants:
        isolated_model_ids: dict[str, list[str]] = {}
        for variant in requested_result_variants:
            variant_log = root / f"vllm-server-{variant}.log"
            process = _start_vllm(
                model=model_name,
                port=vllm_port,
                max_model_len=max_model_len,
                gpu_memory_utilization=vllm_memory_utilization,
                max_loras=1,
                log_path=variant_log,
                enforce_eager=enforce_eager,
                gdn_prefill_backend=gdn_prefill_backend,
                max_logprobs=max_logprobs,
            )
            try:
                base_name = _await_server(process, base_url, variant_log)
                _load_vllm_adapters(
                    base_url,
                    {runtime_adapter_names[variant]: runtime_adapter_paths[variant]},
                )
                model_ids = _vllm_model_ids(base_url)
                if runtime_adapter_names[variant] not in model_ids:
                    raise RuntimeError(
                        "vLLM acknowledged adapter loading but /models does not expose the exact request identity: "
                        f"{runtime_adapter_names[variant]!r}; available={model_ids!r}"
                    )
                base_vllm_names[variant] = base_name
                isolated_model_ids[variant] = model_ids
                vllm_base_scores[variant] = _vllm_scores(
                    base_url=base_url,
                    model_name=base_name,
                    prompt_token_ids=prompt_ids,
                    token_ids=requested_ids,
                )
                vllm_scores[variant] = _vllm_scores(
                    base_url=base_url,
                    model_name=runtime_adapter_names[variant],
                    prompt_token_ids=prompt_ids,
                    token_ids=requested_ids,
                )
            finally:
                _stop_process(process)
        model_ids_after_load = isolated_model_ids
    else:
        process = _start_vllm(
            model=model_name,
            port=vllm_port,
            max_model_len=max_model_len,
            gpu_memory_utilization=vllm_memory_utilization,
            max_loras=len(runtime_adapter_names),
            log_path=server_log,
            enforce_eager=enforce_eager,
            gdn_prefill_backend=gdn_prefill_backend,
            max_logprobs=max_logprobs,
        )
        try:
            base_name = _await_server(process, base_url, server_log)
            _load_vllm_adapters(
                base_url,
                _runtime_adapter_loads(runtime_adapter_names, runtime_adapter_paths),
            )
            model_ids = _vllm_model_ids(base_url)
            missing_model_ids = sorted(set(runtime_adapter_names.values()) - set(model_ids))
            if missing_model_ids:
                raise RuntimeError(
                    "vLLM acknowledged adapter loading but /models does not expose the exact request identities: "
                    f"{missing_model_ids!r}; available={model_ids!r}"
                )
            base_scores = _vllm_scores(
                base_url=base_url,
                model_name=base_name,
                prompt_token_ids=prompt_ids,
                token_ids=requested_ids,
            )
            for variant in requested_result_variants:
                base_vllm_names[variant] = base_name
                vllm_base_scores[variant] = base_scores
                vllm_scores[variant] = _vllm_scores(
                    base_url=base_url,
                    model_name=runtime_adapter_names[variant],
                    prompt_token_ids=prompt_ids,
                    token_ids=requested_ids,
                )
            model_ids_after_load = model_ids
        finally:
            _stop_process(process)

    results: dict[str, Any] = {}
    for variant in requested_result_variants:
        hf_effect: list[float] = []
        vllm_effect: list[float] = []
        for index, wanted in enumerate(requested_ids):
            hf_effect.extend(
                _effect_vector(
                    hf_scores["base"][index],
                    # The evaluator-path adapter is byte-identical in weights
                    # to the "full" disposable copy; reuse its HF pass rather
                    # than introduce a fourth redundant model forward.
                    hf_scores["full" if variant == EVALUATOR_PATH_VARIANT else variant][index],
                    reference_token_id=references[index],
                    token_ids=wanted,
                )
            )
            vllm_effect.extend(
                _effect_vector(
                    vllm_base_scores[variant][index],
                    vllm_scores[variant][index],
                    reference_token_id=references[index],
                    token_ids=wanted,
                )
            )
        comparison = compare_effects(hf_effect, vllm_effect)
        results[variant] = {
            "summary": comparison,
            "verdict": _variant_verdict(comparison),
            "hf_effect_vector": hf_effect,
            "vllm_effect_vector": vllm_effect,
        }

    report = {
        "schema": SCHEMA,
        "model": model_name,
        "adapter": {
            "path": str(vllm_adapter),
            "adapter_model_sha256": _sha256(vllm_adapter / "adapter_model.safetensors"),
            "hf_path": str(hf_source_adapter),
            "hf_adapter_model_sha256": _sha256(hf_source_adapter / "adapter_model.safetensors"),
        },
        "data": {
            "path": str(Path(data).resolve()),
            "sha256": _sha256(Path(data).resolve()),
            "samples": len(prompts),
            "question_ids": [prompt.question_id for prompt in prompts],
            "question_ids_sha256": _json_sha256([prompt.question_id for prompt in prompts]),
            "chat_prompt_token_lengths": [len(ids) for ids in prompt_ids],
        },
        "token_protocol": {
            "generation_boundary": "tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)",
            "candidate_label_surface_forms": ["X", " X", "\\nX", "(X)"],
            "top_token_count": top_token_count,
            "requested_token_ids": requested_ids,
            "reference_token_ids": references,
            "vllm_score_transport": VLLM_SCORE_TRANSPORT,
            "vllm_allowed_token_ids": "requested_token_ids",
            "vllm_response_token_ids": "exactly_requested_token_ids",
            "comparison": (
                "adapter-minus-base relative next-token score; neutral processed-logprob "
                "differences under the allowed-token restricted softmax equal logit differences"
            ),
            "requested_result_variants": list(requested_result_variants),
        },
        "adapter_variants": {
            "vllm": variants,
            "hf": hf_variants,
        },
        "backends": {
            "hf": {"torch_dtype": "bfloat16", "batch_size": hf_batch_size},
            "vllm": {
                "base_model_ids": base_vllm_names,
                "runtime_adapter_names": runtime_adapter_names,
                "model_ids_after_load": model_ids_after_load,
                "port": vllm_port,
                "max_model_len": max_model_len,
                "max_logprobs": max_logprobs,
                "logprobs_mode": VLLM_LOGPROBS_MODE,
                "score_transport": VLLM_SCORE_TRANSPORT,
                "parity_sampling": dict(VLLM_PARITY_SAMPLING),
                "gpu_memory_utilization": vllm_memory_utilization,
                "server_log": str(server_log),
                "enforce_eager": enforce_eager,
                "isolate_vllm_variants": isolate_vllm_variants,
                "parallel_isolated_vllm_variants": parallel_isolated_plan is not None,
                "parallel_isolated_server_plan": (
                    _parallel_isolated_vllm_report_metadata(root=root, plan=parallel_isolated_plan)
                    if parallel_isolated_plan is not None
                    else None
                ),
                "gdn_prefill_backend": gdn_prefill_backend,
            },
        },
        "results": results,
    }
    _write_report(root / "report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument(
        "--hf-adapter",
        type=Path,
        help="Untouched PEFT adapter for HF when --adapter is a vLLM compatibility copy.",
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=DEFAULT_PROBE_SAMPLES)
    parser.add_argument("--top-token-count", type=int, default=DEFAULT_TOP_TOKEN_COUNT)
    parser.add_argument("--hf-batch-size", type=int, default=DEFAULT_HF_BATCH_SIZE)
    parser.add_argument("--vllm-port", type=int, default=DEFAULT_VLLM_PORT)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--vllm-memory-utilization", type=float, default=DEFAULT_VLLM_MEMORY_UTILIZATION)
    parser.add_argument(
        "--max-logprobs",
        type=int,
        default=DEFAULT_MAX_LOGPROBS,
        help=(
            "vLLM server logprob admission cap. It is pinned to the 29-token "
            "maximum of this frozen parity protocol."
        ),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=RESULT_VARIANTS,
        default=list(RESULT_VARIANTS),
        help="Subset of adapter variants to probe. Defaults to the full attestation set.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable vLLM CUDA graphs for a controlled numerical diagnostic.",
    )
    parser.add_argument(
        "--isolate-vllm-variants",
        action="store_true",
        help="Run each requested adapter variant in a fresh one-slot vLLM server.",
    )
    parser.add_argument(
        "--parallel-isolated-vllm-variants",
        action="store_true",
        help=(
            "Run the complete four-variant isolated attestation concurrently. "
            "Requires --isolate-vllm-variants and four unique --vllm-device-tokens."
        ),
    )
    parser.add_argument(
        "--vllm-device-tokens",
        nargs=4,
        metavar="DEVICE",
        help=(
            "Exactly four unique CUDA_VISIBLE_DEVICES tokens, assigned in canonical "
            "full/linear_only/self_attn_only/evaluator_path order for parallel isolation."
        ),
    )
    parser.add_argument(
        "--gdn-prefill-backend",
        choices=GDN_PREFILL_BACKENDS,
        help="Optional diagnostic override for vLLM's Gated DeltaNet prefill backend.",
    )
    args = parser.parse_args(argv)
    try:
        report = run_probe(
            model_name=args.model,
            adapter=args.adapter,
            hf_adapter=args.hf_adapter,
            data=args.data,
            output_dir=args.output_dir,
            samples=args.samples,
            top_token_count=args.top_token_count,
            hf_batch_size=args.hf_batch_size,
            vllm_port=args.vllm_port,
            max_model_len=args.max_model_len,
            vllm_memory_utilization=args.vllm_memory_utilization,
            max_logprobs=args.max_logprobs,
            result_variants=args.variants,
            enforce_eager=args.enforce_eager,
            isolate_vllm_variants=args.isolate_vllm_variants,
            parallel_isolated_vllm_variants=args.parallel_isolated_vllm_variants,
            vllm_device_tokens=args.vllm_device_tokens,
            gdn_prefill_backend=args.gdn_prefill_backend,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    verdicts = ", ".join(f"{variant}={result['verdict']}" for variant, result in report["results"].items())
    print(f"wrote: {args.output_dir.resolve() / 'report.json'}; {verdicts}")


if __name__ == "__main__":
    main()
