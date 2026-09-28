"""Measured, hash-bound parity for Qwen3.5-9B ordinary Q/V-only adapters.

This is a distinct attestation, never a relaxation of the original two-family
contract. All trained components must have a nonzero agreeing effect; the
absent DeltaNet component must be measured as zero in both runtimes.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

SCOPE = "qwen35-9b-ordinary-qv-only-v1"
ATTESTATION_SCHEMA = "qwen35-vllm-ordinary-qv-parity-attestation-v1"
NOOP_VERDICT = "verified_absent_family_zero_effect"
ACTIVE_VERDICT = "hf_vllm_effects_agree"
VARIANTS = ("full", "linear_only", "self_attn_only", "evaluator_path")
ORDINARY_LAYERS = tuple(range(3, 32, 4))
MODULES = tuple(f"{i}.self_attn.{p}_proj" for i in ORDINARY_LAYERS for p in ("q", "v"))
PREFIXES = ("base_model.model.model.layers.", "base_model.model.model.language_model.layers.")


def same_model_snapshot(left: Any, right: Any) -> bool:
    """Accept literal identity or two existing absolute aliases of one directory.

    Never infer equality from a model name, revision basename or similar-looking
    path. Isambard's /scratch and /lus/.../scratch aliases resolve to the same
    actual snapshot, while adapter configuration bytes remain untouched.
    """
    if not isinstance(left, str) or not isinstance(right, str) or not left or not right:
        return False
    if left == right:
        return True
    a, b = Path(left), Path(right)
    return a.is_absolute() and b.is_absolute() and a.is_dir() and b.is_dir() and a.samefile(b)


def sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tensors(path: Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    tensors = load_file(str(path / "adapter_model.safetensors"), device="cpu")
    normalized = {}
    for key, value in tensors.items():
        prefixes = [p for p in PREFIXES if key.startswith(p)]
        if len(prefixes) != 1:
            raise ValueError(f"unexpected scope tensor: {key}")
        name = key.removeprefix(prefixes[0])
        if name in normalized:
            raise ValueError("duplicate normalized adapter tensor")
        normalized[name] = value
    return normalized


def scope_identity(adapter: str | Path) -> dict[str, Any]:
    """Check actual A/B pairs, all eight ordinary layers, and target config."""
    root = Path(adapter).resolve()
    cfg = json.loads((root / "adapter_config.json").read_text())
    targets = cfg.get("target_modules")
    if not isinstance(targets, list) or set(targets) not in (
        {"q_proj", "v_proj"}, {"model.layers." + m for m in MODULES}
    ):
        raise ValueError("ordinary Q/V scope requires exactly its ordinary-attention targets")
    tensors = _tensors(root)
    expected = {f"{m}.lora_{p}.weight" for m in MODULES for p in ("A", "B")}
    if set(tensors) != expected:
        raise ValueError("ordinary Q/V scope requires exactly 16 complete pairs; no DeltaNet/MLP/extra tensors")
    rank = cfg.get("r")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("adapter rank must be a positive integer")
    shapes = {}
    for module in MODULES:
        a, b = (tensors[f"{module}.lora_{p}.weight"] for p in ("A", "B"))
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != rank or b.shape[1] != rank:
            raise ValueError("ordinary Q/V adapter ranks do not match its configuration")
        for p, tensor in (("A", a), ("B", b)):
            if not bool(tensor.isfinite().all()) or min(tensor.shape) <= 0:
                raise ValueError("non-finite or empty adapter tensor")
            shapes[f"{module}.lora_{p}.weight"] = list(tensor.shape)
    return {"profile": SCOPE, "adapter_model_sha256": sha(root / "adapter_model.safetensors"),
            "adapter_config_sha256": sha(root / "adapter_config.json"),
            "family_lora_b_tensor_counts": {"linear_attn": 0, "self_attn": 16},
            "modules": list(MODULES), "shapes": shapes, "expected_noop_variants": ["linear_only"]}


def scoped_verdict(variant: str, summary: Mapping[str, Any]) -> str:
    if variant != "linear_only":
        raise ValueError("only the absent DeltaNet variant has a scoped no-op verdict")
    values = [summary.get("hf_max_abs"), summary.get("vllm_max_abs")]
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0 <= v < 1e-5 for v in values):
        return NOOP_VERDICT
    return "unexpected_effect_in_absent_family"


def _validate_vectors(result: Mapping[str, Any], *, variant: str, length: int) -> None:
    hf, vl = result.get("hf_effect_vector"), result.get("vllm_effect_vector")
    for vector in (hf, vl):
        if not isinstance(vector, list) or len(vector) != length or length < 1:
            raise ValueError("missing or wrong-length numerical effect vector")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vector):
            raise ValueError("non-finite numerical effect vector")
    hn = math.sqrt(sum(v * v for v in hf))
    vn = math.sqrt(sum(v * v for v in vl))
    cosine = sum(a * b for a, b in zip(hf, vl, strict=True)) / (hn * vn) if hn and vn else None
    maximums = {"hf_max_abs": max(map(abs, hf)), "vllm_max_abs": max(map(abs, vl))}
    summary = result.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("missing numerical summary")
    for name, value in {**maximums, "cosine_similarity": cosine}.items():
        actual = summary.get(name)
        if value is None:
            if actual is not None:
                raise ValueError("summary does not match measured vectors")
        elif isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("summary does not match measured vectors")
    if variant == "linear_only":
        if result.get("verdict") != NOOP_VERDICT or scoped_verdict(variant, maximums) != NOOP_VERDICT:
            raise ValueError("absent family did not measure zero in both runtimes")
    elif result.get("verdict") != ACTIVE_VERDICT or min(maximums.values()) < 1e-5 or cosine is None or cosine < 0.90:
        raise ValueError("trained adapter effect fails the original nonzero/agreement thresholds")


def validate_scope_report(adapter: str | Path, report: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute scope and check real diagnostic payloads and measured vectors."""
    import torch

    root = Path(adapter).resolve()
    identity = scope_identity(root)
    cfg = json.loads((root / "adapter_config.json").read_text())
    if not same_model_snapshot(report.get("model"), cfg.get("base_model_name_or_path")):
        raise ValueError("scope parity model differs from the adapter base model")
    if report.get("schema") != "qwen35-lora-runtime-parity-v1" or report.get("adapter_scope") != identity:
        raise ValueError("report does not bind the actual ordinary Q/V-only scope")
    a = report.get("adapter", {})
    if a.get("path") != str(root) or a.get("adapter_model_sha256") != identity["adapter_model_sha256"]:
        raise ValueError("report adapter identity mismatch")
    hf_path = Path(a.get("hf_path", ""))
    if not hf_path.is_absolute():
        raise ValueError("HF adapter path must be absolute")
    hf_identity = scope_identity(hf_path)
    if a.get("hf_adapter_model_sha256") != hf_identity["adapter_model_sha256"]:
        raise ValueError("raw HF adapter hash mismatch")
    if hf_identity["adapter_config_sha256"] != identity["adapter_config_sha256"]:
        raise ValueError("compatibility copy changed adapter configuration")
    payload = _tensors(root)
    raw = _tensors(hf_path)
    if any(not torch.equal(payload[k], raw[k]) for k in payload):
        raise ValueError("compatibility translation changed tensor payloads")
    variants = report.get("adapter_variants", {})
    for backend in ("hf", "vllm"):
        entries = variants.get(backend, {})
        if set(entries) != {"full", "linear_only", "self_attn_only"}:
            raise ValueError("missing diagnostic adapter variants")
        for name, entry in entries.items():
            p = Path(entry.get("path", ""))
            if not p.is_absolute() or entry.get("adapter_sha256") != sha(p / "adapter_model.safetensors"):
                raise ValueError("diagnostic adapter hash mismatch")
            if sha(p / "adapter_config.json") != identity["adapter_config_sha256"]:
                raise ValueError("diagnostic adapter configuration changed")
            actual = _tensors(p)
            if set(actual) != set(payload):
                raise ValueError("diagnostic adapter scope changed")
            for key, value in actual.items():
                expected = torch.zeros_like(payload[key]) if name == "linear_only" and ".lora_B." in key else payload[key]
                if not torch.equal(value, expected):
                    raise ValueError("diagnostic adapter is not the declared exact family isolation")
    protocol = report.get("token_protocol", {})
    requested = protocol.get("requested_token_ids")
    if protocol.get("requested_result_variants") != list(VARIANTS) or not isinstance(requested, list) or not requested:
        raise ValueError("scope attestation needs the complete four-variant token protocol")
    if any(not isinstance(row, list) or not row for row in requested):
        raise ValueError("invalid requested token set")
    results = report.get("results", {})
    if set(results) != set(VARIANTS):
        raise ValueError("scope attestation needs all four measured results")
    for variant in VARIANTS:
        if not isinstance(results[variant], Mapping):
            raise ValueError("invalid numerical result mapping")
        _validate_vectors(results[variant], variant=variant, length=sum(map(len, requested)))
    return identity


def make_scope_attestation(adapter: str | Path, report_path: str | Path) -> dict[str, Any]:
    root, path = Path(adapter).resolve(), Path(report_path).resolve()
    report = json.loads(path.read_text())
    identity = validate_scope_report(root, report)
    return {"schema": ATTESTATION_SCHEMA, "adapter_path": str(root),
            "adapter_model_sha256": identity["adapter_model_sha256"],
            "adapter_config_sha256": identity["adapter_config_sha256"],
            "model": report.get("model"), "report_path": str(path), "report_sha256": sha(path),
            "scope": identity, "required_variants": list(VARIANTS),
            "verdicts": {v: report["results"][v]["verdict"] for v in VARIANTS}}


def verified_scope_attestation(adapter: str | Path, attestation: Mapping[str, Any]) -> bool:
    try:
        root = Path(adapter).resolve()
        path = Path(attestation.get("report_path", ""))
        if not path.is_absolute() or attestation.get("schema") != ATTESTATION_SCHEMA:
            return False
        return dict(attestation) == make_scope_attestation(root, path)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError):
        return False
