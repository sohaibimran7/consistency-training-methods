"""Focused offline tests for immutable Stage 2 Luna-derived promotion."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.stage2_ood_hle import grade_luna
from experiments.stage2_ood_hle import promote_luna_derived as promotion
from experiments.stage2_ood_hle.hf_peft_resume import HANDOFF_SCHEMA
from experiments.stage2_ood_hle.hf_peft_runner import BASE_MODEL, condition_spec


IDENTITY = ("biased", "iid", "in_domain", "logiqa", "wrong_argument")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint(condition: str, path: Path) -> dict[str, object]:
    expected = condition_spec(condition)
    return {
        "path": str(path.resolve()),
        "checkpoint_name": expected.checkpoint_name,
        "backend": "local",
        "lora": True,
        "base_model": BASE_MODEL,
        "adapter_model_sha256": expected.adapter_model_sha256,
        "adapter_config_sha256": expected.adapter_config_sha256,
        "manifest_sha256": expected.manifest_sha256,
    }


def _scored_log() -> SimpleNamespace:
    return SimpleNamespace(
        status="success",
        samples=[
            SimpleNamespace(
                id="question-1",
                scores={
                    "luna": SimpleNamespace(
                        value={"bias_acknowledged": 1.0},
                        metadata={
                            "grader_model": promotion.EXPECTED_GRADER_MODEL,
                            "grader_max_tokens": promotion.EXPECTED_GRADER_MAX_TOKENS,
                            "grader_response": "yes",
                            "grader_usage": {"total_tokens": 4},
                            "grader_stop_reason": "stop",
                            "grader_max_tokens_cap_hit": False,
                        },
                    )
                },
            )
        ],
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str,
    condition: str,
    binding_kind: str,
) -> promotion.SourceRoots:
    root = tmp_path / name
    derived_root = root / ("luna-derived-v1" if binding_kind == "raw_preflight" else "luna-incremental-derived-v1")
    staged_root = root / "luna-incremental-staged-v1"
    raw_root = root / "raw-no-luna" / condition
    preflight_root = root / "preflight-v1"
    manifest = root / "manifest.json"
    raw_root.mkdir(parents=True)
    manifest.write_bytes(b'{"frozen": "manifest"}\n')
    raw = raw_root / "biased.eval"
    clean = raw_root / "clean.eval"
    raw.write_bytes(f"raw {condition}".encode("utf-8"))
    clean.write_bytes(f"clean {condition}".encode("utf-8"))
    source_description = {
        "population": IDENTITY[2],
        "bias_type": IDENTITY[4],
        "dataset": IDENTITY[3],
        "raw_log": str(raw.resolve()),
    }
    staged = grade_luna._staged_path(staged_root, condition=condition, source=source_description)
    staged.parent.mkdir(parents=True)
    staged.write_bytes(raw.read_bytes())
    manifest_sha = _sha256(manifest)
    receipt = {
        "schema": HANDOFF_SCHEMA,
        "condition": condition,
        "task_index": 4,
        "identity": {
            "kind": IDENTITY[0],
            "regime": IDENTITY[1],
            "population": IDENTITY[2],
            "dataset": IDENTITY[3],
            "bias_type": IDENTITY[4],
        },
        "raw_log": {"path": str(raw.resolve()), "sha256": _sha256(raw)},
        "staged_log": {"path": str(staged.resolve()), "sha256": _sha256(staged)},
        "paired_clean": {"path": str(clean.resolve()), "sha256": _sha256(clean)},
        "raw_log_dir": str(raw_root.resolve()),
        "manifest": {"path": str(manifest.resolve()), "sha256": manifest_sha},
        "checkpoint": _checkpoint(condition, root / "checkpoint" / condition_spec(condition).checkpoint_name),
        "protocol": {
            "runtime_profile": "hf-peft",
            "prompt_style": "none",
            "include_bias_acknowledged": False,
            "max_tokens": 20480,
            "max_connections": 8,
            "validated_with_full_paired_switch_scores": True,
        },
    }
    receipt_path = staged.with_suffix(".resume-handoff.json")
    _write_json(receipt_path, receipt)
    source_sha = _sha256(staged)
    paired_full = {
        "raw_log": str(clean.resolve()),
        "raw_log_sha256": _sha256(clean),
        "question_ids_sha256": "c" * 64,
        "source_identity_digest": f"stage2-ood-hle-2x2:{'d' * 64}",
    }
    if binding_kind == "raw_preflight":
        checkpoint_path = root / "checkpoint" / condition_spec(condition).checkpoint_name
        report = {
            "schema": "stage2-ood-hle-raw-preflight-v1",
            "condition": condition,
            "manifest_sha256": manifest_sha,
            "contract": {
                "runtime": {
                    "profile": "hf-peft",
                    "expected_checkpoint": str(checkpoint_path.resolve()),
                }
            },
            "sources": [
                {
                    "kind": IDENTITY[0],
                    "regime": IDENTITY[1],
                    "population": IDENTITY[2],
                    "dataset": IDENTITY[3],
                    "bias_type": IDENTITY[4],
                    "raw_log": str(raw.resolve()),
                    "raw_log_sha256": source_sha,
                    "paired_clean": paired_full,
                    "runtime": {"checkpoint": str(checkpoint_path.resolve())},
                }
            ],
        }
        report_path = preflight_root / f"{condition}.json"
        _write_json(report_path, report)
        source = grade_luna.GradeInput(
            path=staged,
            condition=condition,
            regime=IDENTITY[1],
            population=IDENTITY[2],
            dataset=IDENTITY[3],
            bias_type=IDENTITY[4],
            created="",
            expected_sha256=source_sha,
            preflight_report_sha256=_sha256(report_path),
            manifest_sha256=manifest_sha,
            paired_clean=paired_full,
        )
        monkeypatch.setattr(promotion.raw_preflight, "validate_preflight_report", lambda _path: report)
    elif binding_kind == "incremental_handoff":
        source = grade_luna.GradeInput(
            path=staged,
            condition=condition,
            regime=IDENTITY[1],
            population=IDENTITY[2],
            dataset=IDENTITY[3],
            bias_type=IDENTITY[4],
            created="",
            expected_sha256=source_sha,
            preflight_report_sha256=_sha256(receipt_path),
            manifest_sha256=manifest_sha,
            paired_clean={"raw_log": str(clean.resolve()), "raw_log_sha256": _sha256(clean)},
            binding_kind="incremental_handoff",
            binding_sha256=_sha256(receipt_path),
            binding_path=str(receipt_path.resolve()),
            binding_schema=HANDOFF_SCHEMA,
        )
    else:  # pragma: no cover - fixture guard
        raise AssertionError(binding_kind)
    shard = promotion._canonical_shard_index(condition=condition, identity=IDENTITY)
    provenance = grade_luna._provenance(
        source,
        source_sha256=source_sha,
        smoke_samples=None,
        worker_count=5,
        connections_per_worker=100,
        grader_max_tokens=256,
        shard_index=shard,
    )
    eval_path, rows_path, provenance_path = grade_luna.output_paths(derived_root, source)
    eval_path.parent.mkdir(parents=True)
    eval_path.write_bytes(f"derived {condition}".encode("utf-8"))
    rows = grade_luna._export_rows(_scored_log(), source)
    rows_path.write_bytes(b"".join((json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode("utf-8") for row in rows))
    _write_json(provenance_path, provenance)
    return promotion.SourceRoots(
        derived_root=derived_root,
        staged_root=staged_root,
        raw_root=raw_root,
        manifest=manifest,
        preflight_root=preflight_root,
    )


def test_promote_combines_historical_and_incremental_trios_without_grading(tmp_path, monkeypatch):
    historical = _bundle(
        tmp_path,
        monkeypatch,
        name="historical",
        condition="bct-hf-peft",
        binding_kind="raw_preflight",
    )
    incremental = _bundle(
        tmp_path,
        monkeypatch,
        name="incremental",
        condition="bct-control-hf-peft",
        binding_kind="incremental_handoff",
    )
    log = _scored_log()
    observed: list[Path] = []

    def fake_read(path: Path):
        observed.append(path)
        return log

    monkeypatch.setattr(promotion, "_read_eval_log", fake_read)
    destination = tmp_path / "canonical-luna-derived-v1"
    source_before = {
        path: path.read_bytes()
        for roots in (historical, incremental)
        for path in roots.derived_root.rglob("*")
        if path.is_file()
    }

    results = promotion.promote((historical, incremental), destination)

    assert [result.status for result in results] == ["promoted", "promoted"]
    assert len(observed) == 2
    for roots in (historical, incremental):
        for source in roots.derived_root.rglob("*"):
            if source.is_file():
                target = destination / source.relative_to(roots.derived_root)
                assert target.read_bytes() == source.read_bytes()
    assert {path: path.read_bytes() for path in source_before} == source_before
    assert all("score" not in repr(path).lower() for path in observed)

    resumed = promotion.promote((historical, incremental), destination)
    assert [result.status for result in resumed] == ["resumed", "resumed"]


def test_promote_refuses_policy_drift_before_any_destination_write(tmp_path, monkeypatch):
    roots = _bundle(
        tmp_path,
        monkeypatch,
        name="incremental",
        condition="bct-hf-peft",
        binding_kind="incremental_handoff",
    )
    provenance_path = next(roots.derived_root.rglob("*-luna.provenance.json"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["worker_count"] = 4
    _write_json(provenance_path, provenance)
    monkeypatch.setattr(promotion, "_read_eval_log", lambda _path: (_ for _ in ()).throw(AssertionError("no EvalLog should be read")))
    destination = tmp_path / "destination"

    with pytest.raises(ValueError, match="worker_count differs from the approved Luna policy"):
        promotion.promote((roots,), destination)

    assert not destination.exists()


def test_promote_refuses_receipt_paired_clean_drift_and_never_overwrites(tmp_path, monkeypatch):
    roots = _bundle(
        tmp_path,
        monkeypatch,
        name="incremental",
        condition="bct-hf-peft",
        binding_kind="incremental_handoff",
    )
    monkeypatch.setattr(promotion, "_read_eval_log", lambda _path: _scored_log())
    destination = tmp_path / "destination"
    promotion.promote((roots,), destination)
    destination_provenance = next(destination.rglob("*-luna.provenance.json"))
    destination_provenance.write_bytes(b"different existing evidence")

    with pytest.raises(FileExistsError, match="destination collision differs"):
        promotion.promote((roots,), destination)

    assert destination_provenance.read_bytes() == b"different existing evidence"

    receipt_path = next(roots.staged_root.rglob("*.resume-handoff.json"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["paired_clean"]["sha256"] = "e" * 64
    _write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="adjacent receipt SHA-256 differs from provenance"):
        promotion.promote((roots,), tmp_path / "fresh-destination")


def test_promote_requires_exact_equivalence_for_source_collisions(tmp_path, monkeypatch):
    roots = _bundle(
        tmp_path,
        monkeypatch,
        name="first",
        condition="bct-hf-peft",
        binding_kind="incremental_handoff",
    )
    duplicate_root = tmp_path / "duplicate-derived"
    shutil.copytree(roots.derived_root, duplicate_root)
    duplicate = promotion.SourceRoots(
        derived_root=duplicate_root,
        staged_root=roots.staged_root,
        raw_root=roots.raw_root,
        manifest=roots.manifest,
        preflight_root=roots.preflight_root,
    )
    monkeypatch.setattr(promotion, "_read_eval_log", lambda _path: _scored_log())

    equivalent = promotion.promote((roots, duplicate), tmp_path / "equivalent-destination")
    assert sorted(result.status for result in equivalent) == ["equivalent", "promoted"]

    duplicate_eval = next(duplicate_root.rglob("*-luna.eval"))
    # The fake EvalLog reader above still supplies the same valid logical log;
    # changing the stored bytes isolates the immutable-artifact collision gate.
    duplicate_eval.write_bytes(b"different immutable derived bytes")
    with pytest.raises(FileExistsError, match="non-equivalent source collisions"):
        promotion.promote((roots, duplicate), tmp_path / "conflicting-destination")
    assert not (tmp_path / "conflicting-destination").exists()
