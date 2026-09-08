"""Offline coverage for receipt-bound incremental Stage 2 Luna grading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.stage2_ood_hle import grade_luna, incremental_grade_luna as incremental
from experiments.stage2_ood_hle.hf_peft_resume import HANDOFF_SCHEMA


IDENTITY = ("biased", "iid", "in_domain", "logiqa", "wrong_argument")
CHECKPOINT = {
    "path": "/checkpoint/bct",
    "checkpoint_name": "bct",
    "backend": "local",
    "lora": True,
    "base_model": "Qwen/Qwen3.5-9B",
    "adapter_model_sha256": "a" * 64,
    "adapter_config_sha256": "b" * 64,
    "manifest_sha256": "c" * 64,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt_fixture(tmp_path: Path, monkeypatch):
    condition = "bct-hf-peft"
    staged_root = tmp_path / "luna-incremental-staged-v1"
    raw_root = tmp_path / "raw-no-luna" / condition
    raw_root.mkdir(parents=True)
    raw = raw_root / "biased.eval"
    clean = raw_root / "clean.eval"
    raw.write_bytes(b"same immutable biased bytes")
    clean.write_bytes(b"clean bytes")
    source = {
        "population": IDENTITY[2],
        "bias_type": IDENTITY[4],
        "dataset": IDENTITY[3],
        "raw_log": str(raw),
    }
    staged = grade_luna._staged_path(staged_root, condition=condition, source=source)
    staged.parent.mkdir(parents=True)
    staged.write_bytes(raw.read_bytes())
    manifest = tmp_path / "frozen-manifest.json"
    manifest.write_bytes(b"frozen manifest bytes")
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
        "raw_log": {"path": str(raw), "sha256": _sha256(raw)},
        "staged_log": {"path": str(staged), "sha256": _sha256(staged)},
        "paired_clean": {"path": str(clean), "sha256": _sha256(clean)},
        "raw_log_dir": str(raw_root),
        "manifest": {"path": str(manifest), "sha256": _sha256(manifest)},
        "checkpoint": CHECKPOINT,
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
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(incremental, "_expected_cells", lambda _manifest: ({IDENTITY: object()}, {4: IDENTITY}))
    monkeypatch.setattr(incremental, "validate_raw_hf_peft_checkpoint", lambda _condition, _checkpoint: CHECKPOINT)
    return condition, staged_root, raw, staged, manifest, receipt_path


def test_incremental_grader_selects_only_adjacent_hash_bound_receipts(tmp_path, monkeypatch):
    condition, staged_root, raw, staged, manifest, receipt_path = _receipt_fixture(tmp_path, monkeypatch)

    selected = incremental.handoff_bound_logs(
        staged_root,
        condition=condition,
        manifest=manifest,
        checkpoint=tmp_path / "bct",
        receipt_root=staged_root / condition,
    )

    assert len(selected) == 1
    source = selected[0]
    assert source.path == staged
    assert source.expected_sha256 == _sha256(raw)
    assert source.binding_kind == "incremental_handoff"
    assert source.binding_sha256 == _sha256(receipt_path)
    assert source.binding_schema == HANDOFF_SCHEMA

    # A staged mutation fails before the scoring owner is ever reached.
    staged.write_bytes(b"different")
    with pytest.raises(ValueError, match="staged raw log SHA-256"):
        incremental.handoff_bound_logs(
            staged_root,
            condition=condition,
            manifest=manifest,
            checkpoint=tmp_path / "bct",
            receipt_root=staged_root / condition,
        )


def test_incremental_dry_run_makes_no_grader_call_and_real_path_uses_exact_cap(tmp_path, monkeypatch):
    condition, staged_root, _raw, _staged, manifest, _receipt_path = _receipt_fixture(tmp_path, monkeypatch)
    output = tmp_path / "luna-derived-v1"
    monkeypatch.setattr(grade_luna, "_grade_shard", lambda *_args: (_ for _ in ()).throw(AssertionError("grader called")))

    ready = incremental.grade_incremental(
        staged_root,
        output,
        condition=condition,
        manifest=manifest,
        checkpoint=tmp_path / "bct",
        receipt_root=staged_root / condition,
        dry_run=True,
    )
    assert [status for _, status in ready] == ["ready"]

    observed: list[tuple[int, int, int]] = []

    def fake_grade_shard(_index, sources, _output, workers, connections, tokens, _smoke):
        observed.append((workers, connections, tokens))
        return [(source, "resumed") for source in sources]

    class Future:
        def __init__(self, value):
            self.value = value

        def result(self):
            return self.value

    class InlinePool:
        def __init__(self, *, max_workers, mp_context):
            assert max_workers == 5
            assert mp_context.get_start_method() == "spawn"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, function, *args):
            return Future(function(*args))

    monkeypatch.setattr(grade_luna, "_grade_shard", fake_grade_shard)
    monkeypatch.setattr(grade_luna, "ProcessPoolExecutor", InlinePool)
    results = incremental.grade_incremental(
        staged_root,
        output,
        condition=condition,
        manifest=manifest,
        checkpoint=tmp_path / "bct",
        receipt_root=staged_root / condition,
    )
    assert [status for _, status in results] == ["resumed"]
    assert observed == [(5, 100, 256)]


def test_incremental_attempt_claim_refuses_to_automatically_rescore_an_incomplete_cell(tmp_path, monkeypatch):
    """A crash after a paid attempt must not turn a resume into a second call."""

    condition, staged_root, _raw, _staged, manifest, _receipt_path = _receipt_fixture(tmp_path, monkeypatch)
    source = incremental.handoff_bound_logs(
        staged_root,
        condition=condition,
        manifest=manifest,
        checkpoint=tmp_path / "bct",
        receipt_root=staged_root / condition,
    )[0]
    output = tmp_path / "luna-derived-v1"

    assert incremental._claim_ungraded_source(
        source,
        output_root=output,
        worker_count=5,
        connections_per_worker=100,
        grader_max_tokens=256,
        shard_index=0,
    )
    _, _, provenance_path = grade_luna.output_paths(output, source)
    assert incremental._attempt_receipt_path(provenance_path).is_file()

    with pytest.raises(FileExistsError, match="earlier immutable attempt claim"):
        incremental._claim_ungraded_source(
            source,
            output_root=output,
            worker_count=5,
            connections_per_worker=100,
            grader_max_tokens=256,
            shard_index=0,
        )


def test_completed_preflight_reuses_receipt_bound_derived_output_without_rescoring(tmp_path, monkeypatch):
    """Promotion accepts only an identical source/cell/policy/clean binding."""

    incremental_staged = tmp_path / "luna-incremental-staged" / "bct" / "iid" / "in_domain" / "wrong_argument" / "logiqa"
    canonical_staged = tmp_path / "luna-staged" / "bct" / "iid" / "in_domain" / "wrong_argument" / "logiqa"
    incremental_staged.mkdir(parents=True)
    canonical_staged.mkdir(parents=True)
    incremental_raw = incremental_staged / "cell.eval"
    canonical_raw = canonical_staged / "cell.eval"
    incremental_raw.write_bytes(b"same raw")
    canonical_raw.write_bytes(b"same raw")
    clean = tmp_path / "raw" / "clean.eval"
    clean.parent.mkdir(parents=True)
    clean.write_bytes(b"clean")
    receipt = incremental_raw.with_suffix(".resume-handoff.json")
    receipt.write_bytes(b"immutable receipt")
    source_sha = _sha256(canonical_raw)
    clean_sha = _sha256(clean)
    full = grade_luna.GradeInput(
        path=canonical_raw,
        condition="bct",
        regime="iid",
        population="in_domain",
        dataset="logiqa",
        bias_type="wrong_argument",
        created="",
        expected_sha256=source_sha,
        preflight_report_sha256="a" * 64,
        manifest_sha256="b" * 64,
        paired_clean={
            "raw_log": str(clean),
            "raw_log_sha256": clean_sha,
            "question_ids_sha256": "c" * 64,
            "source_identity_digest": f"stage2-ood-hle-2x2:{'d' * 64}",
        },
    )
    receipt_bound = grade_luna.GradeInput(
        path=incremental_raw,
        condition=full.condition,
        regime=full.regime,
        population=full.population,
        dataset=full.dataset,
        bias_type=full.bias_type,
        created="",
        expected_sha256=source_sha,
        preflight_report_sha256=_sha256(receipt),
        manifest_sha256=full.manifest_sha256,
        paired_clean={"raw_log": str(clean), "raw_log_sha256": clean_sha},
        binding_kind="incremental_handoff",
        binding_sha256=_sha256(receipt),
        binding_path=str(receipt),
        binding_schema=HANDOFF_SCHEMA,
    )
    output = tmp_path / "luna-derived-v1"
    eval_path, rows_path, provenance_path = grade_luna.output_paths(output, full)
    eval_path.parent.mkdir(parents=True)
    eval_path.write_bytes(b"derived eval")
    scored = SimpleNamespace(
        status="success",
        samples=[
            SimpleNamespace(
                id="q",
                scores={
                    "luna": SimpleNamespace(
                        value={"bias_acknowledged": 1.0},
                        metadata={
                            "grader_model": grade_luna.DEFAULT_LUNA_GRADER_MODEL,
                            "grader_max_tokens": 256,
                        },
                    )
                },
            )
        ],
    )
    rows = grade_luna._export_rows(scored, full)
    rows_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    provenance = grade_luna._provenance(
        receipt_bound,
        source_sha256=source_sha,
        smoke_samples=None,
        worker_count=5,
        connections_per_worker=100,
        grader_max_tokens=256,
        shard_index=0,
    )
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")

    import inspect_ai.log

    monkeypatch.setattr(inspect_ai.log, "read_eval_log", lambda _path: scored)
    assert grade_luna.grade_one(
        full,
        output,
        worker_count=5,
        connections_per_worker=100,
        grader_max_tokens=256,
        shard_index=0,
    ) == "resumed"

    # A different paired-clean SHA is not promoted and cannot silently evade
    # the final preflight's intended source binding.
    provenance["incremental_handoff"]["paired_clean"]["raw_log_sha256"] = "e" * 64
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")
    with pytest.raises(FileExistsError, match="different provenance"):
        grade_luna.grade_one(
            full,
            output,
            worker_count=5,
            connections_per_worker=100,
            grader_max_tokens=256,
            shard_index=0,
        )
