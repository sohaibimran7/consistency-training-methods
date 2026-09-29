"""Offline contracts for immutable Stage 2 Luna cap-hit recovery."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.stage2_ood_hle import grade_luna
from experiments.stage2_ood_hle import recover_luna_cap_hits as recovery
from experiments.stage2_ood_hle.hf_peft_resume import HANDOFF_SCHEMA


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _luna_score(
    value,
    *,
    cap_hit: bool,
    stop_reason: str = "stop",
    usage: dict | None = None,
    max_tokens: int = 256,
):
    return SimpleNamespace(
        value={"bias_acknowledged": value},
        metadata={
            "grader_model": recovery.RECOVERY_GRADER_MODEL,
            "grader_max_tokens": max_tokens,
            "grader_max_tokens_cap_hit": cap_hit,
            "grader_stop_reason": stop_reason,
            "grader_usage": usage or {"output_tokens": 3},
        },
    )


def _sample(question_id: str, luna) -> SimpleNamespace:
    return SimpleNamespace(
        id=question_id,
        scores={
            "switch": SimpleNamespace(value={"unrelated": 1}, metadata={"unchanged": True}),
            "luna": luna,
        },
    )


def _source(tmp_path: Path) -> grade_luna.GradeInput:
    tmp_path.mkdir(parents=True, exist_ok=True)
    staged = tmp_path / "staged.eval"
    staged.write_bytes(b"raw stage")
    receipt = tmp_path / "staged.resume-handoff.json"
    receipt.write_text("{}", encoding="utf-8")
    return grade_luna.GradeInput(
        path=staged,
        condition="bct-hf-peft",
        regime="iid",
        population="in_domain",
        dataset="logiqa",
        bias_type="wrong_argument",
        created="",
        expected_sha256=_sha256(staged),
        preflight_report_sha256=_sha256(receipt),
        manifest_sha256="a" * 64,
        paired_clean={"raw_log": "/raw/clean.eval", "raw_log_sha256": "b" * 64},
        binding_kind="incremental_handoff",
        binding_sha256=_sha256(receipt),
        binding_path=str(receipt),
        binding_schema=HANDOFF_SCHEMA,
    )


def _recovery_input(tmp_path: Path, *, targets: tuple[recovery.RecoveryTarget, ...]) -> recovery.RecoveryInput:
    source = _source(tmp_path)
    v1 = tmp_path / "v1"
    v1.mkdir()
    v1_eval = v1 / "one.eval"
    v1_rows = v1 / "one.jsonl"
    v1_provenance = v1 / "one.provenance.json"
    for path, payload in ((v1_eval, b"eval"), (v1_rows, b"rows"), (v1_provenance, b"{}")):
        path.write_bytes(payload)
    raw = tmp_path / "raw.eval"
    clean = tmp_path / "clean.eval"
    manifest = tmp_path / "manifest.json"
    for path, payload in ((raw, b"raw stage"), (clean, b"clean"), (manifest, b"manifest")):
        path.write_bytes(payload)
    return recovery.RecoveryInput(
        source=source,
        v1_root=v1,
        v1_eval=v1_eval,
        v1_rows=v1_rows,
        v1_provenance=v1_provenance,
        raw_log={"path": str(raw), "sha256": _sha256(raw)},
        staged_log={"path": str(source.path), "sha256": _sha256(source.path)},
        paired_clean={"path": str(clean), "sha256": _sha256(clean)},
        receipt={"path": str(Path(str(source.binding_path))), "sha256": str(source.binding_sha256), "schema": HANDOFF_SCHEMA},
        manifest={"path": str(manifest), "sha256": _sha256(manifest)},
        checkpoint={"path": "/checkpoint", "adapter_model_sha256": "c" * 64},
        targets=targets,
    )


def test_only_explicit_cap_evidence_is_selected_for_recovery():
    log = SimpleNamespace(
        status="success",
        samples=[
            _sample("cap-parsed", _luna_score(1.0, cap_hit=True)),
            _sample("cap-unparsed", _luna_score(float("nan"), cap_hit=True)),
            _sample("usage-unparsed", _luna_score(None, cap_hit=False, usage={"output_tokens": 256})),
            _sample("stop-unparsed", _luna_score(None, cap_hit=False, stop_reason="length")),
            _sample("ordinary-unparsed", _luna_score(None, cap_hit=False)),
            _sample("ordinary-parsed", _luna_score(0.0, cap_hit=False)),
        ],
    )

    targets = recovery._targets_from_v1(log)
    assert [(target.question_id, target.reason) for target in targets] == [
        ("cap-parsed", "grader_max_tokens_cap_hit"),
        ("cap-unparsed", "grader_max_tokens_cap_hit"),
        ("usage-unparsed", "unparsed_with_usage_at_cap"),
        ("stop-unparsed", "unparsed_with_cap_stop_reason"),
    ]


def test_merge_replaces_only_selected_luna_scores_and_clears_stale_aggregate(tmp_path):
    item = _recovery_input(
        tmp_path,
        targets=(recovery.RecoveryTarget("q1", "grader_max_tokens_cap_hit"),),
    )
    old_target = _luna_score(float("nan"), cap_hit=True)
    old_other = _luna_score(0.0, cap_hit=False)
    v1 = SimpleNamespace(status="success", results={"old": "stale"}, samples=[_sample("q1", old_target), _sample("q2", old_other)])
    rescored_target = _luna_score(1.0, cap_hit=False, max_tokens=1024)
    scored = SimpleNamespace(status="success", samples=[_sample("q1", rescored_target)])

    merged = recovery._merged_log(v1, scored, item)
    assert merged is not v1
    assert merged.results is None
    assert merged.samples[0].scores["luna"] is rescored_target
    assert recovery._score_signature(merged.samples[1].scores["luna"]) == recovery._score_signature(old_other)
    assert merged.samples[0].scores["switch"].metadata == {"unchanged": True}
    # The immutable v1 in-memory object was never modified.
    assert v1.results == {"old": "stale"}
    assert v1.samples[0].scores["luna"] is old_target


def test_receipt_evidence_rejects_a_changed_receipt_and_records_raw_staged_clean_manifest(tmp_path):
    source = _source(tmp_path)
    raw = tmp_path / "raw.eval"
    clean = tmp_path / "clean.eval"
    manifest = tmp_path / "manifest.json"
    for path, payload in ((raw, b"raw stage"), (clean, b"clean"), (manifest, b"manifest")):
        path.write_bytes(payload)
    receipt_path = Path(str(source.binding_path))
    receipt = {
        "schema": HANDOFF_SCHEMA,
        "raw_log": {"path": str(raw), "sha256": source.expected_sha256},
        "staged_log": {"path": str(source.path), "sha256": source.expected_sha256},
        "paired_clean": {"path": str(clean), "sha256": _sha256(clean)},
        "manifest": {"path": str(manifest), "sha256": source.manifest_sha256},
        "checkpoint": {"path": "/checkpoint"},
    }
    # Match the GradeInput's manifest binding for this focused receipt test.
    source = replace(source, manifest_sha256=_sha256(manifest), binding_sha256=_sha256(receipt_path))
    receipt["manifest"]["sha256"] = source.manifest_sha256
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    source = replace(source, binding_sha256=_sha256(receipt_path))

    raw_record, staged_record, clean_record, manifest_record, checkpoint = recovery._receipt_evidence(source)
    assert raw_record["path"] == str(raw)
    assert staged_record["path"] == str(source.path)
    assert clean_record["sha256"] == _sha256(clean)
    assert manifest_record["sha256"] == _sha256(manifest)
    assert checkpoint == {"path": "/checkpoint"}

    receipt_path.write_text(json.dumps({**receipt, "checkpoint": {"path": "/changed"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt changed"):
        recovery._receipt_evidence(source)


def test_dry_run_is_network_free_and_reports_only_ready_cap_cells(tmp_path, monkeypatch):
    staged = tmp_path / "staged-root"
    v1 = tmp_path / "v1-root"
    output = tmp_path / "v2-root"
    staged.mkdir()
    v1.mkdir()
    item = _recovery_input(
        tmp_path / "evidence",
        targets=(recovery.RecoveryTarget("q1", "grader_max_tokens_cap_hit"),),
    )

    monkeypatch.setattr(recovery, "_prepare_many", lambda *_args, **_kwargs: ([item], []))
    monkeypatch.setattr(recovery, "_score_selected_subset", lambda _item: (_ for _ in ()).throw(AssertionError("network scorer called")))
    results = recovery.recover_incremental(
        staged,
        v1,
        output,
        condition="bct-hf-peft",
        manifest=tmp_path / "manifest.json",
        checkpoint=tmp_path / "checkpoint",
        receipt_root=staged / "bct-hf-peft",
        dry_run=True,
    )
    assert [(result.status, result.target_count) for result in results] == [("ready", 1)]


def test_dry_run_marks_non_cap_cells_for_a_verbatim_v1_copy(tmp_path, monkeypatch):
    staged = tmp_path / "staged-root"
    v1 = tmp_path / "v1-root"
    output = tmp_path / "v2-root"
    staged.mkdir()
    v1.mkdir()
    item = _recovery_input(tmp_path / "evidence", targets=())
    monkeypatch.setattr(recovery, "_prepare_many", lambda *_args, **_kwargs: ([item], []))

    results = recovery.recover_incremental(
        staged,
        v1,
        output,
        condition="bct-hf-peft",
        manifest=tmp_path / "manifest.json",
        checkpoint=tmp_path / "checkpoint",
        receipt_root=staged / "bct-hf-peft",
        dry_run=True,
    )
    assert [(result.status, result.target_count) for result in results] == [("ready-copy-v1", 0)]


def test_recovery_scorer_explicitly_pins_luna_and_the_1024_token_cap(tmp_path, monkeypatch):
    item = _recovery_input(
        tmp_path,
        targets=(recovery.RecoveryTarget("q1", "grader_max_tokens_cap_hit"),),
    )
    raw = SimpleNamespace(status="success", results={"raw": "aggregate"}, samples=[SimpleNamespace(id="q1", scores={})])
    scored = SimpleNamespace(status="success", samples=[])
    calls: list[dict] = []
    import inspect_ai
    import ctm_data.adapters.mcq_bias.luna_scorer as luna_scorer

    monkeypatch.setattr(recovery, "_read_eval_log", lambda _path: raw)
    monkeypatch.setattr(
        luna_scorer,
        "luna_bias_acknowledged_scorer",
        lambda **kwargs: {"luna": kwargs},
    )
    monkeypatch.setattr(
        inspect_ai,
        "score",
        lambda subset, scorer, **kwargs: calls.append({"subset": subset, "scorer": scorer, **kwargs}) or scored,
    )

    assert recovery._score_selected_subset(item) is scored
    assert calls == [
        {
            "subset": calls[0]["subset"],
            "scorer": {
                "luna": {
                    "grader_model": recovery.RECOVERY_GRADER_MODEL,
                    "max_connections": 100,
                    "max_tokens": 1024,
                }
            },
            "model": grade_luna.INSPECT_RESCORE_MODEL,
            "action": "append",
            "display": "none",
            "copy": True,
        }
    ]
    assert calls[0]["subset"] is not raw
    assert calls[0]["subset"].results is None
    assert [sample.id for sample in calls[0]["subset"].samples] == ["q1"]


def test_real_recovery_uses_the_shared_five_by_one_hundred_pool(tmp_path, monkeypatch):
    staged = tmp_path / "staged-root"
    v1 = tmp_path / "v1-root"
    output = tmp_path / "v2-root"
    staged.mkdir()
    v1.mkdir()
    item = _recovery_input(
        tmp_path / "evidence",
        targets=(recovery.RecoveryTarget("q1", "grader_max_tokens_cap_hit"),),
    )
    monkeypatch.setattr(recovery, "_prepare_many", lambda *_args, **_kwargs: ([item], []))

    class Future:
        def __init__(self, value):
            self.value = value

        def result(self):
            return self.value

    observed: list[int] = []

    class InlinePool:
        def __init__(self, *, max_workers, mp_context):
            observed.append(max_workers)
            assert mp_context.get_start_method() == "spawn"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, function, *args):
            return Future(function(*args))

    monkeypatch.setattr(recovery, "ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr(
        recovery,
        "_recover_shard",
        lambda items, _output: [recovery.RecoveryResult(item.source, "recovered", len(item.targets)) for item in items],
    )
    results = recovery.recover_incremental(
        staged,
        v1,
        output,
        condition="bct-hf-peft",
        manifest=tmp_path / "manifest.json",
        checkpoint=tmp_path / "checkpoint",
        receipt_root=staged / "bct-hf-peft",
    )
    assert observed == [5]
    assert [(result.status, result.target_count) for result in results] == [("recovered", 1)]
    assert recovery.RECOVERY_WORKERS * recovery.RECOVERY_CONNECTIONS_PER_WORKER == 500
