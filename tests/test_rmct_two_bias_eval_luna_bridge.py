"""Focused offline tests for the r005 post-hoc Luna custody bridge."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.rmct_two_bias_eval import luna_bridge as bridge
from experiments.rmct_two_bias_eval.contract import HELD_OUT_BIASES, SEEN_BIASES


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Make a byte-bound r005 task-N layout without needing Inspect or Luna."""

    root = tmp_path / "r005"
    raw_root = root / "stage2" / "raw"
    receipt_root = root / "stage2" / "receipts"
    attempt_root = root / "stage2" / "attempts" / "preserved"
    raw_root.mkdir(parents=True)
    launch = root / "launch-contract.json"
    launch.write_bytes(b"sealed r005 launch contract\n")
    evaluation_receipt_path = root / "runtime" / "evaluation-receipt.json"
    evaluation_receipt_path.parent.mkdir(parents=True)
    evaluation_receipt_path.write_bytes(b"sealed r005 evaluation receipt\n")
    preflight_path = root / "stage2" / "preflight" / f"{bridge.R005_CONDITION}.json"
    preflight_path.parent.mkdir(parents=True)
    preflight_path.write_bytes(b"sealed r005 preflight report\n")

    clean_cells = (
        ("in_domain", "logiqa"),
        ("in_domain", "hellaswag"),
        ("hle", "hle"),
    )
    clean_records: dict[tuple[str, str], dict[str, object]] = {}
    sources: list[dict[str, object]] = []
    raw_paths: dict[int, Path] = {}
    for index, (population, dataset) in enumerate(clean_cells, start=1):
        payload = f"clean task {index}\n".encode()
        digest = hashlib.sha256(payload).hexdigest()
        path = raw_root / f"task-{index:03d}" / f"{digest}.eval"
        path.parent.mkdir(parents=True)
        path.write_bytes(payload)
        raw_paths[index] = path
        source = {
            "kind": "unbiased",
            "regime": "clean",
            "population": population,
            "dataset": dataset,
            "bias_type": None,
            "raw_log": str(path.resolve()),
            "raw_log_sha256": digest,
            "sample_count": 100,
            "evaluation_bias_status": None,
        }
        sources.append(source)
        clean_records[(population, dataset)] = {
            "raw_log": str(path.resolve()),
            "raw_log_sha256": digest,
        }

    task_index = 4
    for bias_type in (*SEEN_BIASES, *HELD_OUT_BIASES):
        for population, dataset in clean_cells:
            payload = f"biased task {task_index} {bias_type}\n".encode()
            digest = hashlib.sha256(payload).hexdigest()
            path = raw_root / f"task-{task_index:03d}" / f"{digest}.eval"
            path.parent.mkdir(parents=True)
            path.write_bytes(payload)
            raw_paths[task_index] = path
            sources.append(
                {
                    "kind": "biased",
                    "regime": "legacy-layout-only",
                    "population": population,
                    "dataset": dataset,
                    "bias_type": bias_type,
                    "raw_log": str(path.resolve()),
                    "raw_log_sha256": digest,
                    "sample_count": 100,
                    "evaluation_bias_status": "seen" if bias_type in SEEN_BIASES else "held_out",
                    "paired_clean": dict(clean_records[(population, dataset)]),
                }
            )
            task_index += 1

    launch_sha = _sha256(launch)
    receipt_sha = _sha256(evaluation_receipt_path)
    for index, raw_path in raw_paths.items():
        attempt = attempt_root / f"task-{index:03d}" / "attempt.eval"
        attempt.parent.mkdir(parents=True, exist_ok=True)
        attempt.write_bytes(f"attempt {index}\n".encode())
        _write_json(
            receipt_root / f"task-{index:03d}.json",
            {
                "schema": bridge.R005_TASK_RECEIPT_SCHEMA,
                "task_index": index,
                "launch_contract_sha256": launch_sha,
                "evaluation_receipt_sha256": receipt_sha,
                "canonical_log": _identity(raw_path),
                "attempt_log": _identity(attempt),
            },
        )

    receipt = {
        "condition": bridge.R005_CONDITION,
        "raw_generation": {"raw_log_root": str(raw_root.resolve())},
        "runtime": {"profile": "vllm", "exact": "r005"},
        "science": {
            "seen_biases": list(SEEN_BIASES),
            "held_out_biases": list(HELD_OUT_BIASES),
            "bias_status_source": "bias_type_not_legacy_substrate_regime",
            "preserve_per_bias_results": True,
        },
    }
    report = {
        "condition": bridge.R005_CONDITION,
        "raw_root": str(raw_root.resolve()),
        "evaluation_receipt": {"path": str(evaluation_receipt_path.resolve()), "sha256": receipt_sha},
        "contract": {"runtime": receipt["runtime"]},
        "sources": sources,
    }
    calls: list[Path] = []

    def verify(path: Path):
        calls.append(Path(path).resolve())
        return receipt

    monkeypatch.setattr(bridge.contract, "verify_evaluation_receipt", verify)
    monkeypatch.setattr(bridge.contract, "validate_evaluation_receipt", lambda _path: receipt)
    monkeypatch.setattr(bridge.raw_preflight, "validate_preflight_report", lambda path: report)
    return {
        "root": root,
        "raw_root": raw_root,
        "evaluation_receipt": evaluation_receipt_path,
        "preflight": preflight_path,
        "output": tmp_path / "luna-derived-separate",
        "calls": calls,
    }


def test_bridge_selects_only_the_receipt_bound_biased_task_004_to_021_matrix(tmp_path, monkeypatch):
    layout = _fixture(tmp_path, monkeypatch)

    sources = bridge.build_grade_inputs(
        layout["raw_root"],
        evaluation_receipt=layout["evaluation_receipt"],
        preflight_report=layout["preflight"],
        output_root=layout["output"],
    )

    assert layout["calls"] == [layout["evaluation_receipt"].resolve()]
    assert [source.task_index for source in sources] == list(range(4, 22))
    assert len(sources) == 18
    assert {source.bias_type for source in sources if source.evaluation_bias_status == "seen"} == set(SEEN_BIASES)
    assert {source.bias_type for source in sources if source.evaluation_bias_status == "held_out"} == set(HELD_OUT_BIASES)
    assert all(source.staged_path.is_relative_to(layout["output"]) for source in sources)
    assert all(not source.staged_path.exists() for source in sources)


def test_dry_run_is_read_only_and_has_no_grader_side_effect(tmp_path, monkeypatch):
    layout = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(bridge, "grade_sources", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("grader called")))

    results = bridge.run_bridge(
        layout["raw_root"],
        evaluation_receipt=layout["evaluation_receipt"],
        preflight_report=layout["preflight"],
        output_root=layout["output"],
        mode="dry-run",
    )

    assert len(results) == 18
    assert {status for _, status in results} == {"ready"}
    assert not layout["output"].exists()


def test_bridge_rejects_an_output_namespace_inside_the_sealed_evaluation_root(tmp_path, monkeypatch):
    layout = _fixture(tmp_path, monkeypatch)

    with pytest.raises(bridge.LunaBridgeError, match="separate from"):
        bridge.build_grade_inputs(
            layout["raw_root"],
            evaluation_receipt=layout["evaluation_receipt"],
            preflight_report=layout["preflight"],
            output_root=layout["root"] / "unsafe-derived",
        )


def test_staging_is_resumable_and_refuses_a_changed_raw_source(tmp_path, monkeypatch):
    layout = _fixture(tmp_path, monkeypatch)
    source = bridge.build_grade_inputs(
        layout["raw_root"],
        evaluation_receipt=layout["evaluation_receipt"],
        preflight_report=layout["preflight"],
        output_root=layout["output"],
    )[0]

    assert bridge.stage_input(source) == "staged"
    assert bridge.stage_input(source) == "resumed"
    receipt = json.loads(source.staged_path.with_suffix(".staging.json").read_text(encoding="utf-8"))
    assert receipt["scientific_labels"]["seen_biases"] == list(SEEN_BIASES)
    assert receipt["scientific_labels"]["held_out_biases"] == list(HELD_OUT_BIASES)

    source.raw_path.write_bytes(b"changed after source selection")
    with pytest.raises(bridge.LunaBridgeError, match="changed before staging"):
        bridge.stage_input(source)


def _scored_log(sample_count: int) -> SimpleNamespace:
    return SimpleNamespace(
        status="success",
        samples=[
            SimpleNamespace(
                id=f"question-{index}",
                scores={
                    "luna_bias_acknowledged_scorer": SimpleNamespace(
                        value={"bias_acknowledged": float(index % 2)},
                        metadata={
                            "grader_model": bridge.DEFAULT_LUNA_GRADER_MODEL,
                            "grader_max_tokens": bridge.DEFAULT_MAX_TOKENS,
                            "grader_response": "Answer: YES",
                            "grader_usage": {"total_tokens": 3},
                            "grader_stop_reason": "stop",
                            "grader_max_tokens_cap_hit": False,
                        },
                    )
                },
            )
            for index in range(sample_count)
        ],
    )


def test_score_path_reuses_the_exact_pinned_luna_policy_and_resumes(tmp_path, monkeypatch):
    layout = _fixture(tmp_path, monkeypatch)
    source = bridge.build_grade_inputs(
        layout["raw_root"],
        evaluation_receipt=layout["evaluation_receipt"],
        preflight_report=layout["preflight"],
        output_root=layout["output"],
    )[0]
    bridge.stage_input(source)
    raw = SimpleNamespace(status="success", samples=[SimpleNamespace(id="raw", scores={}) for _ in range(100)])
    scored = _scored_log(100)
    calls: list[dict[str, object]] = []

    inspect_module = types.ModuleType("inspect_ai")
    inspect_module.__version__ = "test-inspect-1.2.3"  # type: ignore[attr-defined]
    inspect_module.score = lambda input_log, scorer, **kwargs: calls.append(  # type: ignore[attr-defined]
        {"input": input_log, "scorer": scorer, **kwargs}
    ) or scored
    log_module = types.ModuleType("inspect_ai.log")
    log_module.read_eval_log = lambda path: scored if str(path).endswith("-luna.eval") else raw  # type: ignore[attr-defined]
    log_module.write_eval_log = lambda _log, path: Path(path).write_bytes(b"derived eval")  # type: ignore[attr-defined]
    luna_module = types.ModuleType("ctm_data.adapters.mcq_bias.luna_scorer")
    luna_module.luna_bias_acknowledged_scorer = lambda **kwargs: {"pinned-luna": kwargs}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "inspect_ai", inspect_module)
    monkeypatch.setitem(sys.modules, "inspect_ai.log", log_module)
    monkeypatch.setitem(sys.modules, "ctm_data.adapters.mcq_bias.luna_scorer", luna_module)

    assert bridge.grade_one(source, layout["output"]) == "graded"
    assert calls == [
        {
            "input": raw,
            "scorer": {
                "pinned-luna": {
                    "grader_model": bridge.DEFAULT_LUNA_GRADER_MODEL,
                    "max_connections": 100,
                    "max_tokens": 256,
                }
            },
            "model": bridge.INSPECT_RESCORE_MODEL,
            "action": "append",
            "display": "none",
            "copy": True,
        }
    ]
    assert bridge.grade_one(source, layout["output"]) == "resumed"
    eval_path, rows_path, provenance_path = bridge.output_paths(layout["output"], source)
    assert eval_path.is_file() and rows_path.is_file()
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["scientific_labels"]["seen_biases"] == list(SEEN_BIASES)
    assert provenance["luna_policy"]["aggregate_connection_limit"] == bridge.DEFAULT_MAX_CONNECTIONS
    assert provenance["luna_policy"]["reasoning_effort"] == "low"
    assert provenance["luna_policy"]["inspect_version"] == "test-inspect-1.2.3"
    assert provenance["luna_policy"]["scorer_source"]["sha256"] == _sha256(bridge.LUNA_SCORER_SOURCE)
    assert provenance["luna_policy"]["config_source"]["sha256"] == _sha256(bridge.LUNA_CONFIG_SOURCE)
    assert bridge.DEFAULT_WORKERS * bridge.DEFAULT_CONNECTIONS_PER_WORKER == bridge.DEFAULT_MAX_CONNECTIONS


def test_an_incomplete_attempt_claim_is_never_automatically_rescored(tmp_path, monkeypatch):
    layout = _fixture(tmp_path, monkeypatch)
    source = bridge.build_grade_inputs(
        layout["raw_root"],
        evaluation_receipt=layout["evaluation_receipt"],
        preflight_report=layout["preflight"],
        output_root=layout["output"],
    )[0]
    bridge.stage_input(source)
    assert bridge._claim_ungraded_source(
        source,
        output_root=layout["output"],
        worker_count=bridge.DEFAULT_WORKERS,
        connections_per_worker=bridge.DEFAULT_CONNECTIONS_PER_WORKER,
        shard_index=0,
        smoke_samples=None,
        luna_policy=bridge._luna_policy_identity(inspect_version="test-inspect-1.2.3"),
    )
    with pytest.raises(FileExistsError, match="earlier immutable Luna attempt claim"):
        bridge._claim_ungraded_source(
            source,
            output_root=layout["output"],
            worker_count=bridge.DEFAULT_WORKERS,
            connections_per_worker=bridge.DEFAULT_CONNECTIONS_PER_WORKER,
            shard_index=0,
            smoke_samples=None,
            luna_policy=bridge._luna_policy_identity(inspect_version="test-inspect-1.2.3"),
        )


def test_portable_stage_only_bundle_revalidates_local_copies_without_reopening_source_paths(tmp_path, monkeypatch):
    layout = _fixture(tmp_path, monkeypatch)

    bundle, bundle_status, staged = bridge.stage_portable_bundle(
        layout["raw_root"],
        evaluation_receipt=layout["evaluation_receipt"],
        preflight_report=layout["preflight"],
        output_root=layout["output"],
    )

    assert bundle_status == "written"
    assert len(staged) == 18
    assert bundle.name == bridge.PORTABLE_BUNDLE_FILENAME
    document = json.loads(bundle.read_text(encoding="utf-8"))
    assert len(document["sources"]) == 18
    assert len(document["task_receipts"]) == 21
    assert document["scientific_labels"]["seen_biases"] == list(SEEN_BIASES)
    assert document["scientific_labels"]["held_out_biases"] == list(HELD_OUT_BIASES)

    transferred_root = tmp_path / "transferred-r005-portable-bundle"
    shutil.copytree(bundle.parent, transferred_root)
    bundle = transferred_root / bridge.PORTABLE_BUNDLE_FILENAME

    # A transferred bundle must not touch the source host's raw paths again.
    # Change the fixture source after capture, then make native receipt
    # validation fail loudly if the portable path tries to reopen it.
    first_native = layout["raw_root"] / "task-004"
    next(first_native.glob("*.eval")).write_bytes(b"native source is no longer available locally")
    monkeypatch.setattr(
        bridge.contract,
        "verify_evaluation_receipt",
        lambda _path: (_ for _ in ()).throw(AssertionError("portable validation reopened native custody")),
    )

    portable_sources = bridge.portable_grade_inputs(bundle)
    assert [source.task_index for source in portable_sources] == list(range(4, 22))
    assert all(source.portable_bundle_path == bundle for source in portable_sources)
    assert all(source.staged_path.is_relative_to(transferred_root) for source in portable_sources)
    bridge._verify_staged_source(portable_sources[0])

    local_output = tmp_path / "local-luna-derived"
    ready = bridge.grade_staged_bundle(bundle, local_output, mode="dry-run")
    assert len(ready) == 18
    assert {status for _, status in ready} == {"ready"}
    assert not local_output.exists()


def test_unreadable_staged_log_never_creates_a_paid_attempt_claim(tmp_path, monkeypatch):
    layout = _fixture(tmp_path, monkeypatch)
    source = bridge.build_grade_inputs(
        layout["raw_root"],
        evaluation_receipt=layout["evaluation_receipt"],
        preflight_report=layout["preflight"],
        output_root=layout["output"],
    )[0]
    bridge.stage_input(source)
    score_calls: list[object] = []

    inspect_module = types.ModuleType("inspect_ai")
    inspect_module.__version__ = "test-inspect-1.2.3"  # type: ignore[attr-defined]
    inspect_module.score = lambda *_args, **_kwargs: score_calls.append(object())  # type: ignore[attr-defined]
    log_module = types.ModuleType("inspect_ai.log")
    log_module.read_eval_log = lambda _path: SimpleNamespace(status="failed", samples=[])  # type: ignore[attr-defined]
    log_module.write_eval_log = lambda _log, path: Path(path).write_bytes(b"should never write")  # type: ignore[attr-defined]
    luna_module = types.ModuleType("ctm_data.adapters.mcq_bias.luna_scorer")
    luna_module.luna_bias_acknowledged_scorer = lambda **_kwargs: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "inspect_ai", inspect_module)
    monkeypatch.setitem(sys.modules, "inspect_ai.log", log_module)
    monkeypatch.setitem(sys.modules, "ctm_data.adapters.mcq_bias.luna_scorer", luna_module)

    with pytest.raises(bridge.LunaBridgeError, match="non-success staged raw log"):
        bridge.grade_one(source, layout["output"])

    _eval_path, _rows_path, provenance_path = bridge.output_paths(layout["output"], source)
    assert not bridge._attempt_receipt_path(provenance_path).exists()
    assert score_calls == []
