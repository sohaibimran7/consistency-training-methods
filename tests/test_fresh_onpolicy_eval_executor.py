"""Offline contracts for the fresh RMCT four-GPU evaluation executor."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import execute_fresh_onpolicy_evals as executor
from scripts import launch_fresh_onpolicy_evals as launcher


def _context(tmp_path: Path) -> executor.ExecutionContext:
    root = tmp_path / "evidence"
    contract = root / "launch.json"
    commands = {
        name: {"name": name, "argv": ["fake-python", name], "cwd": str(Path(__file__).parents[1])}
        for name in (
            "revalidate_fresh_contract",
            "stage1_raw_generation",
            "stage1_raw_preflight",
            "stage1_stage_hash_bound_raw_logs",
            "stage1_luna_verbalisation",
            "stage1_tbsr_and_luna_analysis",
            "stage2_raw_generation",
            "stage2_raw_preflight",
            "stage2_stage_hash_bound_raw_logs",
            "stage2_luna_verbalisation",
            "stage2_tbsr_and_luna_analysis",
        )
    }
    return executor.ExecutionContext(
        contract_path=contract,
        contract_sha256="a" * 64,
        document={"target": "rmct-main"},
        commands=commands,
        output_root=root,
        condition="rmct-main",
        checkpoint=tmp_path / "checkpoint",
        stage1_raw=root / "stage1" / "raw",
        stage2_raw=root / "stage2" / "raw",
    )


def _cell(index: int, *, kind: str | None = None) -> executor.ValidatedCell:
    return executor.ValidatedCell(
        task_index=index,
        kind=kind or ("clean" if index <= 4 else "biased"),
        identity={"index": index},
        path=Path(f"/tmp/task-{index}.eval"),
        sha256=f"{index:064x}",
        created="2026-08-03T00:00:00Z",
        runtime={"profile": "hf-peft"},
    )


def test_gpu_parser_requires_exactly_four_distinct_physical_devices():
    assert executor.parse_physical_gpus("4,5,6,7") == (4, 5, 6, 7)
    with pytest.raises(ValueError, match="exactly four"):
        executor.parse_physical_gpus("4,5,6")
    with pytest.raises(ValueError, match="duplicate"):
        executor.parse_physical_gpus("4,4,6,7")
    with pytest.raises(ValueError, match="non-negative"):
        executor.parse_physical_gpus("4,-1,6,7")


def test_stage1_resumes_valid_cells_and_never_starts_bias_before_clean_barrier(tmp_path, monkeypatch):
    context = _context(tmp_path)
    present = {1, 3, 5}
    launches: list[list[int]] = []

    def audit(_context, *, validate_biased: bool):
        return executor.Audit(
            selected={index: _cell(index) for index in present},
            archived=(),
        )

    def launch(_command, *, task_indices, gpus):
        launches.append(list(task_indices))
        # The biased group must observe all clean cells already selected.
        if any(index >= 5 for index in task_indices):
            assert {1, 2, 3, 4} <= present
        present.update(task_indices)
        assert list(gpus) == [4, 5, 6, 7]
        return []

    monkeypatch.setattr(executor, "_audit_stage1", audit)
    monkeypatch.setattr(executor, "_run_missing_tasks", launch)

    result = executor._run_raw_stage(context, stage="stage1", gpus=(4, 5, 6, 7))

    assert launches == [[2, 4], [6, 7, 8]]
    assert result["clean_reused"] == 2
    assert result["clean_generated"] == 2
    assert result["biased_reused"] == 1
    assert result["biased_generated"] == 3


def test_worker_groups_expose_one_explicit_gpu_and_round_robin_missing_cells(monkeypatch):
    observed: dict[str, list[int]] = {}

    def fake_run(argv, *, cwd, env, check):
        assert check is False
        assert env["CUDA_VISIBLE_DEVICES"] in {"4", "5", "6", "7"}
        assert "VLLM_BASE_URL" not in env
        indices = [int(argv[index + 1]) for index, token in enumerate(argv) if token == "--task-index"]
        observed[env["CUDA_VISIBLE_DEVICES"]] = indices
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("VLLM_BASE_URL", "http://stale-server")
    monkeypatch.setattr(executor.subprocess, "run", fake_run)

    failures = executor._run_missing_tasks(
        {"argv": ["fake-python", "scripts/run_evals.py"]},
        task_indices=[1, 2, 3, 4, 5, 6],
        gpus=(4, 5, 6, 7),
    )

    assert failures == []
    assert observed == {"4": [1, 5], "5": [2, 6], "6": [3], "7": [4]}


def test_retryable_raw_logs_move_only_to_recoverable_condition_archive(tmp_path):
    context = _context(tmp_path)
    raw = context.stage2_raw
    raw.mkdir(parents=True)
    partial = raw / "broken.eval"
    partial.write_bytes(b"partial inspect payload")

    records = executor._archive_candidates(
        [executor.ArchiveCandidate(partial, "partial_payload:ValueError", executor._sha256_file(partial))],
        context=context,
        stage="stage2",
        raw_root=raw,
    )

    assert not partial.exists()
    assert len(records) == 1
    destination = Path(records[0]["destination"])
    assert destination.is_file()
    assert destination.read_bytes() == b"partial inspect payload"
    assert destination.is_relative_to(context.output_root / "_archive" / "rmct-main" / "stage2")
    assert destination.parent.parent.joinpath("archive-receipt.json").is_file()
    assert list((context.output_root / "_archive").rglob("archive-receipt.json"))


def test_executor_runs_cpu_steps_in_the_written_contract_order(tmp_path, monkeypatch):
    context = _context(tmp_path)
    observed: list[str] = []

    monkeypatch.setattr(executor, "load_execution_context", lambda _path: context)
    monkeypatch.setattr(
        executor,
        "_run_raw_stage",
        lambda _context, *, stage, gpus: observed.append(f"{stage}-raw") or {"stage": stage},
    )
    monkeypatch.setattr(executor, "_run_contract_command", lambda command: observed.append(command["name"]))

    result = executor.execute_contract(context.contract_path, gpus=(0, 1, 2, 3))

    assert observed == [
        "revalidate_fresh_contract",
        "stage1-raw",
        "stage1_raw_preflight",
        "stage1_stage_hash_bound_raw_logs",
        "stage1_luna_verbalisation",
        "stage1_tbsr_and_luna_analysis",
        "stage2-raw",
        "stage2_raw_preflight",
        "stage2_stage_hash_bound_raw_logs",
        "stage2_luna_verbalisation",
        "stage2_tbsr_and_luna_analysis",
    ]
    assert [step["name"] for step in result["steps"]] == [
        "revalidate_fresh_contract",
        "stage1_raw_generation",
        "stage1_raw_preflight",
        "stage1_stage_hash_bound_raw_logs",
        "stage1_luna_verbalisation",
        "stage1_tbsr_and_luna_analysis",
        "stage2_raw_generation",
        "stage2_raw_preflight",
        "stage2_stage_hash_bound_raw_logs",
        "stage2_luna_verbalisation",
        "stage2_tbsr_and_luna_analysis",
    ]


def test_raw_only_runs_both_raw_matrices_and_staging_but_defers_paid_luna(tmp_path, monkeypatch):
    context = _context(tmp_path)
    observed: list[str] = []

    monkeypatch.setattr(executor, "load_execution_context", lambda _path: context)
    monkeypatch.setattr(
        executor,
        "_run_raw_stage",
        lambda _context, *, stage, gpus: observed.append(f"{stage}-raw") or {"stage": stage},
    )
    monkeypatch.setattr(executor, "_run_contract_command", lambda command: observed.append(command["name"]))
    monkeypatch.setattr(executor, "_raw_only_receipt", lambda _context, *, gpus: "written")

    result = executor.execute_contract(context.contract_path, gpus=(4, 5, 6, 7), raw_only=True)

    assert observed == [
        "revalidate_fresh_contract",
        "stage1-raw",
        "stage1_raw_preflight",
        "stage1_stage_hash_bound_raw_logs",
        "stage2-raw",
        "stage2_raw_preflight",
        "stage2_stage_hash_bound_raw_logs",
    ]
    assert "stage1_luna_verbalisation" not in observed
    assert "stage2_luna_verbalisation" not in observed
    assert result["raw_only_receipt"]["status"] == "written"
    assert result["raw_only_receipt"]["deferred_commands"] == [
        "stage1_luna_verbalisation",
        "stage1_tbsr_and_luna_analysis",
        "stage2_luna_verbalisation",
        "stage2_tbsr_and_luna_analysis",
    ]


def test_raw_only_receipt_is_write_once_and_binds_every_validated_raw_cell(tmp_path, monkeypatch):
    context = _context(tmp_path)
    stage1 = executor.Audit(selected={index: _cell(index) for index in range(1, 9)}, archived=())
    stage2 = executor.Audit(
        selected={
            index: _cell(index, kind="clean" if index <= 3 else "biased")
            for index in range(1, 22)
        },
        archived=(),
    )
    monkeypatch.setattr(executor, "_audit_stage1", lambda _context, *, validate_biased: stage1)
    monkeypatch.setattr(executor, "_audit_stage2", lambda _context, *, validate_biased: stage2)

    assert executor._raw_only_receipt(context, gpus=(4, 5, 6, 7)) == "written"
    assert executor._raw_only_receipt(context, gpus=(4, 5, 6, 7)) == "resumed"
    receipt = context.output_root / "_executor-receipts" / "rmct-main" / "raw-only-complete.json"
    document = json.loads(receipt.read_text(encoding="utf-8"))
    assert len(document["stages"]["stage1"]) == 8
    assert len(document["stages"]["stage2"]) == 21
    with pytest.raises(FileExistsError, match="differing"):
        executor._raw_only_receipt(context, gpus=(0, 1, 2, 3))


def test_current_rmct_targets_follow_the_approved_rng_repair_four_gpu_plan():
    expected_path = (
        "experiments/rmct_paper_vast_dense_models/stage1/"
        "qwen3_5_9b_rmct_paper_fidelity_isambard_phase2_4gpu_20260803.yaml"
    )
    expected_experiment = "rmct_paper_isambard_phase2_qwen3_5_9b_rng_repair_4gpu_20260803"
    for target in ("rmct-main", "rmct-control"):
        current = launcher.CURRENT_TARGETS[target]
        assert current.plan_relative_path == expected_path
        assert current.experiment == expected_experiment
        assert current.runtime_profile == "hf-peft"


def test_executor_source_is_raw_resumable_and_never_recursively_deletes_outputs():
    source = (Path(__file__).parents[1] / "scripts" / "execute_fresh_onpolicy_evals.py").read_text(encoding="utf-8")
    assert "shutil.move" in source
    assert "partial-or-non-success" in source
    assert "rm -rf" not in source
