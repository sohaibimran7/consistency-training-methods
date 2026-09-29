from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from experiments.methods_two_bias_convergence import continuation, plan, train


def sealed_run(root, *, method="bct", step=16, plan_hash=plan.PARENT_PLAN_SHA):
    root.mkdir(parents=True)
    (root / ".training.lock").touch()
    checkpoint = root / "checkpoints" / f"step-{step:06d}"
    checkpoint.mkdir(parents=True)
    state = {"step": step, "pending": [], "decision": "continue"}
    loop = {"step": step, "method": method, "plan_sha256": plan_hash, "convergence": state}
    for name in ("adapter_model.safetensors", "optimizer.pt"):
        (checkpoint / name).write_bytes(b"exact sealed payload")
    plan.immutable_json(checkpoint / "adapter_config.json", {})
    plan.immutable_json(checkpoint / "manifest.json", {"loop_state": loop})
    plan.immutable_json(checkpoint / "window-metrics.json", [])
    receipt = {"method": method, "plan_sha256": plan_hash, "convergence": state,
               "checkpoint": str(checkpoint.relative_to(root)),
               "checkpoint_files": train.checkpoint_identity(checkpoint)}
    plan.immutable_json(root / "receipts" / f"step-{step:06d}.json", receipt)
    plan.immutable_json(root / "state.json", receipt)
    if method == "bct":
        plan.immutable_json(root / "base-targets/target.json", {"tokens": [1, 2, 3]})
    return receipt


@pytest.mark.parametrize("method", ["bct", "opct"])
def test_import_preserves_parent_optimizer_and_cache_bytes(tmp_path, monkeypatch, method):
    parent = tmp_path / "parent/runs" / method
    receipt = sealed_run(parent, method=method)
    before = plan.sha256(parent / "state.json")
    target = tmp_path / "recovery/runs" / method
    monkeypatch.setattr(continuation, "job_state", lambda job: "TIMEOUT")
    monkeypatch.setattr(continuation.subprocess, "check_output", lambda *a, **k: "")
    continuation.import_parent(parent, target, method=method, parent_job="6451365")
    assert json.loads((target / "state.json").read_text()) == receipt
    assert plan.sha256(parent / "state.json") == before
    state, path = train.load_resume(target, "new-plan", method, parent_plan_hash=plan.PARENT_PLAN_SHA)
    assert state["step"] == 16 and path.endswith("step-000016")
    if method == "bct":
        assert (target / "base-targets/target.json").read_bytes() == (parent / "base-targets/target.json").read_bytes()
    with pytest.raises(ValueError, match="re-import"):
        continuation.import_parent(parent, target, method=method, parent_job="6451365")


def test_import_refuses_an_old_queued_successor(tmp_path, monkeypatch):
    parent = tmp_path / "parent/runs/bct"
    sealed_run(parent)
    monkeypatch.setattr(continuation, "job_state", lambda job: "COMPLETED")
    monkeypatch.setattr(continuation.subprocess, "check_output", lambda args, **kwargs:
                        "9001|ctm-two-bias-bct\n" if args[0] == "squeue" else f"JobId=9001 WorkDir={tmp_path / 'parent'}")
    with pytest.raises(ValueError, match="duplicate fork"):
        continuation.import_parent(parent, tmp_path / "recovery", method="bct", parent_job="6451365")


def test_successor_prequeues_once_and_removes_initial_import_environment(tmp_path, monkeypatch):
    run_dir = tmp_path / "bct"
    sealed_run(run_dir)
    monkeypatch.setenv("CTM_METHODS_RECOVERY_PARENT_JOB", "6000")
    monkeypatch.setenv("CTM_METHODS_RECOVERY_PARENT_RUN", "/old/run")
    calls = []

    def submit(args, *, text, env):
        calls.append(args)
        assert "--dependency=afterany:7000" in args
        assert env["CTM_METHODS_PREDECESSOR_JOB"] == "7000"
        assert "CTM_METHODS_RECOVERY_PARENT_JOB" not in env
        assert "CTM_METHODS_RECOVERY_PARENT_RUN" not in env
        return "8000;isambard\n"

    monkeypatch.setattr(continuation.subprocess, "check_output", submit)
    for _ in range(2):
        assert continuation.queue_successor(run_dir, method="bct", current_job="7000", script=Path("/runner.sbatch")) == "8000"
    assert len(calls) == 1
    state = json.loads((run_dir / "state.json").read_text())
    state["convergence"]["decision"] = "plateau"
    train.atomic_json(run_dir / "state.json", state)
    assert continuation.queue_successor(run_dir, method="bct", current_job="8000", script=Path("/runner.sbatch")) is None
    assert len(calls) == 1


def test_interrupted_checkpoint_write_cannot_replace_sealed_state(tmp_path):
    run_dir = tmp_path / "bct"
    sealed_run(run_dir)
    before = (run_dir / "state.json").read_bytes()

    class FailingBackend:
        async def save_checkpoint(self, *, name, log_dir, **kwargs):
            path = log_dir / "checkpoints" / name
            path.mkdir(parents=True)
            (path / "optimizer.pt").write_bytes(b"incomplete")
            raise RuntimeError("simulated interruption during checkpoint write")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        asyncio.run(train.seal_checkpoint(FailingBackend(), run_dir=run_dir, method="bct",
                                         state={"step": 17, "pending": [.3], "decision": "continue"},
                                         plan_hash="new-plan", window_metrics=[{"step": 17, "loss": .3}]))
    assert (run_dir / "state.json").read_bytes() == before
    assert not (run_dir / "checkpoints/step-000017").exists()
    assert train.load_resume(run_dir, plan.PARENT_PLAN_SHA, "bct")[0]["step"] == 16
