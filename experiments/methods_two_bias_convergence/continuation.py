"""Fail-closed Slurm continuation for the approved BCT/OPCT execution repair."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from . import plan
from .train import atomic_json, checkpoint_identity, load_resume


def job_state(job_id: str) -> str:
    if not re.fullmatch(r"[0-9]+", job_id):
        raise ValueError("expected a numeric Slurm job ID")
    result = subprocess.check_output(
        ["sacct", "-X", "-j", job_id, "--noheader", "--parsable2", "--format=JobIDRaw,State"],
        text=True,
    )
    states = [line.split("|")[1].split()[0] for line in result.splitlines()
              if line.split("|")[0] == job_id]
    if len(states) != 1:
        raise ValueError(f"cannot verify predecessor {job_id}: {result!r}")
    return states[0]


def require_progress(*, scheduler_state: str, starting_step: int, state: dict) -> None:
    if scheduler_state not in {"COMPLETED", "TIMEOUT"}:
        raise ValueError(f"predecessor {scheduler_state}: not an authorized automatic retry")
    if state["decision"] == "continue" and state["step"] <= starting_step:
        raise ValueError("predecessor made no sealed optimizer progress; stop instead of looping")


def import_parent(parent_run: Path, run_dir: Path, *, method: str, parent_job: str) -> None:
    """Copy one verified optimizer checkpoint; never modify the old trajectory."""
    if job_state(parent_job) not in {"COMPLETED", "TIMEOUT"}:
        raise ValueError("parent is not a completed/timed-out training job")
    if (run_dir / "state.json").exists():
        raise ValueError("refusing to re-import over an existing recovery trajectory")
    queued = subprocess.check_output(["squeue", "--noheader", "--user", os.environ["USER"],
                                      "--format=%i|%j"], text=True)
    for line in queued.splitlines():
        job_id, name = line.split("|", 1)
        if not name.endswith("-" + method) or job_id == parent_job:
            continue
        detail = subprocess.check_output(["scontrol", "show", "job", job_id], text=True)
        workdirs = [Path(field.split("=", 1)[1]) for field in detail.split() if field.startswith("WorkDir=")]
        if any(path.resolve() == parent_run.parent.parent.resolve() for path in workdirs):
            raise ValueError(f"old trajectory still has active/queued job {job_id}; refuse a duplicate fork")
    with (parent_run / ".training.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state, _ = load_resume(parent_run, plan.PARENT_PLAN_SHA, method)
        if state["step"] == 0:
            raise ValueError("parent has no sealed checkpoint")
        receipt = json.loads((parent_run / "state.json").read_text())
        relative = Path(receipt["checkpoint"])
        if relative != Path("checkpoints") / f"step-{state['step']:06d}":
            raise ValueError("unexpected parent checkpoint path")
        destination = run_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(parent_run / relative, destination)
        if checkpoint_identity(destination) != receipt["checkpoint_files"]:
            raise ValueError("copied checkpoint differs from the sealed parent")
        if method == "bct":
            shutil.copytree(parent_run / "base-targets", run_dir / "base-targets")
        plan.immutable_json(run_dir / "receipts" / f"step-{state['step']:06d}.json", receipt)
        plan.immutable_json(run_dir / "parent-import.json", {
            "parent_run": str(parent_run), "parent_job": parent_job,
            "parent_state_sha256": plan.sha256(parent_run / "state.json"),
            "checkpoint": receipt, "optimizer_preserved": True,
        })
        atomic_json(run_dir / "state.json", receipt)


def prepare(run_dir: Path, *, method: str, plan_path: Path, current_job: str) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parent_job = os.environ.get("CTM_METHODS_RECOVERY_PARENT_JOB")
        if parent_job:
            import_parent(Path(os.environ["CTM_METHODS_RECOVERY_PARENT_RUN"]), run_dir,
                          method=method, parent_job=parent_job)
        state, _ = load_resume(run_dir, plan.sha256(plan_path), method, parent_plan_hash=plan.PARENT_PLAN_SHA)
        predecessor = os.environ.get("CTM_METHODS_PREDECESSOR_JOB")
        if predecessor:
            previous = json.loads((run_dir / "jobs" / f"{predecessor}.json").read_text())
            require_progress(scheduler_state=job_state(predecessor), starting_step=previous["starting_step"], state=state)
        elif not parent_job:
            raise ValueError("recovery must have an explicit initial parent or predecessor")
        plan.immutable_json(run_dir / "jobs" / f"{current_job}.json", {
            "job_id": current_job, "method": method, "starting_step": state["step"],
            "plan_sha256": plan.sha256(plan_path), "predecessor": predecessor or parent_job,
        })
        return state


def queue_successor(run_dir: Path, *, method: str, current_job: str, script: Path) -> str | None:
    state = json.loads((run_dir / "state.json").read_text())["convergence"]
    if state["decision"] != "continue":
        return None
    receipt_path = run_dir / "jobs" / f"{current_job}-successor.json"
    if receipt_path.exists():
        return json.loads(receipt_path.read_text())["job_id"]
    environment = dict(os.environ)
    environment.pop("CTM_METHODS_RECOVERY_PARENT_JOB", None)
    environment.pop("CTM_METHODS_RECOVERY_PARENT_RUN", None)
    environment["CTM_METHODS_PREDECESSOR_JOB"] = current_job
    result = subprocess.check_output(
        ["sbatch", "--parsable", f"--dependency=afterany:{current_job}",
         f"--job-name=ctm-recovery-v5-{method}", str(script)], text=True, env=environment,
    ).strip()
    job_id = result.split(";")[0]
    if not re.fullmatch(r"[0-9]+", job_id):
        raise ValueError(f"uncertain successor submission: {result!r}; reconcile before retrying")
    plan.immutable_json(receipt_path, {"job_id": job_id, "predecessor": current_job})
    return job_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "queue"))
    args = parser.parse_args()
    method = os.environ["CTM_METHOD"]
    if method not in {"bct", "opct"}:
        raise ValueError("execution recovery is only approved for BCT/OPCT")
    run_dir = Path(os.environ["CTM_METHODS_RUN_ROOT"]) / method
    current_job = os.environ["SLURM_JOB_ID"]
    if args.action == "prepare":
        result = prepare(run_dir, method=method, plan_path=Path(os.environ["CTM_METHODS_PLAN"]), current_job=current_job)
    else:
        result = queue_successor(run_dir, method=method, current_job=current_job,
                                 script=Path(os.environ["REPO_DIR"]) / "infra/isambard/run_methods_two_bias_convergence.sbatch")
    print(json.dumps({"action": args.action, "result": result}), flush=True)


if __name__ == "__main__":
    main()
