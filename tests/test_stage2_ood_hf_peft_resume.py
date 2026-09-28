from __future__ import annotations

import hashlib
import json
import os
import subprocess
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.stage2_ood_hle import hf_peft_resume as resume


def _identity(index: int) -> tuple[str, str, str, str, str | None]:
    if index <= 3:
        population, dataset = (("in_domain", "logiqa"), ("in_domain", "hellaswag"), ("hle", "hle"))[index - 1]
        return "unbiased", "iid", population, dataset, None
    # Every synthetic biased task shares the first synthetic clean population
    # while retaining a unique bias identity, so the incremental handoff can
    # exercise its clean-pair binding without a real frozen matrix.
    return "biased", "iid", "in_domain", "logiqa", f"bias-{index}"


def _install_fake_matrix(tmp_path: Path, monkeypatch, *, successful: set[int], failed: set[int] = set()):
    manifest = tmp_path / "frozen-manifest.json"
    manifest.write_bytes(b"frozen")
    checkpoint = tmp_path / "bct"
    checkpoint.mkdir()
    raw_root = tmp_path / "raw-no-luna" / "bct-hf-peft"
    raw_root.mkdir(parents=True)
    specs = []
    expected = {}
    identity_by_path: dict[Path, tuple[str, str, str, str, str | None]] = {}
    logs = {}
    for index in range(1, 22):
        identity = _identity(index)
        spec = SimpleNamespace(
            kind=identity[0],
            regime=identity[1],
            population=identity[2],
            dataset=identity[3],
            bias_type=identity[4],
        )
        specs.append(spec)
        expected[identity] = spec
        if index not in successful and index not in failed:
            continue
        path = raw_root / f"task-{index}.eval"
        path.write_bytes(f"task={index}".encode())
        identity_by_path[path.resolve()] = identity
        task = "stage2_ood_unbiased" if identity[0] == "unbiased" else "stage2_ood_biased"
        logs[path.resolve()] = SimpleNamespace(
            status="success" if index in successful else "error",
            eval=SimpleNamespace(task=task, created=f"2026-08-03T00:{index:02d}:00Z"),
        )

    monkeypatch.setattr(resume, "ood_task_specs", lambda _manifest: specs)
    monkeypatch.setattr(resume.raw_preflight, "validate_manifest", lambda _manifest: {})
    monkeypatch.setattr(resume.raw_preflight, "_expected_cell_specs", lambda _specs: expected)
    monkeypatch.setattr(resume.raw_preflight, "_validate_runtime_contract", lambda **_kwargs: {"profile": "hf-peft"})
    monkeypatch.setattr(
        resume.raw_preflight,
        "_read_eval_log",
        lambda path, *, header_only: logs[Path(path).resolve()],
    )
    monkeypatch.setattr(
        resume.raw_preflight,
        "_parse_candidate_identity",
        lambda _evaluation, *, task_name, path: identity_by_path[Path(path).resolve()],
    )
    monkeypatch.setattr(
        resume.raw_preflight, "_validate_header", lambda _log, *, path, spec, raw_root: logs[path].eval.created
    )
    monkeypatch.setattr(
        resume.raw_preflight,
        "_assert_runtime",
        lambda _path, *, runtime: ("Qwen/Qwen3.5-9B", {"profile": runtime["profile"], "max_connections": 8}),
    )
    monkeypatch.setattr(
        resume,
        "build_launch_contract",
        lambda **kwargs: {
            "condition": kwargs["condition"],
            "checkpoint": {"path": str(Path(kwargs["checkpoint"]).resolve()), "adapter_model_sha256": "a" * 64},
            "manifest": str(Path(kwargs["manifest"]).resolve()),
            "raw_log_dir": str(Path(kwargs["raw_log_dir"]).resolve()),
            "max_connections": kwargs["max_connections"],
        },
    )
    return manifest, checkpoint, raw_root, logs


def test_prepare_archives_only_failed_eval_logs_and_binds_existing_success_hashes(tmp_path, monkeypatch):
    manifest, checkpoint, raw_root, _logs = _install_fake_matrix(tmp_path, monkeypatch, successful={1, 2}, failed={4})
    contract = raw_root / "resume-contract.json"
    archive = tmp_path / "_archive" / "bct-hf-peft" / "resume-test"

    state = resume.prepare_resume(
        condition="bct-hf-peft",
        checkpoint=checkpoint,
        manifest=manifest,
        raw_log_dir=raw_root,
        contract_path=contract,
        archive_dir=archive,
    )

    assert state["contract_status"] == "written"
    assert state["selected_task_indices"] == [1, 2]
    assert state["missing_clean_task_indices"] == [3]
    assert state["missing_biased_task_indices"] == list(range(4, 22))
    assert not (raw_root / "task-4.eval").exists()
    assert (archive / "partial-or-non-success" / "task-4.eval").is_file()
    document = json.loads(contract.read_text())
    assert document["schema"] == resume.RESUME_SCHEMA
    assert [entry["task_index"] for entry in document["initial_successes"]] == [1, 2]
    assert (
        document["initial_successes"][0]["raw_log_sha256"]
        == hashlib.sha256((raw_root / "task-1.eval").read_bytes()).hexdigest()
    )


def test_contract_refuses_changed_inherited_success_before_new_generation(tmp_path, monkeypatch):
    manifest, checkpoint, raw_root, _logs = _install_fake_matrix(tmp_path, monkeypatch, successful={1, 2, 3})
    contract = raw_root / "resume-contract.json"
    resume.prepare_resume(
        condition="bct-hf-peft",
        checkpoint=checkpoint,
        manifest=manifest,
        raw_log_dir=raw_root,
        contract_path=contract,
        archive_dir=tmp_path / "_archive" / "bct-hf-peft" / "resume-test",
    )
    (raw_root / "task-1.eval").write_bytes(b"changed")

    with pytest.raises(ValueError, match="changed or disappeared"):
        resume.resume_status(
            condition="bct-hf-peft",
            checkpoint=checkpoint,
            manifest=manifest,
            raw_log_dir=raw_root,
            contract_path=contract,
        )


def test_incremental_handoff_requires_clean_barrier_and_stages_one_verified_biased_cell(tmp_path, monkeypatch):
    manifest, checkpoint, raw_root, _logs = _install_fake_matrix(tmp_path, monkeypatch, successful={1, 2, 3, 4})
    contract = raw_root / "resume-contract.json"
    resume.prepare_resume(
        condition="bct-hf-peft",
        checkpoint=checkpoint,
        manifest=manifest,
        raw_log_dir=raw_root,
        contract_path=contract,
        archive_dir=tmp_path / "_archive" / "bct-hf-peft" / "resume-test",
    )
    monkeypatch.setattr(resume, "_validated_selected_full", lambda selected, *, requested: {})
    staged_root = tmp_path / "incremental-staged"

    records = resume.stage_incremental_handoff(
        condition="bct-hf-peft",
        checkpoint=checkpoint,
        manifest=manifest,
        raw_log_dir=raw_root,
        contract_path=contract,
        output_root=staged_root,
        task_indices=[4],
        all_biased=False,
        allow_inflight_partials=False,
    )

    assert len(records) == 1
    staged = Path(records[0]["staged_log"])
    assert staged.is_file()
    assert staged.read_bytes() == (raw_root / "task-4.eval").read_bytes()
    receipt = json.loads(staged.with_suffix(".resume-handoff.json").read_text())
    assert receipt["schema"] == resume.HANDOFF_SCHEMA
    assert receipt["task_index"] == 4
    assert receipt["protocol"]["validated_with_full_paired_switch_scores"] is True


def test_incremental_handoff_rejects_biased_work_until_all_clean_cells_exist(tmp_path, monkeypatch):
    manifest, checkpoint, raw_root, _logs = _install_fake_matrix(tmp_path, monkeypatch, successful={1, 2, 4})
    contract = raw_root / "resume-contract.json"
    resume.prepare_resume(
        condition="bct-hf-peft",
        checkpoint=checkpoint,
        manifest=manifest,
        raw_log_dir=raw_root,
        contract_path=contract,
        archive_dir=tmp_path / "_archive" / "bct-hf-peft" / "resume-test",
    )

    with pytest.raises(ValueError, match="requires all exact clean cells"):
        resume.stage_incremental_handoff(
            condition="bct-hf-peft",
            checkpoint=checkpoint,
            manifest=manifest,
            raw_log_dir=raw_root,
            contract_path=contract,
            output_root=tmp_path / "incremental-staged",
            task_indices=[4],
            all_biased=False,
            allow_inflight_partials=False,
        )


def test_active_handoff_contract_stages_all_completed_biases_without_touching_inflight_raw_logs(tmp_path, monkeypatch):
    """The live RMCT path must not archive another GPU's partial EvalLog."""

    manifest, checkpoint, raw_root, _logs = _install_fake_matrix(
        tmp_path,
        monkeypatch,
        successful={1, 2, 3, 4, 5, 7},
        failed={6, 8},
    )
    contract = tmp_path / "contracts" / "bct-hf-peft.active-handoff.json"
    raw_before = {path.relative_to(raw_root): path.read_bytes() for path in raw_root.rglob("*.eval")}

    state = resume.establish_active_handoff_contract(
        condition="bct-hf-peft",
        checkpoint=checkpoint,
        manifest=manifest,
        raw_log_dir=raw_root,
        contract_path=contract,
    )

    assert state["schema"] == resume.ACTIVE_HANDOFF_SCHEMA
    assert state["contract_status"] == "written"
    assert state["raw_eval_logs_mutated"] is False
    assert state["selected_task_indices"] == [1, 2, 3, 4, 5, 7]
    assert {Path(item["path"]).name for item in state["unmodified_inflight_candidates"]} == {"task-6.eval", "task-8.eval"}
    contract_document = json.loads(contract.read_text(encoding="utf-8"))
    assert contract_document["active_handoff_policy"] == resume.ACTIVE_HANDOFF_POLICY
    assert [entry["task_index"] for entry in contract_document["initial_successes"]] == [1, 2, 3, 4, 5, 7]
    assert {path.relative_to(raw_root): path.read_bytes() for path in raw_root.rglob("*.eval")} == raw_before

    # The synthetic logs have only headers, so keep this focused on the active
    # raw-preservation and selection contract rather than Inspect score parsing.
    monkeypatch.setattr(resume, "_validated_selected_full", lambda selected, *, requested: {})
    records = resume.stage_incremental_handoff(
        condition="bct-hf-peft",
        checkpoint=checkpoint,
        manifest=manifest,
        raw_log_dir=raw_root,
        contract_path=contract,
        output_root=tmp_path / "luna-incremental-staged-v1",
        task_indices=None,
        all_biased=True,
        allow_inflight_partials=True,
    )

    assert [record["task_index"] for record in records] == [4, 5, 7]
    assert {path.relative_to(raw_root): path.read_bytes() for path in raw_root.rglob("*.eval")} == raw_before
    assert (raw_root / "task-6.eval").is_file()
    assert (raw_root / "task-8.eval").is_file()


def test_active_handoff_contract_must_be_external_to_the_live_raw_directory(tmp_path, monkeypatch):
    manifest, checkpoint, raw_root, _logs = _install_fake_matrix(tmp_path, monkeypatch, successful={1, 2, 3, 4})
    raw_before = {path.relative_to(raw_root): path.read_bytes() for path in raw_root.rglob("*.eval")}

    with pytest.raises(ValueError, match="outside the raw EvalLog directory"):
        resume.establish_active_handoff_contract(
            condition="bct-hf-peft",
            checkpoint=checkpoint,
            manifest=manifest,
            raw_log_dir=raw_root,
            contract_path=raw_root / "active-handoff-contract.json",
        )

    assert {path.relative_to(raw_root): path.read_bytes() for path in raw_root.rglob("*.eval")} == raw_before


def _write_fake_resume_python(path: Path) -> None:
    """Emulate launcher subprocess boundaries without Inspect, a model, or a network."""

    path.write_text(
        textwrap.dedent("""\
            #!/usr/bin/env bash
            set -euo pipefail
            capture=${CTM_FAKE_CAPTURE:?}
            first=${1:-}
            if [ "$first" = "-c" ]; then
              if [[ ${2:-} == *"document = json.loads"* ]]; then
                field=${3:-}
                cat >/dev/null
                case "$field" in
                  missing_clean_task_indices) printf '1 2 3\\n' ;;
                  missing_biased_task_indices) printf '4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21\\n' ;;
                  *) exit 91 ;;
                esac
              else
                printf '%s\\n' '{"include_bias_acknowledged":false,"prompt_style":"none"}'
              fi
              exit 0
            fi
            if [ "$first" = "-m" ]; then
              module=${2:-}
              command=${3:-}
              case "$module:$command" in
                experiments.stage2_ood_hle.hf_peft_resume:prepare)
                  printf '%s\\n' '{"missing_clean_task_indices":[1,2,3],"missing_biased_task_indices":[4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]}'
                  ;;
                experiments.stage2_ood_hle.hf_peft_resume:verify-clean|experiments.stage2_ood_hle.hf_peft_resume:status)
                  printf '%s\\n' '{"missing_clean_task_indices":[],"missing_biased_task_indices":[4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]}'
                  ;;
                experiments.stage2_ood_hle.hf_peft_resume:handoff)
                  printf '%s\\n' '[]'
                  ;;
                experiments.stage2_ood_hle.raw_preflight:*)
                  printf '%s\\n' 'written: fake-preflight.json'
                  ;;
                *) exit 92 ;;
              esac
              exit 0
            fi
            [ "$first" = "scripts/run_evals.py" ] || exit 93
            task=""
            isolate=0
            max_connections=0
            while [ "$#" -gt 0 ]; do
              case "$1" in
                --task-index) task=$2; shift 2 ;;
                --isolate-tasks) isolate=1; shift ;;
                --generation-config)
                  [[ $2 == *'"max_connections":8'* ]] || exit 94
                  [[ $2 == *'"max_tokens":20480'* ]] || exit 95
                  max_connections=8
                  shift 2
                  ;;
                *) shift ;;
              esac
            done
            [ -n "$task" ] && [ "$isolate" = 1 ] && [ "$max_connections" = 8 ] || exit 96
            printf 'gpu=%s task=%s\\n' "${CUDA_VISIBLE_DEVICES:-}" "$task" > "$capture/task-$task"
            """),
        encoding="utf-8",
    )
    path.chmod(0o755)


@pytest.mark.parametrize("condition", ["bct-hf-peft", "opct-phase2-hf-peft"])
def test_resume_launcher_distributes_only_missing_tasks_and_keeps_final_gate_cpu_only(tmp_path, condition):
    root = Path(__file__).resolve().parents[1]
    launcher = root / "scripts" / "ctm_stage2_ood_hf_peft_resume_condition_20260803.sh"
    fake_python = tmp_path / "fake-python"
    _write_fake_resume_python(fake_python)
    repo = tmp_path / "repo"
    frozen = tmp_path / "frozen"
    checkpoint = tmp_path / "bct"
    run_root = tmp_path / "run"
    capture = tmp_path / "capture"
    repo.mkdir()
    frozen.mkdir()
    checkpoint.mkdir()
    capture.mkdir()
    (frozen / "manifest.json").write_text("{}", encoding="utf-8")

    environment = {
        **os.environ,
        "CTM_OOD_REPO": str(repo),
        "CTM_OOD_PY": str(fake_python),
        "CTM_OOD_FROZEN": str(frozen),
        "CTM_OOD_RUN_ROOT": str(run_root),
        "CTM_OOD_CONDITION": condition,
        "CTM_OOD_CHECKPOINT": str(checkpoint),
        "CTM_OOD_EXPECTED_CHECKPOINT": str(checkpoint),
        "CTM_OOD_TEST_MODE": "1",
        "CTM_OOD_GPUS": "0,4,5,6",
        "CTM_FAKE_CAPTURE": str(capture),
    }
    subprocess.run(["bash", str(launcher)], cwd=tmp_path, env=environment, check=True, text=True)

    assignments = {
        int(path.name.removeprefix("task-")): path.read_text(encoding="utf-8").strip()
        for path in capture.glob("task-*")
    }
    assert set(assignments) == set(range(1, 22))
    assert {index for index, record in assignments.items() if "gpu=0" in record} == {1, 4, 8, 12, 16, 20}
    assert {index for index, record in assignments.items() if "gpu=4" in record} == {2, 5, 9, 13, 17, 21}
    assert {index for index, record in assignments.items() if "gpu=5" in record} == {3, 6, 10, 14, 18}
    assert {index for index, record in assignments.items() if "gpu=6" in record} == {7, 11, 15, 19}
    runner_logs = list((run_root / "runners").glob(f"raw-no-luna-{condition}-resume-*.log"))
    main_logs = [path for path in runner_logs if "-task" not in path.name]
    assert len(main_logs) == 1
    assert f"Stage 2 OOD native-HF/PEFT raw/no-Luna condition complete: {condition}" in main_logs[0].read_text()


def test_resume_launcher_is_syntactically_valid_and_preserves_the_strict_contract():
    root = Path(__file__).resolve().parents[1]
    launcher = root / "scripts" / "ctm_stage2_ood_hf_peft_resume_condition_20260803.sh"
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    text = launcher.read_text(encoding="utf-8")
    assert "--isolate-tasks" in text
    assert '"max_connections":8' in text
    assert '"max_tokens":20480' in text
    assert '"prompt_style": "none"' in text
    assert "verify-clean" in text
    assert "raw_preflight" in text
    assert "--allow-inflight-partials" in text
    assert "grade_luna" not in text
    assert "rm -rf" not in text
