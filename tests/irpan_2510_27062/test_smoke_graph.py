from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from scripts import run_evals, train_bct, train_opct, train_rlct
from scripts.irpan_2510_27062 import cli as adapter_cli
from scripts.run_experiment import (
    command_argv,
    compile_experiment,
    initial_context,
    load_experiment_source,
)

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "experiments/paper_reproductions/irpan_2510_27062/debug/smoke.yaml"
ADAPTER_MODULE = "scripts.irpan_2510_27062"


def _temporary_smoke_source(tmp_path: Path) -> dict[str, Any]:
    source = load_experiment_source(SMOKE)
    root = tmp_path / "paper-smoke"
    fixtures = root / "fixtures"
    source["name"] = "irpan-2510-27062-test-smoke"
    source["spec"]["artifact_root"] = str(root)

    sycophancy = source["spec"]["data"]["sycophancy"]
    sycophancy["training_artifact"] = str(fixtures / "sycophancy-training.jsonl")
    sycophancy["bct_result_export"] = str(fixtures / "sycophancy-bct-results.jsonl")
    sycophancy["validation"]["mmlu"] = str(fixtures / "mmlu-validation.jsonl")
    sycophancy["final"]["mmlu"] = str(fixtures / "mmlu-final.jsonl")

    jailbreak = source["spec"]["data"]["jailbreak"]
    jailbreak["training_artifact"] = str(fixtures / "jailbreak-training.jsonl")
    jailbreak["bct_result_export"] = str(fixtures / "jailbreak-bct-results.jsonl")
    jailbreak["validation"]["harmbench"] = str(fixtures / "harmbench-validation.jsonl")
    jailbreak["validation"]["or_bench"] = str(fixtures / "or-bench-validation.jsonl")
    jailbreak["final"]["clearharm"] = str(fixtures / "clearharm-final.jsonl")
    jailbreak["final"]["wildguardtest"] = str(fixtures / "wildguardtest-final.jsonl")
    jailbreak["final"]["xstest"] = str(fixtures / "xstest-final.jsonl")
    jailbreak["final"]["wildjailbreak"] = str(fixtures / "wildjailbreak-final.jsonl")
    return source


def _run_adapter_entry(entry: dict[str, Any], context: dict[str, Any]) -> None:
    argv = command_argv(entry, context)
    assert argv[1:3] == ["-m", ADAPTER_MODULE]
    adapter_cli.main(argv[3:])


def _backend_must_not_initialize(*_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("a checked smoke command initialized a backend")


def test_compiled_smoke_graph_reaches_every_actual_method_and_eval_parser(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    plan = compile_experiment(_temporary_smoke_source(tmp_path))
    context = initial_context(plan)

    _run_adapter_entry(plan["data_generation"], context)
    for entry in plan["data_preparation"]:
        _run_adapter_entry(entry, context)
    capsys.readouterr()

    monkeypatch.setattr(train_bct, "build_backend", _backend_must_not_initialize)
    monkeypatch.setattr(train_rlct, "build_backend", _backend_must_not_initialize)
    monkeypatch.setattr(train_opct, "build_backend", _backend_must_not_initialize)
    monkeypatch.setattr(run_evals, "run_task_evals", _backend_must_not_initialize)

    executed_training: set[str] = set()
    for entry in plan["training"]:
        argv = command_argv(entry, context)
        script = Path(argv[1]).name
        args = argv[2:]
        if script == "train_bct.py":
            monkeypatch.setattr(sys, "argv", [script, *args])
            train_bct.main()
        elif script == "train_rlct.py":
            train_rlct.main(args)
        elif script == "train_opct.py":
            train_opct.main(args)
        else:  # pragma: no cover - a new training entry must make routing explicit
            raise AssertionError(f"unexpected training script: {script}")
        executed_training.add(entry["name"])
        capsys.readouterr()

    assert executed_training == {
        f"{domain}_{method}"
        for domain in ("sycophancy", "jailbreak")
        for method in ("bct", "rmct", "act", "attct", "mlpct", "opct")
    }

    executed_evaluations = []
    for entry in plan["evaluation"]:
        argv = command_argv(entry, context)
        assert Path(argv[1]).name == "run_evals.py"
        run_evals.main(argv[2:])
        executed_evaluations.append(entry["name"])
        capsys.readouterr()

    # Seven candidate states run both clean/wrong-suggestion MMLU validation,
    # then the explicit selected checkpoints run the 28 final routes.
    assert len(executed_evaluations) == 42
    assert len(executed_evaluations) == len(set(executed_evaluations))
