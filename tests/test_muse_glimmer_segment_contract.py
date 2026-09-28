"""Custody and no-cap checks for the Muse RMCT segment runner."""

from __future__ import annotations

import json
import math
from pathlib import Path

from experiments.muse_glimmer_rmct_replication import plan
from infra.isambard import muse_glimmer_rmct_segment_contract as contract
from scripts.run_experiment import _argument_tokens


ROOT = Path(__file__).resolve().parents[1]


def test_segment_runner_compiles_only_the_explicit_no_cap_switch(tmp_path):
    snapshot = tmp_path / plan.MODEL_REVISION
    snapshot.mkdir()
    (snapshot / "config.json").write_text('{"model_type":"muse_glimmer"}\n', encoding="utf-8")

    tokens = _argument_tokens(plan.segment_args(ROOT, 0, model_snapshot=snapshot))

    assert "--no-max-new-tokens" in tokens
    assert "--max-new-tokens" not in tokens
    assert "--max-tokens" not in tokens


def test_segment_sbatch_sources_pinned_runtime_and_has_no_generation_cap():
    source = (ROOT / "infra" / "isambard" / "run_muse_glimmer_rmct_segment.sbatch").read_text(
        encoding="utf-8"
    )

    assert "source infra/isambard/muse_glimmer_cuda129_runtime_env.sh" in source
    assert "model-snapshot.json" in source
    assert 'export TMPDIR="$ctm_muse_job_tmp"' in source
    assert "SLURM_TMPDIR" not in source
    assert "HF_HUB_OFFLINE=1" in source
    assert "TRANSFORMERS_OFFLINE=1" in source
    assert "--gpus=4" in source
    assert "max-new-tokens" not in source
    assert "max-tokens" not in source


def test_segment_launcher_preserves_venv_python_spelling():
    source = (ROOT / "infra" / "isambard" / "run_muse_glimmer_rmct_segment.py").read_text(
        encoding="utf-8"
    )

    assert 'Path(os.environ.get("CTM_MUSE_RUNTIME_PYTHON", ""))' in source
    assert 'Path(os.environ.get("CTM_MUSE_RUNTIME_PYTHON", "")).resolve()' not in source


def test_convergence_window_allows_explicit_zero_count_observations(tmp_path):
    run = tmp_path / "logs" / plan.CONDITION_NAME / plan.run_name(0)
    run.mkdir(parents=True)
    records = []
    expected_count = 0
    expected_sum = 0.0
    for step in range(1, plan.UPDATES_PER_SEGMENT + 1):
        record = {"step": step}
        for perturbation in contract.EXPECTED_TRAINING_PERTURBATIONS:
            count = 0 if step == 1 and perturbation == 1 else 2
            absolute_sum = 0.0 if count == 0 else 0.1
            record[f"train/consistency_gap_abs_sum_{perturbation}"] = absolute_sum
            record[f"train/consistency_gap_abs_count_{perturbation}"] = count
            if count:
                record[f"train/consistency_gap_abs_mean_{perturbation}"] = absolute_sum / count
            expected_count += count
            expected_sum += absolute_sum
        records.append(record)
    (run / "metrics.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    window = contract._window_metrics(tmp_path, 0)

    assert window["resolved_observations"] == expected_count
    assert window["expected_observations"] == 64
    assert math.isclose(window["weighted_abs_gap"], expected_sum / expected_count)
