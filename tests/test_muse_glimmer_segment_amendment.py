"""Regression checks for the validation-only Muse boundary amendment."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from experiments.muse_glimmer_rmct_replication import plan
from infra.isambard import muse_glimmer_rmct_segment_amendment as amendment


ROOT = Path(__file__).resolve().parents[1]


def test_metric_progress_replays_skipped_optimizer_updates(tmp_path: Path) -> None:
    run = tmp_path / "logs" / plan.CONDITION_NAME / plan.run_name(0)
    run.mkdir(parents=True)
    skipped = {7, 8, 14}
    optimizer_step = 0
    rows = []
    for step in range(1, 17):
        is_skipped = int(step in skipped)
        optimizer_step += 1 - is_skipped
        rows.append(
            {
                "step": step,
                "train/skipped_empty_batch": is_skipped,
                "train/optimizer_step": optimizer_step,
                "rollout/grader_failure_count": 0,
            }
        )
    rows.append({"step": 16, "final_checkpoint": "file:///ignored"})
    (run / "metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    progress = amendment._metric_progress(tmp_path, 0)

    assert progress["global_step_end"] == 16
    assert progress["optimizer_step"] == 13
    assert progress["realized_optimizer_updates"] == 13
    assert progress["skipped_empty_batches"] == 3
    assert progress["skipped_global_steps"] == [7, 8, 14]


def test_lora_coverage_uses_exact_trainable_and_safetensor_names(tmp_path: Path) -> None:
    names = []
    tensors = {}
    for layer in range(2):
        for family, projections in amendment.EXPECTED_FAMILY_PROJECTIONS.items():
            for projection in projections:
                for kind in ("A", "B"):
                    stem = (
                        f"base_model.model.model.language_model.layers.{layer}."
                        f"{family}.{projection}.lora_{kind}"
                    )
                    names.append(stem + ".default.weight")
                    tensors[stem + ".weight"] = torch.zeros((1, 1))
    save_file(tensors, tmp_path / "adapter_model.safetensors")
    adapter = {"target_modules": sorted(amendment.EXPECTED_TARGET_MODULES)}
    manifest = {"trainable_parameter_names": names}

    amendment._validate_lora_coverage(tmp_path, adapter, manifest)


def test_amended_orchestration_preserves_frozen_training_and_no_cap_contract() -> None:
    runner = (ROOT / "infra/isambard/run_muse_glimmer_rmct_segment_amended.py").read_text(
        encoding="utf-8"
    )
    sbatch = (ROOT / "infra/isambard/run_muse_glimmer_rmct_segment_amended.sbatch").read_text(
        encoding="utf-8"
    )
    submitter = (ROOT / "infra/isambard/submit_muse_glimmer_rmct_next_amended.sh").read_text(
        encoding="utf-8"
    )
    validator = (ROOT / "infra/isambard/muse_glimmer_rmct_segment_amendment.py").read_text(
        encoding="utf-8"
    )

    assert "from infra.isambard.run_muse_glimmer_rmct_segment import main as frozen_main" in runner
    assert "amendment.validate_attestation(root)" in runner
    assert "amendment.install()" in runner
    assert "run_muse_glimmer_rmct_segment_amended.py" in sbatch
    assert "source infra/isambard/muse_glimmer_cuda129_runtime_env.sh" in sbatch
    assert "--gpus=4" in sbatch
    assert '--reservation="$ctm_muse_reservation"' in submitter
    assert "sbatch --test-only" in submitter
    assert "comparison_terminal_global_step" in submitter
    assert '"output_token_cap": None' in validator
    assert '"generation_termination": "model_eos_only"' in validator
    for source in (runner, sbatch, submitter, validator):
        assert "--max-new-tokens" not in source
        assert "--max-tokens" not in source
        assert "--max-output-tokens" not in source

