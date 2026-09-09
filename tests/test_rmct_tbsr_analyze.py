import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import experiments.rmct_tbsr.analyze as analyze_module
from experiments.rmct_tbsr.analyze import CELL_SPECS, _write_json, build_report, load_inspect_cells, model_matches
from experiments.rmct_tbsr.constants import MODELS, TRAINING_COUNTS, TRAINING_SPLIT
from experiments.switch_gate.analyze import Observation


def _observation(
    question_id: str,
    *,
    model: str = MODELS[0],
    dataset: str = "logiqa",
    bias: str = "wrong_argument",
    parsed: bool = True,
    clean_target: int = 0,
    toward: int = 0,
    away: int = 0,
) -> Observation:
    if not parsed:
        return Observation(
            model=model,
            question_id=question_id,
            source_dataset=dataset,
            bias_type=bias,
            joint_parse=False,
            clean_target=None,
            toward=None,
            away=None,
            net_switch=None,
            abs_switch=None,
        )
    return Observation(
        model=model,
        question_id=question_id,
        source_dataset=dataset,
        bias_type=bias,
        clean_target=clean_target,
        toward=toward,
        away=away,
        net_switch=toward - away,
        abs_switch=toward + away,
    )


def _expected_training() -> dict[str, set[str]]:
    return {dataset: {f"{dataset}-{index}" for index in range(count)} for dataset, count in TRAINING_COUNTS.items()}


def _expected_hle() -> set[str]:
    return {f"hle-{index}" for index in range(100)}


def test_report_has_exact_matrix_order_tbsr_math_and_explicit_missing_partial_cells():
    records = [
        _observation("logiqa-0", toward=1),
        _observation("logiqa-1"),
        _observation("logiqa-2", clean_target=1, away=1),
        _observation("logiqa-3", parsed=False),
    ]
    report = build_report(
        {(MODELS[0], TRAINING_SPLIT, "logiqa", "wrong_argument"): records},
        expected_training=_expected_training(),
        expected_hle=_expected_hle(),
    )

    assert len(report["rows"]) == len(MODELS) * 8 == 24
    assert [(row["scope"], row["dataset"], row["bias_type"]) for row in report["rows"][:8]] == [
        (spec.scope, spec.dataset, spec.bias_type) for spec in CELL_SPECS
    ]
    first = report["rows"][0]
    assert first["data_status"] == "partial"
    assert first["tbsr"] == {"numerator_toward": 1, "eligible_clean_non_target": 2, "rate": 0.5}
    assert first["joint_parse"] == {"numerator": 3, "total": 4, "rate": 0.75}
    assert first["counts"]["clean_target"] == 1
    assert first["counts"]["away"] == 1
    assert first["counts"]["net_switch_sum"] == 0
    missing = report["rows"][1]
    assert missing["data_status"] == "missing"
    assert missing["tbsr"]["rate"] is None
    assert missing["joint_parse"] == {"numerator": 0, "total": 0, "rate": None}
    held_out = report["optional_micro_summaries"][1]
    assert held_out["summary"] == "paper_held_out_hle_excluding_wrong_argument"
    assert "wrong_argument" not in held_out["included_biases"]
    assert report["analysis_policy"]["descriptive_only"] is True


def test_report_rejects_extra_ids_and_score_contract_rejects_incoherent_switches():
    key = (MODELS[0], TRAINING_SPLIT, "logiqa", "wrong_argument")
    with pytest.raises(ValueError, match="outside the exact frozen population"):
        build_report(
            {key: [_observation("not-frozen")]},
            expected_training=_expected_training(),
            expected_hle=_expected_hle(),
        )

    with pytest.raises(ValueError, match=r"abs_switch must equal abs\(net_switch\)"):
        Observation(
            model=MODELS[0],
            question_id="q",
            source_dataset="logiqa",
            bias_type="wrong_argument",
            clean_target=0,
            toward=0,
            away=0,
            net_switch=0,
            abs_switch=1,
        )


def _score(clean=0, toward=0, away=None, net=0, absolute=0):
    return {
        "unbiased_matches_bias": clean,
        "towards_bias_switch": toward,
        "away_from_bias_switch": away,
        "net_switch": net,
        "abs_switch": absolute,
    }


def _fake_log(
    path: Path,
    *,
    created: str,
    model: str,
    split: str,
    toward: int,
    frozen_file: Path,
):
    del path
    return SimpleNamespace(
        status="success",
        eval=SimpleNamespace(
            model=model,
            created=created,
            task="experiments.switch_gate.tasks/switch_gate_biased",
            task_args={
                "dataset": "logiqa",
                "bias_type": "wrong_argument",
                "prompt_style": "none",
                "split": split,
                "frozen_file": str(frozen_file),
            },
        ),
        samples=[
            SimpleNamespace(
                id="logiqa-0",
                metadata={
                    "source_dataset": "logiqa",
                    "bias_type": "wrong_argument",
                    "prompt_style": "none",
                    "variant": "biased",
                },
                scores={"switch": SimpleNamespace(value=_score(toward=toward, net=toward, absolute=toward))},
            )
        ],
    )


def test_log_loader_filters_model_and_task_and_selects_latest_success(tmp_path, monkeypatch):
    old = tmp_path / "old.eval"
    new = tmp_path / "new.eval"
    wrong_model = tmp_path / "wrong-model.eval"
    wrong_split = tmp_path / "wrong-split.eval"
    wrong_artifact = tmp_path / "wrong-artifact.eval"
    expected_artifact = tmp_path / "training.jsonl"
    expected_artifact.write_text("exact", encoding="utf-8")
    other_artifact = tmp_path / "other.jsonl"
    other_artifact.write_text("other", encoding="utf-8")
    for path in (old, new, wrong_model, wrong_split, wrong_artifact):
        path.write_text(path.name, encoding="utf-8")
    logs = {
        str(old): _fake_log(
            old,
            created="2026-07-29T10:00:00Z",
            model=f"provider/{MODELS[0]}",
            split=TRAINING_SPLIT,
            toward=0,
            frozen_file=expected_artifact,
        ),
        str(new): _fake_log(
            new,
            created="2026-07-29T11:00:00Z",
            model=f"provider/{MODELS[0]}",
            split=TRAINING_SPLIT,
            toward=1,
            frozen_file=expected_artifact,
        ),
        str(wrong_model): _fake_log(
            wrong_model,
            created="2026-07-29T12:00:00Z",
            model=MODELS[1],
            split=TRAINING_SPLIT,
            toward=0,
            frozen_file=expected_artifact,
        ),
        str(wrong_split): _fake_log(
            wrong_split,
            created="2026-07-29T13:00:00Z",
            model=MODELS[0],
            split="screen",
            toward=0,
            frozen_file=expected_artifact,
        ),
        str(wrong_artifact): _fake_log(
            wrong_artifact,
            created="2026-07-29T14:00:00Z",
            model=MODELS[0],
            split=TRAINING_SPLIT,
            toward=0,
            frozen_file=other_artifact,
        ),
    }
    inspect_package = types.ModuleType("inspect_ai")
    inspect_log = types.ModuleType("inspect_ai.log")
    inspect_log.read_eval_log = lambda path, header_only=False: logs[path]
    inspect_package.log = inspect_log
    monkeypatch.setitem(sys.modules, "inspect_ai", inspect_package)
    monkeypatch.setitem(sys.modules, "inspect_ai.log", inspect_log)
    monkeypatch.setattr(
        analyze_module,
        "_discover_log_paths",
        lambda location: [old, new, wrong_model, wrong_split, wrong_artifact],
    )

    records, identities = load_inspect_cells(
        {MODELS[0]: "shared"},
        expected_files={(TRAINING_SPLIT, "logiqa", "wrong_argument"): expected_artifact},
    )

    key = (MODELS[0], TRAINING_SPLIT, "logiqa", "wrong_argument")
    assert records[key][0].toward == 1
    assert [identity["path"] for identity in identities if identity["selected"]] == [str(new.resolve())]
    assert len(identities) == 4
    rejected = next(identity for identity in identities if identity["path"] == str(wrong_artifact.resolve()))
    assert rejected["selection_reason"] == "frozen_file_does_not_match_exact_cell_artifact"
    assert model_matches(MODELS[0], MODELS[0])
    assert model_matches(MODELS[0], f"tinker-sampling/{MODELS[0]}")
    assert not model_matches(MODELS[0], MODELS[1])


def test_json_output_is_stable_and_refuses_overwrite(tmp_path):
    path = tmp_path / "analysis.json"
    report = build_report({}, expected_training=_expected_training(), expected_hle=_expected_hle())

    _write_json(path, report)
    first = path.read_bytes()
    assert json.loads(first)["schema_version"] == "rmct-tbsr-analysis-v1"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _write_json(path, report)
    assert path.read_bytes() == first
