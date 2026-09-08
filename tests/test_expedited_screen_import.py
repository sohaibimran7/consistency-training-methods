from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "ctm_expedited_screen_import",
    ROOT / "experiments" / "rmct_paper_vast_dense_models" / "import_expedited_screen.py",
)
assert _SPEC is not None and _SPEC.loader is not None
expedited = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(expedited)
AUDIT = ROOT / "experiments" / "switch_gate" / "audit.json"
LOCK = ROOT / "experiments" / "switch_gate" / "lock.json"
FULL_SHARED_PLAN = ROOT / "experiments" / "rmct_paper_vast_dense_models" / "shared_data.yaml"
EXPEDITED_PLAN = ROOT / "experiments" / "rmct_paper_vast_dense_models" / "expedited_screen" / "shared_data_import.yaml"
SOURCE = ROOT / "artifacts" / "switch-gate" / "source" / "hle-text-mc.jsonl"


def _source_rows(count: int = 112) -> list[dict]:
    return [
        {
            "source_id": f"source-{index}",
            "question": f"Question {index}?",
            "options": [f"Option {index} A", f"Option {index} B", f"Option {index} C"],
            "answer": "A",
        }
        for index in range(count)
    ]


def _suite_fixture(tmp_path: Path):
    source_rows = _source_rows()
    stage1_rows = expedited._stage1_source_rows(source_rows, seed="42", count=100)
    stage1_ids = {expedited._question_id(row) for row in stage1_rows}
    outside = [row for row in source_rows if expedited._question_id(row) not in stage1_ids]
    imported_source = [*stage1_rows[:88], *outside]
    assert len(imported_source) == 100

    unbiased_rows = []
    for source in imported_source:
        unbiased_rows.append(
            {
                "question": source["question"],
                "question_id": expedited._question_id(source),
                "source_dataset": "hle-text-mc",
                "prompt_style": "none",
                "unbiased_messages": [
                    {
                        "role": "user",
                        "content": expedited._parsed_input(source) + expedited._ANSWER_FORMAT_INSTRUCTION,
                    }
                ],
                "ground_truth": source["answer"],
            }
        )

    def payload(rows: list[dict]) -> bytes:
        return b"".join((json.dumps(row) + "\n").encode() for row in rows)

    files = {
        "unbiased": (tmp_path / "unbiased.jsonl", payload(unbiased_rows), unbiased_rows),
    }
    for bias in expedited.BIAS_ORDER:
        rows = [
            {
                **row,
                "biased_messages": [{"role": "user", "content": f"frozen {bias} prompt"}],
                "bias_type": bias,
                "biased_option": expedited._biased_option(source),
                "biasing_text": f"frozen {bias} bias",
            }
            for row, source in zip(unbiased_rows, imported_source, strict=True)
        ]
        files[bias] = (tmp_path / f"{bias}.jsonl", payload(rows), rows)

    ids = [row["question_id"] for row in unbiased_rows]
    slug = hashlib.sha1("\n".join(sorted(ids)).encode()).hexdigest()[:10]
    for name, (_, frozen_payload, rows) in list(files.items()):
        model = "_args-vllm-google-gemma-4-31b-it" if name == "wrong_argument" else ""
        files[name] = (
            tmp_path / f"hle-text-mc_{name}_none_n100_seed42{model}_ids-{slug}.jsonl",
            frozen_payload,
            rows,
        )
    audit = {
        "frozen_hle_evaluation": {
            "question_count": 100,
            "question_id_slug": slug,
        }
    }
    return source_rows, files, [{"question_id": question_id} for question_id in ids], audit


def test_validator_requires_exact_complete_suite_and_labels_known_divergence(tmp_path):
    source_rows, files, question_ids, audit = _suite_fixture(tmp_path)

    divergence = expedited._validate_suite(
        source_rows=source_rows,
        files=files,
        question_id_rows=question_ids,
        audit=audit,
    )

    assert divergence["overlap_count"] == 88
    assert divergence["imported_only_count"] == 12
    assert divergence["full_stage1_only_count"] == 12

    path, payload, rows = files["wrong_argument"]
    files["wrong_argument"] = (path, payload, rows[:92])
    with pytest.raises(ValueError, match="exactly 100 rows, found 92"):
        expedited._validate_suite(
            source_rows=source_rows,
            files=files,
            question_id_rows=question_ids,
            audit=audit,
        )


def test_full_stage1_reference_remains_unrestricted_100_pool_with_accepted_wrong_argument_floor():
    spec = expedited._full_stage1_spec(FULL_SHARED_PLAN)

    assert spec["n_questions"] == 100
    assert spec["min_n_questions"] == 92
    assert spec["seed"] == "42"
    assert spec["prompt_style"] == "none"
    assert spec["argument_model"] == "openrouter/google/gemma-4-31b-it"
    assert spec["shared_root"] == "artifacts/rmct-hle-dense-models-shared"


def test_expedited_plan_is_offline_opt_in_and_uses_only_the_disjoint_12h_root():
    import yaml

    plan = yaml.safe_load(EXPEDITED_PLAN.read_text(encoding="utf-8"))
    assert set(plan) == {"name", "data_generation"}
    assert len(plan["data_generation"]) == 1
    entry = plan["data_generation"][0]
    assert entry["resource"] == "cpu"
    assert entry["command"][-1] == "experiments.rmct_paper_vast_dense_models.import_expedited_screen"
    assert entry["args"]["output_root"] == "artifacts/rmct-hle-dense-models-shared-expedited-12h"
    assert entry["args"]["full_shared_plan"] == "experiments/rmct_paper_vast_dense_models/shared_data.yaml"


@pytest.mark.skipif(not SOURCE.is_file(), reason="gated local HLE artifacts are not present")
def test_real_audited_suite_import_is_byte_exact_and_never_uses_full_root(tmp_path, monkeypatch):
    output = tmp_path / "rmct-hle-dense-models-shared-expedited-12h"
    monkeypatch.setattr(expedited, "EXPEDITED_OUTPUT_ROOT", str(output))

    manifest = expedited.import_expedited_screen(
        repository_root=ROOT,
        audit_path=AUDIT,
        lock_path=LOCK,
        full_shared_plan=FULL_SHARED_PLAN,
        output_root=output,
    )

    assert manifest["classification"] == "EXPEDITED_SCREEN_ONLY_NOT_FULL_STAGE1"
    assert manifest["divergence_from_full_stage1"]["compatible_with_full_stage1"] is False
    question_pool = manifest["divergence_from_full_stage1"]["question_pool"]
    assert question_pool["overlap_count"] == 88
    assert question_pool["imported_ordered_question_ids_sha256"] == "e2d1f7231801b12fb293bfc95ef17859dae311b68a312174fe51b5f27b35a918"
    assert question_pool["full_stage1_ordered_question_ids_sha256"] == "fa24a642f717d21a4b805b2ee2ce584ec24aee6aed6fe00d053ef26e0f855d0e"
    assert manifest["consumer_contract"]["forbidden_full_stage1_task_factory"] == "mcq_bias.tasks:suite_tasks"
    for relative, identity in manifest["copies"].items():
        copied = output / relative
        source = ROOT / identity["source_path"]
        assert copied.read_bytes() == source.read_bytes()
        assert hashlib.sha256(copied.read_bytes()).hexdigest() == identity["content_sha256"]

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        expedited.import_expedited_screen(
            repository_root=ROOT,
            audit_path=AUDIT,
            lock_path=LOCK,
            full_shared_plan=FULL_SHARED_PLAN,
            output_root=output,
        )
