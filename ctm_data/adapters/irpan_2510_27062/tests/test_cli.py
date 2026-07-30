from __future__ import annotations

import json
from pathlib import Path

from ctm_data.adapters.irpan_2510_27062.artifacts import producer_identity, write_artifact
from ctm_data.adapters.irpan_2510_27062.bct_targets import (
    make_fixture_generator_identity,
    read_bct_target_requests,
    read_bct_training_data,
)
from ctm_data.adapters.irpan_2510_27062.cli import main
from ctm_data.adapters.irpan_2510_27062.sycophancy import (
    PROMPT_PAIR_ARTIFACT_KIND,
    build_sycophancy_pairs,
    normalize_arc_rows,
)


def test_inventory_command_writes_no_source_rows(tmp_path: Path, capsys) -> None:
    output = tmp_path / "inventory.json"
    main(["inventory", "--output", str(output)])
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert set(payload["sources"]) == {
        "arc",
        "openbookqa",
        "bbh",
        "mmlu",
        "harmbench",
        "or_bench",
        "clearharm",
        "wildguardtest",
        "xstest",
        "wildjailbreak",
    }
    assert "paper_facts" in payload["reproduction_boundary"]
    assert payload["partitions"]["harmbench"].keys() == {"training", "validation"}
    assert "wildjailbreak" in capsys.readouterr().out


def test_materialize_and_verify_commands(tmp_path: Path, capsys) -> None:
    source = tmp_path / "or-bench.jsonl"
    source.write_text(json.dumps({"id": "fixture-1", "prompt": "Describe a museum safety checklist."}) + "\n")
    output = tmp_path / "eval.jsonl"
    main(
        [
            "materialize-eval",
            "--source-path",
            str(source),
            "--output",
            str(output),
            "--source",
            "or_bench",
            "--subset",
            "fixture",
            "--split",
            "validation",
            "--revision",
            "fixture-revision",
            "--expected-count",
            "1",
            "--expected-count-mode",
            "strict",
        ]
    )
    assert output.is_file()
    main(["verify", str(output), "--kind", "eval_source"])
    printed = capsys.readouterr().out
    assert '"row_count": 1' in printed
    assert '"artifact_kind": "eval_source"' in printed


def test_training_view_and_bct_cli_chain_is_offline_and_loader_native(tmp_path: Path, capsys) -> None:
    source = tmp_path / "sycophancy-source.jsonl"
    normalized = normalize_arc_rows(
        [
            {
                "id": "arc-smoke",
                "question": "Which symbol is first?",
                "choices": {"label": ["A", "B"], "text": ["alpha", "beta"]},
                "answerKey": "A",
            }
        ],
        subset="fixture",
        split="train",
        revision="fixture-v1",
    )
    pairs = build_sycophancy_pairs(normalized)
    write_artifact(
        source,
        pairs,
        artifact_kind=PROMPT_PAIR_ARTIFACT_KIND,
        role="training",
        producer=producer_identity("cli-fixture", __file__),
        config={"fixture": True},
    )

    view = tmp_path / "view.jsonl"
    requests = tmp_path / "requests.jsonl"
    targets = tmp_path / "targets.jsonl"
    bct = tmp_path / "bct.jsonl"
    main(["export-training-view", "--domain", "sycophancy", "--input", str(source), "--output", str(view)])
    main(["build-bct-requests", "--training-view", str(view), "--output", str(requests)])

    request_rows, _manifest = read_bct_target_requests(requests)
    results = tmp_path / "results.jsonl"
    results.write_text(
        "".join(
            json.dumps(
                {
                    "pair_id": row["pair_id"],
                    "clean_prompt_sha256": row["clean_prompt_sha256"],
                    "reference_messages_sha256": row["reference_messages_sha256"],
                    "request_record_sha256": row["request_record_sha256"],
                    "response": "ANSWER: A",
                    "metadata": {"fixture": True},
                }
            )
            + "\n"
            for row in request_rows
        ),
        encoding="utf-8",
    )
    main(
        [
            "import-bct-targets",
            "--requests",
            str(requests),
            "--results",
            str(results),
            "--output",
            str(targets),
            "--generator-identity",
            json.dumps(make_fixture_generator_identity("cli-v1")),
            "--decoding-parameters",
            json.dumps({"temperature": 0, "max_tokens": 16}),
        ]
    )
    main(
        [
            "export-bct-training",
            "--training-view",
            str(view),
            "--targets",
            str(targets),
            "--output",
            str(bct),
        ]
    )

    rows, manifest = read_bct_training_data(bct)
    assert rows[0]["messages"][-1] == {"role": "assistant", "content": "ANSWER: A"}
    assert manifest["provenance"]["role"] == "training"
    assert "bct_training_jsonl" in capsys.readouterr().out


def test_validation_selection_cli_writes_a_validation_only_audit(tmp_path: Path) -> None:
    observations = tmp_path / "observations.json"
    observations.write_text(
        json.dumps(
            [
                {"candidate_id": "a", "benchmark": "harmbench", "stage": "validation", "metric": "harmful_asr", "value": 0.2},
                {"candidate_id": "a", "benchmark": "or_bench", "stage": "validation", "metric": "answered_benign", "value": 0.8},
                {"candidate_id": "b", "benchmark": "harmbench", "stage": "validation", "metric": "harmful_asr", "value": 0.1},
                {"candidate_id": "b", "benchmark": "or_bench", "stage": "validation", "metric": "answered_benign", "value": 0.9},
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "selection.json"

    main(["select-validation", "--input", str(observations), "--output", str(output)])

    audit = json.loads(output.read_text(encoding="utf-8"))
    assert audit["selected_candidate_id"] == "b"
    assert audit["input_stage"] == "validation"
