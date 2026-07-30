from __future__ import annotations

import json
from pathlib import Path

from scripts.irpan_2510_27062 import cli
from scripts.irpan_2510_27062.cli import main


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


def test_materialize_retained_pairs_routes_to_shared_pair_artifact(tmp_path: Path, monkeypatch, capsys) -> None:
    retained = tmp_path / "retained.jsonl"
    output = tmp_path / "pairs.jsonl"
    calls = []

    def fake_materialize(source: Path, target: Path) -> dict:
        calls.append((source, target))
        return {
            "artifact_schema": "ctm.prompt_pairs",
            "row_count": 3,
            "content_sha256": "a" * 64,
        }

    monkeypatch.setattr(cli, "materialize_retained_prompt_pairs", fake_materialize)
    main(
        [
            "materialize-retained-pairs",
            "--retained",
            str(retained),
            "--output",
            str(output),
        ]
    )

    assert calls == [(retained, output)]
    assert '"artifact_schema": "ctm.prompt_pairs"' in capsys.readouterr().out


def test_validation_selection_cli_writes_a_validation_only_audit(tmp_path: Path) -> None:
    observations = tmp_path / "observations.json"
    observations.write_text(
        json.dumps(
            [
                {
                    "candidate_id": "a",
                    "benchmark": "harmbench",
                    "stage": "validation",
                    "metric": "harmful_asr",
                    "value": 0.2,
                },
                {
                    "candidate_id": "a",
                    "benchmark": "or_bench",
                    "stage": "validation",
                    "metric": "answered_benign",
                    "value": 0.8,
                },
                {
                    "candidate_id": "b",
                    "benchmark": "harmbench",
                    "stage": "validation",
                    "metric": "harmful_asr",
                    "value": 0.1,
                },
                {
                    "candidate_id": "b",
                    "benchmark": "or_bench",
                    "stage": "validation",
                    "metric": "answered_benign",
                    "value": 0.9,
                },
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "selection.json"

    main(
        [
            "select-validation",
            "--domain",
            "jailbreak",
            "--input",
            str(observations),
            "--output",
            str(output),
        ]
    )

    audit = json.loads(output.read_text(encoding="utf-8"))
    assert audit["selected_candidate_id"] == "b"
    assert audit["input_stage"] == "validation"
