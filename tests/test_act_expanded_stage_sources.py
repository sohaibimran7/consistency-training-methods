from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pytest

from experiments.act_expanded import stage_sources


def _write_arrow(path: Path, rows: list[dict[str, object]]) -> Path:
    table = pa.Table.from_pylist(rows)
    with path.open("wb") as handle:
        with pa.ipc.new_stream(handle, table.schema) as writer:
            writer.write_table(table)
    return path


def test_stages_logiqa_and_hellaswag_from_local_arrow_with_pinned_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logiqa_rows = [
        {"context": "Context", "query": "Question?", "options": ["A", "B"], "correct_option": 1}
    ]
    hellaswag_rows = [
        {
            "ind": 1,
            "activity_label": "activity",
            "ctx_a": "first",
            "ctx_b": "second",
            "ctx": "Prompt",
            "endings": ["ending zero", "ending one"],
            "source_id": "source",
            "split": "train",
            "split_type": "indomain",
            "label": "0",
        }
    ]
    monkeypatch.setattr(
        stage_sources,
        "PINNED_SOURCES",
        {
            "logiqa": {
                "repository": "lucasmccabe/logiqa",
                "revision": "logiqa-pin",
                "split": "train",
                "row_count": 1,
                "arrow_fields": tuple(logiqa_rows[0]),
            },
            "hellaswag": {
                "repository": "Rowan/hellaswag",
                "revision": "hellaswag-pin",
                "split": "train",
                "row_count": 1,
                "arrow_fields": tuple(hellaswag_rows[0]),
            },
        },
    )

    logiqa = stage_sources.stage_pinned_arrow(
        dataset="logiqa",
        arrow_source=_write_arrow(tmp_path / "logiqa.arrow", logiqa_rows),
        output_dir=tmp_path / "staged",
    )
    hellaswag = stage_sources.stage_pinned_arrow(
        dataset="hellaswag",
        arrow_source=_write_arrow(tmp_path / "hellaswag.arrow", hellaswag_rows),
        output_dir=tmp_path / "staged",
    )
    assert json.loads(logiqa.data_path.read_text().strip()) == {
        "ground_truth_idx": 1,
        "options": ["A", "B"],
        "question": "Context\nQuestion?",
    }
    assert json.loads(hellaswag.data_path.read_text().strip()) == {
        "ground_truth_idx": 0,
        "options": ["ending zero", "ending one"],
        "question": "Prompt",
    }
    for result, revision in ((logiqa, "logiqa-pin"), (hellaswag, "hellaswag-pin")):
        manifest = json.loads(result.manifest_path.read_text())
        assert manifest["revision"] == revision
        assert manifest["assertions"] == {
            "local_only": True,
            "no_network_or_model_call": True,
            "source_order_preserved": True,
        }
        assert result.data_path.name.endswith(f"{result.content_sha256}.jsonl")
        assert result.manifest_path.name.endswith(f"{result.manifest_sha256}.json")

    resumed = stage_sources.stage_pinned_arrow(
        dataset="logiqa",
        arrow_source=tmp_path / "logiqa.arrow",
        output_dir=tmp_path / "staged",
    )
    assert resumed.status == "resumed"
    assert resumed.data_path == logiqa.data_path
