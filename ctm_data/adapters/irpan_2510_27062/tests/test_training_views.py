"""Offline tests for the canonical paired-prompt training boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ctm.artifacts import artifact_manifest_path
from ctm.training.bct_targets import prepare_paired_prompts
from ctm.training.consistency_data import build_consistency_datums
from ctm.training.opct import OPCTConfig, validate_opct_samples
from ctm.training.sft import load_samples
from ctm_data.adapters.irpan_2510_27062.artifacts import (
    MANIFEST_SHA256_FIELD,
    producer_identity,
    write_artifact,
)
from ctm_data.adapters.irpan_2510_27062.schema import (
    make_derived_record,
    make_source_record,
    sha256_json,
    sha256_text,
)
from ctm_data.adapters.irpan_2510_27062.sycophancy import (
    PROMPT_PAIR_ARTIFACT_KIND,
    build_sycophancy_pairs,
    normalize_arc_rows,
)
from ctm_data.adapters.irpan_2510_27062.training_views import (
    JAILBREAK_DOMAIN,
    SYCOPHANCY_DOMAIN,
    TRAINING_VIEW_ARTIFACT_KIND,
    TrainingViewError,
    materialize_training_view,
    read_training_view,
)


def _arc_rows() -> list[dict[str, Any]]:
    return [
        {
            "id": identifier,
            "question": question,
            "choices": {
                "label": ["A", "B", "C"],
                "text": ["alpha", "beta", "gamma"],
            },
            "answerKey": answer,
        }
        for identifier, question, answer in (
            ("arc-2", "Which token is second?", "B"),
            ("arc-1", "Which token is first?", "A"),
        )
    ]


def _write_sycophancy_source(path: Path, *, role: str = "training", kind: str = PROMPT_PAIR_ARTIFACT_KIND) -> None:
    normalized = normalize_arc_rows(
        _arc_rows(),
        subset="fixture",
        split="train",
        revision="fixture-arc-v1",
    )
    rows = build_sycophancy_pairs(normalized, wrong_option_seed=17)
    write_artifact(
        path,
        rows,
        artifact_kind=kind,
        role=role,
        producer=producer_identity("training-view-sycophancy-fixture", __file__),
        config={"fixture": "sycophancy"},
    )


def _act_rows() -> list[dict[str, Any]]:
    source = make_source_record(
        record_type="harmbench_prompt",
        source="harmbench",
        source_key="fixture-behavior",
        payload={"prompt": "Describe a fictional museum safety audit."},
        metadata={"fixture": True},
    )
    core = source["payload"]["prompt"]
    shared = "[Shared instruction] Keep the answer concise."
    clean = f"{core}\n\n{shared}"
    rows: list[dict[str, Any]] = []
    for wrapper_id, prefix in reversed(
        (
            ("role_play_v1", "Fictional role-play: "),
            ("marked_v1", "<audit-item>"),
        )
    ):
        candidate_id = f"{source['example_id']}:wrapper:{wrapper_id}"
        wrapped = f"{prefix}{core}\n\n{shared}"
        rows.append(
            make_derived_record(
                record_type="act_training_export",
                example_id=f"{candidate_id}:training:act:fixture_v1",
                source="harmbench",
                source_key=f"fixture-behavior::{wrapper_id}::act",
                payload={
                    "source_id": candidate_id,
                    "candidate_id": candidate_id,
                    "reference_messages": [{"role": "user", "content": clean}],
                    "variant_messages": [{"role": "user", "content": wrapped}],
                    "alignment_text": core,
                    "alignment_text_sha256": sha256_text(core),
                    "clean_prompt_sha256": sha256_text(clean),
                    "wrapped_prompt_sha256": sha256_text(wrapped),
                    "training_export_version": "fixture_v1",
                },
                parent_hashes=[source["content_sha256"]],
                metadata={"training_method": "act"},
            )
        )
    return rows


def _write_jailbreak_source(
    path: Path,
    *,
    role: str = "training",
    kind: str = "act_training_exports",
    record_type: str = "act_training_export",
) -> None:
    rows = _act_rows()
    if record_type != "act_training_export":
        rows = [
            make_derived_record(
                record_type=record_type,
                example_id=row["example_id"],
                source=row["source"],
                source_key=row["source_key"],
                payload=row["payload"],
                parent_hashes=row["parent_hashes"],
                metadata=row["metadata"],
            )
            for row in rows
        ]
    write_artifact(
        path,
        rows,
        artifact_kind=kind,
        role=role,
        producer=producer_identity("training-view-jailbreak-fixture", __file__),
        config={"fixture": "jailbreak"},
    )


class _CharacterTokenizer:
    """Fast-tokenizer-shaped fixture; character IDs preserve matching suffixes."""

    chat_template = None

    def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool) -> dict[str, Any]:
        assert not add_special_tokens
        assert return_offsets_mapping
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


@pytest.mark.parametrize(
    ("domain", "source_writer"),
    [
        (SYCOPHANCY_DOMAIN, _write_sycophancy_source),
        (JAILBREAK_DOMAIN, _write_jailbreak_source),
    ],
)
def test_training_view_is_deterministic_immutable_and_lineage_complete(
    tmp_path: Path,
    domain: str,
    source_writer,
) -> None:
    source_a = tmp_path / f"{domain}-a.source.jsonl"
    source_b = tmp_path / f"{domain}-b.source.jsonl"
    source_writer(source_a)
    source_writer(source_b)
    view_a = tmp_path / f"{domain}-a.view.jsonl"
    view_b = tmp_path / f"{domain}-b.view.jsonl"
    manifest_a = materialize_training_view(source_a, view_a, domain=domain)
    manifest_b = materialize_training_view(source_b, view_b, domain=domain)
    rows_a, verified_a = read_training_view(view_a, expected_domain=domain)
    rows_b, _ = read_training_view(view_b, expected_domain=domain)

    assert rows_a == rows_b
    assert manifest_a["content_sha256"] == manifest_b["content_sha256"]
    assert manifest_a["content_sha256"] == verified_a["content_sha256"]
    assert verified_a["provenance"]["role"] == "training"
    assert verified_a["provenance"]["artifact_kind"] == TRAINING_VIEW_ARTIFACT_KIND
    assert verified_a["provenance"]["source_parent_identity"]["content_sha256"]
    assert verified_a["provenance"]["source_parent_identity"][MANIFEST_SHA256_FIELD]
    assert verified_a["provenance"]["producer"]["code_sha256"]
    assert verified_a["provenance"]["config_sha256"] == sha256_json(verified_a["provenance"]["config"])
    assert [row["pair_id"] for row in rows_a] == list(dict.fromkeys(row["pair_id"] for row in rows_a))
    assert all(row["source_id"] == row["pair_id"] for row in rows_a)
    assert all(row["reference_messages"] and row["variant_messages"] for row in rows_a)
    if domain == SYCOPHANCY_DOMAIN:
        assert all(row["choice_labels"] == [choice["label"] for choice in row["choices"]] for row in rows_a)
        assert all(row["correct_label"] in row["choice_labels"] for row in rows_a)
        assert all(row["suggested_wrong_label"] in row["choice_labels"] for row in rows_a)
    else:
        assert len({row["example_id"] for row in rows_a}) == 1
        assert len({row["variant_id"] for row in rows_a}) == len(rows_a)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        materialize_training_view(source_a, view_a, domain=domain)


@pytest.mark.parametrize(
    ("domain", "source_writer"),
    [
        (SYCOPHANCY_DOMAIN, _write_sycophancy_source),
        (JAILBREAK_DOMAIN, _write_jailbreak_source),
    ],
)
def test_same_pair_file_passes_actual_act_attct_mlpct_and_opct_boundaries(
    tmp_path: Path,
    domain: str,
    source_writer,
) -> None:
    source = tmp_path / f"{domain}.source.jsonl"
    view = tmp_path / f"{domain}.pairs.jsonl"
    source_writer(source)
    materialize_training_view(source, view, domain=domain)

    # This is the public loader used by the unified ACT/AttCT/MLPCT script.
    samples = load_samples(view)
    assert samples == read_training_view(view)[0]
    tokenizer = _CharacterTokenizer()
    for method in ("act", "attct", "mlpct"):
        datums, skipped = build_consistency_datums(
            tokenizer,
            samples,
            alignment_text_field="alignment_text",
        )
        assert method  # Documents that the same public path serves all three methods.
        assert len(datums) == len(samples)
        assert skipped == 0

    # OPCT's public pre-backend validator accepts those exact loaded objects.
    config = OPCTConfig()
    assert validate_opct_samples(samples, config) == samples

    # The repository's BCT prompt preparer can also select clean/variant fields
    # directly; source_id is the canonical pair key, not an inferred row index.
    prepared = prepare_paired_prompts(
        samples,
        source_messages_field="reference_messages",
        main_messages_field="variant_messages",
        control_messages_field="reference_messages",
    )
    assert [prompt.source_id for prompt in prepared] == [row["pair_id"] for row in samples]


def test_training_view_rejects_nontraining_role_wrong_kind_and_wrong_record_type(tmp_path: Path) -> None:
    eval_source = tmp_path / "eval.jsonl"
    _write_sycophancy_source(eval_source, role="final_eval")
    with pytest.raises(TrainingViewError, match="role.*expected 'training'"):
        materialize_training_view(eval_source, tmp_path / "eval-view.jsonl", domain=SYCOPHANCY_DOMAIN)

    wrong_kind = tmp_path / "wrong-kind.jsonl"
    _write_sycophancy_source(wrong_kind, kind="sycophancy_lookalike")
    with pytest.raises(TrainingViewError, match="artifact kind"):
        materialize_training_view(wrong_kind, tmp_path / "wrong-kind-view.jsonl", domain=SYCOPHANCY_DOMAIN)

    wrong_type = tmp_path / "wrong-type.jsonl"
    _write_jailbreak_source(wrong_type, record_type="bct_training_export")
    with pytest.raises(TrainingViewError, match="record_type"):
        materialize_training_view(wrong_type, tmp_path / "wrong-type-view.jsonl", domain=JAILBREAK_DOMAIN)


def test_training_view_reader_rejects_role_and_row_tampering(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    view = tmp_path / "view.jsonl"
    _write_sycophancy_source(source)
    materialize_training_view(source, view, domain=SYCOPHANCY_DOMAIN)

    manifest_path = artifact_manifest_path(view)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["provenance"]["role"] = "final_eval"
    unsigned = {key: value for key, value in manifest.items() if key != MANIFEST_SHA256_FIELD}
    manifest[MANIFEST_SHA256_FIELD] = sha256_json(unsigned)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="role.*expected 'training'"):
        read_training_view(view)

    # Rebuild a fresh pair, then alter its bytes without touching the sidecar.
    second_view = tmp_path / "second-view.jsonl"
    materialize_training_view(source, second_view, domain=SYCOPHANCY_DOMAIN)
    second_view.write_text(second_view.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        read_training_view(second_view)
