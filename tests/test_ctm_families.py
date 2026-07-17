"""Offline tests for the shared prompt-family artifact contract."""

import json

import pytest

from ctm.settings.families import (
    FAMILY_SCHEMA_VERSION,
    FamilyValidationError,
    load_families,
    make_family_perturbations,
    select_fixed_variants,
    validate_family,
    write_frozen_artifact,
)


def _family(source_id="item-1", n_variants=3):
    return {
        "schema_version": FAMILY_SCHEMA_VERSION,
        "source_id": source_id,
        "source": "unit",
        "reference_messages": [{"role": "user", "content": "plain"}],
        "variants": [
            {
                "variant_id": f"v-{index}",
                "messages": [{"role": "user", "content": f"variant {index}"}],
                "axes": {"fold": index},
            }
            for index in range(n_variants)
        ],
        "metadata": {"valence": "benign"},
    }


def test_family_validation_and_fixed_perturbations():
    row = validate_family(_family(), min_variants=3)
    perturbations = make_family_perturbations(2)
    assert perturbations[0](row)["messages"][0]["content"] == "plain"
    assert perturbations[1](row)["messages"][0]["content"] == "variant 0"
    assert perturbations[2](row)["messages"][0]["content"] == "variant 1"

    controls = make_family_perturbations(2, control=True)
    assert all(fn(row)["messages"][0]["content"] == "plain" for fn in controls)


def test_validation_rejects_mixed_or_short_schema():
    row = _family(n_variants=1)
    row["schema_version"] = 99
    with pytest.raises(FamilyValidationError, match="schema_version"):
        validate_family(row)
    with pytest.raises(FamilyValidationError, match="needs at least 2"):
        validate_family(_family(n_variants=1), min_variants=2)


def test_fixed_variant_selection_is_order_independent_and_exact():
    variants = _family(n_variants=5)["variants"]
    selected = select_fixed_variants(variants, source_id="x", n_variants=3, seed="s")
    reversed_selected = select_fixed_variants(list(reversed(variants)), source_id="x", n_variants=3, seed="s")
    assert selected == reversed_selected
    assert len(selected) == 3
    with pytest.raises(FamilyValidationError, match="needs 6"):
        select_fixed_variants(variants, source_id="x", n_variants=6)


def test_load_families_is_strict_and_takes_fixed_prefix(tmp_path):
    path = tmp_path / "families.jsonl"
    write_frozen_artifact(path, [_family(f"item-{i}") for i in range(3)], provenance={"revision": "unit"})
    rows = load_families(path, n_datapoints=2, n_variants=2)
    assert [row["source_id"] for row in rows] == ["item-0", "item-1"]
    assert all(len(row["variants"]) == 2 for row in rows)
    assert load_families(path, n_datapoints=0) == []
    with pytest.raises(ValueError, match="only 3/4 requested families"):
        load_families(path, n_datapoints=4)


def test_load_families_rejects_content_changed_after_freeze(tmp_path):
    path = tmp_path / "families.jsonl"
    write_frozen_artifact(path, [_family()], provenance={"revision": "unit"})
    path.write_text(json.dumps(_family("changed")) + "\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_families(path)


def test_frozen_artifact_has_identity_manifest_and_never_overwrites(tmp_path):
    path = tmp_path / "train.jsonl"
    manifest = write_frozen_artifact(path, [_family("b"), _family("a")], provenance={"revision": "abc"})
    assert manifest["row_count"] == 2
    assert manifest["provenance"] == {"revision": "abc"}
    disk = json.loads(path.with_suffix(".jsonl.manifest.json").read_text())
    assert disk["content_sha256"] == manifest["content_sha256"]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_frozen_artifact(path, [_family("a")], provenance={"revision": "different"})
