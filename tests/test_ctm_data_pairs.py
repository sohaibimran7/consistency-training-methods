from __future__ import annotations

from collections import UserDict

import pytest

from ctm_data.pairs import PairRowError, canonical_pair_row, canonical_pair_rows, make_pair_row


def _messages(text):
    return [UserDict(role="user", content=text, provider_field={"kept": True})]


def test_canonical_pair_row_copies_messages_and_preserves_extra_metadata():
    metadata = {"nested": [1, 2]}
    row = {
        "reference_messages": _messages("reference"),
        "variant_messages": _messages("variant"),
        "source_id": "example-1",
        "metadata": metadata,
        "custom_score": 0.25,
    }

    result = canonical_pair_row(row)

    assert result == {
        "reference_messages": [{"role": "user", "content": "reference", "provider_field": {"kept": True}}],
        "variant_messages": [{"role": "user", "content": "variant", "provider_field": {"kept": True}}],
        "source_id": "example-1",
        "metadata": metadata,
        "custom_score": 0.25,
    }
    assert type(result["reference_messages"][0]) is dict
    assert result["metadata"] is metadata
    assert row["reference_messages"][0].__class__ is UserDict


@pytest.mark.parametrize("field", ["reference_messages", "variant_messages"])
def test_canonical_pair_row_requires_both_nonempty_message_sequences(field):
    row = {
        "reference_messages": [{"role": "user", "content": "reference"}],
        "variant_messages": [{"role": "user", "content": "variant"}],
    }
    row[field] = []

    with pytest.raises(PairRowError, match=rf"{field!r} must not be empty"):
        canonical_pair_row(row)

    del row[field]
    with pytest.raises(PairRowError, match=rf"missing {field!r}"):
        canonical_pair_row(row)


@pytest.mark.parametrize(
    ("messages", "error"),
    [
        ("not-a-sequence-of-mappings", "non-empty sequence"),
        (["not-a-mapping"], "message 1 must be a mapping"),
    ],
)
def test_canonical_pair_row_rejects_non_message_shapes(messages, error):
    with pytest.raises(PairRowError, match=error):
        canonical_pair_row(
            {
                "reference_messages": messages,
                "variant_messages": [{"role": "user", "content": "variant"}],
            }
        )


def test_canonical_pair_rows_materializes_iterables_and_reports_row_number():
    valid = {
        "reference_messages": [{"role": "user", "content": "reference"}],
        "variant_messages": [{"role": "user", "content": "variant"}],
        "index": 1,
    }

    with pytest.raises(PairRowError, match="pair row 2"):
        canonical_pair_rows(row for row in (valid, {"reference_messages": []}))


def test_make_pair_row_flattens_metadata_without_allowing_field_override():
    row = make_pair_row(
        reference_messages=[{"role": "user", "content": "reference"}],
        variant_messages=[{"role": "user", "content": "variant"}],
        metadata={"source_id": "example-1", "partition": "train"},
    )

    assert row == {
        "source_id": "example-1",
        "partition": "train",
        "reference_messages": [{"role": "user", "content": "reference"}],
        "variant_messages": [{"role": "user", "content": "variant"}],
    }
    with pytest.raises(PairRowError, match="cannot override"):
        make_pair_row(
            reference_messages=[{"role": "user", "content": "reference"}],
            variant_messages=[{"role": "user", "content": "variant"}],
            metadata={"reference_messages": []},
        )
