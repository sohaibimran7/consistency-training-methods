from __future__ import annotations

from collections import UserDict
from types import SimpleNamespace

import pytest

import ctm_data.sources.huggingface as huggingface_module
from ctm_data.sources import (
    HuggingFaceSource,
    LoadedRows,
    LocalSource,
    RowSource,
    SourceIdentity,
    SourceRowError,
    load_huggingface_rows,
    load_local_rows,
)


def test_source_identity_is_plain_serializable_metadata():
    identity = SourceIdentity(
        loader="example",
        location="owner/data",
        config="default",
        split="train",
        revision="abc123",
    )

    assert identity.as_dict() == {
        "loader": "example",
        "location": "owner/data",
        "config": "default",
        "split": "train",
        "revision": "abc123",
    }
    with pytest.raises(ValueError, match="revision"):
        SourceIdentity(loader="example", location="owner/data", revision=" ")


def test_local_source_satisfies_generic_row_source_protocol(tmp_path):
    path = tmp_path / "rows.data"
    path.write_text('[{"id": 1}]', encoding="utf-8")
    source = LocalSource(path, format="json")

    assert isinstance(source, RowSource)
    result = source.load()
    assert isinstance(result, LoadedRows)
    assert result.rows == [{"id": 1}]
    assert result.source.as_dict() == {
        "loader": "local",
        "location": str(path.resolve()),
        "format": "json",
    }


def test_json_requires_an_array_of_mapping_rows(tmp_path):
    path = tmp_path / "rows.txt"
    path.write_text('{"id": 1}', encoding="utf-8")
    with pytest.raises(SourceRowError, match="top-level array"):
        load_local_rows(path, format="json")

    path.write_text('[{"id": 1}, 2]', encoding="utf-8")
    with pytest.raises(SourceRowError, match="row 2 must be a mapping"):
        load_local_rows(path, format="json")


def test_local_format_is_explicit_not_inferred_from_suffix(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_text('[{"id": 1}]', encoding="utf-8")

    assert load_local_rows(path, format="json").rows == [{"id": 1}]
    with pytest.raises(ValueError, match="unsupported local format"):
        LocalSource(path, format="yaml")  # type: ignore[arg-type]


def test_jsonl_uses_physical_newlines_not_unicode_line_separators(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"text": "left\u2028middle\u2029right", "metadata": {"kept": true}}\n' '{"text": "second"}\n',
        encoding="utf-8",
    )

    result = load_local_rows(path, format="jsonl")

    assert result.rows == [
        {"text": "left\u2028middle\u2029right", "metadata": {"kept": True}},
        {"text": "second"},
    ]


def test_jsonl_reports_the_physical_line_for_invalid_json(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"ok": true}\n\nnot-json\n', encoding="utf-8")

    with pytest.raises(SourceRowError, match="line 3 is invalid JSON"):
        load_local_rows(path, format="jsonl")


@pytest.mark.parametrize(
    ("format_name", "delimiter"),
    [("csv", ","), ("tsv", "\t")],
)
def test_delimited_sources_preserve_columns_and_quoted_newlines(tmp_path, format_name, delimiter):
    path = tmp_path / f"rows.{format_name}"
    if format_name == "csv":
        path.write_text('id,text,extra\n1,"first, value",keep\n2,"two\nlines",also\n', encoding="utf-8")
    else:
        path.write_text(f"id{delimiter}text{delimiter}extra\n1{delimiter}first{delimiter}keep\n", encoding="utf-8")

    result = load_local_rows(path, format=format_name)  # type: ignore[arg-type]

    expected = (
        [
            {"id": "1", "text": "first, value", "extra": "keep"},
            {"id": "2", "text": "two\nlines", "extra": "also"},
        ]
        if format_name == "csv"
        else [{"id": "1", "text": "first", "extra": "keep"}]
    )
    assert result.rows == expected


@pytest.mark.parametrize(
    ("contents", "error"),
    [
        ("id,id\n1,2\n", "duplicate header names"),
        ("id,\n1,2\n", "header names must be non-empty"),
        ("id,text\n1\n", "1 values for 2 columns"),
        ("id,text\n1,two,extra\n", "3 values for 2 columns"),
    ],
)
def test_csv_rejects_ambiguous_or_ragged_rows(tmp_path, contents, error):
    path = tmp_path / "rows.csv"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(SourceRowError, match=error):
        load_local_rows(path, format="csv")


def test_huggingface_import_is_lazy_and_selection_is_forwarded_exactly(monkeypatch):
    assert "datasets" not in vars(huggingface_module)
    calls = []

    class CustomMapping(UserDict):
        pass

    selected = [
        CustomMapping(id=1, keep=False, metadata={"source": "upstream"}),
        CustomMapping(id=2, keep=True, extra="untouched"),
    ]

    def fake_load_dataset(dataset, config, *, split, revision):
        calls.append((dataset, config, split, revision))
        return selected

    def fake_import(name):
        assert name == "datasets"
        return SimpleNamespace(load_dataset=fake_load_dataset)

    monkeypatch.setattr(huggingface_module, "import_module", fake_import)

    result = load_huggingface_rows(
        "owner/data",
        config="configuration",
        split="train[:10]",
        revision="full-revision",
    )

    assert calls == [("owner/data", "configuration", "train[:10]", "full-revision")]
    assert result.rows == [
        {"id": 1, "keep": False, "metadata": {"source": "upstream"}},
        {"id": 2, "keep": True, "extra": "untouched"},
    ]
    assert all(type(row) is dict for row in result.rows)
    assert result.source.as_dict() == {
        "loader": "huggingface",
        "location": "owner/data",
        "config": "configuration",
        "split": "train[:10]",
        "revision": "full-revision",
    }


@pytest.mark.parametrize("field", ["dataset", "config", "split", "revision"])
def test_huggingface_selection_fields_are_all_required_and_nonempty(field):
    values = {
        "dataset": "owner/data",
        "config": "default",
        "split": "train",
        "revision": "abc123",
    }
    values[field] = ""

    with pytest.raises(ValueError, match=field):
        HuggingFaceSource(**values)


def test_huggingface_rejects_non_mapping_rows_without_filtering_them(monkeypatch):
    fake_module = SimpleNamespace(load_dataset=lambda *args, **kwargs: [{"id": 1}, None, {"id": 3}])
    monkeypatch.setattr(huggingface_module, "import_module", lambda name: fake_module)

    with pytest.raises(SourceRowError, match="row 2 must be a mapping"):
        load_huggingface_rows("owner/data", config="default", split="train", revision="abc123")
