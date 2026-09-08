"""Load native ``mcq_bias`` rows selected by the experiment config."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from ctm.artifacts import plain_file_identity

_FROZEN_ROW_FIELDS = frozenset(
    {
        "question",
        "question_id",
        "source_dataset",
        "prompt_style",
        "unbiased_messages",
        "biased_messages",
        "bias_type",
        "ground_truth",
        "biased_option",
        "biasing_text",
    }
)
_PROMPT_STYLES = frozenset({"none", "encourage_cot"})


def _validate_frozen_row(row: object, *, path: Path, line_number: int) -> dict:
    """Validate and return one native ``mcq_bias`` row unchanged."""

    location = f"{path}:{line_number}"
    if not isinstance(row, dict):
        raise ValueError(f"{location}: frozen row must be a JSON object")

    missing = sorted(_FROZEN_ROW_FIELDS - row.keys())
    if missing:
        raise ValueError(f"{location}: missing mcq_bias frozen field(s): {', '.join(missing)}")

    for field in (
        "question",
        "question_id",
        "source_dataset",
        "prompt_style",
        "bias_type",
        "ground_truth",
        "biased_option",
        "biasing_text",
    ):
        if not isinstance(row[field], str):
            raise ValueError(f"{location}: {field} must be a string")
    if row["bias_type"] == "are_you_sure":
        raise NotImplementedError(
            f"{location}: are_you_sure requires staged multi-turn generation; "
            "CTM currently supports single-generation training prompts only"
        )
    biased_option = row["biased_option"]
    if not biased_option.strip():
        raise ValueError(f"{location}: biased_option must be a non-empty answer label")
    if biased_option.startswith("NOT"):
        negated = biased_option.removeprefix("NOT ") if biased_option.startswith("NOT ") else ""
        if not negated.strip() or biased_option != f"NOT {negated.strip()}":
            raise ValueError(f"{location}: malformed negated biased_option {biased_option!r}; expected 'NOT <answer>'")
    if row["prompt_style"] not in _PROMPT_STYLES:
        raise ValueError(
            f"{location}: unknown prompt_style {row['prompt_style']!r}; expected one of {sorted(_PROMPT_STYLES)}"
        )

    for field in ("unbiased_messages", "biased_messages"):
        messages = row[field]
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{location}: {field} must be a non-empty message list")
        for index, message in enumerate(messages):
            if (
                not isinstance(message, dict)
                or not isinstance(message.get("role"), str)
                or not message["role"].strip()
                or not isinstance(message.get("content"), str)
                or not message["content"].strip()
            ):
                raise ValueError(f"{location}: {field}[{index}] must contain non-empty string role/content fields")

    return dict(row)


def _validated_row_offset(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("row_offset must be a non-negative integer")
    return value


def _validated_row_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("row_count must be a positive integer")
    return value


def file_identity(
    path: str | Path,
    *,
    row_offset: int | None = None,
    row_count: int | None = None,
) -> dict:
    """Record an explicit frozen file and, when applicable, its exact row slice.

    ``row_offset`` is zero-based while the inclusive range is recorded in the
    human-facing one-based convention used by frozen selection manifests.
    Passing either selection value requires both so provenance cannot describe
    an ambiguous prefix or open-ended range.
    """

    if (row_offset is None) != (row_count is None):
        raise ValueError("row_offset and row_count must be supplied together for file provenance")

    identity = plain_file_identity(path)
    provenance: dict[str, object] = {"source": "explicit_mcq_bias_file"}
    if row_offset is not None and row_count is not None:
        offset = _validated_row_offset(row_offset)
        count = _validated_row_count(row_count)
        provenance["selection"] = {
            "method": "exact_contiguous_source_rows_without_reserialization",
            "row_offset": offset,
            "row_count": count,
            "source_rows_1_based_inclusive": [offset + 1, offset + count],
        }
    identity["provenance"] = provenance
    return identity


def load_paths(
    paths: Sequence[str | Path],
    *,
    n_datapoints: int | None = None,
    path_limits: Mapping[str, int] | None = None,
    row_offset: int | None = None,
) -> list[dict]:
    """Load the exact native files selected by the experiment.

    ``path_limits`` gives each file an explicit cap. Otherwise
    ``n_datapoints`` is divided as evenly as possible across the files.

    ``row_offset`` enables one exact contiguous slice of one frozen source.
    It is zero-based and deliberately cannot be combined with multi-file
    loading or ``path_limits``: either would make the selected source range
    ambiguous. Rows are read directly from the input JSONL; no derived JSONL
    is created or reserialized.
    """

    selected = [Path(path) for path in paths]
    if not selected:
        raise ValueError("sycophancy training requires at least one data_path")

    if row_offset is not None:
        offset = _validated_row_offset(row_offset)
        if len(selected) != 1:
            raise ValueError("row_offset requires exactly one data_path")
        if path_limits is not None:
            raise ValueError("row_offset cannot be combined with path_limits")
    else:
        offset = 0

    if path_limits is not None:
        selected_names = {str(path) for path in selected}
        unknown = sorted(set(path_limits) - selected_names)
        if unknown:
            raise ValueError(f"path_limits names paths not present in data_paths: {unknown}")
        limits = []
        for path in selected:
            limit = path_limits.get(str(path))
            if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
                raise ValueError(f"path_limits[{str(path)!r}] must be a positive integer")
            limits.append(limit)
    else:
        if not isinstance(n_datapoints, int) or isinstance(n_datapoints, bool) or n_datapoints < 1:
            raise ValueError("n_datapoints must be a positive integer when path_limits is omitted")
        quotient, remainder = divmod(n_datapoints, len(selected))
        if quotient == 0:
            raise ValueError("n_datapoints must be at least the number of data_paths")
        limits = [quotient + (index < remainder) for index in range(len(selected))]

    datapoints: list[dict] = []
    for path, limit in zip(selected, limits):
        skipped = 0
        loaded = 0
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
                validated = _validate_frozen_row(row, path=path, line_number=line_number)
                if skipped < offset:
                    skipped += 1
                    continue
                datapoints.append(validated)
                loaded += 1
                if loaded >= limit:
                    break
        if skipped < offset:
            raise ValueError(
                f"{path} contains only {skipped} rows; row_offset {offset} requires at least {offset + 1} rows"
            )
        if loaded < limit:
            if row_offset is None:
                raise ValueError(f"{path} contains only {loaded}/{limit} requested rows")
            raise ValueError(
                f"{path} contains only {loaded}/{limit} requested rows after row_offset {offset}; "
                f"requested source rows {offset + 1}--{offset + limit}"
            )
    return datapoints


def make_perturbation_fns():
    """Return the unbiased and biased prompts exactly as frozen by ``mcq_bias``."""

    def unbiased(datapoint: dict) -> dict:
        return {"messages": datapoint["unbiased_messages"]}

    def biased(datapoint: dict) -> dict:
        return {"messages": datapoint["biased_messages"]}

    return unbiased, biased
