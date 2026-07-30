"""Materialize and combine native ``mcq_bias`` prompt pairs for training."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ctm.artifacts import artifact_manifest_path, write_atomic_bytes, write_verified_jsonl_artifact
from ctm.pairs import canonical_pair_row
from ctm_data.adapters.mcq_bias.data import file_identity
from ctm_data.adapters.mcq_bias.dataset_specs import parse_dataset_cli_tokens

PAIR_ARTIFACT_SCHEMA = "ctm.prompt_pairs"
PAIR_ARTIFACT_SCHEMA_VERSION = 1


def interleave_rows(per_file_rows: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Round-robin files so a global prefix remains balanced across datasets."""

    result: list[dict[str, Any]] = []
    iterators = [iter(rows) for rows in per_file_rows]
    while iterators:
        for iterator in list(iterators):
            try:
                result.append(next(iterator))
            except StopIteration:
                iterators.remove(iterator)
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number}: row must be a JSON object")
            rows.append(row)
    return rows


def native_rows_to_prompt_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Adapt native frozen MCQ rows to CTM's reusable flat pair schema."""

    pairs: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        labels = row.get("option_labels")
        if not isinstance(labels, list) or len(labels) < 2 or not all(isinstance(label, str) for label in labels):
            raise ValueError(
                f"native mcq_bias row {index} has no option_labels; rebuild with a prompt family "
                "that records its parser contract"
            )
        question_id = row.get("question_id")
        source = row.get("source_dataset")
        if not isinstance(question_id, str) or not question_id or not isinstance(source, str) or not source:
            raise ValueError(f"native mcq_bias row {index} has invalid question/source identity")
        pair = canonical_pair_row(
            {
                "pair_id": f"mcq-bias:{row['bias_type']}:{question_id}",
                "source_id": question_id,
                "source": source,
                "reference_messages": row["unbiased_messages"],
                "variant_messages": row["biased_messages"],
                "alignment_text": row["question"],
                "metadata": {
                    "bias_type": row["bias_type"],
                    "correct_label": row["ground_truth"],
                    "biased_option": row["biased_option"],
                    "valid_labels": labels,
                    "prompt_family": row.get("prompt_family", "chua"),
                    "prompt_style": row["prompt_style"],
                    "wrong_option_seed": row.get("wrong_option_seed"),
                    "biasing_text": row.get("biasing_text", ""),
                },
            }
        )
        pairs.append(pair)
    return pairs


def main(argv: list[str] | None = None) -> None:
    from mcq_bias.pipeline.records import PROMPT_FAMILIES, PROMPT_STYLES
    from mcq_bias.tasks import BIAS_TYPES, mcq_bias

    parser = argparse.ArgumentParser(
        description="Materialize native mcq_bias files and publish one balanced training JSONL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bias-type", required=True, choices=BIAS_TYPES)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--prompt-style", default="none", choices=list(PROMPT_STYLES))
    parser.add_argument("--prompt-family", default="chua", choices=list(PROMPT_FAMILIES))
    parser.add_argument("--wrong-option-seed")
    parser.add_argument("--n-questions", type=int, default=250)
    parser.add_argument("--min-n-questions", type=int)
    parser.add_argument("--seed", default="42")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--argument-model")
    parser.add_argument("--generate-missing-arguments", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument(
        "--output-format",
        choices=("native", "prompt_pairs"),
        default="native",
        help="Native mcq_bias rows or the shared ctm.prompt_pairs training schema",
    )
    parser.add_argument("-y", "--yes", action="store_true")
    args = parser.parse_args(argv)
    try:
        dataset_specs = parse_dataset_cli_tokens(args.datasets)
    except ValueError as exc:
        parser.error(str(exc))

    if args.n_questions < 1:
        parser.error("--n-questions must be >= 1")
    if args.min_n_questions is not None and not 1 <= args.min_n_questions <= args.n_questions:
        parser.error("--min-n-questions must be between 1 and --n-questions")
    if args.generate_missing_arguments and args.bias_type != "wrong_argument":
        parser.error("--generate-missing-arguments only applies to wrong_argument")
    if args.output.resolve() == args.manifest_output.resolve():
        parser.error("--output and --manifest-output must be different paths")
    if (
        args.output_format == "prompt_pairs"
        and args.manifest_output.resolve() != artifact_manifest_path(args.output).resolve()
    ):
        parser.error("prompt_pairs output requires --manifest-output OUTPUT.manifest.json")
    existing = [str(path) for path in (args.output, args.manifest_output) if path.exists()]
    if existing:
        parser.error(f"refusing to overwrite existing output(s): {existing}")

    print("\nmcq_bias training-data materialization:")
    print(f"  bias_type={args.bias_type}")
    print(f"  datasets={[spec.as_dict() for spec in dataset_specs]}")
    print(
        f"  prompt_style={args.prompt_style}, n_questions={args.n_questions}, "
        f"min_n_questions={args.min_n_questions}, seed={args.seed}"
    )
    print(f"  dataset_dir={args.dataset_dir}")
    print(f"  output={args.output}")
    print(f"  manifest={args.manifest_output}")
    if not args.yes and input("\nProceed? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return

    argument_model = args.argument_model
    if args.bias_type == "wrong_argument" and argument_model is None:
        from mcq_bias.pipeline.wrong_arguments import DEFAULT_ARGUMENT_MODEL

        argument_model = DEFAULT_ARGUMENT_MODEL

    source_paths = []
    per_dataset_rows = []
    for spec in dataset_specs:
        # Constructing the public task materializes its frozen rows if absent;
        # no model evaluation is run here.
        task = mcq_bias(
            bias_type=args.bias_type,
            **spec.as_dict(include_defaults=False),
            prompt_style=args.prompt_style,
            prompt_family=args.prompt_family,
            wrong_option_seed=args.wrong_option_seed,
            n_questions=args.n_questions,
            min_n_questions=args.min_n_questions,
            seed=args.seed,
            argument_model=argument_model,
            generate_missing_arguments=args.generate_missing_arguments,
            dataset_dir=str(args.dataset_dir),
        )
        source_path = Path(task.metadata["dataset_file"])
        rows = _read_jsonl(source_path)
        floor = args.n_questions if args.min_n_questions is None else args.min_n_questions
        if not floor <= len(rows) <= args.n_questions:
            raise ValueError(
                f"{source_path} contains {len(rows)} rows; expected between {floor} and {args.n_questions}"
            )
        source_paths.append(source_path)
        per_dataset_rows.append(rows)

    native_rows = interleave_rows(per_dataset_rows)
    rows = native_rows_to_prompt_pairs(native_rows) if args.output_format == "prompt_pairs" else native_rows
    payload = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for row in rows)
    selection = {
        "bias_type": args.bias_type,
        "datasets": [spec.dataset for spec in dataset_specs],
        "dataset_specs": [spec.as_dict() for spec in dataset_specs],
        "prompt_style": args.prompt_style,
        "prompt_family": args.prompt_family,
        "wrong_option_seed": args.wrong_option_seed,
        "n_questions": args.n_questions,
        "min_n_questions": args.min_n_questions,
        "seed": args.seed,
        "argument_model": argument_model,
    }
    if args.output_format == "prompt_pairs":
        manifest = write_verified_jsonl_artifact(
            args.output,
            rows,
            artifact_schema=PAIR_ARTIFACT_SCHEMA,
            schema_version=PAIR_ARTIFACT_SCHEMA_VERSION,
            provenance={
                "kind": "mcq_bias_prompt_pairs",
                "selection": selection,
                "sources": [file_identity(path) for path in source_paths],
                "merge": "round_robin",
            },
            row_validator=canonical_pair_row,
            nonempty=True,
        )
        print(f"Materialized {len(rows)} balanced prompt pairs to {args.output}")
        return

    manifest = {
        "schema_version": 1,
        "kind": "mcq_bias_training_selection",
        "written_at": datetime.now(UTC).isoformat(),
        "selection": selection,
        "sources": [file_identity(path) for path in source_paths],
        "merge": "round_robin",
        "row_count": len(rows),
        "output": {
            "path": str(args.output.resolve()),
            "content_sha256": hashlib.sha256(payload).hexdigest(),
        },
    }
    write_atomic_bytes(args.output, payload)
    write_atomic_bytes(
        args.manifest_output,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(f"Materialized {len(rows)} balanced training rows to {args.output}")


if __name__ == "__main__":
    main()
