"""Materialize and combine native ``mcq_bias`` prompt pairs for training."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ctm.artifacts import write_atomic_bytes
from ctm_data.adapters.mcq_bias.data import file_identity
from ctm_data.adapters.mcq_bias.dataset_specs import parse_dataset_cli_tokens


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
                raise ValueError(f"{path}:{line_number}: row must be a JSON object")
            rows.append(row)
    return rows


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

    rows = interleave_rows(per_dataset_rows)
    payload = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for row in rows)
    manifest = {
        "schema_version": 1,
        "kind": "mcq_bias_training_selection",
        "written_at": datetime.now(timezone.utc).isoformat(),
        "selection": {
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
        },
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
