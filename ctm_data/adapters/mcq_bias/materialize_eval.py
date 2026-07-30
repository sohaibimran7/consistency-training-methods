"""Materialize a native mcq-bias evaluation suite without evaluating a model."""

from __future__ import annotations

import argparse
from pathlib import Path

from ctm_data.adapters.mcq_bias.dataset_specs import parse_dataset_cli_tokens


def main(argv: list[str] | None = None) -> None:
    from mcq_bias.pipeline.records import PROMPT_FAMILIES, PROMPT_STYLES
    from mcq_bias.tasks import BIAS_TYPES, suite_tasks

    parser = argparse.ArgumentParser(
        description="Freeze all datasets needed by an mcq-bias evaluation suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bias-types", nargs="+", required=True, choices=BIAS_TYPES)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--prompt-style", default="none", choices=list(PROMPT_STYLES))
    parser.add_argument("--prompt-family", default="chua", choices=list(PROMPT_FAMILIES))
    parser.add_argument("--wrong-option-seed")
    parser.add_argument("--n-questions", type=int, default=250)
    parser.add_argument("--min-n-questions", type=int)
    parser.add_argument("--seed", default="42")
    parser.add_argument("--argument-model")
    parser.add_argument("--generate-missing-arguments", action="store_true")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("-y", "--yes", action="store_true")
    args = parser.parse_args(argv)
    try:
        dataset_specs = parse_dataset_cli_tokens(args.datasets)
    except ValueError as exc:
        parser.error(str(exc))

    if args.n_questions < 1:
        parser.error("--n-questions must be at least 1")
    if args.min_n_questions is not None and not 1 <= args.min_n_questions <= args.n_questions:
        parser.error("--min-n-questions must be between 1 and --n-questions")

    print("\nmcq_bias evaluation-data materialization:")
    print(f"  bias_types={args.bias_types}")
    print(f"  datasets={[spec.as_dict() for spec in dataset_specs]}")
    print(f"  prompt_style={args.prompt_style}, n_questions={args.n_questions}, seed={args.seed}")
    print(f"  dataset_dir={args.dataset_dir}")
    if not args.yes and input("\nProceed? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return

    tasks = suite_tasks(
        bias_types=args.bias_types,
        datasets=[spec.as_dict(include_defaults=False) for spec in dataset_specs],
        prompt_style=args.prompt_style,
        prompt_family=args.prompt_family,
        wrong_option_seed=args.wrong_option_seed,
        n_questions=args.n_questions,
        min_n_questions=args.min_n_questions,
        seed=args.seed,
        argument_model=args.argument_model,
        generate_missing_arguments=args.generate_missing_arguments,
        dataset_dir=str(args.dataset_dir),
        include_bias_acknowledged=False,
        skip_unbuildable=False,
    )
    expected = len(dataset_specs) * (len(args.bias_types) + 1)
    if len(tasks) != expected:
        raise RuntimeError(f"materialized {len(tasks)} tasks; expected {expected}")
    print(f"Materialized the frozen inputs for {len(tasks)} evaluation tasks.")


if __name__ == "__main__":
    main()
