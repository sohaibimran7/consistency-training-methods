"""Validate and inspect a checked-in CTM experiment catalogue JSON file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ctm.experiments.catalog import (
    CatalogError,
    find_experiment,
    list_experiments,
    load_catalog,
    render_catalog_markdown,
    render_experiment_markdown,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without reading catalogue data."""

    parser = argparse.ArgumentParser(
        description="Validate and inspect an offline CTM experiment catalogue.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate a catalogue JSON document")
    validate.add_argument("catalog", type=Path, help="catalogue JSON path")

    list_command = commands.add_parser("list", help="list recorded experiment ids, statuses, and questions")
    list_command.add_argument("catalog", type=Path, help="catalogue JSON path")

    show = commands.add_parser("show", help="show one experiment and every recorded exact location")
    show.add_argument("catalog", type=Path, help="catalogue JSON path")
    show.add_argument("experiment_id", help="stable experiment id")

    render = commands.add_parser("render", help="render the whole catalogue as Markdown")
    render.add_argument("catalog", type=Path, help="catalogue JSON path")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the catalogue CLI and return a conventional process status."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        catalog = load_catalog(args.catalog)
        if args.command == "validate":
            print(f"Valid catalogue: {len(catalog.experiments)} experiment(s).")
        elif args.command == "list":
            for experiment in list_experiments(catalog):
                print(f"{experiment.id}\t{experiment.status}\t{experiment.question}")
        elif args.command == "show":
            # Resolve the id before rendering so a missing id is reported by
            # argparse in the same friendly form as a malformed catalogue.
            find_experiment(catalog, args.experiment_id)
            print(render_experiment_markdown(catalog, args.experiment_id), end="")
        elif args.command == "render":
            print(render_catalog_markdown(catalog), end="")
        else:  # pragma: no cover - argparse constrains the subcommand.
            parser.error(f"unknown command: {args.command}")
    except CatalogError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
