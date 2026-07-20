"""Sample frozen-base completions once and build matched BCT/control files."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from ctm.backends.cli import add_backend_args, build_backend, describe_backend  # noqa: E402
from ctm.backends.renderers import get_renderer_and_tokenizer  # noqa: E402
from ctm.core.config import LoRAConfig  # noqa: E402
from ctm.training.bct_targets import (  # noqa: E402
    generate_bct_rows,
    prepare_paired_prompts,
    write_bct_target_artifacts,
)


def _read_rows(paths: list[Path], limit: int | None) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                if limit is not None and len(rows) >= limit:
                    return rows
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: row must be a JSON object")
                rows.append(value)
    if limit is not None and len(rows) < limit:
        raise ValueError(f"input files contain only {len(rows)}/{limit} requested rows")
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one frozen-base completion per paired prompt and emit matched BCT/control datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", nargs="+", type=Path, required=True, help="Paired-prompt JSONL file(s), in order")
    parser.add_argument("--limit", type=int, help="Exact total row count to use across the input files")
    parser.add_argument("--source-messages-field", default="reference_messages")
    parser.add_argument("--main-messages-field", default="variant_messages")
    parser.add_argument("--control-messages-field", default="reference_messages")
    parser.add_argument("--main-output", type=Path, required=True)
    parser.add_argument("--control-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--lora-rank", type=int, default=8, help="Temporary backend LoRA rank; targets use the frozen base")
    add_backend_args(parser)
    parser.add_argument("-y", "--yes", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be >= 1")
    if args.temperature < 0:
        parser.error("--temperature must be >= 0")
    if args.max_concurrency < 1:
        parser.error("--max-concurrency must be >= 1")
    if args.lora_rank < 1:
        parser.error("--lora-rank must be >= 1")
    if args.local_full_finetune:
        parser.error("BCT target generation requires frozen-base access; do not pass --local-full-finetune")
    missing = [str(path) for path in args.data if not path.is_file()]
    if missing:
        parser.error(f"input file(s) not found: {missing}")
    existing = [str(path) for path in (args.main_output, args.control_output, args.manifest_output) if path.exists()]
    if existing:
        parser.error(f"refusing to overwrite existing output(s): {existing}")

    try:
        rows = _read_rows(args.data, args.limit)
        prompts = prepare_paired_prompts(
            rows,
            source_messages_field=args.source_messages_field,
            main_messages_field=args.main_messages_field,
            control_messages_field=args.control_messages_field,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print("\nBCT target preparation:")
    print(f"  model={args.model}")
    print(f"  backend={describe_backend(args)}")
    print(f"  rows={len(prompts)}, data={[str(path) for path in args.data]}")
    print(f"  source={args.source_messages_field}")
    print(f"  main={args.main_messages_field} -> {args.main_output}")
    print(f"  control={args.control_messages_field} -> {args.control_output}")
    print(
        f"  generation=max_tokens={args.max_tokens}, temperature={args.temperature}, "
        f"max_concurrency={args.max_concurrency}"
    )
    print(f"  manifest={args.manifest_output}")
    if not args.yes and input("\nProceed? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return

    backend = build_backend(args)
    backend.setup(model=args.model, lora=LoRAConfig(rank=args.lora_rank))
    renderer, tokenizer = get_renderer_and_tokenizer(args.model)
    main_rows, control_rows = asyncio.run(
        generate_bct_rows(
            prompts,
            sampler=backend.base_sampler(),
            renderer=renderer,
            tokenizer=tokenizer,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_concurrency=args.max_concurrency,
        )
    )
    manifest = write_bct_target_artifacts(
        main_rows=main_rows,
        control_rows=control_rows,
        main_output=args.main_output,
        control_output=args.control_output,
        manifest_output=args.manifest_output,
        source_files=args.data,
        model=args.model,
        backend_name=type(backend).__name__,
        source_messages_field=args.source_messages_field,
        main_messages_field=args.main_messages_field,
        control_messages_field=args.control_messages_field,
        generation_config={
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "max_concurrency": args.max_concurrency,
        },
    )
    print(f"\nPrepared {manifest['row_count']} shared targets.")


if __name__ == "__main__":
    main()
