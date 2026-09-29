"""Freeze a portable, allowlisted source/data bundle without touching other runs."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from . import plan


def package(repository: Path, plan_path: Path, output: Path, *, alignment_audit: Path | None = None) -> dict:
    document = plan.verify(repository, plan_path)
    files = set(document["sources"])
    files.update((str(plan.POOL), str(plan.MANIFEST), "pyproject.toml", "requirements.txt",
                  "experiments/methods_two_bias_convergence/README.md"))
    for optional in ("uv.lock", "ctm_data/adapters/mcq_bias/plot_registry.toml"):
        if (repository / optional).is_file():
            files.add(optional)
    identities = {relative: plan.sha256(repository / relative) for relative in sorted(files)}
    if alignment_audit is not None:
        from .train import require_alignment_audit
        require_alignment_audit(alignment_audit)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to replace a source bundle: {output}")
    with tarfile.open(output, "x:gz") as archive:
        for relative in sorted(files):
            path = repository / relative
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"bundle input is not a regular file: {relative}")
            archive.add(path, arcname=relative, recursive=False)
        archive.add(plan_path, arcname="campaign-plan.json", recursive=False)
        if alignment_audit is not None:
            archive.add(alignment_audit, arcname="alignment-audit.json", recursive=False)
    receipt = {"bundle": str(output), "sha256": plan.sha256(output), "bytes": output.stat().st_size,
               "plan_sha256": plan.sha256(plan_path), "files": identities,
               "alignment_audit_sha256": plan.sha256(alignment_audit) if alignment_audit is not None else None,
               "deployment": "extract_into_fresh_isolated_directory_only", "submitted": False}
    plan.immutable_json(output.with_suffix(output.suffix + ".manifest.json"), receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alignment-audit", type=Path)
    args = parser.parse_args()
    receipt = package(args.repository.resolve(), args.plan.resolve(), args.output.resolve(),
                      alignment_audit=args.alignment_audit.resolve() if args.alignment_audit else None)
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}, sort_keys=True))


if __name__ == "__main__":
    main()
