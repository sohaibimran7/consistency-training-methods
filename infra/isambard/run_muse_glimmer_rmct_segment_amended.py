#!/usr/bin/env python3
"""Run a frozen Muse RMCT segment under the attested boundary amendment."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infra.isambard import muse_glimmer_rmct_segment_amendment as amendment  # noqa: E402
from infra.isambard.run_muse_glimmer_rmct_segment import main as frozen_main  # noqa: E402


def main() -> int:
    # The frozen runner requires --repo-root and performs the authoritative
    # argument parsing itself.  Resolve only that value to validate the
    # write-once amendment before any guard, model load, or generation.
    try:
        index = sys.argv.index("--repo-root")
        root = Path(sys.argv[index + 1]).resolve()
    except (ValueError, IndexError) as exc:
        raise SystemExit("error: amended Muse runner requires --repo-root") from exc
    amendment.validate_attestation(root)
    amendment.install()
    return frozen_main()


if __name__ == "__main__":
    raise SystemExit(main())
