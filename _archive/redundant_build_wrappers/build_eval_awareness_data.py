#!/usr/bin/env python3
"""CLI wrapper for the EvalAwareBench data adapter."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ctm_data.adapters.eval_awareness.builder import main


if __name__ == "__main__":
    raise SystemExit(main())
