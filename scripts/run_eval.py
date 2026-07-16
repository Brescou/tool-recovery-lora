#!/usr/bin/env python3
"""Run objective smoke evaluation on curated fixtures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tool_recovery_lora.eval.runner import run_smoke_eval

DEFAULT_SMOKE = Path("data/eval/smoke.jsonl")


def main() -> int:
    """Entry point for ``make eval``."""
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SMOKE
    results = run_smoke_eval(path)
    print(json.dumps(results, indent=2))
    if results["args_exact"] < 1.0:
        print("smoke eval: FAIL (args_exact < 1.0)", file=sys.stderr)
        return 1
    print("smoke eval: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
