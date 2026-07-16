#!/usr/bin/env python3
"""Bake-off: LoRA vs Claude tools-API prompting on the smoke set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tool_recovery_lora.bakeoff.runner import run_bakeoff

DEFAULT_SMOKE = Path("data/eval/smoke.jsonl")
DEFAULT_ADAPTER = Path("adapters/qwen25-3b-recovery")
DEFAULT_OUT = Path("data/eval/bakeoff")


def main() -> int:
    """CLI entry for ``make bakeoff``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_SMOKE)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--skip-lora",
        action="store_true",
        help="Reuse existing live_results.json instead of re-running LoRA",
    )
    parser.add_argument(
        "--lora-results",
        type=Path,
        default=Path("data/eval/live_results.json"),
    )
    parser.add_argument(
        "--claude-model",
        default="claude-sonnet-4-20250514",
    )
    args = parser.parse_args()

    try:
        results = run_bakeoff(
            smoke_path=args.path,
            adapter_dir=args.adapter_dir,
            out_dir=args.out_dir,
            limit=args.limit,
            skip_lora=args.skip_lora,
            lora_results_path=args.lora_results,
            claude_model=args.claude_model,
        )
    except Exception as exc:
        print(f"bakeoff failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({"rows": results["rows"]}, indent=2))
    print(f"wrote {results['results_path']}")
    print(f"wrote {args.out_dir / 'accuracy_vs_cost.svg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
