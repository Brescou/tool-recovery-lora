#!/usr/bin/env python3
"""Run objective smoke evaluation (fixture self-check or live LoRA)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tool_recovery_lora.eval.runner import run_smoke_eval

DEFAULT_SMOKE = Path("data/eval/smoke.jsonl")
DEFAULT_ADAPTER = Path("adapters/qwen25-3b-recovery")


def main() -> int:
    """Entry point for ``make eval`` / ``make eval-live``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("fixture", "live"),
        default="fixture",
        help="fixture=self-check gold labels; live=generate with LoRA",
    )
    parser.add_argument("--path", type=Path, default=DEFAULT_SMOKE)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write full JSON results (incl. details).",
    )
    parser.add_argument(
        "--min-args-exact",
        type=float,
        default=None,
        help="Fail if args_exact below threshold (fixture 1.0, live 0.0).",
    )
    args = parser.parse_args()

    if args.mode == "fixture":
        results = run_smoke_eval(args.path)
        threshold = 1.0 if args.min_args_exact is None else args.min_args_exact
        printable = results
    else:
        from tool_recovery_lora.eval.live_runner import run_live_eval

        results = run_live_eval(
            args.path,
            adapter_dir=args.adapter_dir,
            limit=args.limit,
        )
        threshold = 0.0 if args.min_args_exact is None else args.min_args_exact
        printable = {key: value for key, value in results.items() if key != "details"}

    print(json.dumps(printable, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")

    if results["args_exact"] < threshold:
        print(
            f"eval FAIL: args_exact={results['args_exact']:.3f} < {threshold}",
            file=sys.stderr,
        )
        return 1
    print(f"eval PASS ({args.mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
