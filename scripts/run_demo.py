#!/usr/bin/env python3
"""Run the LangGraph tool-call recovery demo with the fine-tuned LoRA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tool_recovery_lora.demo.graph import list_scenarios, run_recovery_demo
from tool_recovery_lora.demo.scenarios import SCENARIOS

DEFAULT_ADAPTER = Path("adapters/qwen25-3b-recovery")


def _print_banner(title: str) -> None:
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def main() -> int:
    """CLI entry for ``make demo``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        default="missing_args",
        choices=sorted(SCENARIOS),
        help="Recovery scenario to run",
    )
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument(
        "--list",
        action="store_true",
        help="List scenarios and exit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable final state only",
    )
    args = parser.parse_args()

    if args.list:
        for scenario in list_scenarios():
            print(f"{scenario.id}: {scenario.title}")
        return 0

    scenario = SCENARIOS[args.scenario]
    if not args.json:
        _print_banner(f"tool-recovery-lora demo · {scenario.title}")
        print(f"user: {scenario.user_text}")
        print(
            f"injected bad call: {scenario.bad_call.name} "
            f"{scenario.bad_call.arguments}"
        )
        print(f"tool error: {scenario.error_message}")
        print("running LoRA repair via LangGraph…")

    try:
        state = run_recovery_demo(
            args.scenario, adapter_dir=args.adapter_dir
        )
    except Exception as exc:
        print(f"demo failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(state, indent=2, default=str))
    else:
        _print_banner("conversation so far")
        for line in state.get("messages_preview", []):
            print(line)
        _print_banner("LoRA repair")
        print(state.get("generation", "").strip())
        print(f"latency_ms: {state.get('latency_ms', 0):.0f}")
        _print_banner("validation")
        print(f"predicted: {state.get('predicted')}")
        print(f"expected:  {scenario.expected.model_dump()}")
        print(f"scores:    {state.get('scores')}")
        print(f"ok:        {state.get('ok')}")

    return 0 if state.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
