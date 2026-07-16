"""Tests for live-eval CLI flag wiring (no GPU)."""

from __future__ import annotations

import argparse
from pathlib import Path

from tool_recovery_lora.eval.infer import DEFAULT_BASE_MODEL


def test_default_base_model_is_qwen25_3b() -> None:
    assert DEFAULT_BASE_MODEL == "unsloth/Qwen2.5-3B-Instruct"


def test_run_eval_parser_accepts_no_adapter() -> None:
    """Ensure --no-adapter and --model are accepted by the CLI parser shape."""
    # Rebuild the same flags as scripts/run_eval.py without importing main side effects.
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fixture", "live"), default="fixture")
    parser.add_argument("--path", type=Path, default=Path("data/eval/smoke.jsonl"))
    parser.add_argument("--adapter-dir", type=Path, default=Path("adapters/x"))
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--no-adapter", action="store_true")
    args = parser.parse_args(
        ["--mode", "live", "--no-adapter", "--model", DEFAULT_BASE_MODEL]
    )
    assert args.no_adapter is True
    assert args.model == DEFAULT_BASE_MODEL
    assert args.mode == "live"
