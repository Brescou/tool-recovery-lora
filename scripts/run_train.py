#!/usr/bin/env python3
"""Run Unsloth QLoRA training for tool-call recovery."""

from __future__ import annotations

import argparse
import sys

from tool_recovery_lora.train.config import TrainConfig


def main() -> int:
    """CLI entry for ``make train``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Short run (max_steps=20) to validate the pipeline.",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=float, default=None)
    args = parser.parse_args()

    overrides: dict = {}
    if args.smoke:
        overrides["max_steps"] = 20
        overrides["output_dir"] = "adapters/qwen25-3b-recovery-smoke"
        overrides["logging_steps"] = 1
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.max_seq_length is not None:
        overrides["max_seq_length"] = args.max_seq_length
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.epochs is not None:
        overrides["num_train_epochs"] = args.epochs

    config = TrainConfig(**overrides)
    print("TrainConfig:")
    for key, value in config.model_dump().items():
        print(f"  {key}={value}")

    try:
        from tool_recovery_lora.train.runner import run_training
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        output_dir = run_training(config)
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"training failed: {exc}", file=sys.stderr)
        raise

    print(f"saved adapters to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
