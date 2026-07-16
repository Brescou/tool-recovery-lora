#!/usr/bin/env python3
"""Training entrypoint stub (real Unsloth loop lands in Phase 2)."""

from __future__ import annotations

from tool_recovery_lora.train.config import TrainConfig


def main() -> int:
    """Print config and exit unless the train extra is installed later."""
    config = TrainConfig()
    print("TrainConfig:")
    print(f"  base_model      = {config.base_model}")
    print(f"  lora_rank       = {config.lora_rank}")
    print(f"  lora_alpha      = {config.lora_alpha}")
    print(f"  max_seq_length  = {config.max_seq_length}")
    print(f"  output_dir      = {config.output_dir}")
    print(f"  train_jsonl     = {config.train_jsonl}")
    print(f"  val_jsonl       = {config.val_jsonl}")
    print()
    print(
        "Phase 0 stub: Unsloth training is not wired yet. "
        "Curate data/curated/*.jsonl, then implement Phase 2."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
