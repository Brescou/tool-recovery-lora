#!/usr/bin/env python3
"""Generate curated train/val/eval JSONL for meeting_prep tool recovery."""

from __future__ import annotations

import argparse
from pathlib import Path

from tool_recovery_lora.data.generate import (
    generate_examples,
    split_train_val_eval,
    write_jsonl,
)
from tool_recovery_lora.data.loader import load_jsonl, validate_no_id_overlap

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """CLI entry for ``make dataset``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-total", type=int, default=1100)
    parser.add_argument("--n-val", type=int, default=120)
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--correct-ratio", type=float, default=0.6)
    args = parser.parse_args()

    examples = generate_examples(
        args.n_total,
        seed=args.seed,
        correct_ratio=args.correct_ratio,
    )
    train, val, eval_set = split_train_val_eval(
        examples,
        n_eval=args.n_eval,
        n_val=args.n_val,
        seed=args.seed,
    )
    validate_no_id_overlap(train, val)
    validate_no_id_overlap(train, eval_set)
    validate_no_id_overlap(val, eval_set)

    train_path = REPO_ROOT / "data" / "curated" / "train.jsonl"
    val_path = REPO_ROOT / "data" / "curated" / "val.jsonl"
    eval_path = REPO_ROOT / "data" / "eval" / "smoke.jsonl"

    write_jsonl(train_path, train)
    write_jsonl(val_path, val)
    write_jsonl(eval_path, eval_set)

    # Round-trip load to catch schema issues early.
    loaded_train = load_jsonl(train_path)
    loaded_val = load_jsonl(val_path)
    loaded_eval = load_jsonl(eval_path)

    n_correct = sum(1 for ex in loaded_train if ex.kind == "correct")
    n_recovery = sum(1 for ex in loaded_train if ex.kind == "recovery")
    print(f"wrote {train_path} ({len(loaded_train)} train)")
    print(f"  correct={n_correct} recovery={n_recovery}")
    print(f"wrote {val_path} ({len(loaded_val)} val)")
    print(f"wrote {eval_path} ({len(loaded_eval)} smoke eval)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
