"""Tests for synthetic dataset generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from tool_recovery_lora.data.generate import (
    generate_examples,
    split_train_val_eval,
    write_jsonl,
)
from tool_recovery_lora.data.loader import load_jsonl, validate_no_id_overlap


def test_generate_examples_ratio_and_kinds() -> None:
    examples = generate_examples(100, seed=7, correct_ratio=0.6)
    assert len(examples) == 100
    n_correct = sum(1 for ex in examples if ex.kind == "correct")
    assert n_correct == 60
    recovery = [ex for ex in examples if ex.kind == "recovery"]
    assert len(recovery) == 40
    error_types = {ex.meta["error_type"] for ex in recovery if ex.meta}
    assert error_types == {"missing_args", "wrong_tool", "invalid_json"}


def test_generate_is_deterministic() -> None:
    a = generate_examples(20, seed=99)
    b = generate_examples(20, seed=99)
    assert [ex.model_dump() for ex in a] == [ex.model_dump() for ex in b]


def test_split_disjoint_ids(tmp_path: Path) -> None:
    examples = generate_examples(80, seed=1)
    train, val, eval_set = split_train_val_eval(
        examples, n_eval=10, n_val=15, seed=1
    )
    validate_no_id_overlap(train, val)
    validate_no_id_overlap(train, eval_set)
    validate_no_id_overlap(val, eval_set)
    assert len(train) + len(val) + len(eval_set) == 80

    path = tmp_path / "train.jsonl"
    write_jsonl(path, train)
    loaded = load_jsonl(path)
    assert len(loaded) == len(train)


def test_split_rejects_oversized_holds() -> None:
    examples = generate_examples(10, seed=2)
    with pytest.raises(ValueError, match="n_eval \\+ n_val"):
        split_train_val_eval(examples, n_eval=6, n_val=5, seed=2)
