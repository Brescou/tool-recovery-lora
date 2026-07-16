"""Tests for JSONL loading and id-overlap checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from tool_recovery_lora.data.loader import load_jsonl, validate_no_id_overlap
from tool_recovery_lora.data.schema import Message, TraceExample

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_PATH = REPO_ROOT / "data" / "eval" / "smoke.jsonl"


def test_load_smoke_jsonl() -> None:
    examples = load_jsonl(SMOKE_PATH)
    assert len(examples) >= 2
    kinds = {example.kind for example in examples}
    assert "correct" in kinds
    assert "recovery" in kinds


def test_load_jsonl_rejects_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_jsonl(path)


def test_validate_no_id_overlap_raises() -> None:
    shared = TraceExample(
        id="dup",
        kind="correct",
        messages=[Message(role="user", content="x")],
    )
    with pytest.raises(ValueError, match="overlap"):
        validate_no_id_overlap([shared], [shared])


def test_validate_no_id_overlap_ok() -> None:
    left = TraceExample(
        id="a",
        kind="correct",
        messages=[Message(role="user", content="x")],
    )
    right = TraceExample(
        id="b",
        kind="recovery",
        messages=[Message(role="user", content="y")],
    )
    validate_no_id_overlap([left], [right])
