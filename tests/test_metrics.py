"""Tests for objective tool-call metrics."""

from __future__ import annotations

from pathlib import Path

from tool_recovery_lora.data.schema import Message, ToolCall
from tool_recovery_lora.eval.metrics import (
    extract_last_assistant_tool_call,
    score_tool_call,
)
from tool_recovery_lora.eval.runner import run_smoke_eval

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_PATH = REPO_ROOT / "data" / "eval" / "smoke.jsonl"


def test_extract_last_assistant_tool_call() -> None:
    messages = [
        Message(
            role="assistant",
            tool_calls=[ToolCall(name="wrong", arguments={})],
        ),
        Message(role="tool", content="ERROR", tool_call_id="1", name="wrong"),
        Message(
            role="assistant",
            tool_calls=[ToolCall(name="meeting_prep", arguments={"company": "X"})],
        ),
    ]
    result = extract_last_assistant_tool_call(messages)
    assert result is not None
    assert result.name == "meeting_prep"


def test_score_tool_call_exact_match() -> None:
    expected = ToolCall(name="meeting_prep", arguments={"company": "Acme"})
    predicted = ToolCall(name="meeting_prep", arguments={"company": "Acme"})
    scores = score_tool_call(predicted, expected)
    assert scores == {
        "name_match": True,
        "args_json_valid": True,
        "args_exact": True,
        "core_args_exact": True,
    }


def test_score_tool_call_core_ignores_context_mismatch() -> None:
    expected = ToolCall(
        name="meeting_prep",
        arguments={
            "company": "Acme",
            "person": "Jane",
            "meeting_goal": "discovery",
            "context": "A",
        },
    )
    predicted = ToolCall(
        name="meeting_prep",
        arguments={
            "company": "Acme",
            "person": "Jane",
            "meeting_goal": "discovery",
            "context": "B",
        },
    )
    scores = score_tool_call(predicted, expected)
    assert scores["args_exact"] is False
    assert scores["core_args_exact"] is True


def test_score_tool_call_missing_prediction() -> None:
    expected = ToolCall(name="meeting_prep", arguments={})
    scores = score_tool_call(None, expected)
    assert scores["name_match"] is False
    assert scores["args_exact"] is False
    assert scores["core_args_exact"] is False


def test_run_smoke_eval_perfect_on_fixtures() -> None:
    results = run_smoke_eval(SMOKE_PATH)
    assert results["n_examples"] >= 3
    assert results["name_match"] == 1.0
    assert results["args_exact"] == 1.0
