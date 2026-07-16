"""Smoke evaluation runner over curated JSONL fixtures."""

from __future__ import annotations

from pathlib import Path

from tool_recovery_lora.data.loader import load_jsonl
from tool_recovery_lora.data.schema import ToolCall, TraceExample
from tool_recovery_lora.eval.metrics import (
    extract_last_assistant_tool_call,
    score_tool_call,
)


def _expected_tool_call(example: TraceExample) -> ToolCall:
    """Read ``meta['expected_tool_call']`` or fall back to last assistant call."""
    if example.meta and "expected_tool_call" in example.meta:
        return ToolCall.model_validate(example.meta["expected_tool_call"])
    predicted = extract_last_assistant_tool_call(example.messages)
    if predicted is None:
        raise ValueError(
            f"example {example.id!r} has no expected_tool_call in meta "
            "and no assistant tool_calls"
        )
    return predicted


def run_smoke_eval(path: Path) -> dict[str, float]:
    """Score fixtures for self-consistency (Phase 0; no live model).

    Each example's last assistant tool call is compared to
    ``meta['expected_tool_call']``.

    Args:
        path: Path to smoke JSONL.

    Returns:
        Aggregate rates for ``name_match``, ``args_json_valid``, ``args_exact``,
        plus ``n_examples``.
    """
    examples = load_jsonl(path)
    if not examples:
        raise ValueError(f"no examples in {path}")

    totals = {"name_match": 0, "args_json_valid": 0, "args_exact": 0}
    for example in examples:
        expected = _expected_tool_call(example)
        predicted = extract_last_assistant_tool_call(example.messages)
        scores = score_tool_call(predicted, expected)
        for key in totals:
            totals[key] += int(scores[key])

    n_examples = float(len(examples))
    return {
        "n_examples": n_examples,
        "name_match": totals["name_match"] / n_examples,
        "args_json_valid": totals["args_json_valid"] / n_examples,
        "args_exact": totals["args_exact"] / n_examples,
    }
