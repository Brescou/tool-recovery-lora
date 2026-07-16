"""Objective metrics for tool-call evaluation (no LLM-as-judge)."""

from __future__ import annotations

from tool_recovery_lora.data.schema import Message, ToolCall


def extract_last_assistant_tool_call(messages: list[Message]) -> ToolCall | None:
    """Return the last tool call from the last assistant message that has any.

    Args:
        messages: Conversation turns.

    Returns:
        The last ``ToolCall`` found, or ``None``.
    """
    for message in reversed(messages):
        if message.role != "assistant":
            continue
        if message.tool_calls:
            return message.tool_calls[-1]
    return None


def score_tool_call(
    predicted: ToolCall | None,
    expected: ToolCall,
) -> dict[str, bool]:
    """Score a predicted tool call against an expected reference.

    Args:
        predicted: Model (or fixture) prediction; ``None`` if missing.
        expected: Ground-truth tool call.

    Returns:
        Flags ``name_match``, ``args_json_valid``, ``args_exact``.
    """
    if predicted is None:
        return {
            "name_match": False,
            "args_json_valid": False,
            "args_exact": False,
        }

    # Arguments are already a dict after schema parse → always JSON-valid.
    args_json_valid = isinstance(predicted.arguments, dict)
    name_match = predicted.name == expected.name
    args_exact = name_match and predicted.arguments == expected.arguments
    return {
        "name_match": name_match,
        "args_json_valid": args_json_valid,
        "args_exact": args_exact,
    }
