"""Parse tool calls from model generation text (Qwen-style)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from tool_recovery_lora.data.schema import ToolCall

_TOOL_CALL_BLOCK = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedToolCall:
    """A tool call plus whether ``arguments`` was a JSON object in the raw text."""

    call: ToolCall
    args_as_object: bool


def _coerce_arguments(raw: Any) -> dict[str, Any] | None:
    """Normalize arguments to a dict; return None if invalid.

    String arguments are still accepted for backward compatibility with
    models trained before the object-arguments template fix.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
        return None
    return None


def parse_tool_calls_detailed(text: str) -> list[ParsedToolCall]:
    """Extract tool calls and whether each had object-form arguments.

    Args:
        text: Raw model generation.

    Returns:
        Parsed calls (may be empty).
    """
    calls: list[ParsedToolCall] = []
    for match in _TOOL_CALL_BLOCK.finditer(text):
        payload_raw = match.group(1).strip()
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or "name" not in payload:
            continue
        raw_args = payload.get("arguments", {})
        args_as_object = isinstance(raw_args, dict)
        arguments = _coerce_arguments(raw_args)
        if arguments is None:
            continue
        calls.append(
            ParsedToolCall(
                call=ToolCall(name=str(payload["name"]), arguments=arguments),
                args_as_object=args_as_object,
            )
        )
    return calls


def parse_tool_calls_from_text(text: str) -> list[ToolCall]:
    """Extract tool calls from Qwen ``<tool_call>`` XML blocks."""
    return [item.call for item in parse_tool_calls_detailed(text)]


def parse_first_tool_call(text: str) -> ToolCall | None:
    """Return the first parsed tool call, or ``None``."""
    calls = parse_tool_calls_from_text(text)
    return calls[0] if calls else None


def parse_first_tool_call_detailed(text: str) -> ParsedToolCall | None:
    """Return the first parsed tool call with object-form flag, or ``None``."""
    calls = parse_tool_calls_detailed(text)
    return calls[0] if calls else None
