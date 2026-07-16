"""Parse tool calls from model generation text (Qwen-style)."""

from __future__ import annotations

import json
import re
from typing import Any

from tool_recovery_lora.data.schema import ToolCall

_TOOL_CALL_BLOCK = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)


def _coerce_arguments(raw: Any) -> dict[str, Any] | None:
    """Normalize arguments to a dict; return None if invalid."""
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


def parse_tool_calls_from_text(text: str) -> list[ToolCall]:
    """Extract tool calls from Qwen ``<tool_call>`` XML blocks.

    Args:
        text: Raw model generation.

    Returns:
        Parsed tool calls (may be empty).
    """
    calls: list[ToolCall] = []
    for match in _TOOL_CALL_BLOCK.finditer(text):
        payload_raw = match.group(1).strip()
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or "name" not in payload:
            continue
        arguments = _coerce_arguments(payload.get("arguments", {}))
        if arguments is None:
            # Keep the call with empty args but mark via skipping exact match later;
            # still expose name if present.
            continue
        calls.append(ToolCall(name=str(payload["name"]), arguments=arguments))
    return calls


def parse_first_tool_call(text: str) -> ToolCall | None:
    """Return the first parsed tool call, or ``None``."""
    calls = parse_tool_calls_from_text(text)
    return calls[0] if calls else None
