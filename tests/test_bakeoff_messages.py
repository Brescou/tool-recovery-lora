"""Tests for Anthropic message mapping (no network)."""

from __future__ import annotations

from tool_recovery_lora.bakeoff.claude_baseline import _messages_to_anthropic
from tool_recovery_lora.data.schema import Message, ToolCall


def test_messages_to_anthropic_recovery_arc() -> None:
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="prep Globex"),
        Message(
            role="assistant",
            tool_calls=[ToolCall(name="meeting_prep", arguments={"company": "X"})],
        ),
        Message(
            role="tool",
            content="ERROR: missing_args",
            tool_call_id="call_1",
            name="meeting_prep",
        ),
    ]
    system, api_messages = _messages_to_anthropic(messages)
    assert system == "sys"
    assert api_messages[0]["role"] == "user"
    assert api_messages[1]["role"] == "assistant"
    assert api_messages[1]["content"][0]["type"] == "tool_use"
    assert api_messages[2]["role"] == "user"
    assert api_messages[2]["content"][0]["type"] == "tool_result"
    assert api_messages[2]["content"][0]["is_error"] is True
