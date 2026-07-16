"""Tests for chat-template message formatting."""

from __future__ import annotations

from tool_recovery_lora.data.schema import Message, ToolCall, TraceExample
from tool_recovery_lora.train.format_messages import (
    example_to_hf_messages,
    message_to_hf,
)


def test_message_to_hf_tool_call_arguments_are_object() -> None:
    message = Message(
        role="assistant",
        tool_calls=[ToolCall(name="meeting_prep", arguments={"company": "Acme"})],
    )
    payload = message_to_hf(message)
    assert payload["role"] == "assistant"
    assert payload["tool_calls"][0]["type"] == "function"
    assert payload["tool_calls"][0]["function"]["name"] == "meeting_prep"
    assert payload["tool_calls"][0]["function"]["arguments"] == {"company": "Acme"}
    assert isinstance(payload["tool_calls"][0]["function"]["arguments"], dict)


def test_example_to_hf_messages_preserves_recovery_arc() -> None:
    example = TraceExample(
        id="t1",
        kind="recovery",
        messages=[
            Message(role="system", content="sys"),
            Message(role="user", content="prep Acme"),
            Message(role="assistant", content='{"broken"'),
            Message(
                role="tool",
                content="ERROR",
                tool_call_id="x",
                name="meeting_prep",
            ),
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(name="meeting_prep", arguments={"company": "Acme"})
                ],
            ),
        ],
    )
    messages = example_to_hf_messages(example)
    assert len(messages) == 5
    assert messages[2]["content"].startswith("{")
    assert messages[-1]["tool_calls"][0]["function"]["name"] == "meeting_prep"
    assert isinstance(
        messages[-1]["tool_calls"][0]["function"]["arguments"],
        dict,
    )
