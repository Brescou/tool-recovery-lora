"""Tests for Qwen tool-call parsing and prompt truncation."""

from __future__ import annotations

from tool_recovery_lora.data.schema import Message, ToolCall, TraceExample
from tool_recovery_lora.eval.parse import (
    parse_first_tool_call,
    parse_first_tool_call_detailed,
    parse_tool_calls_from_text,
)
from tool_recovery_lora.eval.prompt import gold_assistant_index, prompt_messages


def test_parse_tool_call_block() -> None:
    text = (
        "Sure.\n<tool_call>\n"
        '{"name": "meeting_prep", "arguments": {"company": "Acme", "person": "Jane"}}\n'
        "</tool_call>\n"
    )
    calls = parse_tool_calls_from_text(text)
    assert len(calls) == 1
    assert calls[0].name == "meeting_prep"
    assert calls[0].arguments["company"] == "Acme"
    detailed = parse_first_tool_call_detailed(text)
    assert detailed is not None
    assert detailed.args_as_object is True


def test_parse_arguments_as_json_string_still_tolerated() -> None:
    text = (
        '<tool_call>\n{"name": "meeting_prep", '
        '"arguments": "{\\"company\\": \\"Globex\\"}"}\n</tool_call>'
    )
    call = parse_first_tool_call(text)
    assert call is not None
    assert call.arguments == {"company": "Globex"}
    detailed = parse_first_tool_call_detailed(text)
    assert detailed is not None
    assert detailed.args_as_object is False


def test_prompt_messages_drops_gold_assistant() -> None:
    example = TraceExample(
        id="p1",
        kind="recovery",
        messages=[
            Message(role="system", content="sys"),
            Message(role="user", content="prep"),
            Message(
                role="assistant",
                tool_calls=[ToolCall(name="wrong", arguments={})],
            ),
            Message(role="tool", content="ERROR", tool_call_id="1", name="wrong"),
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(name="meeting_prep", arguments={"company": "Acme"})
                ],
            ),
        ],
    )
    assert gold_assistant_index(example) == 4
    prompt = prompt_messages(example)
    assert len(prompt) == 4
    assert prompt[-1].role == "tool"
