"""Tests for TraceExample schema validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tool_recovery_lora.data.schema import (
    TRACE_SCHEMA_VERSION,
    Message,
    ToolCall,
    TraceExample,
)


def test_trace_example_accepts_valid_correct_payload() -> None:
    example = TraceExample(
        id="ex-1",
        kind="correct",
        messages=[
            Message(role="user", content="hi"),
            Message(
                role="assistant",
                tool_calls=[ToolCall(name="meeting_prep", arguments={"company": "A"})],
            ),
        ],
    )
    assert example.schema_version == TRACE_SCHEMA_VERSION
    assert example.kind == "correct"


def test_trace_example_rejects_wrong_schema_version() -> None:
    with pytest.raises(ValidationError):
        TraceExample(
            id="ex-bad",
            schema_version="v0",
            kind="correct",
            messages=[Message(role="user", content="x")],
        )


def test_trace_example_rejects_empty_messages() -> None:
    with pytest.raises(ValidationError):
        TraceExample(id="ex-empty", kind="correct", messages=[])
