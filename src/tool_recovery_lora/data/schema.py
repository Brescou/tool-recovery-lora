"""Pydantic models for tool-recovery traces (schema tool_recovery_trace_v1)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

TRACE_SCHEMA_VERSION = "tool_recovery_trace_v1"

Role = Literal["system", "user", "assistant", "tool"]
ExampleKind = Literal["correct", "recovery"]


class ToolCall(BaseModel):
    """A single structured tool invocation."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """One turn in a tool-calling conversation."""

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    @field_validator("tool_calls")
    @classmethod
    def _empty_tool_calls_to_none(
        cls, value: list[ToolCall] | None
    ) -> list[ToolCall] | None:
        if value is not None and len(value) == 0:
            return None
        return value


class TraceExample(BaseModel):
    """One curated training or eval example."""

    id: str
    schema_version: str = TRACE_SCHEMA_VERSION
    kind: ExampleKind
    messages: list[Message]
    meta: dict[str, Any] | None = None

    @field_validator("schema_version")
    @classmethod
    def _check_schema_version(cls, value: str) -> str:
        if value != TRACE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {value!r}; "
                f"expected {TRACE_SCHEMA_VERSION!r}"
            )
        return value

    @field_validator("messages")
    @classmethod
    def _require_messages(cls, value: list[Message]) -> list[Message]:
        if not value:
            raise ValueError("messages must be non-empty")
        return value
