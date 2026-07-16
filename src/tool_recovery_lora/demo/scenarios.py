"""Fixed demo scenarios for tool-call recovery."""

from __future__ import annotations

from dataclasses import dataclass

from tool_recovery_lora.data.generate import SYSTEM_PROMPT, TOOL_NAME
from tool_recovery_lora.data.schema import Message, ToolCall


@dataclass(frozen=True)
class DemoScenario:
    """One scripted recovery demo."""

    id: str
    title: str
    user_text: str
    bad_call: ToolCall
    error_message: str
    expected: ToolCall


SCENARIOS: dict[str, DemoScenario] = {
    "missing_args": DemoScenario(
        id="missing_args",
        title="Repair missing meeting_goal",
        user_text=(
            "Prepare a briefing for Globex with Bob Martinez about partnership. "
            "Context: Warm intro from an existing customer."
        ),
        bad_call=ToolCall(
            name=TOOL_NAME,
            arguments={"company": "Globex", "person": "Bob Martinez"},
        ),
        error_message=(
            "ERROR: missing_args — meeting_goal is required when provided by the user"
        ),
        expected=ToolCall(
            name=TOOL_NAME,
            arguments={
                "company": "Globex",
                "person": "Bob Martinez",
                "meeting_goal": "partnership",
                "context": "Warm intro from an existing customer.",
            },
        ),
    ),
    "wrong_tool": DemoScenario(
        id="wrong_tool",
        title="Repair wrong tool name",
        user_text=(
            "Brief me before my call with Alice Nguyen (Initech). "
            "Focus on renewal."
        ),
        bad_call=ToolCall(
            name="research_analysis",
            arguments={"query": "Initech Alice Nguyen renewal"},
        ),
        error_message=(
            f"ERROR: wrong_tool — unknown tool for this task; use {TOOL_NAME}"
        ),
        expected=ToolCall(
            name=TOOL_NAME,
            arguments={
                "company": "Initech",
                "person": "Alice Nguyen",
                "meeting_goal": "renewal",
            },
        ),
    ),
}


def scenario_seed_messages(scenario: DemoScenario) -> list[Message]:
    """Build the conversation up to (and including) the tool error turn."""
    return [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content=scenario.user_text),
        Message(role="assistant", content=None, tool_calls=[scenario.bad_call]),
        Message(
            role="tool",
            content=scenario.error_message,
            tool_call_id=f"demo_{scenario.id}",
            name=scenario.bad_call.name,
        ),
    ]
