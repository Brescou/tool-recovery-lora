"""Unit tests for demo scenarios (no GPU)."""

from __future__ import annotations

from tool_recovery_lora.demo.scenarios import SCENARIOS, scenario_seed_messages


def test_scenarios_have_recovery_arc() -> None:
    assert "missing_args" in SCENARIOS
    assert "wrong_tool" in SCENARIOS
    for scenario in SCENARIOS.values():
        messages = scenario_seed_messages(scenario)
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert messages[2].role == "assistant"
        assert messages[2].tool_calls
        assert messages[3].role == "tool"
        assert "ERROR" in (messages[3].content or "")
        assert scenario.expected.name == "meeting_prep"
