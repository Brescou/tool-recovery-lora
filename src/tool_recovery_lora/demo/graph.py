"""LangGraph recovery demo: bad tool call → error → LoRA repair → validate."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from tool_recovery_lora.data.schema import ToolCall
from tool_recovery_lora.demo.scenarios import (
    SCENARIOS,
    DemoScenario,
    scenario_seed_messages,
)
from tool_recovery_lora.eval.infer import generate_from_hf_messages, load_infer_model
from tool_recovery_lora.eval.metrics import score_tool_call
from tool_recovery_lora.eval.parse import parse_first_tool_call
from tool_recovery_lora.train.format_messages import message_to_hf


class RecoveryState(TypedDict, total=False):
    """State for the recovery demo graph."""

    scenario_id: str
    messages_preview: list[str]
    generation: str
    predicted: dict[str, Any] | None
    scores: dict[str, bool]
    latency_ms: float
    ok: bool


def _require_langgraph() -> Any:
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise ImportError(
            "langgraph is required for the demo. "
            "Install with: uv sync --extra train"
        ) from exc
    return END, StateGraph


def build_recovery_graph(model: Any, tokenizer: Any) -> Any:
    """Compile a LangGraph that repairs an injected bad tool call."""
    end, state_graph = _require_langgraph()

    def load_scenario(state: RecoveryState) -> RecoveryState:
        scenario = SCENARIOS[state["scenario_id"]]
        seed = scenario_seed_messages(scenario)
        preview = [
            f"[{message.role}] "
            f"{(message.content or '')[:120] or _tool_preview(message)}"
            for message in seed
        ]
        return {"messages_preview": preview}

    def repair_with_lora(state: RecoveryState) -> RecoveryState:
        scenario = SCENARIOS[state["scenario_id"]]
        seed = scenario_seed_messages(scenario)
        hf_messages = [
            message_to_hf(message, call_id_prefix=f"demo_{index}")
            for index, message in enumerate(seed)
        ]
        text, latency_ms = generate_from_hf_messages(
            model, tokenizer, hf_messages, max_new_tokens=256
        )
        predicted = parse_first_tool_call(text)
        return {
            "generation": text,
            "predicted": predicted.model_dump() if predicted else None,
            "latency_ms": latency_ms,
        }

    def validate(state: RecoveryState) -> RecoveryState:
        scenario = SCENARIOS[state["scenario_id"]]
        predicted_raw = state.get("predicted")
        predicted = (
            ToolCall.model_validate(predicted_raw) if predicted_raw else None
        )
        scores = score_tool_call(predicted, scenario.expected)
        return {"scores": scores, "ok": bool(scores["core_args_exact"])}

    graph = state_graph(RecoveryState)
    graph.add_node("load_scenario", load_scenario)
    graph.add_node("repair_with_lora", repair_with_lora)
    graph.add_node("validate", validate)
    graph.set_entry_point("load_scenario")
    graph.add_edge("load_scenario", "repair_with_lora")
    graph.add_edge("repair_with_lora", "validate")
    graph.add_edge("validate", end)
    return graph.compile()


def _tool_preview(message: Any) -> str:
    if message.tool_calls:
        call = message.tool_calls[0]
        return f"tool_call {call.name}({call.arguments})"
    return ""


def run_recovery_demo(
    scenario_id: str,
    *,
    adapter_dir: Path,
) -> RecoveryState:
    """Load the LoRA, run one recovery scenario through LangGraph.

    Args:
        scenario_id: Key in ``SCENARIOS``.
        adapter_dir: Trained adapter path.

    Returns:
        Final graph state.
    """
    if scenario_id not in SCENARIOS:
        known = ", ".join(sorted(SCENARIOS))
        raise KeyError(f"unknown scenario {scenario_id!r}; choose from: {known}")

    model, tokenizer = load_infer_model(adapter_dir=adapter_dir)
    app = build_recovery_graph(model, tokenizer)
    return app.invoke({"scenario_id": scenario_id})


def list_scenarios() -> list[DemoScenario]:
    """Return available demo scenarios."""
    return list(SCENARIOS.values())
