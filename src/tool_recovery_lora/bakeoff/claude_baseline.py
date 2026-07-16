"""Anthropic Claude baseline: prompting + native tools API (no fine-tune)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from tool_recovery_lora.data.generate import TOOL_NAME
from tool_recovery_lora.data.schema import Message, ToolCall, TraceExample
from tool_recovery_lora.eval.metrics import score_tool_call
from tool_recovery_lora.eval.prompt import prompt_messages
from tool_recovery_lora.eval.runner import expected_tool_call

MEETING_PREP_TOOL: dict[str, Any] = {
    "name": TOOL_NAME,
    "description": (
        "Prepare a structured sales meeting brief for langgraph-agent-stack. "
        "Call this tool with company (required), person, "
        "meeting_goal, and optional context."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "company": {"type": "string"},
            "person": {"type": "string"},
            "meeting_goal": {"type": "string"},
            "context": {"type": "string"},
        },
        "required": ["company"],
    },
}

# USD per 1k tokens (aligned with langgraph-agent-stack cost table).
_DEFAULT_PRICE_IN = 0.003
_DEFAULT_PRICE_OUT = 0.015


def load_dotenv_files(paths: list[Path]) -> None:
    """Load KEY=VALUE pairs into os.environ if not already set."""
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value


def _messages_to_anthropic(
    messages: list[Message],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Split system prompt and map turns to Anthropic Messages API shape."""
    system: str | None = None
    api_messages: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] = []

    def flush_tool_results() -> None:
        nonlocal pending_tool_results
        if pending_tool_results:
            api_messages.append({"role": "user", "content": pending_tool_results})
            pending_tool_results = []

    for index, message in enumerate(messages):
        if message.role == "system":
            system = message.content or system
            continue
        if message.role == "user":
            flush_tool_results()
            api_messages.append({"role": "user", "content": message.content or ""})
            continue
        if message.role == "assistant":
            flush_tool_results()
            if message.tool_calls:
                blocks: list[dict[str, Any]] = []
                if message.content:
                    blocks.append({"type": "text", "text": message.content})
                for call_index, call in enumerate(message.tool_calls):
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": message.tool_call_id
                            or f"bad_{index}_{call_index}",
                            "name": call.name,
                            "input": call.arguments,
                        }
                    )
                api_messages.append({"role": "assistant", "content": blocks})
            else:
                api_messages.append(
                    {"role": "assistant", "content": message.content or ""}
                )
            continue
        if message.role == "tool":
            tool_use_id = message.tool_call_id or f"bad_{index}"
            # Fix id to match prior assistant tool_use if we generated one.
            if api_messages and api_messages[-1]["role"] == "assistant":
                content = api_messages[-1]["content"]
                if isinstance(content, list):
                    for block in content:
                        if block.get("type") == "tool_use":
                            tool_use_id = block["id"]
                            break
            pending_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": message.content or "",
                    "is_error": True,
                }
            )
    flush_tool_results()
    return system, api_messages


def _extract_tool_call(response: Any) -> ToolCall | None:
    """Take the first tool_use block from an Anthropic response."""
    for block in response.content:
        if getattr(block, "type", None) == "tool_use":
            arguments = dict(block.input) if block.input is not None else {}
            return ToolCall(name=str(block.name), arguments=arguments)
    return None


def _estimate_cost_usd(
    input_tokens: int,
    output_tokens: int,
    *,
    price_in: float = _DEFAULT_PRICE_IN,
    price_out: float = _DEFAULT_PRICE_OUT,
) -> float:
    return (input_tokens / 1000.0) * price_in + (output_tokens / 1000.0) * price_out


def run_claude_eval(
    examples: list[TraceExample],
    *,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 512,
) -> dict[str, Any]:
    """Evaluate Claude with native tools on the same traces as LoRA live eval."""
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "anthropic package required. Install with: uv sync --extra bakeoff"
        ) from exc

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    client = anthropic.Anthropic(api_key=api_key)
    totals = {
        "name_match": 0,
        "args_json_valid": 0,
        "args_exact": 0,
        "core_args_exact": 0,
        "recovery_exact": 0,
        "recovery_core_exact": 0,
        "n_recovery": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
    }
    latencies: list[float] = []
    details: list[dict[str, Any]] = []

    for example in examples:
        expected = expected_tool_call(example)
        system, api_messages = _messages_to_anthropic(prompt_messages(example))
        start = time.perf_counter()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system or "You are a helpful tool-calling assistant.",
            tools=[MEETING_PREP_TOOL],
            tool_choice={"type": "any"},
            messages=api_messages,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(latency_ms)

        usage = response.usage
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        cost = _estimate_cost_usd(input_tokens, output_tokens)
        totals["input_tokens"] += input_tokens
        totals["output_tokens"] += output_tokens
        totals["cost_usd"] += cost

        predicted = _extract_tool_call(response)
        scores = score_tool_call(predicted, expected)
        for key in (
            "name_match",
            "args_json_valid",
            "args_exact",
            "core_args_exact",
        ):
            totals[key] += int(scores[key])
        if example.kind == "recovery":
            totals["n_recovery"] += 1
            totals["recovery_exact"] += int(scores["args_exact"])
            totals["recovery_core_exact"] += int(scores["core_args_exact"])

        details.append(
            {
                "id": example.id,
                "kind": example.kind,
                "latency_ms": round(latency_ms, 1),
                "scores": scores,
                "expected": expected.model_dump(),
                "predicted": predicted.model_dump() if predicted else None,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": round(cost, 6),
            }
        )

    n_examples = float(len(examples))
    n_recovery = float(totals["n_recovery"]) or 1.0
    return {
        "provider": "anthropic",
        "model": model,
        "n_examples": n_examples,
        "name_match": totals["name_match"] / n_examples,
        "args_json_valid": totals["args_json_valid"] / n_examples,
        "args_exact": totals["args_exact"] / n_examples,
        "core_args_exact": totals["core_args_exact"] / n_examples,
        "recovery_exact": totals["recovery_exact"] / n_recovery,
        "recovery_core_exact": totals["recovery_core_exact"] / n_recovery,
        "n_recovery": float(totals["n_recovery"]),
        "latency_ms_mean": sum(latencies) / len(latencies),
        "latency_ms_p50": sorted(latencies)[len(latencies) // 2],
        "input_tokens": totals["input_tokens"],
        "output_tokens": totals["output_tokens"],
        "cost_usd_total": round(totals["cost_usd"], 6),
        "cost_usd_per_request": round(totals["cost_usd"] / n_examples, 6),
        "details": details,
    }
