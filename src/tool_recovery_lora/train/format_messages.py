"""Convert TraceExample messages to chat-template-ready dicts."""

from __future__ import annotations

import json
from typing import Any

from tool_recovery_lora.data.schema import Message, ToolCall, TraceExample


def _tool_call_to_hf(tool_call: ToolCall, call_id: str) -> dict[str, Any]:
    """Map our ToolCall to HuggingFace/OpenAI-style tool_call dict."""
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": tool_call.name,
            "arguments": json.dumps(tool_call.arguments, ensure_ascii=False),
        },
    }


def message_to_hf(message: Message, *, call_id_prefix: str = "call") -> dict[str, Any]:
    """Convert one Message to a dict accepted by ``apply_chat_template``."""
    if message.role == "assistant" and message.tool_calls:
        tool_calls = [
            _tool_call_to_hf(tool_call, f"{call_id_prefix}_{index}")
            for index, tool_call in enumerate(message.tool_calls)
        ]
        payload: dict[str, Any] = {"role": "assistant", "tool_calls": tool_calls}
        if message.content:
            payload["content"] = message.content
        return payload

    payload = {"role": message.role, "content": message.content or ""}
    if message.role == "tool":
        if message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        if message.name:
            payload["name"] = message.name
    return payload


def example_to_hf_messages(example: TraceExample) -> list[dict[str, Any]]:
    """Convert a full TraceExample to HF chat messages."""
    return [
        message_to_hf(message, call_id_prefix=f"{example.id}_{index}")
        for index, message in enumerate(example.messages)
    ]


def render_example_text(example: TraceExample, tokenizer: Any) -> str:
    """Apply the tokenizer chat template to one example.

    Args:
        example: Curated trace.
        tokenizer: HuggingFace tokenizer with ``apply_chat_template``.

    Returns:
        Single training string ending with the generation prompt turned off.
    """
    messages = example_to_hf_messages(example)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
