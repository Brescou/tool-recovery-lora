"""Build generation prompts from TraceExample conversations."""

from __future__ import annotations

from tool_recovery_lora.data.schema import Message, TraceExample
from tool_recovery_lora.train.format_messages import message_to_hf


def gold_assistant_index(example: TraceExample) -> int:
    """Index of the final assistant turn that holds the gold tool call.

    Raises:
        ValueError: If no assistant tool_calls turn exists.
    """
    last_idx: int | None = None
    for index, message in enumerate(example.messages):
        if message.role == "assistant" and message.tool_calls:
            last_idx = index
    if last_idx is None:
        raise ValueError(f"example {example.id!r} has no assistant tool_calls")
    return last_idx


def prompt_messages(example: TraceExample) -> list[Message]:
    """Return messages before the gold assistant turn (generation target)."""
    index = gold_assistant_index(example)
    return list(example.messages[:index])


def prompt_hf_messages(example: TraceExample) -> list[dict]:
    """HF chat dicts for ``apply_chat_template(..., add_generation_prompt=True)``."""
    return [
        message_to_hf(message, call_id_prefix=f"{example.id}_p{index}")
        for index, message in enumerate(prompt_messages(example))
    ]
