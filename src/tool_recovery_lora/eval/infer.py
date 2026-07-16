"""Load fine-tuned Unsloth model and generate tool-call completions."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from tool_recovery_lora.data.schema import TraceExample
from tool_recovery_lora.eval.prompt import prompt_hf_messages
from tool_recovery_lora.train.config import TrainConfig

DEFAULT_BASE_MODEL = TrainConfig().base_model


def load_infer_model(
    *,
    model_name: str | None = None,
    adapter_dir: Path | None = None,
    max_seq_length: int = 1024,
) -> tuple[Any, Any]:
    """Load a 4-bit model for inference (base-only or base+LoRA adapter).

    Args:
        model_name: Hub id for the base model when not using an adapter dir
            as the primary load path (e.g. ``unsloth/Qwen2.5-3B-Instruct``).
        adapter_dir: Directory produced by ``make train``. When set alone,
            Unsloth loads base + adapter from this path. Ignored when
            ``model_name`` is set without loading adapters (vanilla baseline).
        max_seq_length: Context length.

    Returns:
        ``(model, tokenizer)`` ready for generation.
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise ImportError(
            "Train/infer deps missing. Install with: uv sync --extra train"
        ) from exc

    if model_name:
        load_name = model_name
    elif adapter_dir is not None:
        if not adapter_dir.is_dir():
            raise FileNotFoundError(adapter_dir)
        load_name = str(adapter_dir)
    else:
        raise ValueError("Provide model_name and/or adapter_dir")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=load_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_from_hf_messages(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    max_new_tokens: int = 256,
) -> tuple[str, float]:
    """Generate an assistant continuation from HF chat messages.

    Args:
        model: Unsloth/PEFT model in inference mode.
        tokenizer: Matching tokenizer.
        messages: Chat messages (no trailing assistant gold turn).
        max_new_tokens: Generation budget.

    Returns:
        ``(decoded_new_text, latency_ms)``.
    """
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    start = time.perf_counter()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0

    new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    return text, latency_ms


def generate_completion(
    model: Any,
    tokenizer: Any,
    example: TraceExample,
    *,
    max_new_tokens: int = 256,
) -> tuple[str, float]:
    """Generate an assistant continuation for one eval example.

    Args:
        model: Unsloth/PEFT model in inference mode.
        tokenizer: Matching tokenizer.
        example: Trace to evaluate.
        max_new_tokens: Generation budget.

    Returns:
        ``(decoded_new_text, latency_ms)``.
    """
    return generate_from_hf_messages(
        model,
        tokenizer,
        prompt_hf_messages(example),
        max_new_tokens=max_new_tokens,
    )
