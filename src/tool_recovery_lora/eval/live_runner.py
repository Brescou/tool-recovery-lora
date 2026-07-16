"""Live model evaluation over smoke JSONL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tool_recovery_lora.data.loader import load_jsonl
from tool_recovery_lora.eval.infer import (
    DEFAULT_BASE_MODEL,
    generate_completion,
    load_infer_model,
)
from tool_recovery_lora.eval.metrics import score_tool_call
from tool_recovery_lora.eval.parse import parse_first_tool_call
from tool_recovery_lora.eval.runner import expected_tool_call


def run_live_eval(
    path: Path,
    *,
    adapter_dir: Path | None = None,
    model_name: str | None = None,
    no_adapter: bool = False,
    max_seq_length: int = 1024,
    max_new_tokens: int = 256,
    limit: int | None = None,
) -> dict[str, Any]:
    """Generate predictions and score objectively (LoRA or vanilla base).

    Args:
        path: Smoke/eval JSONL.
        adapter_dir: Trained adapter directory (LoRA mode).
        model_name: Hub id for vanilla base model when ``no_adapter``.
        no_adapter: If True, load ``model_name`` without LoRA weights.
        max_seq_length: Model context.
        max_new_tokens: Generation budget.
        limit: Optional cap on number of examples (debug).

    Returns:
        Aggregate metrics plus per-example rows under ``details``.
    """
    examples = load_jsonl(path)
    if not examples:
        raise ValueError(f"no examples in {path}")
    if limit is not None:
        examples = examples[:limit]

    if no_adapter:
        resolved_model = model_name or DEFAULT_BASE_MODEL
        model, tokenizer = load_infer_model(
            model_name=resolved_model,
            max_seq_length=max_seq_length,
        )
        model_label = resolved_model
        adapter_label = None
    else:
        if adapter_dir is None:
            raise ValueError("adapter_dir is required unless no_adapter=True")
        model, tokenizer = load_infer_model(
            adapter_dir=adapter_dir,
            max_seq_length=max_seq_length,
        )
        model_label = model_name or DEFAULT_BASE_MODEL
        adapter_label = str(adapter_dir)

    totals = {
        "name_match": 0,
        "args_json_valid": 0,
        "args_exact": 0,
        "core_args_exact": 0,
        "recovery_exact": 0,
        "recovery_core_exact": 0,
        "n_recovery": 0,
    }
    latencies: list[float] = []
    details: list[dict[str, Any]] = []

    for example in examples:
        expected = expected_tool_call(example)
        text, latency_ms = generate_completion(
            model,
            tokenizer,
            example,
            max_new_tokens=max_new_tokens,
        )
        latencies.append(latency_ms)
        predicted = parse_first_tool_call(text)
        if predicted is None and "<tool_call>" in text:
            scores = {
                "name_match": False,
                "args_json_valid": False,
                "args_exact": False,
                "core_args_exact": False,
            }
        else:
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
                "generation_preview": text[:400],
            }
        )

    n_examples = float(len(examples))
    n_recovery = float(totals["n_recovery"]) or 1.0
    return {
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
        "model_name": model_label,
        "adapter_dir": adapter_label,
        "no_adapter": no_adapter,
        "details": details,
    }
