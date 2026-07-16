"""Unsloth QLoRA training runner for tool-recovery traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tool_recovery_lora.data.loader import load_jsonl
from tool_recovery_lora.train.config import TrainConfig
from tool_recovery_lora.train.format_messages import render_example_text


def _require_train_deps() -> tuple[Any, Any, Any, Any]:
    """Import heavy train deps or raise a clear error."""
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise ImportError(
            "Train dependencies missing. Install with: uv sync --extra train"
        ) from exc

    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    return FastLanguageModel, Dataset, SFTConfig, SFTTrainer


def build_text_dataset(
    jsonl_path: Path,
    tokenizer: Any,
) -> Any:
    """Load JSONL traces and render chat-template text rows."""
    _, Dataset, _, _ = _require_train_deps()
    examples = load_jsonl(jsonl_path)
    texts = [render_example_text(example, tokenizer) for example in examples]
    return Dataset.from_dict({"text": texts})


def run_training(config: TrainConfig) -> Path:
    """Run QLoRA fine-tuning and save LoRA adapters.

    Args:
        config: Training hyperparameters and paths.

    Returns:
        Path to the saved adapter directory.
    """
    FastLanguageModel, _, SFTConfig, SFTTrainer = _require_train_deps()

    if not config.train_jsonl.is_file():
        raise FileNotFoundError(config.train_jsonl)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=config.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
    )

    train_dataset = build_text_dataset(config.train_jsonl, tokenizer)
    eval_dataset = None
    if config.val_jsonl.is_file() and config.max_steps is None:
        eval_dataset = build_text_dataset(config.val_jsonl, tokenizer)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    sft_args: dict[str, Any] = {
        "output_dir": str(config.output_dir),
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "logging_steps": config.logging_steps,
        "warmup_ratio": config.warmup_ratio,
        "optim": "adamw_8bit",
        "seed": config.seed,
        "report_to": "none",
        "save_strategy": "epoch" if config.max_steps is None else "no",
        "fp16": not _bf16_supported(),
        "bf16": _bf16_supported(),
        "dataset_text_field": "text",
        "max_seq_length": config.max_seq_length,
        "packing": False,
    }
    if config.max_steps is not None:
        sft_args["max_steps"] = config.max_steps
        sft_args["num_train_epochs"] = 1.0
    else:
        sft_args["num_train_epochs"] = float(config.num_train_epochs)
        if eval_dataset is not None:
            sft_args["eval_strategy"] = "epoch"
            sft_args["per_device_eval_batch_size"] = 1

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(**sft_args),
    )
    train_result = trainer.train()
    model.save_pretrained(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))

    metrics_path = config.output_dir / "train_metrics.json"
    metrics = {
        "train_runtime": getattr(train_result, "metrics", {}).get("train_runtime"),
        "train_loss": getattr(train_result, "metrics", {}).get("train_loss"),
        "metrics": getattr(train_result, "metrics", {}),
        "config": config.model_dump(mode="json"),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return config.output_dir


def _bf16_supported() -> bool:
    """Return True if the GPU supports bf16 (Ampere+)."""
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except Exception:
        return False
