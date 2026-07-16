"""Training configuration for Unsloth QLoRA."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class TrainConfig(BaseModel):
    """Hyperparameters for a local QLoRA run on RTX 3080 10GB."""

    base_model: str = "unsloth/Qwen2.5-3B-Instruct"
    lora_rank: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    max_seq_length: int = Field(default=1024, ge=128)
    output_dir: Path = Path("adapters/qwen25-3b-recovery")
    train_jsonl: Path = Path("data/curated/train.jsonl")
    val_jsonl: Path = Path("data/curated/val.jsonl")
    num_train_epochs: float = Field(default=1.0, gt=0)
    per_device_train_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0)
    warmup_ratio: float = Field(default=0.03, ge=0.0)
    logging_steps: int = Field(default=5, ge=1)
    seed: int = Field(default=42, ge=0)
    max_steps: int | None = Field(
        default=None,
        description="If set, overrides epochs (useful for --smoke runs).",
    )
