"""Training configuration for Unsloth QLoRA (Phase 0 stub)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class TrainConfig(BaseModel):
    """Hyperparameters for a local QLoRA run on RTX 3080 10GB."""

    base_model: str = "unsloth/Qwen2.5-3B-Instruct"
    lora_rank: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    max_seq_length: int = Field(default=2048, ge=128)
    output_dir: Path = Path("adapters/qwen25-3b-recovery")
    train_jsonl: Path = Path("data/curated/train.jsonl")
    val_jsonl: Path = Path("data/curated/val.jsonl")
