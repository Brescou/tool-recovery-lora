"""Tests for TrainConfig defaults."""

from __future__ import annotations

from pathlib import Path

from tool_recovery_lora.train.config import TrainConfig


def test_train_config_defaults() -> None:
    config = TrainConfig()
    assert config.base_model == "unsloth/Qwen2.5-3B-Instruct"
    assert config.lora_rank == 16
    assert config.lora_alpha == 32
    assert config.max_seq_length == 1024
    assert config.output_dir == Path("adapters/qwen25-3b-recovery")
    assert config.per_device_train_batch_size == 1
    assert config.gradient_accumulation_steps == 8
    assert config.max_steps is None
