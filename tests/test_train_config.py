"""Tests for TrainConfig defaults."""

from __future__ import annotations

from pathlib import Path

from tool_recovery_lora.train.config import TrainConfig


def test_train_config_defaults() -> None:
    config = TrainConfig()
    assert config.base_model == "unsloth/Qwen2.5-3B-Instruct"
    assert config.lora_rank == 16
    assert config.lora_alpha == 32
    assert config.max_seq_length == 2048
    assert config.output_dir == Path("adapters/qwen25-3b-recovery")
