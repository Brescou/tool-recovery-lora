# Phase 2 — First Unsloth QLoRA run Implementation Plan

**Goal:** Wire real QLoRA training with Unsloth on `unsloth/Qwen2.5-3B-Instruct`, save adapters under `adapters/qwen25-3b-recovery`, document loss notes.

**Defaults (3080 10GB):** seq 2048 (fall back 1024 if OOM), batch 1, grad_accum 8, lr 2e-4, 1 epoch, rank 16 / alpha 32.

**Tasks:** format messages → train runner → CLI flags (`--smoke`) → install train extra → smoke then full run.
