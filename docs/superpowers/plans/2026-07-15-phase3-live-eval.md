# Phase 3 — Live objective eval Implementation Plan

**Goal:** Score the fine-tuned adapter on smoke JSONL with real generation (not fixture self-check).

**Approach:** Build prompt from messages before the gold assistant turn → Unsloth generate → parse `<tool_call>` / JSON → score name/args. Keep `make eval` as fixture mode for CPU CI; add `make eval-live`.
