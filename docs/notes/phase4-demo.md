# Phase 4 — LangGraph recovery demo

**Date:** 2026-07-15

## What this is

A local LangGraph demo that **injects a bad tool call**, surfaces the tool error, then lets the fine-tuned LoRA **repair** it into a valid `meeting_prep` call.

This is intentional: the LoRA was trained for tool-calling recovery, not for `MeetingPrepOutput` JSON. Pointing `StructuredLLMPack` at the LoRA would be the wrong contract.

## Run

```bash
uv sync --extra train
make demo
# or:
uv run python scripts/run_demo.py --scenario wrong_tool
uv run python scripts/run_demo.py --list
```

## Graph

```
load_scenario → repair_with_lora → validate → END
```

## Stack coupling (next)

Later, a stack agent node can treat domain packs as tools and call this adapter via an OpenAI-compatible local server. Phase 4 keeps the demo in-repo so the learning loop stays focused.
