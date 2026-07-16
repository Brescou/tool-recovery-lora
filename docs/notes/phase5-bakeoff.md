# Phase 5 — Bake-off notes

**Date:** 2026-07-15

## Command

```bash
uv sync --extra train --extra bakeoff
export ANTHROPIC_API_KEY=sk-ant-...   # real key required
make bakeoff
# or reuse LoRA aggregates:
uv run python scripts/run_bakeoff.py --skip-lora
```

Artifacts: `data/eval/bakeoff/bakeoff_results.json`, `accuracy_vs_cost.svg`  
Committed chart stub: [`docs/assets/accuracy_vs_cost.svg`](../assets/accuracy_vs_cost.svg)

## Results (2026-07-15)

| System | core_args_exact | args_exact | latency mean | USD / req |
|--------|-----------------|------------|--------------|-----------|
| **LoRA Qwen2.5-3B** (local 3080) | **100%** | 56% | ~2.05 s | **$0** (no API) |
| Claude Sonnet (tools API) | *skipped* | — | — | — |

Claude was **not** measured: `langgraph-agent-stack/.env` still has a placeholder `ANTHROPIC_API_KEY` (`your-an…`), which returns HTTP 401. The bake-off harness is wired and will fill Claude metrics + the cost chart as soon as a real key is exported.

## What the harness does

1. LoRA: same smoke set as Phase 3 (`make eval-live` path)
2. Claude: native Anthropic **tools API** (prompting, no fine-tune) on identical recovery/correct traces
3. Costs: Claude from token usage × `$0.003 / $0.015` per 1k in/out; LoRA counted as `$0` API cost
4. SVG: dark scatter **core accuracy vs USD/request** for LinkedIn

## GPT

No `OPENAI_API_KEY` on this machine. Optional follow-up: mirror `claude_baseline.py` for OpenAI tools.

## Next

```bash
export ANTHROPIC_API_KEY=...   # real key
make bakeoff                   # full LoRA + Claude comparison
```
