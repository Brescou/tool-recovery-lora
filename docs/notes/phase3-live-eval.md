# Phase 3 live eval notes

**Date:** 2026-07-15  
**Adapter:** `adapters/qwen25-3b-recovery`  
**Set:** `data/eval/smoke.jsonl` (50 examples, 20 recovery)

## Command

```bash
uv sync --extra dev --extra train
make eval       # fixture self-check (CPU)
make eval-live  # LoRA generation (GPU)
```

## Results

| Metric | Value |
|--------|-------|
| name_match | **1.00** |
| args_json_valid | **1.00** |
| core_args_exact (company/person/meeting_goal) | **1.00** |
| args_exact (incl. optional context) | 0.56 |
| recovery_core_exact | **1.00** (20/20) |
| recovery_exact | 0.75 |
| latency_ms_mean | ~2047 |
| latency_ms_p50 | ~2073 |

## Takeaways

1. The LoRA reliably emits valid `<tool_call>` JSON and the correct `meeting_prep` tool.
2. Core meeting fields are exact on the full smoke set, including recovery trajectories.
3. Strict `args_exact` is dragged down by optional `context`: the Phase 1 generator sometimes put context in labels without putting it in the user text. Generator fixed for future regenerations (`context` always surfaced in the user turn when non-empty). Re-`make dataset` + retrain will tighten full exact match.
4. Model still emits arguments as a JSON **string** inside the tool_call object (training template artifact); the parser accepts both string and object forms.

## Next (Phase 4)

Wire the adapter into `langgraph-agent-stack` as a local provider and demo a visible recovery path.
