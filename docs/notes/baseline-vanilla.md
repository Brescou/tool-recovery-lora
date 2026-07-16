# Baseline vanilla vs LoRA

**Date:** 2026-07-16  
**Set:** `data/eval/smoke.jsonl` (50 examples, 20 recovery)  
**Harness:** same live eval + Qwen `<tool_call>` parser as Phase 3

## Commands

```bash
make eval-vanilla   # base model, no LoRA
make eval-live      # LoRA adapter
```

Equivalent CLI:

```bash
uv run python scripts/run_eval.py --mode live --no-adapter \
  --model unsloth/Qwen2.5-3B-Instruct \
  --out data/eval/vanilla_results.json
```

## Side-by-side metrics

| Metric | Vanilla (no adapter) | LoRA (`qwen25-3b-recovery`) |
|--------|----------------------|-----------------------------|
| model | `unsloth/Qwen2.5-3B-Instruct` | base + `adapters/qwen25-3b-recovery` |
| n_examples | 50 | 50 |
| name_match | **0.04** | **1.00** |
| args_json_valid | **0.04** | **1.00** |
| core_args_exact | **0.04** | **1.00** |
| args_exact | **0.00** | 0.56 |
| recovery_core_exact | **0.10** (2/20) | **1.00** (20/20) |
| recovery_exact | **0.00** | 0.75 |
| latency_ms_mean | ~3979 | ~2047 |
| latency_ms_p50 | ~3865 | ~2073 |

Vanilla source: `data/eval/vanilla_results.json` (2026-07-16).  
LoRA source: Phase 3 live eval ([phase3-live-eval.md](phase3-live-eval.md)).

## Takeaways

1. **The LoRA is doing the work.** Without it, the base Instruct model almost never emits a parseable `meeting_prep` tool call on this harness (4% name/core match).
2. Recovery is near-zero vanilla (`recovery_core_exact` 10%) vs perfect with LoRA (100%).
3. Vanilla is also slower (~4.0 s vs ~2.0 s mean) — it tends to ramble instead of emitting a tight `<tool_call>` block.
4. This baseline is the blocking control for claiming fine-tune gains before any Claude bake-off narrative.

## Artifacts

- Vanilla JSON: `data/eval/vanilla_results.json` (gitignored)
- LoRA JSON: `data/eval/live_results.json` (gitignored)
