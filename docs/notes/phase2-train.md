# Phase 2 training notes

**Date:** 2026-07-15  
**Hardware:** RTX 3080 10GB (desktop ~3.7GB reserved → ~6GB free)  
**Base:** `unsloth/Qwen2.5-3B-Instruct`  
**Method:** QLoRA 4-bit, rank 16, alpha 32, Unsloth 2026.7.3

## Config

| Knob | Value |
|------|-------|
| max_seq_length | 1024 (fit 3080 with desktop load) |
| batch / accum | 1 / 8 |
| lr | 2e-4 |
| epochs | 1 |
| train size | 930 |
| val size | 120 |

## Smoke (20 steps)

- Loss: **4.81 → 0.66**
- Runtime: ~54s
- Output: `adapters/qwen25-3b-recovery-smoke/`

## Full run (1 epoch)

- Steps: **117**
- Train loss (avg): **0.444**
- Eval loss: **0.100**
- Runtime: ~5.0 min
- Output: `adapters/qwen25-3b-recovery/`

Loss dropped quickly in the first ~20% of the epoch, then plateaued near ~0.10 — expected for a short, highly structured synthetic curriculum.

## Commands

```bash
uv sync --extra dev --extra train
make train                          # full epoch
uv run python scripts/run_train.py --smoke
```

## Next (Phase 3)

Wire live model inference into `make eval` (load base + adapter, score smoke JSONL objectively) instead of fixture self-consistency only.
