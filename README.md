# tool-recovery-lora

QLoRA fine-tuning for **agentic tool-call recovery** — small open-weight model, LangGraph demo via [`langgraph-agent-stack`](https://github.com/Brescou/langgraph-agent-stack).

Learning-first project: close the PyTorch / fine-tuning gap while staying in agentic MLOps. Bake-off vs frontier models is phase 5, not v1.

## Stack

| Piece | Choice |
|-------|--------|
| Base model | `unsloth/Qwen2.5-3B-Instruct` |
| Method | QLoRA (rank 16 / alpha 32) via Unsloth — Phase 2+ |
| Hardware | RTX 3080 10GB (local) |
| Trace schema | `tool_recovery_trace_v1` |
| Demo pack | `meeting_prep` |

## Quick start

```bash
uv sync --extra dev --extra train
make test
make dataset   # regenerate train/val/smoke JSONL
make eval      # fixture self-check
make eval-live # LoRA generation on smoke set (GPU)
make train     # Unsloth QLoRA on RTX 3080
```

## Layout

```
src/tool_recovery_lora/   # data · train · eval
data/eval/smoke.jsonl     # objective smoke fixtures
data/curated/             # train/val JSONL (Phase 1)
adapters/                 # LoRA weights (gitignored)
docs/                     # design + plans + export contract
```

## Docs

- [Design spec](docs/superpowers/specs/2026-07-15-tool-recovery-lora-design.md)
- [Phase 0 plan](docs/superpowers/plans/2026-07-15-phase0-scaffold.md)
- [Stack export contract](docs/stack-export-contract.md)
- [Dataset format](data/README.md)

## Phases

0. Scaffold ← done  
1. Curated curriculum dataset ← done  
2. First Unsloth QLoRA run ← done ([notes](docs/notes/phase2-train.md))  
3. Objective eval iteration (live model) ← **done** ([notes](docs/notes/phase3-live-eval.md))  
4. LangGraph recovery demo  
5. Optional Claude/GPT bake-off  

## License

MIT
