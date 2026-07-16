# Design: tool-recovery-lora

**Date:** 2026-07-15  
**Repo:** https://github.com/Brescou/tool-recovery-lora  
**Status:** Approved for implementation planning

## Goal

Fine-tune a small open-weight Instruct model with QLoRA so it can perform **agentic tool-calling with recovery** (repair after invalid JSON, wrong tool, or missing args), then run it inside a LangGraph demo hosted by `langgraph-agent-stack`.

Primary success criterion is **pedagogy**: close the PyTorch / fine-tuning gap while staying in the agentic domain. Beating a frontier model on cost/accuracy is explicitly **phase 2**, not v1.

## Context

| Asset | Role |
|-------|------|
| `langgraph-agent-stack` | Trace generator (error injection + export) and demo host |
| `tool-recovery-lora` (this repo) | Dataset curation, Unsloth QLoRA train, objective eval, LoRA adapters |
| Hardware | RTX 3080 10GB, 62GB RAM, Ryzen 7 5800X, CUDA 13 / driver 580 |
| Portfolio | Dark minimal docs, consistent with `brescou.github.io` |

## Decisions locked

| Decision | Choice |
|----------|--------|
| Deliverable v1 | Agentic demo (learning-first), not a publishable bake-off |
| Domain | Tool-call **recovery** on stack tools/packs |
| Coupling | **Medium**: stack = traces + demo; this repo = data/train/eval |
| Base model | Qwen Instruct ~3–4B (exact Hub ID fixed at scaffold via Unsloth support) |
| Train stack | Unsloth + QLoRA (not Axolotl) |
| Compute | Local 3080 only for v1 |
| Timeline | Open-ended exploration |
| Big-model comparison | Deferred to phase 5 |

## Approach

**Error mining from the stack** (not mass synthetic teacher data):

1. Run pack scenarios in `langgraph-agent-stack`.
2. Inject tool errors (invalid JSON, wrong tool name, missing required args).
3. Capture multi-turn trajectories where a teacher or repair path produces a correct follow-up tool call.
4. Export ChatML / messages JSONL into this repo.
5. Curate with a light curriculum, train QLoRA, serve adapter in a stack demo.

Light curriculum borrowed for pedagogy:

- Block A: correct single-turn tool calls (~60% of train)
- Block B: recovery after injected failure (~40%)
- Small multi-turn tail optional if volume allows

## Architecture

```
langgraph-agent-stack                         tool-recovery-lora
─────────────────────                         ──────────────────
packs + tools                                 data/raw, data/curated, data/eval
error injection + trace export  ──────────►   src/.../data (validate)
                                              src/.../train (Unsloth QLoRA)
local LoRA-compatible provider  ◄──────────   adapters/ (LoRA weights)
demo pack run                                 src/.../eval (objective metrics)
```

### Responsibilities

**This repo**

- Validate and version curated JSONL
- Train and log runs (loss, hyperparams)
- Objective smoke eval (`make eval`)
- Document learning notes (dark minimal)

**langgraph-agent-stack (minimal additive changes)**

- Trace export path for tool-calling conversations
- Controlled error injection for recovery trajectories
- Optional provider wiring so a pack can call the fine-tuned local model

### Non-goals (v1)

- Axolotl, multi-GPU, full-weight merge required for demo
- LLM-as-judge harness
- GPT/Claude bake-off and LinkedIn cost chart
- Dataset > ~2k examples
- Folding train code into `langgraph-agent-stack`

## Dataset

| Split | Purpose | Target size |
|-------|---------|-------------|
| Train | Curriculum mix (correct + recovery) | ~700–1000 |
| Val | Held-out same mix | ~100–150 |
| Smoke eval | Fixed cases for `make eval` | ~50 |
| **Total curated** | | **~800–1200** |

**Format:** message lists compatible with Unsloth / Qwen tool-calling (roles `system` / `user` / `assistant` / `tool`, structured `tool_calls` where applicable). Exact schema documented in `data/README.md` at scaffold time and validated by typed loaders + tests.

**Quality bar:**

- Every recovery example has a clear failure signal then a correct repair turn
- No train/eval leakage (eval ids disjoint)
- Raw exports may be large and gitignored; curated train/val preferably versioned (or Git LFS if heavy)

## Training

| Knob | v1 default |
|------|------------|
| Method | QLoRA via Unsloth |
| Rank / alpha | 16 / 32 (starting point; tune later) |
| Seq length | 2048–4096 depending on VRAM headroom |
| Artifact | LoRA adapter only under `adapters/` (gitignored) |
| Entry | `make train` |

Close desktop GPU consumers before long runs (Xorg/Cursor currently use ~3GB). Effective training budget ~8–9GB if the session is cleaned up.

## Evaluation (v1)

Objective only:

- Exact tool name match
- Args JSON parse + required-field / schema validity
- Recovery success: after an injected error turn, next assistant tool call is correct
- Local latency (ms) as informational metric

No LLM-as-judge in v1. `make eval` runs the ~50 smoke cases.

## Demo (v1)

- One stack pack configured to use the local fine-tuned model
- Visible scripted path: bad tool call → recovery → success
- Docs/screenshots dark-themed for portfolio consistency

## Repo layout

```
tool-recovery-lora/
├── src/tool_recovery_lora/
│   ├── data/
│   ├── train/
│   └── eval/
├── scripts/
├── data/
│   ├── raw/          # gitignored if large
│   ├── curated/
│   └── eval/
├── adapters/         # gitignored
├── docs/
│   └── superpowers/
│       └── specs/
├── tests/
├── Makefile          # test, train, eval
├── pyproject.toml    # uv, Python ≥3.10, type hints
└── README.md
```

Conventions: conventional commits, static typing, `make test`, Black-compatible style (88 cols, double quotes) for Python.

## Phases

| Phase | Focus | Done when |
|-------|--------|-----------|
| 0 | Scaffold + stack export stub | `make test` green, README |
| 1 | Curated curriculum dataset | train/val/eval JSONL ready |
| 2 | First Unsloth run | adapter + loss notes |
| 3 | Objective eval + iterate | stable smoke metrics |
| 4 | LangGraph recovery demo | live scenario |
| 5 | (Later) Claude/GPT bake-off + cost chart | LinkedIn-ready |

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| 10GB VRAM OOM | 3–4B only, QLoRA, shorter seq, close GUI GPU users |
| Weak recovery signal in traces | Explicit error injection + manual curation pass |
| Stack coupling drift | Keep export schema versioned; this repo owns validation |
| Scope creep into bake-off | Phase 5 gate; v1 success = demo + learning |

## Open items at scaffold time

- Exact Qwen Hub model ID (pick stable Unsloth-supported Instruct 3–4B)
- Trace JSON schema version string shared with stack export
- Which pack is the first demo target
