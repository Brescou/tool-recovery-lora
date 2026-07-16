# Phase 0 — Repo scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline) or superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold `tool-recovery-lora` so `make test` / `make eval` work with typed trace loaders, fixture data, and train/eval stubs — no Unsloth GPU run yet.

**Architecture:** Package `tool_recovery_lora` owns schema validation (`tool_recovery_trace_v1`), JSONL I/O, objective eval metrics, and train config stubs. Heavy deps (torch/unsloth) stay behind an optional extra. Stack export is documented as a contract; glue lives in `scripts/`.

**Tech Stack:** Python ≥3.12, uv, hatchling, pydantic v2, pytest, ruff; optional `train` extra for Unsloth later.

## Global Constraints

- Python ≥3.12, type hints, Black-compatible (88 cols, double quotes) via ruff
- Conventional commits; `make test` / `make train` / `make eval`
- Base model ID locked: `unsloth/Qwen2.5-3B-Instruct`
- Trace schema version: `tool_recovery_trace_v1`
- First demo pack target: `meeting_prep` (in `langgraph-agent-stack`)
- No bake-off / LLM-as-judge in Phase 0
- Adapters and raw dumps gitignored

---

### Task 1: Project skeleton (pyproject, Makefile, gitignore, package layout)

**Files:**
- Create: `pyproject.toml`, `Makefile`, `.gitignore`, `README.md`
- Create: `src/tool_recovery_lora/__init__.py`
- Create: `data/raw/.gitkeep`, `data/curated/.gitkeep`, `data/eval/.gitkeep`, `adapters/.gitkeep`
- Create: `scripts/.gitkeep`

**Interfaces:**
- Produces: installable package `tool-recovery-lora`, `make test|eval|train|lint`

- [ ] **Step 1: Write `pyproject.toml`** with core deps (pydantic, pytest, ruff) and optional `[project.optional-dependencies] train` placeholder list for unsloth/torch (not required for Phase 0 tests).
- [ ] **Step 2: Write `Makefile`** mirroring stack style: `install`, `test`, `eval`, `train`, `lint`, `format`.
- [ ] **Step 3: Write `.gitignore`** covering `adapters/**`, `data/raw/**`, `.venv`, `__pycache__`, `*.pt`, wandb, etc. Keep `data/curated` and `data/eval` fixtures tracked.
- [ ] **Step 4: Commit** `chore: scaffold project layout`

---

### Task 2: Trace schema + JSONL loader (TDD)

**Files:**
- Create: `src/tool_recovery_lora/data/schema.py`
- Create: `src/tool_recovery_lora/data/loader.py`
- Create: `src/tool_recovery_lora/data/__init__.py`
- Create: `tests/test_schema.py`, `tests/test_loader.py`
- Create: `data/eval/smoke.jsonl` (3–5 fixture examples)

**Interfaces:**
- Produces:
  - `TRACE_SCHEMA_VERSION = "tool_recovery_trace_v1"`
  - `ToolCall(name: str, arguments: dict[str, Any])`
  - `Message(role: Literal["system","user","assistant","tool"], content: str | None, tool_calls: list[ToolCall] | None, tool_call_id: str | None, name: str | None)`
  - `TraceExample(id: str, schema_version: str, kind: Literal["correct","recovery"], messages: list[Message], meta: dict[str, Any] | None)`
  - `load_jsonl(path: Path) -> list[TraceExample]`
  - `validate_no_id_overlap(train: list[TraceExample], eval_set: list[TraceExample]) -> None` (raises `ValueError`)

- [ ] **Step 1: Write failing tests** for schema parse + loader + id overlap.
- [ ] **Step 2: Implement schema + loader.**
- [ ] **Step 3: Add smoke fixtures** covering one `correct` and one `recovery` trajectory.
- [ ] **Step 4: `make test` green; commit** `feat: add tool_recovery_trace_v1 schema and loader`

---

### Task 3: Objective eval stub

**Files:**
- Create: `src/tool_recovery_lora/eval/metrics.py`
- Create: `src/tool_recovery_lora/eval/runner.py`
- Create: `src/tool_recovery_lora/eval/__init__.py`
- Create: `tests/test_metrics.py`
- Create: `scripts/run_eval.py`

**Interfaces:**
- Consumes: `TraceExample`, `ToolCall`
- Produces:
  - `extract_last_assistant_tool_call(messages) -> ToolCall | None`
  - `score_tool_call(predicted: ToolCall | None, expected: ToolCall) -> dict[str, bool]`
    keys: `name_match`, `args_json_valid`, `args_exact`
  - `run_smoke_eval(path: Path) -> dict[str, float]` — for Phase 0, scores **expected self-consistency** of fixtures (last assistant tool call vs `meta["expected_tool_call"]`), not a live model

- [ ] **Step 1: Failing tests for metrics.**
- [ ] **Step 2: Implement metrics + runner.**
- [ ] **Step 3: Wire `make eval` → `uv run python scripts/run_eval.py`.**
- [ ] **Step 4: Commit** `feat: add objective smoke eval harness`

---

### Task 4: Train config stub (no GPU)

**Files:**
- Create: `src/tool_recovery_lora/train/config.py`
- Create: `src/tool_recovery_lora/train/__init__.py`
- Create: `scripts/run_train.py`
- Create: `tests/test_train_config.py`

**Interfaces:**
- Produces: `TrainConfig` pydantic model with defaults:
  - `base_model: str = "unsloth/Qwen2.5-3B-Instruct"`
  - `lora_rank: int = 16`, `lora_alpha: int = 32`
  - `max_seq_length: int = 2048`
  - `output_dir: Path = Path("adapters/qwen25-3b-recovery")`
- `scripts/run_train.py` exits with clear message if `train` extra missing; does not download models in Phase 0.

- [ ] **Step 1: Tests for TrainConfig defaults.**
- [ ] **Step 2: Implement config + stub CLI.**
- [ ] **Step 3: Commit** `feat: add TrainConfig stub for Unsloth QLoRA`

---

### Task 5: Docs + stack export contract

**Files:**
- Create: `data/README.md`
- Modify: `README.md` (dark-portfolio friendly, concise)
- Create: `docs/stack-export-contract.md`

**Interfaces:**
- Documents JSONL fields, curriculum kinds, and how `langgraph-agent-stack` should export traces (version `tool_recovery_trace_v1`, pack `meeting_prep`).

- [ ] **Step 1: Write data + export contract docs.**
- [ ] **Step 2: Polish root README** with goals, make targets, hardware note.
- [ ] **Step 3: Commit** `docs: add README and stack export contract`

---

## Spec coverage (self-review)

| Spec item | Task |
|-----------|------|
| Repo layout | Task 1 |
| Schema / ChatML messages | Task 2 |
| `make test` / `make eval` / `make train` | Tasks 1, 3, 4 |
| Model ID Qwen 3B Unsloth | Task 4 |
| Objective metrics (name, JSON, recovery meta) | Task 3 |
| Stack coupling / export | Task 5 |
| No Unsloth install required for CI-like `make test` | Tasks 1–3 |
| Adapters gitignored | Task 1 |

Phase 1+ (real dataset, GPU train, demo wiring) are **out of scope** for this plan.
