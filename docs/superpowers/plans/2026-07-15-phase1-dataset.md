# Phase 1 — Curated curriculum dataset Implementation Plan

> **For agentic workers:** Execute inline. Checkbox tracking optional.

**Goal:** Produce curated `train.jsonl` / `val.jsonl` (~800–1200 total) plus ~50 smoke eval examples for `meeting_prep` tool-call recovery, grounded in the real stack input schema.

**Architecture:** Deterministic synthetic generator in this repo (seeded) that emits `tool_recovery_trace_v1` traces. Uses `MeetingPrepInput` fields (`company`, `person`, `meeting_goal`, `context`) as the tool signature. Stack-side live export remains Phase 1.5 / later; this phase honors the export contract shape.

**Tech Stack:** Existing package + stdlib; no Unsloth yet.

## Global Constraints

- Schema `tool_recovery_trace_v1`
- ~60% `correct` / ~40% `recovery`
- Recovery types: `missing_args`, `wrong_tool`, `invalid_json`
- No train/eval id overlap
- Reproducible via `--seed`

---

### Task 1: Generator module + tests
### Task 2: CLI `scripts/generate_dataset.py` + `make dataset`
### Task 3: Generate curated files + expand smoke to ~50
### Task 4: Verify `make test` / `make eval` + commit
