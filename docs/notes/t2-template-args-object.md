# T2 — Template arguments object + dataset regen

**Date:** 2026-07-16

## Problem

Training used `json.dumps(arguments)` before the Qwen chat template's `| tojson`.
That double-encoded args, so the LoRA learned:

```text
{"name": "meeting_prep", "arguments": "{\"company\": \"...\"}"}
```

instead of:

```text
{"name": "meeting_prep", "arguments": {"company": "..."}}
```

## Fixes

1. **`format_messages._tool_call_to_hf`** — pass `arguments` as a `dict` (JSON object).
2. **Parser** — still tolerates string args for old checkpoints; live eval now reports **`args_as_object_rate`**.
3. **`make dataset`** — regenerated with context always surfaced in the user turn when non-empty (Phase 3 fix).

## Commands

```bash
make dataset
make test
# after retrain:
make eval-live   # expect args_as_object_rate → 1.0
```

## Note

Existing adapter `adapters/qwen25-3b-recovery` was trained with the string form.
Retrain (Phase 2 again) is required for the model to emit object args natively.
