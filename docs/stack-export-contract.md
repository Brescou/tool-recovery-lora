# Stack export contract

How [`langgraph-agent-stack`](https://github.com/Brescou/langgraph-agent-stack) should emit traces for this repo.

## Version

- **Schema:** `tool_recovery_trace_v1`
- **Consumer:** `tool_recovery_lora.data.load_jsonl`
- **First demo pack:** `meeting_prep`

## Export shape

One JSON object per line (JSONL), fields as in [`data/README.md`](../data/README.md).

Minimum `meta` for recovery examples:

```json
{
  "pack_id": "meeting_prep",
  "error_type": "missing_args",
  "expected_tool_call": {
    "name": "meeting_prep",
    "arguments": {
      "company": "Globex",
      "person": "Bob",
      "meeting_goal": "partnership"
    }
  }
}
```

`error_type` values used in fixtures: `missing_args`, `wrong_tool`, `invalid_json` (extend as needed).

## Error injection (producer side)

For recovery trajectories the stack should:

1. Run a normal pack scenario.
2. Force a first assistant tool call to fail (`missing_args` / `wrong_tool` / `invalid_json`).
3. Emit the tool error as a `role=tool` message.
4. Capture the repaired assistant `tool_calls` turn.
5. Set `meta.expected_tool_call` to the **repaired** call.

Correct examples omit the error turn; `kind` is `"correct"`.

## Drop location

Write exports under this repo's `data/raw/` (gitignored). Curate into `data/curated/` before training.

## Status

Phase 4 demo lives in this repo (`make demo`). Stack-side exporter / pack-as-tool agent node remains optional follow-up.
