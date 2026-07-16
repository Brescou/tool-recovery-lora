# Dataset format

Traces use schema version **`tool_recovery_trace_v1`** (one JSON object per line).

## Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Stable unique id (no train/eval overlap) |
| `schema_version` | string | Must be `tool_recovery_trace_v1` |
| `kind` | `"correct"` \| `"recovery"` | Curriculum block |
| `messages` | array | Conversation turns |
| `meta` | object \| null | Pack id, error type, `expected_tool_call`, etc. |

### Message

| Field | Type | Notes |
|-------|------|-------|
| `role` | `system` \| `user` \| `assistant` \| `tool` | |
| `content` | string \| null | |
| `tool_calls` | array \| null | Assistant structured calls |
| `tool_call_id` | string \| null | For `tool` role |
| `name` | string \| null | Tool name on `tool` role |

### Tool call

```json
{"name": "meeting_prep", "arguments": {"company": "Acme", "person": "Jane", "meeting_goal": "discovery"}}
```

## Curriculum targets (Phase 1)

| Split | Path | Size |
|-------|------|------|
| Train | `data/curated/train.jsonl` | ~700–1000 (~60% correct / ~40% recovery) |
| Val | `data/curated/val.jsonl` | ~100–150 |
| Smoke | `data/eval/smoke.jsonl` | ~50 (3 fixtures shipped in Phase 0) |

## Directories

- `raw/` — stack exports (gitignored)
- `curated/` — reviewed JSONL for training
- `eval/` — held-out smoke / regression set
