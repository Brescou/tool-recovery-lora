"""Load and validate tool-recovery JSONL datasets."""

from __future__ import annotations

import json
from pathlib import Path

from tool_recovery_lora.data.schema import TraceExample


def load_jsonl(path: Path) -> list[TraceExample]:
    """Load a JSONL file of ``TraceExample`` rows.

    Args:
        path: Path to a ``.jsonl`` file.

    Returns:
        Parsed and validated examples in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a line is invalid JSON or fails schema validation.
    """
    if not path.is_file():
        raise FileNotFoundError(path)

    examples: list[TraceExample] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON ({exc})"
                ) from exc
            try:
                examples.append(TraceExample.model_validate(payload))
            except Exception as exc:
                raise ValueError(
                    f"{path}:{line_number}: schema validation failed ({exc})"
                ) from exc
    return examples


def validate_no_id_overlap(
    train: list[TraceExample],
    eval_set: list[TraceExample],
) -> None:
    """Raise if any example id appears in both train and eval.

    Args:
        train: Training examples.
        eval_set: Evaluation examples.

    Raises:
        ValueError: When overlapping ids are found.
    """
    train_ids = {example.id for example in train}
    eval_ids = {example.id for example in eval_set}
    overlap = train_ids & eval_ids
    if overlap:
        sorted_ids = ", ".join(sorted(overlap))
        raise ValueError(f"train/eval id overlap: {sorted_ids}")
