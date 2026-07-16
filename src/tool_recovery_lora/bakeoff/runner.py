"""Compare LoRA live eval vs frontier prompting baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tool_recovery_lora.bakeoff.claude_baseline import (
    load_dotenv_files,
    run_claude_eval,
)
from tool_recovery_lora.data.loader import load_jsonl
from tool_recovery_lora.data.schema import TraceExample
from tool_recovery_lora.eval.live_runner import run_live_eval


def _summary_row(label: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": label,
        "model": result.get("model") or result.get("adapter_dir") or label,
        "n_examples": result["n_examples"],
        "name_match": result["name_match"],
        "core_args_exact": result["core_args_exact"],
        "args_exact": result["args_exact"],
        "recovery_core_exact": result.get("recovery_core_exact"),
        "latency_ms_mean": result["latency_ms_mean"],
        "cost_usd_total": result.get("cost_usd_total", 0.0),
        "cost_usd_per_request": result.get("cost_usd_per_request", 0.0),
    }


def render_accuracy_cost_svg(rows: list[dict[str, Any]], path: Path) -> None:
    """Write a minimal dark SVG scatter: core_args_exact vs cost/request."""
    plotted = [
        row
        for row in rows
        if row.get("core_args_exact") is not None
        and row.get("cost_usd_per_request") is not None
    ]
    if not plotted:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="120">'
            '<rect width="100%" height="100%" fill="#0b0f14"/>'
            '<text x="24" y="64" fill="#e5e7eb" font-size="16">'
            "No bake-off points to plot yet.</text></svg>\n",
            encoding="utf-8",
        )
        return

    width, height = 720, 420
    margin = 60
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    max_cost = max(float(row["cost_usd_per_request"]) for row in plotted)
    max_cost = max(max_cost, 1e-6)

    def x_pos(cost: float) -> float:
        return margin + (cost / max_cost) * plot_w

    def y_pos(acc: float) -> float:
        return margin + (1.0 - acc) * plot_h

    points: list[str] = []
    labels: list[str] = []
    colors = ["#5eead4", "#fbbf24", "#f472b6"]
    for index, row in enumerate(plotted):
        color = colors[index % len(colors)]
        cx = x_pos(float(row["cost_usd_per_request"]))
        cy = y_pos(float(row["core_args_exact"]))
        points.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="10" fill="{color}" />'
        )
        label = (
            f'{row["label"]} ({row["core_args_exact"]:.0%} @ '
            f'${row["cost_usd_per_request"]:.4f})'
        )
        labels.append(
            f'<text x="{cx + 14:.1f}" y="{cy + 4:.1f}" fill="#e5e7eb" '
            f'font-family="IBM Plex Sans, sans-serif" font-size="14">'
            f"{label}</text>"
        )

    mid_y = margin + plot_h / 2
    svg_parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b0f14"/>',
        f'<text x="{margin}" y="32" fill="#e5e7eb" '
        'font-family="IBM Plex Sans, sans-serif" '
        'font-size="18" font-weight="600">',
        "Tool-call core accuracy vs cost / request",
        "</text>",
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" '
        f'y2="{height - margin}" stroke="#374151" />',
        f'<line x1="{margin}" y1="{height - margin}" '
        f'x2="{width - margin}" y2="{height - margin}" '
        'stroke="#374151" />',
        f'<text x="16" y="{mid_y}" fill="#9ca3af" font-size="12" '
        f'transform="rotate(-90 16,{mid_y})">'
        "core_args_exact</text>",
        f'<text x="{width / 2}" y="{height - 16}" fill="#9ca3af" '
        'font-size="12" text-anchor="middle">USD / request</text>',
        *points,
        *labels,
        "</svg>",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg_parts), encoding="utf-8")


def run_bakeoff(
    *,
    smoke_path: Path,
    adapter_dir: Path,
    out_dir: Path,
    limit: int | None = None,
    skip_lora: bool = False,
    lora_results_path: Path | None = None,
    claude_model: str = "claude-sonnet-4-20250514",
) -> dict[str, Any]:
    """Run LoRA + Claude bake-off and write artifacts."""
    load_dotenv_files(
        [
            Path("langgraph-agent-stack") / ".env",
            Path("..") / "langgraph-agent-stack" / ".env",
            Path("/home/brescou/Project/langgraph-agent-stack/.env"),
            Path(".env"),
        ]
    )
    examples = load_jsonl(smoke_path)
    if limit is not None:
        examples = examples[:limit]

    if skip_lora and lora_results_path and lora_results_path.is_file():
        lora = json.loads(lora_results_path.read_text(encoding="utf-8"))
        lora["cost_usd_total"] = 0.0
        lora["cost_usd_per_request"] = 0.0
        lora["model"] = str(adapter_dir)
    else:
        lora = run_live_eval(
            smoke_path,
            adapter_dir=adapter_dir,
            limit=limit,
        )
        lora["cost_usd_total"] = 0.0
        lora["cost_usd_per_request"] = 0.0
        lora["model"] = str(adapter_dir)

    claude_examples: list[TraceExample] = examples
    claude: dict[str, Any]
    try:
        claude = run_claude_eval(claude_examples, model=claude_model)
    except Exception as exc:  # noqa: BLE001 — surface auth/network in artifact
        claude = {
            "provider": "anthropic",
            "model": claude_model,
            "status": "skipped",
            "error": str(exc),
            "n_examples": float(len(claude_examples)),
            "name_match": None,
            "args_json_valid": None,
            "args_exact": None,
            "core_args_exact": None,
            "recovery_exact": None,
            "recovery_core_exact": None,
            "n_recovery": None,
            "latency_ms_mean": None,
            "latency_ms_p50": None,
            "cost_usd_total": None,
            "cost_usd_per_request": None,
            "details": [],
        }

    rows = [_summary_row("LoRA Qwen2.5-3B", lora)]
    if claude.get("status") != "skipped":
        rows.append(_summary_row("Claude (tools API)", claude))
    else:
        rows.append(
            {
                "label": "Claude (skipped)",
                "model": claude_model,
                "n_examples": claude["n_examples"],
                "name_match": None,
                "core_args_exact": None,
                "args_exact": None,
                "recovery_core_exact": None,
                "latency_ms_mean": None,
                "cost_usd_total": None,
                "cost_usd_per_request": None,
                "error": claude.get("error"),
            }
        )
    payload = {
        "rows": rows,
        "lora": {key: value for key, value in lora.items() if key != "details"},
        "claude": {
            key: value for key, value in claude.items() if key != "details"
        },
        "claude_details": claude.get("details"),
        "lora_details": lora.get("details"),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "bakeoff_results.json"
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    render_accuracy_cost_svg(rows, out_dir / "accuracy_vs_cost.svg")
    return {
        "n_examples": float(len(examples)),
        "rows": rows,
        "results_path": str(results_path),
    }
