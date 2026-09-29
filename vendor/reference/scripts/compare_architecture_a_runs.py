"""Compare two Architecture-A JSON runs and select a winner.

Winner rule:
1) Higher overall accuracy
2) Fewer total failures (failed_evidence + skipped_questions)
3) Lower average latency
4) Lower estimated Exa cost
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_payload(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _summary_metrics(payload: dict[str, Any]) -> dict[str, float | int | None]:
    summary = payload.get("summary", {})
    skipped = summary.get("skipped_questions")
    failed = summary.get("failed_evidence")
    total_failures = None
    if isinstance(skipped, int) and isinstance(failed, int):
        total_failures = skipped + failed
    return {
        "accuracy": summary.get("accuracy"),
        "total_failures": total_failures,
        "average_latency_ms": summary.get("average_latency_ms"),
        "estimated_exa_cost_usd": summary.get("estimated_exa_cost_usd"),
        "processed_questions": summary.get("processed_questions"),
    }


def _build_correctness_map(payload: dict[str, Any]) -> dict[str, bool]:
    rows = payload.get("results", [])
    mapping: dict[str, bool] = {}
    for row in rows:
        row_uid = row.get("row_uid")
        is_correct = row.get("is_correct")
        if isinstance(row_uid, str) and isinstance(is_correct, bool):
            mapping[row_uid] = is_correct
    return mapping


def _compare_runs(
    fulltext: dict[str, Any],
    highlights: dict[str, Any],
) -> dict[str, Any]:
    full_metrics = _summary_metrics(fulltext)
    high_metrics = _summary_metrics(highlights)

    def _as_high(v: float | int | None) -> float:
        return float(v) if v is not None else float("-inf")

    def _as_low(v: float | int | None) -> float:
        return float(v) if v is not None else float("inf")

    tie_break_trace: list[str] = []
    winner = "tie"
    criterion = "none"

    full_acc = _as_high(full_metrics["accuracy"])
    high_acc = _as_high(high_metrics["accuracy"])
    if full_acc != high_acc:
        winner = "full_text" if full_acc > high_acc else "highlights"
        criterion = "accuracy"
        tie_break_trace.append(
            f"accuracy: full_text={full_metrics['accuracy']} vs highlights={high_metrics['accuracy']}"
        )
    else:
        tie_break_trace.append(
            f"accuracy tie at {full_metrics['accuracy']}"
        )
        full_fail = _as_low(full_metrics["total_failures"])
        high_fail = _as_low(high_metrics["total_failures"])
        if full_fail != high_fail:
            winner = "full_text" if full_fail < high_fail else "highlights"
            criterion = "fewer_failures"
            tie_break_trace.append(
                f"total_failures: full_text={full_metrics['total_failures']} vs highlights={high_metrics['total_failures']}"
            )
        else:
            tie_break_trace.append(
                f"failure tie at {full_metrics['total_failures']}"
            )
            full_lat = _as_low(full_metrics["average_latency_ms"])
            high_lat = _as_low(high_metrics["average_latency_ms"])
            if full_lat != high_lat:
                winner = "full_text" if full_lat < high_lat else "highlights"
                criterion = "lower_latency"
                tie_break_trace.append(
                    f"average_latency_ms: full_text={full_metrics['average_latency_ms']} vs highlights={high_metrics['average_latency_ms']}"
                )
            else:
                tie_break_trace.append(
                    f"latency tie at {full_metrics['average_latency_ms']}"
                )
                full_cost = _as_low(full_metrics["estimated_exa_cost_usd"])
                high_cost = _as_low(high_metrics["estimated_exa_cost_usd"])
                if full_cost != high_cost:
                    winner = "full_text" if full_cost < high_cost else "highlights"
                    criterion = "lower_cost"
                    tie_break_trace.append(
                        f"estimated_exa_cost_usd: full_text={full_metrics['estimated_exa_cost_usd']} vs highlights={high_metrics['estimated_exa_cost_usd']}"
                    )
                else:
                    tie_break_trace.append(
                        f"cost tie at {full_metrics['estimated_exa_cost_usd']}"
                    )

    full_map = _build_correctness_map(fulltext)
    high_map = _build_correctness_map(highlights)
    shared_uids = sorted(set(full_map).intersection(set(high_map)))
    full_better = sum(1 for uid in shared_uids if full_map[uid] and not high_map[uid])
    high_better = sum(1 for uid in shared_uids if high_map[uid] and not full_map[uid])
    equal = len(shared_uids) - full_better - high_better

    return {
        "winner": winner,
        "deciding_criterion": criterion,
        "tie_break_trace": tie_break_trace,
        "full_text_metrics": full_metrics,
        "highlights_metrics": high_metrics,
        "head_to_head": {
            "shared_rows": len(shared_uids),
            "full_text_better_rows": full_better,
            "highlights_better_rows": high_better,
            "equal_rows": equal,
        },
        "deltas": {
            "accuracy": (
                (full_metrics["accuracy"] or 0.0) - (high_metrics["accuracy"] or 0.0)
                if full_metrics["accuracy"] is not None and high_metrics["accuracy"] is not None
                else None
            ),
            "total_failures": (
                (full_metrics["total_failures"] or 0) - (high_metrics["total_failures"] or 0)
                if full_metrics["total_failures"] is not None and high_metrics["total_failures"] is not None
                else None
            ),
            "average_latency_ms": (
                (full_metrics["average_latency_ms"] or 0.0)
                - (high_metrics["average_latency_ms"] or 0.0)
                if full_metrics["average_latency_ms"] is not None
                and high_metrics["average_latency_ms"] is not None
                else None
            ),
            "estimated_exa_cost_usd": (
                (full_metrics["estimated_exa_cost_usd"] or 0.0)
                - (high_metrics["estimated_exa_cost_usd"] or 0.0)
                if full_metrics["estimated_exa_cost_usd"] is not None
                and high_metrics["estimated_exa_cost_usd"] is not None
                else None
            ),
        },
    }


def _write_markdown(path: str, comparison: dict[str, Any]) -> None:
    winner = comparison["winner"]
    criterion = comparison["deciding_criterion"]
    full = comparison["full_text_metrics"]
    high = comparison["highlights_metrics"]
    h2h = comparison["head_to_head"]

    lines = [
        "# A/B Comparison: Full Text vs Highlights",
        "",
        f"- Winner: **{winner}**",
        f"- Deciding criterion: **{criterion}**",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Full Text | Highlights |",
        "|---|---:|---:|",
        f"| Accuracy | {full['accuracy']} | {high['accuracy']} |",
        f"| Total failures | {full['total_failures']} | {high['total_failures']} |",
        f"| Avg latency (ms) | {full['average_latency_ms']} | {high['average_latency_ms']} |",
        f"| Estimated Exa cost (USD) | {full['estimated_exa_cost_usd']} | {high['estimated_exa_cost_usd']} |",
        "",
        "## Head-to-Head (shared row_uid)",
        "",
        f"- Shared rows: {h2h['shared_rows']}",
        f"- Full text better rows: {h2h['full_text_better_rows']}",
        f"- Highlights better rows: {h2h['highlights_better_rows']}",
        f"- Equal rows: {h2h['equal_rows']}",
        "",
        "## Tie-Break Trace",
        "",
    ]
    for step in comparison["tie_break_trace"]:
        lines.append(f"- {step}")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two Architecture-A JSON result files.")
    parser.add_argument("--fulltext", required=True, help="Path to full-text run JSON")
    parser.add_argument("--highlights", required=True, help="Path to highlights run JSON")
    parser.add_argument("--out-json", required=True, help="Comparison JSON output path")
    parser.add_argument("--out-md", required=True, help="Comparison markdown output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    full_payload = _load_payload(args.fulltext)
    high_payload = _load_payload(args.highlights)

    comparison = _compare_runs(full_payload, high_payload)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    _write_markdown(args.out_md, comparison)

    print(f"Winner: {comparison['winner']} ({comparison['deciding_criterion']})")
    print(f"Comparison JSON: {args.out_json}")
    print(f"Comparison Markdown: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
