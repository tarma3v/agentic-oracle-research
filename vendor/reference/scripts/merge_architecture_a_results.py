"""
Merge Architecture-A batch results into combined CSV + JSON.

CSV merge is intentionally text-level (not a CSV parse) because some batch CSVs use
different quoting conventions while still being valid CSV. We keep the header from
the first input file and append all subsequent rows from the remaining files.

JSON merge concatenates the per-question `results` arrays and recomputes the summary
from the merged results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _default_results_dir() -> Path:
    """
    Prefer the cleaned-up folder if it exists; otherwise fall back to legacy layout.
    """
    new_dir = Path("results/architecture_a_results")
    return new_dir if new_dir.exists() else Path("results")


def _default_output_csv() -> Path:
    return Path("results/architecture_a_results.csv")


def _default_output_json() -> Path:
    return Path("results/architecture_a_results.json")


def _default_inputs_csv() -> list[str]:
    base = _default_results_dir()
    return [
        str(base / "architecture_a_first100.csv"),
        str(base / "architecture_a_second_100.csv"),
        str(base / "architecture_a_third_100.csv"),
        str(base / "architecture_a_fourth_100.csv"),
        str(base / "architecture_a_last50.csv"),
    ]


def _default_inputs_json() -> list[str]:
    base = _default_results_dir()
    return [
        str(base / "architecture_a_first100.json"),
        str(base / "architecture_a_second_100.json"),
        str(base / "architecture_a_third_100.json"),
        str(base / "architecture_a_fourth_100.json"),
        str(base / "architecture_a_last50.json"),
    ]


def _read_lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    # Keep line endings consistent when writing the merged file.
    return text.splitlines()


def merge_csvs(inputs: list[Path], output: Path) -> None:
    if not inputs:
        raise ValueError("No input CSVs provided.")

    all_lines: list[str] = []

    for idx, path in enumerate(inputs):
        if not path.exists():
            raise FileNotFoundError(f"Missing input CSV: {path}")

        lines = _read_lines(path)
        if not lines:
            raise ValueError(f"Input CSV is empty: {path}")

        if idx == 0:
            all_lines.extend(lines)
        else:
            # Skip header row for subsequent inputs.
            all_lines.extend(lines[1:])

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(all_lines) + "\n", encoding="utf-8")


def _normalize_truth(value: str | None) -> str | None:
    if value is None:
        return None
    return str(value).strip().upper()


def _accuracy(correct: int, total: int) -> float | None:
    return (correct / total) if total > 0 else None


def _compute_summary_from_results(results: list[dict]) -> dict:
    processed = len(results)
    correct = sum(1 for r in results if bool(r.get("is_correct")))

    def _agent_acc(prefix: str) -> float | None:
        total = 0
        corr = 0
        for r in results:
            decision = _normalize_truth(r.get(f"{prefix}_decision"))
            truth = _normalize_truth(r.get("ground_truth"))
            if decision is None or truth is None:
                continue
            total += 1
            if decision == truth:
                corr += 1
        return _accuracy(corr, total)

    unanimous = 0
    majority = 0
    for r in results:
        try:
            yes = int(r.get("yes_votes", 0))
            no = int(r.get("no_votes", 0))
        except Exception:
            continue
        if yes == 3 or no == 3:
            unanimous += 1
        if yes != no:
            majority += 1

    return {
        "total_questions": processed,
        "processed_questions": processed,
        "skipped_questions": 0,
        "failed_evidence": 0,
        "accuracy": _accuracy(correct, processed),
        "gpt4o_accuracy": _agent_acc("gpt4o"),
        "deepseek_accuracy": _agent_acc("deepseek"),
        "llama_accuracy": _agent_acc("llama"),
        "unanimous_rate": _accuracy(unanimous, processed),
        "majority_rate": _accuracy(majority, processed),
        "average_latency_ms": None,
        "runtime_seconds": None,
        "estimated_exa_cost_usd": None,
    }


def merge_jsons(inputs: list[Path], output: Path) -> None:
    payloads = []
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(f"Missing input JSON: {path}")
        payloads.append(json.loads(path.read_text(encoding="utf-8")))

    merged_results: list[dict] = []
    for p in payloads:
        merged_results.extend(list(p.get("results", [])))

    summary = _compute_summary_from_results(merged_results)

    # Weighted averages / sums where available from batch summaries.
    total_latency_weight = 0.0
    total_latency_weighted = 0.0
    runtime_sum = 0.0
    runtime_any = False
    cost_sum = 0.0
    cost_any = False

    for p in payloads:
        s = p.get("summary", {}) or {}
        processed = int(s.get("processed_questions") or 0)
        avg_latency = s.get("average_latency_ms")
        if avg_latency is not None and processed > 0:
            total_latency_weight += processed
            total_latency_weighted += float(avg_latency) * processed

        if s.get("runtime_seconds") is not None:
            runtime_sum += float(s["runtime_seconds"])
            runtime_any = True
        if s.get("estimated_exa_cost_usd") is not None:
            cost_sum += float(s["estimated_exa_cost_usd"])
            cost_any = True

    if total_latency_weight > 0:
        summary["average_latency_ms"] = total_latency_weighted / total_latency_weight
    if runtime_any:
        summary["runtime_seconds"] = runtime_sum
    if cost_any:
        summary["estimated_exa_cost_usd"] = cost_sum

    merged_run_config = {
        "merged_from": [str(p) for p in inputs],
        "total_questions": len(merged_results),
    }
    first_cfg = (payloads[0].get("run_config", {}) or {}) if payloads else {}
    for k in (
        "retrieval_mode",
        "num_results",
        "fulltext_max_characters",
        "highlights_max_characters",
        "cache_dir",
    ):
        if k in first_cfg:
            merged_run_config[k] = first_cfg.get(k)

    out_payload = {
        "run_config": merged_run_config,
        "summary": summary,
        "results": merged_results,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge Architecture-A batch results into combined CSV + JSON."
    )
    parser.add_argument(
        "--output-csv",
        default=str(_default_output_csv()),
        help="Output CSV path (default: results/architecture_a_results.csv)",
    )
    parser.add_argument(
        "--output-json",
        default=str(_default_output_json()),
        help="Output JSON path (default: results/architecture_a_results.json)",
    )
    parser.add_argument("--inputs-csv", nargs="*", default=_default_inputs_csv())
    parser.add_argument("--inputs-json", nargs="*", default=_default_inputs_json())
    args = parser.parse_args()

    inputs_csv = [Path(p) for p in args.inputs_csv]
    output_csv = Path(args.output_csv)
    inputs_json = [Path(p) for p in args.inputs_json]
    output_json = Path(args.output_json)

    merge_csvs(inputs=inputs_csv, output=output_csv)
    merge_jsons(inputs=inputs_json, output=output_json)

    # Print a small sanity check for humans running the script.
    merged_lines = _read_lines(output_csv)
    total_rows = max(0, len(merged_lines) - 1)
    print(f"Wrote {output_csv} ({total_rows} data rows).")
    print(f"Wrote {output_json}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

