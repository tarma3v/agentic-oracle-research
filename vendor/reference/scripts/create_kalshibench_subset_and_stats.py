#!/usr/bin/env python3
"""Create a random KalshiBench subset and compute dataset/subset stats."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from datasets import load_dataset, load_from_disk


DATASET_ID = "2084Collective/kalshibench-v2"
DEFAULT_LOCAL_DATASET_DIR = Path("cache/kalshibench-v2")
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_MIN_CLOSE_DATE = date(2025, 5, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a random KalshiBench subset and write distribution stats."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_LOCAL_DATASET_DIR,
        help="Local Dataset.save_to_disk() directory (default: cache/kalshibench-v2).",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=450,
        help="Number of rows in random subset (default: 450).",
    )
    parser.add_argument(
        "--min-close-date",
        type=str,
        default=DEFAULT_MIN_CLOSE_DATE.isoformat(),
        help=(
            "Keep only markets with close_time on/after this date "
            "(YYYY-MM-DD, default: 2025-05-01)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Output directory for subset + stats files (default: results).",
    )
    return parser.parse_args()


def load_rows(dataset_dir: Path) -> list[dict[str, Any]]:
    if dataset_dir.exists():
        ds = load_from_disk(str(dataset_dir))
    else:
        ds = load_dataset(DATASET_ID, split="train")
    return list(ds)


def yes_stats(rows: list[dict[str, Any]]) -> tuple[int, int, float]:
    total = len(rows)
    yes_count = sum(
        1 for row in rows if str(row.get("ground_truth", "")).strip().lower() == "yes"
    )
    yes_pct = (yes_count / total * 100.0) if total else 0.0
    return yes_count, total, round(yes_pct, 2)


def category_distribution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(str(row.get("category", "Unknown")) for row in rows)
    total = len(rows)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [
        {
            "category": category,
            "count": count,
            "pct": round((count / total * 100.0) if total else 0.0, 2),
        }
        for category, count in ordered
    ]


def filter_rows_by_min_close_date(
    rows: list[dict[str, Any]], min_close_date: date
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        close_time = str(row.get("close_time", ""))
        # close_time is ISO-like UTC; compare its YYYY-MM-DD prefix.
        close_day = close_time[:10]
        if close_day >= min_close_date.isoformat():
            filtered.append(row)
    return filtered


def main() -> int:
    args = parse_args()
    all_rows = load_rows(args.dataset_dir)
    try:
        min_close_date = date.fromisoformat(args.min_close_date)
    except ValueError as exc:
        raise SystemExit("--min-close-date must be in YYYY-MM-DD format") from exc

    rows = filter_rows_by_min_close_date(all_rows, min_close_date)
    if args.subset_size < 1:
        raise SystemExit("--subset-size must be >= 1")
    if args.subset_size > len(rows):
        raise SystemExit(
            "--subset-size "
            f"({args.subset_size}) exceeds filtered dataset size ({len(rows)})."
        )

    rng = random.Random(args.seed)
    subset = rng.sample(rows, args.subset_size)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    subset_csv = args.out_dir / f"kalshibench_subset_{args.subset_size}_seed{args.seed}.csv"
    stats_json = args.out_dir / f"kalshibench_stats_{args.subset_size}_seed{args.seed}.json"

    with subset_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(subset)

    full_yes_count, full_total, full_yes_pct = yes_stats(all_rows)
    filtered_yes_count, filtered_total, filtered_yes_pct = yes_stats(rows)
    subset_yes_count, subset_total, subset_yes_pct = yes_stats(subset)

    stats = {
        "dataset_path": str(args.dataset_dir),
        "dataset_size": len(all_rows),
        "filtered_dataset_size": len(rows),
        "min_close_date": min_close_date.isoformat(),
        "subset_size": args.subset_size,
        "subset_seed": args.seed,
        "outputs": {
            "subset_csv": str(subset_csv),
            "stats_json": str(stats_json),
        },
        "full_dataset": {
            "yes_count": full_yes_count,
            "yes_pct": full_yes_pct,
            "category_distribution": category_distribution(all_rows),
        },
        "filtered_dataset": {
            "yes_count": filtered_yes_count,
            "yes_pct": filtered_yes_pct,
            "category_distribution": category_distribution(rows),
        },
        "subset": {
            "yes_count": subset_yes_count,
            "yes_pct": subset_yes_pct,
            "category_distribution": category_distribution(subset),
        },
        "notes": {
            "yes_percentage_definition": "100 * (ground_truth == 'yes') / total_rows"
        },
    }

    stats_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats["outputs"], indent=2))
    print(
        f"Full dataset yes%: {full_yes_pct} ({full_yes_count}/{full_total})\n"
        f"Filtered dataset yes%: {filtered_yes_pct} ({filtered_yes_count}/{filtered_total})\n"
        f"Subset yes%: {subset_yes_pct} ({subset_yes_count}/{subset_total})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
