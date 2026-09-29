"""Export incorrect rows from Architecture A and B result CSVs.

This script filters the saved experiment outputs down to rows where
`is_correct` is false and writes one CSV per architecture.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List


DEFAULT_ARCH_A_INPUT = Path("results/architecture_a_results.csv")
DEFAULT_ARCH_B_INPUT = Path("results/Final_Architecture_B_results.csv")
DEFAULT_OUTPUT_DIR = Path("results/failure_cases")


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _row_for_csv(row: dict) -> dict:
    out = {}
    for key, value in row.items():
        if isinstance(value, str) and value:
            out[key] = value.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
        else:
            out[key] = value
    return out


def _filter_failures(input_path: Path) -> tuple[list[str], list[dict]]:
    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [row for row in reader if not _parse_bool(row.get("is_correct"))]
    return fieldnames, rows


def _write_csv(fieldnames: List[str], rows: Iterable[dict], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow(_row_for_csv(row))
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export incorrect rows from Architecture A/B result CSVs."
    )
    parser.add_argument(
        "--arch-a-input",
        type=Path,
        default=DEFAULT_ARCH_A_INPUT,
        help="Architecture A results CSV (default: results/architecture_a_results.csv)",
    )
    parser.add_argument(
        "--arch-b-input",
        type=Path,
        default=DEFAULT_ARCH_B_INPUT,
        help=(
            "Architecture B results CSV "
            "(default: results/architecture_b_results/architecture_b_results.csv)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write failure CSVs into (default: results/failure_cases)",
    )
    args = parser.parse_args()

    outputs = [
        ("architecture_a_failures.csv", args.arch_a_input),
        ("architecture_b_failures.csv", args.arch_b_input),
    ]

    for output_name, input_path in outputs:
        if not input_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_path}")

        fieldnames, failures = _filter_failures(input_path)
        output_path = args.output_dir / output_name
        count = _write_csv(fieldnames, failures, output_path)
        print(f"Wrote {count} failure rows to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
