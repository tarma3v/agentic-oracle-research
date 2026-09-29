#!/usr/bin/env python3
"""Analyze prediction-market results by unique vs multi-instance market views."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

YES_NO = {"YES", "NO"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build unique-by-market and multi-instance-only analysis bundles."
    )
    parser.add_argument("--input-csv", required=True, help="Input results CSV to analyze.")
    parser.add_argument(
        "--market-key",
        default="id",
        help="Column used to identify markets (default: id).",
    )
    parser.add_argument(
        "--decision-col",
        default="final_decision",
        help="Preferred final decision column (default: final_decision).",
    )
    parser.add_argument(
        "--truth-col",
        default="ground_truth",
        help="Ground-truth column (default: ground_truth).",
    )
    parser.add_argument(
        "--category-col",
        default="category",
        help="Category column (default: category).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/market_instance_analysis",
        help="Output directory (default: results/market_instance_analysis).",
    )
    return parser.parse_args()


def normalize_decision(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def wilson_interval(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    phat = correct / n
    z2 = z**2
    denom = 1.0 + (z2 / n)
    center = (phat + (z2 / (2.0 * n))) / denom
    half_width = (
        z * np.sqrt((phat * (1.0 - phat) + (z2 / (4.0 * n))) / n) / denom
    )
    return float(center - half_width), float(center + half_width)


def dedupe_keep_first(df: pd.DataFrame, market_key: str) -> pd.DataFrame:
    return df.drop_duplicates(subset=[market_key], keep="first").copy()


def filter_multi_instance(df: pd.DataFrame, market_key: str) -> pd.DataFrame:
    counts = df.groupby(market_key, dropna=False)[market_key].transform("size")
    return df.loc[counts > 1].copy()


def detect_decision_columns(
    df: pd.DataFrame, preferred_decision_col: str, truth_col: str
) -> list[str]:
    cols: list[str] = []
    if preferred_decision_col in df.columns and preferred_decision_col != truth_col:
        cols.append(preferred_decision_col)

    for col in df.columns:
        if col == truth_col or col in cols:
            continue
        if col.endswith("_decision") or col.endswith("_final"):
            cols.append(col)
    return cols


def _split_snapshot(df: pd.DataFrame, market_key: str, truth_col: str) -> dict[str, float | int]:
    truth = normalize_decision(df[truth_col])
    yes_count = int((truth == "YES").sum())
    no_count = int((truth == "NO").sum())
    row_count = int(len(df))
    return {
        "row_count": row_count,
        "unique_markets": int(df[market_key].nunique(dropna=False)),
        "yes_count": yes_count,
        "no_count": no_count,
        "yes_pct": float((yes_count / row_count) if row_count else 0.0),
        "no_pct": float((no_count / row_count) if row_count else 0.0),
    }


def compute_overall_metrics(
    df: pd.DataFrame,
    *,
    split_name: str,
    market_key: str,
    truth_col: str,
    decision_cols: Iterable[str],
) -> pd.DataFrame:
    cols = [
        "split",
        "system",
        "total_rows",
        "unique_markets",
        "yes_count",
        "no_count",
        "yes_pct",
        "no_pct",
        "attempted_n",
        "correct_n",
        "accuracy",
        "wilson_ci_low",
        "wilson_ci_high",
    ]
    snapshot = _split_snapshot(df, market_key, truth_col)
    truth = normalize_decision(df[truth_col])
    rows: list[dict[str, float | int | str]] = []
    for col in decision_cols:
        pred = normalize_decision(df[col])
        valid = truth.isin(YES_NO) & pred.isin(YES_NO)
        attempted_n = int(valid.sum())
        correct_n = int(((pred == truth) & valid).sum())
        accuracy = float(correct_n / attempted_n) if attempted_n else float("nan")
        ci_low, ci_high = wilson_interval(correct_n, attempted_n)
        rows.append(
            {
                "split": split_name,
                "system": col,
                "total_rows": snapshot["row_count"],
                "unique_markets": snapshot["unique_markets"],
                "yes_count": snapshot["yes_count"],
                "no_count": snapshot["no_count"],
                "yes_pct": snapshot["yes_pct"],
                "no_pct": snapshot["no_pct"],
                "attempted_n": attempted_n,
                "correct_n": correct_n,
                "accuracy": accuracy,
                "wilson_ci_low": ci_low,
                "wilson_ci_high": ci_high,
            }
        )

    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols).sort_values(["accuracy", "attempted_n"], ascending=[False, False])


def compute_category_distribution(df: pd.DataFrame, category_col: str) -> pd.DataFrame:
    category = df[category_col].fillna("Unknown").astype(str)
    counts = category.value_counts(dropna=False)
    total = len(df)
    out = pd.DataFrame(
        {
            "category": counts.index,
            "count": counts.values,
            "pct": counts.values / total if total else 0.0,
            "total_rows": total,
        }
    )
    return out.sort_values(["count", "category"], ascending=[False, True]).reset_index(drop=True)


def compute_multi_instance_market_consistency(
    df: pd.DataFrame,
    *,
    market_key: str,
    truth_col: str,
    decision_cols: Iterable[str],
) -> pd.DataFrame:
    rows: list[dict[str, int | str | float]] = []
    truth = normalize_decision(df[truth_col])
    for col in decision_cols:
        pred = normalize_decision(df[col])
        valid = truth.isin(YES_NO) & pred.isin(YES_NO)
        scored = pd.DataFrame(
            {
                market_key: df[market_key],
                "_is_valid": valid,
                "_is_correct": (pred == truth) & valid,
            }
        )
        grouped = scored.groupby(market_key, dropna=False).agg(
            total_instances=("_is_valid", "size"),
            scored_instances=("_is_valid", "sum"),
            correct_instances=("_is_correct", "sum"),
        )

        no_scored = int((grouped["scored_instances"] == 0).sum())
        all_correct = int(
            ((grouped["scored_instances"] > 0) & (grouped["correct_instances"] == grouped["scored_instances"])).sum()
        )
        all_incorrect = int(
            ((grouped["scored_instances"] > 0) & (grouped["correct_instances"] == 0)).sum()
        )
        mixed = int(
            (
                (grouped["scored_instances"] > 0)
                & (grouped["correct_instances"] > 0)
                & (grouped["correct_instances"] < grouped["scored_instances"])
            ).sum()
        )
        total_markets = int(len(grouped))
        rows.append(
            {
                "system": col,
                "total_multi_markets": total_markets,
                "all_instances_correct_markets": all_correct,
                "all_instances_incorrect_markets": all_incorrect,
                "mixed_correctness_markets": mixed,
                "no_scored_instances_markets": no_scored,
                "all_instances_correct_pct": (all_correct / total_markets) if total_markets else float("nan"),
                "mixed_correctness_pct": (mixed / total_markets) if total_markets else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _write_bundle(
    bundle_df: pd.DataFrame,
    *,
    split_name: str,
    market_key: str,
    truth_col: str,
    category_col: str,
    decision_cols: list[str],
    output_dir: Path,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    bundle_out = output_dir / split_name
    bundle_out.mkdir(parents=True, exist_ok=True)

    overall = compute_overall_metrics(
        bundle_df,
        split_name=split_name,
        market_key=market_key,
        truth_col=truth_col,
        decision_cols=decision_cols,
    )
    by_category = compute_category_distribution(bundle_df, category_col)

    overall.to_csv(bundle_out / "overall_metrics.csv", index=False)
    by_category.to_csv(bundle_out / "category_metrics.csv", index=False)
    consistency = pd.DataFrame()
    if split_name == "multi_instance_only":
        consistency = compute_multi_instance_market_consistency(
            bundle_df,
            market_key=market_key,
            truth_col=truth_col,
            decision_cols=decision_cols,
        )
        consistency.to_csv(bundle_out / "market_consistency_metrics.csv", index=False)
    return _split_snapshot(bundle_df, market_key, truth_col), consistency


def _best_system_text(overall: pd.DataFrame) -> str:
    if overall.empty:
        return "N/A (no decision columns with YES/NO predictions found)"
    row = overall.sort_values(["accuracy", "attempted_n"], ascending=[False, False]).iloc[0]
    if pd.isna(row["accuracy"]):
        return "N/A (no attempted YES/NO predictions)"
    return (
        f"{row['system']} "
        f"(acc={row['accuracy']:.3f}, 95% CI=[{row['wilson_ci_low']:.3f}, {row['wilson_ci_high']:.3f}], "
        f"attempted={int(row['attempted_n'])})"
    )


def _write_summary_markdown(
    output_dir: Path,
    unique_snapshot: dict[str, float | int],
    multi_snapshot: dict[str, float | int],
    unique_overall: pd.DataFrame,
    multi_overall: pd.DataFrame,
    multi_consistency: pd.DataFrame,
) -> None:
    lines = [
        "# Unique vs Multi-Instance Market Analysis",
        "",
        "## Side-by-side summary",
        "",
        "| Metric | unique_by_id | multi_instance_only |",
        "|---|---:|---:|",
        f"| Rows | {unique_snapshot['row_count']} | {multi_snapshot['row_count']} |",
        f"| Unique markets | {unique_snapshot['unique_markets']} | {multi_snapshot['unique_markets']} |",
        f"| YES count | {unique_snapshot['yes_count']} | {multi_snapshot['yes_count']} |",
        f"| NO count | {unique_snapshot['no_count']} | {multi_snapshot['no_count']} |",
        f"| YES % | {unique_snapshot['yes_pct']:.2%} | {multi_snapshot['yes_pct']:.2%} |",
        f"| NO % | {unique_snapshot['no_pct']:.2%} | {multi_snapshot['no_pct']:.2%} |",
        "",
        "## Best-performing decision column",
        "",
        f"- unique_by_id: {_best_system_text(unique_overall)}",
        f"- multi_instance_only: {_best_system_text(multi_overall)}",
        "",
        "## Multi-instance market-level consistency",
        "",
    ]
    if multi_consistency.empty:
        lines.extend(["No decision columns were available for market-level consistency."])
    else:
        lines.extend(
            [
                "| System | Multi-markets | All instances correct | Mixed correctness | All instances incorrect | No scored instances |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in multi_consistency.sort_values(
            ["all_instances_correct_pct", "mixed_correctness_pct"], ascending=[False, False]
        ).iterrows():
            lines.append(
                f"| {row['system']} | {int(row['total_multi_markets'])} | "
                f"{int(row['all_instances_correct_markets'])} | "
                f"{int(row['mixed_correctness_markets'])} | "
                f"{int(row['all_instances_incorrect_markets'])} | "
                f"{int(row['no_scored_instances_markets'])} |"
            )
    lines.extend(
        [
            "",
            "## Notes",
        "",
        "- `unique_by_id` keeps first occurrence by original file order for each market key.",
        "- `multi_instance_only` keeps all rows where market key frequency is greater than 1.",
        "- These are independent analytical views and are not expected to have matching row counts.",
        ]
    )
    (output_dir / "summary_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    required = [args.market_key, args.truth_col, args.category_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    decision_cols = detect_decision_columns(df, args.decision_col, args.truth_col)

    unique_df = dedupe_keep_first(df, args.market_key)
    multi_df = filter_multi_instance(df, args.market_key)

    unique_snapshot, _ = _write_bundle(
        unique_df,
        split_name="unique_by_id",
        market_key=args.market_key,
        truth_col=args.truth_col,
        category_col=args.category_col,
        decision_cols=decision_cols,
        output_dir=output_dir,
    )
    multi_snapshot, multi_consistency = _write_bundle(
        multi_df,
        split_name="multi_instance_only",
        market_key=args.market_key,
        truth_col=args.truth_col,
        category_col=args.category_col,
        decision_cols=decision_cols,
        output_dir=output_dir,
    )

    unique_overall = pd.read_csv(output_dir / "unique_by_id" / "overall_metrics.csv")
    multi_overall = pd.read_csv(output_dir / "multi_instance_only" / "overall_metrics.csv")
    _write_summary_markdown(
        output_dir=output_dir,
        unique_snapshot=unique_snapshot,
        multi_snapshot=multi_snapshot,
        unique_overall=unique_overall,
        multi_overall=multi_overall,
        multi_consistency=multi_consistency,
    )

    metadata = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(input_path),
        "output_dir": str(output_dir),
        "columns": {
            "market_key": args.market_key,
            "truth_col": args.truth_col,
            "category_col": args.category_col,
            "preferred_decision_col": args.decision_col,
            "decision_cols_used": decision_cols,
        },
        "split_counts": {
            "source_rows": int(len(df)),
            "unique_by_id_rows": int(len(unique_df)),
            "multi_instance_only_rows": int(len(multi_df)),
            "source_unique_markets": int(df[args.market_key].nunique(dropna=False)),
            "multi_instance_market_count": int(multi_df[args.market_key].nunique(dropna=False)),
        },
        "multi_instance_consistency": multi_consistency.to_dict(orient="records"),
        "assumptions": {
            "market_identity": "market_key column",
            "unique_rule": "keep first occurrence per market key by original file order",
            "multi_rule": "keep rows where market key frequency > 1",
            "truth_and_predictions_normalization": "uppercased and trimmed before scoring",
            "accuracy_denominator": "rows where both truth and prediction are YES/NO",
            "ci_method": "95% Wilson interval",
        },
    }
    (output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "decision_cols_used": decision_cols}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
