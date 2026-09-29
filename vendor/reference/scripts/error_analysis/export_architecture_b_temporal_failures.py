"""Export likely temporal-reasoning failure cases from Architecture B.

Heuristic:
- final Architecture B answer is wrong,
- all three round-2 model decisions are wrong,
- round-2 reasoning traces contain explicit date/year language and definitive
  language ("confirms", "explicitly states", "therefore", etc.),
- and/or the evidence packet has no sources within 30 days of the resolution date.

The goal is to surface cases where the models appear to treat stale or
temporally mismatched evidence as if it were conclusive.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd


DEFAULT_RESULTS = Path("results/Final_Architecture_B_results.csv")
DEFAULT_CACHE_DIR = Path("cache/evidence")
DEFAULT_OUTPUT_DIR = Path("results/temporal_reasoning_analysis")

DATE_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b"
    r"|\b20\d{2}\b"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    re.I,
)

DEFINITIVE_RE = re.compile(
    r"\b(confirms?|explicitly states?|definitive|definitively|proves?|shows?|"
    r"indicates|demonstrates|therefore|thus|already|has already|is now|"
    r"no evidence|insufficient evidence|insufficient information|lack of evidence|"
    r"cannot determine|cannot be determined|does not mention|not explicitly mention)\b",
    re.I,
)


def _row_cache_path(cache_dir: Path, row: pd.Series) -> Path:
    return cache_dir / f"{row['retrieval_mode']}__{row['row_uid']}.json"


def _load_packet(cache_path: Path) -> Optional[dict]:
    if not cache_path.exists():
        return None
    return json.loads(cache_path.read_text())


def _source_gap_stats(packet: dict) -> dict[str, Optional[int]]:
    resolution_date = pd.to_datetime(packet.get("resolution_date"), utc=True, errors="coerce")
    if pd.isna(resolution_date):
        return {
            "num_sources": None,
            "sources_within_14d": None,
            "sources_within_30d": None,
            "min_days_from_resolution": None,
        }

    days: list[int] = []
    within_14 = 0
    within_30 = 0
    for source in packet.get("sources", []):
        published = pd.to_datetime(source.get("published_date"), utc=True, errors="coerce")
        if pd.isna(published):
            continue
        gap_days = abs((resolution_date - published).days)
        days.append(gap_days)
        if gap_days <= 14:
            within_14 += 1
        if gap_days <= 30:
            within_30 += 1

    return {
        "num_sources": len(packet.get("sources", [])),
        "sources_within_14d": within_14,
        "sources_within_30d": within_30,
        "min_days_from_resolution": min(days) if days else None,
    }


def _annotate_rows(df: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    out = df.copy()

    out["all_round2_wrong"] = (
        (out["gpt4o_round2_decision"] != out["ground_truth"])
        & (out["deepseek_round2_decision"] != out["ground_truth"])
        & (out["llama_round2_decision"] != out["ground_truth"])
    )

    reasoning_cols = [
        "gpt4o_round2_reasoning",
        "deepseek_round2_reasoning",
        "llama_round2_reasoning",
    ]
    for col in reasoning_cols:
        out[f"{col}_date"] = out[col].fillna("").map(lambda text: bool(DATE_RE.search(str(text))))
        out[f"{col}_definitive"] = out[col].fillna("").map(
            lambda text: bool(DEFINITIVE_RE.search(str(text)))
        )

    out["date_all_round2"] = out[[f"{c}_date" for c in reasoning_cols]].all(axis=1)
    out["date_any_round2"] = out[[f"{c}_date" for c in reasoning_cols]].any(axis=1)
    out["definitive_all_round2"] = out[[f"{c}_definitive" for c in reasoning_cols]].all(axis=1)
    out["definitive_any_round2"] = out[[f"{c}_definitive" for c in reasoning_cols]].any(axis=1)

    stats = []
    for _, row in out.iterrows():
        packet = _load_packet(_row_cache_path(cache_dir, row))
        if not packet:
            stats.append(
                {
                    "num_sources": None,
                    "sources_within_14d": None,
                    "sources_within_30d": None,
                    "min_days_from_resolution": None,
                    "resolution_year": None,
                }
            )
            continue
        gap_stats = _source_gap_stats(packet)
        resolution_date = pd.to_datetime(packet.get("resolution_date"), utc=True, errors="coerce")
        gap_stats["resolution_year"] = int(resolution_date.year) if not pd.isna(resolution_date) else None
        stats.append(gap_stats)

    stats_df = pd.DataFrame(stats)
    out = pd.concat([out.reset_index(drop=True), stats_df.reset_index(drop=True)], axis=1)
    return out


def _filter_rows(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    strict = (
        df["all_round2_wrong"]
        & df["date_all_round2"]
        & df["definitive_all_round2"]
        & ((df["sources_within_30d"] == 0) | (df["min_days_from_resolution"] >= 30))
    )

    broad = df["all_round2_wrong"] & (
        strict
        | (df["date_any_round2"] & df["definitive_any_round2"])
        | (df["sources_within_30d"] == 0)
    )

    if mode == "strict":
        return df[strict].copy()
    if mode == "broad":
        return df[broad].copy()
    raise ValueError(f"Unsupported mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export likely Architecture B temporal-reasoning failure cases."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Architecture B results CSV (default: results/Final_Architecture_B_results.csv)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Evidence cache directory (default: cache/evidence)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: results/temporal_reasoning_analysis)",
    )
    parser.add_argument(
        "--mode",
        choices=["strict", "broad", "both"],
        default="both",
        help="Which filter to export (default: both)",
    )
    args = parser.parse_args()

    if not args.results.exists():
        raise FileNotFoundError(f"Results CSV not found: {args.results}")

    df = pd.read_csv(args.results)
    if "is_correct" not in df.columns:
        raise ValueError("Results CSV is missing the is_correct column")

    wrong_df = df[df["is_correct"] == False].copy()
    annotated = _annotate_rows(wrong_df, args.cache_dir)

    modes = ["strict", "broad"] if args.mode == "both" else [args.mode]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    keep_columns = [
        "row_uid",
        "row_number",
        "retrieval_mode",
        "question_id",
        "question_text",
        "resolution_criteria",
        "category",
        "ground_truth",
        "final_decision",
        "all_round2_wrong",
        "date_all_round2",
        "date_any_round2",
        "definitive_all_round2",
        "definitive_any_round2",
        "num_sources",
        "sources_within_14d",
        "sources_within_30d",
        "min_days_from_resolution",
        "resolution_year",
        "gpt4o_round2_reasoning",
        "deepseek_round2_reasoning",
        "llama_round2_reasoning",
    ]

    for mode in modes:
        filtered = _filter_rows(annotated, mode)
        output_path = args.output_dir / f"architecture_b_temporal_failures_{mode}.csv"
        filtered.loc[:, keep_columns].to_csv(output_path, index=False)
        print(f"Wrote {len(filtered)} rows to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
