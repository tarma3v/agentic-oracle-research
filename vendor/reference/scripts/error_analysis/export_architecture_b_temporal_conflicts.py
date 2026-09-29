"""Export likely temporal-conflict cases from Architecture B.

This is a stricter detector than the earlier broad temporal heuristic.
It looks for explicit date claims in the reasoning traces that conflict with
the cached evidence packet chronology:

* a sentence mentions `Source N` and a date that does not match the published
  date of that source,
* a sentence cites a source whose published date is after the question's
  resolution date,
* a sentence uses a post-cutoff date as if it were evidence.

The goal is to surface genuine chronology mistakes rather than merely
date-heavy reasoning.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


DEFAULT_RESULTS = Path("results/Final_Architecture_B_results.csv")
DEFAULT_CACHE_DIR = Path("cache/evidence")
DEFAULT_OUTPUT_DIR = Path("results/temporal_conflict_analysis")

SOURCE_REF_RE = re.compile(r"\bSource\s+(\d+)\b", re.I)
ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
MDY_DATE_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+"
    r"\d{1,2}(?:,\s*\d{4})?\b",
    re.I,
)
MONTH_YEAR_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4}\b",
    re.I,
)
YEAR_RE = re.compile(r"\b20\d{2}\b")

DATE_KEYWORDS_RE = re.compile(r"\b(published|dated|on|before|after|by|as of|source)\b", re.I)
POST_CUTOFF_HINT_RE = re.compile(
    r"\b(therefore|thus|so|proves?|shows?|confirms?|explicitly states?|indicates|"
    r"already|has already|is now)\b",
    re.I,
)


@dataclass
class TemporalConflict:
    row_uid: str
    question_id: str
    model: str
    round_label: str
    conflict_type: str
    sentence: str


def _row_cache_path(cache_dir: Path, row: pd.Series) -> Path:
    return cache_dir / f"{row['retrieval_mode']}__{row['row_uid']}.json"


def _load_packet(cache_path: Path) -> Optional[dict]:
    if not cache_path.exists():
        return None
    return json.loads(cache_path.read_text())


def _parse_date(text: str) -> Optional[pd.Timestamp]:
    value = text.strip().rstrip(",.")
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if not pd.isna(ts):
        return ts
    # Handle bare month-year phrases by pinning to the first of the month.
    for fmt in ("%B %Y", "%b %Y"):
        try:
            return pd.Timestamp(pd.to_datetime(value, format=fmt, utc=True))
        except Exception:
            continue
    return None


def _extract_dates(sentence: str) -> list[str]:
    found: list[str] = []
    found.extend(ISO_DATE_RE.findall(sentence))
    found.extend(MDY_DATE_RE.findall(sentence))
    found.extend(MONTH_YEAR_RE.findall(sentence))
    # Deduplicate while preserving order.
    seen = set()
    ordered: list[str] = []
    for item in found:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(item)
    return ordered


def _split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []
    return re.split(r"(?<=[.!?])\s+", cleaned)


def _source_published_dates(packet: dict) -> list[Optional[pd.Timestamp]]:
    out: list[Optional[pd.Timestamp]] = []
    for source in packet.get("sources", []):
        out.append(pd.to_datetime(source.get("published_date"), utc=True, errors="coerce"))
    return out


def _date_distance_days(a: pd.Timestamp, b: pd.Timestamp) -> Optional[int]:
    if pd.isna(a) or pd.isna(b):
        return None
    return abs((a - b).days)


def _sentence_conflicts(
    sentence: str,
    packet: dict,
    resolution_date: pd.Timestamp,
    source_dates: list[Optional[pd.Timestamp]],
) -> list[str]:
    conflicts: list[str] = []
    source_refs = [int(match) for match in SOURCE_REF_RE.findall(sentence)]
    date_strings = _extract_dates(sentence)
    parsed_dates = [d for d in (_parse_date(text) for text in date_strings) if d is not None]
    lower_sentence = sentence.lower()
    has_source_or_date_language = bool(DATE_KEYWORDS_RE.search(sentence))

    for source_idx in source_refs:
        if source_idx < 1 or source_idx > len(source_dates):
            continue
        pub_date = source_dates[source_idx - 1]
        if pd.isna(pub_date):
            continue
        if not pd.isna(resolution_date) and pub_date > resolution_date:
            conflicts.append(
                f"Source {source_idx} published after resolution date ({pub_date.date()} > {resolution_date.date()})"
            )

        # If the sentence explicitly states a date near a source reference,
        # flag mismatches against the actual source publication date.
        for claimed_date in parsed_dates:
            distance = _date_distance_days(claimed_date, pub_date)
            if distance is not None and distance > 7:
                conflicts.append(
                    f"Source {source_idx} date mismatch ({claimed_date.date()} vs {pub_date.date()})"
                )
                break

    # If a sentence mentions a date and uses it assertively, flag dates after
    # the resolution cutoff as suspicious chronology usage.
    for claimed_date in parsed_dates:
        if not pd.isna(resolution_date) and claimed_date > resolution_date and (
            has_source_or_date_language or POST_CUTOFF_HINT_RE.search(lower_sentence)
        ):
            conflicts.append(
                f"Post-cutoff date used as evidence ({claimed_date.date()} after {resolution_date.date()})"
            )

    return list(dict.fromkeys(conflicts))


def _collect_row_conflicts(row: pd.Series, packet: dict) -> list[TemporalConflict]:
    conflicts: list[TemporalConflict] = []
    resolution_date = pd.to_datetime(packet.get("resolution_date"), utc=True, errors="coerce")
    source_dates = _source_published_dates(packet)

    reasoning_fields = [
        ("gpt4o", "round1", row.get("gpt4o_round1_reasoning", "")),
        ("gpt4o", "round2", row.get("gpt4o_round2_reasoning", "")),
        ("deepseek", "round1", row.get("deepseek_round1_reasoning", "")),
        ("deepseek", "round2", row.get("deepseek_round2_reasoning", "")),
        ("llama", "round1", row.get("llama_round1_reasoning", "")),
        ("llama", "round2", row.get("llama_round2_reasoning", "")),
    ]

    for model_name, round_label, reasoning in reasoning_fields:
        for sentence in _split_sentences(str(reasoning)):
            sentence_conflicts = _sentence_conflicts(
                sentence=sentence,
                packet=packet,
                resolution_date=resolution_date,
                source_dates=source_dates,
            )
            for conflict in sentence_conflicts:
                conflicts.append(
                    TemporalConflict(
                        row_uid=str(row.get("row_uid")),
                        question_id=str(row.get("question_id")),
                        model=model_name,
                        round_label=round_label,
                        conflict_type=conflict,
                        sentence=sentence,
                    )
                )

    return conflicts


def _annotate(df: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        packet = _load_packet(_row_cache_path(cache_dir, row))
        if not packet:
            continue
        conflicts = _collect_row_conflicts(row, packet)
        if not conflicts:
            continue

        source_dates = _source_published_dates(packet)
        resolution_date = pd.to_datetime(packet.get("resolution_date"), utc=True, errors="coerce")
        rows.append(
            {
                "row_uid": row["row_uid"],
                "row_number": row["row_number"],
                "retrieval_mode": row["retrieval_mode"],
                "question_id": row["question_id"],
                "question_text": row["question_text"],
                "resolution_criteria": row["resolution_criteria"],
                "category": row["category"],
                "ground_truth": row["ground_truth"],
                "final_decision": row["final_decision"],
                "is_correct": row["is_correct"],
                "num_sources": len(packet.get("sources", [])),
                "sources_after_resolution": int(
                    sum(
                        1
                        for dt in source_dates
                        if not pd.isna(dt) and not pd.isna(resolution_date) and dt > resolution_date
                    )
                ),
                "min_days_from_resolution": None,
                "conflict_count": len(conflicts),
                "conflict_models": "; ".join(sorted({c.model for c in conflicts})),
                "conflict_rounds": "; ".join(sorted({c.round_label for c in conflicts})),
                "conflict_types": " | ".join(sorted({c.conflict_type for c in conflicts})),
                "example_sentence": conflicts[0].sentence if conflicts else "",
                "gpt4o_round1_reasoning": row.get("gpt4o_round1_reasoning", ""),
                "gpt4o_round2_reasoning": row.get("gpt4o_round2_reasoning", ""),
                "deepseek_round1_reasoning": row.get("deepseek_round1_reasoning", ""),
                "deepseek_round2_reasoning": row.get("deepseek_round2_reasoning", ""),
                "llama_round1_reasoning": row.get("llama_round1_reasoning", ""),
                "llama_round2_reasoning": row.get("llama_round2_reasoning", ""),
            }
        )

        if not pd.isna(resolution_date):
            valid_deltas = [
                abs((resolution_date - dt).days)
                for dt in source_dates
                if not pd.isna(dt)
            ]
            rows[-1]["min_days_from_resolution"] = min(valid_deltas) if valid_deltas else None

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export Architecture B rows with temporal reasoning conflicts."
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
        help="Output directory (default: results/temporal_conflict_analysis)",
    )
    args = parser.parse_args()

    if not args.results.exists():
        raise FileNotFoundError(f"Results CSV not found: {args.results}")

    df = pd.read_csv(args.results)
    if "is_correct" not in df.columns:
        raise ValueError("Results CSV is missing the is_correct column")

    wrong_df = df[df["is_correct"] == False].copy()
    annotated = _annotate(wrong_df, args.cache_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "architecture_b_temporal_conflicts.csv"
    annotated.to_csv(output_path, index=False)
    print(f"Wrote {len(annotated)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
