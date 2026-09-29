"""Export likely hallucination cases from Architecture B.

Heuristic:
- Split each reasoning trace into sentences.
- For each sentence, compare it against the cached evidence packet contents
  (source titles, highlights, and text) using fuzzy string similarity.
- If a sentence contains a factual cue (source refs, numbers, dates, or
  quotation markers) but fails to match the packet contents, flag it as a
  hallucination candidate.

This is intentionally conservative: it is meant to surface claims that seem
to reference facts not present anywhere in the evidence packet.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import pandas as pd


DEFAULT_RESULTS = Path("results/Final_Architecture_B_results.csv")
DEFAULT_CACHE_DIR = Path("cache/evidence")
DEFAULT_OUTPUT_DIR = Path("results/hallucination_analysis")

SOURCE_REF_RE = re.compile(r"\bSource\s+(\d+)\b", re.I)
DATE_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}"
    r"(?:,\s*\d{4})?\b|\b\d{4}-\d{2}-\d{2}\b|\b20\d{2}\b",
    re.I,
)
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
QUOTE_RE = re.compile(r"[\"“”]")
QUOTED_SPAN_RE = re.compile(r"[\"“](.+?)[\"”]")
NUMERIC_PHRASE_RE = re.compile(
    r"\b\d+(?:\.\d+)?(?:\s*(?:million|billion|thousand|votes|days|weeks|months|years|"
    r"percent|%|times|points|goals|streams|wins|seats|cut|dollars|usd|bps|basis points))?"
    r"(?:\s+[a-zA-Z][a-zA-Z'-]{1,20}){0,3}\b",
    re.I,
)


@dataclass
class HallucinationHit:
    row_uid: str
    question_id: str
    model: str
    round_label: str
    sentence: str
    best_score: float
    matched_source: str


def _row_cache_path(cache_dir: Path, row: pd.Series) -> Path:
    return cache_dir / f"{row['retrieval_mode']}__{row['row_uid']}.json"


def _load_packet(cache_path: Path) -> Optional[dict]:
    if not cache_path.exists():
        return None
    return json.loads(cache_path.read_text())


def _split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []
    return re.split(r"(?<=[.!?])\s+", cleaned)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def _tokens(text: str) -> set[str]:
    return {tok for tok in _normalize(text).split() if tok}


def _packet_content(packet: dict) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    for idx, source in enumerate(packet.get("sources", []), start=1):
        pieces = [f"Source {idx}", source.get("title", ""), source.get("text", "")]
        pieces.extend(source.get("highlights", []) or [])
        blob = "\n".join(p for p in pieces if p)
        chunks.append((f"Source {idx}: {source.get('title', '')}", blob))
    return chunks


def _factual_sentence(sentence: str) -> bool:
    return bool(
        SOURCE_REF_RE.search(sentence)
        or DATE_RE.search(sentence)
        or NUMBER_RE.search(sentence)
        or QUOTE_RE.search(sentence)
    )


def _extract_claim_spans(sentence: str) -> list[str]:
    spans: list[str] = []
    spans.extend(match.group(1).strip() for match in QUOTED_SPAN_RE.finditer(sentence))
    spans.extend(match.group(0).strip() for match in DATE_RE.finditer(sentence))
    spans.extend(match.group(0).strip() for match in NUMERIC_PHRASE_RE.finditer(sentence))
    spans.extend(f"Source {match}" for match in SOURCE_REF_RE.findall(sentence))
    # Deduplicate while preserving order and ignore tiny fragments.
    seen = set()
    ordered: list[str] = []
    for span in spans:
        normalized = _normalize(span)
        if len(normalized.split()) < 2 and not SOURCE_REF_RE.search(span):
            continue
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(span)
    return ordered


def _best_match(span: str, packet_chunks: list[tuple[str, str]]) -> tuple[float, str]:
    normalized_span = _normalize(span)
    span_tokens = _tokens(span)
    if not normalized_span:
        return 0.0, ""

    best_score = 0.0
    best_label = ""
    for label, content in packet_chunks:
        normalized_content = _normalize(content)
        if not normalized_content:
            continue
        content_tokens = _tokens(content)
        # Prefer overlap of the important tokens in the claim span.
        token_overlap = (
            len(span_tokens & content_tokens) / len(span_tokens)
            if span_tokens
            else 0.0
        )
        seq = SequenceMatcher(None, normalized_span, normalized_content).ratio()
        score = max(token_overlap, seq)
        if score > best_score:
            best_score = score
            best_label = label
    return best_score, best_label


def _collect_hits(row: pd.Series, packet: dict, threshold: float) -> list[HallucinationHit]:
    hits: list[HallucinationHit] = []
    packet_chunks = _packet_content(packet)

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
            if not _factual_sentence(sentence):
                continue
            claim_spans = _extract_claim_spans(sentence)
            if not claim_spans:
                continue
            span_scores: list[tuple[float, str]] = []
            for span in claim_spans:
                span_scores.append(_best_match(span, packet_chunks))
            best_score, label = max(span_scores, key=lambda item: item[0])
            if best_score < threshold:
                hits.append(
                    HallucinationHit(
                        row_uid=str(row.get("row_uid")),
                        question_id=str(row.get("question_id")),
                        model=model_name,
                        round_label=round_label,
                        sentence=sentence,
                        best_score=best_score,
                        matched_source=label,
                    )
                )

    return hits


def _summarize_hits(row: pd.Series, packet: dict, hits: list[HallucinationHit]) -> dict:
    source_titles = [s.get("title", "") for s in packet.get("sources", [])]
    source_dates = [s.get("published_date") for s in packet.get("sources", [])]
    return {
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
        "packet_source_count": len(packet.get("sources", [])),
        "packet_source_titles": " | ".join(source_titles[:5]),
        "packet_source_dates": " | ".join([d for d in source_dates[:5] if d]),
        "hit_count": len(hits),
        "hit_models": "; ".join(sorted({h.model for h in hits})),
        "hit_rounds": "; ".join(sorted({h.round_label for h in hits})),
        "best_score_min": min((h.best_score for h in hits), default=None),
        "best_score_avg": (sum(h.best_score for h in hits) / len(hits)) if hits else None,
        "example_sentence": hits[0].sentence if hits else "",
        "example_best_source": hits[0].matched_source if hits else "",
        "gpt4o_round1_reasoning": row.get("gpt4o_round1_reasoning", ""),
        "gpt4o_round2_reasoning": row.get("gpt4o_round2_reasoning", ""),
        "deepseek_round1_reasoning": row.get("deepseek_round1_reasoning", ""),
        "deepseek_round2_reasoning": row.get("deepseek_round2_reasoning", ""),
        "llama_round1_reasoning": row.get("llama_round1_reasoning", ""),
        "llama_round2_reasoning": row.get("llama_round2_reasoning", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export likely hallucination cases from Architecture B."
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
        help="Output directory (default: results/hallucination_analysis)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.32,
        help="SequenceMatcher threshold below which a factual sentence is flagged.",
    )
    args = parser.parse_args()

    if not args.results.exists():
        raise FileNotFoundError(f"Results CSV not found: {args.results}")

    df = pd.read_csv(args.results)
    if "is_correct" not in df.columns:
        raise ValueError("Results CSV is missing the is_correct column")

    wrong_df = df[df["is_correct"] == False].copy()
    rows = []
    for _, row in wrong_df.iterrows():
        packet = _load_packet(_row_cache_path(args.cache_dir, row))
        if not packet:
            continue
        hits = _collect_hits(row, packet, threshold=args.threshold)
        if not hits:
            continue
        rows.append(_summarize_hits(row, packet, hits))

    output_df = pd.DataFrame(rows)
    if not output_df.empty:
        output_df = output_df.sort_values(
            by=["best_score_min", "hit_count", "question_id"],
            ascending=[True, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "architecture_b_hallucination_candidates.csv"
    output_df.to_csv(output_path, index=False)
    print(f"Wrote {len(output_df)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
