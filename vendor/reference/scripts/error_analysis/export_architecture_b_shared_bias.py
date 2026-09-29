"""Export unanimous-wrong Architecture B cases with shared-reasoning similarity.

This script finds rows where all three round-2 model decisions are wrong and
unanimous (same YES/NO answer), then scores the three round-2 reasoning traces
for lexical cosine similarity. It also extracts shared source references from
the reasoning so you can inspect whether the models are leaning on the same
evidence.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


DEFAULT_RESULTS = Path("results/Final_Architecture_B_results.csv")
DEFAULT_OUTPUT_DIR = Path("results/shared_bias_analysis")

TOKEN_RE = re.compile(r"\b[a-z0-9]+\b", re.I)
SOURCE_REF_RE = re.compile(r"\bSource\s+(\d+)\b", re.I)


def _tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text or "") if tok]


def _with_bigrams(tokens: list[str]) -> list[str]:
    if len(tokens) < 2:
        return tokens
    bigrams = [f"{tokens[i]}__{tokens[i + 1]}" for i in range(len(tokens) - 1)]
    return tokens + bigrams


def _cosine_similarity_matrix(texts: list[str]) -> np.ndarray:
    token_docs = [_with_bigrams(_tokenize(text)) for text in texts]
    vocab: dict[str, int] = {}
    for doc in token_docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)

    if not vocab:
        return np.zeros((len(texts), len(texts)), dtype=float)

    n_docs = len(token_docs)
    df = Counter()
    for doc in token_docs:
        df.update(set(doc))

    idf = np.zeros(len(vocab), dtype=float)
    for token, idx in vocab.items():
        idf[idx] = math.log((1 + n_docs) / (1 + df[token])) + 1.0

    vectors = np.zeros((n_docs, len(vocab)), dtype=float)
    for i, doc in enumerate(token_docs):
        counts = Counter(doc)
        doc_len = max(len(doc), 1)
        for token, count in counts.items():
            vectors[i, vocab[token]] = (count / doc_len) * idf[vocab[token]]

    norms = np.linalg.norm(vectors, axis=1)
    sims = np.zeros((n_docs, n_docs), dtype=float)
    for i in range(n_docs):
        for j in range(n_docs):
            denom = norms[i] * norms[j]
            sims[i, j] = float(vectors[i].dot(vectors[j]) / denom) if denom else 0.0
    return sims


def _extract_source_refs(text: str) -> list[str]:
    return SOURCE_REF_RE.findall(text or "")


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _score_row(row: pd.Series) -> dict:
    texts = [
        str(row.get("gpt4o_round2_reasoning", "")),
        str(row.get("deepseek_round2_reasoning", "")),
        str(row.get("llama_round2_reasoning", "")),
    ]
    sims = _cosine_similarity_matrix(texts)
    pairwise = {
        "cos_gpt4o_deepseek": float(sims[0, 1]),
        "cos_gpt4o_llama": float(sims[0, 2]),
        "cos_deepseek_llama": float(sims[1, 2]),
    }
    pairwise["cosine_mean"] = float(np.mean([pairwise[k] for k in pairwise]))
    pairwise["cosine_min"] = float(np.min([pairwise[k] for k in pairwise]))
    pairwise["cosine_max"] = float(np.max([pairwise[k] for k in pairwise]))

    refs = [_extract_source_refs(text) for text in texts]
    common_refs = sorted(set(refs[0]) & set(refs[1]) & set(refs[2]), key=int)
    pairwise["common_source_refs"] = ";".join(common_refs)
    pairwise["common_source_ref_count"] = len(common_refs)
    pairwise["source_ref_jaccard_mean"] = float(
        np.mean([_jaccard(refs[0], refs[1]), _jaccard(refs[0], refs[2]), _jaccard(refs[1], refs[2])])
    )

    pairwise["gpt4o_round2_refs"] = ";".join(refs[0])
    pairwise["deepseek_round2_refs"] = ";".join(refs[1])
    pairwise["llama_round2_refs"] = ";".join(refs[2])
    return pairwise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export unanimous-wrong Architecture B rows ranked by reasoning similarity."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Architecture B results CSV (default: results/Final_Architecture_B_results.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: results/shared_bias_analysis)",
    )
    args = parser.parse_args()

    if not args.results.exists():
        raise FileNotFoundError(f"Results CSV not found: {args.results}")

    df = pd.read_csv(args.results)
    unanimous_wrong = df[
        (df["gpt4o_round2_decision"] == df["deepseek_round2_decision"])
        & (df["gpt4o_round2_decision"] == df["llama_round2_decision"])
        & (df["gpt4o_round2_decision"] != df["ground_truth"])
    ].copy()

    rows = []
    for _, row in unanimous_wrong.iterrows():
        scored = _score_row(row)
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
                "gpt4o_round2_decision": row["gpt4o_round2_decision"],
                "deepseek_round2_decision": row["deepseek_round2_decision"],
                "llama_round2_decision": row["llama_round2_decision"],
                "cos_gpt4o_deepseek": scored["cos_gpt4o_deepseek"],
                "cos_gpt4o_llama": scored["cos_gpt4o_llama"],
                "cos_deepseek_llama": scored["cos_deepseek_llama"],
                "cosine_mean": scored["cosine_mean"],
                "cosine_min": scored["cosine_min"],
                "cosine_max": scored["cosine_max"],
                "source_ref_jaccard_mean": scored["source_ref_jaccard_mean"],
                "common_source_ref_count": scored["common_source_ref_count"],
                "common_source_refs": scored["common_source_refs"],
                "gpt4o_round2_refs": scored["gpt4o_round2_refs"],
                "deepseek_round2_refs": scored["deepseek_round2_refs"],
                "llama_round2_refs": scored["llama_round2_refs"],
                "gpt4o_round2_reasoning": row.get("gpt4o_round2_reasoning", ""),
                "deepseek_round2_reasoning": row.get("deepseek_round2_reasoning", ""),
                "llama_round2_reasoning": row.get("llama_round2_reasoning", ""),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            by=["cosine_mean", "common_source_ref_count", "cosine_min", "question_id"],
            ascending=[False, False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "architecture_b_shared_bias_candidates.csv"
    out.to_csv(output_path, index=False)
    print(f"Wrote {len(out)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
