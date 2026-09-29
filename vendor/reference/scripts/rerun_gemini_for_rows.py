"""
Rerun Gemini agent for specific row_numbers in an Architecture-A results CSV.

Intended use: fill in missing gemini_decision/gemini_reasoning after transient API failures,
then recompute majority-vote final_decision/yes_votes/no_votes/is_correct for those rows.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional
import time

import pandas as pd
from dotenv import load_dotenv

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from src.agents.base import AgentResolution, Decision
from src.agents.google_agent import GeminiAgent
from src.retrieval import ExaOracleRetriever
from src.resolution.aggregation import MajorityVoteStrategy


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _backup_file(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    backup = path.with_suffix(path.suffix + f".bak_{_now_stamp()}")
    shutil.copy2(path, backup)
    return backup


def _parse_rows(value: str) -> list[int]:
    rows: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        rows.append(int(part))
    return rows


def _as_optional_str(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str) and (value.strip().lower() == "nan" or value.strip() == ""):
        return None
    return str(value)


def _as_optional_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _as_optional_decision(value) -> Optional[Decision]:
    text = _as_optional_str(value)
    if not text:
        return None
    text = text.strip().upper()
    if text not in ("YES", "NO"):
        return None
    return Decision(text)


def _reconstruct_agent_resolution(
    *,
    agent_name: str,
    question_id: str,
    decision_value,
    confidence_value,
    reasoning_value,
    error_value,
) -> AgentResolution:
    return AgentResolution(
        agent_name=agent_name,
        question_id=question_id,
        decision=_as_optional_decision(decision_value),
        confidence=_as_optional_float(confidence_value),
        reasoning=_as_optional_str(reasoning_value) or "",
        timestamp=datetime.utcnow().isoformat() + "Z",
        latency_ms=0,
        error=_as_optional_str(error_value),
    )


def _iter_target_indices(df: pd.DataFrame, row_numbers: Iterable[int]) -> list[int]:
    if "row_number" not in df.columns:
        raise ValueError("CSV missing required column: row_number")
    wanted = set(int(r) for r in row_numbers)
    indices = df.index[df["row_number"].astype(int).isin(wanted)].tolist()
    missing = sorted(wanted - set(int(df.loc[i, "row_number"]) for i in indices))
    if missing:
        raise ValueError(f"Row numbers not found in CSV: {missing}")
    return indices


def _cache_key_for_row(row: pd.Series) -> str:
    retrieval_mode = _as_optional_str(row.get("retrieval_mode")) or "highlights"
    row_uid = _as_optional_str(row.get("row_uid"))
    if not row_uid:
        raise ValueError("CSV row missing row_uid")
    if row_uid.startswith(f"{retrieval_mode}__"):
        return row_uid
    return f"{retrieval_mode}__{row_uid}"


def _update_json_payload(
    *,
    json_path: Path,
    updated_rows: dict[str, dict],
) -> None:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    results = payload.get("results", [])
    for item in results:
        row_uid = item.get("row_uid")
        if row_uid in updated_rows:
            item.update(updated_rows[row_uid])
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _should_retry(error: str) -> bool:
    lowered = error.lower()
    return any(
        token in lowered
        for token in (
            "resource_exhausted",
            "429",
            "rate",
            "try again later",
            "temporarily unavailable",
            "nodename nor servname provided",
            "name or service not known",
            "temporary failure in name resolution",
            "timed out",
            "timeout",
            "connection reset",
            "connection aborted",
            "connection refused",
        )
    )


def _rerun_gemini_with_retries(
    gemini: GeminiAgent,
    evidence,
    *,
    max_attempts: int = 3,
) -> AgentResolution:
    delays = [10.0, 30.0, 60.0]
    last: AgentResolution | None = None
    for attempt in range(max_attempts):
        res = gemini.resolve(evidence)
        last = res
        if res.error is None and res.decision is not None:
            return res
        if res.error and _should_retry(res.error) and attempt < max_attempts - 1:
            time.sleep(delays[min(attempt, len(delays) - 1)])
            continue
        return res
    return last if last is not None else gemini.resolve(evidence)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rerun Gemini for selected row_numbers and update Architecture-A CSV."
    )
    parser.add_argument(
        "--csv",
        default="results/architecture_a_results/architecture_a_first100.csv",
        help="Path to Architecture-A results CSV to overwrite",
    )
    parser.add_argument(
        "--rows",
        required=True,
        help='Comma-separated row_numbers (e.g. "3,5,14")',
    )
    parser.add_argument(
        "--cache-dir",
        default="cache/evidence",
        help="Evidence cache directory used by scripts/test_architecture_a.py",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="Optional JSON results file to update in-place as well",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    load_dotenv(project_root / ".env")

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    row_numbers = _parse_rows(args.rows)
    df = pd.read_csv(csv_path)

    target_indices = _iter_target_indices(df, row_numbers=row_numbers)

    gemini = GeminiAgent()
    aggregator = MajorityVoteStrategy()

    updated_rows_for_json: dict[str, dict] = {}
    updated_count = 0
    skipped_count = 0

    for idx in target_indices:
        row = df.loc[idx]
        question_id = _as_optional_str(row.get("question_id")) or "unknown"
        cache_key = _cache_key_for_row(row)

        evidence = ExaOracleRetriever.load_from_cache(cache_key, cache_dir=args.cache_dir)
        if evidence is None:
            raise FileNotFoundError(
                f"Missing cached evidence for row_number={int(row['row_number'])}: "
                f"{Path(args.cache_dir) / (cache_key + '.json')}"
            )

        gemini_res = _rerun_gemini_with_retries(gemini, evidence)
        if gemini_res.error is not None or gemini_res.decision is None:
            skipped_count += 1
            continue

        df.at[idx, "gemini_decision"] = gemini_res.decision.value
        df.at[idx, "gemini_confidence"] = gemini_res.confidence
        df.at[idx, "gemini_reasoning"] = gemini_res.reasoning
        df.at[idx, "gemini_error"] = gemini_res.error

        gpt_res = _reconstruct_agent_resolution(
            agent_name=_as_optional_str(row.get("gpt4o_agent_name"))
            or _as_optional_str(row.get("gpt4o_model"))
            or "gpt",
            question_id=question_id,
            decision_value=row.get("gpt4o_decision"),
            confidence_value=row.get("gpt4o_confidence"),
            reasoning_value=row.get("gpt4o_reasoning"),
            error_value=row.get("gpt4o_error"),
        )
        deepseek_res = _reconstruct_agent_resolution(
            agent_name="deepseek-v3",
            question_id=question_id,
            decision_value=row.get("deepseek_decision"),
            confidence_value=row.get("deepseek_confidence"),
            reasoning_value=row.get("deepseek_reasoning"),
            error_value=row.get("deepseek_error"),
        )

        aggregated = aggregator.aggregate([gpt_res, deepseek_res, gemini_res], total_latency_ms=0)
        df.at[idx, "final_decision"] = aggregated.final_decision.value
        df.at[idx, "yes_votes"] = aggregated.yes_votes
        df.at[idx, "no_votes"] = aggregated.no_votes

        ground_truth = _as_optional_str(row.get("ground_truth"))
        if ground_truth:
            df.at[idx, "is_correct"] = aggregated.final_decision.value == ground_truth.strip().upper()

        row_uid = _as_optional_str(row.get("row_uid"))
        if row_uid:
            updated_rows_for_json[row_uid] = {
                "gemini_decision": df.at[idx, "gemini_decision"],
                "gemini_confidence": df.at[idx, "gemini_confidence"],
                "gemini_reasoning": df.at[idx, "gemini_reasoning"],
                "gemini_error": df.at[idx, "gemini_error"],
                "final_decision": df.at[idx, "final_decision"],
                "yes_votes": int(df.at[idx, "yes_votes"]),
                "no_votes": int(df.at[idx, "no_votes"]),
                "is_correct": bool(df.at[idx, "is_correct"]),
            }
        updated_count += 1

    json_path = Path(args.json_path) if args.json_path else None

    if args.dry_run:
        print(
            f"Dry run: would overwrite {csv_path} and attempt {len(target_indices)} rows "
            f"(update={updated_count}, skipped={skipped_count})."
        )
        if json_path:
            print(f"Dry run: would also update JSON {json_path}.")
        return 0

    _backup_file(csv_path)
    df.to_csv(csv_path, index=False)

    if json_path:
        if not json_path.exists():
            raise FileNotFoundError(f"JSON not found: {json_path}")
        _backup_file(json_path)
        _update_json_payload(json_path=json_path, updated_rows=updated_rows_for_json)

    print(
        f"Attempted {len(target_indices)} rows: updated={updated_count}, skipped={skipped_count}. "
        f"Overwrote {csv_path}."
    )
    if json_path:
        print(f"Also updated JSON: {json_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
