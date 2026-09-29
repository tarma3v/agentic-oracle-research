"""End-to-end Architecture-B (multi-LLM debate) pipeline test script."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, List, Literal, Optional, Tuple

from tqdm import tqdm

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv

load_dotenv(_project_root / ".env")

from src.data import load_kalshi_questions_csv
from src.agents.openai_agent import GPT4oAgent
from src.resolution.debate import MultiLLMDebateRunner
from src.retrieval import ExaOracleRetriever


EXA_COST_PER_QUESTION = 0.015
OPENAI_AGENT_NAME = GPT4oAgent.AGENT_NAME
LLAMA_AGENT_NAME = "llama-3.3-70b-turbo"
GEMINI_AGENT_NAME = "gemini-2.0-flash"
DEFAULT_QUESTIONS_CSV = "data/kalshibench_v2_evaluation_Filtered.csv"


@dataclass
class QuestionResult:
    row_uid: str
    row_number: int
    retrieval_mode: str
    resolution_time_ms: Optional[float]
    question_id: str
    question_text: str
    resolution_criteria: str
    category: str
    ground_truth: str

    gpt4o_round1_decision: Optional[str]
    gpt4o_round1_confidence: Optional[float]
    gpt4o_round1_latency_ms: Optional[int]
    gpt4o_round1_reasoning: str
    gpt4o_round1_error: Optional[str]
    gpt4o_round2_decision: Optional[str]
    gpt4o_round2_confidence: Optional[float]
    gpt4o_round2_latency_ms: Optional[int]
    gpt4o_round2_reasoning: str
    gpt4o_round2_revised: Optional[bool]
    gpt4o_round2_convergence_notes: Optional[str]
    gpt4o_round2_error: Optional[str]

    claude_round1_decision: Optional[str]
    claude_round1_confidence: Optional[float]
    claude_round1_latency_ms: Optional[int]
    claude_round1_reasoning: str
    claude_round1_error: Optional[str]
    claude_round2_decision: Optional[str]
    claude_round2_confidence: Optional[float]
    claude_round2_latency_ms: Optional[int]
    claude_round2_reasoning: str
    claude_round2_revised: Optional[bool]
    claude_round2_convergence_notes: Optional[str]
    claude_round2_error: Optional[str]

    deepseek_round1_decision: Optional[str]
    deepseek_round1_confidence: Optional[float]
    deepseek_round1_latency_ms: Optional[int]
    deepseek_round1_reasoning: str
    deepseek_round1_error: Optional[str]
    deepseek_round2_decision: Optional[str]
    deepseek_round2_confidence: Optional[float]
    deepseek_round2_latency_ms: Optional[int]
    deepseek_round2_reasoning: str
    deepseek_round2_revised: Optional[bool]
    deepseek_round2_convergence_notes: Optional[str]
    deepseek_round2_error: Optional[str]

    llama_round1_decision: Optional[str]
    llama_round1_confidence: Optional[float]
    llama_round1_latency_ms: Optional[int]
    llama_round1_reasoning: str
    llama_round1_error: Optional[str]
    llama_round2_decision: Optional[str]
    llama_round2_confidence: Optional[float]
    llama_round2_latency_ms: Optional[int]
    llama_round2_reasoning: str
    llama_round2_revised: Optional[bool]
    llama_round2_convergence_notes: Optional[str]
    llama_round2_error: Optional[str]

    final_decision: str
    round1_yes_votes: int
    round1_no_votes: int
    round2_yes_votes: int
    round2_no_votes: int
    tie_break_method: str
    is_correct: bool

    def to_row(self) -> dict:
        return asdict(self)


@dataclass
class RunSummary:
    total_questions: int
    processed_questions: int
    skipped_questions: int
    failed_evidence: int

    accuracy: Optional[float]

    gpt4o_round1_accuracy: Optional[float]
    gpt4o_round2_accuracy: Optional[float]
    claude_round1_accuracy: Optional[float]
    claude_round2_accuracy: Optional[float]
    deepseek_round1_accuracy: Optional[float]
    deepseek_round2_accuracy: Optional[float]
    llama_round1_accuracy: Optional[float]
    llama_round2_accuracy: Optional[float]

    revision_rate: Optional[float]
    round2_unanimous_rate: Optional[float]

    average_total_latency_ms: Optional[float]
    average_round1_agent_latency_ms: Optional[float]
    average_round2_agent_latency_ms: Optional[float]
    average_resolution_time_ms: Optional[float]

    runtime_seconds: float
    estimated_exa_cost_usd: float

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize_ground_truth(value: str) -> str:
    return value.strip().upper()


def _build_row_uid(row_number: int, question) -> str:
    fingerprint = json.dumps(
        {
            "row_number": row_number,
            "question_id": question.id,
            "question_text": question.question,
            "resolution_criteria": question.description,
            "resolution_date": question.close_time,
            "ground_truth": question.ground_truth,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:12]
    return f"row{row_number:04d}_{question.id}_{digest}"


def _accuracy(correct: int, total: int) -> Optional[float]:
    return (correct / total) if total > 0 else None


def _decision_eq(decision: Optional[str], ground_truth: str) -> bool:
    return decision is not None and decision == ground_truth


def _select_questions(
    all_questions,
    question_ids: Optional[List[str]],
    category: Optional[str],
    start_row: int,
    n: int,
    verbose: bool,
):
    if question_ids:
        question_map = {q.id: q for q in all_questions}
        selected = [question_map[qid] for qid in question_ids if qid in question_map]
        if verbose:
            missing = [qid for qid in question_ids if qid not in question_map]
            if missing:
                print(f"Warning: {len(missing)} question IDs not found")
        return selected

    if category:
        filtered = [q for q in all_questions if q.category == category]
    else:
        filtered = list(all_questions)

    start_idx = max(0, start_row - 1)
    if start_idx >= len(filtered):
        return []

    end_idx = min(start_idx + n, len(filtered))
    return filtered[start_idx:end_idx]


def get_evidence(
    question,
    row_uid: str,
    retriever,
    cache_only: bool,
    cache_dir: str,
    retrieval_mode: Literal["full_text", "highlights"],
    num_results: int,
    fulltext_max_characters: int,
    highlights_max_characters: Optional[int],
):
    cache_key = f"{retrieval_mode}__{row_uid}"
    evidence = ExaOracleRetriever.load_from_cache(cache_key, cache_dir=cache_dir)
    if evidence:
        return evidence, True

    if cache_only:
        return None, True

    evidence = retriever.retrieve_and_cache(
        cache_key=cache_key,
        question_id=question.id,
        question_text=question.question,
        resolution_criteria=question.description,
        resolution_date=question.close_time,
        retrieval_mode=retrieval_mode,
        num_results=num_results,
        fulltext_max_characters=fulltext_max_characters,
        highlights_max_characters=highlights_max_characters,
        cache_dir=cache_dir,
    )
    return evidence, False


def process_question(
    question,
    row_uid: str,
    row_number: int,
    retrieval_mode: Literal["full_text", "highlights"],
    evidence,
    runner: MultiLLMDebateRunner,
    use_claude: bool,
    use_gemini: bool,
) -> Tuple[QuestionResult, int]:
    aggregated = runner.resolve(evidence)
    traces = {trace.agent_name: trace for trace in aggregated.agent_traces}

    gpt4o = traces.get(OPENAI_AGENT_NAME) or traces.get("gpt-4o")
    claude = traces.get("claude-haiku") if use_claude else None
    deepseek = traces.get("deepseek-v3") if not use_claude else None
    third_agent_name = GEMINI_AGENT_NAME if use_gemini else LLAMA_AGENT_NAME
    third_agent = traces.get(third_agent_name)

    ground_truth = _normalize_ground_truth(question.ground_truth)
    final_decision = aggregated.final_decision.value

    def _round_decision(trace, round_num: int) -> Optional[str]:
        if trace is None:
            return None
        response = trace.round1 if round_num == 1 else trace.round2
        return response.decision.value if response.decision else None

    def _round_conf(trace, round_num: int) -> Optional[float]:
        if trace is None:
            return None
        return trace.round1.confidence if round_num == 1 else trace.round2.confidence

    def _round_reasoning(trace, round_num: int) -> str:
        if trace is None:
            return ""
        return trace.round1.reasoning if round_num == 1 else trace.round2.reasoning

    def _round_latency(trace, round_num: int) -> Optional[int]:
        if trace is None:
            return None
        return trace.round1.latency_ms if round_num == 1 else trace.round2.latency_ms

    def _round_error(trace, round_num: int) -> Optional[str]:
        if trace is None:
            return "missing"
        return trace.round1.error if round_num == 1 else trace.round2.error

    def _round_revised(trace) -> Optional[bool]:
        return trace.round2.revised if trace else None

    def _round_conv_notes(trace) -> Optional[str]:
        return trace.round2.convergence_notes if trace else None

    result = QuestionResult(
        row_uid=row_uid,
        row_number=row_number,
        retrieval_mode=retrieval_mode,
        resolution_time_ms=None,
        question_id=question.id,
        question_text=question.question,
        resolution_criteria=question.description,
        category=question.category,
        ground_truth=ground_truth,

        gpt4o_round1_decision=_round_decision(gpt4o, 1),
        gpt4o_round1_confidence=_round_conf(gpt4o, 1),
        gpt4o_round1_latency_ms=_round_latency(gpt4o, 1),
        gpt4o_round1_reasoning=_round_reasoning(gpt4o, 1),
        gpt4o_round1_error=_round_error(gpt4o, 1),
        gpt4o_round2_decision=_round_decision(gpt4o, 2),
        gpt4o_round2_confidence=_round_conf(gpt4o, 2),
        gpt4o_round2_latency_ms=_round_latency(gpt4o, 2),
        gpt4o_round2_reasoning=_round_reasoning(gpt4o, 2),
        gpt4o_round2_revised=_round_revised(gpt4o),
        gpt4o_round2_convergence_notes=_round_conv_notes(gpt4o),
        gpt4o_round2_error=_round_error(gpt4o, 2),

        claude_round1_decision=_round_decision(claude, 1),
        claude_round1_confidence=_round_conf(claude, 1),
        claude_round1_latency_ms=_round_latency(claude, 1),
        claude_round1_reasoning=_round_reasoning(claude, 1),
        claude_round1_error=_round_error(claude, 1),
        claude_round2_decision=_round_decision(claude, 2),
        claude_round2_confidence=_round_conf(claude, 2),
        claude_round2_latency_ms=_round_latency(claude, 2),
        claude_round2_reasoning=_round_reasoning(claude, 2),
        claude_round2_revised=_round_revised(claude),
        claude_round2_convergence_notes=_round_conv_notes(claude),
        claude_round2_error=_round_error(claude, 2),

        deepseek_round1_decision=_round_decision(deepseek, 1),
        deepseek_round1_confidence=_round_conf(deepseek, 1),
        deepseek_round1_latency_ms=_round_latency(deepseek, 1),
        deepseek_round1_reasoning=_round_reasoning(deepseek, 1),
        deepseek_round1_error=_round_error(deepseek, 1),
        deepseek_round2_decision=_round_decision(deepseek, 2),
        deepseek_round2_confidence=_round_conf(deepseek, 2),
        deepseek_round2_latency_ms=_round_latency(deepseek, 2),
        deepseek_round2_reasoning=_round_reasoning(deepseek, 2),
        deepseek_round2_revised=_round_revised(deepseek),
        deepseek_round2_convergence_notes=_round_conv_notes(deepseek),
        deepseek_round2_error=_round_error(deepseek, 2),

        llama_round1_decision=_round_decision(third_agent, 1),
        llama_round1_confidence=_round_conf(third_agent, 1),
        llama_round1_latency_ms=_round_latency(third_agent, 1),
        llama_round1_reasoning=_round_reasoning(third_agent, 1),
        llama_round1_error=_round_error(third_agent, 1),
        llama_round2_decision=_round_decision(third_agent, 2),
        llama_round2_confidence=_round_conf(third_agent, 2),
        llama_round2_latency_ms=_round_latency(third_agent, 2),
        llama_round2_reasoning=_round_reasoning(third_agent, 2),
        llama_round2_revised=_round_revised(third_agent),
        llama_round2_convergence_notes=_round_conv_notes(third_agent),
        llama_round2_error=_round_error(third_agent, 2),

        final_decision=final_decision,
        round1_yes_votes=aggregated.round1_yes_votes,
        round1_no_votes=aggregated.round1_no_votes,
        round2_yes_votes=aggregated.round2_yes_votes,
        round2_no_votes=aggregated.round2_no_votes,
        tie_break_method=aggregated.tie_break_method,
        is_correct=(final_decision == ground_truth),
    )
    return result, aggregated.total_latency_ms


def compute_summary(
    results: List[QuestionResult],
    runtime_seconds: float,
    total_questions: int,
    skipped_questions: int,
    failed_evidence: int,
    total_latency_ms: List[int],
    resolution_time_ms: List[float],
    exa_cost_usd: float,
    use_claude: bool,
) -> RunSummary:
    processed_questions = len(results)
    overall_correct = sum(1 for r in results if r.is_correct)

    gpt4o_r1_total = sum(1 for r in results if r.gpt4o_round1_decision)
    gpt4o_r1_correct = sum(
        1 for r in results if _decision_eq(r.gpt4o_round1_decision, r.ground_truth)
    )
    gpt4o_r2_total = sum(1 for r in results if r.gpt4o_round2_decision)
    gpt4o_r2_correct = sum(
        1 for r in results if _decision_eq(r.gpt4o_round2_decision, r.ground_truth)
    )

    claude_r1_total = sum(1 for r in results if r.claude_round1_decision) if use_claude else 0
    claude_r1_correct = (
        sum(1 for r in results if _decision_eq(r.claude_round1_decision, r.ground_truth))
        if use_claude
        else 0
    )
    claude_r2_total = sum(1 for r in results if r.claude_round2_decision) if use_claude else 0
    claude_r2_correct = (
        sum(1 for r in results if _decision_eq(r.claude_round2_decision, r.ground_truth))
        if use_claude
        else 0
    )

    deepseek_r1_total = (
        sum(1 for r in results if r.deepseek_round1_decision) if not use_claude else 0
    )
    deepseek_r1_correct = (
        sum(1 for r in results if _decision_eq(r.deepseek_round1_decision, r.ground_truth))
        if not use_claude
        else 0
    )
    deepseek_r2_total = (
        sum(1 for r in results if r.deepseek_round2_decision) if not use_claude else 0
    )
    deepseek_r2_correct = (
        sum(1 for r in results if _decision_eq(r.deepseek_round2_decision, r.ground_truth))
        if not use_claude
        else 0
    )

    llama_r1_total = sum(1 for r in results if r.llama_round1_decision)
    llama_r1_correct = sum(
        1 for r in results if _decision_eq(r.llama_round1_decision, r.ground_truth)
    )
    llama_r2_total = sum(1 for r in results if r.llama_round2_decision)
    llama_r2_correct = sum(
        1 for r in results if _decision_eq(r.llama_round2_decision, r.ground_truth)
    )

    revised_flags = []
    for r in results:
        if r.gpt4o_round2_revised is not None:
            revised_flags.append(r.gpt4o_round2_revised)
        if use_claude and r.claude_round2_revised is not None:
            revised_flags.append(r.claude_round2_revised)
        if not use_claude and r.deepseek_round2_revised is not None:
            revised_flags.append(r.deepseek_round2_revised)
        if r.llama_round2_revised is not None:
            revised_flags.append(r.llama_round2_revised)

    round2_unanimous = sum(1 for r in results if (r.round2_yes_votes == 3 or r.round2_no_votes == 3))

    avg_total_latency = sum(total_latency_ms) / len(total_latency_ms) if total_latency_ms else None
    avg_resolution_time_ms = (
        sum(resolution_time_ms) / len(resolution_time_ms) if resolution_time_ms else None
    )
    round1_latencies = [
        latency
        for r in results
        for latency in (
            r.gpt4o_round1_latency_ms,
            r.claude_round1_latency_ms if use_claude else r.deepseek_round1_latency_ms,
            r.llama_round1_latency_ms,
        )
        if latency is not None
    ]
    round2_latencies = [
        latency
        for r in results
        for latency in (
            r.gpt4o_round2_latency_ms,
            r.claude_round2_latency_ms if use_claude else r.deepseek_round2_latency_ms,
            r.llama_round2_latency_ms,
        )
        if latency is not None
    ]

    return RunSummary(
        total_questions=total_questions,
        processed_questions=processed_questions,
        skipped_questions=skipped_questions,
        failed_evidence=failed_evidence,
        accuracy=_accuracy(overall_correct, processed_questions),
        gpt4o_round1_accuracy=_accuracy(gpt4o_r1_correct, gpt4o_r1_total),
        gpt4o_round2_accuracy=_accuracy(gpt4o_r2_correct, gpt4o_r2_total),
        claude_round1_accuracy=_accuracy(claude_r1_correct, claude_r1_total),
        claude_round2_accuracy=_accuracy(claude_r2_correct, claude_r2_total),
        deepseek_round1_accuracy=_accuracy(deepseek_r1_correct, deepseek_r1_total),
        deepseek_round2_accuracy=_accuracy(deepseek_r2_correct, deepseek_r2_total),
        llama_round1_accuracy=_accuracy(llama_r1_correct, llama_r1_total),
        llama_round2_accuracy=_accuracy(llama_r2_correct, llama_r2_total),
        revision_rate=(sum(1 for flag in revised_flags if flag) / len(revised_flags))
        if revised_flags
        else None,
        round2_unanimous_rate=_accuracy(round2_unanimous, processed_questions),
        average_total_latency_ms=avg_total_latency,
        average_round1_agent_latency_ms=(
            sum(round1_latencies) / len(round1_latencies) if round1_latencies else None
        ),
        average_round2_agent_latency_ms=(
            sum(round2_latencies) / len(round2_latencies) if round2_latencies else None
        ),
        average_resolution_time_ms=avg_resolution_time_ms,
        runtime_seconds=runtime_seconds,
        estimated_exa_cost_usd=exa_cost_usd,
    )


def _row_for_csv(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if isinstance(v, str) and v:
            out[k] = v.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
        else:
            out[k] = v
    return out


def _question_result_fieldnames(use_claude: bool) -> List[str]:
    hidden_prefix = "deepseek_" if use_claude else "claude_"
    fields = list(QuestionResult.__dataclass_fields__.keys())
    return [field for field in fields if not field.startswith(hidden_prefix)]


def _summary_fieldnames(use_claude: bool) -> List[str]:
    hidden_prefix = "deepseek_" if use_claude else "claude_"
    fields = list(RunSummary.__dataclass_fields__.keys())
    return [field for field in fields if not field.startswith(hidden_prefix)]


def write_csv(results: Iterable[QuestionResult], path: str, use_claude: bool) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = _question_result_fieldnames(use_claude=use_claude)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for result in results:
            filtered = {field: result.to_row().get(field) for field in fieldnames}
            writer.writerow(_row_for_csv(filtered))


def write_json(
    results: Iterable[QuestionResult],
    summary: RunSummary,
    path: str,
    use_claude: bool,
    run_config: Optional[dict[str, Any]] = None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_fields = _question_result_fieldnames(use_claude=use_claude)
    summary_fields = _summary_fieldnames(use_claude=use_claude)
    payload = {
        "run_config": run_config or {},
        "summary": {field: summary.to_dict().get(field) for field in summary_fields},
        "results": [{field: r.to_row().get(field) for field in result_fields} for r in results],
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def print_summary(summary: RunSummary, use_claude: bool, use_gemini: bool) -> None:
    print("\n" + "=" * 60)
    print("Run summary")
    print(f"Total questions: {summary.total_questions}")
    print(f"Processed: {summary.processed_questions}")
    print(f"Skipped (cache-only): {summary.skipped_questions}")
    print(f"Evidence failures: {summary.failed_evidence}")

    if summary.accuracy is not None:
        print(f"Overall accuracy: {summary.accuracy:.3f}")

    if summary.gpt4o_round1_accuracy is not None:
        print(f"{OPENAI_AGENT_NAME} round 1 accuracy: {summary.gpt4o_round1_accuracy:.3f}")
    if summary.gpt4o_round2_accuracy is not None:
        print(f"{OPENAI_AGENT_NAME} round 2 accuracy: {summary.gpt4o_round2_accuracy:.3f}")

    provider_label = "Claude" if use_claude else "DeepSeek"
    provider_r1 = summary.claude_round1_accuracy if use_claude else summary.deepseek_round1_accuracy
    provider_r2 = summary.claude_round2_accuracy if use_claude else summary.deepseek_round2_accuracy
    if provider_r1 is not None:
        print(f"{provider_label} round 1 accuracy: {provider_r1:.3f}")
    if provider_r2 is not None:
        print(f"{provider_label} round 2 accuracy: {provider_r2:.3f}")

    if summary.llama_round1_accuracy is not None:
        print(
            f"{'Gemini' if use_gemini else 'Llama'} round 1 accuracy: {summary.llama_round1_accuracy:.3f}"
        )
    if summary.llama_round2_accuracy is not None:
        print(
            f"{'Gemini' if use_gemini else 'Llama'} round 2 accuracy: {summary.llama_round2_accuracy:.3f}"
        )

    if summary.revision_rate is not None:
        print(f"Revision rate: {summary.revision_rate:.3f}")
    if summary.round2_unanimous_rate is not None:
        print(f"Round 2 unanimous rate: {summary.round2_unanimous_rate:.3f}")

    if summary.average_total_latency_ms is not None:
        print(f"Avg total latency (ms): {summary.average_total_latency_ms:.1f}")
    if summary.average_round1_agent_latency_ms is not None:
        print(f"Avg round 1 agent latency (ms): {summary.average_round1_agent_latency_ms:.1f}")
    if summary.average_round2_agent_latency_ms is not None:
        print(f"Avg round 2 agent latency (ms): {summary.average_round2_agent_latency_ms:.1f}")
    if summary.average_resolution_time_ms is not None:
        print(f"Avg resolution time (ms): {summary.average_resolution_time_ms:.1f}")

    print(f"Runtime (s): {summary.runtime_seconds:.1f}")
    print(f"Estimated Exa cost (USD): {summary.estimated_exa_cost_usd:.2f}")
    print("=" * 60 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Architecture-B end-to-end pipeline test")
    parser.add_argument("-n", type=int, default=10, help="Number of questions (default: 10)")
    parser.add_argument(
        "--start-row",
        type=int,
        default=1,
        help="1-indexed starting row in selected CSV/category view (default: 1)",
    )
    parser.add_argument(
        "--questions-csv",
        default=DEFAULT_QUESTIONS_CSV,
        help="Questions CSV path (default: data/kalshibench_v2_evaluation_Filtered.csv)",
    )
    parser.add_argument("--question-ids", nargs="+", help="Specific question IDs (overrides -n)")
    parser.add_argument("--category", help="Filter by category")
    parser.add_argument("--cache-only", action="store_true", help="Only use cached evidence")
    parser.add_argument("--cache-dir", default="cache/evidence", help="Evidence cache directory")
    parser.add_argument(
        "--retrieval-mode",
        choices=["full_text", "highlights"],
        default="highlights",
        help="Exa retrieval mode (default: highlights)",
    )
    parser.add_argument(
        "--num-results",
        type=int,
        default=10,
        help="Number of Exa results per question (default: 10)",
    )
    parser.add_argument(
        "--fulltext-max-characters",
        type=int,
        default=4000,
        help="Max characters per source for full_text mode",
    )
    parser.add_argument(
        "--highlights-max-characters",
        type=int,
        default=2000,
        help="Max characters for Exa highlights. Set 0 to use Exa defaults.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="results/architecture_b_results.csv",
        help="Output CSV path",
    )
    parser.add_argument("--json", dest="json_output", help="Optional JSON with summary stats")
    parser.add_argument(
        "--use-claude",
        action="store_true",
        help="Use Claude instead of DeepSeek for the third agent slot.",
    )
    parser.add_argument(
        "--use-gemini",
        action="store_true",
        help="Use Gemini instead of Llama for the third agent slot.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would run without executing")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose progress output")
    parser.add_argument(
        "--cache-start-row",
        type=int,
        default=None,
        help=(
            "1-based row number to start cache keys/row_uid at without skipping rows. "
            "If set, all rows are processed but cache keys start at this value."
        ),
    )

    args = parser.parse_args()
    if args.start_row < 1:
        print("Error: --start-row must be >= 1")
        return 1
    if args.cache_start_row is not None and args.cache_start_row < 1:
        print("Error: --cache-start-row must be >= 1")
        return 1

    print(f"Loading questions from CSV: {args.questions_csv}")
    all_questions = load_kalshi_questions_csv(args.questions_csv)
    questions = _select_questions(
        all_questions,
        question_ids=args.question_ids,
        category=args.category,
        start_row=args.start_row,
        n=args.n,
        verbose=args.verbose,
    )

    if args.dry_run:
        print(f"Dry run: {len(questions)} questions selected.")
        print(f"Start row: {args.start_row}")
        if args.cache_start_row is not None:
            print(f"Cache start row: {args.cache_start_row}")
        if questions:
            end_row = args.start_row + len(questions) - 1
            print(f"Selected row range: {args.start_row}-{end_row}")
        if args.question_ids:
            print(f"Question IDs: {', '.join(q.id for q in questions)}")
        if args.category:
            print(f"Category filter: {args.category}")
        print(f"Questions CSV: {args.questions_csv}")
        print(f"Retrieval mode: {args.retrieval_mode}")
        second_agent = "Claude" if args.use_claude else "DeepSeek"
        third_agent = "Gemini" if args.use_gemini else "Llama"
        print(f"Agents: {OPENAI_AGENT_NAME}, {second_agent}, {third_agent}")
        print(f"Exa num results: {args.num_results}")
        if args.cache_only:
            print("Cache-only mode enabled.")
        else:
            print(f"Full-text max characters: {args.fulltext_max_characters}")
            highlights_max = (
                "Exa defaults" if args.highlights_max_characters == 0 else args.highlights_max_characters
            )
            print(f"Highlights max characters: {highlights_max}")
        print(f"Cache dir: {args.cache_dir}")
        print(f"Would write CSV to: {args.output}")
        if args.json_output:
            print(f"Would write JSON to: {args.json_output}")
        return 0

    if not questions:
        print("No questions selected. Exiting.")
        return 1

    if args.cache_only:
        print("Cache-only mode enabled. Cache misses will be skipped.")
    else:
        estimated_cost = len(questions) * EXA_COST_PER_QUESTION
        print(f"Estimated Exa cost if no cache hits: ${estimated_cost:.2f}")

    retriever = None
    if not args.cache_only:
        print("Initializing Exa retriever...")
        retriever = ExaOracleRetriever()

    if args.use_claude and args.use_gemini:
        from src.agents.anthropic_agent import ClaudeAgent
        from src.agents.google_agent import GeminiAgent

        runner = MultiLLMDebateRunner(agents=[GPT4oAgent(), ClaudeAgent(), GeminiAgent()])
    elif args.use_claude:
        from src.agents.anthropic_agent import ClaudeAgent
        from src.agents.llama_agent import LlamaAgent

        runner = MultiLLMDebateRunner(agents=[GPT4oAgent(), ClaudeAgent(), LlamaAgent()])
    elif args.use_gemini:
        from src.agents.deepseek_agent import DeepSeekAgent
        from src.agents.google_agent import GeminiAgent

        runner = MultiLLMDebateRunner(agents=[GPT4oAgent(), DeepSeekAgent(), GeminiAgent()])
    else:
        runner = MultiLLMDebateRunner()

    total_latency_ms: List[int] = []
    resolution_time_ms: List[float] = []
    results: List[QuestionResult] = []
    skipped_questions = 0
    failed_evidence = 0
    exa_cache_hits = 0
    exa_retrievals = 0

    print(
        f"Starting run for {len(questions)} questions "
        f"({'cache-only' if args.cache_only else f'cache + retrieval ({args.retrieval_mode})'})."
    )

    start_time = time.perf_counter()
    highlights_max_characters = (
        None if args.highlights_max_characters == 0 else args.highlights_max_characters
    )

    cache_start = args.cache_start_row or args.start_row
    for row_number, question in enumerate(
        tqdm(questions, desc="Processing questions", disable=not args.verbose),
        start=cache_start,
    ):
        question_start = time.perf_counter()
        row_uid = _build_row_uid(row_number=row_number, question=question)
        try:
            evidence, from_cache = get_evidence(
                question=question,
                row_uid=row_uid,
                retriever=retriever,
                cache_only=args.cache_only,
                cache_dir=args.cache_dir,
                retrieval_mode=args.retrieval_mode,
                num_results=args.num_results,
                fulltext_max_characters=args.fulltext_max_characters,
                highlights_max_characters=highlights_max_characters,
            )
        except Exception as exc:
            failed_evidence += 1
            print(f"Evidence retrieval failed for {question.id} ({row_uid}): {exc}")
            continue

        if evidence is None:
            skipped_questions += 1
            if args.verbose:
                print(f"Skipping {question.id} ({row_uid}): cache miss in cache-only mode")
            continue

        if from_cache:
            exa_cache_hits += 1
        else:
            exa_retrievals += 1

        result, latency_ms = process_question(
            question=question,
            row_uid=row_uid,
            row_number=row_number,
            retrieval_mode=args.retrieval_mode,
            evidence=evidence,
            runner=runner,
            use_claude=args.use_claude,
            use_gemini=args.use_gemini,
        )
        result.resolution_time_ms = (time.perf_counter() - question_start) * 1000.0
        results.append(result)
        total_latency_ms.append(latency_ms)
        resolution_time_ms.append(result.resolution_time_ms)

    runtime_seconds = time.perf_counter() - start_time
    estimated_exa_cost = exa_retrievals * EXA_COST_PER_QUESTION

    summary = compute_summary(
        results=results,
        runtime_seconds=runtime_seconds,
        total_questions=len(questions),
        skipped_questions=skipped_questions,
        failed_evidence=failed_evidence,
        total_latency_ms=total_latency_ms,
        resolution_time_ms=resolution_time_ms,
        exa_cost_usd=estimated_exa_cost,
        use_claude=args.use_claude,
    )

    write_csv(results, args.output, use_claude=args.use_claude)
    if args.json_output:
        write_json(
            results,
            summary,
            args.json_output,
            use_claude=args.use_claude,
            run_config={
                "retrieval_mode": args.retrieval_mode,
                "num_results": args.num_results,
                "fulltext_max_characters": args.fulltext_max_characters,
                "highlights_max_characters": highlights_max_characters,
                "cache_only": args.cache_only,
                "cache_dir": args.cache_dir,
                "start_row": args.start_row,
                "n": args.n,
                "category": args.category,
                "question_ids": args.question_ids or [],
                "use_claude": args.use_claude,
                "use_gemini": args.use_gemini,
            },
        )

    if args.verbose:
        print(f"Cache hits: {exa_cache_hits}")
        print(f"Exa retrievals: {exa_retrievals}")
    print_summary(summary, use_claude=args.use_claude, use_gemini=args.use_gemini)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
