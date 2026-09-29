"""End-to-end Architecture-A pipeline test script."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, List, Literal, Optional, Tuple

from tqdm import tqdm

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv

load_dotenv(_project_root / ".env")

from src.data import load_kalshi_questions_csv
from src.agents.openai_agent import GPT4oAgent
from src.retrieval import ExaOracleRetriever
from src.resolution.runner import MultiAgentRunner


EXA_COST_PER_QUESTION = 0.015
OPENAI_AGENT_NAME = GPT4oAgent.AGENT_NAME
LLAMA_AGENT_NAME = "llama-3.3-70b-turbo"
GEMINI_AGENT_NAME = "gemini-2.0-flash"
DEFAULT_QUESTIONS_CSV = "data/kalshibench_v2_evaluation_Filtered.csv"


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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
    gpt4o_decision: Optional[str]
    gpt4o_confidence: Optional[float]
    gpt4o_reasoning: str
    gpt4o_error: Optional[str]
    claude_decision: Optional[str]
    claude_confidence: Optional[float]
    claude_reasoning: str
    claude_error: Optional[str]
    deepseek_decision: Optional[str]
    deepseek_confidence: Optional[float]
    deepseek_reasoning: str
    deepseek_error: Optional[str]
    llama_decision: Optional[str]
    llama_confidence: Optional[float]
    llama_reasoning: str
    llama_error: Optional[str]
    final_decision: str
    yes_votes: int
    no_votes: int
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
    gpt4o_accuracy: Optional[float]
    claude_accuracy: Optional[float]
    deepseek_accuracy: Optional[float]
    llama_accuracy: Optional[float]
    unanimous_rate: Optional[float]
    majority_rate: Optional[float]
    average_latency_ms: Optional[float]
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
    end_idx = start_idx + n
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
    runner: MultiAgentRunner,
    use_claude: bool,
    use_gemini: bool,
) -> Tuple[QuestionResult, int]:
    aggregated = runner.resolve(evidence)
    agent_results = {r.agent_name: r for r in aggregated.agent_resolutions}

    gpt4o = agent_results.get(OPENAI_AGENT_NAME) or agent_results.get("gpt-4o")
    claude = agent_results.get("claude-haiku") if use_claude else None
    deepseek = agent_results.get("deepseek-v3") if not use_claude else None
    third_agent_name = GEMINI_AGENT_NAME if use_gemini else LLAMA_AGENT_NAME
    third_agent = agent_results.get(third_agent_name)

    ground_truth = _normalize_ground_truth(question.ground_truth)
    final_decision = aggregated.final_decision.value
    is_correct = final_decision == ground_truth

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
        gpt4o_decision=gpt4o.decision.value if gpt4o and gpt4o.decision else None,
        gpt4o_confidence=gpt4o.confidence if gpt4o else None,
        gpt4o_reasoning=gpt4o.reasoning if gpt4o else "",
        gpt4o_error=gpt4o.error if gpt4o else "missing",
        claude_decision=claude.decision.value if claude and claude.decision else None,
        claude_confidence=claude.confidence if claude else None,
        claude_reasoning=claude.reasoning if claude else "",
        claude_error=claude.error if claude else "missing",
        deepseek_decision=deepseek.decision.value if deepseek and deepseek.decision else None,
        deepseek_confidence=deepseek.confidence if deepseek else None,
        deepseek_reasoning=deepseek.reasoning if deepseek else "",
        deepseek_error=deepseek.error if deepseek else "missing",
        llama_decision=third_agent.decision.value if third_agent and third_agent.decision else None,
        llama_confidence=third_agent.confidence if third_agent else None,
        llama_reasoning=third_agent.reasoning if third_agent else "",
        llama_error=third_agent.error if third_agent else "missing",
        final_decision=final_decision,
        yes_votes=aggregated.yes_votes,
        no_votes=aggregated.no_votes,
        is_correct=is_correct,
    )
    return result, aggregated.total_latency_ms


def _accuracy(correct: int, total: int) -> Optional[float]:
    return (correct / total) if total > 0 else None


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
    correct = sum(1 for r in results if r.is_correct)

    gpt4o_total = sum(1 for r in results if r.gpt4o_decision)
    gpt4o_correct = sum(
        1 for r in results if r.gpt4o_decision and r.gpt4o_decision == r.ground_truth
    )
    claude_total = sum(1 for r in results if r.claude_decision) if use_claude else 0
    claude_correct = (
        sum(1 for r in results if r.claude_decision and r.claude_decision == r.ground_truth)
        if use_claude
        else 0
    )
    deepseek_total = sum(1 for r in results if r.deepseek_decision) if not use_claude else 0
    deepseek_correct = (
        sum(1 for r in results if r.deepseek_decision and r.deepseek_decision == r.ground_truth)
        if not use_claude
        else 0
    )
    llama_total = sum(1 for r in results if r.llama_decision)
    llama_correct = sum(
        1 for r in results if r.llama_decision and r.llama_decision == r.ground_truth
    )

    unanimous = sum(1 for r in results if (r.yes_votes == 3 or r.no_votes == 3))
    majority = sum(1 for r in results if (r.yes_votes != r.no_votes))

    avg_latency = (
        sum(total_latency_ms) / len(total_latency_ms) if total_latency_ms else None
    )
    avg_resolution_time_ms = (
        sum(resolution_time_ms) / len(resolution_time_ms) if resolution_time_ms else None
    )

    return RunSummary(
        total_questions=total_questions,
        processed_questions=processed_questions,
        skipped_questions=skipped_questions,
        failed_evidence=failed_evidence,
        accuracy=_accuracy(correct, processed_questions),
        gpt4o_accuracy=_accuracy(gpt4o_correct, gpt4o_total),
        claude_accuracy=_accuracy(claude_correct, claude_total),
        deepseek_accuracy=_accuracy(deepseek_correct, deepseek_total),
        llama_accuracy=_accuracy(llama_correct, llama_total),
        unanimous_rate=_accuracy(unanimous, processed_questions),
        majority_rate=_accuracy(majority, processed_questions),
        average_latency_ms=avg_latency,
        average_resolution_time_ms=avg_resolution_time_ms,
        runtime_seconds=runtime_seconds,
        estimated_exa_cost_usd=exa_cost_usd,
    )


def _row_for_csv(row: dict) -> dict:
    """Replace newlines in string fields so one logical row = one CSV line."""
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
    if summary.gpt4o_accuracy is not None:
        print(f"{OPENAI_AGENT_NAME} accuracy: {summary.gpt4o_accuracy:.3f}")
    provider_accuracy = summary.claude_accuracy if use_claude else summary.deepseek_accuracy
    provider_label = "Claude" if use_claude else "DeepSeek"
    if provider_accuracy is not None:
        print(f"{provider_label} accuracy: {provider_accuracy:.3f}")
    if summary.llama_accuracy is not None:
        print(f"{'Gemini' if use_gemini else 'Llama'} accuracy: {summary.llama_accuracy:.3f}")

    if summary.unanimous_rate is not None:
        print(f"Unanimous rate: {summary.unanimous_rate:.3f}")
    if summary.majority_rate is not None:
        print(f"Majority rate: {summary.majority_rate:.3f}")
    if summary.average_latency_ms is not None:
        print(f"Avg latency (ms): {summary.average_latency_ms:.1f}")
    if summary.average_resolution_time_ms is not None:
        print(f"Avg resolution time (ms): {summary.average_resolution_time_ms:.1f}")

    print(f"Runtime (s): {summary.runtime_seconds:.1f}")
    print(f"Estimated Exa cost (USD): {summary.estimated_exa_cost_usd:.2f}")
    print("=" * 60 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Architecture-A end-to-end pipeline test")
    parser.add_argument(
        "-n",
        type=int,
        default=10,
        help="Number of questions (default: 10)",
    )
    parser.add_argument(
        "--questions-csv",
        default=DEFAULT_QUESTIONS_CSV,
        help="Questions CSV path (default: data/kalshibench_v2_evaluation_Filtered.csv)",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=1,
        help="1-based start row number for cache keys/selection (default: 1)",
    )
    parser.add_argument(
        "--cache-start-row",
        type=int,
        default=None,
        help=(
            "1-based row number to start cache keys/row_uid at without skipping rows. "
            "If set, all rows are processed but cache keys start at this value."
        ),
    )
    parser.add_argument(
        "--question-ids",
        nargs="+",
        help="Specific question IDs (overrides -n)",
    )
    parser.add_argument(
        "--category",
        help="Filter by category",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Only use cached evidence",
    )
    parser.add_argument(
        "--cache-dir",
        default="cache/evidence",
        help="Evidence cache directory",
    )
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
        default=None,
        help=(
            "Output CSV path. If omitted, writes to "
            "results/architecture_a_results/runs/architecture_a_run_<timestamp>.csv"
        ),
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        help="Optional JSON with summary stats",
    )
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run without executing",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose progress output",
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = str(
            Path("results")
            / "architecture_a_results"
            / "runs"
            / f"architecture_a_run_{_now_stamp()}.csv"
        )

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
            highlights_max = "Exa defaults" if args.highlights_max_characters == 0 else args.highlights_max_characters
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

        runner = MultiAgentRunner(agents=[GPT4oAgent(), ClaudeAgent(), GeminiAgent()])
    elif args.use_claude:
        from src.agents.anthropic_agent import ClaudeAgent
        from src.agents.llama_agent import LlamaAgent

        runner = MultiAgentRunner(agents=[GPT4oAgent(), ClaudeAgent(), LlamaAgent()])
    elif args.use_gemini:
        from src.agents.deepseek_agent import DeepSeekAgent
        from src.agents.google_agent import GeminiAgent

        runner = MultiAgentRunner(agents=[GPT4oAgent(), DeepSeekAgent(), GeminiAgent()])
    else:
        runner = MultiAgentRunner()

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
        tqdm(
        questions,
        desc="Processing questions",
        disable=not args.verbose,
        ),
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
