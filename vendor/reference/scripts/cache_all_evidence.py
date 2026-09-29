"""Pre-fetch and cache Exa evidence for all KalshiBench questions"""

import sys
import time
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_kalshi_questions_csv
from src.retrieval import ExaOracleRetriever


DEFAULT_QUESTIONS_CSV = "data/kalshibench_v2_evaluation_Filtered.csv"


def cache_all_evidence(
    questions_csv: str = DEFAULT_QUESTIONS_CSV,
    output_dir: str = "cache/evidence",
    skip_existing: bool = True,
    delay_seconds: float = 0.1
):
    """
    Retrieve and cache evidence for all KalshiBench questions.
    
    """
    print(f"Loading questions from CSV: {questions_csv}")
    questions = load_kalshi_questions_csv(questions_csv)
    print(f"Loaded {len(questions)} questions")
    
    print("\nInitializing Exa retriever...")
    retriever = ExaOracleRetriever()
    
    cache_path = Path(output_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    if skip_existing:
        existing_ids = {f.stem for f in cache_path.glob("*.json")}
        questions_to_process = [q for q in questions if q.id not in existing_ids]
        print(f"Found {len(existing_ids)} cached, processing {len(questions_to_process)} remaining")
    else:
        questions_to_process = questions
        print(f"Processing all {len(questions_to_process)} questions")
    
    successes = 0
    failures = []
    
    for question in tqdm(questions_to_process, desc="Retrieving evidence"):
        try:
            retriever.retrieve_and_cache(
                question_id=question.id,
                question_text=question.question,
                resolution_criteria=question.description,
                resolution_date=question.close_time,
                cache_dir=output_dir
            )
            successes += 1
            
            time.sleep(delay_seconds)
            
        except Exception as e:
            failures.append((question.id, str(e)))
            print(f"\nError processing {question.id}: {e}")
    
    # Summary
    print("\n" + "="*60)
    print(f"Successfully cached: {successes}")
    print(f"Failures: {len(failures)}")
    
    if failures:
        print("\nFailed question IDs:")
        for qid, error in failures[:10]: 
            print(f"  - {qid}: {error[:100]}")
    
    cost_per_question = 0.015 
    total_cost = successes * cost_per_question
    print(f"\nEstimated cost: ${total_cost:.2f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cache Exa evidence for KalshiBench")
    parser.add_argument(
        "--questions-csv",
        default=DEFAULT_QUESTIONS_CSV,
        help="Questions CSV path (default: data/kalshibench_v2_evaluation_Filtered.csv)"
    )
    parser.add_argument(
        "--output-dir",
        default="cache/evidence",
        help="Output directory for cached evidence"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch even if evidence exists"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between API calls (seconds)"
    )
    
    args = parser.parse_args()
    
    cache_all_evidence(
        questions_csv=args.questions_csv,
        output_dir=args.output_dir,
        skip_existing=not args.force,
        delay_seconds=args.delay
    )
