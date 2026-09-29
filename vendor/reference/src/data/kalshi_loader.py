"""Load KalshiBench dataset from HuggingFace (or a repo-local saved copy)."""

import csv
from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Optional
from datasets import load_dataset, load_from_disk


@dataclass
class KalshiQuestion:
    """A single question from the KalshiBench dataset"""
    
    id: str
    question: str
    description: str  # Resolution criteria
    category: str
    close_time: str   # ISO timestamp - use as resolution_date for Exa
    ground_truth: str # "yes" or "no"
    series_ticker: Optional[str] = None
    source: Optional[str] = None
    
    def __repr__(self) -> str:
        return f"KalshiQuestion(id='{self.id}', category='{self.category}', ground_truth='{self.ground_truth}')"


DATASET_ID = "2084Collective/kalshibench-v2"
DEFAULT_LOCAL_DIRNAME = "cache/kalshibench-v2"
DEFAULT_SUBSET_CSV_RELATIVE = "results/kalshibench_subset_450_seed42.csv"


def _repo_root() -> Path:
    # src/data/kalshi_loader.py -> repo root is 2 parents up from src/
    return Path(__file__).resolve().parents[2]


def _default_local_path() -> Path:
    return _repo_root() / DEFAULT_LOCAL_DIRNAME


def _default_subset_csv_path() -> Path:
    return _repo_root() / DEFAULT_SUBSET_CSV_RELATIVE


def _resolve_local_path(local_path: Optional[str | os.PathLike[str]]) -> Optional[Path]:
    env_path = os.getenv("KALSHI_BENCH_DATA_DIR")
    if local_path is None and env_path:
        local_path = env_path
    if local_path is None:
        local_path = _default_local_path()
    p = Path(local_path).expanduser()
    return p


def _optional_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = value.strip()
    return v if v else None


def load_kalshi_questions_csv(
    csv_path: Optional[str | os.PathLike[str]] = None,
) -> List[KalshiQuestion]:
    """
    Load Kalshi questions from a CSV file (defaults to the 450-row subset).
    """
    if csv_path is None:
        resolved = _default_subset_csv_path()
    else:
        resolved = Path(csv_path).expanduser()
        if not resolved.is_absolute():
            resolved = _repo_root() / resolved

    if not resolved.exists():
        raise FileNotFoundError(
            f"Questions CSV not found at {resolved}. "
            "Generate it with: python scripts/create_kalshibench_subset_and_stats.py"
        )

    questions: List[KalshiQuestion] = []
    with resolved.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(
                KalshiQuestion(
                    id=row["id"],
                    question=row["question"],
                    description=row["description"],
                    category=row["category"],
                    close_time=row["close_time"],
                    ground_truth=row["ground_truth"],
                    series_ticker=_optional_str(row.get("series_ticker")),
                    source=_optional_str(row.get("source")),
                )
            )
    return questions


def load_kalshi_bench(local_path: Optional[str | os.PathLike[str]] = None) -> List[KalshiQuestion]:
    """
    Load full KalshiBench v2 from HuggingFace.
    
    Returns:
        List of 1,530 prediction market questions with ground truth
    """
    local_dir = _resolve_local_path(local_path)
    if local_dir is not None and local_dir.exists():
        try:
            ds = load_from_disk(str(local_dir))
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                f"Found local KalshiBench dataset at {local_dir}, but failed to load it.\n"
                "Re-download with: python scripts/download_kalshibench_v2.py --force\n"
                "Or point KALSHI_BENCH_DATA_DIR to a valid `Dataset.save_to_disk()` directory."
            ) from e
    else:
        ds = load_dataset(DATASET_ID, split="train")
    
    questions = []
    for row in ds:
        question = KalshiQuestion(
            id=row["id"],
            question=row["question"],
            description=row["description"],
            category=row["category"],
            close_time=row["close_time"],
            ground_truth=row["ground_truth"],
            series_ticker=row.get("series_ticker"),
            source=row.get("source")
        )
        questions.append(question)
    
    return questions


def load_by_category(category: str) -> List[KalshiQuestion]:
    """
    Filter KalshiBench to specific category.
    
    Args:
        category: One of 16 categories (e.g., "Politics", "Financials", "World")
        
    Returns:
        Filtered list of questions
    """
    all_questions = load_kalshi_bench()
    return [q for q in all_questions if q.category == category]


def get_categories() -> List[str]:
    """
    Get list of all unique categories in KalshiBench.
    
    Returns:
        List of category names
    """
    questions = load_kalshi_bench()
    return sorted(set(q.category for q in questions))


def get_sample(n: int = 10) -> List[KalshiQuestion]:
    """
    Get a small sample for testing.
    
    Args:
        n: Number of questions to sample
        
    Returns:
        Random sample of n questions
    """
    import random
    questions = load_kalshi_bench()
    return random.sample(questions, min(n, len(questions)))
