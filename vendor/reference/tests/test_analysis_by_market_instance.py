"""Tests for scripts/analysis_by_market_instance.py."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "analysis_by_market_instance.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("analysis_by_market_instance", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dedupe_keeps_first_occurrence_stably():
    mod = _load_module()
    df = pd.DataFrame(
        [
            {"id": "M1", "final_decision": "yes"},
            {"id": "M2", "final_decision": "no"},
            {"id": "M1", "final_decision": "no"},
            {"id": "M3", "final_decision": "yes"},
        ]
    )
    out = mod.dedupe_keep_first(df, "id")
    assert out["id"].tolist() == ["M1", "M2", "M3"]
    assert out["final_decision"].tolist() == ["yes", "no", "yes"]


def test_multi_instance_filter_only_returns_groups_gt_one():
    mod = _load_module()
    df = pd.DataFrame(
        [
            {"id": "M1"},
            {"id": "M2"},
            {"id": "M1"},
            {"id": "M3"},
            {"id": "M3"},
        ]
    )
    out = mod.filter_multi_instance(df, "id")
    assert set(out["id"].tolist()) == {"M1", "M3"}
    assert len(out) == 4


def test_normalize_decision_handles_case_and_whitespace():
    mod = _load_module()
    s = pd.Series([" yes ", "No", "  YES", "nO  "])
    assert mod.normalize_decision(s).tolist() == ["YES", "NO", "YES", "NO"]


def test_accuracy_ci_handles_zero_attempts_without_crash():
    mod = _load_module()
    df = pd.DataFrame(
        [
            {"id": "M1", "ground_truth": "maybe", "final_decision": "unknown", "category": "X"},
            {"id": "M2", "ground_truth": "", "final_decision": "", "category": "X"},
        ]
    )
    metrics = mod.compute_overall_metrics(
        df,
        split_name="test",
        market_key="id",
        truth_col="ground_truth",
        decision_cols=["final_decision"],
    )
    assert len(metrics) == 1
    row = metrics.iloc[0]
    assert int(row["attempted_n"]) == 0
    assert pd.isna(row["accuracy"])
    assert pd.isna(row["wilson_ci_low"])
    assert pd.isna(row["wilson_ci_high"])


def test_multi_instance_consistency_counts_all_correct_incorrect_mixed():
    mod = _load_module()
    df = pd.DataFrame(
        [
            # M1: both correct
            {"id": "M1", "ground_truth": "yes", "final_decision": "yes"},
            {"id": "M1", "ground_truth": "no", "final_decision": "no"},
            # M2: both incorrect
            {"id": "M2", "ground_truth": "yes", "final_decision": "no"},
            {"id": "M2", "ground_truth": "no", "final_decision": "yes"},
            # M3: mixed
            {"id": "M3", "ground_truth": "yes", "final_decision": "yes"},
            {"id": "M3", "ground_truth": "no", "final_decision": "yes"},
        ]
    )
    out = mod.compute_multi_instance_market_consistency(
        df,
        market_key="id",
        truth_col="ground_truth",
        decision_cols=["final_decision"],
    )
    assert len(out) == 1
    row = out.iloc[0]
    assert int(row["total_multi_markets"]) == 3
    assert int(row["all_instances_correct_markets"]) == 1
    assert int(row["all_instances_incorrect_markets"]) == 1
    assert int(row["mixed_correctness_markets"]) == 1
    assert int(row["no_scored_instances_markets"]) == 0


def test_integration_outputs_expected_files(tmp_path):
    output_dir = tmp_path / "market_instance_analysis"
    input_csv = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "kalshibench_subset_450_seed42_complement_exclude_ids_keep_dups.csv"
    )
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input-csv",
            str(input_csv),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )

    expected = [
        output_dir / "summary_comparison.md",
        output_dir / "analysis_metadata.json",
        output_dir / "unique_by_id" / "overall_metrics.csv",
        output_dir / "unique_by_id" / "category_metrics.csv",
        output_dir / "multi_instance_only" / "overall_metrics.csv",
        output_dir / "multi_instance_only" / "category_metrics.csv",
        output_dir / "multi_instance_only" / "market_consistency_metrics.csv",
    ]
    for path in expected:
        assert path.exists(), f"Expected artifact missing: {path}"
