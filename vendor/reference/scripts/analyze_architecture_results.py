#!/usr/bin/env python3
"""Comprehensive analysis for Architecture A/B experiment outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest

YES_NO = {"YES", "NO"}
BASELINE_MODELS = ["gpt4o", "deepseek", "llama"]


def normalize_decision(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def load_data(arch_a_path: Path, arch_b_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    a = pd.read_csv(arch_a_path)
    b = pd.read_csv(arch_b_path)

    for frame in [a, b]:
        frame["ground_truth"] = normalize_decision(frame["ground_truth"])
        frame["final_decision"] = normalize_decision(frame["final_decision"])
        frame["category"] = frame["category"].fillna("Unknown")

    for model in BASELINE_MODELS:
        a[f"{model}_decision"] = normalize_decision(a[f"{model}_decision"])
        b[f"{model}_round2_decision"] = normalize_decision(b[f"{model}_round2_decision"])

    return a, b


def majority_vote(df: pd.DataFrame, decision_cols: List[str]) -> pd.Series:
    yes_votes = (df[decision_cols] == "YES").sum(axis=1)
    no_votes = (df[decision_cols] == "NO").sum(axis=1)
    return np.where(yes_votes >= no_votes, "YES", "NO")


def weighted_vote(df: pd.DataFrame) -> pd.Series:
    yes_score = sum(
        np.where(df[f"{model}_decision"] == "YES", df[f"{model}_confidence"].fillna(0), 0)
        for model in BASELINE_MODELS
    )
    no_score = sum(
        np.where(df[f"{model}_decision"] == "NO", df[f"{model}_confidence"].fillna(0), 0)
        for model in BASELINE_MODELS
    )
    return np.where(yes_score >= no_score, "YES", "NO")


def accuracy(series: pd.Series, truth: pd.Series) -> float:
    valid = series.isin(YES_NO)
    if valid.sum() == 0:
        return float("nan")
    return (series[valid] == truth[valid]).mean()


def category_accuracy(df: pd.DataFrame, pred_col: str, truth_col: str = "ground_truth") -> pd.DataFrame:
    rows = []
    for category, g in df.groupby("category"):
        pred = g[pred_col]
        valid = pred.isin(YES_NO)
        rows.append(
            {
                "category": category,
                "n": len(g),
                "attempted": int(valid.sum()),
                "accuracy": (pred[valid] == g.loc[valid, truth_col]).mean() if valid.sum() else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["accuracy", "n"], ascending=[False, False])


def disagreement_rate_a(a: pd.DataFrame) -> pd.Series:
    cols = [f"{m}_decision" for m in BASELINE_MODELS]
    return a[cols].nunique(axis=1) > 1


def error_correlation(df: pd.DataFrame, pred_cols: Dict[str, str]) -> pd.DataFrame:
    errs = {}
    for label, col in pred_cols.items():
        errs[label] = (df[col].isin(YES_NO)) & (df[col] != df["ground_truth"])
    err_df = pd.DataFrame(errs)
    return err_df.corr()


def pairwise_wrong_together(df: pd.DataFrame, pred_cols: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for a_label, a_col in pred_cols.items():
        for b_label, b_col in pred_cols.items():
            subset = df[df[a_col].isin(YES_NO) & df[b_col].isin(YES_NO)]
            a_wrong = subset[a_col] != subset["ground_truth"]
            b_wrong = subset[b_col] != subset["ground_truth"]
            rows.append(
                {
                    "model_a": a_label,
                    "model_b": b_label,
                    "p_b_wrong_given_a_wrong": float(b_wrong[a_wrong].mean()) if a_wrong.sum() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def mcnemar_exact(y_true: pd.Series, pred_a: pd.Series, pred_b: pd.Series) -> Dict[str, float]:
    mask = pred_a.isin(YES_NO) & pred_b.isin(YES_NO)
    truth = y_true[mask]
    a_ok = pred_a[mask] == truth
    b_ok = pred_b[mask] == truth
    b_only = int((~a_ok & b_ok).sum())
    a_only = int((a_ok & ~b_ok).sum())
    n = a_only + b_only
    p = (
        float(binomtest(k=min(a_only, b_only), n=n, p=0.5, alternative="two-sided").pvalue)
        if n
        else 1.0
    )
    return {
        "n_pairs": int(mask.sum()),
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "discordant": n,
        "p_value": p,
    }


def build_overlap(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    a2 = a.copy()
    b2 = b.copy()
    key_cols = ["question_text", "ground_truth"]
    a2["_match_key"] = a2[key_cols].astype(str).agg("||".join, axis=1)
    b2["_match_key"] = b2[key_cols].astype(str).agg("||".join, axis=1)
    a2["_occ"] = a2.groupby("_match_key").cumcount()
    b2["_occ"] = b2.groupby("_match_key").cumcount()

    keep_a = ["_match_key", "_occ", "category", "ground_truth", "final_decision", "weighted_final", "a_disagree"]
    keep_b = ["_match_key", "_occ", "final_decision", "round1_yes_votes", "round1_no_votes", "round2_yes_votes", "round2_no_votes"]
    merged = a2[keep_a].merge(b2[keep_b], on=["_match_key", "_occ"], how="inner", suffixes=("_a", "_b"))
    return merged


def architecture_b_revision_analysis(b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    revision_rows = []
    for model in BASELINE_MODELS:
        r1_col = f"{model}_round1_decision"
        r2_col = f"{model}_round2_decision"
        round1 = normalize_decision(b[r1_col])
        round2 = normalize_decision(b[r2_col])
        truth = normalize_decision(b["ground_truth"])
        revised = (round1 != round2) & round1.isin(YES_NO) & round2.isin(YES_NO) & truth.isin(YES_NO)
        model_rows = b.loc[revised, ["row_uid", "question_id", "category", "ground_truth"]].copy()
        model_rows["model"] = model
        model_rows["round1_decision"] = round1[revised].values
        model_rows["round2_decision"] = round2[revised].values
        model_rows["round1_correct"] = (round1[revised] == truth[revised]).values
        model_rows["round2_correct"] = (round2[revised] == truth[revised]).values
        model_rows["flip_direction"] = np.select(
            [
                model_rows["round1_correct"] & ~model_rows["round2_correct"],
                ~model_rows["round1_correct"] & model_rows["round2_correct"],
                model_rows["round1_correct"] & model_rows["round2_correct"],
            ],
            ["correct_to_incorrect", "incorrect_to_correct", "correct_to_correct"],
            default="incorrect_to_incorrect",
        )
        revision_rows.append(model_rows)

    revisions = pd.concat(revision_rows, ignore_index=True)

    direction_summary = (
        revisions.groupby(["model", "flip_direction"])
        .size()
        .reset_index(name="count")
        .sort_values(["model", "count"], ascending=[True, False])
    )

    total_agent_pairs = int(len(b) * len(BASELINE_MODELS))
    total_revisions = int(len(revisions))
    correct_to_incorrect = int((revisions["flip_direction"] == "correct_to_incorrect").sum())
    incorrect_to_correct = int((revisions["flip_direction"] == "incorrect_to_correct").sum())
    correct_to_correct = int((revisions["flip_direction"] == "correct_to_correct").sum())
    incorrect_to_incorrect = int((revisions["flip_direction"] == "incorrect_to_incorrect").sum())

    row_flip_counts = sum(normalize_decision(b[f"{m}_round1_decision"]) != normalize_decision(b[f"{m}_round2_decision"]) for m in BASELINE_MODELS)

    summary = {
        "n_rows": int(len(b)),
        "total_agent_pairs": total_agent_pairs,
        "total_revisions": total_revisions,
        "revision_rate_over_agent_pairs": float(total_revisions / total_agent_pairs) if total_agent_pairs else float("nan"),
        "rows_with_any_revision": int((row_flip_counts > 0).sum()),
        "rows_with_any_revision_rate": float((row_flip_counts > 0).mean()) if len(b) else float("nan"),
        "rows_with_1_revision": int((row_flip_counts == 1).sum()),
        "rows_with_2_revisions": int((row_flip_counts == 2).sum()),
        "rows_with_3_revisions": int((row_flip_counts == 3).sum()),
        "correct_to_incorrect": correct_to_incorrect,
        "incorrect_to_correct": incorrect_to_correct,
        "correct_to_correct": correct_to_correct,
        "incorrect_to_incorrect": incorrect_to_incorrect,
        "round1_accuracy_on_revised_cases": float(revisions["round1_correct"].mean()) if total_revisions else float("nan"),
        "round2_accuracy_on_revised_cases": float(revisions["round2_correct"].mean()) if total_revisions else float("nan"),
        "net_change_in_correct_predictions_on_revised_cases": int(incorrect_to_correct - correct_to_incorrect),
    }

    overall_rows = [
        {"model": "all", "flip_direction": "correct_to_incorrect", "count": correct_to_incorrect},
        {"model": "all", "flip_direction": "incorrect_to_correct", "count": incorrect_to_correct},
        {"model": "all", "flip_direction": "correct_to_correct", "count": correct_to_correct},
        {"model": "all", "flip_direction": "incorrect_to_incorrect", "count": incorrect_to_incorrect},
    ]
    direction_summary = pd.concat([direction_summary, pd.DataFrame(overall_rows)], ignore_index=True)

    return revisions, direction_summary, summary


def architecture_b_signal_analysis(
    b: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    signal_df = b.copy()
    conf_cols = [f"{model}_round2_confidence" for model in BASELINE_MODELS]
    for col in conf_cols:
        signal_df[col] = pd.to_numeric(signal_df[col], errors="coerce")

    signal_df["avg_round2_confidence"] = signal_df[conf_cols].mean(axis=1)
    signal_df["is_unanimous"] = (signal_df["round2_yes_votes"] == 3) | (signal_df["round2_no_votes"] == 3)
    signal_df["vote_pattern"] = np.where(signal_df["is_unanimous"], "unanimous_3_to_0", "split_2_to_1")
    signal_df["is_correct"] = signal_df["final_decision"] == signal_df["ground_truth"]

    confidence_median = float(signal_df["avg_round2_confidence"].median())
    signal_df["confidence_bin"] = np.where(
        signal_df["avg_round2_confidence"] >= confidence_median,
        "high",
        "low",
    )

    unanimity_summary = (
        signal_df.groupby("vote_pattern")
        .agg(
            n=("row_uid", "size"),
            accuracy=("is_correct", "mean"),
            avg_confidence=("avg_round2_confidence", "mean"),
        )
        .reset_index()
    )

    confidence_summary = (
        signal_df.groupby("confidence_bin")
        .agg(
            n=("row_uid", "size"),
            accuracy=("is_correct", "mean"),
            avg_confidence=("avg_round2_confidence", "mean"),
        )
        .reset_index()
    )

    joint_summary = (
        signal_df.groupby(["vote_pattern", "confidence_bin"])
        .agg(
            n=("row_uid", "size"),
            accuracy=("is_correct", "mean"),
            avg_confidence=("avg_round2_confidence", "mean"),
        )
        .reset_index()
        .sort_values(["vote_pattern", "confidence_bin"])
    )

    curve = signal_df.sort_values(
        ["is_unanimous", "avg_round2_confidence"],
        ascending=[False, False],
    ).reset_index(drop=True)
    curve["rank"] = np.arange(1, len(curve) + 1)
    curve["auto_resolved_fraction"] = curve["rank"] / len(curve)
    curve["auto_resolved_accuracy"] = curve["is_correct"].expanding().mean()
    curve["signal_bucket"] = np.where(curve["is_unanimous"], "unanimous", "split")

    summary = {
        "confidence_split_method": "median",
        "confidence_median": confidence_median,
        "signal_strength_sort": "Sort by final-round unanimity first, then descending average round-2 confidence.",
        "unanimous_count": int(signal_df["is_unanimous"].sum()),
        "split_count": int((~signal_df["is_unanimous"]).sum()),
    }

    curve = curve[
        [
            "row_uid",
            "question_id",
            "category",
            "vote_pattern",
            "confidence_bin",
            "avg_round2_confidence",
            "is_correct",
            "rank",
            "auto_resolved_fraction",
            "auto_resolved_accuracy",
            "signal_bucket",
        ]
    ]

    return unanimity_summary, confidence_summary, joint_summary, curve, summary


def save_plots(
    output_dir: Path,
    metrics_df: pd.DataFrame,
    cat_df: pd.DataFrame,
    err_corr_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    signal_curve_df: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 5))
    plot_df = metrics_df.sort_values("accuracy", ascending=False)
    sns.barplot(data=plot_df, x="accuracy", y="system", hue="system", palette="viridis", legend=False)
    plt.xlim(0, 1)
    plt.title("Overall Accuracy Across Systems")
    plt.tight_layout()
    plt.savefig(output_dir / "overall_accuracy.png", dpi=180)
    plt.close()

    pivot = cat_df.pivot(index="category", columns="system", values="accuracy")
    plt.figure(figsize=(12, max(5, len(pivot) * 0.35)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1)
    plt.title("Category-wise Accuracy")
    plt.tight_layout()
    plt.savefig(output_dir / "category_accuracy_heatmap.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.heatmap(err_corr_df, annot=True, fmt=".2f", cmap="coolwarm", vmin=0, vmax=1)
    plt.title("Pairwise Error Correlation (Architecture A baselines)")
    plt.tight_layout()
    plt.savefig(output_dir / "error_correlation_heatmap.png", dpi=180)
    plt.close()

    if len(overlap_df):
        counts = {
            "A wrong → B right": int(((overlap_df["final_decision_a"] != overlap_df["ground_truth"]) & (overlap_df["final_decision_b"] == overlap_df["ground_truth"])) .sum()),
            "A right → B wrong": int(((overlap_df["final_decision_a"] == overlap_df["ground_truth"]) & (overlap_df["final_decision_b"] != overlap_df["ground_truth"])) .sum()),
            "Both right": int(((overlap_df["final_decision_a"] == overlap_df["ground_truth"]) & (overlap_df["final_decision_b"] == overlap_df["ground_truth"])) .sum()),
            "Both wrong": int(((overlap_df["final_decision_a"] != overlap_df["ground_truth"]) & (overlap_df["final_decision_b"] != overlap_df["ground_truth"])) .sum()),
        }
        plt.figure(figsize=(8, 4.5))
        sns.barplot(x=list(counts.values()), y=list(counts.keys()), hue=list(counts.keys()), palette="magma", legend=False)
        plt.title("Architecture A vs B Head-to-Head (Overlapping Questions)")
        plt.tight_layout()
        plt.savefig(output_dir / "head_to_head_transitions.png", dpi=180)
        plt.close()

    if len(signal_curve_df):
        plt.figure(figsize=(8, 5))
        plt.plot(signal_curve_df["auto_resolved_fraction"], signal_curve_df["auto_resolved_accuracy"], linewidth=2.2)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("Fraction Auto-Resolved")
        plt.ylabel("Accuracy on Auto-Resolved Set")
        plt.title("Architecture B Coverage-Accuracy Curve")
        plt.tight_layout()
        plt.savefig(output_dir / "architecture_b_coverage_accuracy_curve.png", dpi=180)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch-a", default="results/architecture_a_results.csv")
    parser.add_argument("--arch-b", default="results/Master_450_markets_ArchitectureB.csv")
    parser.add_argument("--output-dir", default="results/comprehensive_analysis")
    parser.add_argument(
        "--timestamped-output",
        action="store_true",
        help="Append a timestamp to the output directory name.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.timestamped_output:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_dir.parent / f"{output_dir.name}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    a, b = load_data(Path(args.arch_a), Path(args.arch_b))

    a["weighted_final"] = weighted_vote(a)
    a["majority_final"] = majority_vote(a, [f"{m}_decision" for m in BASELINE_MODELS])
    a["a_disagree"] = disagreement_rate_a(a)
    b["consensus_round1"] = (b["round1_yes_votes"] == 3) | (b["round1_no_votes"] == 3)
    b["consensus_round2"] = (b["round2_yes_votes"] == 3) | (b["round2_no_votes"] == 3)
    b["convergence_stage"] = np.select(
        [b["consensus_round1"], (~b["consensus_round1"] & b["consensus_round2"])],
        ["round1", "round2"],
        default="no_consensus",
    )

    metrics = []
    for model in BASELINE_MODELS:
        metrics.append({"system": f"A baseline {model}", "accuracy": accuracy(a[f"{model}_decision"], a["ground_truth"]), "n": len(a)})
    metrics.extend(
        [
            {"system": "Architecture A majority", "accuracy": accuracy(a["majority_final"], a["ground_truth"]), "n": len(a)},
            {"system": "Architecture A weighted", "accuracy": accuracy(a["weighted_final"], a["ground_truth"]), "n": len(a)},
        ]
    )
    for model in BASELINE_MODELS:
        metrics.append({"system": f"B final-round {model}", "accuracy": accuracy(b[f"{model}_round2_decision"], b["ground_truth"]), "n": len(b)})
    metrics.append({"system": "Architecture B deliberative final", "accuracy": accuracy(b["final_decision"], b["ground_truth"]), "n": len(b)})
    metrics_df = pd.DataFrame(metrics)

    cat_tables = []
    for label, (df, col) in {
        "A majority": (a, "majority_final"),
        "A weighted": (a, "weighted_final"),
        "A baseline gpt4o": (a, "gpt4o_decision"),
        "A baseline deepseek": (a, "deepseek_decision"),
        "A baseline llama": (a, "llama_decision"),
        "B deliberative final": (b, "final_decision"),
        "B final-round gpt4o": (b, "gpt4o_round2_decision"),
        "B final-round deepseek": (b, "deepseek_round2_decision"),
        "B final-round llama": (b, "llama_round2_decision"),
    }.items():
        table = category_accuracy(df, col)
        table["system"] = label
        cat_tables.append(table)
    cat_df = pd.concat(cat_tables, ignore_index=True)

    disagree_by_cat = a.groupby("category")["a_disagree"].mean().reset_index(name="disagreement_rate")

    convergence = (
        b.groupby("convergence_stage")
        .agg(n=("convergence_stage", "size"), accuracy=("is_correct", "mean"))
        .reset_index()
        .sort_values("n", ascending=False)
    )
    revision_rows, revision_direction_summary, revision_summary = architecture_b_revision_analysis(b)
    unanimity_summary, confidence_summary, joint_signal_summary, signal_curve, signal_summary = architecture_b_signal_analysis(b)

    err_corr = error_correlation(
        a,
        {
            "gpt4o": "gpt4o_decision",
            "deepseek": "deepseek_decision",
            "llama": "llama_decision",
        },
    )
    wrong_together = pairwise_wrong_together(
        a,
        {
            "gpt4o": "gpt4o_decision",
            "deepseek": "deepseek_decision",
            "llama": "llama_decision",
        },
    )

    overlap = build_overlap(a, b)

    mcnemar_rows = []
    comparisons = [
        ("A majority", a["majority_final"], "A weighted", a["weighted_final"], a["ground_truth"]),
    ]
    if len(overlap):
        comparisons.extend(
            [
                (
                    "A majority (overlap)",
                    overlap["final_decision_a"],
                    "B final",
                    overlap["final_decision_b"],
                    overlap["ground_truth"],
                ),
                (
                    "A weighted (overlap)",
                    overlap["weighted_final"],
                    "B final",
                    overlap["final_decision_b"],
                    overlap["ground_truth"],
                ),
            ]
        )
    for left_name, left_pred, right_name, right_pred, y in comparisons:
        res = mcnemar_exact(y, left_pred, right_pred)
        res.update({"model_a": left_name, "model_b": right_name})
        mcnemar_rows.append(res)
    mcnemar_df = pd.DataFrame(mcnemar_rows)

    if len(overlap):
        overlap["a_correct"] = overlap["final_decision_a"] == overlap["ground_truth"]
        overlap["b_correct"] = overlap["final_decision_b"] == overlap["ground_truth"]
        overlap_summary = {
            "n_overlap": int(len(overlap)),
            "a_accuracy_overlap": float(overlap["a_correct"].mean()),
            "b_accuracy_overlap": float(overlap["b_correct"].mean()),
            "a_wrong_b_right": int((~overlap["a_correct"] & overlap["b_correct"]).sum()),
            "a_right_b_wrong": int((overlap["a_correct"] & ~overlap["b_correct"]).sum()),
            "both_right": int((overlap["a_correct"] & overlap["b_correct"]).sum()),
            "both_wrong": int((~overlap["a_correct"] & ~overlap["b_correct"]).sum()),
        }
    else:
        overlap_summary = {"n_overlap": 0}

    metrics_df.to_csv(output_dir / "overall_metrics.csv", index=False)
    cat_df.to_csv(output_dir / "category_metrics.csv", index=False)
    disagree_by_cat.to_csv(output_dir / "disagreement_by_category.csv", index=False)
    convergence.to_csv(output_dir / "convergence_summary.csv", index=False)
    err_corr.to_csv(output_dir / "error_correlation_matrix.csv")
    wrong_together.to_csv(output_dir / "pairwise_wrong_together.csv", index=False)
    mcnemar_df.to_csv(output_dir / "mcnemar_tests.csv", index=False)
    overlap.to_csv(output_dir / "head_to_head_overlap_rows.csv", index=False)
    revision_rows.to_csv(output_dir / "architecture_b_revision_rows.csv", index=False)
    revision_direction_summary.to_csv(output_dir / "architecture_b_revision_directions.csv", index=False)
    unanimity_summary.to_csv(output_dir / "architecture_b_unanimous_vs_split.csv", index=False)
    confidence_summary.to_csv(output_dir / "architecture_b_confidence_bins.csv", index=False)
    joint_signal_summary.to_csv(output_dir / "architecture_b_signal_2x2.csv", index=False)
    signal_curve.to_csv(output_dir / "architecture_b_coverage_accuracy_curve.csv", index=False)

    with open(output_dir / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "overall_metrics": metrics,
                "convergence": convergence.to_dict(orient="records"),
                "architecture_b_revisions": revision_summary,
                "architecture_b_signal_analysis": {
                    "summary": signal_summary,
                    "unanimous_vs_split": unanimity_summary.to_dict(orient="records"),
                    "confidence_bins": confidence_summary.to_dict(orient="records"),
                    "signal_2x2": joint_signal_summary.to_dict(orient="records"),
                },
                "mcnemar": mcnemar_df.to_dict(orient="records"),
                "head_to_head": overlap_summary,
                "notes": [
                    "Architecture B file contains 2 rounds of deliberation; no explicit round-3 columns detected.",
                    "Head-to-head overlap matched using (question_text, ground_truth, occurrence index).",
                    "Architecture B revision analysis uses actual round-1 vs round-2 decision changes rather than the *_round2_revised flags.",
                    "Architecture B confidence bins use a median split on average round-2 confidence across the three models.",
                ],
            },
            f,
            indent=2,
        )

    save_plots(output_dir, metrics_df, cat_df, err_corr, overlap, signal_curve)


if __name__ == "__main__":
    main()
