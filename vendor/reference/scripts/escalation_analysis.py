#!/usr/bin/env python3
"""Escalation-oriented analysis for Architecture B confidence and consensus signals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

YES_NO = {"YES", "NO"}
MODELS = ["gpt4o", "deepseek", "llama"]


def normalize_decision(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def load_architecture_b(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ground_truth"] = normalize_decision(df["ground_truth"])
    df["final_decision"] = normalize_decision(df["final_decision"])
    return df


def load_architecture_a(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ground_truth"] = normalize_decision(df["ground_truth"])
    df["final_decision"] = normalize_decision(df["final_decision"])
    for model in MODELS:
        df[f"{model}_decision"] = normalize_decision(df[f"{model}_decision"])
    return df


def build_signal_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    signal_df = df.copy()
    conf_cols = [f"{model}_round2_confidence" for model in MODELS]
    for col in conf_cols:
        signal_df[col] = pd.to_numeric(signal_df[col], errors="coerce")

    signal_df["avg_round2_confidence"] = signal_df[conf_cols].mean(axis=1)
    signal_df["is_correct"] = signal_df["final_decision"] == signal_df["ground_truth"]
    signal_df["is_unanimous"] = (signal_df["round2_yes_votes"] == 3) | (signal_df["round2_no_votes"] == 3)
    signal_df["vote_pattern"] = np.where(signal_df["is_unanimous"], "unanimous_3_to_0", "split_2_to_1")

    confidence_median = float(signal_df["avg_round2_confidence"].median())
    signal_df["confidence_bin"] = np.where(signal_df["avg_round2_confidence"] >= confidence_median, "high", "low")
    return signal_df, confidence_median


def build_signal_frame_architecture_a(df: pd.DataFrame) -> pd.DataFrame:
    signal_df = df.copy()
    conf_cols = [f"{model}_confidence" for model in MODELS]
    decision_cols = [f"{model}_decision" for model in MODELS]
    for col in conf_cols:
        signal_df[col] = pd.to_numeric(signal_df[col], errors="coerce")

    signal_df["avg_confidence"] = signal_df[conf_cols].mean(axis=1)
    signal_df["is_correct"] = signal_df["final_decision"] == signal_df["ground_truth"]
    signal_df["is_unanimous"] = signal_df[decision_cols].nunique(axis=1) == 1
    signal_df["vote_pattern"] = np.where(signal_df["is_unanimous"], "unanimous_3_to_0", "split_2_to_1")
    signal_df["agreement_status"] = np.where(signal_df["is_unanimous"], "Unanimous", "Split")
    confidence_median = float(signal_df["avg_confidence"].median())
    signal_df["confidence_bin"] = np.where(signal_df["avg_confidence"] >= confidence_median, "High", "Low")
    signal_df["composite_score"] = signal_df["is_unanimous"].astype(int) + signal_df["avg_confidence"]
    signal_df.attrs["confidence_median"] = confidence_median
    return signal_df


def summarize_signal(signal_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unanimity = (
        signal_df.groupby("vote_pattern")
        .agg(
            n=("row_uid", "size"),
            accuracy=("is_correct", "mean"),
            avg_confidence=("avg_round2_confidence", "mean"),
        )
        .reset_index()
    )
    confidence = (
        signal_df.groupby("confidence_bin")
        .agg(
            n=("row_uid", "size"),
            accuracy=("is_correct", "mean"),
            avg_confidence=("avg_round2_confidence", "mean"),
        )
        .reset_index()
    )
    joint = (
        signal_df.groupby(["vote_pattern", "confidence_bin"])
        .agg(
            n=("row_uid", "size"),
            accuracy=("is_correct", "mean"),
            avg_confidence=("avg_round2_confidence", "mean"),
        )
        .reset_index()
        .sort_values(["vote_pattern", "confidence_bin"])
    )
    return unanimity, confidence, joint


def build_coverage_curve(signal_df: pd.DataFrame) -> pd.DataFrame:
    curve = signal_df.sort_values(["is_unanimous", "avg_round2_confidence"], ascending=[False, False]).reset_index(drop=True)
    curve["rank"] = np.arange(1, len(curve) + 1)
    curve["auto_resolved_fraction"] = curve["rank"] / len(curve)
    curve["auto_resolved_accuracy"] = curve["is_correct"].expanding().mean()
    return curve[
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
        ]
    ]


def build_coverage_curve_architecture_a(signal_df: pd.DataFrame) -> pd.DataFrame:
    curve = signal_df.sort_values(["composite_score"], ascending=[False]).reset_index(drop=True)
    curve["rank"] = np.arange(1, len(curve) + 1)
    curve["auto_resolved_fraction"] = curve["rank"] / len(curve)
    curve["auto_resolved_accuracy"] = curve["is_correct"].expanding().mean()
    return curve[
        [
            "row_uid",
            "question_id",
            "category",
            "vote_pattern",
            "agreement_status",
            "confidence_bin",
            "avg_confidence",
            "composite_score",
            "is_correct",
            "rank",
            "auto_resolved_fraction",
            "auto_resolved_accuracy",
        ]
    ]


def summarize_signal_architecture_a(signal_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    agreement = (
        signal_df.groupby("agreement_status")
        .agg(
            Count=("row_uid", "size"),
            Accuracy=("is_correct", "mean"),
        )
        .reset_index()
    )
    confidence = (
        signal_df.groupby("confidence_bin")
        .agg(
            Count=("row_uid", "size"),
            Accuracy=("is_correct", "mean"),
        )
        .reset_index()
    )
    joint = (
        signal_df.groupby(["agreement_status", "confidence_bin"])
        .agg(
            Count=("row_uid", "size"),
            Accuracy=("is_correct", "mean"),
        )
        .reset_index()
        .sort_values(["agreement_status", "confidence_bin"])
    )
    return agreement, confidence, joint


def build_comparison_curve(a_curve: pd.DataFrame, b_curve: pd.DataFrame) -> pd.DataFrame:
    if len(a_curve) != len(b_curve):
        raise ValueError("Architecture A and B curves must be built on datasets of the same size for direct comparison.")

    comparison = pd.DataFrame(
        {
            "rank": a_curve["rank"],
            "fraction_auto_resolved": a_curve["auto_resolved_fraction"],
            "architecture_a_accuracy": a_curve["auto_resolved_accuracy"],
            "architecture_b_accuracy": b_curve["auto_resolved_accuracy"],
        }
    )
    comparison["accuracy_gap_a_minus_b"] = comparison["architecture_a_accuracy"] - comparison["architecture_b_accuracy"]
    return comparison


def write_summary(
    output_dir: Path,
    source_path: Path,
    confidence_median: float,
    unanimity: pd.DataFrame,
    confidence: pd.DataFrame,
    joint: pd.DataFrame,
    curve: pd.DataFrame,
) -> None:
    checkpoints = []
    for frac in [0.10, 0.20, 0.25, 0.50, 0.75, 1.00]:
        rank = max(1, int(frac * len(curve)))
        row = curve.iloc[rank - 1]
        checkpoints.append(
            {
                "fraction_auto_resolved": float(row["auto_resolved_fraction"]),
                "rank": int(row["rank"]),
                "accuracy": float(row["auto_resolved_accuracy"]),
            }
        )

    summary = {
        "source_file": str(source_path),
        "n_questions": int(len(curve)),
        "confidence_split_method": "median",
        "confidence_median": confidence_median,
        "signal_strength_sort": "Final-round unanimity first, then descending average round-2 confidence.",
        "unanimous_vs_split": unanimity.to_dict(orient="records"),
        "confidence_bins": confidence.to_dict(orient="records"),
        "signal_2x2": joint.to_dict(orient="records"),
        "coverage_accuracy_checkpoints": checkpoints,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def write_comparison_summary(
    output_dir: Path,
    arch_a_path: Path,
    arch_b_path: Path,
    comparison: pd.DataFrame,
) -> None:
    checkpoints = []
    for frac in [0.10, 0.20, 0.25, 0.50, 0.75, 1.00]:
        rank = max(1, int(frac * len(comparison)))
        row = comparison.iloc[rank - 1]
        checkpoints.append(
            {
                "fraction_auto_resolved": float(row["fraction_auto_resolved"]),
                "rank": int(row["rank"]),
                "architecture_a_accuracy": float(row["architecture_a_accuracy"]),
                "architecture_b_accuracy": float(row["architecture_b_accuracy"]),
                "accuracy_gap_a_minus_b": float(row["accuracy_gap_a_minus_b"]),
            }
        )

    summary = {
        "architecture_a_source_file": str(arch_a_path),
        "architecture_b_source_file": str(arch_b_path),
        "n_questions": int(len(comparison)),
        "architecture_a_overall_accuracy": float(comparison.iloc[-1]["architecture_a_accuracy"]),
        "architecture_b_overall_accuracy": float(comparison.iloc[-1]["architecture_b_accuracy"]),
        "a_better_prefix_count": int((comparison["accuracy_gap_a_minus_b"] > 0).sum()),
        "b_better_prefix_count": int((comparison["accuracy_gap_a_minus_b"] < 0).sum()),
        "tied_prefix_count": int((comparison["accuracy_gap_a_minus_b"] == 0).sum()),
        "coverage_accuracy_checkpoints": checkpoints,
    }
    with open(output_dir / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def write_architecture_a_summary(
    output_dir: Path,
    arch_a_path: Path,
    confidence_median: float,
    agreement: pd.DataFrame,
    confidence: pd.DataFrame,
    joint: pd.DataFrame,
    curve: pd.DataFrame,
) -> None:
    checkpoints = []
    for frac in [0.10, 0.20, 0.25, 0.50, 0.75, 1.00]:
        rank = max(1, int(frac * len(curve)))
        row = curve.iloc[rank - 1]
        checkpoints.append(
            {
                "fraction_auto_resolved": float(row["auto_resolved_fraction"]),
                "rank": int(row["rank"]),
                "accuracy": float(row["auto_resolved_accuracy"]),
                "composite_score": float(row["composite_score"]),
            }
        )

    summary = {
        "source_file": str(arch_a_path),
        "n_questions": int(len(curve)),
        "confidence_split_method": "median",
        "confidence_median": confidence_median,
        "coverage_score_definition": "composite_score = unanimity_indicator + average_confidence",
        "agreement_status": agreement.to_dict(orient="records"),
        "confidence_bins": confidence.to_dict(orient="records"),
        "signal_2x2": joint.to_dict(orient="records"),
        "coverage_accuracy_checkpoints": checkpoints,
    }
    with open(output_dir / "architecture_a_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def write_readme(
    output_dir: Path,
    confidence_median: float,
    unanimity: pd.DataFrame,
    confidence: pd.DataFrame,
    joint: pd.DataFrame,
    curve: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    def fmt_pct(value: float) -> str:
        return f"{100 * value:.2f}%"

    uni_rows = {
        row["vote_pattern"]: row for _, row in unanimity.iterrows()
    }
    conf_rows = {
        row["confidence_bin"]: row for _, row in confidence.iterrows()
    }
    joint_rows = {
        (row["vote_pattern"], row["confidence_bin"]): row for _, row in joint.iterrows()
    }

    checkpoint_lines = []
    for frac in [0.10, 0.20, 0.25, 0.50, 0.75, 1.00]:
        rank = max(1, int(frac * len(curve)))
        row = curve.iloc[rank - 1]
        checkpoint_lines.append(
            f"- Top {int(frac * 100)}% auto-resolved: n={rank}, accuracy={fmt_pct(row['auto_resolved_accuracy'])}"
        )

    a_better_prefixes = int((comparison["accuracy_gap_a_minus_b"] > 0).sum())
    b_better_prefixes = int((comparison["accuracy_gap_a_minus_b"] < 0).sum())
    tied_prefixes = int((comparison["accuracy_gap_a_minus_b"] == 0).sum())
    final_row = comparison.iloc[-1]

    text = f"""# Escalation Analysis

This folder contains Architecture B escalation-style signal analysis based on final-round consensus and confidence.

## Setup

- Source: `results/Final_Architecture_B_results.csv`
- Questions analyzed: `{len(curve)}`
- Confidence split: median average round-2 confidence across the three models
- Median threshold: `{confidence_median:.4f}`
- Signal strength ordering for coverage curve: unanimous questions first, then descending average round-2 confidence

## Unanimous vs Split

- `unanimous_3_to_0`: n={int(uni_rows['unanimous_3_to_0']['n'])}, accuracy={fmt_pct(uni_rows['unanimous_3_to_0']['accuracy'])}
- `split_2_to_1`: n={int(uni_rows['split_2_to_1']['n'])}, accuracy={fmt_pct(uni_rows['split_2_to_1']['accuracy'])}

## Confidence Bins

- `high`: n={int(conf_rows['high']['n'])}, accuracy={fmt_pct(conf_rows['high']['accuracy'])}
- `low`: n={int(conf_rows['low']['n'])}, accuracy={fmt_pct(conf_rows['low']['accuracy'])}

## Joint 2x2

- `unanimous_3_to_0 + high`: n={int(joint_rows[('unanimous_3_to_0', 'high')]['n'])}, accuracy={fmt_pct(joint_rows[('unanimous_3_to_0', 'high')]['accuracy'])}
- `unanimous_3_to_0 + low`: n={int(joint_rows[('unanimous_3_to_0', 'low')]['n'])}, accuracy={fmt_pct(joint_rows[('unanimous_3_to_0', 'low')]['accuracy'])}
- `split_2_to_1 + high`: n={int(joint_rows[('split_2_to_1', 'high')]['n'])}, accuracy={fmt_pct(joint_rows[('split_2_to_1', 'high')]['accuracy'])}
- `split_2_to_1 + low`: n={int(joint_rows[('split_2_to_1', 'low')]['n'])}, accuracy={fmt_pct(joint_rows[('split_2_to_1', 'low')]['accuracy'])}

## Coverage-Accuracy Checkpoints

{chr(10).join(checkpoint_lines)}

## Architecture A vs B Comparison

- Comparison source files: `results/Final_Architecture_A_results.csv` vs `results/Final_Architecture_B_results.csv`
- Architecture A overall accuracy: {fmt_pct(final_row['architecture_a_accuracy'])}
- Architecture B overall accuracy: {fmt_pct(final_row['architecture_b_accuracy'])}
- Prefixes where A is higher: {a_better_prefixes}
- Prefixes where B is higher: {b_better_prefixes}
- Tied prefixes: {tied_prefixes}
- Note: A is better overall, but it does not strictly dominate B at every single coverage prefix.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def save_curve_plot(curve: pd.DataFrame, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    plt.plot(curve["auto_resolved_fraction"], curve["auto_resolved_accuracy"], linewidth=2.2)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Fraction Auto-Resolved")
    plt.ylabel("Accuracy on Auto-Resolved Set")
    plt.title("Architecture B Coverage-Accuracy Curve")
    plt.tight_layout()
    plt.savefig(output_dir / "coverage_accuracy_curve.png", dpi=180)
    plt.close()


def save_comparison_plot(comparison: pd.DataFrame, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8.8, 5.6), facecolor="white")

    arch_a_color = "#3A8A5C"
    arch_b_color = "#C26A4A"

    ax.plot(
        comparison["fraction_auto_resolved"],
        comparison["architecture_a_accuracy"],
        linewidth=2.4,
        color=arch_a_color,
        label="Architecture A",
    )
    ax.plot(
        comparison["fraction_auto_resolved"],
        comparison["architecture_b_accuracy"],
        linewidth=2.4,
        color=arch_b_color,
        label="Architecture B",
    )

    annotation_points = [
        {
            "x": 0.474,
            "y": 0.9787,
            "label": "Unanimous + High Conf.\n(47.4% coverage, 97.9% accuracy)",
            "xytext": (32, -42),
        },
        {
            "x": 0.823,
            "y": 0.8834,
            "label": "All Unanimous\n(82.3% coverage, 88.3% accuracy)",
            "xytext": (28, 30),
        },
    ]
    for point in annotation_points:
        ax.scatter(
            point["x"],
            point["y"],
            s=64,
            color=arch_a_color,
            edgecolor="white",
            linewidth=0.9,
            zorder=5,
        )
        ax.annotate(
            point["label"],
            xy=(point["x"], point["y"]),
            xytext=point["xytext"],
            textcoords="offset points",
            fontsize=8.5,
            color="#444444",
            ha="left",
            va="center",
            arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.8},
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0.70, 1.02)
    ax.set_xlabel("Coverage (Fraction Auto-Resolved)")
    ax.set_ylabel("Accuracy on Auto-Resolved Set")
    ax.set_title("Coverage-Accuracy Tradeoff for Escalation Policy")
    ax.yaxis.set_major_locator(mticker.FixedLocator([0.75, 0.80, 0.85, 0.90, 0.95, 1.00]))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis="both", color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_dir / "coverage_accuracy_comparison.png", dpi=180, facecolor="white")
    plt.close(fig)


def save_curve_plot_architecture_a(curve: pd.DataFrame, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    plt.plot(curve["auto_resolved_fraction"], curve["auto_resolved_accuracy"], linewidth=2.2)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Fraction Auto-Resolved")
    plt.ylabel("Accuracy on Auto-Resolved Set")
    plt.title("Architecture A Coverage-Accuracy Curve")
    plt.tight_layout()
    plt.savefig(output_dir / "architecture_a_coverage_accuracy_curve.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch-a", default="results/Final_Architecture_A_results.csv")
    parser.add_argument("--arch-b", default="results/Final_Architecture_B_results.csv")
    parser.add_argument("--output-dir", default="results/escalation_analysis")
    args = parser.parse_args()

    arch_a_path = Path(args.arch_a)
    arch_b_path = Path(args.arch_b)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    a_df = load_architecture_a(arch_a_path)
    df = load_architecture_b(arch_b_path)
    signal_df, confidence_median = build_signal_frame(df)
    unanimity, confidence, joint = summarize_signal(signal_df)
    curve = build_coverage_curve(signal_df)
    a_signal_df = build_signal_frame_architecture_a(a_df)
    a_confidence_median = float(a_signal_df.attrs["confidence_median"])
    a_agreement, a_confidence, a_joint = summarize_signal_architecture_a(a_signal_df)
    a_curve = build_coverage_curve_architecture_a(a_signal_df)
    comparison = build_comparison_curve(a_curve, curve)

    unanimity.to_csv(output_dir / "unanimous_vs_split.csv", index=False)
    confidence.to_csv(output_dir / "confidence_bins.csv", index=False)
    joint.to_csv(output_dir / "signal_2x2.csv", index=False)
    curve.to_csv(output_dir / "coverage_accuracy_curve.csv", index=False)
    curve.to_csv(output_dir / "architecture_b_coverage_accuracy_curve.csv", index=False)
    a_agreement.to_csv(output_dir / "architecture_a_agreement_status.csv", index=False)
    a_confidence.to_csv(output_dir / "architecture_a_confidence_bins.csv", index=False)
    a_joint.to_csv(output_dir / "architecture_a_signal_2x2.csv", index=False)
    a_curve.to_csv(output_dir / "architecture_a_coverage_accuracy_curve.csv", index=False)
    comparison.to_csv(output_dir / "coverage_accuracy_comparison.csv", index=False)
    write_summary(output_dir, arch_b_path, confidence_median, unanimity, confidence, joint, curve)
    write_architecture_a_summary(output_dir, arch_a_path, a_confidence_median, a_agreement, a_confidence, a_joint, a_curve)
    write_comparison_summary(output_dir, arch_a_path, arch_b_path, comparison)
    write_readme(output_dir, confidence_median, unanimity, confidence, joint, curve, comparison)
    save_curve_plot(curve, output_dir)
    save_curve_plot_architecture_a(a_curve, output_dir)
    save_comparison_plot(comparison, output_dir)


if __name__ == "__main__":
    main()
