# Escalation Analysis

This folder contains Architecture B escalation-style signal analysis based on final-round consensus and confidence.

## Setup

- Source: `results/Final_Architecture_B_results.csv`
- Questions analyzed: `1189`
- Confidence split: median average round-2 confidence across the three models
- Median threshold: `0.8067`
- Signal strength ordering for coverage curve: unanimous questions first, then descending average round-2 confidence

## Unanimous vs Split

- `unanimous_3_to_0`: n=1052, accuracy=78.99%
- `split_2_to_1`: n=137, accuracy=54.01%

## Confidence Bins

- `high`: n=600, accuracy=85.83%
- `low`: n=589, accuracy=66.21%

## Joint 2x2

- `unanimous_3_to_0 + high`: n=522, accuracy=89.85%
- `unanimous_3_to_0 + low`: n=530, accuracy=68.30%
- `split_2_to_1 + high`: n=78, accuracy=58.97%
- `split_2_to_1 + low`: n=59, accuracy=47.46%

## Coverage-Accuracy Checkpoints

- Top 10% auto-resolved: n=118, accuracy=100.00%
- Top 20% auto-resolved: n=237, accuracy=99.16%
- Top 25% auto-resolved: n=297, accuracy=97.64%
- Top 50% auto-resolved: n=594, accuracy=87.04%
- Top 75% auto-resolved: n=891, accuracy=80.58%
- Top 100% auto-resolved: n=1189, accuracy=76.11%

## Architecture A vs B Comparison

- Comparison source files: `results/Final_Architecture_A_results.csv` vs `results/Final_Architecture_B_results.csv`
- Architecture A overall accuracy: 82.93%
- Architecture B overall accuracy: 76.11%
- Prefixes where A is higher: 989
- Prefixes where B is higher: 0
- Tied prefixes: 200
- Note: A is better overall, but it does not strictly dominate B at every single coverage prefix.
