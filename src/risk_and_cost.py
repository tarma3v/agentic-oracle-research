"""Reproducible arithmetic, not a new model experiment. Python 3.10+, stdlib only.

The observed cell (12 errors / 563 accepted out of 1189) is from Table 12 of
Kota (2026), https://arxiv.org/html/2605.30802v1 . Cost values are hypothetical.
Binomial bounds require iid Bernoulli errors and a gate fixed independently
of evaluation labels. The paper cell does not establish those assumptions.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


def binomial_cdf(k: int, n: int, p: float) -> float:
    if p <= 0:
        return 1.0
    if p >= 1:
        return float(k >= n)
    terms = [math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
             + j * math.log(p) + (n - j) * math.log1p(-p)
             for j in range(k + 1)]
    top = max(terms)
    return math.exp(top) * sum(math.exp(t - top) for t in terms)


def upper_binomial(k: int, n: int, delta: float = .05) -> float:
    """One-sided Clopper-Pearson upper bound; no threshold-search correction."""
    if not 0 <= k <= n or n < 1 or not 0 < delta < 1:
        raise ValueError('Require 0 <= k <= n, n >= 1, 0 < delta < 1')
    if k == n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(90):
        mid = (lo + hi) / 2
        if binomial_cdf(k, n, mid) > delta:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main() -> None:
    # Verify mathematical edge cases, independently of source data.
    assert abs(upper_binomial(0, 100) - (1 - .05 ** .01)) < 1e-12
    assert upper_binomial(100, 100) == 1.0
    assert upper_binomial(1, 100) > upper_binomial(0, 100)
    total, accepted, errors = 1189, 563, 12
    coverage = accepted / total
    risk = errors / accepted
    # C_AI applies to every incoming question, C_H to each deferred question.
    ai_cost, human_cost = .05, 5.0
    scenarios = []
    for loss in [10, 100, 1000, 10000]:
        hybrid = ai_cost + (1 - coverage) * human_cost + coverage * risk * loss
        scenarios.append({'loss_per_wrong_automatic_resolution_usd': loss,
                          'hybrid_cost_per_incoming_question_usd': hybrid,
                          'human_only_cost_per_incoming_question_usd': human_cost,
                          'savings_per_incoming_question_usd': human_cost - hybrid})
    result = {
        'kind': 'arithmetic_on_reported_counts_and_hypothetical_costs',
        'source': 'https://arxiv.org/html/2605.30802v1#S4.T12',
        'n_total': total, 'n_accepted': accepted, 'n_errors_accepted': errors,
        'coverage': coverage, 'selective_error_rate': risk,
        'one_sided_95pct_binomial_error_upper_bound_assuming_iid_fixed_gate': upper_binomial(errors, accepted),
        'warning': 'Post-hoc paper gate and correlated event rows: bound is illustrative, not a certified deployment guarantee.',
        'zero_error_required_independent_accepted_cases': {
            str(alpha): math.ceil(math.log(.05) / math.log1p(-alpha))
            for alpha in [.05, .01, .005, .001]
        },
        'cost_assumptions': {'ai_cost_per_incoming_question_usd': ai_cost,
                             'human_cost_per_deferred_question_usd': human_cost,
                             'human_error_rate': 0,
                             'latency_and_disputes_included': False,
                             'status': 'illustrative assumptions, not measured market costs'},
        'break_even_loss_usd': (coverage * human_cost - ai_cost) / (coverage * risk),
        'cost_scenarios': scenarios,
    }
    out = Path(__file__).resolve().parents[1] / 'results' / 'risk_and_cost.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
