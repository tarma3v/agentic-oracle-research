"""Reproducible arithmetic, not a new model experiment. Python 3.10+, stdlib only.

The observed cell (12 errors / 563 accepted out of 1189) is from Table 12 of
Kota (2026), https://arxiv.org/html/2605.30802v1 . Cost values are hypothetical.
Binomial bounds require iid Bernoulli errors and a gate fixed independently
of evaluation labels. The paper cell does not establish those assumptions.
The sample-size table conditions on the realized error count; it is not a
prospective power calculation for a difference between two selective policies.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


def binomial_cdf(k: int, n: int, p: float) -> float:
    if not 0 <= k <= n or not 0 <= p <= 1:
        raise ValueError('Require 0 <= k <= n and 0 <= p <= 1')
    if p <= 0:
        return 1.0
    if p >= 1:
        return float(k >= n)
    # Recurrence avoids subtracting large lgamma values for small k / large n.
    terms = [n * math.log1p(-p)]
    for j in range(1, k + 1):
        terms.append(terms[-1] + math.log(n - j + 1) - math.log(j)
                     + math.log(p) - math.log1p(-p))
    top = max(terms)
    return min(1.0, math.exp(top) * math.fsum(math.exp(t - top) for t in terms))


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


def required_accepted_cases(errors: int, risk_limit: float, delta: float = .05) -> int:
    """Smallest n whose one-sided CP upper bound with k errors is <= risk_limit.

    Inversion of the exact binomial CDF; no normal approximation. The returned
    integer depends on having exactly this many observed errors, an independently
    fixed gate, and independent accepted cases. It does not correct for searching
    over thresholds, repeated looks, dependent contracts, or multiple endpoints.
    """
    if errors < 0 or not isinstance(errors, int) or not 0 < risk_limit < 1 or not 0 < delta < 1:
        raise ValueError('Require nonnegative integer errors and risk_limit, delta in (0,1)')
    lo, hi = errors, max(errors + 1, 1)
    while binomial_cdf(errors, hi, risk_limit) > delta:
        hi *= 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if binomial_cdf(errors, mid, risk_limit) <= delta:
            hi = mid
        else:
            lo = mid
    return hi


def round_for_serialization(value):
    """Suppress irrelevant platform floating-point differences in saved outputs."""
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, dict):
        return {key: round_for_serialization(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_for_serialization(item) for item in value]
    return value


def main() -> None:
    # Verify mathematical edge cases, independently of source data.
    assert abs(upper_binomial(0, 100) - (1 - .05 ** .01)) < 1e-12
    assert upper_binomial(100, 100) == 1.0
    assert upper_binomial(1, 100) > upper_binomial(0, 100)
    required_cases = {
        str(alpha): {str(k): required_accepted_cases(k, alpha) for k in [0, 2, 5]}
        for alpha in [.05, .02, .01, .0066, .005, .0026, .001]
    }
    # Check the defining inequality and minimality rather than rounded bounds.
    for alpha, counts in required_cases.items():
        for k, n in counts.items():
            assert binomial_cdf(int(k), n, float(alpha)) <= .05
            assert binomial_cdf(int(k), n - 1, float(alpha)) > .05
        assert counts['0'] == math.ceil(math.log(.05) / math.log1p(-float(alpha)))
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
    proposer_scenarios = []
    bond = 750.0
    for reward, proposal_cost in [(5., 0.), (5., .05), (5., 3.), (2., .05)]:
        risk_break_even = (reward - proposal_cost) / (bond + reward)
        proposer_scenarios.append({
            'bond_usd_equivalent': bond,
            'reward_usd_equivalent_assumed': reward,
            'all_in_cost_per_proposal_usd_assumed': proposal_cost,
            'break_even_error_rate_under_full_slashing_assumption': risk_break_even,
            'required_accepted_cases_zero_errors_at_95pct': required_accepted_cases(0, risk_break_even),
            'profit_per_proposal_at_reported_selective_risk_usd_illustrative': (
                (1 - risk) * reward - risk * bond - proposal_cost),
        })
    result = {
        'kind': 'arithmetic_on_reported_counts_and_hypothetical_costs',
        'source': 'https://arxiv.org/html/2605.30802v1#S4.T12',
        'n_total': total, 'n_accepted': accepted, 'n_errors_accepted': errors,
        'coverage': coverage, 'selective_error_rate': risk,
        'one_sided_95pct_binomial_error_upper_bound_assuming_iid_fixed_gate': upper_binomial(errors, accepted),
        'warning': 'Post-hoc paper gate and correlated event rows: bound is illustrative, not a certified deployment guarantee.',
        'zero_error_required_independent_accepted_cases': {
            str(alpha): required_accepted_cases(0, alpha)
            for alpha in [.05, .02, .01, .005, .001]
        },
        'required_independent_accepted_cases_by_risk_limit_and_observed_errors': required_cases,
        'sample_size_interpretation': {
            'confidence_level_one_sided': .95,
            'method': 'Exact binomial inversion / one-sided Clopper-Pearson upper bound',
            'inequality': 'min n: sum_{j=0}^{k} choose(n,j) alpha^j (1-alpha)^(n-j) <= 0.05',
            'unit': 'independent accepted evaluation case, not every incoming or correlated market row',
            'limitations': 'Conditional on realized error count; fixed gate; no threshold search or multiple-look correction; not power for coverage uplift.',
            'incoming_sample_conversion': 'n_accepted / coverage is an expectation for planning, not a guarantee of enough accepted cases.',
        },
        'cost_assumptions': {'ai_cost_per_incoming_question_usd': ai_cost,
                             'human_cost_per_deferred_question_usd': human_cost,
                             'human_error_rate': 0,
                             'latency_and_disputes_included': False,
                             'status': 'illustrative assumptions, not measured market costs'},
        'break_even_loss_usd': (coverage * human_cost - ai_cost) / (coverage * risk),
        'cost_scenarios': scenarios,
        'uma_proposer_economics': {
            'role': 'Private expected profit of a proposer; separate from operator/human replacement cost and user settlement harm.',
            'primary_sources': [
                'https://docs.polymarket.com/concepts/resolution',
                'https://github.com/Polymarket/uma-ctf-adapter',
            ],
            'documentation_checked_as_of': '2026-09-30',
            'documented_context': 'Polymarket documents a typical $750 pUSD bond, return plus a reward when correct and undisputed, and loss for an incorrect or premature proposal. The adapter stores reward per market.',
            'reward_status': '$5 and $2 are sensitivity assumptions; no universal/current fixed proposal reward was verified. Read request-specific bond, currency and reward before any live evaluation.',
            'profit_formula': 'E[profit] = (1-epsilon)*R - epsilon*B - c',
            'break_even_formula': 'epsilon_star = (R-c)/(B+R), requiring R>c for positive tolerance',
            'strong_assumptions': [
                'Every wrong or premature proposal is rejected and loses the entire bond B.',
                'Every correct proposal returns the bond and earns R; dispute windfalls and failures are ignored.',
                'c is the all-in per-proposal cost, including any gas, inference, search, capital lockup and operations counted in the scenario.',
                'The reward and bond are expressed in comparable USD-equivalent amounts; token/depeg risk is ignored.',
                'Equating the reported dataset error rate with rejected-proposal probability is illustrative only and is not empirically established.',
            ],
            'excluded_claim': 'Proposer profit does not certify settlement accuracy, social welfare, or attacker resistance; bond is not the full loss per incorrect payout.',
            'scenarios': proposer_scenarios,
        },
    }
    result = round_for_serialization(result)
    out = Path(__file__).resolve().parents[1] / 'results' / 'risk_and_cost.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
