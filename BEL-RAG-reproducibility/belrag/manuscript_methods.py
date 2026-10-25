"""Manuscript-aligned mathematical reference utilities (Sections 6.2, 6.3, 7.5).

Not the historical trained conditional evidence model. The caller must supply
validated conditional likelihoods and outcome probabilities; no probabilities
are invented from retrieval similarity or from a fitted demonstration model.
"""
from __future__ import annotations
from math import log, log2, exp, isfinite
from typing import Mapping, Callable, Sequence

OUTCOMES = ('support', 'contradiction', 'neutral')

def entropy_bits(p: float) -> float:
    if not isfinite(p) or not 0 <= p <= 1:
        raise ValueError('posterior must be a probability')
    return 0.0 if p in (0., 1.) else -p*log2(p)-(1-p)*log2(1-p)

def conditional_posterior(prior: float, conditional_lrs: Sequence[float]) -> float:
    """Posterior odds from *conditional* likelihood ratios, in sequence.

    The caller, not this function, must calculate each LR conditioned on the
    preceding evidence history. Marginal LRs are not automatically valid.
    """
    if not 0 < prior < 1:
        raise ValueError('prior must be strictly between zero and one')
    odds = log(prior/(1-prior))
    for lr in conditional_lrs:
        if not isfinite(lr) or lr <= 0:
            raise ValueError('conditional likelihood ratio must be finite and positive')
        odds += log(lr)
    return 1/(1+exp(-odds)) if odds >= 0 else exp(odds)/(1+exp(odds))

def expected_information_gain(current_p: float,
                              outcome_probabilities: Mapping[str, float],
                              next_posteriors: Mapping[str, float]) -> float:
    """H(p_t) - sum_y rho_y H(p_(t+1)(y,e)), in bits.

    All three outcomes are mandatory. This quantity is information gain only
    when the caller supplies a coherent predictive model.
    """
    if set(outcome_probabilities) != set(OUTCOMES) or set(next_posteriors) != set(OUTCOMES):
        raise ValueError('support, contradiction and neutral outcomes are all required')
    probs = list(outcome_probabilities.values())
    if any(not isfinite(v) or v < 0 or v > 1 for v in probs) or abs(sum(probs)-1) > 1e-9:
        raise ValueError('outcome probabilities must sum to one')
    return entropy_bits(current_p) - sum(outcome_probabilities[y] * entropy_bits(next_posteriors[y]) for y in OUTCOMES)

def acquisition_utility(current_p: float,
                        outcome_probabilities: Mapping[str, float],
                        next_posteriors: Mapping[str, float],
                        cost: float, reference_cost: float, lambda_bits: float) -> float:
    if not all(isfinite(v) for v in (cost, reference_cost, lambda_bits)) or cost <= 0 or reference_cost <= 0 or lambda_bits < 0:
        raise ValueError('cost/reference cost must be positive and lambda nonnegative')
    return expected_information_gain(current_p, outcome_probabilities, next_posteriors) - lambda_bits*(cost/reference_cost)

def counterfactual_scores(support_ids: Sequence[str], posterior_for_subset: Callable[[tuple[str,...]],float],
                          tau: float = .9, alpha: float = .5) -> dict:
    """Threshold-relative necessity/sufficiency, recomputing each subset.

    posterior_for_subset must use the same dependence-aware model for all subsets.
    """
    if not 0 < tau < 1 or not 0 <= alpha <= 1:
        raise ValueError('invalid thresholds')
    ids = tuple(support_ids)
    if len(set(ids)) != len(ids):
        raise ValueError('support IDs must be unique')
    if not ids or posterior_for_subset(ids) < tau:
        return {'feasible':False,'necessity':0.,'sufficiency':0.,'cfs':0.}
    necessity = sum(posterior_for_subset(tuple(x for x in ids if x != e)) < tau for e in ids)/len(ids)
    sufficiency = sum(posterior_for_subset((e,)) >= tau for e in ids)/len(ids)
    return {'feasible':True,'necessity':necessity,'sufficiency':sufficiency,
            'cfs':alpha*necessity+(1-alpha)*sufficiency}
