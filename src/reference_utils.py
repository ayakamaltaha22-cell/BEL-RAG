"""Small executable utilities, NOT a full BEL-RAG implementation."""
import math

def prior_corrected_lr(s, pi, eps=0.01):
    if not 0 < pi < 1:
        raise ValueError('pi must be in (0,1)')
    s = min(1-eps, max(eps, s))
    return (s/(1-s)) / (pi/(1-pi))

def update_posterior(prior, lr):
    if not 0 < prior < 1 or lr < 0:
        raise ValueError('invalid prior or LR')
    odds = (prior/(1-prior))*lr
    return odds/(1+odds)

def min_max(values):
    if not values: return []
    lo, hi = min(values), max(values)
    return [0.0 for _ in values] if lo == hi else [(x-lo)/(hi-lo) for x in values]

if __name__ == '__main__':
    assert math.isclose(prior_corrected_lr(0.5, 0.5), 1)
    assert math.isclose(update_posterior(0.5, 1), 0.5)
    assert min_max([2,2]) == [0.0,0.0]
    print('Reference utilities smoke tests passed')
