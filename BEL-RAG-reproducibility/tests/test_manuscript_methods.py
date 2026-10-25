import math
import pytest
from belrag.manuscript_methods import (conditional_posterior, expected_information_gain,
    acquisition_utility, counterfactual_scores)

def test_conditional_prior_correction_not_duplicated():
    assert conditional_posterior(.5,[3]) == pytest.approx(.75)
    assert conditional_posterior(.5,[3,1]) == pytest.approx(.75)
    assert conditional_posterior(.5,[3,1/3]) == pytest.approx(.5)

def test_expected_information_gain_and_units():
    rho={'support':.5,'contradiction':.5,'neutral':0.}
    post={'support':.9,'contradiction':.1,'neutral':.5}
    gain=expected_information_gain(.5,rho,post)
    assert gain == pytest.approx(1-(-.9*math.log2(.9)-.1*math.log2(.1)))
    assert acquisition_utility(.5,rho,post,10,10,.05) == pytest.approx(gain-.05)

def test_invalid_predictor_rejected():
    with pytest.raises(ValueError):
        expected_information_gain(.5,{'support':1.},{'support':.9})

def test_counterfactual_removal_and_isolation():
    def posterior(ids):
        return conditional_posterior(.5,[5 for _ in ids])
    result=counterfactual_scores(('a','b'),posterior)
    assert result['feasible']
    assert result['necessity']==1 and result['sufficiency']==0
    assert result['cfs']==.5
    assert not counterfactual_scores((),posterior)['feasible']
