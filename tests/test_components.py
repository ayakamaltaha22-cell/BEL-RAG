from belrag.core import *
from belrag.metrics import *
def test_policy():
 e=Passage('1','test','a');assert Policy().allow(e)
 assert not Policy().allow(Passage('2','test','b',license='restricted'))
def test_prior():assert abs(prior_corrected_lr(.5,.5)-1)<1e-10
def test_audit():
 items=[Passage('1','a','a',2.5),Passage('2','b','b',1.0)]
 p=Posterior();assert exhaustive(items,p,.9)[0].id=='1'
 assert audit(items,p,.9)['feasible']
def test_metrics():assert exact_match('Paris','paris')==1
