from src.audit_benchmark import Case,greedy,exhaustive,audit,make_case

def test_joint_sufficiency():
    c=Case([.3,.3],[0,1],{(0,1):3.2})
    assert c.posterior([0])<.9 and c.posterior([1])<.9
    assert c.posterior([0,1])>=.9
    assert len(exhaustive(c,.9))==2

def test_redundancy_does_not_double_count():
    c=Case([2.,2.],[0,0],{})
    assert c.posterior([0,1])==c.posterior([0])

def test_conflict_reduces_posterior():
    c=Case([3.,-2.],[0,1],{})
    assert c.posterior([0,1])<c.posterior([0])

def test_greedy_feasibility_and_optimality_bound():
    for kind in ('ordinary','redundant','conflicting','jointly_sufficient','synergy_trap','mixed'):
        for j in range(10):
            c=make_case(19+j,kind,3+j%10,j)
            g=greedy(c,.9);o=exhaustive(c,.9)
            assert not g or c.posterior(g)>=.9
            assert not o or (c.posterior(o)>=.9 and (not g or len(o)<=len(g)))
            a=audit(c,g,.9)
            assert 0<=a['cfs']<=1
