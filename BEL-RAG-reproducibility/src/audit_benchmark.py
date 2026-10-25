"""Reproducible controlled stress test of BEL-RAG Section 7 support-set search.
All instances are GENERATED; no manuscript benchmark observations are represented.
Posterior is a declared joint log-odds model with cluster deduplication and
pairwise interaction terms, not the unreleased benchmark conditional model.
"""
import argparse,csv,json,math,random,time,statistics
from pathlib import Path
from itertools import combinations

def sigmoid(x):
    if x>=0: return 1/(1+math.exp(-x))
    z=math.exp(x);return z/(1+z)

class Case:
    def __init__(self,weights,clusters,pairs,prior=.5,kind='ordinary',case_id=''):
        self.weights=weights;self.clusters=clusters;self.pairs=pairs
        self.prior=prior;self.kind=kind;self.case_id=case_id;self.calls=0
    def posterior(self,ids):
        self.calls+=1
        # deterministic cluster aggregation: strongest absolute evidence per source
        bycluster={}
        for i in ids:
            cl=self.clusters[i]
            if cl not in bycluster or abs(self.weights[i])>abs(self.weights[bycluster[cl]]):
                bycluster[cl]=i
        selected=set(bycluster.values())
        logodds=math.log(self.prior/(1-self.prior))+sum(self.weights[i] for i in selected)
        for (a,b),interaction in self.pairs.items():
            if a in selected and b in selected:logodds+=interaction
        return sigmoid(logodds)

def greedy(case,tau):
    remaining=set(range(len(case.weights)));chosen=[]
    while remaining:
        p=case.posterior(chosen)
        if p>=tau:return chosen
        ranked=sorted(((case.posterior(chosen+[i])-p,i) for i in remaining),key=lambda x:(-x[0],x[1]))
        gain,i=ranked[0]
        if gain<=1e-12:break # manuscript's positive marginal greedy rule
        chosen.append(i);remaining.remove(i)
    return chosen if case.posterior(chosen)>=tau else []

def exhaustive(case,tau):
    n=len(case.weights)
    for k in range(1,n+1):
        for sub in combinations(range(n),k):
            if case.posterior(sub)>=tau:return list(sub)
    return []

def audit(case,subset,tau,alpha=.5):
    if not subset:return dict(necessity=0.,sufficiency=0.,cfs=0.)
    feasible=case.posterior(subset)>=tau
    necessity=sum(feasible and case.posterior([j for j in subset if j!=i])<tau for i in subset)/len(subset)
    sufficiency=sum(case.posterior([i])>=tau for i in subset)/len(subset)
    return dict(necessity=necessity,sufficiency=sufficiency,cfs=alpha*necessity+(1-alpha)*sufficiency)

def make_case(seed,kind,n,idx):
    r=random.Random(seed);weights=[r.uniform(-.6,1.65) for _ in range(n)]
    clusters=list(range(n));pairs={}
    if kind=='redundant':
        weights[0]=r.uniform(1.1,1.9)
        for i in range(1,min(n,5)):clusters[i]=0;weights[i]=weights[0]*r.uniform(.85,1.02)
    elif kind=='conflicting':
        for i in range(n):weights[i]=r.uniform(1.0,2.0) if i%2==0 else -r.uniform(.6,2.1)
    elif kind=='jointly_sufficient':
        weights=[r.uniform(-.4,.25) for _ in range(n)]
        a,b=r.sample(range(n),2);weights[a]=r.uniform(.4,.9);weights[b]=r.uniform(.4,.9)
        pairs[tuple(sorted((a,b)))]=r.uniform(2.1,3.4)
    elif kind=='synergy_trap':
        weights=[r.uniform(-.2,.1) for _ in range(n)]
        a,b=r.sample(range(n),2);weights[a]=.10;weights[b]=.10
        pairs[tuple(sorted((a,b)))]=3.4
        weights[next(i for i in range(n) if i not in (a,b))]=1.5
    elif kind=='mixed':
        for _ in range(n//3):
            a,b=r.sample(range(n),2)
            pairs[tuple(sorted((a,b)))]=r.uniform(-.8,1.6)
    return Case(weights,clusters,pairs,kind=kind,case_id=f'{kind}-{idx:04d}')

def run(outdir,seed=13,per_kind=50,tau=.9):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    kinds=['ordinary','redundant','conflicting','jointly_sufficient','synergy_trap','mixed']
    rows=[];cases=[]
    for k,kind in enumerate(kinds):
        for j in range(per_kind):
            n=3+(j*7+k*3)%10
            case=make_case(seed+100003*k+7919*j,kind,n,j)
            cases.append({'id':case.case_id,'kind':kind,'n':n,'weights':case.weights,'clusters':case.clusters,'pair_interactions':[[a,b,v] for (a,b),v in sorted(case.pairs.items())],'prior':case.prior})
            case.calls=0;t=time.perf_counter_ns();g=greedy(case,tau);gt=time.perf_counter_ns()-t;gc=case.calls
            case.calls=0;t=time.perf_counter_ns();o=exhaustive(case,tau);ot=time.perf_counter_ns()-t;oc=case.calls
            ga=audit(case,g,tau);oa=audit(case,o,tau)
            gf=bool(g);of=bool(o)
            assert not gf or case.posterior(g)>=tau-1e-12
            assert not of or case.posterior(o)>=tau-1e-12
            assert not gf or of
            rows.append({'case_id':case.case_id,'scenario':kind,'n':n,'greedy_feasible':int(gf),'optimal_feasible':int(of),
                         'greedy_size':len(g) if gf else '', 'optimal_size':len(o) if of else '',
                         'approx_ratio':len(g)/len(o) if gf and of else '',
                         'greedy_optimal':int(len(g)==len(o)) if gf and of else '',
                         'greedy_calls':gc,'exhaustive_calls':oc,'greedy_ns':gt,'exhaustive_ns':ot,
                         'greedy_cfs':ga['cfs'],'optimal_cfs':oa['cfs'],
                         'greedy_necessity':ga['necessity'],'optimal_necessity':oa['necessity'],
                         'greedy_sufficiency':ga['sufficiency'],'optimal_sufficiency':oa['sufficiency'],
                         'greedy_ids':json.dumps(g),'optimal_ids':json.dumps(o)})
    with (outdir/'per_case.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (outdir/'cases.json').write_text(json.dumps({'seed':seed,'tau':tau,'cases':cases},indent=2))
    def summarize(rs):
        feasible=[r for r in rs if r['optimal_feasible']]
        both=[r for r in feasible if r['greedy_feasible']]
        return {'n':len(rs),'optimal_feasible':len(feasible),'greedy_feasible':len(both),
                'greedy_feasibility_recall':len(both)/len(feasible) if feasible else None,
                'optimality_rate_conditional_on_both_feasible':sum(r['greedy_optimal'] for r in both)/len(both) if both else None,
                'mean_approx_ratio_conditional_on_both_feasible':statistics.mean(r['approx_ratio'] for r in both) if both else None,
                'max_approx_ratio':max((r['approx_ratio'] for r in both),default=None),
                'mean_greedy_posterior_evaluations':statistics.mean(r['greedy_calls'] for r in rs),
                'mean_exhaustive_posterior_evaluations':statistics.mean(r['exhaustive_calls'] for r in rs),
                'median_greedy_ms':statistics.median(r['greedy_ns'] for r in rs)/1e6,
                'median_exhaustive_ms':statistics.median(r['exhaustive_ns'] for r in rs)/1e6}
    summary={'evaluation':'measured computational results on generated controlled cases; NOT original NQ/HotpotQA/MetaQA/StrategyQA',
             'posterior_model':'declared joint log-odds with source-cluster deduplication and pairwise interactions',
             'seed':seed,'tau':tau,'per_kind':per_kind,'overall':summarize(rows),
             'by_scenario':{kind:summarize([r for r in rows if r['scenario']==kind]) for kind in kinds}}
    (outdir/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/audit_stress');p.add_argument('--seed',type=int,default=13);p.add_argument('--per-kind',type=int,default=50);p.add_argument('--tau',type=float,default=.9)
    a=p.parse_args();run(a.out,a.seed,a.per_kind,a.tau)
