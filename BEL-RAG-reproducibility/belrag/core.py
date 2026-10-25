from dataclasses import dataclass
import math,itertools
@dataclass(frozen=True)
class Passage:
 id:str
 text:str
 cluster:str
 log_lr:float=0.
 cost:float=1.
 license:str='permitted'
 age_days:int=0
 jurisdiction:str='global'
 privacy:str='public'
 role:str='validation'
@dataclass
class Policy:
 max_age_days:int=3650
 def allow(self,e):return e.license=='permitted' and e.age_days<=self.max_age_days and e.jurisdiction=='global' and e.privacy=='public' and e.role=='validation'
def sigmoid(x):return 1/(1+math.exp(-x)) if x>=0 else math.exp(x)/(1+math.exp(x))
def prior_corrected_lr(q,pi):
 if not 0<q<1 or not 0<pi<1:raise ValueError('probabilities')
 return q/(1-q)/(pi/(1-pi))
class Posterior:
 def __init__(self,prior=.5,interactions=None):
  if not 0<prior<1:raise ValueError('prior')
  self.prior=prior;self.interactions=interactions or {}
 def __call__(self,subset):
  clusters={}
  for e in subset:
   if e.cluster not in clusters or abs(e.log_lr)>abs(clusters[e.cluster].log_lr):clusters[e.cluster]=e
  chosen={e.id:e for e in clusters.values()}
  logit=math.log(self.prior/(1-self.prior))+sum(e.log_lr for e in chosen.values())
  logit+=sum(v for (a,b),v in self.interactions.items() if a in chosen and b in chosen)
  return sigmoid(logit)
def greedy(items,posterior,tau=.9):
 chosen=[];remaining=sorted(items,key=lambda e:e.id)
 while remaining and posterior(chosen)<tau:
  p=posterior(chosen);e=max(remaining,key=lambda x:(posterior(chosen+[x])-p,-remaining.index(x)))
  if posterior(chosen+[e])-p<=1e-12:break
  chosen.append(e);remaining.remove(e)
 return chosen if chosen and posterior(chosen)>=tau else []
def exhaustive(items,posterior,tau=.9):
 if len(items)>12:raise ValueError('exhaustive limited to 12 passages')
 for k in range(1,len(items)+1):
  for group in itertools.combinations(sorted(items,key=lambda e:e.id),k):
   if posterior(group)>=tau:return list(group)
 return []
def audit(items,posterior,tau=.9,alpha=.5,method='greedy'):
 selected=(greedy if method=='greedy' else exhaustive)(items,posterior,tau)
 if not selected:return dict(feasible=False,support=[],necessity=0.,sufficiency=0.,cfs=0.)
 n=len(selected);nec=sum(posterior([x for x in selected if x.id!=e.id])<tau for e in selected)/n
 suf=sum(posterior([e])>=tau for e in selected)/n
 return dict(feasible=True,support=[e.id for e in selected],necessity=nec,sufficiency=suf,cfs=alpha*nec+(1-alpha)*suf)
def lease(items,policy,posterior,tau=.9,kappa=.6,budget=8.,max_passages=8,rounds=3,lam=.05):
 selected=[];trace=[];remaining=budget;used=set()
 for round_id in range(rounds):
  while len(selected)<max_passages and posterior(selected)<tau:
   p=posterior(selected)
   def entropy(x):return -x*math.log2(x)-(1-x)*math.log2(1-x) if 0<x<1 else 0
   eligible=[e for e in items if policy.allow(e) and e.id not in used and e.cost<=remaining]
   if not eligible:break
   scored=sorted([(entropy(p)-entropy(posterior(selected+[e]))-lam*e.cost,e.id,e) for e in eligible],key=lambda z:(-z[0],z[1]))
   if scored[0][0]<=0:break
   utility,_,e=scored[0];used.add(e.id);selected.append(e);remaining-=e.cost
   trace.append(dict(round=round_id+1,id=e.id,posterior=posterior(selected),utility_proxy_bits=utility))
  result=audit(selected,posterior,tau)
  if posterior(selected)>=tau and result['cfs']>=kappa and result['feasible']:
   return dict(decision='ACCEPT',audit=result,trace=trace,remaining_budget=remaining)
  if not any(policy.allow(e) and e.id not in used and e.cost<=remaining for e in items):break
 return dict(decision='ABSTAIN',audit=result,trace=trace,remaining_budget=remaining)
