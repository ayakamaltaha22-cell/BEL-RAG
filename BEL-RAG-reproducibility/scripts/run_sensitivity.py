"""Controlled synthetic sensitivity and ablation checks, not manuscript benchmark."""
import csv,argparse,json
from pathlib import Path
from belrag.core import Passage,Policy,Posterior,lease
ap=argparse.ArgumentParser();ap.add_argument('--out',default='results/synthetic_sensitivity.csv');a=ap.parse_args()
items=[Passage(f'e{i}',f'passage {i}',f'c{i}',log_lr=v) for i,v in enumerate([1.4,1.1,.7,-.4,.8,.2,1.0])]
rows=[]
for tau in (.7,.8,.9,.95):
 for lam in (.01,.05,.1,.2):
  for cap in (3,5,8):
   for ca in (True,False):
    result=lease(items,Policy(),Posterior(),tau=tau,lam=lam,max_passages=cap,kappa=.6 if ca else 0.)
    rows.append(dict(tau=tau,lambda_bits=lam,max_passages=cap,ca_enabled=ca,decision=result['decision'],leased=len(result['trace']),cfs=result['audit']['cfs'],origin='synthetic controlled sweep'))
p=Path(a.out);p.parent.mkdir(parents=True,exist_ok=True)
with p.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print(json.dumps(dict(rows=len(rows),path=str(p),origin='synthetic controlled sweep')))
