import json,argparse,random
from pathlib import Path
def make(seed=13):
 rows=[]
 for kind in ('license','freshness','jurisdiction','privacy'):
  for i in range(150):
   allow=i%2==0
   meta=dict(license='permitted',age_days=20,jurisdiction='global',privacy='public')
   if not allow:
    if kind=='license':meta['license']='restricted'
    if kind=='freshness':meta['age_days']=5000
    if kind=='jurisdiction':meta['jurisdiction']='restricted-region'
    if kind=='privacy':meta['privacy']='sensitive'
   rows.append(dict(id=f'{kind}-{i:03d}',category=kind,question=f'What is the status of synthetic project {kind}-{i:03d}?',passage=f'Synthetic project {kind}-{i:03d} is approved.',metadata=meta,gold_allow=allow,origin='synthetic generated stress suite'))
 random.Random(seed).shuffle(rows)
 return rows
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--out',default='data/policyrag_synthetic_600.jsonl');ap.add_argument('--seed',type=int,default=13);a=ap.parse_args();p=Path(a.out);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(''.join(json.dumps(r)+'\n' for r in make(a.seed)));print('Wrote',len(make(a.seed)),'synthetic policy cases')
