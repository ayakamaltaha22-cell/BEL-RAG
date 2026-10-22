import json,argparse
from belrag.core import Passage,Policy
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--data',default='data/policyrag_synthetic_600.jsonl');a=ap.parse_args();counts=dict(tp=0,tn=0,fp=0,fn=0)
for line in Path(a.data).read_text().splitlines():
 r=json.loads(line);e=Passage(id=r['id'],text=r['passage'],cluster=r['id'],**r['metadata']);v=Policy().allow(e);y=r['gold_allow']
 counts['tp' if v and y else 'tn' if not v and not y else 'fp' if v else 'fn']+=1
print(json.dumps(dict(origin='synthetic generated policy test',n=sum(counts.values()),confusion=counts),indent=2))
