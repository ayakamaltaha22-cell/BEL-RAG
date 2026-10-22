import re
def norm(s):return ' '.join(re.findall(r'[a-z0-9]+',str(s).lower()))
def exact_match(a,b):return int(norm(a)==norm(b))
def f1(a,b):
 from collections import Counter
 x=Counter(norm(a).split());y=Counter(norm(b).split());c=sum((x&y).values())
 return 2*c/(sum(x.values())+sum(y.values())) if x or y else 1.
def brier(p,y):return sum((a-b)**2 for a,b in zip(p,y))/len(y)
def ece(p,y,bins=15):
 total=0
 for b in range(bins):
  ids=[i for i,v in enumerate(p) if min(bins-1,int(v*bins))==b]
  if ids:total+=len(ids)/len(p)*abs(sum(p[i] for i in ids)/len(ids)-sum(y[i] for i in ids)/len(ids))
 return total
