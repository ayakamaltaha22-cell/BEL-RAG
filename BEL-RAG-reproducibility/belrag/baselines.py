import re,math
from collections import Counter
def bm25(query,passages):
 docs=[re.findall(r'\w+',e.text.lower()) for e in passages];q=re.findall(r'\w+',query.lower());N=len(docs);avg=sum(map(len,docs))/max(1,N);df=Counter(t for d in docs for t in set(d))
 def score(d):
  c=Counter(d);return sum(math.log(1+(N-df[t]+.5)/(df[t]+.5))*c[t]*1.9/(c[t]+.9*(.4+.6*len(d)/max(1,avg))) for t in q)
 return [e for _,e in sorted(zip(map(score,docs),passages),key=lambda x:(-x[0],x[1].id))]
def dense(query,passages,model_name='BAAI/bge-base-en-v1.5'):
 from sentence_transformers import SentenceTransformer
 model=SentenceTransformer(model_name);v=model.encode([query]+[e.text for e in passages],normalize_embeddings=True)
 return [passages[i] for i in sorted(range(len(passages)),key=lambda i:(-float(v[0]@v[i+1]),passages[i].id))]
def fixed_depth(ranked,k=5):return ranked[:k]
def hybrid(query,passages):
 a=dense(query,passages);b=bm25(query,passages);ia={e.id:i for i,e in enumerate(a)};ib={e.id:i for i,e in enumerate(b)}
 return sorted(passages,key=lambda e:(ia[e.id]+ib[e.id],e.id))
