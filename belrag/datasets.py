import json
from pathlib import Path
def load(name,path):
 if name not in ('nq','hotpotqa','metaqa','strategyqa'):raise ValueError(name)
 path=Path(path)
 if path.suffix=='.jsonl':
  with path.open() as f:rows=[json.loads(x) for x in f if x.strip()]
 else:
  raw=json.loads(path.read_text());rows=raw if isinstance(raw,list) else raw['data']
 for i,r in enumerate(rows):
  yield {'id':str(r.get('_id',r.get('id',i))),'question':r.get('question',r.get('query','')),'answer':r.get('answer',r.get('answers',r.get('short_answers',[]))),'raw':r}
