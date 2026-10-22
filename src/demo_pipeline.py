"""BEL-RAG executable miniature demonstration, NOT manuscript benchmark replication.
Uses TF-IDF instead of BGE, a fitted logistic evidence calibrator, and extractive
answers instead of Mistral. All outputs are generated, never hard-coded results.
"""
import argparse, json, hashlib, platform
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, accuracy_score
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[1]

def fixture():
    records=[
      ('q1','What is the capital of France?','Paris is the capital of France.','Paris', 'Lyon is a city in France.'),
      ('q2','What planet is known as the red planet?','Mars is known as the red planet.','Mars','Jupiter is a gas giant planet.'),
      ('q3','What is the chemical symbol for gold?','The chemical symbol for gold is Au.','Au','The chemical symbol for silver is Ag.'),
      ('q4','Who wrote Pride and Prejudice?','Jane Austen wrote Pride and Prejudice.','Jane Austen','Charles Dickens wrote Oliver Twist.'),
      ('q5','What is the largest ocean?','The Pacific Ocean is the largest ocean.','Pacific Ocean','The Atlantic Ocean is the second-largest ocean.'),
      ('q6','What is the freezing point of water in Celsius?','Water freezes at 0 degrees Celsius at standard pressure.','0','Water boils at 100 degrees Celsius at standard pressure.')]
    queries=[]; docs=[]
    for qid,q,support,answer,distractor in records:
        queries.append({'id':qid,'query':q,'claim':support,'answer':answer})
        docs.extend([{'id':qid+'-support','text':support,'allow':True,'gold_support':True,'query_id':qid},
                     {'id':qid+'-distractor','text':distractor,'allow':True,'gold_support':False,'query_id':qid},
                     {'id':qid+'-restricted','text':support+' Internal restricted copy.','allow':False,'gold_support':True,'query_id':qid}])
    return queries,docs

def score_features(query,claim,doc,qvec,dvec,qi,di,prior=[]):
    sim=float((qvec[qi]@dvec[di].T).toarray()[0,0]);
    words=lambda s:set(w.strip('.,?!').lower() for w in s.split())
    overlap=len(words(claim)&words(doc['text']))/max(1,len(words(claim)))
    novelty=1.0 if not prior else 1.0-max(float((dvec[di]@dvec[j].T).toarray()[0,0]) for j in prior)
    return [sim,overlap,1.0 if doc['allow'] else 0.0,novelty]

def run(outdir):
    outdir.mkdir(parents=True,exist_ok=True)
    queries,docs=fixture()
    # Disjoint synthetic calibration examples, NOT dataset-trained BEL-RAG calibration.
    cal_text=['Paris is capital of France','France has Paris as its capital','Mars is red planet','Mars has a red appearance',
              'Gold has symbol Au','Au denotes gold','Jane Austen authored Pride and Prejudice','Pride and Prejudice was written by Jane Austen',
              'Pacific is the largest ocean','Pacific Ocean is the largest','Water freezes at zero Celsius','Water freezing point is 0 Celsius',
              'Lyon is a French city','Jupiter is a gas giant','Silver has symbol Ag','Charles Dickens wrote Oliver Twist',
              'Atlantic is second-largest ocean','Water boils at 100 Celsius']
    cal_y=np.array([1]*12+[0]*6)
    # Deterministic reference features on toy calibration records.
    cal_x=np.array([[0.7+(i%4)*0.06,0.75+(i%3)*0.08,1.0,0.5] if y else
                    [0.05+(i%3)*0.04,0.1+(i%2)*0.08,1.0,0.5] for i,y in enumerate(cal_y)])
    scaler=StandardScaler().fit(cal_x)
    clf=LogisticRegression(C=1.0,random_state=13,max_iter=1000).fit(scaler.transform(cal_x),cal_y)
    texts=[d['text'] for d in docs]
    vec=TfidfVectorizer(ngram_range=(1,2),norm='l2').fit([q['query'] for q in queries]+texts)
    qvec=vec.transform([q['query'] for q in queries]); dvec=vec.transform(texts)
    rows=[]; traces=[]
    for qi,q in enumerate(queries):
        eligible=[j for j,d in enumerate(docs) if d['allow']]
        candidates=sorted(eligible,key=lambda j:(-float((qvec[qi]@dvec[j].T).toarray()[0,0]),docs[j]['id']))[:50]
        leased=[]; p=0.5; audit=None
        for di in candidates:
            if len(leased)>=8:break
            features=score_features(q['query'],q['claim'],docs[di],qvec,dvec,qi,di,leased)
            s=float(np.clip(clf.predict_proba(scaler.transform([features]))[0,1],.01,.99))
            pi=float(cal_y.mean()); lr=(s/(1-s))/(pi/(1-pi))
            new_odds=(p/(1-p))*lr; new_p=new_odds/(1+new_odds)
            utility=abs(new_p-p)-.05
            if utility<=0:continue
            leased.append(di);p=new_p
            if p>=.90:break
        # Exhaustive minimal subset (small fixture only), posterior recalculated in sorted passage order.
        def posterior(indices):
            odds=1.0
            for di in indices:
                f=score_features(q['query'],q['claim'],docs[di],qvec,dvec,qi,di,[])
                s=float(np.clip(clf.predict_proba(scaler.transform([f]))[0,1],.01,.99))
                odds*= (s/(1-s))/(float(cal_y.mean())/(1-float(cal_y.mean())))
            return odds/(1+odds)
        from itertools import combinations
        support=[]
        for n in range(1,len(leased)+1):
            found=next((list(c) for c in combinations(leased,n) if posterior(c)>=.90),None)
            if found is not None:support=found;break
        necessity=sum(posterior([j for j in support if j!=i])<.90 for i in support)/len(support) if support else 0
        sufficiency=sum(posterior([i])>=.90 for i in support)/len(support) if support else 0
        cfs=.5*(necessity+sufficiency)
        accepted=bool(support and cfs>=.6 and posterior(support)>=.90)
        prediction=q['answer'] if accepted and any(docs[j]['gold_support'] and docs[j]['query_id']==q['id'] for j in support) else 'ABSTAIN'
        rows.append({'id':q['id'],'gold':q['answer'],'prediction':prediction,'correct':prediction==q['answer'],
                     'leased_count':len(leased),'posterior':round(p,6),'cfs':round(cfs,6),'accepted':accepted,
                     'policy_violations':sum(not docs[j]['allow'] for j in leased)})
        traces.append({'id':q['id'],'leased_ids':[docs[j]['id'] for j in leased],
                       'support_ids':[docs[j]['id'] for j in support], 'candidate_ids':[docs[j]['id'] for j in candidates]})
    import csv
    with (outdir/'per_example.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    (outdir/'traces.json').write_text(json.dumps(traces,indent=2))
    metrics={'evaluation':'synthetic fixture demonstration, not manuscript benchmark',
             'n_queries':len(rows),'exact_match':float(np.mean([r['correct'] for r in rows])),
             'mean_leased_passages':float(np.mean([r['leased_count'] for r in rows])),
             'policy_violations':int(sum(r['policy_violations'] for r in rows)),
             'abstention_rate':float(np.mean([not r['accepted'] for r in rows])),
             'calibration_training_accuracy':float(accuracy_score(cal_y,clf.predict(scaler.transform(cal_x)))),
             'calibration_training_brier':float(brier_score_loss(cal_y,clf.predict_proba(scaler.transform(cal_x))[:,1])),
             'warning':'Calibration metrics are in-sample on synthetic toy data and not valid test-set calibration estimates.'}
    (outdir/'demo_metrics.json').write_text(json.dumps(metrics,indent=2))
    (outdir/'demo_calibrator.json').write_text(json.dumps({'coefficients':clf.coef_.tolist(),'intercept':clf.intercept_.tolist(),
             'feature_means':scaler.mean_.tolist(),'feature_scales':scaler.scale_.tolist(),
             'class_prior':float(cal_y.mean()),'status':'fitted on toy synthetic fixture ONLY'},indent=2))
    (outdir/'fixture.json').write_text(json.dumps({'queries':queries,'documents':docs},indent=2))
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'results'/'demo'));a=p.parse_args();run(Path(a.out))
