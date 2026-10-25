# BEL-RAG extended research package

This extends the earlier executable demonstration without deleting it. See IMPLEMENTATION_STATUS.md for implemented and missing elements.

Install demo dependencies: `python -m pip install -r requirements-demo.txt`. For optional transformer/dense retrieval: `python -m pip install -r requirements-full.txt` (model weights must be downloaded separately).

Generate 600 **synthetic** policy cases: `python scripts/make_policyrag.py`.
Evaluate policy labels: `python -m scripts.evaluate_policy`.
Run 300 controlled adversarial audit cases: `python -m src.audit_benchmark --out results/audit_stress --seed 13 --per-kind 50 --tau 0.90`.
Run tests: `python -m pytest -q`.

Dataset adapters accept normalized JSON/JSONL question/answer records for NQ, HotpotQA-distractor, MetaQA and StrategyQA. Public datasets are not included in this ZIP: obtain their exact release and permitted split and log checksums. With externally produced predictions CSV containing `id,prediction`, evaluate via `python -m scripts.evaluate_predictions --dataset hotpotqa --data PATH.json --predictions PATH.csv`. This is a metric evaluation command, **not** a full RAG benchmark run.

Reference model identifiers in the manuscript: `mistralai/Mistral-7B-Instruct-v0.3`, `BAAI/bge-base-en-v1.5`. Optional wrappers do not reproduce the original experiments without fitted benchmark models, corpus snapshots, full baselines and trace logs.

Synthetic parameter/CA sweep: `python -m scripts.run_sensitivity`. The CA toggle is a controlled decision-gate ablation, not the original published ablation.
