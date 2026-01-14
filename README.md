# BEL-RAG (Reference Implementation)

This is a **research/reference codebase** implementing the core ideas described in the attached manuscript:
**BEL-RAG: Bayesian Evidence-Leasing RAG with Counterfactual Auditing and Policy-Aware Retrieval**.

It provides end-to-end plumbing for the four modules:
1. Policy-Aware Retrieval (PAR)
2. Bayesian Evidence Leasing (EL)
3. Counterfactual Auditing (CA) with Minimal-Support + CFS
4. Calibrated Generation (CG) (LLM adapter interface + structured reporting)

> Note: This repo is **self-contained** and runnable without external LLM APIs. It includes
> a lightweight default "calibrator" and "generator" so you can run the pipeline on a toy corpus.
> For real experiments, plug in your embedding model / retriever / LLM via the provided interfaces.

## Quickstart

```bash
python -m belrag.demo
```

## Package layout

- `belrag/policy.py` : policy configuration, metadata checks, query profiling
- `belrag/retrieval.py` : TF-IDF vector retriever + chunk store
- `belrag/leasing.py` : sequential Bayesian evidence leasing with stop criteria (tau / cost budget)
- `belrag/auditing.py` : minimal-support greedy selection, necessity/sufficiency tests, CFS
- `belrag/generation.py` : calibrated generation/reporting (pluggable LLM adapter)
- `belrag/pipeline.py` : orchestration
- `belrag/types.py` : shared dataclasses

## License

MIT (for the reference implementation code).
