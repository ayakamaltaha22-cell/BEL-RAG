# Scope and provenance

This repository is a research reference and evaluation harness, NOT a complete replication of the manuscript tables.

Implemented: synthetic 600-case policy generator, deterministic metadata policy evaluator, Bayesian prior correction, source-cluster-aware joint-log-odds surrogate, greedy/exhaustive audit and CFS, bounded leasing with proxy utility, BM25 retrieval, optional BGE dense retrieval, evaluation metrics, benchmark data adapters, controlled adversarial audit benchmark and tests.

Still requiring original trained artifacts / additional development: full validated Mistral hypothesis proposal/decomposition, fitted six-feature calibrator and conditional likelihood model, trained three-outcome expected utility, multi-claim dependency adjudication, faithful native implementations of SELF-RAG/CRAG/Adaptive-RAG/Bayesian RAG, exact benchmark splits and full historical experiment reproduction. An optional Mistral generation wrapper exists but is not wired into a validated full pipeline.

The generated policy cases are synthetic, not the manuscript's original PolicyRAG Stress Suite; labels follow deterministic metadata predicates and are not evidence of real-world generalization. Never present generated outcomes as historical paper measurements.

October 2026 update: Added `belrag/manuscript_methods.py`, tests, reported-sweep transcription, and alignment review. Existing `belrag/core.py` leasing utility remains a demonstration proxy, not the expected three-outcome utility in Section 6.3.2.
