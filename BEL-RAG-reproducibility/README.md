> **25 October 2026 manuscript alignment update:** Read [MANUSCRIPT_ALIGNMENT_2026-10-25.md](MANUSCRIPT_ALIGNMENT_2026-10-25.md) before using these materials. The original uploaded repository is retained, with a new mathematically specified three-outcome utility module and tests. The synthetic demonstrations are not historical benchmark replications.

# BEL-RAG reproducibility-study scaffold

**Status: proposed, not a replication of the manuscript's reported experiments.**

This archive includes a machine-readable *reference* configuration, proposed prompts, dataset manifest templates, and a small executable utility module. The numerical values for C, tau, lambda, maximum leased passages and related grids are user-confirmed as adopted for a proposed configuration, but no original experiment logs, fitted calibrator, exact dataset snapshots, original source code, or per-example outputs were supplied to this package.

## Contents
- `configs/reference.yaml`: parameter values, documented original settings, and explicit unresolved fields.
- `configs/search_grids.yaml`: proposed tuning grids.
- `prompts/`: three proposed prompt templates.
- `data/dataset_manifest.json`: dataset names, public landing pages, and missing release/checksum fields.
- `src/reference_utils.py`: basic prior correction, posterior update, and normalization utilities; **not the complete BEL-RAG system**.
- `results/STATUS.md`: explanation of missing experimental outputs.

## Quick smoke test
`python src/reference_utils.py`

## Before publishing claims of reproducibility
Pin exact dataset/model revisions and hashes; fill dataset counts, original prompts, fitted calibration coefficients and normalization statistics; implement PAR, EL, CA, CG and baseline pipelines; archive raw logs, predictions, per-seed metrics, policy cases and statistical scripts; rerun all manuscript tables. Do not cite this scaffold as evidence that the manuscript's numbers have been replicated.

## GitHub
These materials were prepared from a user-supplied archive described as the GitHub package. No live repository update or verification was performed here. Upload the reviewed files to the intended repository manually.

## Executable synthetic demonstration (added)

Run `python src/demo_pipeline.py` (requires `pip install -r requirements-demo.txt`).
Run `python tests/test_demo.py` for an end-to-end smoke test.
Actual outputs from the synthetic demonstration are saved under `results/demo/`:
`demo_metrics.json`, `per_example.csv`, `traces.json`, `demo_calibrator.json`, `fixture.json`.

**Critical scope limits:** This is a lightweight *partial reference implementation*, not the
full manuscript BEL-RAG system. It demonstrates policy prefiltering, TF-IDF candidate retrieval,
a toy fitted logistic evidence model, prior-corrected sequential odds updating, bounded
leasing, exhaustive support-subset auditing, a CFS decision and extractive answer reporting.
It does NOT implement the manuscript's BGE/FAISS retriever, Mistral generation, cross-encoder,
NLI, calibrated conditional evidence-dependence modeling, full utility lookahead, true
multi-round audit feedback, or benchmark adapters. Its toy calibration features and labels
are deliberately synthetic and are **not** scientifically valid estimates of passage
entailment. Results cannot be cited as the paper's benchmark results or used to validate
Tables 1–5. The sample outputs are measured by running the demo, not fabricated, but are
limited to six synthetic examples; in-sample calibration scores are not held-out results.
The main reference YAML remains a proposed full-scale configuration, not an executable
configuration for this reduced demonstration.

## Controlled counterfactual-audit approximation experiment (new)

Run from the repository root:

```bash
python -m src.audit_benchmark --out results/audit_stress --seed 13 --per-kind 50 --tau 0.90
python -m pytest -q tests/test_audit_benchmark.py
```

The script deterministically constructs **300 generated controlled cases** (six scenario classes, 50 each; 3–12 candidate passages), measures greedy marginal-posterior selection against exhaustive minimum-cardinality search, and writes `cases.json`, `per_case.csv`, and `summary.json`. The generated cases are reproducible from the seed and contain their complete weights, source clusters, and pairwise interaction terms. Posterior evaluations are counted; nanosecond wall times are recorded but are hardware-dependent and should not be generalized from a single machine. A feasibility failure is **not** counted as a cardinality approximation ratio. CFS is computed using the binary necessity and singleton-sufficiency averages with alpha=0.5.

**Scope and limitations.** This is an actual executed computational study on **synthetically generated controlled evidence**, not on the manuscript's Natural Questions, HotpotQA, MetaQA, StrategyQA or PolicyRAG benchmark records. The joint log-odds model with cluster deduplication and pairwise interaction terms is explicitly defined in `src/audit_benchmark.py` and is not the manuscript's unreleased fitted six-feature dependence-aware benchmark model. It evaluates Section 7's subset-search behavior, not the end-to-end BGE/FAISS, Mistral, PAR/EL/CA/CG benchmark. It must not be used to claim independent reproduction of the manuscript's Tables 1–5. The `synergy_trap` cases deliberately expose non-optimal greedy solutions; they are not sampled from a real-world distribution.

For a benchmark-level comparison, provide frozen original claim/evidence records, the fitted conditional posterior callable, source-cluster identifiers, and exact audit thresholds; replace the controlled posterior function and rerun the same evaluation on the held-out benchmark evidence sets.

### Coverage map against the complete manuscript

| Manuscript component | Existing public demo | New audit experiment | Benchmark replication status |
|---|---|---|---|
| 4.4 draft/atomic claims and evidence separation | toy fixed claims | not tested | original traces unavailable |
| 5 policy-aware retrieval | allow/block toy prefilter | not tested | benchmark policy files unavailable |
| 6 calibrated EL/utility/termination | simplified toy update | not tested | fitted outcome predictor unavailable |
| 7 support search, necessity/sufficiency, CFS | exhaustive on tiny fixture | greedy vs exhaustive with full subset posterior and adversarial interactions | controlled synthetic study only |
| 8 calibrated final generation | extractive demo | not tested | Mistral pipeline unavailable |
| 9 benchmark datasets/metrics | six synthetic questions | 300 generated evidence sets | not reproduced |

The new experiment **extends rather than replaces** the original package. The source manuscript has inconsistencies between some documented reference and sensitivity settings, and its CFS weight sweep labels do not match the necessity/sufficiency alpha; this implementation uses the explicit Section 7 alpha=0.5 and tau=0.90 definitions and does not silently reconcile the historical benchmark.
