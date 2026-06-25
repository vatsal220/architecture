# Toronto Data Platform — Citizen Engagement Recommendation System

A reference architecture **and** a working ML training prototype for a
recommendation capability on the City of Toronto Data Platform (TDP): surfacing
relevant recreation programs, community services, and events to citizens based on
historical interaction patterns.

---

## The Pipeline

A single entry point runs the seven canonical stages end-to-end. Each stage is
idempotent and writes to a data-lake-style zone (Raw → Standard → Curated →
Features → Models), mirroring the target S3 lake.

| Stage | Module | What it does |
|-------|--------|--------------|
| `01_load_data` | `src/data/loader.py` | `MovieLensLoader` fetches MovieLens `ml-100k` (auto-download + cache) → Raw zone |
| `02_validate_data` | `src/data/quality.py` | 6 input quality checks (blocking gate) |
| `03_generate_features` | `src/feature/engineer.py` | Implicit-confidence matrix + user/item features |
| `04_train_model` | `src/model/train.py` | Hybrid SVD matrix factorisation + popularity |
| `05_evaluate_model` | `src/model/evaluate.py` | P/R/MAP/NDCG@K, coverage, diversity, cold-start |
| `06_generate_recommendations` | `src/model/inference.py` | Batch top-N slates + post-scoring quality gate |
| `07_publish_results` | `src/model/deploy.py` | Publish to Curated + auditable run manifest |

The model is registered with lineage and passed through a **metric promotion
gate** (`src/model/registry.py`) — the same gate CI/CD would enforce for
Dev→Test→Prod promotion.

---

## Quick Start

Requires Python 3.11 (works on 3.9+). On the first run the loader **fetches the
MovieLens `ml-100k` dataset automatically** (~5 MB) and caches it under
`data/raw/`; later runs reuse the cache and need no network.

### Option A — local venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the whole pipeline (single entry point):
python run_pipeline.py
# or:
make train
```

### Option B — Docker

```bash
docker compose up pipeline --build   # train + batch-score once
docker compose up serving --build    # then serve the latest model on :8080
```

---

## Outputs

After a run, inspect:

| File | Contents |
|------|----------|
| `outputs/reports/data_quality_input.json` | Input quality report (every check, severity, action) |
| `outputs/reports/data_quality_post_scoring.json` | Post-scoring quality report |
| `outputs/reports/evaluation.json` | Ranking metrics incl. K-sweep and cold-start slice |
| `outputs/models/<version>/metadata.json` | Model lineage (git SHA, config, data fingerprint, metrics) |
| `outputs/models/registry.json` | Registry index + `latest` pointer |
| `outputs/publish/latest_manifest.json` | Auditable batch manifest (model, metrics, quality, promotion) |
| `outputs/publish/recommendations_<batch>.csv` | Human-readable recommendation slates |

A representative run (~3s, full 100k interactions / 943 users) yields at K=10:
Precision ≈ 0.16, Recall ≈ 0.11, MAP ≈ 0.09, NDCG ≈ 0.18, Coverage ≈ 0.36,
Diversity ≈ 0.75 on **MovieLens `ml-100k`**. The deliverable is the measurement +
gating machinery, not the absolute score.

---

## Data Quality & Validation

Six input checks (`src/data/quality.py`), each with a severity. **BLOCK** aborts
the run (above its threshold); **WARN** repairs and continues. All outcomes are
persisted for audit.

| # | Check | Why it exists | Blocks? | Failure handling |
|---|-------|---------------|---------|------------------|
| 1 | **Schema validation** | Downstream assumes the canonical schema; a missing/typed-wrong column is a contract breach | **Yes** | Abort (cannot repair) |
| 2 | **Missing identifiers** | A recommendation needs both citizen and program; nulls can't be imputed | **Yes**, above tolerance | Quarantine rows; abort if fraction > threshold |
| 3 | **Duplicate records** | Duplicates over-weight a pair and bias the confidence matrix | No (WARN) | Drop duplicates (keep first) |
| 4 | **Invalid timestamps** (null/non-positive/future) | Recency & the temporal split depend on them; a future timestamp leaks the future | No (WARN) | Quarantine offending rows |
| 5 | **Invalid interaction values** (out of range) | Distort confidence weighting; a high rate signals an upstream bug | **Yes**, above tolerance | Quarantine; abort if fraction > threshold |
| 6 | **Reference integrity** (items) | Content features & diversity need item metadata; items can't be enriched/explained | **Yes**, above tolerance | Drop items; abort if fraction > threshold |
| (7) | **Minimum volume floor** | A tiny batch usually means a broken extract; CF degrades to noise | **Yes** | Abort |

Five **post-scoring** checks then guard the generated recommendations
(completeness, reference integrity, no duplicate items in a slate, slate
completeness, catalogue-coverage floor).

> MovieLens `ml-100k` is already clean, so on real data these gates typically
> report all-clear. Each check is exercised against crafted defects (nulls,
> out-of-range ratings, future timestamps, items, duplicates) in
> `tests/unit/test_quality.py`.

---

## Model Evaluation

Ranking-aware metrics (`src/model/evaluate.py`), because the division consumes an
**ordered slate**, not rating predictions:

* **Precision@K / Recall@K** — relevance of, and need-coverage by, the slate.
* **MAP@K** — rewards ranking relevant items higher.
* **NDCG@K** — position-discounted gain; the **promotion-gate** metric.
* **Coverage** — catalogue share recommended (popularity-collapse & equity guard).
* **Diversity** — intra-list category dissimilarity (avoids near-duplicate slates).
* **Cold-start slice** — the same metrics on low-history users.

Why these and not RMSE: we serve rankings, so ordering and position are what
matter to the citizen experience and to equity. Full rationale in
[`architecture/solution_architecture.md`](architecture/solution_architecture.md) §9.4.

---

## Development notes

Trunk-based development: features land via short-lived branches and integrate
into `main`, which acts as the production branch
([trunk-based workflow](https://trunkbaseddevelopment.com/)). Python 3.11 is the
target (3.9+ supported). Runtime dependencies are intentionally minimal and
pure-wheel so they install identically on a laptop, in CI, and in the container
images deployed to EKS.

### Environment variables

Account-scoped credentials and the active `ENV` are read from the process
environment or a local `.env` (never committed). Behaviour-defining parameters
live in `configs/pipeline.yaml`. See `src/utils/config.py`.

### Testing

```bash
make test-unit
make test-e2e
```

CI (`.github/workflows/test-executor.yml`) runs the unit suite, the e2e smoke
test, and a real pipeline run on every push/PR to `main`/`staging`.