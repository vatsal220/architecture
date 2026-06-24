# Recommendation Architecture Design — Toronto Data Platform (TDP)
**Document owner:** Vatsal Patel  
**Status:** Proposed  
**Audience:** TDP Platform Team  
**Companion documents:**  
- [`tool_selection.md`](./tool_selection.md)  
- [`decision_log.md`](./decision_log.md)   
- [`operational_scenarios.md`](./operational_scenarios.md)  

---
## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Goals, Non-Goals, Open Questions](#goals-non-goals-open-questions)
3. [Functional Requirements & Resources](#functional-requirements--resources)
4. [Data Ingestion](#data-ingestion)
5. [Feature Engineering](#feature-engineering)
6. [Model Training](#model-training)
7. [Modelling](#modelling)
8. [Inferencing](#inferencing)
9. [Data Quality & Logging](#data-quality--logging)
10. [Multi-Account Deployment](#multi-account-deployment)
11. [Security & Governance (Risk Assessment)](#security--governance-risk-assessment)
12. [Challenges and Risks](#challenges-and-risks)
13. [User Acceptance Testing](#user-acceptance-testing)
14. [Documentation](#documentation)
15. [Appendix](#appendix)

---
## Executive Summary  
The document outlines an reusable architecture for providing recommendations for the Toronto Data Platform for the City of Toronto. These recommendations will be provided to improve citizen engagement by surfacing relevant services, programs, events and resources from historical interaction patterns. This document outlines a batch first, AWS native recommendation architecture built on the existing TDP data lake (raw → standard → curated → features → models) and Kubernetes, deployed across the Dev/Test/Prod multi-account structure with Lake Formation governance, KMS encryption and CodePipeline promotion.  

The accompanying prototype implements the full seven-stage pipeline end to end on a MovieLens public dataset reframed as citizens x recreation programs. It demonstrates the data quality gates, feature engineering, a hybrid matrix factorization + popularity model, ranking evaluations (precision, recall, MAP, NDCG@K, coverage, diversity, cold-start), a local model registry with a promotion gate, and a published recommendations artifact with an auditable run manifest. A FastAPI service then serves the registered model over HTTP, the runnable form of EKS real time path with Kubernetes manifests under deploy/k8s. The loader fetches MovieLens ml-100k automatically on first run; the pipeline then completes in ~3 seconds and is the literal, runnable encoding of the architecture below.

This design is optimized for engineering judgement, governability and operability on a public sector platform, not for the last point of recommendation accuracy*. Every choice below favours the simpler solution that is correct, observable, secure, and cheap to run, with clear seams to scale up when (and only when) the need is demonstrated.

**Phase 1 - Batch Recommendations**  
A pipeline which runs on a cron expression that generates a ranked slate of programs per citizen, writes them to the curated zone and exposes them through Athena or an API the division already consumes. This covers the overwhelming majority of civic use cases (a citizen browsing programs does not need millisecond personalization) at a fraction of the cost and operational burden of real-time serving.  

**Phase 2 - Real Time Re-Ranking**  
Executing a FastAPI service on EKS can be backed by an online feature store (ElastiCache Redis) and a two-stage candidate-generate + re-rank flow behind the same contracts when a latency-sensitive surface (e.g. interactive portal) justifies it.  

---
## Goals, Non-Goals, Open Questions
### Goals

| # | Goal | Execution |
|---|------|-----------|
| G1 | Recommend relevant programs/services per citizen from historical interactions. | Hybrid CF + popularity model, batch slates. |
| G2 | Run entirely within the governed AWS multi-account platform. | EKS + S3 lake + Lake Formation + KMS + ECR + CodePipeline. |
| G3 | Reproducible, testable ML pipeline. | Config-driven 7-stage DAG, deterministic seeds, unit/e2e tests. |
| G4 | Strong data-quality and governance posture. | Input and post scoring data quality checks, blocking gates, audit manifest. |
| G5 | Handle cold-start (new citizens / new programs) gracefully. | Popularity + content-based fallback path. |
| G6 | Operable: monitorable, retrainable, with incident runbooks. | CloudWatch metrics, drift monitors, runbooks. |
| G7 | Safe promotion Dev→Test→Prod with a quality gate. | Metric-gated model registry promotion. |

### Non-Goals

| # | Non-Goal | Rationale |
|---|----------|-----------|
| N1 | Real-time / sub-100ms personalization in Phase 1. | Civic browsing tolerates daily freshness; defer cost & ops until justified. |
| N2 | Deep-learning sequence models (e.g. transformers, SASRec). | Data volume and explainability needs don't warrant them yet; revisit at scale. |
| N3 | Cross-division identity resolution / a unified citizen graph. | Major privacy/governance program in its own right; out of scope. |
| N4 | Collecting new PII or new tracking. | We use only data the platform already holds under existing consent. |
| N5 | A bespoke feature platform. | Reuse S3+Glue (offline) and Redis/Feast (online) rather than build one. |

### Open Questions
These require division / governance input before Prod rollout:

1. Under what legal basis / consent were the source interactions collected, and does "personalised recommendation" fall within that purpose? (Drives whether opt-out is required.)  
2. Is there a stable pseudonymous citizen id available in the Curated zone, or must we operate session/household-level? Affects personalization depth and re-identification risk.  
3. Is a "relevant" recommendation a *registration*, an *attendance*, a *click*, or a *repeat visit*? This defines the label and the evaluation target.  
4. Which channel renders recommendations (portal, email, staff tool) and what freshness/latency does it actually need. Decides whether Phase 2 real-time is ever required.  
5. How much structured program metadata (category, location, age band, cost) is reliably available for content-based cold-start?

---
## Functional Requirements & Resources
### Functional Requirements
| ID | Requirement |
|----|-------------|
| FR1 | Ingest interaction, service metadata, and event data into the Raw zone on a scheduled cadence. |
| FR2 | Validate and cleanse data into Standard/Curated zones with blocking quality gates. |
| FR3 | Generate and version reusable ML features with train/serve consistency. |
| FR4 | Train, evaluate and register a ranking model through running experiments. |
| FR5 | Produce a top N ranked slate per eligible citizen on a cron schedule. |
| FR6 | Publish recommendations to a governed, queryable location with an audit manifest. |
| FR7 | Monitor data quality, model performance, and operational health; alert on breach. |
| FR8 | Promote artifacts Dev → Staging → Prod via CI/CD with a quality gate and rollback. |

### Non-Functional Requirements
| ID | Requirement | Target (Phase 1) |
|----|-------------|------------------|
| NFR1 | Batch freshness | Recommendations no older than 24h. |
| NFR2 | Pipeline runtime | Full daily run < 2h at city scale. |
| NFR3 | Reproducibility | Re-runnable from `(config + data snapshot + git SHA)`. |
| NFR4 | Encryption | At rest (KMS) and in transit (TLS) everywhere. |
| NFR5 | Auditability | Every published batch traceable to model, data, config. |
| NFR6 | Availability | Batch job 99% successful nightly; auto-retry + alert. |
| NFR7 | Cost | Run on serverless/managed spot where possible. |

### AWS Resources by Platform Layer

| Concern | AWS Service | Role in this Solution |
|---------|-------------|-----------------------|
| Data lake storage | **Amazon S3** | Raw/Standard/Curated/Features/Models zones, per-account buckets |
| Cataloging / ETL | **AWS Glue** (Crawler, Catalog, Jobs) | Schema discovery, Standard/Curated transforms, table registration |
| Fine-grained governance | **AWS Lake Formation** | Table/column/row permissions, tag-based access, cross-account grants |
| Identity | **AWS IAM** | Execution roles (least privilege), cross-account assume-role |
| Encryption | **AWS KMS** | CMKs per zone/account; envelope encryption for S3, EBS, Redis |
| Ad-hoc query | **Amazon Athena** | Divisional access to Curated recommendations + analytics |
| Compute platform | **Amazon EKS** (Kubernetes) | Runs all ML jobs, the batch CronJob, and the FastAPI service; node autoscaling via Karpenter, spot for batch/training |
| Container registry | **Amazon ECR** | Stores the pipeline + serving images promoted across accounts |
| Pipeline orchestration | **Argo Workflows** (on EKS) | The 7-stage DAG as a retriable workflow; CronWorkflow for retrains |
| ML jobs (ETL/features/train/eval/score) | **Kubernetes Jobs** (image on EKS) | Each pipeline stage runs as a container step |
| Real-time serving | **FastAPI** on EKS (Deployment + ALB Ingress + HPA) | Low-latency recommendation API |
| Batch serving | **Kubernetes CronJob** | Nightly batch scoring → Curated zone |
| Model registry + experiment tracking | **MLflow** (self-hosted on EKS; S3 + RDS backend) | Versioning, lineage, promotion approval, run tracking |
| Online feature store | **Amazon ElastiCache (Redis)** | Low-latency features for real-time serving |
| Drift / model monitoring | **Evidently** + **Prometheus/Grafana** | Data + model drift, service metrics |
| GitOps deployment | **Argo CD** | Syncs `deploy/k8s` manifests to each cluster |
| CI/CD | **AWS CodePipeline / CodeBuild** (+ GitHub Actions) | Build, test, push to ECR, infra deploy, model promotion |
| Observability | **Amazon CloudWatch** (Container Insights) + **EventBridge** | Logs, metrics, dashboards, alarms, schedule triggers |
| Notifications | **Amazon SNS** | Alert routing to on-call / division |

---
## Data Ingestion
**How interaction data enters the platform.**
| Aspect | Approach |
|--------|----------|
| **Interaction data** (registrations, attendance, repeat visits) | Periodic extract from source systems landed to **Raw** S3 (CSV/Parquet/JSON). Where a database source exists, **AWS DMS** or a scheduled **Glue** job; where files are dropped, an S3 landing prefix with EventBridge trigger. |
| **Service / program metadata** | Reference extract (program id, name, category, location, age band, cost, season) to Raw; refreshed on change or daily. The catalogue is small and slowly changing. |
| **Event data** | Event listings (one-off/seasonal) ingested like metadata; flagged with validity windows so expired events are excluded at scoring (§10.4). |
| **Historical activity** | Bulk backfill to Raw once, then incremental. Used for the initial training corpus. |

**Principles.** Raw is **immutable and append-only** (the system of record copy); all cleansing happens downstream so we can always reprocess. Every object is KMS-encrypted and partitioned (`source/dt=YYYY-MM-DD/`). A **Glue Crawler**
registers schemas into the Catalog; Lake Formation governs access from Raw onward. Ingestion is **idempotent** (re-landing a partition is safe).

---
## Feature Engineering
| Feature Component | Process |
|----------|--------|
| **Feature generation** | From the Standard/Curated interaction history we build: (a) a sparse **user × item implicit-confidence matrix** (`c = 1 + α·w`, where `w` blends a normalised engagement signal with **exponential recency decay** so recent activity counts more); (b) **user features** (activity count, recency, category breadth, preferred category); (c) **item features** (recency-weighted popularity, average engagement, category one-hots used for content cold-start and diversity). |
| **Feature storage** | Features are written to the **Features** S3 zone as Parquet + sparse-matrix artifacts, registered in Glue (the **offline** store, point-in-time correct for training/batch). For real-time, an **online** store on **ElastiCache Redis** (optionally fronted by **Feast** for a unified feature API) serves low-latency lookups. |
| **Feature reuse** | Features are defined once in `src/feature/engineer.py` and consumed by training, evaluation, batch inference, and the FastAPI service. A Glue-registered offline store (or Feast feature views) makes them discoverable and reusable across future models/divisions. |
| **Train/inference consistency** | The **same module** computes features for training and serving, and the **fitted artefacts** (id maps, category vocabulary, recency reference date) are persisted with the model so batch scoring and the API apply identical transforms. In production the offline (S3) and online (Redis) stores are populated from the same job, giving offline/online parity and preventing training-serving skew. |

---
## Model Training
### Caveats
* **Implicit, biased feedback.** We only observe positive engagement; absence is not a true negative (a citizen may simply never have seen a program). The model and metrics treat unobserved items as candidates, not as negatives.
* **Popularity bias.** Engagement concentrates on a few popular programs; without care the model recommends the already-popular. We counter with recency weighting and we *monitor* coverage/diversity, not just accuracy.
* **Feedback loops.** Once recommendations drive engagement, future training data is partly caused by the model. We log served slates so this can be measured and (later) corrected (e.g. inverse-propensity weighting).
* **Selection bias in evaluation.** Offline metrics are computed only on citizens with enough history; the cold-start population is reported separately.

### Cold-start problem
| Cold Entity | Strategy |
|-------------|----------|
| **New citizen** (no/low history) | Route to the **popularity + content** path: recency-weighted popular programs, optionally blended with content affinity if any signal exists (e.g. first interaction's category, or supplied preferences). Triggered when history < `cold_start_min_interactions`. |
| **New program** (no interactions) | **Content-based**: represent the program by its metadata (category one-hots; extensible to location/age/cost) so it can be recommended by similarity before it accrues interactions. |
| **Both** | Fall back to globally popular / editorially curated slates the division can override. |

This is implemented in `HybridRecommender.score_users` (per-user routing) and
exercised by the prototype; the evaluation reports a dedicated **cold-start
slice** (least-active quartile of evaluable users).

### Retraining Cadence
| Trigger | Cadence / Condition |
|---------|---------------------|
| **Scheduled** | Full retrain **weekly** (program catalogue and seasonal demand shift slowly); features refreshed nightly for scoring. |
| **Event-driven** | Retrain on **seasonal boundaries** (e.g. summer/winter program launches) and on large catalogue updates. |
| **Drift-driven** | Retrain when monitoring detects input drift or a sustained drop in online/offline quality beyond threshold (§11.4). |
| **Guardrail** | Every retrain must pass the **promotion gate** (NDCG ≥ floor) before it can replace the incumbent; otherwise the incumbent is retained and on-call is alerted. |

### Input Data (Training)
Cleansed, de-duplicated interactions from the **Standard/Curated** zones joined to program metadata, restricted to a configurable history window, split by a **per-user temporal holdout** (most-recent fraction held out) to prevent leakage
and to simulate "predict the next program".

---
## Modelling
### Model Selection
A **hybrid** of two deliberately simple, explainable components:

1. **Collaborative filtering via matrix factorisation** — `TruncatedSVD` on the implicit-confidence matrix yields user/item latent factors; score = dot product. Captures "citizens like you also engaged with…".
2. **Recency-weighted popularity** — robust prior for cold-start and a safety fallback.

Routed per user by history depth. *Why this tier and why not LightFM / implicit / deep models is argued in detail in*
[`tool_selection.md`](./tool_selection.md). In short: it is correct, cheap, explainable to a public-sector stakeholder, has trivial ops, and establishes the metric baseline that any fancier model must beat to justify its cost.

### Experimentation
* **Hypothesis-driven:** each experiment changes one lever (factors, recency half-life, confidence α, hybrid threshold) against a fixed split and seed.
* **Tracked:** parameters, metrics, dataset fingerprint, and git SHA are logged. On AWS this is **MLflow Tracking** (self-hosted on EKS); the prototype writes the same tuple to `metadata.json` per run.
* **Comparable:** all runs report the identical metric suite at identical K so they are directly rankable.

### Experiment Structure
```
experiment (e.g. "recency-decay-sweep")
└── trials (one per hyper-parameter set)
    ├── params      : n_factors, alpha, half_life, k, seed
    ├── inputs      : data fingerprint, feature version, git SHA
    ├── metrics     : P@K, R@K, MAP@K, NDCG@K, coverage, diversity, cold-start
    └── artifacts   : model.npz, id_maps.json, evaluation.json, manifest
```
Determinism (fixed seeds, pinned deps, config-as-code) makes a trial reproducible from its recorded tuple.

### Model Tuning  
| Aspect | Approach |
|--------|----------|
| **Tunable levers** | `n_factors` (latent dimensionality), implicit-confidence `alpha`, recency `half_life_days`, hybrid `cold_start_min_interactions`, ranking `K`. |
| **Search strategy** | Start with a small **grid / sweep** over the few levers that matter (the parameter space is tiny), driven by config overrides (`--set model.svd_mf.n_factors=128`). Scale to **Optuna** (Bayesian HPO, run as parallel K8s Jobs) when the model graduates to FM/LightFM with a larger space. |
| **Objective** | Maximise **NDCG@K** on the temporal holdout (the promotion-gate metric), subject to coverage/diversity floors so tuning can't trade away equity. |
| **Guardrails** | All trials share one split + seed for comparability; the best trial must still clear the promotion gate before it can be registered/promoted. |
| **Cost control** | Tuning runs in Dev/Test on managed/spot compute; the cheap baseline means sweeps are fast and inexpensive. |

### Evaluation
Ranking-aware metrics, because the division consumes an **ordered slate**:
| Metric | Why it is appropriate |
|--------|-----------------------|
| **Precision@K** | Of the K shown, how many were relevant — directly the citizen's experience. |
| **Recall@K** | Of relevant programs, how many we surfaced — coverage of citizen need. |
| **MAP@K** | Rewards placing relevant items **higher** in the slate. |
| **NDCG@K** | Position-discounted gain; our **promotion gate** metric. |
| **Coverage** | Fraction of the catalogue ever recommended — guards against popularity collapse and supports **equity** (are niche/under-used programs reachable?). |
| **Diversity** | Mean intra-list category dissimilarity — avoids 10 near-identical suggestions. |
| **Cold-start slice** | Same metrics on low-history users — the population most likely to fail silently. |

RMSE/accuracy-style metrics are intentionally excluded: we serve rankings, not rating predictions. Online, these are complemented by **A/B-tested engagement KPIs** (CTR, registration lift) owned with the division.

### Model Version Control
Every trained model is an **versioned artifact** with a metadata record capturing the lineage tuple `(version, git SHA, config, data fingerprint, metrics, timestamp, env)`. A registry index tracks all versions and the `latest` pointer; a **metric-based promotion gate** decides eligibility to advance to the next account. On AWS this maps to the **Model Registry** (registered models, version stages, approval transitions), backed by S3 + RDS.
See `src/model/registry.py` (and `src/model/load_model` for the serving-side counterpart used by the FastAPI app).

---
## Inferencing
### Caveats
* **Exclude already-engaged programs** so we recommend *next* actions, not past ones (configurable).
* **Respect validity windows** — never recommend an expired event or a program outside its registration window.
* **Stable, explainable slates** — batch slates change at most daily, avoiding a jarring citizen experience and easing support.
* **Eligibility & suppression** — honour any opt-out / suppression list before publishing (governance, Open Question 1).

### Batch (default serving mode)
Nightly, score **every eligible citizen**, produce a top-N ranked slate, run the **post-scoring quality gate**, and publish to Curated. Implemented as chunked scoring in `src/model/inference.py`; on AWS a **Kubernetes CronJob** running the pipeline image (`deploy/k8s/base/batch-cronjob.yaml`). This is the default because civic browsing tolerates daily freshness and batch is dramatically cheaper and simpler to govern than always-on services.

### Real-Time (FastAPI Service on EKS)
The **FastAPI** app (`src/serving/app.py`) loads the registered model and serves ranked slates over HTTP — `GET /recommend/{user_id}`, `POST /recommend` (batch), plus `/health`, `/ready`, `/metadata`, `/metrics`. It is deployed as a stateless, horizontally-scaled **Deployment** behind an **AWS ALB Ingress** with an **HPA**
(`deploy/k8s/`), and reuses the exact same `HybridRecommender` as training/batch (via `model.registry.load_model`), so online and offline behaviour cannot drift.

For **Phase 2** low-latency personalization at scale, the same service backs onto an **online feature store (Redis)** and a two-stage **candidate generation → re-rank** flow. The output contract is identical, so consumers are unaffected.

### Scoring Eligibility & Input Data  
* **Eligible users:** present in the trained user map (≥1 valid interaction), or routed to the cold-start path. Suppressed/opted-out citizens are excluded.
* **Eligible items:** in-catalogue programs with valid (non-expired) windows.
* **Input data:** the persisted model artifact + feature artefacts + current item catalogue. No raw PII is needed at scoring time beyond ids.

### Scoring Output
A tidy, versioned table — the **only** interface consumers use:

| column | meaning |
|--------|---------|
| `user_id` | citizen id |
| `item_id` | recommended program id |
| `rank` | 1..N position in the slate |
| `score` | model score (for ordering/diagnostics) |
| `category` | program category (for filtering/explainability) |
| `model_path` / `batch_id` | lineage back to the producing model & run |
| `generated_at` | freshness |

Written as partitioned Parquet (machine contract) + CSV preview (divisional), accompanied by a **run manifest** tying the batch to model version, metrics, and quality-gate outcomes.

---
## Data Quality & Logging
### Input Data Quality

| # | Check | Severity | Why | Failure Handling |
|---|-------|----------|-----|------------------|
| 1 | Schema validation | BLOCK | Downstream assumes the canonical schema | Abort (cannot repair a missing column) |
| 2 | Missing identifiers | BLOCK* | A recommendation needs citizen + program | Quarantine rows; abort if > threshold |
| 3 | Duplicate records | WARN | Duplicates over-weight a pair / bias confidence | Drop duplicates (keep first) |
| 4 | Invalid timestamps (null/non-positive/future) | WARN | Recency & temporal split depend on them; future = leakage | Quarantine offending rows |
| 5 | Invalid interaction values (out of range) | BLOCK* | Distorts confidence; high rate = upstream bug | Quarantine; abort if > threshold |

### Post-scoring Data Quality
Catches failure modes that appear only *after* inference:

| Check | Severity | Catches |
|-------|----------|---------|
| Scoring completeness | BLOCK | All eligible users scored |
| Recommendation reference integrity | BLOCK | No out-of-catalogue items recommended |
| No duplicate items within a slate | BLOCK | Degenerate/buggy ranking |
| Slate completeness | WARN | Users with fewer than top-N items |
| Catalogue coverage floor | WARN | Popularity collapse / equity regression |

### Logging
* **Structured JSON logs** (`src/utils/logger.py`) with level, source line, and timestamp — ship cleanly to **CloudWatch Logs** and are queryable.
* **Run manifest** per batch (model version, git SHA, data fingerprint, metrics, quality outcomes, promotion decision) — the auditable record of "what produced these recommendations".
* **Lineage tuple** on every model artifact for full traceability.

### Model monitoring
| Layer | What we watch | Mechanism | Action on breach |
|-------|---------------|-----------|------------------|
| **Data drift** | Distribution shift in interactions/features vs training | **Evidently** drift job (Argo CronWorkflow) → Prometheus/CloudWatch | Alarm → notification (Slack) → investigate / retrain |
| **Model quality (offline)** | NDCG/MAP/coverage per retrain vs baseline | Eval stage + registry gate | Block promotion; keep incumbent |
| **Model quality (online)** | CTR / registration lift vs control | A/B test + division KPIs | Roll back to previous model version |
| **Operational health** | Workflow/Job success, runtime, cost, data freshness | CloudWatch Container Insights + Argo metrics | Auto-retry; page on-call |
| **Service health (API)** | Latency, error rate, request volume, pod saturation | FastAPI `/metrics` → Prometheus/Grafana; ALB metrics | HPA scale-out; page on-call; rollback |

Runbooks for each alarm are in [`operational_scenarios.md`](./operational_scenarios.md).

---
## Multi-Account Deployment
### Dev/Staging/Prod Separation
| Account | Purpose | Data | Who |
|---------|---------|------|-----|
| **Dev** | Experimentation, feature/model development, pipeline build | Sample / masked / public (e.g. MovieLens) | Engineers (broad, sandboxed). |
| **Staging** | Full pipeline on **prod-like** governed data; integration & UAT | Prod-like (governed copy) | CI/CD + QA + division UAT. |
| **Prod** | Scheduled batch + FastAPI serving | Real Curated data | CI/CD / Argo CD only; no human write access. |  

Each account has its **own EKS cluster** (and ECR repos, KMS keys, Lake Formation governance). Hard account boundaries provide blast-radius isolation.

### Cross-Account Access
* A **shared model registry** + versioned artifacts in S3, accessed via **cross-account IAM assume-role** (pods use **IRSA**) with least privilege.
* **ECR cross-account pull** so each cluster runs the same approved image.
* **Lake Formation** grants cross-account data access at table/column/row granularity rather than bucket-wide IAM.
* KMS key policies permit only the specific roles/service accounts that need decrypt in each account.

### CI/CD strategy
* **GitHub Actions** (`.github/workflows/test-executor.yml`) runs unit + e2e + a pipeline smoke run on every PR/push as the first quality gate.
* Trunk-based development; main is the production branch (see README).
* Automated rollback: revert the manifest/image-tag commit (Argo CD restores the prior state) and/or re-point serving to the previous approved model version.

---
## Security & Governance (Risk Assessment)
| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| PII exposure / re-identification | Low | High | Pseudonymisation (de-duplications), KMS, Lake Formation, minimisation, governance review |
| Popularity bias / poor equity | Medium | Medium | Recency weighting, coverage/diversity monitoring, equity slices |
| Training-serving skew | Low | High | Shared feature code + persisted parity artefacts; offline (S3) / online (Redis) populated by one job |
| Data drift degrades quality | Medium | Medium | Evidently drift monitor + drift-triggered retrain + promotion gate |
| Bad model reaches Prod | Low | High | Metric promotion gate + manual approval + rollback |
| Upstream data breakage | Medium | High | Blocking quality gates + volume floor + alerting |
| Feedback loop bias | Medium | Medium | Log served slates; future IPS correction |
| Cost overrun | Low | Medium | Batch-first, spot nodes (Karpenter), scale-to-floor HPA, cost alarms |
| **EKS cluster / supply-chain compromise** | Low | High | Hardened nodes, non-root read-only pods, IRSA least-privilege, signed images, network policies, EKS patching |
| **Higher ops burden vs managed ML** (self-hosted MLflow/Argo/serving) | Medium | Medium | GitOps (Argo CD), IaC, runbooks, managed add-ons; reuse a platform-team EKS standard |

---

## Challenges and Risks
* **Implicit feedback & no true negatives** — modelling/eval treat unobserved as candidates; online A/B is the real arbiter.
* **Cold-start dominance in civic data** — Many citizens interact rarely; the content/popularity path and the cold-start metric slice are first-class, not afterthoughts.
* **Consent & privacy** — The gating risk for public-sector ML; surfaced as Open Questions.
* **Organisational** — Clear ownership split between platform team (pipeline) and division (relevance definition, KPIs, UAT sign-off). As the deployment of this solution is cross-team collaboration, we should be understanding and patient to learn from mistakes and be non-judgemental.

---
## User Acceptance Testing
**Functionality**  
The following are the testing criteria for functionality:  
- Returns the requested number of items, respects filters (e.g., exclude already-watched).
- Handles edge cases gracefully: new user with zero history, user with one rating, deleted/unavailable items in catalog.
- Latency/SLA within bounds.

**Business Rules / Guardrails**  
Define explicit pass/fail rules stakeholders agree on upfront:
- No duplicate or already-consumed items
- Diversity constraints (not 10 recommendations all from the same franchise)
- No banned/inappropriate content surfaced
- Freshness — not recommending the same list for weeks
- Popularity bias check — is it just recommending top-N globally popular items to everyone (a common recommender failure mode)

**Qualitative Relevance Review**  
Since we can't get one "correct answer," let's build a set of representative test personas/scenarios and have humans (PM, domain experts, or actual test users) judge the output:
- Cold-start user (no history)
- Power user with rich, varied history
- Niche-taste user (e.g., only watches foreign arthouse films)
- User with a recent behavior shift (binged a new genre)

For each, generate recommendations with explanations ("because you watched X") so reviewers can judge *why* something was surfaced, not just *what*. This explainability layer is often what makes qualitative UAT tractable, reviewers can sanity-check the reasoning chain even without a ground truth label.  

**Controlled Online Experiment**  
For recommenders, true "acceptance" usually can't be fully validated pre-launch — it has to be confirmed via A/B test or interleaving experiment, with predefined success metrics agreed before the test starts (CTR, watch-time, session length, return rate, diversity/coverage metrics). This is effectively our final UAT sign-off, just happening in production with traffic allocation rather than a staging environment.  

---

## Relevent Documentation
| Artifact | Location | Purpose |
|----------|----------|---------|
| Tool selection & justification | [`tool_selection.md`](./tool_selection.md) | Framework choice, alternatives, scaling, ops, cost, prod deployment. |
| Engineering decision log | [`decision_log.md`](./decision_log.md) | Decisions with alternatives & trade-offs. |
| Operational scenarios & runbooks | [`operational_scenarios.md`](./operational_scenarios.md) | Incidents, retraining, UAT, on-call. |
| Project README | [`../README.md`](../README.md) | Setup, how to run, assignment mapping, assumptions, time spent. |
| Config reference | [`../configs/README.md`](../configs/README.md) | Config schema & per-env promotion. |
| Inline code docs | `src/**` docstrings | Stage-level rationale next to the code. |
| Run manifests / reports | `outputs/**` | Per-run evidence (metrics, quality, lineage). |

Documentation is treated as a deliverable: it lives in-repo, is versioned with the code, and is cross-linked so a reviewer can navigate design, code and output.

---
## Appendix

### Glossary
* **Implicit feedback** — engagement inferred from behaviour (attendance), not explicit ratings.
* **Cold-start** — recommending for new users/items with little/no history.
* **NDCG@K** — Normalised Discounted Cumulative Gain; position-aware ranking quality.
* **Coverage** — share of the catalogue ever recommended.
* **Lake zones** — Raw → Standard → Curated → Features → Models progression.
* **Promotion gate** — automated metric check controlling Dev→Test→Prod advance.