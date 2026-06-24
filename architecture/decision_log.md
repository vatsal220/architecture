# Engineering Decision Log

This document outlines key technical decisions for the TDP recommendation solution. Each entry records the **decision**, **alternatives considered**, **selected option**, **justification**, and **trade-off**.

> Context for all decisions: a governed, multi-account AWS public-sector platform where engineering judgment, governability, operability, and total cost of ownership are valued over peak recommendation accuracy.

---
### Serving Mode: Batch vs Real-Time
| | |
|---|---|
| **Decision** | How are recommendations served? |
| **Alternatives** | (a) Nightly **batch** precompute; (b) **real-time** FastAPI service; (c) hybrid candidate-generation + live re-rank. |
| **Selected** | **Batch (a) as the default serving mode**, plus a **FastAPI real-time service (b)** included now, with (c) as the Phase 2 path. |
| **Justification** | Civic program browsing tolerates daily freshness, so batch (cheap, scales to ~zero, simple to govern) is the default. A lean FastAPI service is included so latency-sensitive surfaces can be served without re-architecting. Output contract is serving-mode-agnostic. |
| **Trade-off** | Batch alone has no within-day freshness; running the API adds a (small) always-on footprint. Accepted: HPA scales the API to a low floor and batch covers most demand. |

---

### Model Framework
| | |
|---|---|
| **Decision** | What model + framework for Phase 1? |
| **Alternatives** | Factorization Machines, LightFM, implicit (ALS), TF Recommenders, Surprise, XGBoost re-ranker, bespoke PyTorch. |
| **Selected** | **Hybrid matrix factorisation (`scikit-learn TruncatedSVD`) + recency-weighted popularity**, with **Factorization Machines / LightFM** (as container Jobs) as pre-planned upgrades. |
| **Justification** | Simplest correct, explainable, dependency-light baseline; zero extra ops; reproducible; establishes the metric bar any heavier model must beat. (Full reasoning in `tool_selection.md`.) |
| **Trade-off** | Lower accuracy ceiling than deep/hybrid models; SVD can't natively handle brand-new items (mitigated by the content/popularity path). |

---
### Feature Storage

| | |
|---|---|
| **Decision** | Where/how are features stored and reused? |
| **Alternatives** | (a) Plain S3 (Features zone) + Glue; (b) S3 offline + **Redis** online; (c) **Feast** feature store over S3+Redis; (d) bespoke feature DB. |
| **Selected** | **S3+Glue offline + ElastiCache Redis online**, with **Feast** as an optional unifying layer if feature reuse grows. |
| **Justification** | S3+Glue is cheap and point-in-time-correct for training/batch; Redis gives the low-latency online lookups the FastAPI service needs, all on portable OSS. One job populates both stores so offline/online parity is structural. Feast can be added later for a managed feature API without changing the model code. |
| **Trade-off** | We operate Redis (and possibly Feast) ourselves rather than using a managed feature store; the prototype enforces parity via shared code rather than a running online store. |

---

### Orchestration
| | |
|---|---|
| **Decision** | What runs and orchestrates the ETL + ML pipeline stages? |
| **Alternatives** | (a) **AWS Glue** (Spark) for all ETL; (b) **Argo Workflows on EKS** running container Jobs; (c) **Kubeflow Pipelines**; (d) **Step Functions + Lambda**; (e) EMR. |
| **Selected** | **Argo Workflows on EKS** for the ML pipeline DAG (each stage a container Job); **Glue** for generic lake ETL/cataloguing upstream. |
| **Justification** | Keeps ML code (Python/sklearn) in one portable runtime that is identical for jobs, batch, and serving; Argo gives a retriable DAG that mirrors our 7 stages 1:1 and reuses the platform's existing Kubernetes tooling. Kubeflow was heavier than needed; Step Functions would split logic across Lambda. Glue stays where it's strongest (Spark ETL, Catalog). |
| **Trade-off** | We operate Argo/EKS ourselves (vs a fully-managed orchestrator); justified by portability and reuse of an existing k8s platform. |

---
### Evaluation Metrics
| | |
|---|---|
| **Decision** | What metrics define "good", and which gates promotion? |
| **Alternatives** | RMSE/accuracy; Precision/Recall@K; MAP@K; **NDCG@K**; coverage/diversity; online CTR only. |
| **Selected** | **Ranking suite** (P/R/MAP/NDCG@K + coverage + diversity + cold-start slice); **NDCG@K is the promotion-gate metric**; online A/B KPIs own the final call. |
| **Justification** | Consumers receive an ordered slate, so ranking + position matter (NDCG). Coverage/diversity guard equity and popularity collapse. A single gate metric makes promotion automatable. |
| **Trade-off** | Offline NDCG is an imperfect proxy for real engagement; mitigated by treating online A/B as the source of truth. |

---
### Cold Start Strategy
| | |
|---|---|
| **Decision** | How to recommend for new citizens/programs? |
| **Alternatives** | Ignore cold users; popularity-only; **content-based**; full hybrid with side-feature model (LightFM/two-tower). |
| **Selected** | **Hybrid routing**: collaborative for users with history; **recency-weighted popularity blended with content (category) affinity** for cold users; content similarity for new items. |
| **Justification** | Civic data is cold-start-heavy; a usable, non-empty, explainable slate for newcomers matters more than marginal accuracy for power users. Content path handles brand-new programs SVD can't. |
| **Trade-off** | Cold-start recommendations are less personalised; richer side-feature models deferred to Phase 2. |

---
### Data Quality Design
| | |
|---|---|
| **Decision** | How do quality checks fail the pipeline? |
| **Alternatives** | (a) Hard-fail on any defect; (b) **severity-based** (BLOCK aborts, WARN repairs + continues, BLOCK-above-threshold); (c) log-only. |
| **Selected** | **Severity-based with thresholds** — schema/volume hard-block; ids/ratings block only above a small tolerance; duplicates/timestamps repaired with a warning. |
| **Justification** | Real data always has a small defect rate; hard-failing on every null is brittle and pages people needlessly. Catastrophic rates (upstream breakage) still block. All outcomes are persisted for audit. |
| **Trade-off** | Quarantining rows silently drops some data; bounded by thresholds and fully logged in the quality report. |

---

### Retraining Strategy
| | |
|---|---|
| **Decision** | When and how is the model retrained? |
| **Alternatives** | Manual; fixed daily; **scheduled + event + drift-triggered**; continuous/online learning. |
| **Selected** | **Weekly scheduled retrain + seasonal/event triggers + drift-triggered retrain**, each gated by the promotion check; features refreshed nightly for scoring. |
| **Justification** | Catalogue/demand shift slowly but seasonally; weekly balances freshness and cost, while drift triggers catch surprises. The gate guarantees a worse model never replaces the incumbent. |
| **Trade-off** | More moving parts than a fixed schedule; mitigated by automation and the safety of the promotion gate. |

---

### Multi-Account Deployment & Promotion
| | |
|---|---|
| **Decision** | How do code and models move Dev→Test→Prod? |
| **Alternatives** | Single account with prefixes; **3 accounts (Dev/Test/Prod)**; per-feature ephemeral accounts. |
| **Selected** | **3 hard-separated accounts**, each with its own **EKS cluster + ECR**; identical behaviour-defining config with per-account Kustomize overlays; **CodePipeline build/test + Argo CD (GitOps)** promotion with manual approval + model quality gate; Prod mutated only by Argo CD. |
| **Justification** | Account boundaries give blast-radius isolation and independent KMS/Lake Formation governance. Identical config + image digests make Test predictive of Prod. GitOps makes promotion/rollback a reviewable git operation; no human writes to Prod. |
| **Trade-off** | More infrastructure (per-account clusters) and cross-account IAM/ECR/Lake Formation plumbing; standard cost of a governed enterprise k8s platform. |

---
### Reproducibility
| | |
|---|---|
| **Decision** | How is a run made reproducible? |
| **Alternatives** | Hard-coded params; notebook-driven; **single YAML config + pinned deps + seeds + lineage manifest**. |
| **Selected** | **Config-as-code** (`configs/pipeline.yaml`) + fixed seeds + pinned `requirements.txt` + per-run manifest capturing `(config, git SHA, data fingerprint, metrics)`. |
| **Justification** | A run is reproducible from `(config + data + git SHA)` — essential for audit, debugging, and fair experiment comparison; the same config is the unit promoted across accounts. |
| **Trade-off** | Discipline overhead (everything must flow through config); pays for itself in auditability and promotion safety. |

---
### EKS + FastAPI + Open Source Solutions vs SageMaker
| | |
|---|---|
| **Decision** | What platform runs training, orchestration, registry, and serving? |
| **Alternatives** | (a) **Amazon SageMaker** (Processing/Training/Pipelines/Registry/Feature Store/Endpoints); (b) **EKS (Kubernetes)** with open-source tooling — Argo Workflows, FastAPI, MLflow, Optuna, Redis, Prometheus/Evidently; (c) ECS/Fargate + Step Functions. |
| **Selected** | **EKS + FastAPI + open-source MLOps (b)** — a deliberately **SageMaker-free** stack — while still using AWS for data/identity/governance (S3, Glue, Lake Formation, IAM/IRSA, KMS, ECR, CloudWatch). |
| **Justification** | Portability / no vendor lock-in; reuse of an existing governed EKS platform and the team's Kubernetes skills; one unified container runtime for jobs, batch, and the API; full control of the serving runtime (FastAPI). FastAPI specifically for async throughput, Pydantic validation, OpenAPI docs, and trivial health/metrics endpoints. |
| **Trade-off** | **Higher operational burden** — we own MLflow, Argo, the serving runtime, and EKS itself, versus SageMaker's managed convenience. Mitigated by GitOps (Argo CD), IaC, runbooks, and assuming the platform team already operates EKS. |