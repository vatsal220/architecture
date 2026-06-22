# Tool Selection & Justification

> This document will outline the reasoning behind why specific tools were selected. See also [`decision_log.md`](./decision_log.md) and [`solution_architecture.md`](./solution_architecture.md).

## TL;DR
The choice is **two-layered** — a *model/framework* layer and a *platform* layer:

* **Model framework:** a matrix-factorisation recommender built on **`scikit-learn` (`TruncatedSVD`)** plus a recency-weighted **popularity** baseline. Start with the simplest model that is correct, explainable, and cheap; grow only when measurements justify it.
* **Platform / deployment:** run **SageMaker-free** on **Amazon EKS (Kubernetes)**. Pipeline stages run as **containerised Jobs orchestrated by Argo Workflows**; the model is served by a **FastAPI** service (Deployment + ALB Ingress + HPA); model registry / experiment tracking is **MLflow**; HPO is **Optuna**; online features are **Redis**; monitoring is **Prometheus/Grafana + Evidently**. We still use AWS for data, identity, and governance (S3, Glue, Lake Formation, IAM, KMS, Athena, ECR, CloudWatch, CodePipeline).

The guiding principle: **run the model on portable, vendor-neutral building blocks, behind stable contracts, while inheriting the platform's data and governance services** — and only add complexity (deep models, real-time at scale) when a measurement demands it.

---
## Scikit-Learn (Model Framework)

| Reason | Detail |
|--------|--------|
| **Right complexity for the problem & data** | The interaction matrix is sparse with strong long-tail/popularity structure. Matrix factorisation (latent factors via truncated SVD) captures the dominant collaborative signal well; deeper models add cost and opacity without a proven accuracy need on civic data. |
| **Explainability for a public-sector stakeholder** | "Citizens with similar engagement also liked these programs" + "popular in your category" is defensible to a division and to governance. |
| **Zero extra operational surface** | `scikit-learn`/`scipy`/`numpy` are pure-wheel, ubiquitous, and run in any Python/Kubernetes container. No GPU, no native build, no bespoke serving runtime. |
| **Reproducibility & determinism** | Fixed `random_state` + pinned deps + config-as-code make every run reproducible — essential for audit and a fair experiment baseline. |
| **Fast iteration** | The full pipeline runs in ~2 seconds locally, so engineers iterate quickly and CI runs the real pipeline on every PR. |
| **Clean upgrade path** | The model sits behind a single `score_users` API and a stable output contract, so swapping in LightFM/implicit/a two-tower model later touches one module, not the platform. |

**Net:** scikit-learn is the **baseline any heavier framework must beat to earn its operational cost**. Establishing that baseline first is the disciplined choice.

---
## EKS + FastAPI (Platform Layer) over SageMaker

This is the central platform decision. SageMaker is a perfectly good managed ML platform; we deliberately **do not** use it here, in favour of self-hosted open-source components on EKS.

| Driver | Why EKS + FastAPI + OSS | What we give up vs SageMaker |
|--------|------------------------|------------------------------|
| **Portability / no vendor lock-in** | The whole stack (FastAPI, Argo, MLflow, Optuna, Evidently) is open-source and runs on any Kubernetes — on AWS today, elsewhere tomorrow. Public-sector procurement often values this. | SageMaker's turnkey managed services. |
| **Reuse of an existing platform capability** | Many enterprises already run a governed EKS platform with shared tooling (Argo CD, Prometheus, mesh). Recsys becomes "just another workload", reusing that investment and the team's k8s skills. | SageMaker-specific operational knowledge. |
| **Unified runtime for training + serving** | One container image, one cluster, one set of deployment primitives for jobs, batch CronJobs, and the API — instead of stitching Processing + Training + Pipelines + Endpoints. | Managed autoscaling/patching that SageMaker handles for you. |
| **Control & flexibility** | Full control of the serving runtime (custom routing, A/B, multi-model, request logging) in plain FastAPI; the registry/feature/monitoring layers are swappable. | Having to assemble + operate those layers ourselves. |
| **Cost shape** | Batch CronJobs + HPA-scaled API on spot/Karpenter nodes; scale the API to a small floor and batch to zero between runs. | SageMaker's per-second managed billing and zero cluster overhead. |

**The honest trade-off** is operational burden: we now own MLflow, Argo, and the serving runtime, plus EKS itself. We accept that because (a) it is assumed the platform team already operates EKS, (b) GitOps (Argo CD) + IaC + runbooks keep it manageable, and (c) portability/control outweigh turnkey convenience for this context. This trade-off is logged as **D13** in the decision log.

**Why FastAPI specifically** for serving: async, high-throughput ASGI; first-class **Pydantic** validation and **auto-generated OpenAPI** docs; trivial health/ready probes and a `/metrics` endpoint for Prometheus; tiny, pure-Python footprint that shares the project's dependency set and image.

---
## Investigating Alternative Model Frameworks

| Option | Why Not (for Phase 1) | When to Revisit |
|--------|----------------------|-------------------|
| **LightFM** | Excellent hybrid (implicit + user/item metadata, WARP loss) and a natural Phase 2 choice for richer cold-start. Deferred to avoid a native-build dependency and added tuning surface before the baseline proves the need. | Phase 2, when metadata-rich cold-start is the bottleneck. Runs as a K8s training Job. |
| **TensorFlow Recommenders** | Powerful for large-scale retrieval + rich features + real-time. Overkill for current data volume; heavy GPU/serving footprint and lower explainability. | At large scale or for Phase 2 real-time retrieval (served via FastAPI/KServe). |
| **Surprise** | Designed around **explicit** ratings; our signal is **implicit** engagement, so it's a conceptual mismatch. | Not planned. |
| **XGBoost** | Best as a re-ranker over candidate features, not a first-stage recommender. | Phase 2 learning-to-rank re-ranker. |
| **PyTorch** | Maximum flexibility, maximum maintenance. No justification before simpler models are exhausted. | Only if a custom architecture is proven necessary. |

---
## Scaling Limitations

| Limitation | Detail | Mitigation |
|------------|--------|------------|
| **In-memory matrix** | Truncated SVD on a single user×item matrix is bounded by node memory. | Fine for city scale (10⁵–10⁶ users × 10³–10⁴ programs sparse) on a single large pod. Beyond that: ALS/FM on distributed compute (e.g. Spark on EKS), or two-tower + ANN retrieval. |
| **Full-catalogue scoring** | Batch scores every user × every item (dense per user). | Acceptable for a small catalogue; for large catalogues use ANN/candidate generation (two-stage) served by FastAPI. |
| **API throughput/latency** | A single pod has finite RPS. | Stateless pods + **HPA** (CPU/RPS) + **Karpenter** node autoscaling scale horizontally; add a cache for hot users. |
| **Cold-start ceiling** | SVD alone can't place brand-new items. | Hybrid content path now; LightFM/two-tower with side features later. |
| **Retrain cost grows with history** | Full retrains get heavier over time. | Windowed history, incremental/warm-start, or move to FM/ALS with partial updates. |
| **Single-model, single-division** | One global model. | MLflow registry + the `score_users`/serving contract make multi-model / per-division scaling straightforward. |

---
## 5. Operational Considerations

* **Runtime:** containers on EKS — pipeline stages as Argo-orchestrated **K8s Jobs**, batch as a **CronJob**, serving as a **Deployment**. Node lifecycle via managed node groups / **Karpenter**; cluster + add-ons via IaC.
* **Scheduling:** **EventBridge / Argo CronWorkflow** (retrains) and a **CronJob** (nightly batch).
* **Observability:** structured logs → **CloudWatch Container Insights**; the API exposes `/metrics` to **Prometheus/Grafana**.
* **Failure handling:** idempotent stages, Argo retries, blocking quality gates, and a promotion gate that prevents bad models from advancing.
* **Rollback:** **GitOps** — revert the manifest/image-tag commit and Argo CD restores prior state; serving can also re-point to the previous MLflow model version.
* **Dependencies:** small, pure-wheel, pinned; identical on laptop, CI, and the EKS images — eliminating "works on my machine".
* **Team fit:** plain Python + Kubernetes — broadly maintainable and a good fit for a platform team that already runs EKS.

---
## 6. Cost Considerations
| Driver | Phase 1 (Batch + Lean API on EKS) | Real-Time |
|--------|-----------------------------------|-----------|
| Compute | Short-lived CPU **K8s Jobs/CronJobs** (minutes) on **spot/Karpenter**; API at a small replica floor | Always-on GPU serving (24×7) + GPU training |
| Storage | S3 lake + small artifacts (KMS) + tiny RDS/Redis for MLflow/online | + larger online store + heavier serving infra |
| Serving | A few small FastAPI pods (HPA scales with traffic, down off-peak) | GPU endpoint hosting + autoscaling |
| Ops/people | EKS already operated by the platform team; modest add-on footprint | Higher on-call burden, latency SLAs |
| **Posture** | **Batch scales to ~zero; API scales to a small floor** | Pay-to-idle; only worth it with a proven latency need |

Batch-first on shared EKS is the **cost-responsible default for a public service**: we pay for compute mainly when the nightly job runs, and the API floor is a few small pods.

---
## 7. Production Deployment Considerations

* **Packaging:** two images in **ECR** — pipeline (`Docker/Dockerfile`) and serving (`Docker/Dockerfile.serving`) — the units of promotion. Identical artifact from laptop → CI → EKS; only args + the pod's IAM role (IRSA) differ.
* **Promotion:** images + Kustomize overlays promoted Dev→Test→Prod via CodePipeline build/test and **Argo CD** sync; models promoted through the **MLflow Model Registry** behind a **metric quality gate** + manual approval (`src/model/registry.py::promotion_gate`).
* **Train/serve parity:** the FastAPI service rebuilds the *same* `HybridRecommender` via `model.registry.load_model`; features come from one job that populates both offline (S3) and online (Redis) stores — no skew.
* **Governance baked in:** runs within Lake Formation + KMS + IAM/**IRSA**, with hardened pods, network policies, and an auditable manifest per batch.
* **Designed-for evolution:** the `score_users` API and stable output contract mean we can upgrade the algorithm (FM, LightFM, two-tower) or scale the FastAPI service into a two-stage real-time recommender **without touching consumers** — the architecture, not the model, is the durable asset.