# Operational Scenarios & Runbooks

The following document will act as the production operating guide for the TDP recommendation pipeline: incident runbooks, retraining procedure, rollback, on-call playbooks, and the UAT checklist. Companion to [`solution_architecture.md`](./solution_architecture.md).

---
## 1. Service Overview
| Property | Value |
|----------|-------|
| Platform | Amazon EKS (Argo Workflows for the DAG; CronJob for batch; FastAPI Deployment for serving) |
| Trigger | EventBridge / Argo CronWorkflow (weekly retrain) + CronJob (nightly scoring) |
| Stages | `01_load → 02_validate → 03_features → 04_train → 05_evaluate → register → 06_score → 07_publish` |
| Primary output | Curated `recommendations` table + per-batch manifest; FastAPI service for real-time |
| Health signals | CloudWatch Container Insights, Prometheus/Grafana, FastAPI `/metrics`, run manifest, quality reports |
| Paging | CloudWatch / Prometheus alarm → SNS → on-call |
| Rollback unit | Previous approved model version in the MLflow registry; previous manifest/image via Argo CD |

---
## 2. Incident Management System

1. **Detect**: CloudWatch alarm (job failure, runtime/cost spike, drift, quality-gate block, freshness breach) → SNS → on-call.
2. **Triage**: classify severity:
   * **SEV-1**: no fresh recommendations reaching consumers, or a data/privacy incident.
   * **SEV-2**: pipeline failing but last-good output still served.
   * **SEV-3**: quality degradation within tolerance / single WARN.
3. **Contain**: for model/output issues, **roll back** to the last approved model + last-good published batch (consumers keep reading the previous Curated partition). For data-exposure issues, revoke access via Lake Formation/IAM and engage governance immediately.
4. **Diagnose**: use the run manifest + structured logs + lineage tuple to reproduce locally from `(config + data snapshot + git SHA)`.
5. **Fix & verify**: patch, run pipeline in **Test**, confirm gates pass.
6. **Promote**: through CodePipeline with approval.
7. **Post-incident review**: blameless; capture root cause, add a regression test and/or a new quality check, update this runbook.

---
## 3. Runbooks by Alarm

### 3.1 Pipeline stage failed
* **Symptom:** Argo Workflow step `Failed`/`Error` (pod non-zero exit).
* **Check:** `argo logs <workflow>` / CloudWatch logs for the step's pod; the
  data-quality report if it's stage 02/06.
* **Act:** transient (spot reclaim/throttling) → Argo retry already attempted; retry the node (`argo retry`, idempotent). Persistent → reproduce in Test, fix, promote.
* **Guardrail:** consumers continue serving the previous batch; no rollback needed unless a bad batch was already published.

### 3.2 Input Data-Quality BLOCK
* **Symptom:** `02_validate_data` aborted (e.g. null-id fraction over threshold, schema drift, volume floor).
* **Check:** `outputs/reports/data_quality_input.json` (which rule, fraction).
* **Act:** this is the gate doing its job — **do not bypass**. Engage the upstream data owner; the most common cause is a broken/partial extract. Re-run after the source is fixed.

### 3.3 Post-Scoring Quality BLOCK
* **Symptom:** out-of-catalogue items, duplicate items in a slate, or not all eligible users scored.
* **Check:** `data_quality_post_scoring.json`.
* **Act:** block publish (automatic); previous batch stays live. Reproduce, fix inference/feature bug, re-run.

### 3.4 Data Drift
* **Symptom:** Model Monitor flags input drift, or online CTR/registration lift drops vs control.
* **Act:** trigger an off-cycle retrain; if drift is severe, consider reverting to a known-good model while investigating. Validate with A/B before full rollout.

### 3.5 Freshness / SLA Breach
* **Symptom:** recommendations older than 24h (NFR1).
* **Act:** confirm the schedule fired; check for a silently-stuck job; re-run. Communicate to the division if output is stale.

---
## 4. Retraining Procedure
1. **Triggered** by schedule (weekly), season/event, or drift.
2. Pipeline runs end-to-end in Test on prod-like data.
3. `05_evaluate_model` computes the metric suite; `registry.promotion_gate` compares against the floor (and the incumbent).
4. **Pass** → register new version in MLflow, request approval, bump the Prod overlay (Argo CD syncs); the FastAPI pods pick up the new `latest` (rolling restart) and the next batch uses it.
5. **Fail** → keep incumbent, alert on-call, open an investigation.
6. Every retrain is recorded with full lineage (config, git SHA, data fingerprint, metrics) for audit and comparison.

---
## 5. Rollback Procedure
1. Identify the last **approved** model version in the model registry (and the last-good image tag).
2. Revert the offending commit (model version pin and/or image tag) in the environment overlay; **Argo CD** syncs the cluster back to the prior state — no Prod hand-edits. The FastAPI Deployment rolls back automatically.
3. Consumers reading the Curated `recommendations` table are unaffected during the switch; if a bad batch was published, re-publish from the last-good model.
4. Verify health signals green; record the rollback in the incident log.

Rollback is intentionally **fast and boring**: every model version and image is immutable and retained, and rollback is a git revert.

---
## 6. On-call Playbook (Quick Reference)

| Situation | First action | Citizen impact? |
|-----------|--------------|-----------------|
| Stage failed | Re-run idempotent step | No (previous batch served) |
| Input DQ block | Call upstream owner; don't bypass | No |
| Promotion blocked | None — incumbent retained | No |
| Post-scoring block | Investigate; previous batch served | No |
| Bad batch already live | **Rollback** + re-publish | Yes → act now (SEV-1/2) |
| API degraded / crashloop | Check `/ready`+pods; HPA scale; Argo CD rollback | Maybe (batch table still served) → SEV-2 |
| Drift / KPI drop | Off-cycle retrain; maybe revert | Possibly → SEV-2 |
| Data/privacy exposure | Revoke access; engage governance | Yes → SEV-1 |
