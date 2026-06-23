# Configuration

`pipeline.yaml` is the single, version-controlled source of truth for the
end-to-end recommendation pipeline. A run is reproducible from the tuple
`(pipeline.yaml + dataset + git SHA)`.

## Per-environment promotion

In the target AWS multi-account deployment the **same** `pipeline.yaml` is
promoted Dev → Test → Prod. Only a small, account-specific overlay changes:

| Block     | Local            | Dev / Test / Prod                          |
| --------- | ---------------- | ------------------------------------------ |
| `env`     | `local`          | `dev` / `test` / `prod`                    |
| `paths.*` | `./data`,`./out` | `s3://tdp-<zone>-<acct>/recsys/...`         |
| `aws.*`   | *(absent)*       | account id, KMS key arn, Glue DB, role arn |

Keeping behaviour-defining parameters (model, split, thresholds) identical
across accounts is what makes Test a faithful predictor of Prod. Account
overlays are injected by CodePipeline at deploy time and never hand-edited in
Prod (see `architecture/operational_scenarios.md`).

## Overriding at runtime

```bash
python run_pipeline.py --config configs/pipeline.yaml          # full run
python run_pipeline.py --config configs/pipeline.yaml --stage 04_train_model
python run_pipeline.py --set model.svd_mf.n_factors=128        # ad-hoc override
```
