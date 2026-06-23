# -*- coding: utf-8 -*-
"""Model Deploy
Stage ``07_publish_results``.

In the prototype, "publishing" writes the recommendation slates and a run
manifest to the local ``publish`` zone in both partitioned Parquet (the machine
contract) and CSV (human/divisional preview). On the target platform this stage
is the boundary that promotes outputs into the **Curated** zone of the data
lake, registers/updates the Glue table, and (optionally) loads an online store
for low-latency serving — see architecture/solution_architecture.md
("Recommendation Serving" / "Multi-Account Deployment").

The manifest is the auditable record tying a published batch to the exact model
version, config, metrics, and quality-gate outcomes that produced it.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Dict

import pandas as pd

from utils import logger


def publish_results(cfg, recommendations: pd.DataFrame, metadata: Dict,
                    metrics: Dict, quality: Dict, batch_id: str,
                    promotion: Dict) -> Dict:
    """Write recommendations + manifest to the publish zone; return the manifest."""
    from utils.config import resolve_path

    publish_dir = resolve_path(cfg, "publish_dir")
    recs_dir = resolve_path(cfg, "recommendations_dir")
    publish_dir.mkdir(parents=True, exist_ok=True)
    recs_dir.mkdir(parents=True, exist_ok=True)

    # Partitioned parquet (curated contract) + flat csv (divisional preview).
    parquet_path = recs_dir / f"recommendations_{batch_id}.parquet"
    csv_path = publish_dir / f"recommendations_{batch_id}.csv"
    recommendations.to_parquet(parquet_path, index=False)
    recommendations.sort_values(["user_id", "rank"]).to_csv(csv_path, index=False)

    manifest = {
        "batch_id": batch_id,
        "published_at": dt.datetime.now().isoformat(timespec="seconds"),
        "env": cfg.get_path("env", "local"),
        "model_version": metadata.get("version"),
        "git_sha": metadata.get("git_sha"),
        "data_fingerprint": metadata.get("data_fingerprint"),
        "n_users_scored": int(recommendations["user_id"].nunique()),
        "n_recommendation_rows": int(len(recommendations)),
        "metrics": metrics,
        "input_quality_passed": quality.get("input", {}).get("passed"),
        "post_scoring_quality_passed": quality.get("post", {}).get("passed"),
        "promotion_decision": promotion,
        "artifacts": {
            "recommendations_parquet": str(parquet_path),
            "recommendations_csv": str(csv_path),
            "model_artifact": metadata.get("artefact_path"),
        },
    }
    manifest_path = publish_dir / f"manifest_{batch_id}.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    # Stable pointer to the most recent successful publish.
    with open(publish_dir / "latest_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    logger.info(
        f"07_publish_results: published batch {batch_id} "
        f"({manifest['n_recommendation_rows']} rows) -> {publish_dir}"
    )
    return manifest
