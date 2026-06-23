# -*- coding: utf-8 -*-
"""End-to-end smoke test.

Runs the full seven-stage pipeline on a small, fast MovieLens-format fixture in
an isolated temp directory and asserts the contract of the final outputs:
recommendations exist, the manifest is well-formed, and metrics are present.
This is the regression guard for "does `python run_pipeline.py` still work".
"""
from pathlib import Path

import run_pipeline


def test_pipeline_end_to_end(pipeline_config):
    cfg_path = pipeline_config(n_users=300, n_items=60, n_interactions=6000, n_factors=16)
    result = run_pipeline.run(str(cfg_path), overrides=[], only_stage=None)

    assert result["batch_id"]
    metrics = result["metrics"]
    for key in ("precision_at_k", "recall_at_k", "map_at_k", "ndcg_at_k",
                "coverage", "diversity"):
        assert key in metrics
        assert metrics[key] >= 0.0

    manifest = result["manifest"]
    assert manifest["n_recommendation_rows"] > 0
    assert manifest["input_quality_passed"] is True
    assert manifest["post_scoring_quality_passed"] is True

    recs = Path(manifest["artifacts"]["recommendations_csv"])
    assert recs.exists()
    assert "promotion_decision" in manifest
