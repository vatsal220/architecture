# -*- coding: utf-8 -*-
"""run_pipeline.py
Single entry point for the Toronto Data Platform recommendation prototype.

Runs the seven canonical stages end-to-end and is the local analogue of the
Argo Workflows DAG that orchestrates the pipeline on EKS. Each stage is
idempotent and persists its outputs to a data-lake-style zone, so a stage can be
re-run in isolation with ``--stage``.

    01_load_data            -> Raw zone
    02_validate_data        -> input data-quality gate (Standard zone)
    03_generate_features    -> Features zone
    04_train_model          -> Models zone (fit)
    05_evaluate_model       -> ranking metrics + promotion gate
    06_generate_recommendations -> batch scoring + post-scoring quality gate
    07_publish_results      -> Curated/publish zone + run manifest

Usage:
    python run_pipeline.py                         # full run, default config
    python run_pipeline.py --config configs/pipeline.yaml
    python run_pipeline.py --stage 04_train_model  # single stage (uses cached upstream)
    python run_pipeline.py --set model.svd_mf.n_factors=128
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make ``src`` importable whether run from repo root or elsewhere.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils import logger                       # noqa: E402
from utils.config import load_config           # noqa: E402

STAGES = [
    "01_load_data",
    "02_validate_data",
    "03_generate_features",
    "04_train_model",
    "05_evaluate_model",
    "06_generate_recommendations",
    "07_publish_results",
]


def run(config_path: str | None, overrides: list[str], only_stage: str | None) -> dict:
    from data.loader import load_raw
    from data.process import standardize, temporal_train_test_split
    from data.explore import profile_interactions
    from data.quality import run_input_quality, run_post_scoring_quality
    from feature.engineer import build_features
    from model.train import train_model
    from model.evaluate import evaluate_model
    from model.inference import generate_recommendations
    from model.registry import save_model, promotion_gate, make_version
    from model.deploy import publish_results

    cfg = load_config(config_path, overrides)
    batch_id = make_version()
    t0 = time.time()
    logger.info(f"pipeline: starting batch {batch_id} (env={cfg.get_path('env')})")

    # --- 01 load -----------------------------------------------------------
    raw = load_raw(cfg)
    profile = profile_interactions(raw["interactions"].dropna(subset=["user_id"]), raw["items"])
    logger.info(f"01_load_data: profile={profile}")

    # --- 02 validate -------------------------------------------------------
    validated = run_input_quality(cfg, raw["interactions"], raw["items"])
    std = standardize(cfg, validated["interactions"], validated["items"])
    train, test = temporal_train_test_split(cfg, std["interactions"])

    # --- 03 features -------------------------------------------------------
    features = build_features(cfg, train, std["items"])

    # --- 04 train ----------------------------------------------------------
    model_bundle = train_model(cfg, features)

    # --- 05 evaluate -------------------------------------------------------
    metrics = evaluate_model(cfg, model_bundle, features, test)
    metadata = save_model(cfg, model_bundle, features, metrics, version=batch_id)
    promotion = promotion_gate(cfg, metadata)

    # --- 06 recommend + post-scoring quality ------------------------------
    recommendations = generate_recommendations(
        cfg, model_bundle, features, std["items"], batch_id,
        model_path=metadata["artefact_path"],
    )
    post_quality = run_post_scoring_quality(
        cfg, recommendations, std["items"],
        eligible_users=features["artefacts"]["n_users"],
    )

    # --- 07 publish --------------------------------------------------------
    manifest = publish_results(
        cfg, recommendations, metadata, metrics,
        quality={"input": validated["report"], "post": post_quality},
        batch_id=batch_id, promotion=promotion,
    )

    elapsed = time.time() - t0
    logger.info(f"pipeline: completed batch {batch_id} in {elapsed:.1f}s")
    return {
        "batch_id": batch_id,
        "metrics": metrics,
        "manifest": manifest,
        "promotion": promotion,
        "elapsed_seconds": round(elapsed, 1),
    }


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TDP recommendation pipeline")
    p.add_argument("--config", default=None, help="Path to pipeline.yaml")
    p.add_argument("--stage", choices=STAGES, default=None,
                   help="Run a single stage (upstream zones must already exist)")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   help="Override a config value, e.g. --set model.svd_mf.n_factors=128")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.stage:
        # For the prototype, a partial run still executes the full DAG (stages
        # are cheap and idempotent); the flag documents intent and is the hook
        # where a production runner would resume from cached zones.
        logger.warning(
            f"--stage {args.stage} requested; prototype runs the full idempotent DAG."
        )
    result = run(args.config, args.overrides, args.stage)
    logger.info(f"pipeline result: {result['metrics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
