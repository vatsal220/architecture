# -*- coding: utf-8 -*-
"""Model Registry
Local stand-in for the **Models** S3 zone + an MLflow Model Registry (the
self-hosted, SageMaker-free registry the platform runs on EKS).

Responsibilities (Model Version Control, see solution_architecture.md):
* persist the fitted model factors as a versioned, immutable artefact;
* write a ``metadata.json`` capturing the lineage tuple
  ``(version, git SHA, config, data fingerprint, metrics, timestamp)`` so any
  prediction can be traced back to exactly what produced it;
* maintain a ``registry.json`` index and a ``latest`` pointer;
* expose a metric-based promotion gate (the same gate CodePipeline would call
  before promoting a model Dev -> Test -> Prod).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import datetime as dt
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy import sparse

from utils import logger


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # pragma: no cover - git optional
        return "nogit"


def _data_fingerprint(matrix: sparse.csr_matrix) -> str:
    h = hashlib.sha256()
    h.update(matrix.indices.tobytes())
    h.update(matrix.indptr.tobytes())
    h.update(np.round(matrix.data, 4).tobytes())
    return h.hexdigest()[:16]


def make_version() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def save_model(cfg, model_bundle: Dict, features: Dict, metrics: Optional[Dict],
               version: Optional[str] = None) -> Dict:
    """Persist artefacts + metadata, update the registry index, return metadata."""
    from utils.config import resolve_path

    version = version or make_version()
    models_dir = resolve_path(cfg, "models_dir")
    artefact_dir = models_dir / version
    artefact_dir.mkdir(parents=True, exist_ok=True)

    model = model_bundle["model"]
    np.savez_compressed(
        artefact_dir / "model.npz",
        user_factors=model.svd.user_factors,
        item_factors=model.svd.item_factors,
        popularity=model.pop.item_scores,
        user_history_counts=model.user_history_counts,
        item_content=model.item_content if model.item_content is not None else np.empty(0),
    )

    artefacts = features["artefacts"]
    with open(artefact_dir / "id_maps.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "user_map": {str(k): v for k, v in artefacts["user_map"].items()},
                "item_map": {str(k): v for k, v in artefacts["item_map"].items()},
                "categories": artefacts["categories"],
            },
            fh,
        )

    metadata = {
        "version": version,
        "git_sha": _git_sha(),
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "env": cfg.get_path("env", "local"),
        "model_config": model_bundle["model_config"].to_dict(),
        "data_fingerprint": _data_fingerprint(features["matrix"]),
        "n_users": artefacts["n_users"],
        "n_items": artefacts["n_items"],
        "metrics": metrics or {},
        "artefact_path": str(artefact_dir),
    }
    with open(artefact_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    _update_index(models_dir, metadata)
    logger.info(f"registry: saved model version {version} -> {artefact_dir}")
    return metadata


def _update_index(models_dir: Path, metadata: Dict) -> None:
    index_path = models_dir / "registry.json"
    index = {"models": [], "latest": None}
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as fh:
            index = json.load(fh)
    index["models"].append(
        {
            "version": metadata["version"],
            "git_sha": metadata["git_sha"],
            "created_at": metadata["created_at"],
            "metrics": metadata["metrics"],
        }
    )
    index["latest"] = metadata["version"]
    with open(index_path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)


def _resolve_version(models_dir: Path, version: Optional[str]) -> str:
    """Return the requested version, or the registry's ``latest`` pointer."""
    if version:
        return version
    index_path = models_dir / "registry.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"No model registry at {index_path}; run the pipeline first."
        )
    with open(index_path, "r", encoding="utf-8") as fh:
        index = json.load(fh)
    latest = index.get("latest")
    if not latest:
        raise ValueError("Registry has no 'latest' model version")
    return latest


def load_model(cfg, version: Optional[str] = None) -> Dict:
    """Rebuild a served model bundle from persisted registry artifacts.

    This is the serving-side counterpart to :func:`save_model`. It reconstructs
    the same :class:`HybridRecommender` (plus id maps, categories and metadata)
    used by training/batch, so the FastAPI service and the batch job share one
    scoring implementation and cannot drift apart.
    """
    from utils.config import resolve_path
    # Imported lazily to avoid a heavy import (sklearn) at module load time.
    from model.train import HybridRecommender, SVDRecommender, PopularityRecommender

    models_dir = resolve_path(cfg, "models_dir")
    version = _resolve_version(models_dir, version)
    artefact_dir = models_dir / version
    if not artefact_dir.exists():
        raise FileNotFoundError(f"Model version not found: {artefact_dir}")

    with np.load(artefact_dir / "model.npz", allow_pickle=False) as data:
        user_factors = data["user_factors"]
        item_factors = data["item_factors"]
        popularity = data["popularity"]
        user_history_counts = data["user_history_counts"]
        item_content = data["item_content"]
    item_content = item_content if item_content.size else None

    with open(artefact_dir / "id_maps.json", "r", encoding="utf-8") as fh:
        maps = json.load(fh)
    with open(artefact_dir / "metadata.json", "r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    user_map = {int(k): v for k, v in maps["user_map"].items()}
    item_map = {int(k): v for k, v in maps["item_map"].items()}
    cold_min = int(metadata["model_config"]["cold_start_min_interactions"])

    model = HybridRecommender(
        svd=SVDRecommender(user_factors=user_factors, item_factors=item_factors),
        pop=PopularityRecommender(item_scores=popularity),
        user_history_counts=user_history_counts,
        cold_start_min=cold_min,
        item_content=item_content,
        user_pref_vector=None,
    )
    logger.info(f"registry.load_model: loaded version {version} ({len(user_map)} users)")
    return {
        "model": model,
        "version": version,
        "user_map": user_map,
        "item_map": item_map,
        "categories": maps.get("categories", []),
        "metadata": metadata,
    }


def promotion_gate(cfg, metadata: Dict) -> Dict:
    """Decide whether a model is eligible to promote to the next account.

    The same metric gate the CI/CD promotion job would enforce: a model below
    the configured floor on the promotion metric is blocked. Returns a decision
    record (logged + persisted) rather than raising, so CodePipeline can branch.
    """
    reg = cfg.registry
    metric_name = reg["promote_metric"]
    floor = float(reg["min_promote_value"])
    metric_key = f"{metric_name}_at_k"
    value = metadata.get("metrics", {}).get(metric_key)
    promote = value is not None and value >= floor
    decision = {
        "promote": bool(promote),
        "metric": metric_name,
        "value": value,
        "floor": floor,
        "reason": (
            f"{metric_key}={value} >= floor {floor}" if promote
            else f"{metric_key}={value} below floor {floor}"
        ),
    }
    level = logger.info if promote else logger.warning
    level(f"registry.promotion_gate: {decision['reason']} -> promote={promote}")
    return decision
