# -*- coding: utf-8 -*-
"""Feature Engineer
Stage ``03_generate_features``.

Produces three artefacts that constitute the Features zone:

1. ``interaction_matrix`` - a sparse user x item confidence matrix consumed by
   the matrix-factorisation model. Confidence follows the implicit-feedback
   formulation ``c = 1 + alpha * w`` where ``w`` blends the engagement rating
   with an exponential recency decay (recent engagement is weighted higher).
2. ``user_features`` - per-citizen aggregates (activity, recency, breadth,
   preferred category) used for cold-start blending and monitoring.
3. ``item_features`` - per-program aggregates + category one-hots used for the
   content-based cold-start path and for intra-list diversity in evaluation.

Training/serving consistency: features are computed by this single class from
the same Standard-zone inputs. The fitted artefacts (id maps, category vocab,
recency reference date) are persisted so batch inference and the FastAPI service
reuse *identical* transforms — this is the local stand-in for the offline/online
feature parity provided in production by S3 (offline) + ElastiCache Redis
(online) (see architecture/solution_architecture.md).
"""
from __future__ import annotations

import json
from typing import Dict

import numpy as np
import pandas as pd
from scipy import sparse

from data.process import build_id_maps
from utils import logger


class FeatureEngineer:
    """Builds and persists the Features-zone artefacts from training rows.

    A single instance owns the full fit → transform → persist flow. The fitted
    state (id maps, category vocabulary, recency reference timestamp) is held on
    the instance and written to disk so downstream scoring applies identical
    transforms.

    Usage::

        fe = FeatureEngineer(cfg)
        features = fe.build(train, items)
    """

    def __init__(self, cfg):
        from utils.config import resolve_path

        self.cfg = cfg
        fe = cfg.feature_engineering
        self.alpha = float(fe["implicit_confidence_alpha"])
        self.half_life = float(fe["recency_half_life_days"])
        self.features_dir = resolve_path(cfg, "features_dir")

        # Fitted state, populated by build().
        self.user_map: Dict = {}
        self.item_map: Dict = {}
        self.n_users: int = 0
        self.n_items: int = 0
        self.reference_ts: float = 0.0
        self.categories: list = []

    # -- public API ---------------------------------------------------------
    def build(self, train: pd.DataFrame, items: pd.DataFrame) -> Dict:
        """Fit id maps + recency reference and emit all Features-zone artefacts."""
        self.user_map, self.item_map = build_id_maps(train)
        self.n_users, self.n_items = len(self.user_map), len(self.item_map)
        self.reference_ts = float(train["timestamp"].max())

        matrix = self._build_interaction_matrix(train)
        item_content = self._build_item_content(items)
        user_features = self._build_user_features(train, items)
        item_features = self._build_item_features(train, items)
        artefacts = {
            "user_map": self.user_map,
            "item_map": self.item_map,
            "n_users": self.n_users,
            "n_items": self.n_items,
            "reference_ts": self.reference_ts,
            "categories": self.categories,
        }

        self._persist(matrix, item_content, user_features, item_features)
        logger.info(
            f"03_generate_features: matrix {matrix.shape} nnz={matrix.nnz}, "
            f"{len(self.categories)} categories -> {self.features_dir}"
        )
        return {
            "matrix": matrix,
            "item_content": item_content,
            "user_features": user_features,
            "item_features": item_features,
            "artefacts": artefacts,
        }

    def _recency_weight(self, timestamps: pd.Series) -> np.ndarray:
        """Exponential decay weight in (0, 1]; 1.0 at the reference date."""
        age_days = (self.reference_ts - timestamps.to_numpy()) / 86400.0
        age_days = np.clip(age_days, 0, None)
        return np.power(0.5, age_days / self.half_life)

    def _build_interaction_matrix(self, train: pd.DataFrame) -> sparse.csr_matrix:
        """Implicit-confidence matrix: blend normalised rating with recency."""
        recency = self._recency_weight(train["timestamp"])
        rating = train["rating"].to_numpy(dtype=float)
        rating_norm = rating / rating.max() if rating.max() > 0 else rating
        weight = 0.5 * rating_norm + 0.5 * recency
        confidence = 1.0 + self.alpha * weight

        rows = train["user_id"].map(self.user_map).to_numpy()
        cols = train["item_id"].map(self.item_map).to_numpy()
        return sparse.csr_matrix(
            (confidence, (rows, cols)), shape=(self.n_users, self.n_items)
        )

    def _build_item_content(self, items: pd.DataFrame) -> np.ndarray:
        """Category vocabulary + one-hot item content matrix (cold-start path)."""
        self.categories = sorted(items["category"].unique().tolist())
        cat_index = {c: i for i, c in enumerate(self.categories)}
        inv_item_map = {j: it for it, j in self.item_map.items()}
        cat_lookup = dict(zip(items["item_id"], items["category"]))

        item_content = np.zeros((self.n_items, len(self.categories)), dtype=float)
        for j in range(self.n_items):
            item_id = inv_item_map[j]
            item_content[j, cat_index[cat_lookup[item_id]]] = 1.0
        return item_content

    def _build_user_features(self, train: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
        """Per-citizen aggregates for cold-start blending and monitoring."""
        cat_lookup = dict(zip(items["item_id"], items["category"]))
        merged = train.assign(category=train["item_id"].map(cat_lookup))
        feats = merged.groupby("user_id").agg(
            n_interactions=("item_id", "size"),
            n_unique_items=("item_id", "nunique"),
            avg_rating=("rating", "mean"),
            last_ts=("timestamp", "max"),
            category_breadth=("category", "nunique"),
        ).reset_index()
        feats["days_since_last"] = (self.reference_ts - feats["last_ts"]) / 86400.0
        feats["recency_weight"] = np.power(0.5, feats["days_since_last"] / self.half_life)
        # Preferred category (mode) for content-based cold-start blending.
        pref = merged.groupby("user_id")["category"].agg(
            lambda s: s.value_counts().index[0]
        ).reset_index().rename(columns={"category": "preferred_category"})
        return feats.merge(pref, on="user_id", how="left")

    def _build_item_features(self, train: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
        """Per-program aggregates joined onto the full catalogue."""
        feats = train.groupby("item_id").agg(
            popularity=("user_id", "nunique"),
            n_interactions=("user_id", "size"),
            avg_rating=("rating", "mean"),
        ).reset_index()
        return items.merge(feats, on="item_id", how="left").fillna(
            {"popularity": 0, "n_interactions": 0, "avg_rating": 0.0}
        )

    def _persist(self, matrix: sparse.csr_matrix, item_content: np.ndarray,
                 user_features: pd.DataFrame, item_features: pd.DataFrame) -> None:
        """Write the Features zone so the stage is independently re-runnable."""
        self.features_dir.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(self.features_dir / "interaction_matrix.npz", matrix)
        np.save(self.features_dir / "item_content.npy", item_content)
        user_features.to_parquet(self.features_dir / "user_features.parquet", index=False)
        item_features.to_parquet(self.features_dir / "item_features.parquet", index=False)
        with open(self.features_dir / "feature_artefacts.json", "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "n_users": self.n_users,
                    "n_items": self.n_items,
                    "reference_ts": self.reference_ts,
                    "categories": self.categories,
                    "user_map": {str(k): v for k, v in self.user_map.items()},
                    "item_map": {str(k): v for k, v in self.item_map.items()},
                },
                fh,
            )

def build_features(cfg, train: pd.DataFrame, items: pd.DataFrame) -> Dict:
    """Entry point for stage ``03_generate_features``.

    Thin wrapper that delegates to :class:`FeatureEngineer`, returning the
    interaction matrix, content matrix, user/item feature tables and artefacts.

    Args:
        cfg: Dictionary containing the configuration
        train: DataFrame containing the training data
        items: DataFrame containing the item metadata

    Returns:
        Dict containing the interaction matrix, content matrix, user/item feature tables and artefacts
    """
    return FeatureEngineer(cfg).build(train, items)
