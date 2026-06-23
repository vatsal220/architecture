# -*- coding: utf-8 -*-
"""Model Train
Stage ``04_train_model``.

Implements a **hybrid recommender** that is deliberately simple, dependency-light
and explainable — the right complexity tier for a first production rollout on a
governed platform (see architecture/tool_selection.md for why scikit-learn over
LightFM / a deep two-tower model at this stage):

* ``SVDRecommender`` - matrix factorisation via ``sklearn.TruncatedSVD`` on the
  implicit-confidence user x item matrix. Produces dense user and item latent
  factors; the score for (u, i) is the dot product of their factors. Captures
  collaborative signal for users with history.
* ``PopularityRecommender`` - recency-weighted global popularity. Serves the
  cold-start path (new citizens / new programs) and is the safety fallback.
* ``HybridRecommender`` - routes each user to SVD or popularity based on how
  much history they have, and exposes a single ``score_users`` API so inference
  and evaluation never branch on model internals.

The fitted object is plain numpy and is persisted by ``model.registry``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from model.model_cfg import ModelConfig
from utils import logger

@dataclass
class PopularityRecommender:
    item_scores: np.ndarray  # shape (n_items,)

    @classmethod
    def fit(cls, matrix: sparse.csr_matrix) -> "PopularityRecommender":
        # Sum of confidence per item = recency-weighted popularity.
        scores = np.asarray(matrix.sum(axis=0)).ravel()
        if scores.max() > 0:
            scores = scores / scores.max()
        return cls(item_scores=scores)

    def score_all(self) -> np.ndarray:
        return self.item_scores

@dataclass
class SVDRecommender:
    user_factors: np.ndarray  # (n_users, k)
    item_factors: np.ndarray  # (n_items, k)

    @classmethod
    def fit(cls, matrix: sparse.csr_matrix, n_factors: int, random_state: int) -> "SVDRecommender":
        k = min(n_factors, min(matrix.shape) - 1)
        if k < 2:
            raise ValueError("Matrix too small for SVD; need >=3 users and items")
        svd = TruncatedSVD(n_components=k, random_state=random_state)
        user_factors = svd.fit_transform(matrix)          # (n_users, k)
        item_factors = svd.components_.T                  # (n_items, k)
        logger.info(
            f"04_train_model: TruncatedSVD k={k} "
            f"explained_variance_ratio_sum={svd.explained_variance_ratio_.sum():.4f}"
        )
        return cls(user_factors=user_factors, item_factors=item_factors)

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        return self.user_factors[user_indices] @ self.item_factors.T


class HybridRecommender:
    """Routes between collaborative (SVD) and popularity scoring per user."""

    def __init__(self, svd: SVDRecommender, pop: PopularityRecommender,
                 user_history_counts: np.ndarray, cold_start_min: int,
                 item_content: Optional[np.ndarray] = None,
                 user_pref_vector: Optional[np.ndarray] = None):
        """Initialize the HybridRecommender.

        Args:
            svd: SVDRecommender object
            pop: PopularityRecommender object
            user_history_counts: numpy array containing the number of interactions per user
            cold_start_min: int containing the minimum number of interactions for a user to be considered a cold start
            item_content: numpy array containing the content of the items
            user_pref_vector: numpy array containing the preferred category of the users
        """
        self.svd = svd
        self.pop = pop
        self.user_history_counts = user_history_counts
        self.cold_start_min = cold_start_min
        self.item_content = item_content
        self.user_pref_vector = user_pref_vector

    @classmethod
    def fit(cls, matrix: sparse.csr_matrix, mcfg: ModelConfig,
            item_content: Optional[np.ndarray] = None,
            user_pref_vector: Optional[np.ndarray] = None) -> "HybridRecommender":
        svd = SVDRecommender.fit(matrix, mcfg.n_factors, mcfg.random_state)
        pop = PopularityRecommender.fit(matrix)
        history_counts = np.asarray((matrix > 0).sum(axis=1)).ravel()
        logger.info(
            f"04_train_model: hybrid fitted "
            f"(cold_start_min={mcfg.cold_start_min_interactions})"
        )
        return cls(svd, pop, history_counts, mcfg.cold_start_min_interactions,
                   item_content, user_pref_vector)

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        """Return a (len(user_indices), n_items) dense score matrix."""
        scores = self.svd.score_users(user_indices)
        pop_scores = self.pop.score_all()

        cold_mask = self.user_history_counts[user_indices] < self.cold_start_min
        if cold_mask.any():
            # Cold users: popularity prior, blended with content affinity if known.
            blend = np.tile(pop_scores, (cold_mask.sum(), 1))
            if self.item_content is not None and self.user_pref_vector is not None:
                pref = self.user_pref_vector[user_indices[cold_mask]]
                content_aff = pref @ self.item_content.T
                content_aff = _row_minmax(content_aff)
                blend = 0.5 * blend + 0.5 * content_aff
            scores[cold_mask] = blend
        return scores


def _row_minmax(a: np.ndarray) -> np.ndarray:
    lo = a.min(axis=1, keepdims=True)
    hi = a.max(axis=1, keepdims=True)
    rng = np.where((hi - lo) == 0, 1.0, hi - lo)
    return (a - lo) / rng


def _user_pref_vector(user_features, user_map, categories) -> np.ndarray:
    """One-hot of each user's preferred category, aligned to the user index."""
    cat_index = {c: i for i, c in enumerate(categories)}
    vec = np.zeros((len(user_map), len(categories)), dtype=float)
    for _, row in user_features.iterrows():
        uidx = user_map.get(int(row["user_id"]))
        if uidx is None:
            continue
        pref = row.get("preferred_category")
        if pref in cat_index:
            vec[uidx, cat_index[pref]] = 1.0
    return vec


def train_model(cfg, features: Dict) -> Dict:
    """Stage entry point. Returns the fitted model + the config it was trained with."""
    mcfg = ModelConfig.from_cfg(cfg)
    matrix = features["matrix"]
    artefacts = features["artefacts"]

    user_pref_vector = _user_pref_vector(
        features["user_features"], artefacts["user_map"], artefacts["categories"]
    )
    model = HybridRecommender.fit(
        matrix,
        mcfg,
        item_content=features.get("item_content"),
        user_pref_vector=user_pref_vector,
    )
    return {"model": model, "model_config": mcfg, "artefacts": artefacts}
