# -*- coding: utf-8 -*-
"""Predictor
Thin, framework-agnostic scoring wrapper used by the FastAPI service.

It loads a model bundle from the registry (``model.registry.load_model``) and
exposes a single :meth:`recommend` method. The same :class:`HybridRecommender`
that powers training/batch is reused here, so real-time and batch can never
disagree. Known citizens are scored collaboratively; unknown (cold) citizens
fall back to the popularity prior — the online-serving analogue of the batch
cold-start path.

In production the model artifacts are pulled from the **Models** S3 zone /
MLflow Model Registry at pod start-up (and on a refresh signal); locally they
are read from ``outputs/models/<version>``.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from utils import logger
from utils.config import load_config
from model.registry import load_model


class Predictor:
    """Loads a registered model once and serves top-N program recommendations."""

    def __init__(self, config_path: Optional[str] = None, version: Optional[str] = None):
        self.cfg = load_config(config_path)
        bundle = load_model(self.cfg, version)
        self.model = bundle["model"]
        self.version = bundle["version"]
        self.user_map: Dict[int, int] = bundle["user_map"]
        self.item_map: Dict[int, int] = bundle["item_map"]
        self.categories: List[str] = bundle["categories"]
        self.metadata = bundle["metadata"]
        self._inv_item_map = {j: it for it, j in self.item_map.items()}
        self._n_items = len(self.item_map)
        logger.info(
            f"Predictor ready: model={self.version} "
            f"users={len(self.user_map)} items={self._n_items}"
        )

    @property
    def n_known_users(self) -> int:
        return len(self.user_map)

    def _category_of(self, item_col: int) -> str:
        content = self.model.item_content
        if content is None or not self.categories:
            return "unknown"
        return self.categories[int(content[item_col].argmax())]

    def _popularity_topk(self, k: int) -> np.ndarray:
        scores = self.model.pop.score_all()
        kk = min(k, self._n_items)
        top = np.argpartition(-scores, kth=min(kk, self._n_items - 1))[:kk]
        return top[np.argsort(-scores[top])]

    def recommend(self, user_id: int, k: int = 10) -> Dict:
        """Return a ranked slate for ``user_id`` (cold-start aware)."""
        known = user_id in self.user_map
        if known:
            uidx = np.array([self.user_map[user_id]])
            scores = self.model.score_users(uidx)[0]
            kk = min(k, self._n_items)
            top = np.argpartition(-scores, kth=min(kk, self._n_items - 1))[:kk]
            top = top[np.argsort(-scores[top])]
            top_scores = scores[top]
            strategy = (
                "collaborative"
                if self.model.user_history_counts[uidx[0]] >= self.model.cold_start_min
                else "cold_start_blend"
            )
        else:
            top = self._popularity_topk(k)
            top_scores = self.model.pop.score_all()[top]
            strategy = "cold_start_popularity"

        items = [
            {
                "item_id": int(self._inv_item_map[int(c)]),
                "rank": rank + 1,
                "score": round(float(s), 6),
                "category": self._category_of(int(c)),
            }
            for rank, (c, s) in enumerate(zip(top, top_scores))
        ]
        return {
            "user_id": int(user_id),
            "model_version": self.version,
            "strategy": strategy,
            "known_user": known,
            "recommendations": items,
        }
