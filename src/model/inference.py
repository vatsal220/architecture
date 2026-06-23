# -*- coding: utf-8 -*-
"""Inference
Stage ``06_generate_recommendations``.

Batch scoring of every eligible citizen. The same :class:`HybridRecommender`
API is used here as in evaluation, so offline metrics are a faithful predictor
of what is served. Output is a tidy ``(user_id, item_id, rank, score,
category, model_path, batch_id)`` table — the contract the downstream Curated /
serving layer consumes (see architecture/solution_architecture.md "Scoring
Output").

Scoring eligibility (mirrors the architecture doc):
* a user must exist in the trained user map (had >=1 valid interaction), OR
* be routed to the cold-start popularity/content path;
* already-seen programs are excluded when ``recommendations.exclude_seen`` is
  set, so we never recommend something the citizen already engaged with.
"""
from __future__ import annotations

import datetime as dt
from typing import Dict

import numpy as np
import pandas as pd

from utils import logger


def generate_recommendations(cfg, model_bundle: Dict, features: Dict,
                             items: pd.DataFrame, batch_id: str,
                             model_path: str = "") -> pd.DataFrame:
    """Produce a top-N ranked slate for every user in the trained user map."""
    rc = cfg.recommendations
    top_n = int(rc["top_n"])
    exclude_seen = bool(rc["exclude_seen"])

    model = model_bundle["model"]
    artefacts = features["artefacts"]
    user_map = artefacts["user_map"]
    item_map = artefacts["item_map"]
    inv_item_map = {j: it for it, j in item_map.items()}
    matrix = features["matrix"]
    cat_lookup = dict(zip(items["item_id"], items["category"]))

    user_ids = np.array(sorted(user_map.keys()))
    user_indices = np.array([user_map[u] for u in user_ids])

    # Score in chunks to bound memory (stand-in for distributed batch scoring).
    chunk = 2000
    records = []
    n_items = len(item_map)
    eff_n = min(top_n, n_items)
    for start in range(0, len(user_indices), chunk):
        idx = user_indices[start:start + chunk]
        uids = user_ids[start:start + chunk]
        scores = model.score_users(idx)
        if exclude_seen:
            seen = matrix[idx].toarray() > 0
            scores = np.where(seen, -np.inf, scores)
        topk = np.argpartition(-scores, kth=min(eff_n, n_items - 1), axis=1)[:, :eff_n]
        order = np.argsort(-np.take_along_axis(scores, topk, axis=1), axis=1)
        topk = np.take_along_axis(topk, order, axis=1)
        topscore = np.take_along_axis(scores, topk, axis=1)
        for r, uid in enumerate(uids):
            for rank in range(eff_n):
                col = int(topk[r, rank])
                sc = float(topscore[r, rank])
                if not np.isfinite(sc):
                    continue
                item_id = inv_item_map[col]
                records.append(
                    {
                        "user_id": int(uid),
                        "item_id": int(item_id),
                        "rank": rank + 1,
                        "score": round(sc, 6),
                        "category": cat_lookup.get(item_id, "unknown"),
                        "model_path": model_path,
                        "batch_id": batch_id,
                        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
                    }
                )

    recs = pd.DataFrame.from_records(records)
    logger.info(
        f"06_generate_recommendations: {recs['user_id'].nunique()} users x "
        f"<= {top_n} items = {len(recs)} rows (batch {batch_id})"
    )
    return recs
