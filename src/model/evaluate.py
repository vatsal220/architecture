# -*- coding: utf-8 -*-
"""Evaluate
Stage ``05_evaluate_model``.

Recommendation-specific, ranking-aware metrics (Part 5). Accuracy-style metrics
(RMSE) are deliberately *not* used: the division consumes a ranked slate of
programs, so we measure the quality of the *ordering* and the *experience*:

* ``precision@k`` / ``recall@k`` - did relevant programs appear in the slate?
* ``map@k``                      - rewards placing relevant items higher.
* ``ndcg@k``                     - position-discounted gain; our promotion gate.
* ``coverage``                   - fraction of the catalogue ever recommended
                                   (guards against popularity collapse / equity).
* ``diversity``                  - mean intra-list category dissimilarity
                                   (avoids recommending 10 near-identical items).
* cold-start slice              - the same metrics restricted to low-history
                                   users, because that population is where a
                                   civic recommender most often fails silently.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd

from utils import logger


def _dcg(relevances: List[int]) -> float:
    return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevances))


def precision_at_k(recommended: List[int], relevant: set, k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def recall_at_k(recommended: List[int], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def average_precision_at_k(recommended: List[int], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    score, hits = 0.0, 0
    for idx, item in enumerate(recommended[:k]):
        if item in relevant:
            hits += 1
            score += hits / (idx + 1)
    return score / min(len(relevant), k)


def ndcg_at_k(recommended: List[int], relevant: set, k: int) -> float:
    gains = [1 if item in relevant else 0 for item in recommended[:k]]
    dcg = _dcg(gains)
    ideal = _dcg(sorted(gains, reverse=True)) if any(gains) else 0.0
    ideal = _dcg([1] * min(len(relevant), k)) if relevant else 0.0
    return dcg / ideal if ideal > 0 else 0.0


def catalogue_coverage(all_recommended: set, n_items: int) -> float:
    return len(all_recommended) / n_items if n_items else 0.0


def intra_list_diversity(recommended: List[int], item_content: np.ndarray,
                         item_index: Dict[int, int]) -> float:
    """1 - mean pairwise cosine similarity of recommended items' content vectors."""
    idxs = [item_index[i] for i in recommended if i in item_index]
    if len(idxs) < 2:
        return 0.0
    vecs = item_content[idxs]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = vecs / norms
    sim = unit @ unit.T
    n = len(idxs)
    upper = sim[np.triu_indices(n, k=1)]
    return float(1.0 - upper.mean())


def evaluate_model(cfg, model_bundle: Dict, features: Dict, test: pd.DataFrame) -> Dict:
    """Compute the full metric suite at the configured k (and a k-sweep)."""
    from utils.config import resolve_path

    ev = cfg.evaluation
    k = int(ev["k"])
    k_sweep = [int(x) for x in ev["k_sweep"]]
    model = model_bundle["model"]
    artefacts = features["artefacts"]
    user_map = artefacts["user_map"]
    item_map = artefacts["item_map"]
    inv_item_map = {j: it for it, j in item_map.items()}
    item_content = features["item_content"]
    matrix = features["matrix"]

    # Ground truth: held-out relevant items per user (only users present at train time).
    truth = defaultdict(set)
    for uid, iid in zip(test["user_id"], test["item_id"]):
        if uid in user_map and iid in item_map:
            truth[int(uid)].add(int(iid))
    eval_users = [u for u in truth if truth[u]]
    if not eval_users:
        raise ValueError("No evaluable users with held-out items; check the split")

    user_indices = np.array([user_map[u] for u in eval_users])
    scores = model.score_users(user_indices)

    # Mask items already seen in train so we evaluate genuine next-best items.
    seen = matrix[user_indices].toarray() > 0
    scores = np.where(seen, -np.inf, scores)

    max_k = max(k_sweep + [k])
    topk_idx = np.argpartition(-scores, kth=min(max_k, scores.shape[1] - 1), axis=1)[:, :max_k]
    # sort the partitioned top-k by score
    row_sort = np.argsort(-np.take_along_axis(scores, topk_idx, axis=1), axis=1)
    topk_idx = np.take_along_axis(topk_idx, row_sort, axis=1)

    history_counts = model.user_history_counts
    # Evaluation cold-start slice: truly cold users (<=2 interactions) have no
    # holdout to score against, so we report on the *least-active quartile* of
    # evaluable users as a measurable proxy for low-history performance. The
    # serving-time cold-start path itself triggers at model.cold_start_min.
    eval_history = np.array([history_counts[user_map[u]] for u in eval_users])
    cold_threshold = float(np.quantile(eval_history, 0.25))

    sweep = {}
    all_recommended = set()
    diversities = []
    cold_metrics = defaultdict(list)
    for ki in sorted(set(k_sweep + [k])):
        p, r, m, n = [], [], [], []
        for row, u in enumerate(eval_users):
            rec_cols = topk_idx[row, :ki]
            rec_items = [inv_item_map[c] for c in rec_cols]
            rel = truth[u]
            p.append(precision_at_k(rec_items, rel, ki))
            r.append(recall_at_k(rec_items, rel, ki))
            m.append(average_precision_at_k(rec_items, rel, ki))
            n.append(ndcg_at_k(rec_items, rel, ki))
            if ki == k:
                all_recommended.update(rec_items)
                diversities.append(intra_list_diversity(rec_items, item_content, item_map))
                if history_counts[user_map[u]] <= cold_threshold:
                    cold_metrics["precision"].append(p[-1])
                    cold_metrics["recall"].append(r[-1])
                    cold_metrics["ndcg"].append(n[-1])
        sweep[f"k={ki}"] = {
            "precision": round(float(np.mean(p)), 6),
            "recall": round(float(np.mean(r)), 6),
            "map": round(float(np.mean(m)), 6),
            "ndcg": round(float(np.mean(n)), 6),
        }

    metrics = {
        "k": k,
        "n_eval_users": len(eval_users),
        "precision_at_k": sweep[f"k={k}"]["precision"],
        "recall_at_k": sweep[f"k={k}"]["recall"],
        "map_at_k": sweep[f"k={k}"]["map"],
        "ndcg_at_k": sweep[f"k={k}"]["ndcg"],
        "coverage": round(catalogue_coverage(all_recommended, len(item_map)), 6),
        "diversity": round(float(np.mean(diversities)) if diversities else 0.0, 6),
        "k_sweep": sweep,
    }

    if ev.get("cold_start_report", True):
        metrics["cold_start"] = {
            "slice_definition": f"eval users with train history <= {cold_threshold:.0f} (least-active quartile)",
            "n_cold_users": len(cold_metrics["precision"]),
            "precision_at_k": round(float(np.mean(cold_metrics["precision"])), 6)
            if cold_metrics["precision"] else None,
            "recall_at_k": round(float(np.mean(cold_metrics["recall"])), 6)
            if cold_metrics["recall"] else None,
            "ndcg_at_k": round(float(np.mean(cold_metrics["ndcg"])), 6)
            if cold_metrics["ndcg"] else None,
        }

    reports_dir = resolve_path(cfg, "reports_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "evaluation.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    logger.info(
        f"05_evaluate_model: P@{k}={metrics['precision_at_k']:.4f} "
        f"R@{k}={metrics['recall_at_k']:.4f} MAP@{k}={metrics['map_at_k']:.4f} "
        f"NDCG@{k}={metrics['ndcg_at_k']:.4f} cov={metrics['coverage']:.4f} "
        f"div={metrics['diversity']:.4f}"
    )
    return metrics
