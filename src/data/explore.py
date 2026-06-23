# -*- coding: utf-8 -*-
"""Data Explore
Lightweight exploratory profiling used for logging/observability and to feed the
data-quality report with descriptive context. 
"""
import pandas as pd
from typing import Dict

def profile_interactions(interactions: pd.DataFrame, items: pd.DataFrame) -> Dict:
    """Return a compact descriptive profile of the interaction matrix.

    Reports the figures a reviewer needs to sanity-check sparsity and the
    long-tail nature of engagement (which motivates the modelling choices).

    Args:
        interactions: DataFrame containing the interaction data
        items: DataFrame containing the item metadata

    Returns:
        Dict containing the profile of the interaction matrix
            "n_users": int,
            "n_items": int,
            "n_interactions": int,
            "matrix_density": float,
            "median_interactions_per_user": float,
            "median_interactions_per_item": float,
            "p95_interactions_per_item": float,
            "catalogue_categories": int,
            "cold_start_users_lte2": int,
    """
    n_users = interactions["user_id"].nunique()
    n_items = interactions["item_id"].nunique()
    n_inter = len(interactions)
    density = n_inter / (n_users * n_items) if n_users and n_items else 0.0

    per_user = interactions.groupby("user_id")["item_id"].count()
    per_item = interactions.groupby("item_id")["user_id"].count()
    return {
        "n_users": int(n_users),
        "n_items": int(n_items),
        "n_interactions": int(n_inter),
        "matrix_density": round(float(density), 6),
        "median_interactions_per_user": float(per_user.median()),
        "median_interactions_per_item": float(per_item.median()),
        "p95_interactions_per_item": float(per_item.quantile(0.95)),
        "catalogue_categories": int(items["category"].nunique()),
        "cold_start_users_lte2": int((per_user <= 2).sum()),
    }
