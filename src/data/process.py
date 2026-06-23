# -*- coding: utf-8 -*-
"""Data Process
Standardisation / cleansing that turns validated **Raw** rows into the
**Standard** and **Curated** zones, plus the temporal train/test split used for
honest offline evaluation.

This module assumes Stage 02 has already quarantined defective rows, so it only
performs deterministic, behaviour-preserving transforms:

* contiguous integer index maps for users and items (required by the matrix
  factorisation backend);
* a derived ``event_date`` column;
* a per-user temporal holdout so evaluation never scores on rows the model
  trained on (prevents leakage / inflated metrics).
"""
from __future__ import annotations
from typing import Dict, Tuple
from utils import logger

import numpy as np
import pandas as pd

def build_id_maps(df: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Return user_id -> row index and item_id -> col index mappings.
    
    Args:
        df: DataFrame containing the interaction data

    Returns:
        Tuple containing the user_id -> row index and item_id -> col index mappings
            user_map: Dictionary containing the user_id -> row index mappings
            item_map: Dictionary containing the item_id -> col index mappings
    """
    users = np.sort(df["user_id"].unique())
    items = np.sort(df["item_id"].unique())
    user_map = {int(u): i for i, u in enumerate(users)}
    item_map = {int(it): j for j, it in enumerate(items)}
    return user_map, item_map

def standardize(cfg: dict, interactions: pd.DataFrame, items: pd.DataFrame) -> Dict:
    """Produce the Standard zone: typed, de-duplicated, index-mapped interactions.
    
    Args:
        cfg: Dictionary containing the configuration
        interactions: DataFrame containing the interaction data
        items: DataFrame containing the item metadata

    Returns:
        Dict containing the Standard zone
            interactions: DataFrame containing the typed, de-duplicated, index-mapped interactions
            items: DataFrame containing the item metadata
    """
    from utils.config import resolve_path

    df = interactions.copy()
    df["event_date"] = pd.to_datetime(df["timestamp"], unit="s")
    # Collapse repeated citizen/program interactions to a single max-confidence
    # row (a citizen attending a program many times is strong positive signal).
    df = (
        df.sort_values("timestamp")
        .groupby(["user_id", "item_id"], as_index=False)
        .agg(rating=("rating", "max"),
             timestamp=("timestamp", "max"),
             event_date=("event_date", "max"),
             interaction_count=("rating", "size"))
    )

    standard_dir = resolve_path(cfg, "standard_dir")
    standard_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(standard_dir / "interactions.parquet", index=False)
    items.to_parquet(standard_dir / "items.parquet", index=False)
    logger.info(f"standardize: {len(df)} unique (user,item) rows -> {standard_dir}")
    return {"interactions": df, "items": items}

def temporal_train_test_split(cfg: dict, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user temporal holdout.

    For each user with enough history, the most recent ``test_fraction`` of
    interactions are held out for evaluation; everyone is kept in train. Users
    below ``min_interactions_for_eval`` contribute only to training (their
    sparse history would make ranking metrics meaningless and they are handled
    by the cold-start path at serving time).

    Args:
        cfg: Dictionary containing the configuration
        df: DataFrame containing the interaction data

    Returns:
        Tuple containing the train and test DataFrames
            train: DataFrame containing the training data
            test: DataFrame containing the test data
    """
    sp = cfg.split
    test_fraction = float(sp["test_fraction"])
    min_eval = int(sp["min_interactions_for_eval"])

    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    counts = df.groupby("user_id")["item_id"].transform("size")
    df = df.assign(_rank=df.groupby("user_id").cumcount(), _count=counts)

    eligible = df["_count"] >= min_eval
    # rows whose within-user rank falls in the most recent fraction -> test
    holdout_start = (df["_count"] * (1 - test_fraction)).astype(int)
    is_test = eligible & (df["_rank"] >= holdout_start)

    train = df[~is_test].drop(columns=["_rank", "_count"]).reset_index(drop=True)
    test = df[is_test].drop(columns=["_rank", "_count"]).reset_index(drop=True)
    logger.info(
        f"temporal_train_test_split: train={len(train)} test={len(test)} "
        f"(eval users={test['user_id'].nunique()})"
    )
    return train, test