# -*- coding: utf-8 -*-
"""Integration test: Raw -> validate -> standardize -> split -> features.

Exercises the data + feature stages wired together (the half of the pipeline
that does not depend on BLAS-heavy training), verifying the contracts between
stages: the quality gate cleans rows, the temporal split avoids leakage, and the
feature stage emits a correctly-shaped interaction matrix + artefacts.
"""
import numpy as np
import pytest

from utils.config import load_config
from data.loader import load_raw
from data.quality import run_input_quality
from data.process import standardize, temporal_train_test_split
from feature.engineer import build_features


@pytest.fixture
def cfg(pipeline_config):
    return load_config(str(pipeline_config(n_users=200, n_items=40, n_interactions=4000)))


def test_data_to_features_contract(cfg):
    raw = load_raw(cfg)
    validated = run_input_quality(cfg, raw["interactions"], raw["items"])

    # The gate passes clean MovieLens data through without corrupting it.
    cleaned = validated["interactions"]
    assert cleaned["user_id"].notna().all()
    assert cleaned["item_id"].isin(set(raw["items"]["item_id"])).all()
    assert not cleaned.duplicated().any()

    std = standardize(cfg, cleaned, validated["items"])
    train, test = temporal_train_test_split(cfg, std["interactions"])

    # No leakage: a (user,item) held out for test must not also be in train.
    train_pairs = set(map(tuple, train[["user_id", "item_id"]].values))
    test_pairs = set(map(tuple, test[["user_id", "item_id"]].values))
    assert train_pairs.isdisjoint(test_pairs)

    features = build_features(cfg, train, std["items"])
    matrix = features["matrix"]
    assert matrix.shape[0] == features["artefacts"]["n_users"]
    assert matrix.shape[1] == features["artefacts"]["n_items"]
    assert matrix.nnz == len(train)
    assert np.all(matrix.data >= 1.0)            # confidence = 1 + alpha*w
    assert len(features["artefacts"]["categories"]) >= 1
