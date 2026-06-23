# -*- coding: utf-8 -*-
"""Unit tests for the data-quality checks (Part 4).

These lock in the *contract* of each rule: what it flags, its severity, and
whether it blocks. They are deliberately small and dependency-light so they run
in CI without any data download.
"""
import numpy as np
import pandas as pd
import pytest

from data import quality


@pytest.fixture
def items():
    return pd.DataFrame(
        {
            "item_id": [1, 2, 3],
            "item_name": ["a", "b", "c"],
            "category": ["Aquatics", "Skating", "Camps"],
        }
    )


@pytest.fixture
def good_interactions():
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 3],
            "item_id": [1, 2, 2, 3],
            "rating": [5, 4, 3, 5],
            "timestamp": [1_600_000_000, 1_600_000_100, 1_600_000_200, 1_600_000_300],
        }
    )


def test_schema_detects_missing_column(good_interactions):
    df = good_interactions.drop(columns=["rating"])
    result = quality.check_schema(df)
    assert not result.passed
    assert result.severity == quality.BLOCK


def test_schema_passes_on_valid(good_interactions):
    assert quality.check_schema(good_interactions).passed


def test_missing_identifiers_flags_nulls(good_interactions):
    df = good_interactions.copy()
    df.loc[0, "user_id"] = np.nan
    result = quality.check_missing_identifiers(df, max_null_fraction=0.0)
    assert result.n_violations == 1
    assert not result.passed                      # 0.25 > 0.0 threshold
    assert 0 in result.invalid_index


def test_duplicates_never_block(good_interactions):
    df = pd.concat([good_interactions, good_interactions.iloc[[0]]], ignore_index=True)
    result = quality.check_duplicates(df)
    assert result.severity == quality.WARN
    assert result.passed
    assert result.n_violations == 1


def test_invalid_ratings_threshold(good_interactions):
    df = good_interactions.copy()
    df.loc[0, "rating"] = 99
    blocked = quality.check_invalid_ratings(df, scale=[1, 5], max_fraction=0.0)
    tolerated = quality.check_invalid_ratings(df, scale=[1, 5], max_fraction=0.5)
    assert not blocked.passed
    assert tolerated.passed
    assert blocked.n_violations == 1


def test_invalid_timestamps_flags_future(good_interactions):
    df = good_interactions.copy()
    df.loc[0, "timestamp"] = int(pd.Timestamp("2100-01-01").timestamp())
    result = quality.check_invalid_timestamps(df, grace_days=1)
    assert result.n_violations == 1
    assert 0 in result.invalid_index


def test_reference_integrity_flags_orphans(good_interactions, items):
    df = good_interactions.copy()
    df.loc[0, "item_id"] = 999
    result = quality.check_reference_integrity(df, items, max_fraction=0.0)
    assert not result.passed
    assert result.n_violations == 1


def test_min_rows_blocks_small_batches(good_interactions):
    assert not quality.check_min_rows(good_interactions, min_rows=1000).passed
    assert quality.check_min_rows(good_interactions, min_rows=2).passed
