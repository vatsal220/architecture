# -*- coding: utf-8 -*-
"""Unit tests for the ranking-aware evaluation metrics (Part 5).

Each metric is checked against a hand-computed expected value on a tiny,
fully-deterministic example so regressions in the ranking maths are caught.
"""
import numpy as np
import pytest

from model import evaluate as ev


def test_precision_at_k():
    recommended = [1, 2, 3, 4]
    relevant = {2, 4}
    assert ev.precision_at_k(recommended, relevant, 4) == pytest.approx(0.5)
    assert ev.precision_at_k(recommended, relevant, 2) == pytest.approx(0.5)


def test_recall_at_k():
    recommended = [1, 2, 3, 4]
    relevant = {2, 4, 9}        # 9 is never recommended
    assert ev.recall_at_k(recommended, relevant, 4) == pytest.approx(2 / 3)


def test_average_precision_rewards_ranking():
    relevant = {1, 2}
    top_first = ev.average_precision_at_k([1, 2, 3], relevant, 3)
    bottom = ev.average_precision_at_k([3, 1, 2], relevant, 3)
    assert top_first == pytest.approx(1.0)
    assert top_first > bottom


def test_ndcg_perfect_is_one():
    relevant = {1, 2}
    assert ev.ndcg_at_k([1, 2, 3], relevant, 3) == pytest.approx(1.0)


def test_ndcg_position_discount():
    relevant = {3}
    high = ev.ndcg_at_k([3, 1, 2], relevant, 3)
    low = ev.ndcg_at_k([1, 2, 3], relevant, 3)
    assert high > low


def test_coverage():
    assert ev.catalogue_coverage({1, 2, 3}, 10) == pytest.approx(0.3)


def test_intra_list_diversity_distinct_categories():
    # two items in orthogonal one-hot categories -> max diversity (1.0)
    item_content = np.array([[1.0, 0.0], [0.0, 1.0]])
    item_index = {10: 0, 20: 1}
    assert ev.intra_list_diversity([10, 20], item_content, item_index) == pytest.approx(1.0)


def test_intra_list_diversity_same_category():
    item_content = np.array([[1.0, 0.0], [1.0, 0.0]])
    item_index = {10: 0, 20: 1}
    assert ev.intra_list_diversity([10, 20], item_content, item_index) == pytest.approx(0.0)


def test_empty_relevant_sets_are_safe():
    assert ev.recall_at_k([1, 2], set(), 2) == 0.0
    assert ev.average_precision_at_k([1, 2], set(), 2) == 0.0
