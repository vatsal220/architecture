# -*- coding: utf-8 -*-
"""Integration test for the FastAPI serving layer.

Runs the pipeline to register a model, loads it through the Predictor (the same
HybridRecommender used in training/batch), and exercises the HTTP API with
FastAPI's TestClient — verifying the real-time contract and the cold-start path.
"""
import pytest
from fastapi.testclient import TestClient

import run_pipeline
from serving.predictor import Predictor
import serving.app as app_module


@pytest.fixture(scope="module")
def trained_config(pipeline_config):
    cfg_path = pipeline_config(n_users=200, n_items=40, n_interactions=4000, n_factors=16)
    run_pipeline.run(str(cfg_path), overrides=[], only_stage=None)
    return str(cfg_path)


@pytest.fixture(scope="module")
def predictor(trained_config):
    return Predictor(config_path=trained_config)


def test_predictor_known_user(predictor):
    result = predictor.recommend(user_id=1, k=5)
    assert result["known_user"] is True
    assert len(result["recommendations"]) == 5
    ranks = [r["rank"] for r in result["recommendations"]]
    assert ranks == [1, 2, 3, 4, 5]
    assert result["model_version"]


def test_predictor_cold_user_falls_back(predictor):
    result = predictor.recommend(user_id=10_000_000, k=5)
    assert result["known_user"] is False
    assert result["strategy"] == "cold_start_popularity"
    assert len(result["recommendations"]) == 5


def test_api_endpoints(predictor):
    app_module._predictor = predictor  # inject the loaded model
    client = TestClient(app_module.app)

    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json()["status"] == "ready"

    r = client.get("/recommend/1", params={"k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["user_id"] == 1
    assert len(body["recommendations"]) == 3

    rb = client.post("/recommend", json={"user_ids": [1, 2, 3], "k": 2})
    assert rb.status_code == 200
    assert len(rb.json()) == 3
    assert all(len(u["recommendations"]) == 2 for u in rb.json())

    meta = client.get("/metadata").json()
    assert meta["model_version"] == predictor.version

    assert "tdp_recsys_requests_total" in client.get("/metrics").text
