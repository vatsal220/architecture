# -*- coding: utf-8 -*-
"""FastAPI serving application.

Exposes the recommendation model as a stateless HTTP service suitable for
horizontal scaling on Kubernetes (EKS). Endpoints:

* ``GET  /health``           - liveness/readiness probe (used by k8s probes + ALB)
* ``GET  /ready``            - readiness (model loaded?)
* ``GET  /metadata``         - served model version + lineage
* ``GET  /recommend/{user_id}`` - real-time slate for one citizen
* ``POST /recommend``        - batch of citizens in one request
* ``GET  /metrics``          - Prometheus exposition (scraped by Prometheus on EKS)

Run locally:
    uvicorn serving.app:app --host 0.0.0.0 --port 8080
(or `python -m serving.app`). The container entry point does the same.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from utils import logger
from serving.predictor import Predictor

MAX_K = 50

# Module-level singletons populated at startup.
_predictor: Optional[Predictor] = None
_request_count = 0
_error_count = 0


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the registered model once at start-up (pod readiness depends on it)."""
    global _predictor
    config_path = os.environ.get("PIPELINE_CONFIG")  # defaults to configs/pipeline.yaml
    version = os.environ.get("MODEL_VERSION")         # None => registry 'latest'
    try:
        _predictor = Predictor(config_path=config_path, version=version)
    except Exception as exc:  # pragma: no cover - surfaced via /ready
        logger.error(f"startup: failed to load model: {exc}")
        _predictor = None
    yield


app = FastAPI(
    title="TDP Recommendation Service",
    version="1.0.0",
    description="Real-time citizen program recommendations (FastAPI on EKS).",
    lifespan=lifespan,
)


class RecommendationItem(BaseModel):
    item_id: int
    rank: int
    score: float
    category: str


class RecommendationResponse(BaseModel):
    user_id: int
    model_version: str
    strategy: str
    known_user: bool
    recommendations: List[RecommendationItem]


class BatchRequest(BaseModel):
    user_ids: List[int] = Field(..., min_length=1, max_length=1000)
    k: int = Field(10, ge=1, le=MAX_K)


def get_predictor() -> Predictor:
    if _predictor is None:  # pragma: no cover - guarded by startup
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _predictor


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready", "model_version": _predictor.version}


@app.get("/metadata")
def metadata() -> dict:
    p = get_predictor()
    return {
        "model_version": p.version,
        "n_known_users": p.n_known_users,
        "n_items": len(p.item_map),
        "lineage": p.metadata,
    }


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def recommend(user_id: int, k: int = Query(10, ge=1, le=MAX_K)) -> RecommendationResponse:
    global _request_count, _error_count
    _request_count += 1
    try:
        result = get_predictor().recommend(user_id, k)
        return RecommendationResponse(**result)
    except HTTPException:
        _error_count += 1
        raise
    except Exception as exc:  # pragma: no cover - defensive
        _error_count += 1
        logger.exception("recommend failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/recommend", response_model=List[RecommendationResponse])
def recommend_batch(req: BatchRequest) -> List[RecommendationResponse]:
    global _request_count
    _request_count += 1
    p = get_predictor()
    return [RecommendationResponse(**p.recommend(uid, req.k)) for uid in req.user_ids]


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    """Minimal Prometheus exposition (Prometheus on EKS scrapes this)."""
    loaded = 1 if _predictor is not None else 0
    return (
        "# HELP tdp_recsys_requests_total Total recommendation requests\n"
        "# TYPE tdp_recsys_requests_total counter\n"
        f"tdp_recsys_requests_total {_request_count}\n"
        "# HELP tdp_recsys_errors_total Total request errors\n"
        "# TYPE tdp_recsys_errors_total counter\n"
        f"tdp_recsys_errors_total {_error_count}\n"
        "# HELP tdp_recsys_model_loaded Whether a model is loaded\n"
        "# TYPE tdp_recsys_model_loaded gauge\n"
        f"tdp_recsys_model_loaded {loaded}\n"
    )


def main() -> None:  # pragma: no cover - container entry point
    import uvicorn

    uvicorn.run(
        "serving.app:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8080")),
        workers=int(os.environ.get("WEB_CONCURRENCY", "1")),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
