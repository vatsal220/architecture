# -*- coding: utf-8 -*-
"""Model Config
Typed view over the ``model`` block of ``configs/pipeline.yaml``.

Keeping a single dataclass here means the training, tuning, inference and
registry code all agree on the same hyper-parameter surface, and the exact
config that produced a model is serialised alongside the artefact for
reproducibility / model version control.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict


@dataclass
class ModelConfig:
    primary: str = "svd_mf"
    fallback: str = "popularity"
    n_factors: int = 64
    random_state: int = 42
    popularity_decay: bool = True
    cold_start_min_interactions: int = 3

    @classmethod
    def from_cfg(cls, cfg) -> "ModelConfig":
        m = cfg.model
        return cls(
            primary=m["primary"],
            fallback=m["fallback"],
            n_factors=int(m["svd_mf"]["n_factors"]),
            random_state=int(m["svd_mf"]["random_state"]),
            popularity_decay=bool(m["popularity"]["decay"]),
            cold_start_min_interactions=int(m["hybrid"]["cold_start_min_interactions"]),
        )

    def to_dict(self) -> Dict:
        return asdict(self)
