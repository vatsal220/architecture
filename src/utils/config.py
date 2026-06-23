# -*- coding: utf-8 -*-
"""Config
Configuration utilities for the Toronto Data Platform (TDP) recommendation
pipeline.

Two concerns live here:

1. **Secrets / environment** - account-scoped credentials and the active `ENV`
   are read from the process environment (or a local ``.env`` file). These are
   never written to YAML and never committed.
2. **Pipeline configuration** - all behaviour-defining parameters (dataset,
   features, model, evaluation, thresholds) live in ``configs/pipeline.yaml``
   and are loaded into a light, dot-accessible mapping so that a run is fully
   reproducible from ``(config + dataset + git SHA)``.
"""
from __future__ import annotations

import os
import copy
import datetime as dt
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

load_dotenv()

current_date = str(dt.datetime.now())[0:10]
ENV = os.environ.get("ENV", "STAGING")

# Project root = two levels up from this file (src/utils/config.py -> repo root).
ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "pipeline.yaml"

creds = {
    "db_creds": {
        "user": os.environ.get("DB_USERNAME"),
        "pw": os.environ.get("DB_PASSWORD"),
        "host": os.environ.get("DB_HOST"),
    },
    "s3_creds": {
        "access_key": os.environ.get("AWS_ACCESS_KEY_ID"),
        "secret_key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
        "region": os.environ.get("AWS_REGION"),
    },
}


class Config(dict):
    """A dict that also supports attribute access and recursive ``.get`` paths.

    Example::

        cfg.model.svd_mf.n_factors
        cfg.get_path("model.svd_mf.n_factors", default=64)
    """

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover
        self[name] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """Return a nested value addressed by a dotted key path."""
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def _coerce(value: str) -> Any:
    """Best-effort coercion of a CLI override string into a python scalar."""
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _apply_override(cfg: Dict[str, Any], dotted: str, value: str) -> None:
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = _coerce(value)


def load_config(path: str | os.PathLike | None = None, overrides: list[str] | None = None) -> Config:
    """Load ``configs/pipeline.yaml`` (or ``path``) and apply ``key=value`` overrides.

    Args:
        path: Path to a YAML config. Defaults to ``configs/pipeline.yaml``.
        overrides: Optional list of ``dotted.key=value`` strings (CLI ``--set``).

    Returns:
        Config: A dot-accessible configuration object.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Invalid --set override '{override}', expected key=value")
        key, value = override.split("=", 1)
        _apply_override(data, key.strip(), value.strip())

    data.setdefault("root_dir", str(ROOT_DIR))
    return Config(copy.deepcopy(data))


def resolve_path(cfg: Config, key: str) -> Path:
    """Resolve a ``paths.<key>`` entry to an absolute path under the repo root."""
    rel = cfg.get_path(f"paths.{key}")
    if rel is None:
        raise KeyError(f"paths.{key} is not defined in the config")
    p = Path(rel)
    return p if p.is_absolute() else ROOT_DIR / p
