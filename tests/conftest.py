# -*- coding: utf-8 -*-
"""Shared pytest configuration.

Ensures ``src`` is importable so tests can ``from data import quality`` etc.
whether pytest is invoked via ``make test-unit`` (which sets PYTHONPATH) or
directly.

Also provides ``pipeline_config``: a factory that writes a small
**MovieLens-format** fixture (``u.data`` / ``u.item``) into a temp raw directory
and returns a config pointing at it with auto-download disabled. This keeps the
data/feature/serving tests fast and fully offline while still exercising the
real MovieLens parsing path in ``MovieLensLoader``.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

N_GENRES = 19  # MovieLens ml-100k genre flag count


def _write_movielens_fixture(
    raw_dir: Path, n_users: int, n_items: int, n_interactions: int, seed: int = 42
) -> None:
    """Write tiny ``u.data``/``u.item`` files in the real ml-100k on-disk format.

    The interactions carry genuine collaborative structure (each user prefers a
    genre) so the downstream model has signal to learn, mirroring the shape of
    the real dataset without needing a network download in tests.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Items: id | title | release | video_release | imdb_url | 19 genre flags
    item_genre = rng.integers(0, N_GENRES, size=n_items)
    item_lines = []
    for i in range(1, n_items + 1):
        flags = [0] * N_GENRES
        flags[item_genre[i - 1]] = 1
        fields = [str(i), f"program_{i:04d}", "01-Jan-1995", "", "http://example.org"]
        fields += [str(f) for f in flags]
        item_lines.append("|".join(fields))
    (raw_dir / "u.item").write_text("\n".join(item_lines), encoding="latin-1")

    # Interactions: user prefers a genre; popular items sampled more often.
    user_pref = rng.integers(0, N_GENRES, size=n_users)
    item_pop = rng.power(0.4, size=n_items)
    item_choice = rng.choice(n_items, size=n_interactions, p=item_pop / item_pop.sum())
    user_choice = rng.integers(0, n_users, size=n_interactions)
    match = (user_pref[user_choice] == item_genre[item_choice]).astype(float)
    ratings = np.clip(
        np.round(3.0 + 1.5 * match + rng.normal(0, 0.6, size=n_interactions)), 1, 5
    ).astype(int)
    timestamps = 880_000_000 + rng.integers(0, 60_000_000, size=n_interactions)

    inter = pd.DataFrame(
        {
            "user_id": user_choice + 1,
            "item_id": item_choice + 1,
            "rating": ratings,
            "timestamp": timestamps,
        }
    ).drop_duplicates(subset=["user_id", "item_id"])  # ml-100k has unique (u, i)
    inter.to_csv(raw_dir / "u.data", sep="\t", header=False, index=False)


# Session-scoped so it can back fixtures of any scope (e.g. the module-scoped
# trained model in the serving test) without a ScopeMismatch.
@pytest.fixture(scope="session")
def pipeline_config(tmp_path_factory):
    """Return a factory ``make(**overrides) -> cfg_path`` backed by a local fixture."""

    def _make(n_users=300, n_items=60, n_interactions=6000, **overrides):
        tmp_path = tmp_path_factory.mktemp("pipeline")
        base = yaml.safe_load((ROOT / "configs" / "pipeline.yaml").read_text())
        for key in base["paths"]:
            base["paths"][key] = str(tmp_path / base["paths"][key])

        raw_dir = Path(base["paths"]["raw_dir"])
        _write_movielens_fixture(raw_dir, n_users, n_items, n_interactions)

        base["dataset"]["movielens"]["download"] = False
        base["data_quality"]["min_rows"] = overrides.pop("min_rows", 100)
        if "n_factors" in overrides:
            base["model"]["svd_mf"]["n_factors"] = overrides.pop("n_factors")

        cfg_path = tmp_path / "pipeline.yaml"
        cfg_path.write_text(yaml.safe_dump(base))
        return cfg_path

    return _make
