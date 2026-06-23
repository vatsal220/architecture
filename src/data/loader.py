# -*- coding: utf-8 -*-
"""Data Loader
Stage ``01_load_data`` of the recommendation pipeline.

Design
------
A single :class:`MovieLensLoader` class owns the full fetch → parse → persist
flow. It will fetch the dataset automatically (download + extract the
official ml-100k archive into the raw directory) when the files are not already
present, then parse u.data / u.item into the canonical schema shared by
every downstream stage:

    interactions: user_id, item_id, rating, timestamp
    items:        item_id, item_name, category
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from utils import logger

INTERACTION_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]
ITEM_COLUMNS = ["item_id", "item_name", "category"]

# MovieLens ml-100k genre columns, in the fixed order they appear in ``u.item``
# (columns 5..23). Used as the "program category" in our civic framing.
MOVIELENS_GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

DEFAULT_DOWNLOAD_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


class MovieLensLoader:
    """Fetches and loads the MovieLens ml-100k dataset into the canonical schema.

    The loader is idempotent: once the source files exist in the raw directory
    they are reused, so only the first run incurs a download.

    Usage::

        loader = MovieLensLoader(cfg)
        data = loader.load()              # {"interactions": df, "items": df}
    """

    def __init__(self, cfg: dict) -> None:
        """Initialize the MovieLensLoader.

        Args:
            cfg: Dictionary containing the configuration
        """
        from utils.config import resolve_path

        self.cfg = cfg
        self.ml = cfg.dataset.movielens
        self.raw_dir: Path = resolve_path(cfg, "raw_dir")
        self.interactions_file: str = self.ml.get("interactions_file", "u.data")
        self.items_file: str = self.ml.get("items_file", "u.item")
        self.download_enabled: bool = bool(self.ml.get("download", True))
        self.download_url: str = self.ml.get("download_url", DEFAULT_DOWNLOAD_URL)

    # -- public API ---------------------------------------------------------
    def load(self) -> Dict[str, pd.DataFrame]:
        """Fetch (if needed), parse, persist and return the raw dataset."""
        interactions_path, items_path = self._ensure_local_files()
        interactions = self._read_interactions(interactions_path)
        items = self._read_items(items_path)
        self._persist(interactions, items)
        logger.info(
            "01_load_data: loaded MovieLens ml-100k "
            f"({len(interactions)} interactions, {len(items)} items) into {self.raw_dir}"
        )
        return {"interactions": interactions, "items": items}

    def _ensure_local_files(self) -> Tuple[Path, Path]:
        """Return paths to the source files, downloading the archive if absent."""
        interactions_path = self._locate(self.interactions_file)
        items_path = self._locate(self.items_file)
        if interactions_path and items_path:
            return interactions_path, items_path

        if not self.download_enabled:
            raise FileNotFoundError(
                f"MovieLens source files not found under {self.raw_dir} "
                f"('{self.interactions_file}', '{self.items_file}') and "
                "dataset.movielens.download is disabled. Drop the ml-100k export "
                "there or enable download."
            )

        self._download_and_extract()
        interactions_path = self._locate(self.interactions_file)
        items_path = self._locate(self.items_file)
        if not (interactions_path and items_path):
            raise FileNotFoundError(
                f"Downloaded archive did not contain '{self.interactions_file}' "
                f"and '{self.items_file}'."
            )
        return interactions_path, items_path

    def _locate(self, filename: str) -> Path | None:
        """Find ``filename`` at the raw-dir root or under an extracted ml-100k/ dir."""
        for candidate in (self.raw_dir / filename, self.raw_dir / "ml-100k" / filename):
            if candidate.exists():
                return candidate
        return None

    def _download_and_extract(self) -> None:
        """Download the ml-100k zip and extract it into the raw directory."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"01_load_data: fetching MovieLens ml-100k from {self.download_url}")
        with urllib.request.urlopen(self.download_url) as response:  # noqa: S310
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            archive.extractall(self.raw_dir)
        logger.info(f"01_load_data: extracted ml-100k archive to {self.raw_dir}")

    def _read_interactions(self, path: Path) -> pd.DataFrame:
        """Parse the tab-separated ``u.data`` file into the canonical schema."""
        return pd.read_csv(
            path, sep="\t", names=INTERACTION_COLUMNS, encoding="latin-1"
        )

    def _read_items(self, path: Path) -> pd.DataFrame:
        """Parse the pipe-separated ``u.item`` file, reducing genre flags to one category.

        ``u.item`` layout: id | title | release | video_release | imdb_url |
        followed by 19 binary genre flags. We pick each program's primary
        category as the first flagged genre (argmax).
        """
        raw = pd.read_csv(path, sep="|", header=None, encoding="latin-1")
        genre_cols = list(range(5, raw.shape[1]))
        primary = raw[genre_cols].values.argmax(axis=1)
        categories = [
            MOVIELENS_GENRES[p] if p < len(MOVIELENS_GENRES) else "unknown"
            for p in primary
        ]
        return pd.DataFrame(
            {
                "item_id": raw[0].astype(int),
                "item_name": raw[1].astype(str),
                "category": categories,
            }
        )

    def _persist(self, interactions: pd.DataFrame, items: pd.DataFrame) -> None:
        """Write the canonical Raw zone so the stage is independently re-runnable."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        interactions.to_csv(self.raw_dir / "interactions.csv", index=False)
        items.to_csv(self.raw_dir / "items.csv", index=False)


def load_raw(cfg) -> Dict[str, pd.DataFrame]:
    """Entry point for stage ``01_load_data``.

    Thin wrapper that delegates to :class:`MovieLensLoader`, returning a dict
    with ``interactions`` and ``items`` DataFrames and materialising the Raw zone.
    """
    return MovieLensLoader(cfg).load()
