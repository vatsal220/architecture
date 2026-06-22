# -*- coding: utf-8 -*-
"""Data Quality
Stage ``02_validate_data`` (input quality) and post-scoring quality checks.

Design
------
Every check is a small function that returns a :class:`CheckResult`. A check
declares a *severity*:

* ``BLOCK``  - a violation above the configured threshold aborts the run. Used
  for defects that would silently corrupt the model or the recommendations
  (missing ids, schema drift, too few rows).
* ``WARN``   - logged and surfaced in the report, but the run continues. Used
  for defects we can safely repair or tolerate (duplicates, rare orphan items).

Each check also returns the set of row indices it considers invalid so the
caller can quarantine rather than discard blindly. The full report is persisted
to the reports zone for auditability (a governance requirement on TDP).
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List
from utils import logger

import json
import pandas as pd
import datetime as dt

BLOCK = "BLOCK"
WARN = "WARN"

REQUIRED_INTERACTION_COLUMNS = {
    "user_id": "numeric",
    "item_id": "numeric",
    "rating": "numeric",
    "timestamp": "numeric",
}

@dataclass
class CheckResult:
    name: str
    severity: str
    passed: bool
    n_violations: int
    n_total: int
    action: str
    detail: str = ""
    invalid_index: List[int] = field(default_factory=list)

    @property
    def violation_fraction(self) -> float:
        return (self.n_violations / self.n_total) if self.n_total else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("invalid_index")  # keep the report compact
        d["violation_fraction"] = round(self.violation_fraction, 6)
        return d


class DataQualityError(Exception):
    """Raised when one or more BLOCK-severity checks fail."""

def check_schema(df: pd.DataFrame) -> CheckResult:
    """Rule 1 - Schema validation.

    Why: every downstream stage assumes the canonical interaction schema. A
    missing or non-numeric column is an upstream contract breach and must never
    reach feature engineering.
    Blocks: yes. Failures: abort immediately (cannot repair a missing column).

    Args:
        df: DataFrame containing the interaction data

    Returns:
        CheckResult containing the schema validation result
    """
    missing = [c for c in REQUIRED_INTERACTION_COLUMNS if c not in df.columns]
    non_numeric = [
        c
        for c in REQUIRED_INTERACTION_COLUMNS
        if c in df.columns and not pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce"))
    ]
    problems = []
    if missing:
        problems.append(f"missing columns: {missing}")
    if non_numeric:
        problems.append(f"non-numeric columns: {non_numeric}")
    passed = not problems
    return CheckResult(
        name="schema_validation",
        severity=BLOCK,
        passed=passed,
        n_violations=len(missing) + len(non_numeric),
        n_total=len(REQUIRED_INTERACTION_COLUMNS),
        action="abort run" if not passed else "none",
        detail="; ".join(problems) or "schema ok",
    )


def check_missing_identifiers(df: pd.DataFrame, max_null_fraction: float) -> CheckResult:
    """Rule 2 - Missing identifiers (null user_id / item_id).

    Why: a recommendation is meaningless without knowing the citizen and the
    program. Null ids cannot be imputed without fabricating engagement.
    Blocks: yes (default threshold 0.0). Failures: offending rows quarantined;
    run aborts if the null fraction exceeds the threshold.

    Args:
        df: DataFrame containing the interaction data
        max_null_fraction: Maximum fraction of null values allowed

    Returns:
        CheckResult containing the missing identifiers result
    """
    mask = df["user_id"].isna() | df["item_id"].isna()
    n = int(mask.sum())
    frac = n / len(df) if len(df) else 0.0
    passed = frac <= max_null_fraction
    return CheckResult(
        name="missing_identifiers",
        severity=BLOCK,
        passed=passed,
        n_violations=n,
        n_total=len(df),
        action="quarantine rows; abort if over threshold",
        detail=f"null id fraction {frac:.4f} (threshold {max_null_fraction})",
        invalid_index=df.index[mask].tolist(),
    )


def check_duplicates(df: pd.DataFrame) -> CheckResult:
    """Rule 3 - Duplicate records.

    Why: duplicate interactions over-weight a citizen/program pair and bias the
    confidence matrix. Often a benign re-delivery from upstream ingestion.
    Blocks: no. Failures: duplicates are dropped (keep first) and counted.

    Args:
        df: DataFrame containing the interaction data

    Returns:
        CheckResult containing the duplicate records result
    """
    dup_mask = df.duplicated(keep="first")
    n = int(dup_mask.sum())
    return CheckResult(
        name="duplicate_records",
        severity=WARN,
        passed=True,  # repairable, never blocks
        n_violations=n,
        n_total=len(df),
        action="drop duplicates (keep first)",
        detail=f"{n} exact duplicate rows",
        invalid_index=df.index[dup_mask].tolist(),
    )


def check_invalid_timestamps(df: pd.DataFrame, grace_days: int) -> CheckResult:
    """Rule 4 - Invalid timestamps (unparseable, non-positive, or future).

    Why: features such as recency and the temporal train/test split depend on
    trustworthy timestamps; a future timestamp leaks the future into training.
    Blocks: no (rows quarantined). Failures: offending rows dropped, counted.

    Args:
        df: DataFrame containing the interaction data
        grace_days: Number of days to grace for future timestamps

    Returns:
        CheckResult containing the invalid timestamps result
    """
    ts = pd.to_numeric(df["timestamp"], errors="coerce")
    cutoff = (dt.datetime.now() + dt.timedelta(days=grace_days)).timestamp()
    mask = ts.isna() | (ts <= 0) | (ts > cutoff)
    n = int(mask.sum())
    return CheckResult(
        name="invalid_timestamps",
        severity=WARN,
        passed=True,
        n_violations=n,
        n_total=len(df),
        action="quarantine offending rows",
        detail=f"{n} rows with null/non-positive/future timestamps",
        invalid_index=df.index[mask].tolist(),
    )


def check_invalid_ratings(df: pd.DataFrame, scale, max_fraction: float) -> CheckResult:
    """Rule 5 - Invalid interaction values (rating out of range).

    Why: out-of-range engagement values distort confidence weighting. A high
    rate of them signals an upstream encoding bug worth blocking on.
    Blocks: yes (if fraction exceeds threshold). Failures: rows quarantined.

    Args:
        df: DataFrame containing the interaction data
        scale: Tuple containing the minimum and maximum rating values
        max_fraction: Maximum fraction of invalid ratings allowed

    Returns:
        CheckResult containing the invalid ratings result
    """
    lo, hi = scale
    rating = pd.to_numeric(df["rating"], errors="coerce")
    mask = rating.isna() | (rating < lo) | (rating > hi)
    n = int(mask.sum())
    frac = n / len(df) if len(df) else 0.0
    passed = frac <= max_fraction
    return CheckResult(
        name="invalid_interaction_values",
        severity=BLOCK,
        passed=passed,
        n_violations=n,
        n_total=len(df),
        action="quarantine rows; abort if over threshold",
        detail=f"out-of-range rating fraction {frac:.4f} (threshold {max_fraction})",
        invalid_index=df.index[mask].tolist(),
    )


def check_reference_integrity(df: pd.DataFrame, items: pd.DataFrame, max_fraction: float) -> CheckResult:
    """Rule 6 - Reference integrity (every interaction item exists in metadata).

    Why: content features and category-based diversity require item metadata.
    Orphan items cannot be enriched or explained to the division.
    Blocks: yes (if fraction exceeds threshold). Failures: orphan rows dropped.

    Args:
        df: DataFrame containing the interaction data
        items: DataFrame containing the item metadata
        max_fraction: Maximum fraction of orphan items allowed

    Returns:
        CheckResult containing the reference integrity result
    """
    valid_items = set(items["item_id"].tolist())
    mask = ~df["item_id"].isin(valid_items)
    n = int(mask.sum())
    frac = n / len(df) if len(df) else 0.0
    passed = frac <= max_fraction
    return CheckResult(
        name="reference_integrity",
        severity=BLOCK,
        passed=passed,
        n_violations=n,
        n_total=len(df),
        action="drop orphan rows; abort if over threshold",
        detail=f"orphan-item fraction {frac:.4f} (threshold {max_fraction})",
        invalid_index=df.index[mask].tolist(),
    )


def check_min_rows(df: pd.DataFrame, min_rows: int) -> CheckResult:
    """Rule 7 - Volume floor.

    Why: collaborative filtering degrades to noise on tiny datasets; an
    unexpectedly small batch usually means a broken upstream extract.
    Blocks: yes. Failures: abort the run.

    Args:
        df: DataFrame containing the interaction data
        min_rows: Minimum number of rows required

    Returns:
        CheckResult containing the minimum volume result
    """
    n = len(df)
    passed = n >= min_rows
    return CheckResult(
        name="minimum_volume",
        severity=BLOCK,
        passed=passed,
        n_violations=max(0, min_rows - n),
        n_total=n,
        action="abort run" if not passed else "none",
        detail=f"{n} rows (minimum {min_rows})",
    )

def run_input_quality(cfg, interactions: pd.DataFrame, items: pd.DataFrame) -> Dict:
    """Run all input-quality checks, repair what is repairable, persist a report.

    Returns a dict with the cleaned DataFrame and the structured report. Raises
    :class:`DataQualityError` if any BLOCK check fails.

    Args:
        cfg: Dictionary containing the configuration
        interactions: DataFrame containing the interaction data
        items: DataFrame containing the item metadata

    Returns:
        Dict containing the cleaned DataFrame and the structured report
    """
    from utils.config import resolve_path

    dq = cfg.data_quality
    scale = cfg.dataset.get("rating_scale", [1, 5])

    df = interactions.reset_index(drop=True)
    results: List[CheckResult] = []

    results.append(check_schema(df))
    results.append(check_min_rows(df, int(dq["min_rows"])))
    results.append(check_missing_identifiers(df, float(dq["max_null_id_fraction"])))
    results.append(check_invalid_ratings(df, scale, float(dq["max_invalid_rating_fraction"])))
    results.append(check_invalid_timestamps(df, int(dq["future_timestamp_grace_days"])))
    results.append(check_reference_integrity(df, items, float(dq["max_orphan_item_fraction"])))
    dup_result = check_duplicates(df)
    results.append(dup_result)

    # Repair: drop duplicates + all quarantined invalid rows from non-schema checks.
    drop_idx = set()
    for r in results:
        if r.name in ("missing_identifiers", "invalid_timestamps",
                      "invalid_interaction_values", "reference_integrity",
                      "duplicate_records"):
            drop_idx.update(r.invalid_index)
    cleaned = df.drop(index=list(drop_idx)).reset_index(drop=True)

    # Cast types now that bad rows are gone.
    for col in ("user_id", "item_id", "rating", "timestamp"):
        cleaned[col] = pd.to_numeric(cleaned[col]).astype("int64")

    blocking_failures = [r for r in results if r.severity == BLOCK and not r.passed]

    report = {
        "stage": "02_validate_data",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "rows_in": len(df),
        "rows_out": len(cleaned),
        "rows_dropped": len(drop_idx),
        "passed": not blocking_failures,
        "checks": [r.to_dict() for r in results],
    }

    reports_dir = resolve_path(cfg, "reports_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / "data_quality_input.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    for r in results:
        level = logger.info if r.passed else logger.error
        level(f"02_validate_data [{r.severity}] {r.name}: {r.detail} -> {r.action}")

    if blocking_failures:
        names = ", ".join(r.name for r in blocking_failures)
        raise DataQualityError(f"Blocking data quality checks failed: {names}")

    logger.info(
        f"02_validate_data: {len(cleaned)}/{len(df)} rows passed "
        f"({len(drop_idx)} quarantined). Report -> {out}"
    )
    return {"interactions": cleaned, "items": items, "report": report}


def run_post_scoring_quality(cfg: dict, recommendations: pd.DataFrame, items: pd.DataFrame,
                             eligible_users: int) -> Dict:
    """Post-scoring quality gate over generated recommendations.

    Catches failure modes that only appear *after* inference: empty slates,
    out-of-catalogue items, duplicate items within a slate, and degenerate
    catalogue coverage (a sign of popularity collapse).

    Args:
        cfg: Dictionary containing the configuration
        recommendations: DataFrame containing the recommendations
        items: DataFrame containing the item metadata
        eligible_users: Integer containing the number of eligible users

    Returns:
        Dict containing the recommendations and the structured report
    """
    from utils.config import resolve_path

    top_n = int(cfg.recommendations["top_n"])
    valid_items = set(items["item_id"].tolist())
    checks = []

    counts = recommendations.groupby("user_id")["item_id"].count()
    short_slates = int((counts < top_n).sum())
    checks.append(
        {
            "name": "slate_completeness",
            "severity": WARN,
            "passed": True,
            "detail": f"{short_slates} users with fewer than top_n={top_n} items",
        }
    )

    users_scored = recommendations["user_id"].nunique()
    checks.append(
        {
            "name": "scoring_completeness",
            "severity": BLOCK,
            "passed": users_scored >= eligible_users,
            "detail": f"{users_scored}/{eligible_users} eligible users scored",
        }
    )

    out_of_catalogue = int((~recommendations["item_id"].isin(valid_items)).sum())
    checks.append(
        {
            "name": "recommendation_reference_integrity",
            "severity": BLOCK,
            "passed": out_of_catalogue == 0,
            "detail": f"{out_of_catalogue} recommended items not in catalogue",
        }
    )

    dup_in_slate = int(recommendations.duplicated(subset=["user_id", "item_id"]).sum())
    checks.append(
        {
            "name": "no_duplicate_items_in_slate",
            "severity": BLOCK,
            "passed": dup_in_slate == 0,
            "detail": f"{dup_in_slate} duplicate (user,item) pairs",
        }
    )

    catalogue_coverage = recommendations["item_id"].nunique() / max(1, len(valid_items))
    checks.append(
        {
            "name": "catalogue_coverage_floor",
            "severity": WARN,
            "passed": catalogue_coverage >= 0.05,
            "detail": f"catalogue coverage {catalogue_coverage:.3f}",
        }
    )

    blocking_failures = [c for c in checks if c["severity"] == BLOCK and not c["passed"]]
    report = {
        "stage": "06_generate_recommendations:post_quality",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "passed": not blocking_failures,
        "checks": checks,
    }
    reports_dir = resolve_path(cfg, "reports_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "data_quality_post_scoring.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    for c in checks:
        level = logger.info if c["passed"] else logger.error
        level(f"post_scoring [{c['severity']}] {c['name']}: {c['detail']}")

    if blocking_failures:
        names = ", ".join(c["name"] for c in blocking_failures)
        raise DataQualityError(f"Post-scoring quality checks failed: {names}")
    return report
