"""Association of actual KPI values belonging to the current anomaly inference.

Pure data functions. An unsupervised consensus flag is a candidate, not a
verified fault. No significance test is claimed for arbitrary user uploads.
"""
from __future__ import annotations

import pandas as pd

from src.core.config import KEYS, KPIS

JOIN_COLUMNS = [*KEYS, "Time"]
FLAG_COLUMNS = ["sigma", "iqr", "isolation_forest", "one_class_svm", "consensus"]


def candidate_kpi_correlation(
    history: pd.DataFrame, detections: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return matched consensus observations and their Pearson matrix.

    Explicit one-to-one join prevents associating beam-hour labels to the
    wrong values. First (context) hour has no anomaly output.
    """
    required_history = set(JOIN_COLUMNS + KPIS)
    required_detection = set(JOIN_COLUMNS + FLAG_COLUMNS)
    if not required_history <= set(history.columns):
        raise ValueError("Input history missing required KPI or beam/time columns")
    if not required_detection <= set(detections.columns):
        raise ValueError("Anomaly output has no aligned beam/time identifiers")
    if history.duplicated(JOIN_COLUMNS).any() or detections.duplicated(JOIN_COLUMNS).any():
        raise ValueError("Duplicate beam-hour identifiers in association input")

    merged = detections[JOIN_COLUMNS + FLAG_COLUMNS].merge(
        history[JOIN_COLUMNS + KPIS],
        on=JOIN_COLUMNS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if (merged["_merge"] != "both").any():
        raise ValueError("Detected beam-hours are absent from the uploaded history")
    if merged[KPIS].isna().any().any():
        raise ValueError("Missing KPI values in detected beam-hours")
    candidates = merged.loc[
        merged["consensus"].astype(bool),
        JOIN_COLUMNS + KPIS,
    ].copy()
    if len(candidates) < 3:
        return candidates, pd.DataFrame()
    corr = candidates[KPIS].corr(method="pearson")
    return candidates, corr
