"""Data access helpers for the 5G KPI Streamlit dashboard.

The dashboard reads existing experiment artifacts. It does not retrain models.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = PROJECT_ROOT / "figures"

KPIS = ["DLPRB", "DLThpVol", "DLThpTime", "MR_number"]
DEEP_LEARNING_MODELS = {"LSTM", "GRU", "TRANSFORMER"}


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    """Read a CSV if present; return an empty frame for missing artifacts."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json_if_exists(path: Path) -> dict:
    """Read a JSON file if present; return an empty dict otherwise."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_model_metrics() -> pd.DataFrame:
    """Return model metrics in a common schema across ML and DL families."""
    combined_path = REPORT_DIR / "week3" / "w3_ml_dl_all_metrics.csv"
    if combined_path.exists():
        frame = pd.read_csv(combined_path)
        if "KPI" in frame.columns and "target" not in frame.columns:
            frame = frame.rename(columns={"KPI": "target"})
        if "family" not in frame.columns:
            frame["family"] = "Unknown"
    else:
        frames = []
        traditional_path = REPORT_DIR / "model_metrics.csv"
        deep_path = REPORT_DIR / "w3_deep_learning_metrics.csv"

        if traditional_path.exists():
            traditional = pd.read_csv(traditional_path)
            traditional["family"] = "Traditional ML"
            frames.append(traditional)

        if deep_path.exists():
            deep = pd.read_csv(deep_path)
            deep = deep.rename(columns={"KPI": "target"})
            deep["family"] = "Deep learning"
            frames.append(deep)

        if not frames:
            return pd.DataFrame()
        frame = pd.concat(frames, ignore_index=True, sort=False)

    required = {"target", "model", "MAE", "RMSE", "R2"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()

    for column in ("MAE", "RMSE", "R2"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["target"] = frame["target"].astype(str)
    frame["model"] = frame["model"].astype(str)
    return frame


def available_prediction_models() -> list[str]:
    """List models that have saved prediction series available to the UI."""
    models: set[str] = set()

    traditional_path = REPORT_DIR / "prediction_timeseries.csv"
    if traditional_path.exists():
        frame = pd.read_csv(traditional_path, usecols=["model"])
        models.update(frame["model"].dropna().astype(str).unique())

    for model in DEEP_LEARNING_MODELS:
        path = REPORT_DIR / f"w3_{model.lower()}_hourly_predictions.csv"
        if path.exists():
            models.add(model)

    return sorted(models, key=str.casefold)


def load_prediction_series(model: str, kpi: str) -> pd.DataFrame:
    """Load one model/KPI prediction series using common column names."""
    model_key = model.upper()

    if model_key in DEEP_LEARNING_MODELS:
        path = REPORT_DIR / f"w3_{model_key.lower()}_hourly_predictions.csv"
        frame = read_csv_if_exists(path)
        actual_col = f"{kpi}_actual"
        predicted_col = f"{kpi}_pred"

        if frame.empty or not {"Time", actual_col, predicted_col}.issubset(frame.columns):
            return pd.DataFrame()

        result = frame[["Time", actual_col, predicted_col]].copy()
        result = result.rename(columns={actual_col: "actual", predicted_col: "prediction"})
        result["model"] = model_key
        result["KPI"] = kpi
        return result.sort_values("Time")

    path = REPORT_DIR / "prediction_timeseries.csv"
    frame = read_csv_if_exists(path)
    required = {"target", "model", "target_time", "actual_mean", "prediction_mean"}

    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame()

    selected = frame[
        (frame["target"].astype(str) == kpi)
        & (frame["model"].astype(str).str.casefold() == model.casefold())
    ].copy()

    if selected.empty:
        return pd.DataFrame()

    selected = selected.rename(
        columns={
            "target_time": "Time",
            "actual_mean": "actual",
            "prediction_mean": "prediction",
        }
    )
    selected["model"] = model
    selected["KPI"] = kpi
    return selected[["Time", "actual", "prediction", "model", "KPI"]].sort_values("Time")


def load_anomaly_summary() -> pd.DataFrame:
    return read_csv_if_exists(REPORT_DIR / "w3_anomaly_method_summary.csv")


def load_anomaly_hourly_rates() -> pd.DataFrame:
    return read_csv_if_exists(REPORT_DIR / "w3_anomaly_rate_by_hour.csv")


def load_anomaly_events() -> pd.DataFrame:
    return read_csv_if_exists(REPORT_DIR / "w3_anomaly_candidate_events.csv")


def load_correlation_matrix() -> tuple[pd.DataFrame, str]:
    """Prefer the W4 matrix with significance work; fall back to W3 matrix."""
    w4_path = REPORT_DIR / "w4_anomaly_kpi_correlation_matrix.csv"
    w3_path = REPORT_DIR / "w3_anomaly_kpi_correlations.csv"

    for path, label in (
        (w4_path, "W4 diagnostic output"),
        (w3_path, "W3 exploratory output"),
    ):
        if path.exists():
            frame = pd.read_csv(path, index_col=0)
            frame.index = frame.index.astype(str)
            frame.columns = frame.columns.astype(str)
            return frame, label

    return pd.DataFrame(), "Unavailable"


def load_correlation_tests() -> pd.DataFrame:
    return read_csv_if_exists(REPORT_DIR / "w4_anomaly_kpi_correlation_tests.csv")


def load_preprocessing_summary() -> dict:
    return read_json_if_exists(REPORT_DIR / "preprocessing_summary.json")


def load_experiment_config() -> dict:
    return read_json_if_exists(REPORT_DIR / "w3_anomaly_run_config.json")
