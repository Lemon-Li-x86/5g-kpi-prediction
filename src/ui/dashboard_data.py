"""Data access helpers for the 5G KPI Streamlit dashboard.

The dashboard reads existing experiment artifacts. It does not retrain models.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.core.config import ROOT as PROJECT_ROOT, KPIS
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = PROJECT_ROOT / "figures"
DEEP_LEARNING_MODELS = {"LSTM", "GRU", "TRANSFORMER"}


def list_experiment_runs(category: str, root: Path | None = None) -> list[str]:
    """Discover completed, isolated Web experiment reports for a dashboard page.

    category: 'forecast' accepts train_ml/train_dl; 'anomaly' accepts
    train_anomaly only. Never mix report files from separate runs.
    """
    base = Path(root) if root is not None else PROJECT_ROOT
    experiments = base / "resources" / "experiments"
    allowed = {"forecast": {"train_ml", "train_dl"},
               "anomaly": {"train_anomaly"}}.get(category)
    if allowed is None or not experiments.is_dir():
        return []

    runs = []
    for folder in experiments.iterdir():
        if not folder.is_dir() or folder.is_symlink():
            continue
        request = folder / "requested_parameters.json"
        if not request.is_file():
            continue
        try:
            task = json.loads(request.read_text(encoding="utf-8")).get("task")
        except (OSError, ValueError, TypeError):
            continue
        if task not in allowed:
            continue
        report = folder / "reports"
        expected = {
            "train_ml": "model_metrics.csv",
            "train_dl": "w3_deep_learning_metrics.csv",
            "train_anomaly": "w3_anomaly_method_summary.csv",
        }[task]
        if (report / expected).is_file():
            runs.append(folder.name)
    return sorted(runs, reverse=True)


def run_report_dir(run_id: str, root: Path | None = None) -> Path:
    """Resolve an ID from the enumerated local runs, never arbitrary paths."""
    base = Path(root) if root is not None else PROJECT_ROOT
    all_runs = set(list_experiment_runs("forecast", base))
    all_runs.update(list_experiment_runs("anomaly", base))
    if run_id not in all_runs:
        raise ValueError("Unknown or incomplete experiment version")
    return base / "resources" / "experiments" / run_id / "reports"



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


def load_model_metrics(report_dir: Path | None = None) -> pd.DataFrame:
    """Return model metrics in a common schema across ML and DL families."""
    report_dir = Path(report_dir) if report_dir is not None else REPORT_DIR
    combined_path = report_dir / "week3" / "w3_ml_dl_all_metrics.csv"
    if combined_path.exists():
        frame = pd.read_csv(combined_path)
        if "KPI" in frame.columns and "target" not in frame.columns:
            frame = frame.rename(columns={"KPI": "target"})
        if "family" not in frame.columns:
            frame["family"] = "Unknown"
    else:
        frames = []
        traditional_path = report_dir / "model_metrics.csv"
        deep_path = report_dir / "w3_deep_learning_metrics.csv"

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


def available_prediction_models(report_dir: Path | None = None) -> list[str]:
    """List models that have saved prediction series available to the UI."""
    report_dir = Path(report_dir) if report_dir is not None else REPORT_DIR
    models: set[str] = set()

    traditional_path = report_dir / "prediction_timeseries.csv"
    if traditional_path.exists():
        frame = pd.read_csv(traditional_path, usecols=["model"])
        models.update(frame["model"].dropna().astype(str).unique())

    for model in DEEP_LEARNING_MODELS:
        path = report_dir / f"w3_{model.lower()}_hourly_predictions.csv"
        if path.exists():
            models.add(model)

    return sorted(models, key=str.casefold)


def load_prediction_series(model: str, kpi: str, report_dir: Path | None = None) -> pd.DataFrame:
    """Load one model/KPI prediction series using common column names."""
    report_dir = Path(report_dir) if report_dir is not None else REPORT_DIR
    model_key = model.upper()

    if model_key in DEEP_LEARNING_MODELS:
        path = report_dir / f"w3_{model_key.lower()}_hourly_predictions.csv"
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

    path = report_dir / "prediction_timeseries.csv"
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


def load_anomaly_summary(report_dir: Path | None = None) -> pd.DataFrame:
    report_dir = Path(report_dir) if report_dir is not None else REPORT_DIR
    return read_csv_if_exists(report_dir / "w3_anomaly_method_summary.csv")


def load_anomaly_hourly_rates(report_dir: Path | None = None) -> pd.DataFrame:
    report_dir = Path(report_dir) if report_dir is not None else REPORT_DIR
    return read_csv_if_exists(report_dir / "w3_anomaly_rate_by_hour.csv")


def load_anomaly_events(report_dir: Path | None = None) -> pd.DataFrame:
    report_dir = Path(report_dir) if report_dir is not None else REPORT_DIR
    return read_csv_if_exists(report_dir / "w3_anomaly_candidate_events.csv")


def load_correlation_matrix(report_dir: Path | None = None) -> tuple[pd.DataFrame, str]:
    """Prefer the W4 matrix with significance work; fall back to W3 matrix."""
    report_dir = Path(report_dir) if report_dir is not None else REPORT_DIR
    w4_path = report_dir / "w4_anomaly_kpi_correlation_matrix.csv"
    w3_path = report_dir / "w3_anomaly_kpi_correlations.csv"

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


def load_correlation_tests(report_dir: Path | None = None) -> pd.DataFrame:
    report_dir = Path(report_dir) if report_dir is not None else REPORT_DIR
    return read_csv_if_exists(report_dir / "w4_anomaly_kpi_correlation_tests.csv")


def load_preprocessing_summary() -> dict:
    return read_json_if_exists(REPORT_DIR / "preprocessing_summary.json")


def load_experiment_config(report_dir: Path | None = None) -> dict:
    report_dir = Path(report_dir) if report_dir is not None else REPORT_DIR
    return read_json_if_exists(report_dir / "w3_anomaly_run_config.json")
