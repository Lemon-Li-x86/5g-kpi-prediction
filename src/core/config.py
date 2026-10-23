"""Canonical KPI fields and stable paths relative to the repository root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KPIS = ["DLPRB", "DLThpVol", "DLThpTime", "MR_number"]
KEYS = ["base_station", "cell", "beam"]

# Shared ML feature/inference contract. Keep historical experiment semantics.
ML_LAGS = (1, 2, 24, 168)  # observed hours before the prediction origin
ML_FORECAST_HORIZON = 1     # one-hour-ahead target
ML_HISTORY_HOURS = max(ML_LAGS) + 1
DEFAULT_RANDOM_SEED = 42
DATASET_TEST_START_HOUR = 672  # historical gotsf-ds train/test boundary

# Fixed public dataset manifest used by download, audit and preprocessing.
DATASET_REPO_ID = "netop/gotsf-ds"
DATASET_FILE_DIRECTORY = "data/train"
DATASET_FILES = {
    "DLPRB": "DLPRB_train_0w-5w.csv",
    "DLThpVol": "DLThpVol_train_0w-5w.csv",
    "DLThpTime": "DLThpTime_train_0w-5w.csv",
    "MR_number": "MR_number_train_0w-5w.csv",
}
RAW_DATA = ROOT / "data" / "raw"
PROCESSED_DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"
RESOURCES = ROOT / "resources"
