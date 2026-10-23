"""Canonical KPI fields and stable paths relative to the repository root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KPIS = ["DLPRB", "DLThpVol", "DLThpTime", "MR_number"]
KEYS = ["base_station", "cell", "beam"]
RAW_DATA = ROOT / "data" / "raw"
PROCESSED_DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"
RESOURCES = ROOT / "resources"
