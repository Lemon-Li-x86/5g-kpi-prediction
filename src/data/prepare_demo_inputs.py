"""Prepare small, non-destructive CSV fixtures for final Streamlit acceptance.

Reads the existing local preprocessed KPI dataset and writes only under the
Git-ignored resources/demo_inputs/ folder. Does not touch model weights.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config import ROOT, KPIS, KEYS, ML_HISTORY_HOURS

COLUMNS = ["Time", *KEYS, *KPIS]


def prepare_demo_frames(frame: pd.DataFrame, *, forecast_hours: int = ML_HISTORY_HOURS,
                        anomaly_hours: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing KPI columns: {sorted(missing)}")
    if forecast_hours < ML_HISTORY_HOURS:
        raise ValueError(f"Traditional ML needs {ML_HISTORY_HOURS} consecutive history hours")
    if anomaly_hours < 2:
        raise ValueError("Anomaly detection needs a context hour and at least one test hour")

    df = frame[COLUMNS].copy()
    if df[COLUMNS].isna().any().any():
        raise ValueError("Demo input cannot contain missing KPI/ID/time values")
    df = df.sort_values(KEYS + ["Time"], kind="stable")
    if df.duplicated(KEYS + ["Time"]).any():
        raise ValueError("Duplicate beam-hour records in source dataset")

    last_hours = np.sort(df["Time"].unique())
    if len(last_hours) < max(forecast_hours, anomaly_hours):
        raise ValueError("Not enough hours in source data")
    selected_anomaly_hours = last_hours[-anomaly_hours:]
    if not np.all(np.diff(selected_anomaly_hours) == 1):
        raise ValueError("Anomaly hours are not contiguous")

    groups = df.groupby(KEYS, sort=True, dropna=False)
    first_beam = next(iter(groups))[1]
    forecast = first_beam.tail(forecast_hours).copy()
    if len(forecast) != forecast_hours or not np.all(np.diff(forecast["Time"]) == 1):
        raise ValueError("Selected beam has incomplete forecast history")

    anomaly = df[df["Time"].isin(selected_anomaly_hours)].copy()
    nbeams = df[KEYS].drop_duplicates().shape[0]
    if len(anomaly) != nbeams * anomaly_hours:
        raise ValueError("Missing hours for some beams in anomaly sample")
    # Input inference sorts these beam identifiers; keep CSV ordered for review.
    return forecast.reset_index(drop=True), anomaly.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=Path, default=ROOT / "data" / "processed" / "kpi_timeseries.csv"
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "resources" / "demo_inputs"
    )
    parser.add_argument("--forecast-hours", type=int, default=ML_HISTORY_HOURS)
    parser.add_argument("--anomaly-hours", type=int, default=3)
    args = parser.parse_args()

    if not args.csv.is_file():
        parser.error(f"Processed KPI CSV does not exist: {args.csv}")
    data = pd.read_csv(args.csv, usecols=COLUMNS)
    forecast, anomaly = prepare_demo_frames(
        data, forecast_hours=args.forecast_hours, anomaly_hours=args.anomaly_hours
    )
    args.out.mkdir(parents=True, exist_ok=True)
    prediction_csv = args.out / "forecast_history.csv"
    anomaly_csv = args.out / "anomaly_history.csv"
    forecast.to_csv(prediction_csv, index=False)
    anomaly.to_csv(anomaly_csv, index=False)
    print(f"Prediction input: {prediction_csv} ({len(forecast):,} rows)", flush=True)
    print(f"Anomaly input:    {anomaly_csv} ({len(anomaly):,} rows)", flush=True)
    print("Use these in Streamlit model management. Samples are not fault labels.", flush=True)


if __name__ == "__main__":
    main()
