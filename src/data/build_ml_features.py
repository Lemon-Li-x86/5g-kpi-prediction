from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

INPUT_FILE = PROCESSED_DIR / "kpi_timeseries.csv"

OUTPUT_FILE = PROCESSED_DIR / "ml_features.pkl"
SUMMARY_FILE = REPORT_DIR / "ml_feature_summary.json"


# ============================================================
# Configuration
# ============================================================

KPI_COLUMNS = [
    "DLPRB",
    "DLThpVol",
    "DLThpTime",
    "MR_number",
]

ID_COLUMNS = [
    "base_station",
    "cell",
    "beam",
]

# Historical lags:
# 1 hour, 2 hours, 24 hours, 1 week
LAGS = [
    1,
    2,
    24,
    168,
]

# Predict the next hour
FORECAST_HORIZON = 1

# W3 split:
# target Time < 672 -> train
# target Time >= 672 -> test
TEST_START_TIME = 672


# ============================================================
# Cache check
# ============================================================

def outputs_are_up_to_date():
    """
    Skip feature construction if:
    - feature cache exists
    - feature summary exists
    - neither input CSV nor this script has changed
    """

    if not OUTPUT_FILE.exists():
        return False

    if not SUMMARY_FILE.exists():
        return False

    output_mtime = min(
        OUTPUT_FILE.stat().st_mtime,
        SUMMARY_FILE.stat().st_mtime,
    )

    source_files = [
        INPUT_FILE,
        Path(__file__),
    ]

    for source_file in source_files:

        if not source_file.exists():
            return False

        if source_file.stat().st_mtime > output_mtime:
            return False

    return True


# ============================================================
# Load W3 processed dataset
# ============================================================

def load_processed_data():
    """
    Load only the columns required for W4 feature construction.

    Raw KPI columns are retained for:
    - t+1 regression targets
    - persistence baseline

    Standardised KPI columns are used as ML inputs.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Processed W3 dataset not found: {INPUT_FILE}"
        )

    required_columns = (
        [
            "Time",
            "base_station",
            "cell",
            "beam",
        ]
        + KPI_COLUMNS
        + [
            f"{kpi}_z"
            for kpi in KPI_COLUMNS
        ]
    )

    dtype_map = {
        "Time": np.int16,
        "base_station": np.int16,
        "cell": np.int8,
        "beam": np.int8,
    }

    for kpi in KPI_COLUMNS:
        dtype_map[kpi] = np.float32
        dtype_map[f"{kpi}_z"] = np.float32

    print("=" * 70)
    print("W4 machine-learning feature construction")
    print("=" * 70)

    print("\nLoading W3 processed dataset...")
    print(f"Input: {INPUT_FILE}")

    df = pd.read_csv(
        INPUT_FILE,
        usecols=required_columns,
        dtype=dtype_map,
    )

    print(
        f"Loaded shape: "
        f"{df.shape}"
    )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    if df.duplicated(
        subset=[
            "Time",
            "base_station",
            "cell",
            "beam",
        ]
    ).any():
        raise ValueError(
            "Duplicate Time/beam records detected."
        )

    # Sort each beam chronologically.
    #
    # This is essential before groupby().shift().
    df = (
        df
        .sort_values(
            by=[
                "base_station",
                "cell",
                "beam",
                "Time",
            ]
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# Validate time structure
# ============================================================

def validate_time_structure(df):
    """
    Confirm that all beams contain the same continuous
    time sequence.
    """

    unique_times = np.sort(
        df["Time"].unique()
    )

    if len(unique_times) < 2:
        raise ValueError(
            "Not enough time points."
        )

    if not np.all(
        np.diff(unique_times) == 1
    ):
        raise ValueError(
            "Global Time sequence is not continuous."
        )

    beam_count = (
        df[
            ID_COLUMNS
        ]
        .drop_duplicates()
        .shape[0]
    )

    expected_rows = (
        len(unique_times)
        * beam_count
    )

    if len(df) != expected_rows:
        raise ValueError(
            "Dataset is not a complete "
            "Time × beam panel."
        )

    print(
        f"\nTime range: "
        f"{unique_times[0]} "
        f"-> {unique_times[-1]}"
    )

    print(
        f"Time points: "
        f"{len(unique_times)}"
    )

    print(
        f"Beam count: "
        f"{beam_count}"
    )

    return (
        unique_times,
        beam_count,
    )


# ============================================================
# Feature construction
# ============================================================

def build_features(df):
    """
    Build W4 regression features.

    For each KPI, create:

        current value at time t
        lag 1
        lag 2
        lag 24
        lag 168

    All ML input features use W3 StandardScaler values.

    Raw KPI values are retained for:
        persistence baseline
        t+1 prediction targets

    Targets:

        target_DLPRB
        target_DLThpVol
        target_DLThpTime
        target_MR_number
    """

    print(
        "\nBuilding time-series features..."
    )

    max_lag = max(LAGS)

    # Group rows belonging to the same physical beam.
    grouped = df.groupby(
        ID_COLUMNS,
        sort=False,
    )

    feature_columns = []
    current_raw_columns = []
    target_columns = []

    # --------------------------------------------------------
    # Build features for all four KPIs
    # --------------------------------------------------------

    for kpi in KPI_COLUMNS:

        scaled_column = f"{kpi}_z"

        # ----------------------------------------------------
        # Current standardised value: KPI(t)
        # ----------------------------------------------------

        current_feature = (
            f"{kpi}_z_t"
        )

        df[current_feature] = (
            df[scaled_column]
            .astype(np.float32)
        )

        feature_columns.append(
            current_feature
        )

        # ----------------------------------------------------
        # Historical standardised values
        # ----------------------------------------------------

        for lag in LAGS:

            lag_column = (
                f"{kpi}_z_lag{lag}"
            )

            df[lag_column] = (
                grouped[
                    scaled_column
                ]
                .shift(lag)
                .astype(np.float32)
            )

            feature_columns.append(
                lag_column
            )

        # ----------------------------------------------------
        # Raw current value for persistence baseline
        # ----------------------------------------------------

        current_raw_column = (
            f"{kpi}_raw_t"
        )

        df[current_raw_column] = (
            df[kpi]
            .astype(np.float32)
        )

        current_raw_columns.append(
            current_raw_column
        )

        # ----------------------------------------------------
        # Raw t+1 regression target
        # ----------------------------------------------------

        target_column = (
            f"target_{kpi}"
        )

        df[target_column] = (
            grouped[kpi]
            .shift(
                -FORECAST_HORIZON
            )
            .astype(np.float32)
        )

        target_columns.append(
            target_column
        )

    # --------------------------------------------------------
    # Feature and target timestamps
    # --------------------------------------------------------

    df["feature_time"] = (
        df["Time"]
        .astype(np.int16)
    )

    df["target_time"] = (
        df["Time"]
        + FORECAST_HORIZON
    ).astype(np.int16)

    # --------------------------------------------------------
    # Remove rows without sufficient history or future target
    # --------------------------------------------------------

    min_time = int(
        df["Time"].min()
    )

    max_time = int(
        df["Time"].max()
    )

    first_usable_time = (
        min_time
        + max_lag
    )

    last_usable_time = (
        max_time
        - FORECAST_HORIZON
    )

    usable_mask = (
        (df["feature_time"] >= first_usable_time)
        &
        (df["feature_time"] <= last_usable_time)
    )

    ml_data = (
        df.loc[
            usable_mask,
            (
                [
                    "feature_time",
                    "target_time",
                    "base_station",
                    "cell",
                    "beam",
                ]
                + feature_columns
                + current_raw_columns
                + target_columns
            ),
        ]
        .copy()
    )

    # Original large dataframe no longer needed.
    del df

    # --------------------------------------------------------
    # Train/test split based on TARGET time
    #
    # target 0-671   -> training
    # target 672-839 -> testing
    # --------------------------------------------------------

    ml_data["is_train"] = (
        ml_data["target_time"]
        < TEST_START_TIME
    )

    # --------------------------------------------------------
    # Optimise row order
    #
    # Time-major ordering is convenient for later plotting
    # and prediction analysis.
    # --------------------------------------------------------

    ml_data = (
        ml_data
        .sort_values(
            by=[
                "target_time",
                "base_station",
                "cell",
                "beam",
            ]
        )
        .reset_index(drop=True)
    )

    return (
        ml_data,
        feature_columns,
        current_raw_columns,
        target_columns,
        first_usable_time,
        last_usable_time,
    )


# ============================================================
# Feature dataset validation
# ============================================================

def validate_feature_dataset(
    ml_data,
    feature_columns,
    current_raw_columns,
    target_columns,
    beam_count,
    first_usable_time,
    last_usable_time,
):
    """
    Validate sample counts, time boundaries and
    absence of invalid feature/target values.
    """

    print(
        "\nValidating feature dataset..."
    )

    # --------------------------------------------------------
    # Expected number of usable feature timestamps
    # --------------------------------------------------------

    usable_time_count = (
        last_usable_time
        - first_usable_time
        + 1
    )

    expected_rows = (
        usable_time_count
        * beam_count
    )

    if len(ml_data) != expected_rows:
        raise ValueError(
            f"Unexpected ML sample count: "
            f"{len(ml_data):,} "
            f"(expected {expected_rows:,})"
        )

    # --------------------------------------------------------
    # Validate t -> t+1 relationship
    # --------------------------------------------------------

    if not np.all(
        ml_data["target_time"].to_numpy()
        ==
        (
            ml_data["feature_time"].to_numpy()
            + FORECAST_HORIZON
        )
    ):
        raise ValueError(
            "Feature/target time relationship is invalid."
        )

    # --------------------------------------------------------
    # Check NaN and infinity column-by-column
    #
    # Column-by-column validation avoids creating one
    # extremely large temporary NumPy matrix.
    # --------------------------------------------------------

    numerical_columns = (
        feature_columns
        + current_raw_columns
        + target_columns
    )

    for column in numerical_columns:

        values = (
            ml_data[column]
            .to_numpy()
        )

        if np.isnan(values).any():
            raise ValueError(
                f"{column}: NaN values detected."
            )

        if np.isinf(values).any():
            raise ValueError(
                f"{column}: infinite values detected."
            )

    # --------------------------------------------------------
    # Every beam should contain the same number of samples
    # --------------------------------------------------------

    beam_sample_counts = (
        ml_data
        .groupby(
            ID_COLUMNS,
            sort=False,
        )
        .size()
    )

    if not np.all(
        beam_sample_counts.to_numpy()
        == usable_time_count
    ):
        raise ValueError(
            "Different beams contain different "
            "numbers of ML samples."
        )

    # --------------------------------------------------------
    # Train/test sample counts
    # --------------------------------------------------------

    train_rows = int(
        ml_data[
            "is_train"
        ].sum()
    )

    test_rows = int(
        (
            ~ml_data[
                "is_train"
            ]
        ).sum()
    )

    print(
        f"Feature time range: "
        f"{ml_data['feature_time'].min()} "
        f"-> "
        f"{ml_data['feature_time'].max()}"
    )

    print(
        f"Target time range:  "
        f"{ml_data['target_time'].min()} "
        f"-> "
        f"{ml_data['target_time'].max()}"
    )

    print(
        f"Total samples:      "
        f"{len(ml_data):,}"
    )

    print(
        f"Training samples:   "
        f"{train_rows:,}"
    )

    print(
        f"Testing samples:    "
        f"{test_rows:,}"
    )

    print(
        f"Input features:     "
        f"{len(feature_columns)}"
    )

    print(
        f"Prediction targets: "
        f"{len(target_columns)}"
    )

    return (
        train_rows,
        test_rows,
    )


# ============================================================
# Save feature summary
# ============================================================

def save_feature_summary(
    ml_data,
    feature_columns,
    current_raw_columns,
    target_columns,
    beam_count,
    train_rows,
    test_rows,
):
    """
    Save W4 feature-construction metadata.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "forecast": {
            "horizon_hours": FORECAST_HORIZON,
            "targets": KPI_COLUMNS,
            "target_columns": target_columns,
        },

        "features": {
            "input_representation": (
                "StandardScaler features from W3"
            ),
            "kpi_columns": KPI_COLUMNS,
            "lags_hours": LAGS,
            "feature_count": len(
                feature_columns
            ),
            "feature_columns": (
                feature_columns
            ),
            "current_raw_columns_for_persistence": (
                current_raw_columns
            ),
        },

        "dataset": {
            "rows": int(
                len(ml_data)
            ),
            "columns": int(
                len(ml_data.columns)
            ),
            "beam_count": int(
                beam_count
            ),
            "feature_time_min": int(
                ml_data[
                    "feature_time"
                ].min()
            ),
            "feature_time_max": int(
                ml_data[
                    "feature_time"
                ].max()
            ),
            "target_time_min": int(
                ml_data[
                    "target_time"
                ].min()
            ),
            "target_time_max": int(
                ml_data[
                    "target_time"
                ].max()
            ),
        },

        "split": {
            "method": (
                "chronological split "
                "based on target time"
            ),
            "test_start_target_time": (
                TEST_START_TIME
            ),
            "train_rows": (
                train_rows
            ),
            "test_rows": (
                test_rows
            ),
        },
    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# Main pipeline
# ============================================================

def main(force=False):

    # --------------------------------------------------------
    # Cache
    # --------------------------------------------------------

    if (
        not force
        and outputs_are_up_to_date()
    ):
        print(
            "ML feature dataset is already up to date."
        )

        print(
            f"Feature cache: {OUTPUT_FILE}"
        )

        print(
            f"Summary:       {SUMMARY_FILE}"
        )

        print(
            "Use --force to rebuild."
        )

        return

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_processed_data()

    (
        unique_times,
        beam_count,
    ) = validate_time_structure(
        df
    )

    # --------------------------------------------------------
    # Feature construction
    # --------------------------------------------------------

    (
        ml_data,
        feature_columns,
        current_raw_columns,
        target_columns,
        first_usable_time,
        last_usable_time,
    ) = build_features(
        df
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    (
        train_rows,
        test_rows,
    ) = validate_feature_dataset(
        ml_data,
        feature_columns,
        current_raw_columns,
        target_columns,
        beam_count,
        first_usable_time,
        last_usable_time,
    )

    # --------------------------------------------------------
    # Save feature cache
    # --------------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nSaving ML feature cache..."
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    ml_data.to_pickle(
        OUTPUT_FILE
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    save_feature_summary(
        ml_data,
        feature_columns,
        current_raw_columns,
        target_columns,
        beam_count,
        train_rows,
        test_rows,
    )

    print(
        f"Summary: {SUMMARY_FILE}"
    )

    # --------------------------------------------------------
    # File size
    # --------------------------------------------------------

    output_size_mb = (
        OUTPUT_FILE.stat().st_size
        / 1024
        / 1024
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "ML feature construction completed successfully."
    )

    print(
        f"Rows:               "
        f"{len(ml_data):,}"
    )

    print(
        f"Columns:            "
        f"{len(ml_data.columns)}"
    )

    print(
        f"Feature columns:    "
        f"{len(feature_columns)}"
    )

    print(
        f"Targets:            "
        f"{len(target_columns)}"
    )

    print(
        f"Train rows:         "
        f"{train_rows:,}"
    )

    print(
        f"Test rows:          "
        f"{test_rows:,}"
    )

    print(
        f"Feature cache size: "
        f"{output_size_mb:.1f} MB"
    )

    print(
        "=" * 70
    )


# ============================================================
# Command-line entry point
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Build time-series ML features "
            "for four 5G KPIs."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Rebuild the ML feature cache "
            "even if existing outputs "
            "are up to date."
        ),
    )

    args = parser.parse_args()

    main(
        force=args.force
    )