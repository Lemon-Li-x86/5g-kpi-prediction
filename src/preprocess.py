from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_FILE = PROCESSED_DIR / "kpi_timeseries.csv"
SUMMARY_FILE = REPORT_DIR / "preprocessing_summary.json"


# ============================================================
# Dataset configuration
# ============================================================

FILES = {
    "DLPRB": "DLPRB_train_0w-5w.csv",
    "DLThpVol": "DLThpVol_train_0w-5w.csv",
    "DLThpTime": "DLThpTime_train_0w-5w.csv",
    "MR_number": "MR_number_train_0w-5w.csv",
}

KPI_COLUMNS = [
    "DLPRB",
    "DLThpVol",
    "DLThpTime",
    "MR_number",
]

# First four weeks for training:
# 4 weeks × 7 days × 24 hours = 672 hours
TRAIN_END_TIME = 672


# ============================================================
# Utility functions
# ============================================================

def outputs_are_up_to_date():
    """
    Check whether preprocessing outputs already exist
    and are newer than both:
    1. all raw data files
    2. this preprocessing script

    If so, preprocessing can safely be skipped.
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
        RAW_DIR / filename
        for filename in FILES.values()
    ]

    source_files.append(Path(__file__))

    for source_file in source_files:
        if not source_file.exists():
            return False

        if source_file.stat().st_mtime > output_mtime:
            return False

    return True


def load_kpi_file(kpi_name, file_path):
    """
    Load and validate one raw KPI CSV file.

    Parameters
    ----------
    kpi_name : str
        KPI name, e.g. DLPRB.
    file_path : Path
        Path to the raw CSV file.

    Returns
    -------
    time_values : np.ndarray
        Continuous hourly time indices.
    beam_names : list[str]
        Beam identifiers such as '0_0_0'.
    values : np.ndarray
        KPI matrix with shape [time, beam].
    """

    print(f"\nLoading {kpi_name}...")
    print(f"File: {file_path.name}")

    if not file_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    # Remove accidental whitespace from column names
    df.columns = [
        str(col).strip()
        for col in df.columns
    ]

    # --------------------------------------------------------
    # Recover Time column
    #
    # The original gotsf-ds CSV files store the time index
    # as an unnamed first column, which pandas reads as
    # "Unnamed: 0".
    # --------------------------------------------------------

    if "Time" not in df.columns:
        first_col = df.columns[0]

        if first_col.startswith("Unnamed:"):
            df = df.rename(
                columns={first_col: "Time"}
            )
        else:
            raise ValueError(
                f"{file_path.name}: "
                f"cannot identify Time column"
            )

    # --------------------------------------------------------
    # Validate Time
    # --------------------------------------------------------

    time = pd.to_numeric(
        df["Time"],
        errors="coerce",
    )

    if time.isna().any():
        raise ValueError(
            f"{file_path.name}: invalid Time values"
        )

    if time.duplicated().any():
        raise ValueError(
            f"{file_path.name}: "
            f"duplicate Time values detected"
        )

    time_values = time.to_numpy(
        dtype=np.int32
    )

    if len(time_values) > 1:
        time_diff = np.diff(time_values)

        if not np.all(time_diff == 1):
            raise ValueError(
                f"{file_path.name}: "
                f"Time axis is not continuous"
            )

    # --------------------------------------------------------
    # Validate beam identifiers
    # --------------------------------------------------------

    beam_df = df.drop(
        columns=["Time"]
    )

    beam_names = beam_df.columns.tolist()

    for beam_name in beam_names:
        parts = beam_name.split("_")

        if (
            len(parts) != 3
            or not all(
                part.isdigit()
                for part in parts
            )
        ):
            raise ValueError(
                f"{file_path.name}: "
                f"invalid beam identifier "
                f"'{beam_name}'"
            )

    # --------------------------------------------------------
    # Convert KPI values to numeric
    # --------------------------------------------------------

    values = (
        beam_df
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    # --------------------------------------------------------
    # Validate KPI values
    # --------------------------------------------------------

    if np.isnan(values).any():
        raise ValueError(
            f"{file_path.name}: "
            f"missing or non-numeric KPI values"
        )

    if np.isinf(values).any():
        raise ValueError(
            f"{file_path.name}: "
            f"infinite KPI values detected"
        )

    if (values < 0).any():
        raise ValueError(
            f"{file_path.name}: "
            f"negative KPI values detected"
        )

    print(f"Shape:      {df.shape}")
    print(
        f"Time:       "
        f"{time_values[0]} "
        f"-> {time_values[-1]}"
    )
    print(
        f"Beam count: "
        f"{len(beam_names)}"
    )

    return (
        time_values,
        beam_names,
        values,
    )


def parse_beam_names(beam_names):
    """
    Convert beam identifiers such as:

        12_2_31

    into:

        base_station = 12
        cell = 2
        beam = 31
    """

    base_stations = []
    cells = []
    beams = []

    for name in beam_names:
        parts = name.split("_")

        if (
            len(parts) != 3
            or not all(
                part.isdigit()
                for part in parts
            )
        ):
            raise ValueError(
                f"Invalid beam name: {name}"
            )

        base_station, cell, beam = map(
            int,
            parts,
        )

        base_stations.append(
            base_station
        )
        cells.append(cell)
        beams.append(beam)

    return (
        np.asarray(
            base_stations,
            dtype=np.int16,
        ),
        np.asarray(
            cells,
            dtype=np.int8,
        ),
        np.asarray(
            beams,
            dtype=np.int8,
        ),
    )


def build_long_table(
    reference_time,
    reference_beams,
    loaded_data,
):
    """
    Convert the original wide KPI matrices into
    one long-format time-series table.
    """

    n_times = len(reference_time)
    n_beams = len(reference_beams)

    (
        base_station_ids,
        cell_ids,
        beam_ids,
    ) = parse_beam_names(
        reference_beams
    )

    processed = pd.DataFrame({
        "Time": np.repeat(
            reference_time,
            n_beams,
        ).astype(np.int16),

        "base_station": np.tile(
            base_station_ids,
            n_times,
        ),

        "cell": np.tile(
            cell_ids,
            n_times,
        ),

        "beam": np.tile(
            beam_ids,
            n_times,
        ),
    })

    for kpi_name in KPI_COLUMNS:
        processed[kpi_name] = (
            loaded_data[kpi_name]
            .reshape(-1)
            .astype(np.float32)
        )

    return processed


def add_train_test_split(processed):
    """
    Create a chronological train/test split.

    Time 0-671   -> train
    Time 672-839 -> test
    """

    processed["split"] = np.where(
        processed["Time"]
        < TRAIN_END_TIME,
        "train",
        "test",
    )

    return processed


def standardise_kpis(processed):
    """
    Standardise all KPI columns.

    IMPORTANT:
    The StandardScaler is fitted on TRAINING data only.
    Test data is transformed using the same scaler.

    This prevents data leakage.
    """

    train_mask = (
        processed["split"]
        == "train"
    )

    scaler = StandardScaler()

    scaler.fit(
        processed.loc[
            train_mask,
            KPI_COLUMNS,
        ]
    )

    scaled_values = scaler.transform(
        processed[KPI_COLUMNS]
    ).astype(np.float32)

    for index, kpi_name in enumerate(
        KPI_COLUMNS
    ):
        processed[
            f"{kpi_name}_z"
        ] = scaled_values[:, index]

    return processed, scaler


def validate_processed_data(processed):
    """
    Perform final consistency checks on the
    processed dataset.
    """

    expected_rows = 840 * 2880

    if len(processed) != expected_rows:
        raise ValueError(
            "Unexpected processed row count: "
            f"{len(processed):,} "
            f"(expected {expected_rows:,})"
        )

    if processed.isna().any().any():
        raise ValueError(
            "Processed dataset contains "
            "missing values"
        )

    numeric_columns = (
        KPI_COLUMNS
        + [
            f"{kpi}_z"
            for kpi in KPI_COLUMNS
        ]
    )

    numeric_values = (
        processed[
            numeric_columns
        ]
        .to_numpy()
    )

    if not np.isfinite(
        numeric_values
    ).all():
        raise ValueError(
            "Processed dataset contains "
            "non-finite values"
        )

    duplicate_count = (
        processed
        .duplicated(
            subset=[
                "Time",
                "base_station",
                "cell",
                "beam",
            ]
        )
        .sum()
    )

    if duplicate_count != 0:
        raise ValueError(
            f"Processed dataset contains "
            f"{duplicate_count} duplicate "
            f"time/beam records"
        )


def save_preprocessing_summary(
    processed,
    scaler,
):
    """
    Save preprocessing metadata and scaler
    parameters for reproducibility.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_mask = (
        processed["split"]
        == "train"
    )

    test_mask = (
        processed["split"]
        == "test"
    )

    z_columns = [
        f"{kpi}_z"
        for kpi in KPI_COLUMNS
    ]

    train_z_mean = (
        processed.loc[
            train_mask,
            z_columns,
        ]
        .mean()
    )

    train_z_std = (
        processed.loc[
            train_mask,
            z_columns,
        ]
        .std(ddof=0)
    )

    summary = {
        "dataset": {
            "rows": int(
                len(processed)
            ),
            "columns": int(
                len(
                    processed.columns
                )
            ),
            "time_min": int(
                processed[
                    "Time"
                ].min()
            ),
            "time_max": int(
                processed[
                    "Time"
                ].max()
            ),
            "beam_count": int(
                processed[
                    [
                        "base_station",
                        "cell",
                        "beam",
                    ]
                ]
                .drop_duplicates()
                .shape[0]
            ),
        },

        "split": {
            "method": (
                "chronological"
            ),
            "train_time": [
                0,
                TRAIN_END_TIME - 1,
            ],
            "test_time": [
                TRAIN_END_TIME,
                int(
                    processed[
                        "Time"
                    ].max()
                ),
            ],
            "train_rows": int(
                train_mask.sum()
            ),
            "test_rows": int(
                test_mask.sum()
            ),
        },

        "standardisation": {
            "method": (
                "StandardScaler"
            ),
            "fit_scope": (
                "training data only"
            ),
            "kpi_columns": (
                KPI_COLUMNS
            ),
            "mean": {
                kpi: float(mean)
                for kpi, mean
                in zip(
                    KPI_COLUMNS,
                    scaler.mean_,
                )
            },
            "scale": {
                kpi: float(scale)
                for kpi, scale
                in zip(
                    KPI_COLUMNS,
                    scaler.scale_,
                )
            },
            "train_standardised_mean": {
                column: float(value)
                for column, value
                in train_z_mean.items()
            },
            "train_standardised_std": {
                column: float(value)
                for column, value
                in train_z_std.items()
            },
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
# Main preprocessing pipeline
# ============================================================

def main(force=False):

    # --------------------------------------------------------
    # Skip expensive preprocessing if nothing changed
    # --------------------------------------------------------

    if (
        not force
        and outputs_are_up_to_date()
    ):
        print(
            "Processed dataset is already "
            "up to date."
        )
        print(
            f"CSV: {OUTPUT_FILE}"
        )
        print(
            "Use --force to rebuild it."
        )
        return

    print(
        "=" * 70
    )
    print(
        "5G KPI preprocessing"
    )
    print(
        "=" * 70
    )

    loaded_data = {}

    reference_time = None
    reference_beams = None

    # --------------------------------------------------------
    # 1. Load and validate all four KPI files
    # --------------------------------------------------------

    for (
        kpi_name,
        filename,
    ) in FILES.items():

        file_path = (
            RAW_DIR
            / filename
        )

        (
            time,
            beam_names,
            values,
        ) = load_kpi_file(
            kpi_name,
            file_path,
        )

        if reference_time is None:
            reference_time = time
            reference_beams = (
                beam_names
            )

        else:
            if not np.array_equal(
                time,
                reference_time,
            ):
                raise ValueError(
                    f"{kpi_name}: "
                    f"Time vector does "
                    f"not match reference"
                )

            if (
                beam_names
                != reference_beams
            ):
                raise ValueError(
                    f"{kpi_name}: "
                    f"beam structure does "
                    f"not match reference"
                )

        loaded_data[
            kpi_name
        ] = values

    print(
        "\nAll four KPI files "
        "have matching structure."
    )

    # --------------------------------------------------------
    # 2. Convert wide data to long format
    # --------------------------------------------------------

    processed = build_long_table(
        reference_time,
        reference_beams,
        loaded_data,
    )

    print(
        "\nLong-format dataset created."
    )
    print(
        f"Shape: "
        f"{processed.shape}"
    )

    # Free four large wide matrices once
    # the long table has been constructed.
    del loaded_data

    # --------------------------------------------------------
    # 3. Chronological train/test split
    # --------------------------------------------------------

    processed = (
        add_train_test_split(
            processed
        )
    )

    print(
        "\nTrain/test split:"
    )
    print(
        processed[
            "split"
        ].value_counts()
    )

    # --------------------------------------------------------
    # 4. Standardise KPI columns
    # --------------------------------------------------------

    (
        processed,
        scaler,
    ) = standardise_kpis(
        processed
    )

    print(
        "\nStandardisation completed."
    )

    print(
        "\nScaler mean:"
    )

    for kpi, mean in zip(
        KPI_COLUMNS,
        scaler.mean_,
    ):
        print(
            f"  {kpi}: "
            f"{mean:.6f}"
        )

    print(
        "\nScaler standard deviation:"
    )

    for kpi, scale in zip(
        KPI_COLUMNS,
        scaler.scale_,
    ):
        print(
            f"  {kpi}: "
            f"{scale:.6f}"
        )

    # --------------------------------------------------------
    # 5. Final validation
    # --------------------------------------------------------

    validate_processed_data(
        processed
    )

    train_mask = (
        processed["split"]
        == "train"
    )

    z_columns = [
        f"{kpi}_z"
        for kpi in KPI_COLUMNS
    ]

    print(
        "\nTraining-set "
        "standardised mean:"
    )

    print(
        processed.loc[
            train_mask,
            z_columns,
        ].mean()
    )

    print(
        "\nTraining-set "
        "standardised std:"
    )

    print(
        processed.loc[
            train_mask,
            z_columns,
        ].std(ddof=0)
    )

    # --------------------------------------------------------
    # 6. Save processed CSV
    # --------------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nSaving processed dataset..."
    )
    print(
        f"Output: {OUTPUT_FILE}"
    )

    processed.to_csv(
        OUTPUT_FILE,
        index=False,
        float_format="%.8g",
    )

    # --------------------------------------------------------
    # 7. Save preprocessing metadata
    # --------------------------------------------------------

    save_preprocessing_summary(
        processed,
        scaler,
    )

    print(
        f"Summary: {SUMMARY_FILE}"
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Preprocessing completed successfully."
    )

    print(
        f"Rows:       "
        f"{len(processed):,}"
    )

    print(
        f"Train rows: "
        f"{train_mask.sum():,}"
    )

    print(
        f"Test rows:  "
        f"{(~train_mask).sum():,}"
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
            "Preprocess gotsf-ds "
            "5G KPI data."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Rebuild the processed "
            "dataset even if cached "
            "outputs are up to date."
        ),
    )

    args = parser.parse_args()

    main(
        force=args.force
    )