from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
# Cache check
# ============================================================

def outputs_are_up_to_date():
    """
    Skip preprocessing if:
    - processed CSV exists
    - preprocessing summary exists
    - neither raw data nor this script has changed
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


# ============================================================
# Raw data loading and validation
# ============================================================

def load_kpi_file(kpi_name, file_path):
    """
    Load and validate one raw KPI CSV file.

    Returns
    -------
    time_values : np.ndarray
        Hour indices.
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

    # Remove accidental whitespace
    df.columns = [
        str(col).strip()
        for col in df.columns
    ]

    # --------------------------------------------------------
    # Recover Time column
    #
    # gotsf-ds stores the time index as an unnamed first column.
    # Pandas reads it as "Unnamed: 0".
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
            f"{file_path.name}: duplicate Time values detected"
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
                f"invalid beam identifier '{beam_name}'"
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
        f"{time_values[0]} -> {time_values[-1]}"
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


# ============================================================
# Beam parsing
# ============================================================

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


# ============================================================
# Wide-to-long conversion
# ============================================================

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


# ============================================================
# Chronological train/test split
# ============================================================

def add_train_test_split(processed):
    """
    Chronological split:

    Time 0-671   -> train
    Time 672-839 -> test
    """

    processed["split"] = np.where(
        processed["Time"] < TRAIN_END_TIME,
        "train",
        "test",
    )

    return processed


# ============================================================
# Standardisation + normalisation
# ============================================================

def scale_kpis(processed):
    """
    Generate two alternative scaled representations:

    1. StandardScaler
       mean ~= 0
       standard deviation ~= 1

       Output suffix: _z

    2. MinMaxScaler
       training range -> [0, 1]

       Output suffix: _mm

    IMPORTANT:
    Both scalers are fitted ONLY on training data.
    """

    train_mask = (
        processed["split"] == "train"
    )

    train_data = processed.loc[
        train_mask,
        KPI_COLUMNS,
    ]

    # --------------------------------------------------------
    # Standardisation
    # --------------------------------------------------------

    standard_scaler = StandardScaler()

    standard_scaler.fit(
        train_data
    )

    standard_values = (
        standard_scaler
        .transform(
            processed[KPI_COLUMNS]
        )
        .astype(np.float32)
    )

    for index, kpi_name in enumerate(
        KPI_COLUMNS
    ):
        processed[
            f"{kpi_name}_z"
        ] = standard_values[:, index]

    # --------------------------------------------------------
    # Min-Max normalisation
    # --------------------------------------------------------

    minmax_scaler = MinMaxScaler()

    minmax_scaler.fit(
        train_data
    )

    minmax_values = (
        minmax_scaler
        .transform(
            processed[KPI_COLUMNS]
        )
        .astype(np.float32)
    )

    for index, kpi_name in enumerate(
        KPI_COLUMNS
    ):
        processed[
            f"{kpi_name}_mm"
        ] = minmax_values[:, index]

    return (
        processed,
        standard_scaler,
        minmax_scaler,
    )


# ============================================================
# Final validation
# ============================================================

def validate_processed_data(
    processed,
    expected_rows,
):
    """
    Validate final processed dataset.
    """

    if len(processed) != expected_rows:
        raise ValueError(
            "Unexpected processed row count: "
            f"{len(processed):,} "
            f"(expected {expected_rows:,})"
        )

    if processed.isna().any().any():
        raise ValueError(
            "Processed dataset contains missing values"
        )

    numeric_columns = (
        KPI_COLUMNS
        + [
            f"{kpi}_z"
            for kpi in KPI_COLUMNS
        ]
        + [
            f"{kpi}_mm"
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


# ============================================================
# Save preprocessing summary
# ============================================================

def save_preprocessing_summary(
    processed,
    standard_scaler,
    minmax_scaler,
):
    """
    Save preprocessing metadata and scaler parameters.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_mask = (
        processed["split"] == "train"
    )

    test_mask = (
        processed["split"] == "test"
    )

    z_columns = [
        f"{kpi}_z"
        for kpi in KPI_COLUMNS
    ]

    mm_columns = [
        f"{kpi}_mm"
        for kpi in KPI_COLUMNS
    ]

    # --------------------------------------------------------
    # StandardScaler validation statistics
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MinMaxScaler validation statistics
    # --------------------------------------------------------

    train_mm_min = (
        processed.loc[
            train_mask,
            mm_columns,
        ]
        .min()
    )

    train_mm_max = (
        processed.loc[
            train_mask,
            mm_columns,
        ]
        .max()
    )

    test_mm_min = (
        processed.loc[
            test_mask,
            mm_columns,
        ]
        .min()
    )

    test_mm_max = (
        processed.loc[
            test_mask,
            mm_columns,
        ]
        .max()
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "dataset": {
            "rows": int(
                len(processed)
            ),
            "columns": int(
                len(processed.columns)
            ),
            "time_min": int(
                processed["Time"].min()
            ),
            "time_max": int(
                processed["Time"].max()
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
            "method": "chronological",

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
            "method": "StandardScaler",

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
                    standard_scaler.mean_,
                )
            },

            "scale": {
                kpi: float(scale)
                for kpi, scale
                in zip(
                    KPI_COLUMNS,
                    standard_scaler.scale_,
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

        "normalisation": {
            "method": "MinMaxScaler",

            "fit_scope": (
                "training data only"
            ),

            "feature_range": [
                0.0,
                1.0,
            ],

            "kpi_columns": (
                KPI_COLUMNS
            ),

            "training_data_min": {
                kpi: float(value)
                for kpi, value
                in zip(
                    KPI_COLUMNS,
                    minmax_scaler.data_min_,
                )
            },

            "training_data_max": {
                kpi: float(value)
                for kpi, value
                in zip(
                    KPI_COLUMNS,
                    minmax_scaler.data_max_,
                )
            },

            "train_normalised_min": {
                column: float(value)
                for column, value
                in train_mm_min.items()
            },

            "train_normalised_max": {
                column: float(value)
                for column, value
                in train_mm_max.items()
            },

            "test_normalised_min": {
                column: float(value)
                for column, value
                in test_mm_min.items()
            },

            "test_normalised_max": {
                column: float(value)
                for column, value
                in test_mm_max.items()
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
# Main pipeline
# ============================================================

def main(force=False):

    # --------------------------------------------------------
    # Skip expensive processing when cached outputs are valid
    # --------------------------------------------------------

    if (
        not force
        and outputs_are_up_to_date()
    ):
        print(
            "Processed dataset is already up to date."
        )
        print(
            f"CSV: {OUTPUT_FILE}"
        )
        print(
            f"Summary: {SUMMARY_FILE}"
        )
        print(
            "Use --force to rebuild."
        )
        return

    print("=" * 70)
    print("5G KPI preprocessing")
    print("=" * 70)

    loaded_data = {}

    reference_time = None
    reference_beams = None

    # --------------------------------------------------------
    # 1. Load and validate all KPI files
    # --------------------------------------------------------

    for (
        kpi_name,
        filename,
    ) in FILES.items():

        file_path = (
            RAW_DIR / filename
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
                    f"Time vector does not "
                    f"match reference"
                )

            if (
                beam_names
                != reference_beams
            ):
                raise ValueError(
                    f"{kpi_name}: "
                    f"beam structure does not "
                    f"match reference"
                )

        loaded_data[
            kpi_name
        ] = values

    print(
        "\nAll four KPI files "
        "have matching structure."
    )

    n_times = len(
        reference_time
    )

    n_beams = len(
        reference_beams
    )

    expected_rows = (
        n_times * n_beams
    )

    # --------------------------------------------------------
    # 2. Wide -> long
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
        f"Shape: {processed.shape}"
    )

    print(
        f"Expected rows: "
        f"{expected_rows:,}"
    )

    # Wide matrices are no longer needed
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
    # 4. Standardisation + normalisation
    # --------------------------------------------------------

    (
        processed,
        standard_scaler,
        minmax_scaler,
    ) = scale_kpis(
        processed
    )

    print(
        "\nStandardisation and "
        "normalisation completed."
    )

    # --------------------------------------------------------
    # StandardScaler diagnostics
    # --------------------------------------------------------

    print(
        "\nStandardScaler mean:"
    )

    for kpi, mean in zip(
        KPI_COLUMNS,
        standard_scaler.mean_,
    ):
        print(
            f"  {kpi}: "
            f"{mean:.6f}"
        )

    print(
        "\nStandardScaler "
        "standard deviation:"
    )

    for kpi, scale in zip(
        KPI_COLUMNS,
        standard_scaler.scale_,
    ):
        print(
            f"  {kpi}: "
            f"{scale:.6f}"
        )

    # --------------------------------------------------------
    # MinMaxScaler diagnostics
    # --------------------------------------------------------

    print(
        "\nMinMaxScaler training min/max:"
    )

    for (
        kpi,
        minimum,
        maximum,
    ) in zip(
        KPI_COLUMNS,
        minmax_scaler.data_min_,
        minmax_scaler.data_max_,
    ):
        print(
            f"  {kpi}: "
            f"{minimum:.6f} "
            f"-> {maximum:.6f}"
        )

    # --------------------------------------------------------
    # 5. Final validation
    # --------------------------------------------------------

    validate_processed_data(
        processed,
        expected_rows,
    )

    train_mask = (
        processed["split"]
        == "train"
    )

    z_columns = [
        f"{kpi}_z"
        for kpi in KPI_COLUMNS
    ]

    mm_columns = [
        f"{kpi}_mm"
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

    print(
        "\nTraining-set "
        "normalised min:"
    )

    print(
        processed.loc[
            train_mask,
            mm_columns,
        ].min()
    )

    print(
        "\nTraining-set "
        "normalised max:"
    )

    print(
        processed.loc[
            train_mask,
            mm_columns,
        ].max()
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
    # 7. Save preprocessing summary
    # --------------------------------------------------------

    save_preprocessing_summary(
        processed,
        standard_scaler,
        minmax_scaler,
    )

    print(
        f"Summary: {SUMMARY_FILE}"
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "Preprocessing completed successfully."
    )

    print(
        f"Rows:       "
        f"{len(processed):,}"
    )

    print(
        f"Columns:    "
        f"{len(processed.columns)}"
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
            "Rebuild the processed dataset "
            "even if cached outputs are "
            "already up to date."
        ),
    )

    args = parser.parse_args()

    main(
        force=args.force
    )