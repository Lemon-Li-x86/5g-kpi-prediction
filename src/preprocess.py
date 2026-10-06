from pathlib import Path
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

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

TRAIN_END_TIME = 672


def load_kpi_file(kpi_name, file_path):
    print(f"\nLoading {kpi_name}...")

    df = pd.read_csv(file_path)

    # Remove accidental whitespace from column names
    df.columns = [str(col).strip() for col in df.columns]

    # The dataset stores Time as an unnamed first column
    if "Time" not in df.columns:
        first_col = df.columns[0]

        if first_col.startswith("Unnamed:"):
            df = df.rename(columns={first_col: "Time"})
        else:
            raise ValueError(
                f"{file_path.name}: cannot identify Time column"
            )

    # Check Time column
    time = pd.to_numeric(df["Time"], errors="coerce")

    if time.isna().any():
        raise ValueError(
            f"{file_path.name}: invalid Time values"
        )

    if time.duplicated().any():
        raise ValueError(
            f"{file_path.name}: duplicate Time values"
        )

    time_values = time.to_numpy()

    if len(time_values) > 1:
        time_diff = np.diff(time_values)

        if not np.all(time_diff == 1):
            raise ValueError(
                f"{file_path.name}: Time is not continuous"
            )

    # KPI values
    beam_df = df.drop(columns=["Time"])

    values = (
        beam_df
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32)
    )

    if np.isnan(values).any():
        raise ValueError(
            f"{file_path.name}: missing or invalid KPI values"
        )

    if np.isinf(values).any():
        raise ValueError(
            f"{file_path.name}: infinite KPI values"
        )

    if (values < 0).any():
        raise ValueError(
            f"{file_path.name}: negative KPI values"
        )

    print(f"Shape: {df.shape}")
    print(f"Time: {time_values[0]} -> {time_values[-1]}")
    print(f"Beam count: {beam_df.shape[1]}")

    return time_values, beam_df.columns.tolist(), values

def parse_beam_names(beam_names):
    """
    Convert beam names such as '12_2_31' into
    base_station, cell and beam identifiers.
    """

    base_stations = []
    cells = []
    beams = []

    for name in beam_names:
        parts = name.split("_")

        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            raise ValueError(f"Invalid beam name: {name}")

        base_station, cell, beam = map(int, parts)

        base_stations.append(base_station)
        cells.append(cell)
        beams.append(beam)

    return (
        np.array(base_stations),
        np.array(cells),
        np.array(beams),
    )


def main():
    loaded_data = {}

    reference_time = None
    reference_beams = None

    # 1. Load all four KPI files
    for kpi_name, filename in FILES.items():
        file_path = RAW_DIR / filename

        time, beam_names, values = load_kpi_file(
            kpi_name,
            file_path
        )

        # Use the first KPI as the reference structure
        if reference_time is None:
            reference_time = time
            reference_beams = beam_names
        else:
            if not np.array_equal(time, reference_time):
                raise ValueError(
                    f"{kpi_name}: Time vector does not match"
                )

            if beam_names != reference_beams:
                raise ValueError(
                    f"{kpi_name}: beam columns do not match"
                )

        loaded_data[kpi_name] = values

    print("\nAll four KPI files have matching structure.")

    # 2. Parse beam identifiers
    base_station_ids, cell_ids, beam_ids = parse_beam_names(
        reference_beams
    )

    n_times = len(reference_time)
    n_beams = len(reference_beams)

    # 3. Build long-format table
    processed = pd.DataFrame({
        "Time": np.repeat(reference_time, n_beams),

        "base_station": np.tile(
            base_station_ids,
            n_times
        ),

        "cell": np.tile(
            cell_ids,
            n_times
        ),

        "beam": np.tile(
            beam_ids,
            n_times
        ),
    })

    # 4. Add the four KPI values
    for kpi_name, values in loaded_data.items():
        processed[kpi_name] = values.reshape(-1)
    # 5. Time-based train/test split
        processed["split"] = np.where(
        processed["Time"] < TRAIN_END_TIME,
        "train",
        "test"
    )

    # 6. Standardise KPI values
    #
    # Fit the scaler using TRAINING data only.
    # This prevents information leakage from the test period.

    train_mask = processed["split"] == "train"

    scaler = StandardScaler()

    scaler.fit(
        processed.loc[
            train_mask,
            KPI_COLUMNS
        ]
    )

    scaled_values = scaler.transform(
        processed[KPI_COLUMNS]
    )

    for i, kpi_name in enumerate(KPI_COLUMNS):
        processed[f"{kpi_name}_z"] = scaled_values[:, i]

    print("\nStandardisation completed.")

    print("\nScaler mean:")
    for kpi_name, mean in zip(KPI_COLUMNS, scaler.mean_):
        print(f"{kpi_name}: {mean:.6f}")

    print("\nScaler standard deviation:")
    for kpi_name, scale in zip(KPI_COLUMNS, scaler.scale_):
        print(f"{kpi_name}: {scale:.6f}")



    print("\nTrain/test split created:")
    print(processed["split"].value_counts())

    print("\nLong-format dataset created.")
    print("Shape:", processed.shape)

    print("\nFirst 10 rows:")
    print(processed.head(10))

    print("\nLast 5 rows:")
    print(processed.tail())

    # 5. Save processed dataset
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DIR / "kpi_timeseries.csv"

    print("\nSaving processed dataset...")
    processed.to_csv(
        output_path,
        index=False,
        float_format="%.8g"
    )

    print(f"Saved to: {output_path}")

    print(
        processed.loc[
            processed["split"] == "train",
            ["DLPRB_z", "DLThpVol_z", "DLThpTime_z", "MR_number_z"]
        ].mean()
    )


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()