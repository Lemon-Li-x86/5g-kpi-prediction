"""Audit raw gotsf-ds KPI CSVs and save the observed data-quality report.

Reports describe source structure, time coverage, missing values and beam IDs.
This command reads raw inputs without changing them.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


from src.core.config import RAW_DATA, REPORTS, DATASET_FILES

RAW_DIR = RAW_DATA
REPORT_DIR = REPORTS
FILES = DATASET_FILES


def audit_file(kpi_name: str, file_path: Path):
    print(f"\n{'=' * 70}")
    print(f"Auditing: {kpi_name}")
    print(f"File: {file_path.name}")

    df = pd.read_csv(file_path)

    # Remove accidental whitespace from column names
    df.columns = [str(col).strip() for col in df.columns]

    # The gotsf-ds CSV files store the time index as an unnamed first column.
    # Pandas therefore reads it as "Unnamed: 0".
    if "Time" not in df.columns:
        first_col = df.columns[0]

        if first_col.startswith("Unnamed:"):
            candidate_time = pd.to_numeric(df[first_col], errors="coerce")

            # Validate that the unnamed column behaves like a time index
            if candidate_time.notna().all():
                diffs = np.diff(candidate_time.to_numpy(dtype=float))

                if len(diffs) == 0 or np.all(diffs == 1):
                    df = df.rename(columns={first_col: "Time"})
                    print(
                        f"Detected unnamed time index column "
                        f"'{first_col}' -> renamed to 'Time'"
                    )
                else:
                    raise ValueError(
                        f"{file_path.name}: first unnamed column is not "
                        f"a continuous time index"
                    )
            else:
                raise ValueError(
                    f"{file_path.name}: first unnamed column contains "
                    f"non-numeric values"
                )
        else:
            raise ValueError(
                f"{file_path.name}: missing required 'Time' column"
            )

    time = df["Time"]
    beam_df = df.drop(columns=["Time"])

    # Convert KPI values to numeric.
    # Any non-numeric content becomes NaN so that it is visible in the audit.
    numeric_df = beam_df.apply(pd.to_numeric, errors="coerce")
    values = numeric_df.to_numpy(dtype=float)

    finite_mask = np.isfinite(values)
    finite_values = values[finite_mask]

    missing_count = int(np.isnan(values).sum())
    inf_count = int(np.isinf(values).sum())
    negative_count = int((values < 0).sum())
    zero_count = int((values == 0).sum())

    total_kpi_values = values.size

    beam_names = beam_df.columns.tolist()

    valid_beam_names = []
    invalid_beam_names = []

    base_stations = set()
    cells = set()
    beams = set()

    for name in beam_names:
        parts = name.split("_")

        if len(parts) == 3 and all(part.isdigit() for part in parts):
            bs, cell, beam = map(int, parts)

            valid_beam_names.append(name)
            base_stations.add(bs)
            cells.add((bs, cell))
            beams.add((bs, cell, beam))
        else:
            invalid_beam_names.append(name)

    unique_time = pd.Series(time.dropna().unique()).sort_values()

    if len(unique_time) > 1:
        diffs = np.diff(unique_time.to_numpy(dtype=float))
        min_time_step = float(np.min(diffs))
        max_time_step = float(np.max(diffs))
        non_unit_gaps = int(np.sum(diffs != 1))
    else:
        min_time_step = None
        max_time_step = None
        non_unit_gaps = None

    if finite_values.size > 0:
        statistics = {
            "min": float(np.min(finite_values)),
            "max": float(np.max(finite_values)),
            "mean": float(np.mean(finite_values)),
            "std": float(np.std(finite_values)),
            "p01": float(np.percentile(finite_values, 1)),
            "p25": float(np.percentile(finite_values, 25)),
            "median": float(np.percentile(finite_values, 50)),
            "p75": float(np.percentile(finite_values, 75)),
            "p99": float(np.percentile(finite_values, 99)),
        }
    else:
        statistics = {
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "p01": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p99": None,
        }

    result = {
        "kpi": kpi_name,
        "file": file_path.name,
        "rows": int(df.shape[0]),
        "columns_total": int(df.shape[1]),
        "beam_columns": int(beam_df.shape[1]),
        "base_station_count": len(base_stations),
        "cell_count": len(cells),
        "beam_count": len(beams),
        "invalid_beam_column_count": len(invalid_beam_names),
        "invalid_beam_columns": invalid_beam_names,
        "time_min": (
            float(time.min())
            if pd.api.types.is_numeric_dtype(time)
            else str(time.min())
        ),
        "time_max": (
            float(time.max())
            if pd.api.types.is_numeric_dtype(time)
            else str(time.max())
        ),
        "unique_time_count": int(time.nunique(dropna=True)),
        "time_monotonic_increasing": bool(time.is_monotonic_increasing),
        "duplicate_time_count": int(time.duplicated().sum()),
        "duplicate_full_row_count": int(df.duplicated().sum()),
        "min_time_step": min_time_step,
        "max_time_step": max_time_step,
        "non_unit_time_gap_count": non_unit_gaps,
        "total_kpi_values": int(total_kpi_values),
        "missing_values": missing_count,
        "missing_rate_percent": (
            100 * missing_count / total_kpi_values
            if total_kpi_values
            else None
        ),
        "infinite_values": inf_count,
        "negative_values": negative_count,
        "zero_values": zero_count,
        "zero_rate_percent": (
            100 * zero_count / total_kpi_values
            if total_kpi_values
            else None
        ),
        **statistics,
    }

    print(f"Shape:                  {df.shape}")
    print(f"Beam columns:           {result['beam_columns']}")
    print(f"Base stations:          {result['base_station_count']}")
    print(f"Cells:                  {result['cell_count']}")
    print(f"Time range:             {result['time_min']} -> {result['time_max']}")
    print(f"Unique Time values:     {result['unique_time_count']}")
    print(f"Duplicate Time values:  {result['duplicate_time_count']}")
    print(f"Duplicate full rows:    {result['duplicate_full_row_count']}")
    print(f"Non-unit time gaps:     {result['non_unit_time_gap_count']}")
    print(f"Missing values:         {result['missing_values']}")
    print(f"Infinite values:        {result['infinite_values']}")
    print(f"Negative values:        {result['negative_values']}")
    print(f"Zero values:            {result['zero_values']}")
    print(f"Min / Median / Max:     "
          f"{result['min']} / {result['median']} / {result['max']}")

    return result, df["Time"].copy(), df.columns.tolist()


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []
    time_vectors = {}
    column_sets = {}

    for kpi_name, filename in FILES.items():
        file_path = RAW_DIR / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Raw data file not found: {file_path}"
            )

        result, time_vector, columns = audit_file(
            kpi_name,
            file_path
        )

        summaries.append(result)
        time_vectors[kpi_name] = time_vector
        column_sets[kpi_name] = columns

    reference_kpi = next(iter(FILES))

    same_columns = {
        kpi: column_sets[kpi] == column_sets[reference_kpi]
        for kpi in FILES
    }

    same_time_vector = {
        kpi: time_vectors[kpi].equals(time_vectors[reference_kpi])
        for kpi in FILES
    }

    cross_file_checks = {
        "reference_kpi": reference_kpi,
        "same_columns_as_reference": same_columns,
        "same_time_vector_as_reference": same_time_vector,
    }

    summary_df = pd.DataFrame(summaries)

    csv_path = REPORT_DIR / "raw_data_audit_summary.csv"
    json_path = REPORT_DIR / "raw_data_audit.json"

    summary_df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "files": summaries,
                "cross_file_checks": cross_file_checks,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n{'=' * 70}")
    print("Cross-file checks")
    print("Same columns:")
    for kpi, status in same_columns.items():
        print(f"  {kpi}: {status}")

    print("Same Time vector:")
    for kpi, status in same_time_vector.items():
        print(f"  {kpi}: {status}")

    print("\nAudit completed.")
    print(f"CSV report:  {csv_path}")
    print(f"JSON report: {json_path}")


if __name__ == "__main__":
    main()