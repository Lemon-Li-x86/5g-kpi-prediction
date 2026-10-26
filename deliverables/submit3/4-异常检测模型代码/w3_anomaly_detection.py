
#!/usr/bin/env python3
"""
Condensed W3 / Original Week 6:
5G KPI anomaly detection and preliminary root-cause analysis.

Methods:
    1. Three-sigma rule
    2. IQR
    3. Isolation Forest
    4. One-Class SVM

Note:
    The dataset contains no verified network-fault labels.
    Detected anomalies are candidates, not confirmed faults.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


# ============================================================
# 1. Configuration
# ============================================================

KPIS = [
    "DLPRB",
    "DLThpVol",
    "DLThpTime",
    "MR_number",
]

Z = [f"{k}_z" for k in KPIS]

KEYS = [
    "base_station",
    "cell",
    "beam",
]


# ============================================================
# 2. Load KPI data
# ============================================================

def load_data(path: Path):

    cols = [
        "Time",
        *KEYS,
        *KPIS,
        *Z,
    ]

    df = pd.read_csv(
        path,
        usecols=cols,
    )

    df = df.sort_values(
        KEYS + ["Time"],
        kind="stable",
    )

    counts = (
        df.groupby(
            KEYS,
            sort=False,
            dropna=False,
        )
        .size()
        .to_numpy()
    )

    if not len(counts) or not np.all(counts == counts[0]):
        raise ValueError(
            "Beam/time data must form an equal-length panel"
        )

    beams = len(counts)
    nt = int(counts[0])

    tgrid = (
        df["Time"]
        .to_numpy()
        .reshape(beams, nt)
    )

    timeline = tgrid[0].copy()

    if not np.array_equal(
        tgrid,
        np.broadcast_to(timeline, tgrid.shape),
    ):
        raise ValueError(
            "Misaligned beam time series"
        )

    values = (
        df[Z]
        .to_numpy(dtype=np.float32)
        .reshape(beams, nt, 4)
    )

    raw = (
        df[KPIS]
        .to_numpy(dtype=np.float32)
        .reshape(beams, nt, 4)
    )

    keys = (
        df[KEYS]
        .iloc[::nt]
        .reset_index(drop=True)
    )

    if (
        not np.isfinite(values).all()
        or not np.isfinite(raw).all()
    ):
        raise ValueError(
            "NaN or infinite values in KPI data"
        )

    return values, raw, keys, timeline


# ============================================================
# 3. Statistical anomaly detection
# ============================================================

def detection_thresholds(train, test):

    # Learn thresholds independently for each beam,
    # using training hours only.

    mean = train.mean(
        axis=1,
        keepdims=True,
    )

    std = train.std(
        axis=1,
        keepdims=True,
    )

    beam_z = (
        (test - mean)
        / np.maximum(std, 1e-6)
    )

    # Three-sigma rule
    sigma = np.abs(beam_z) > 3.0

    # IQR method
    q1, q3 = np.quantile(
        train,
        [0.25, 0.75],
        axis=1,
    )

    q1 = q1[:, None, :]
    q3 = q3[:, None, :]

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    # Handle zero-inflated or constant-IQR distributions.
    flat = iqr < 1e-8

    p1, p99 = np.quantile(
        train,
        [0.01, 0.99],
        axis=1,
    )

    lower = np.where(
        flat,
        p1[:, None, :] - 1e-6,
        lower,
    )

    upper = np.where(
        flat,
        p99[:, None, :] + 1e-6,
        upper,
    )

    outlier_iqr = (
        (test < lower)
        | (test > upper)
    )

    return (
        sigma,
        outlier_iqr,
        beam_z,
        int(flat.sum()),
    )


# ============================================================
# 4. Machine-learning feature construction
# ============================================================

def features(values):

    # Current KPI values and first differences.
    # Differences use the current and previous hour only.

    change = np.diff(
        values,
        axis=1,
        prepend=values[:, :1, :],
    )

    return np.concatenate(
        [values, change],
        axis=2,
    ).astype(np.float32)


def sampled_fit(x, limit, seed):

    rng = np.random.default_rng(seed)

    if 0 < limit < len(x):

        return x[
            rng.choice(
                len(x),
                size=limit,
                replace=False,
            )
        ]

    return x


# ============================================================
# 5. Main anomaly-detection pipeline
# ============================================================

def main():

    p = argparse.ArgumentParser(
        description=__doc__
    )

    p.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "data/processed/kpi_timeseries.csv"
        ),
    )

    p.add_argument(
        "--out",
        type=Path,
        default=Path("."),
    )

    p.add_argument(
        "--test-start",
        type=int,
        default=672,
    )

    p.add_argument(
        "--if-train-sample",
        type=int,
        default=80000,
    )

    p.add_argument(
        "--if-contamination",
        type=float,
        default=0.02,
    )

    p.add_argument(
        "--svm-train-sample",
        type=int,
        default=2500,
    )

    p.add_argument(
        "--svm-test-sample",
        type=int,
        default=12000,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--max-events",
        type=int,
        default=2000,
    )

    args = p.parse_args()

    if not 0 < args.if_contamination <= 0.5:
        p.error(
            "--if-contamination must lie in (0, 0.5]"
        )

    # Output directories
    reports = args.out / "reports"
    figures = args.out / "figures"

    reports.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load and split data
    # --------------------------------------------------------

    values, raw, keys, timeline = load_data(
        args.csv
    )

    train_cols = np.flatnonzero(
        timeline < args.test_start
    )

    test_cols = np.flatnonzero(
        timeline >= args.test_start
    )

    if len(train_cols) < 10 or len(test_cols) < 1:
        raise ValueError(
            "Need >=10 training hours and >=1 test hour"
        )

    if not np.array_equal(
        train_cols,
        np.arange(train_cols[-1] + 1),
    ):
        raise ValueError(
            "Training hours must form a chronological prefix"
        )

    train = values[:, train_cols, :]
    test = values[:, test_cols, :]

    n_beams, nt_test, _ = test.shape

    print("=" * 65)
    print("W3: 5G KPI anomaly detection")
    print("=" * 65)

    print(f"Beams: {n_beams:,}")
    print(f"Training hours: {len(train_cols)}")
    print(f"Testing hours: {len(test_cols)}")
    print(f"Test observations: {n_beams * nt_test:,}")

    # --------------------------------------------------------
    # Statistical detection: 3sigma and IQR
    # --------------------------------------------------------

    print("\n[1/4] Statistical anomaly detection")

    (
        sigma_kpi,
        iqr_kpi,
        beam_z,
        fallback_count,
    ) = detection_thresholds(
        train,
        test,
    )

    sigma = (
        sigma_kpi
        .any(axis=2)
        .reshape(-1)
    )

    iqr = (
        iqr_kpi
        .any(axis=2)
        .reshape(-1)
    )

    # --------------------------------------------------------
    # Construct 8-dimensional ML features
    # --------------------------------------------------------

    f = features(values)

    f_train = (
        f[:, train_cols, :]
        .reshape(-1, 8)
    )

    f_test = (
        f[:, test_cols, :]
        .reshape(-1, 8)
    )

    # --------------------------------------------------------
    # Isolation Forest
    # --------------------------------------------------------

    print("\n[2/4] Isolation Forest")

    if_train = sampled_fit(
        f_train,
        args.if_train_sample,
        args.seed,
    )

    print(
        f"Training samples: {len(if_train):,}"
    )

    print(
        f"Testing samples: {len(f_test):,}"
    )

    forest = IsolationForest(
        n_estimators=150,
        max_samples=min(256, len(if_train)),
        contamination=args.if_contamination,
        random_state=args.seed,
        n_jobs=-1,
    )

    forest.fit(if_train)

    if_score = forest.decision_function(
        f_test
    )

    if_flag = if_score < 0

    # --------------------------------------------------------
    # One-Class SVM
    # --------------------------------------------------------

    print("\n[3/4] One-Class SVM")

    rng = np.random.default_rng(
        args.seed
    )

    svm_train = sampled_fit(
        f_train,
        args.svm_train_sample,
        args.seed + 1,
    )

    # The RBF kernel is computationally expensive.
    # Evaluate an explicitly sampled test subset.

    if args.svm_test_sample > 0:

        svm_idx = np.sort(
            rng.choice(
                len(f_test),
                size=min(
                    args.svm_test_sample,
                    len(f_test),
                ),
                replace=False,
            )
        )

    else:

        svm_idx = np.arange(
            len(f_test)
        )

    print(
        f"Training samples: {len(svm_train):,}"
    )

    print(
        f"Testing subset: {len(svm_idx):,}"
    )

    svm = OneClassSVM(
        nu=0.02,
        gamma="scale",
        kernel="rbf",
        cache_size=500,
    )

    svm.fit(svm_train)

    svm_flag = (
        svm.predict(
            f_test[svm_idx]
        ) == -1
    )

    # Save exact SVM test subset
    svm_pd = pd.DataFrame({
        "beam_index": svm_idx // nt_test,
        "Time": (
            timeline[test_cols][
                svm_idx % nt_test
            ]
        ),
        "ocsvm_flag": svm_flag,
        "isolation_forest_flag": if_flag[svm_idx],
        "sigma_flag": sigma[svm_idx],
        "iqr_flag": iqr[svm_idx],
    })

    svm_pd.to_csv(
        reports / "w3_one_class_svm_test_subset.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Compare detection methods
    # --------------------------------------------------------

    print("\n[4/4] Results and KPI association")

    rates = []

    method_results = [
        ("3sigma", sigma),
        ("IQR", iqr),
        ("IsolationForest", if_flag),
        ("OneClassSVM", svm_flag),
    ]

    for name, flags in method_results:

        rates.append({
            "method": name,
            "evaluated_samples": len(flags),
            "flagged_samples": int(flags.sum()),
            "flag_rate": float(flags.mean()),
            "evaluation_scope": (
                "sampled test subset"
                if name == "OneClassSVM"
                else "full test"
            ),
        })

    rates_df = pd.DataFrame(rates)

    rates_df.to_csv(
        reports / "w3_anomaly_method_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Consensus anomaly candidates
    # --------------------------------------------------------

    # A consensus candidate must be detected by IF
    # and at least one statistical method.
    # This is not a confirmed network fault.

    consensus = (
        if_flag
        & (sigma | iqr)
    )

    # --------------------------------------------------------
    # Hourly detection rates
    # --------------------------------------------------------

    times_flat = np.tile(
        timeline[test_cols],
        n_beams,
    )

    hourly = pd.DataFrame({
        "Time": times_flat,
        "3sigma": sigma.astype(int),
        "IQR": iqr.astype(int),
        "IsolationForest": if_flag.astype(int),
        "Consensus": consensus.astype(int),
    })

    hourly = (
        hourly.groupby(
            "Time",
            sort=True,
        )
        .mean()
    )

    hourly.to_csv(
        reports / "w3_anomaly_rate_by_hour.csv"
    )

    fig, ax = plt.subplots(
        figsize=(9, 4)
    )

    for col in hourly.columns:

        ax.plot(
            hourly.index,
            100 * hourly[col],
            label=col,
        )

    ax.set(
        xlabel="Test hour",
        ylabel="Flagged beam-hours (%)",
        title="Unsupervised anomaly screening rates",
    )

    ax.legend(ncol=2)
    fig.tight_layout()

    fig.savefig(
        figures / "w3_anomaly_rate_by_hour.png",
        dpi=160,
    )

    plt.close(fig)

    # --------------------------------------------------------
    # KPI correlation analysis
    # --------------------------------------------------------

    selected_values = (
        test.reshape(-1, 4)[consensus]
    )

    # Fall back to all test observations if
    # there are too few candidate anomalies.

    if len(selected_values) >= 3:

        corr_values = selected_values
        corr_title = "KPI correlations in candidate anomalies"

    else:

        corr_values = test.reshape(-1, 4)
        corr_title = "KPI correlations in all test observations"

    corr = pd.DataFrame(
        corr_values,
        columns=KPIS,
    ).corr()

    corr.to_csv(
        reports / "w3_anomaly_kpi_correlations.csv"
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    im = ax.imshow(
        corr.to_numpy(),
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
    )

    ax.set_xticks(
        range(4),
        KPIS,
        rotation=25,
        ha="right",
    )

    ax.set_yticks(
        range(4),
        KPIS,
    )

    for row in range(4):

        for col in range(4):

            value = corr.iloc[row, col]

            ax.text(
                col,
                row,
                (
                    "N/A"
                    if not np.isfinite(value)
                    else f"{value:.2f}"
                ),
                ha="center",
                va="center",
                fontsize=9,
            )

    fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    ax.set_title(corr_title)

    fig.tight_layout()

    fig.savefig(
        figures / "w3_anomaly_kpi_correlations.png",
        dpi=160,
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Extract suspicious beam-hour events
    # --------------------------------------------------------

    candidates = np.flatnonzero(
        consensus
    )

    # Smaller IF score indicates greater abnormality.

    candidates = candidates[
        np.argsort(
            if_score[candidates]
        )
    ][:max(0, args.max_events)]

    beam_idx = candidates // nt_test
    time_idx = candidates % nt_test

    events = (
        keys.iloc[beam_idx]
        .reset_index(drop=True)
        .copy()
    )

    events.insert(
        0,
        "Time",
        timeline[test_cols][time_idx],
    )

    events["isolation_forest_score"] = (
        if_score[candidates]
    )

    events["sigma_flag"] = sigma[candidates]
    events["iqr_flag"] = iqr[candidates]

    sigma_flat = sigma_kpi.reshape(-1, 4)
    iqr_flat = iqr_kpi.reshape(-1, 4)
    beam_z_flat = beam_z.reshape(-1, 4)
    raw_test_flat = raw[:, test_cols, :].reshape(-1, 4)

    events["n_sigma_kpis"] = (
        sigma_flat[candidates]
        .sum(axis=1)
    )

    events["n_iqr_kpis"] = (
        iqr_flat[candidates]
        .sum(axis=1)
    )

    for j, kpi in enumerate(KPIS):

        events[kpi] = (
            raw_test_flat[candidates, j]
        )

        events[f"{kpi}_beam_z"] = (
            beam_z_flat[candidates, j]
        )

    if len(candidates):

        strongest = np.argmax(
            np.abs(
                beam_z_flat[candidates]
            ),
            axis=1,
        )

        events["top_deviation_kpi"] = [
            KPIS[index]
            for index in strongest
        ]

    else:

        events["top_deviation_kpi"] = []

    events.to_csv(
        reports / "w3_anomaly_candidate_events.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save experiment configuration
    # --------------------------------------------------------

    config = {
        "input": str(args.csv),
        "train_until_exclusive": args.test_start,
        "test_hours": [
            int(timeline[test_cols[0]]),
            int(timeline[test_cols[-1]]),
        ],
        "n_beams": n_beams,
        "test_samples": len(f_test),
        "if_train_sample": len(if_train),
        "svm_train_sample": len(svm_train),
        "svm_test_sample": len(svm_idx),
        "if_contamination": args.if_contamination,
        "consensus_candidates": int(consensus.sum()),
        "seed": args.seed,
    }

    (
        reports / "w3_anomaly_run_config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("Anomaly detection completed")
    print("=" * 65)

    print(
        rates_df.to_string(
            index=False
        )
    )

    print(
        f"\nConsensus anomaly candidates: "
        f"{int(consensus.sum()):,} / {len(consensus):,}"
    )

    print(
        f"\nReports saved to: {reports.resolve()}"
    )

    print(
        f"Figures saved to: {figures.resolve()}"
    )


if __name__ == "__main__":
    main()
