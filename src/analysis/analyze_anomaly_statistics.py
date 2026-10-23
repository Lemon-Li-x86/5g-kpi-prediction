
#!/usr/bin/env python3
"""W4: Statistical diagnostics for 5G KPI anomaly detection.

Run from the repository root:
    python -m src.analysis.analyze_anomaly_statistics

Required:
    src/models/detect_kpi_anomalies.py
    data/processed/kpi_timeseries.csv

Outputs:
    reports/*.csv
    figures/*.png

This script performs data analysis only.
It does not generate an experiment report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import kurtosis, pearsonr, skew, t
from sklearn.ensemble import IsolationForest

from src.core.config import DATASET_TEST_START_HOUR

from src.models.detect_kpi_anomalies import (
    KPIS,
    detection_thresholds,
    features,
    load_data,
    sampled_fit,
)


# ============================================================
# 1. Statistical utilities
# ============================================================

def describe(values):
    """Descriptive statistics for a one-dimensional array."""
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "q01": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "q99": np.nan,
            "zero_fraction": np.nan,
            "skewness": np.nan,
            "excess_kurtosis": np.nan,
        }

    variable = len(x) > 2 and np.std(x) > 0

    return {
        "n": len(x),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
        "median": float(np.median(x)),
        "q01": float(np.quantile(x, 0.01)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
        "q99": float(np.quantile(x, 0.99)),
        "zero_fraction": float(np.mean(np.isclose(x, 0))),
        "skewness": float(skew(x, bias=False)) if variable else np.nan,
        "excess_kurtosis": (
            float(kurtosis(x, fisher=True, bias=False))
            if variable else np.nan
        ),
    }


def holm_adjust(p_values):
    """Holm-Bonferroni correction for multiple hypothesis tests."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(p.shape, np.nan)

    valid = np.flatnonzero(np.isfinite(p))
    if len(valid) == 0:
        return adjusted

    order = valid[np.argsort(p[valid])]
    running_max = 0.0
    m = len(order)

    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running_max = max(running_max, value)
        adjusted[idx] = running_max

    return adjusted


def cluster_robust_slope_test(x, y, clusters):
    """Test zero linear slope using base-station-clustered standard errors."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    clusters = np.asarray(clusters)

    valid = np.isfinite(x) & np.isfinite(y)
    x, y, clusters = x[valid], y[valid], clusters[valid]

    unique_clusters = np.unique(clusters)
    n = len(x)
    g = len(unique_clusters)

    if n < 4 or g < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, g, np.nan

    design = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ beta

    bread = np.linalg.pinv(design.T @ design)
    meat = np.zeros((2, 2))

    for cluster in unique_clusters:
        mask = clusters == cluster
        score = design[mask].T @ residual[mask]
        meat += np.outer(score, score)

    correction = (g / (g - 1)) * (
        (n - 1) / (n - design.shape[1])
    )

    covariance = correction * (bread @ meat @ bread)
    variance = float(covariance[1, 1])

    if not np.isfinite(variance) or variance <= 0:
        return np.nan, g, np.nan

    se = np.sqrt(variance)
    statistic = beta[1] / se
    p_value = float(2 * t.sf(abs(statistic), df=g - 1))

    return p_value, g, float(se)


def cluster_bootstrap_correlation_ci(
    x, y, clusters, iterations, rng
):
    """95% percentile CI for Pearson r using a station-cluster bootstrap."""
    clusters = np.asarray(clusters)
    unique_clusters = np.unique(clusters)

    groups = [
        np.flatnonzero(clusters == cluster)
        for cluster in unique_clusters
    ]

    if len(groups) < 3 or iterations <= 0:
        return np.nan, np.nan

    estimates = []

    for _ in range(iterations):
        selected = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[i] for i in selected])

        xb = x[indices]
        yb = y[indices]

        if np.std(xb) == 0 or np.std(yb) == 0:
            continue

        r = np.corrcoef(xb, yb)[0, 1]

        if np.isfinite(r):
            estimates.append(r)

    if len(estimates) < max(30, iterations // 2):
        return np.nan, np.nan

    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


# ============================================================
# 2. Main analysis
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/processed/kpi_timeseries.csv"),
    )
    parser.add_argument("--out", type=Path, default=Path("."))
    parser.add_argument("--test-start", type=int, default=DATASET_TEST_START_HOUR)
    parser.add_argument("--if-train-sample", type=int, default=80000)
    parser.add_argument("--if-contamination", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iterations", type=int, default=500)

    args = parser.parse_args()

    if args.bootstrap_iterations < 0:
        parser.error("--bootstrap-iterations must be >= 0")

    if not 0 < args.if_contamination <= 0.5:
        parser.error("--if-contamination must be in (0, 0.5]")

    reports = args.out / "reports"
    figures = args.out / "figures"

    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 2.1 Load data and split train/test chronologically
    # --------------------------------------------------------

    values, raw, keys, timeline = load_data(args.csv)

    train_cols = np.flatnonzero(timeline < args.test_start)
    test_cols = np.flatnonzero(timeline >= args.test_start)

    if len(train_cols) < 10 or len(test_cols) == 0:
        raise ValueError("Insufficient training or test hours.")

    if not np.array_equal(
        train_cols, np.arange(train_cols[-1] + 1)
    ):
        raise ValueError("Training data must be a chronological prefix.")

    train = values[:, train_cols, :]
    test = values[:, test_cols, :]

    raw_train = raw[:, train_cols, :]
    raw_test = raw[:, test_cols, :]

    n_beams, n_test_hours, n_kpis = test.shape
    n_test = n_beams * n_test_hours

    # --------------------------------------------------------
    # 2.2 Statistical detectors: 3-sigma and IQR
    # --------------------------------------------------------

    sigma_kpi, iqr_kpi, _, fallback_count = detection_thresholds(
        train, test
    )

    # A beam-hour is flagged if any KPI is flagged.
    sigma = sigma_kpi.any(axis=2).ravel()
    iqr = iqr_kpi.any(axis=2).ravel()

    # --------------------------------------------------------
    # 2.3 Isolation Forest
    # --------------------------------------------------------

    all_features = features(values)

    feature_count = len(KPIS) * 2

    f_train = all_features[:, train_cols, :].reshape(
        -1, feature_count
    )

    f_test = all_features[:, test_cols, :].reshape(
        -1, feature_count
    )

    if_train = sampled_fit(
        f_train,
        args.if_train_sample,
        args.seed,
    )

    forest = IsolationForest(
        n_estimators=150,
        max_samples=min(256, len(if_train)),
        contamination=args.if_contamination,
        random_state=args.seed,
        n_jobs=-1,
    )

    forest.fit(if_train)

    if_score = forest.decision_function(f_test)
    if_flag = if_score < 0

    # Consensus = IF AND (3-sigma OR IQR)
    consensus = if_flag & (sigma | iqr)

    # --------------------------------------------------------
    # 2.4 Export all test-set detector flags
    # --------------------------------------------------------

    raw_test_flat = raw_test.reshape(-1, n_kpis)
    sigma_flat = sigma_kpi.reshape(-1, n_kpis)
    iqr_flat = iqr_kpi.reshape(-1, n_kpis)

    flag_data = {
        "Time": np.tile(timeline[test_cols], n_beams),
        "base_station": np.repeat(
            keys["base_station"].to_numpy(), n_test_hours
        ),
        "cell": np.repeat(
            keys["cell"].to_numpy(), n_test_hours
        ),
        "beam": np.repeat(
            keys["beam"].to_numpy(), n_test_hours
        ),
        "3sigma_any_KPI": sigma.astype(np.uint8),
        "IQR_any_KPI": iqr.astype(np.uint8),
        "IsolationForest": if_flag.astype(np.uint8),
        "Consensus_IF_AND_sigma_OR_IQR": consensus.astype(np.uint8),
    }

    for j, kpi in enumerate(KPIS):
        flag_data[kpi] = raw_test_flat[:, j]
        flag_data[f"3sigma_{kpi}"] = sigma_flat[:, j].astype(np.uint8)
        flag_data[f"IQR_{kpi}"] = iqr_flat[:, j].astype(np.uint8)

    pd.DataFrame(flag_data).to_csv(
        reports / "w4_anomaly_test_flags.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 2.5 Compare AND/OR logic
    # --------------------------------------------------------

    logic_masks = {
        "3sigma": sigma,
        "IQR": iqr,
        "Isolation Forest": if_flag,
        "3sigma OR IQR": sigma | iqr,
        "3sigma AND IQR": sigma & iqr,
        "IF AND 3sigma": if_flag & sigma,
        "IF AND IQR": if_flag & iqr,
        "IF AND (3sigma OR IQR)": consensus,
        "IF AND 3sigma AND IQR": if_flag & sigma & iqr,
        "IF OR 3sigma OR IQR": if_flag | sigma | iqr,
    }

    logic_df = pd.DataFrame([
        {
            "logic": name,
            "evaluated_beam_hours": int(mask.size),
            "flagged_beam_hours": int(mask.sum()),
            "flag_rate": float(mask.mean()),
        }
        for name, mask in logic_masks.items()
    ])

    logic_df.to_csv(
        reports / "w4_anomaly_logic_comparison.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 2.6 Raw KPI descriptive statistics
    # --------------------------------------------------------

    distribution_rows = []

    for j, kpi in enumerate(KPIS):
        for split_name, data in [
            ("train", raw_train[:, :, j]),
            ("test", raw_test[:, :, j]),
        ]:
            distribution_rows.append({
                "KPI": kpi,
                "split": split_name,
                **describe(data),
            })

    pd.DataFrame(distribution_rows).to_csv(
        reports / "w4_anomaly_distribution_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 2.7 Training distribution shape and per-KPI flag rates
    # --------------------------------------------------------

    q1, q3 = np.quantile(train, [0.25, 0.75], axis=1)
    beam_iqr = q3 - q1

    beam_skew = skew(train, axis=1, bias=False)
    beam_kurtosis = kurtosis(
        train, axis=1, fisher=True, bias=False
    )

    shape_rows = []

    for j, kpi in enumerate(KPIS):
        iqr_j = beam_iqr[:, j]
        skew_j = beam_skew[:, j]
        kurt_j = beam_kurtosis[:, j]

        valid_skew = skew_j[np.isfinite(skew_j)]
        valid_kurt = kurt_j[np.isfinite(kurt_j)]

        sigma_j = sigma_kpi[:, :, j]
        iqr_j_flags = iqr_kpi[:, :, j]

        shape_rows.append({
            "KPI": kpi,
            "n_beam_distributions": n_beams,

            "median_training_IQR": float(np.nanmedian(iqr_j)),
            "q25_training_IQR": float(np.nanquantile(iqr_j, 0.25)),
            "q75_training_IQR": float(np.nanquantile(iqr_j, 0.75)),

            "zero_IQR_fallback_distributions": int(
                np.sum(iqr_j < 1e-8)
            ),
            "zero_IQR_fraction": float(
                np.mean(iqr_j < 1e-8)
            ),

            "fraction_abs_skewness_gt_1": (
                float(np.mean(np.abs(valid_skew) > 1))
                if len(valid_skew) else np.nan
            ),

            "fraction_excess_kurtosis_gt_3": (
                float(np.mean(valid_kurt > 3))
                if len(valid_kurt) else np.nan
            ),

            "median_skewness": float(np.nanmedian(skew_j)),
            "median_excess_kurtosis": float(np.nanmedian(kurt_j)),

            "3sigma_flagged_KPI_values": int(sigma_j.sum()),
            "3sigma_KPI_flag_rate": float(sigma_j.mean()),

            "IQR_flagged_KPI_values": int(iqr_j_flags.sum()),
            "IQR_KPI_flag_rate": float(iqr_j_flags.mean()),

            "IQR_only_KPI_flag_rate": float(
                (iqr_j_flags & ~sigma_j).mean()
            ),

            "both_methods_KPI_flag_rate": float(
                (iqr_j_flags & sigma_j).mean()
            ),
        })

    shape_df = pd.DataFrame(shape_rows)

    shape_df.to_csv(
        reports / "w4_anomaly_threshold_shape_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 2.8 KPI correlation and statistical significance
    # --------------------------------------------------------

    candidate_indices = np.flatnonzero(consensus)
    candidate_values = raw_test_flat[candidate_indices]

    station_per_test_record = np.repeat(
        keys["base_station"].to_numpy(), n_test_hours
    )

    candidate_stations = station_per_test_record[candidate_indices]
    n_candidates = len(candidate_values)

    rng = np.random.default_rng(args.seed + 2026)

    if n_candidates >= 2:
        corr = np.corrcoef(candidate_values, rowvar=False)
    else:
        corr = np.full((n_kpis, n_kpis), np.nan)

    pd.DataFrame(
        corr, index=KPIS, columns=KPIS
    ).to_csv(
        reports / "w4_anomaly_kpi_correlation_matrix.csv"
    )

    pair_rows = []

    for i in range(n_kpis):
        for j in range(i + 1, n_kpis):
            x = candidate_values[:, i]
            y = candidate_values[:, j]

            if (
                len(x) >= 3
                and np.std(x) > 0
                and np.std(y) > 0
            ):
                r, p_iid = pearsonr(x, y)

                p_cluster, n_clusters, cluster_se = (
                    cluster_robust_slope_test(
                        x, y, candidate_stations
                    )
                )

                ci_low, ci_high = (
                    cluster_bootstrap_correlation_ci(
                        x,
                        y,
                        candidate_stations,
                        args.bootstrap_iterations,
                        rng,
                    )
                )

            else:
                r = p_iid = p_cluster = cluster_se = np.nan
                n_clusters = len(np.unique(candidate_stations))
                ci_low = ci_high = np.nan

            pair_rows.append({
                "KPI_1": KPIS[i],
                "KPI_2": KPIS[j],
                "n_candidate_beam_hours": len(x),
                "n_base_station_clusters": n_clusters,
                "pearson_r": r,
                "iid_pearson_p_value": p_iid,
                "cluster_robust_p_value": p_cluster,
                "cluster_robust_slope_se": cluster_se,
                "bootstrap_CI95_low": ci_low,
                "bootstrap_CI95_high": ci_high,
            })

    pair_df = pd.DataFrame(pair_rows)

    pair_df["holm_adjusted_p_value"] = holm_adjust(
        pair_df["cluster_robust_p_value"].to_numpy()
    )

    pair_df.to_csv(
        reports / "w4_anomaly_kpi_correlation_tests.csv",
        index=False,
    )

    p_matrix = pd.DataFrame(
        np.ones((n_kpis, n_kpis)),
        index=KPIS,
        columns=KPIS,
    )

    holm_matrix = p_matrix.copy()
    adjusted_lookup = {}

    for row in pair_df.to_dict("records"):
        a, b = row["KPI_1"], row["KPI_2"]

        p_matrix.loc[a, b] = p_matrix.loc[b, a] = (
            row["cluster_robust_p_value"]
        )

        holm_matrix.loc[a, b] = holm_matrix.loc[b, a] = (
            row["holm_adjusted_p_value"]
        )

        adjusted_lookup[tuple(sorted((a, b)))] = (
            row["holm_adjusted_p_value"]
        )

    p_matrix.to_csv(
        reports / "w4_anomaly_kpi_correlation_pvalues.csv"
    )

    holm_matrix.to_csv(
        reports / "w4_anomaly_kpi_correlation_pvalues_holm.csv"
    )

    # --------------------------------------------------------
    # 2.9 Figure 1: per-KPI anomaly flag rates
    # --------------------------------------------------------

    positions = np.arange(n_kpis)
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 4.8))

    ax.bar(
        positions - width / 2,
        100 * shape_df["3sigma_KPI_flag_rate"],
        width,
        label="3-sigma",
    )

    ax.bar(
        positions + width / 2,
        100 * shape_df["IQR_KPI_flag_rate"],
        width,
        label="IQR",
    )

    ax.set_xticks(positions, KPIS)
    ax.set_ylabel("Flagged KPI observations (%)")
    ax.set_title("Per-KPI anomaly flag rates on the test set")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        figures / "w4_anomaly_kpi_flag_rates.png",
        dpi=170,
    )
    plt.close(fig)

    # --------------------------------------------------------
    # 2.10 Figure 2: training distribution shape
    # --------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    skew_data = [
        beam_skew[:, j][np.isfinite(beam_skew[:, j])]
        for j in range(n_kpis)
    ]

    kurt_data = [
        beam_kurtosis[:, j][np.isfinite(beam_kurtosis[:, j])]
        for j in range(n_kpis)
    ]

    axes[0].boxplot(skew_data, labels=KPIS, showfliers=False)
    axes[0].axhline(0, linestyle="--", linewidth=1)
    axes[0].set_title("Training skewness by beam")
    axes[0].set_ylabel("Skewness")

    axes[1].boxplot(kurt_data, labels=KPIS, showfliers=False)
    axes[1].axhline(0, linestyle="--", linewidth=1)
    axes[1].set_title("Training excess kurtosis by beam")
    axes[1].set_ylabel("Excess kurtosis")

    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Training distribution shape by KPI")
    fig.tight_layout()

    fig.savefig(
        figures / "w4_anomaly_training_distribution_shape.png",
        dpi=170,
    )
    plt.close(fig)

    # --------------------------------------------------------
    # 2.11 Figure 3: correlation matrix and significance
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 5.8))

    if n_candidates >= 2 and np.isfinite(corr).any():
        image = ax.imshow(
            corr,
            vmin=-1,
            vmax=1,
            cmap="coolwarm",
        )

        fig.colorbar(
            image,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            label="Pearson r",
        )

        for i in range(n_kpis):
            for j in range(n_kpis):
                value = corr[i, j]

                if i == j:
                    label = f"{value:.2f}" if np.isfinite(value) else "NA"

                else:
                    p_adj = adjusted_lookup[
                        tuple(sorted((KPIS[i], KPIS[j])))
                    ]

                    if not np.isfinite(p_adj):
                        stars = "NA"
                    elif p_adj < 0.001:
                        stars = "***"
                    elif p_adj < 0.01:
                        stars = "**"
                    elif p_adj < 0.05:
                        stars = "*"
                    else:
                        stars = "ns"

                    label = (
                        f"{value:.2f}\n{stars}"
                        if np.isfinite(value)
                        else f"NA\n{stars}"
                    )

                ax.text(
                    j, i, label,
                    ha="center",
                    va="center",
                    fontsize=9,
                )

        ax.set_xticks(
            range(n_kpis), KPIS,
            rotation=25, ha="right",
        )
        ax.set_yticks(range(n_kpis), KPIS)

        ax.set_title(
            "KPI correlations among consensus candidates\n"
            f"n={n_candidates:,} beam-hours; "
            "Holm-adjusted cluster-robust significance"
        )

    else:
        ax.text(
            0.5, 0.5,
            "Insufficient candidates for correlation analysis",
            ha="center",
            va="center",
        )
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(
        figures / "w4_anomaly_kpi_correlation_significance.png",
        dpi=170,
    )
    plt.close(fig)

    # --------------------------------------------------------
    # 3. Console summary
    # --------------------------------------------------------

    print("\nW4 analysis completed.")
    print(f"Training hours: {len(train_cols):,}")
    print(f"Test hours: {len(test_cols):,}")
    print(f"Beams: {n_beams:,}")
    print(f"Evaluated test beam-hours: {n_test:,}")
    print(f"Consensus candidates: {n_candidates:,}")

    print("\nOverall detector flag rates:")
    print(
        f"  3-sigma:           {sigma.mean():.2%}"
    )
    print(
        f"  IQR:               {iqr.mean():.2%}"
    )
    print(
        f"  Isolation Forest:  {if_flag.mean():.2%}"
    )
    print(
        f"  Consensus:         {consensus.mean():.2%}"
    )

    print(f"\nIQR fallback count: {fallback_count:,}")
    print(f"CSV outputs: {reports.resolve()}")
    print(f"Figure outputs: {figures.resolve()}")


if __name__ == "__main__":
    main()
