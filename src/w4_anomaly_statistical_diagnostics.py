#!/usr/bin/env python3
"""W4 statistical diagnostics for the 5G KPI anomaly-detection experiments.

Run from the repository root:
    python src/w4_anomaly_statistical_diagnostics.py

The script reuses the W3 preprocessing and detector definitions, then exports
full-test flag overlaps, KPI distribution diagnostics, and correlation tests.
No verified fault labels are available, so all detected events remain candidates.
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

from w3_anomaly_detection import (
    KEYS,
    KPIS,
    detection_thresholds,
    features,
    load_data,
    sampled_fit,
)


def describe(values: np.ndarray) -> dict:
    """Descriptive statistics for a one-dimensional KPI sample."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return {
            "n": 0, "mean": np.nan, "std": np.nan, "median": np.nan,
            "q01": np.nan, "q25": np.nan, "q75": np.nan, "q99": np.nan,
            "zero_fraction": np.nan, "skewness": np.nan,
            "excess_kurtosis": np.nan,
        }
    nonconstant = finite.size > 2 and np.std(finite) > 0
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else np.nan,
        "median": float(np.median(finite)),
        "q01": float(np.quantile(finite, 0.01)),
        "q25": float(np.quantile(finite, 0.25)),
        "q75": float(np.quantile(finite, 0.75)),
        "q99": float(np.quantile(finite, 0.99)),
        "zero_fraction": float(np.mean(np.isclose(finite, 0.0, atol=1e-8))),
        "skewness": float(skew(finite, bias=False)) if nonconstant else np.nan,
        "excess_kurtosis": (
            float(kurtosis(finite, fisher=True, bias=False))
            if nonconstant else np.nan
        ),
    }


def holm_adjust(p_values: list[float]) -> np.ndarray:
    """Holm-Bonferroni adjustment, preserving the input order."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(p.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return adjusted
    order = valid[np.argsort(p[valid])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        candidate = min(1.0, (m - rank) * p[idx])
        running = max(running, candidate)
        adjusted[idx] = running
    return adjusted


def cluster_robust_slope_test(
    x: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
) -> tuple[float, int, float]:
    """Cluster-robust test of zero slope in y ~ 1 + x.

    The cluster unit is base station. This accommodates dependence among
    candidate beam-hours within a station better than an IID Pearson test.
    With only 30 stations, the resulting inference is still approximate.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    clusters = np.asarray(clusters)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y, clusters = x[keep], y[keep], clusters[keep]
    unique_clusters = np.unique(clusters)
    n, g = len(x), len(unique_clusters)
    if n < 4 or g < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, int(g), np.nan

    design = np.column_stack((np.ones(n), x))
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ beta
    bread = np.linalg.pinv(design.T @ design)
    meat = np.zeros((2, 2), dtype=float)

    for cluster in unique_clusters:
        mask = clusters == cluster
        score = design[mask].T @ residual[mask]
        meat += np.outer(score, score)

    correction = (g / (g - 1)) * ((n - 1) / (n - design.shape[1]))
    covariance = correction * bread @ meat @ bread
    variance = float(covariance[1, 1])
    if not np.isfinite(variance) or variance <= 0:
        return np.nan, int(g), np.nan
    standard_error = np.sqrt(variance)
    statistic = beta[1] / standard_error
    p_value = float(2 * t.sf(abs(statistic), df=g - 1))
    return p_value, int(g), float(standard_error)


def cluster_bootstrap_correlation_ci(
    x: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile CI for Pearson r from a base-station cluster bootstrap."""
    clusters = np.asarray(clusters)
    unique_clusters = np.unique(clusters)
    groups = [np.flatnonzero(clusters == c) for c in unique_clusters]
    if len(groups) < 3 or iterations <= 0:
        return np.nan, np.nan

    boot_r = []
    for _ in range(iterations):
        selected = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[k] for k in selected])
        xb, yb = x[idx], y[idx]
        if np.std(xb) == 0 or np.std(yb) == 0:
            continue
        value = float(np.corrcoef(xb, yb)[0, 1])
        if np.isfinite(value):
            boot_r.append(value)

    if len(boot_r) < max(30, int(iterations * 0.5)):
        return np.nan, np.nan
    low, high = np.quantile(boot_r, [0.025, 0.975])
    return float(low), float(high)


def p_text(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 1e-300:
        return "<1e-300"
    return f"{value:.3g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path,
        default=Path("data/processed/kpi_timeseries.csv"),
        help="Processed long-format KPI dataset used by W3.",
    )
    parser.add_argument("--out", type=Path, default=Path("."))
    parser.add_argument("--test-start", type=int, default=672)
    parser.add_argument("--if-train-sample", type=int, default=80000)
    parser.add_argument("--if-contamination", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    args = parser.parse_args()

    if args.bootstrap_iterations < 0:
        parser.error("--bootstrap-iterations must be >= 0")
    if not 0 < args.if_contamination <= 0.5:
        parser.error("--if-contamination must lie in (0, 0.5]")

    reports = args.out / "reports"
    figures = args.out / "figures"
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    values, raw, keys, timeline = load_data(args.csv)
    train_cols = np.flatnonzero(timeline < args.test_start)
    test_cols = np.flatnonzero(timeline >= args.test_start)
    if len(train_cols) < 10 or len(test_cols) < 1:
        raise ValueError("Need >=10 training hours and >=1 test hour")
    if not np.array_equal(train_cols, np.arange(train_cols[-1] + 1)):
        raise ValueError("Training hours must form a chronological prefix")

    train = values[:, train_cols, :]
    test = values[:, test_cols, :]
    raw_train = raw[:, train_cols, :]
    raw_test = raw[:, test_cols, :]
    n_beams, n_test_hours, _ = test.shape
    n_test = n_beams * n_test_hours

    sigma_kpi, iqr_kpi, beam_z, fallback_count = detection_thresholds(train, test)
    sigma = sigma_kpi.any(axis=2).reshape(-1)
    iqr = iqr_kpi.any(axis=2).reshape(-1)

    # Refit Isolation Forest using the same features, sample seed, and settings
    # as src/w3_anomaly_detection.py so the overlap analysis is reproducible.
    all_features = features(values)
    f_train = all_features[:, train_cols, :].reshape(-1, 8)
    f_test = all_features[:, test_cols, :].reshape(-1, 8)
    if_train = sampled_fit(f_train, args.if_train_sample, args.seed)
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
    consensus = if_flag & (sigma | iqr)

    # Full-test flags are retained so that the logic can be audited exactly.
    test_times = np.tile(timeline[test_cols], n_beams)
    flag_data = {
        "Time": test_times,
        "base_station": np.repeat(keys["base_station"].to_numpy(), n_test_hours),
        "cell": np.repeat(keys["cell"].to_numpy(), n_test_hours),
        "beam": np.repeat(keys["beam"].to_numpy(), n_test_hours),
        "3sigma_any_KPI": sigma.astype(np.uint8),
        "IQR_any_KPI": iqr.astype(np.uint8),
        "IsolationForest": if_flag.astype(np.uint8),
        "Consensus_IF_AND_sigma_OR_IQR": consensus.astype(np.uint8),
    }
    raw_test_flat = raw_test.reshape(-1, len(KPIS))
    sigma_flat = sigma_kpi.reshape(-1, len(KPIS))
    iqr_flat = iqr_kpi.reshape(-1, len(KPIS))
    for j, kpi in enumerate(KPIS):
        flag_data[kpi] = raw_test_flat[:, j]
        flag_data[f"3sigma_{kpi}"] = sigma_flat[:, j].astype(np.uint8)
        flag_data[f"IQR_{kpi}"] = iqr_flat[:, j].astype(np.uint8)
    pd.DataFrame(flag_data).to_csv(
        reports / "w4_anomaly_test_flags.csv", index=False
    )

    # Overall beam-hour rates (any KPI flagged).
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
    logic_df.to_csv(reports / "w4_anomaly_logic_comparison.csv", index=False)

    # Per-KPI distribution summaries on the original KPI scale.
    distribution_rows = []
    for j, kpi in enumerate(KPIS):
        for split_name, arr in (
            ("train", raw_train[:, :, j]),
            ("test", raw_test[:, :, j]),
        ):
            distribution_rows.append({
                "KPI": kpi,
                "split": split_name,
                **describe(arr),
            })
    distribution_df = pd.DataFrame(distribution_rows)
    distribution_df.to_csv(
        reports / "w4_anomaly_distribution_summary.csv", index=False
    )

    # Shape of the per-beam training distributions that determine thresholds.
    q1, q3 = np.quantile(train, [0.25, 0.75], axis=1)
    beam_iqr = q3 - q1
    beam_skew = skew(train, axis=1, bias=False)
    beam_excess_kurtosis = kurtosis(train, axis=1, fisher=True, bias=False)
    shape_rows = []
    for j, kpi in enumerate(KPIS):
        iqr_j = beam_iqr[:, j]
        skew_j = beam_skew[:, j]
        kurt_j = beam_excess_kurtosis[:, j]
        sigma_j = sigma_kpi[:, :, j]
        iqr_flags_j = iqr_kpi[:, :, j]
        shape_rows.append({
            "KPI": kpi,
            "beam_KPI_training_distributions": int(n_beams),
            "median_training_IQR_per_beam": float(np.nanmedian(iqr_j)),
            "q25_training_IQR_per_beam": float(np.nanquantile(iqr_j, 0.25)),
            "q75_training_IQR_per_beam": float(np.nanquantile(iqr_j, 0.75)),
            "zero_IQR_fallback_distributions": int(np.sum(iqr_j < 1e-8)),
            "zero_IQR_fraction": float(np.mean(iqr_j < 1e-8)),
            "share_abs_skewness_gt_1": float(np.nanmean(np.abs(skew_j) > 1)),
            "share_excess_kurtosis_gt_3": float(np.nanmean(kurt_j > 3)),
            "median_per_beam_skewness": float(np.nanmedian(skew_j)),
            "median_per_beam_excess_kurtosis": float(np.nanmedian(kurt_j)),
            "3sigma_flagged_KPI_values": int(sigma_j.sum()),
            "3sigma_KPI_value_flag_rate": float(sigma_j.mean()),
            "IQR_flagged_KPI_values": int(iqr_flags_j.sum()),
            "IQR_KPI_value_flag_rate": float(iqr_flags_j.mean()),
            "IQR_only_KPI_value_flag_rate": float((iqr_flags_j & ~sigma_j).mean()),
            "both_methods_KPI_value_flag_rate": float((iqr_flags_j & sigma_j).mean()),
        })
    shape_df = pd.DataFrame(shape_rows)
    shape_df.to_csv(reports / "w4_anomaly_threshold_shape_summary.csv", index=False)

    # Pairwise correlations among all consensus candidates, with station-level
    # cluster-robust tests and a station-cluster bootstrap CI.
    candidate_indices = np.flatnonzero(consensus)
    candidate_values = raw_test_flat[candidate_indices]
    candidate_station_all = np.repeat(
        keys["base_station"].to_numpy(), n_test_hours
    )
    candidate_stations = candidate_station_all[candidate_indices]
    n_candidates = len(candidate_values)
    rng = np.random.default_rng(args.seed + 2026)

    corr = np.corrcoef(candidate_values, rowvar=False)
    corr_df = pd.DataFrame(corr, index=KPIS, columns=KPIS)
    corr_df.to_csv(reports / "w4_anomaly_kpi_correlation_matrix.csv")

    pair_rows = []
    for i in range(len(KPIS)):
        for j in range(i + 1, len(KPIS)):
            x = candidate_values[:, i]
            y = candidate_values[:, j]
            if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0:
                r, p_iid = pearsonr(x, y)
                p_cluster, n_clusters, cluster_se = cluster_robust_slope_test(
                    x, y, candidate_stations
                )
                ci_low, ci_high = cluster_bootstrap_correlation_ci(
                    x, y, candidate_stations,
                    args.bootstrap_iterations, rng,
                )
            else:
                r, p_iid, p_cluster, n_clusters, cluster_se = (
                    np.nan, np.nan, np.nan, len(np.unique(candidate_stations)), np.nan
                )
                ci_low, ci_high = np.nan, np.nan
            pair_rows.append({
                "KPI_1": KPIS[i],
                "KPI_2": KPIS[j],
                "n_candidate_beam_hours": int(len(x)),
                "n_base_station_clusters": int(n_clusters),
                "pearson_r": float(r),
                "iid_pearson_p_value": float(p_iid),
                "base_station_cluster_robust_p_value": float(p_cluster),
                "cluster_robust_slope_se": float(cluster_se),
                "base_station_cluster_bootstrap_CI95_low": float(ci_low),
                "base_station_cluster_bootstrap_CI95_high": float(ci_high),
            })
    pair_df = pd.DataFrame(pair_rows)
    pair_df["holm_adjusted_cluster_p"] = holm_adjust(
        pair_df["base_station_cluster_robust_p_value"].tolist()
    )
    pair_df.to_csv(reports / "w4_anomaly_kpi_correlation_tests.csv", index=False)

    p_matrix = pd.DataFrame(np.ones((4, 4)), index=KPIS, columns=KPIS)
    holm_matrix = pd.DataFrame(np.ones((4, 4)), index=KPIS, columns=KPIS)
    adjusted_lookup = {}
    for row in pair_df.to_dict("records"):
        a, b = row["KPI_1"], row["KPI_2"]
        p_matrix.loc[a, b] = p_matrix.loc[b, a] = row["base_station_cluster_robust_p_value"]
        holm_matrix.loc[a, b] = holm_matrix.loc[b, a] = row["holm_adjusted_cluster_p"]
        adjusted_lookup[tuple(sorted((a, b)))] = row["holm_adjusted_cluster_p"]
    p_matrix.to_csv(reports / "w4_anomaly_kpi_correlation_pvalues.csv")
    holm_matrix.to_csv(reports / "w4_anomaly_kpi_correlation_pvalues_holm.csv")

    # Figure 1: KPI-level flag rates (the denominator is KPI observations,
    # not beam-hours with at least one KPI flagged).
    x_positions = np.arange(len(KPIS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(
        x_positions - width / 2,
        100 * shape_df["3sigma_KPI_value_flag_rate"].to_numpy(),
        width, label="3-sigma",
    )
    ax.bar(
        x_positions + width / 2,
        100 * shape_df["IQR_KPI_value_flag_rate"].to_numpy(),
        width, label="IQR",
    )
    ax.set_xticks(x_positions, KPIS)
    ax.set_ylabel("Flagged KPI observations (%)")
    ax.set_title("Per-KPI flag rates on the full test set")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "w4_anomaly_kpi_flag_rates.png", dpi=170)
    plt.close(fig)

    # Figure 2: per-beam training distribution shape.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    skew_data = [
        beam_skew[:, j][np.isfinite(beam_skew[:, j])]
        for j in range(len(KPIS))
    ]
    kurt_data = [
        beam_excess_kurtosis[:, j][np.isfinite(beam_excess_kurtosis[:, j])]
        for j in range(len(KPIS))
    ]
    axes[0].boxplot(skew_data, labels=KPIS, showfliers=False)
    axes[0].axhline(0, linewidth=1, linestyle="--")
    axes[0].set_title("Training skewness by beam")
    axes[0].set_ylabel("Skewness")
    axes[1].boxplot(kurt_data, labels=KPIS, showfliers=False)
    axes[1].axhline(0, linewidth=1, linestyle="--")
    axes[1].set_title("Training excess kurtosis by beam")
    axes[1].set_ylabel("Excess kurtosis")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Distribution shape used by per-beam anomaly thresholds")
    fig.tight_layout()
    fig.savefig(figures / "w4_anomaly_training_distribution_shape.png", dpi=170)
    plt.close(fig)

    # Figure 3: candidate correlation matrix with station-cluster-adjusted stars.
    fig, ax = plt.subplots(figsize=(7, 5.8))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(4), KPIS, rotation=25, ha="right")
    ax.set_yticks(range(4), KPIS)
    for i in range(4):
        for j in range(4):
            value = corr[i, j]
            if i == j:
                label = f"{value:.2f}"
            else:
                p_adj = adjusted_lookup[tuple(sorted((KPIS[i], KPIS[j])))]
                stars = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "ns"
                label = f"{value:.2f}\n{stars}"
            ax.text(j, i, label, ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    ax.set_title(
        f"Consensus-candidate KPI correlations\n"
        f"n={n_candidates:,} beam-hours; stars use Holm-adjusted station-cluster p"
    )
    fig.tight_layout()
    fig.savefig(figures / "w4_anomaly_kpi_correlation_significance.png", dpi=170)
    plt.close(fig)

    # Build a report with data-driven conclusions; no distribution claim is
    # hard-coded before the full KPI dataset is analysed.
    method_rows = []
    for name, mask in (
        ("3-sigma", sigma),
        ("IQR", iqr),
        ("Isolation Forest", if_flag),
        ("Consensus: IF AND (3-sigma OR IQR)", consensus),
    ):
        method_rows.append(
            f"| {name} | {int(mask.sum()):,} | {mask.mean():.2%} |"
        )
    method_table = "\n".join(method_rows)

    logic_table = "\n".join(
        f"| {row.logic} | {row.flagged_beam_hours:,} | {row.flag_rate:.2%} |"
        for row in logic_df.itertuples(index=False)
    )
    shape_table = "\n".join(
        f"| {row.KPI} | {row.median_training_IQR_per_beam:.3f} | "
        f"{row.zero_IQR_fallback_distributions:,} | "
        f"{row.share_abs_skewness_gt_1:.1%} | "
        f"{row.share_excess_kurtosis_gt_3:.1%} | "
        f"{row['3sigma_KPI_value_flag_rate']:.2%} | "
        f"{row['IQR_KPI_value_flag_rate']:.2%} |"
        for _, row in shape_df.iterrows()
    )
    corr_table_rows = []
    for row in pair_df.to_dict("records"):
        corr_table_rows.append(
            f"| {row['KPI_1']} – {row['KPI_2']} | "
            f"{row['n_candidate_beam_hours']:,} | {row['pearson_r']:.3f} | "
            f"[{row['base_station_cluster_bootstrap_CI95_low']:.3f}, "
            f"{row['base_station_cluster_bootstrap_CI95_high']:.3f}] | "
            f"{p_text(row['holm_adjusted_cluster_p'])} |"
        )
    corr_table = "\n".join(corr_table_rows)

    iqr_rate = float(iqr.mean())
    sigma_rate = float(sigma.mean())
    rate_ratio = iqr_rate / sigma_rate if sigma_rate else np.nan
    report = f"""# W4: Statistical Diagnostics for KPI Anomaly Detection

## 1. Experimental scope

- Input: {args.csv}
- Training hours: {int(timeline[train_cols[0]])}–{int(timeline[train_cols[-1]])} ({len(train_cols):,} hours)
- Test hours: {int(timeline[test_cols[0]])}–{int(timeline[test_cols[-1]])} ({len(test_cols):,} hours)
- Beams: {n_beams:,}
- Full-test beam-hours: {n_test:,}
- Consensus candidates: {n_candidates:,} ({consensus.mean():.2%})
- IQR fallback count across beam-KPI training distributions: {fallback_count:,}

No verified network-fault labels are available. All detections are anomaly candidates, not confirmed network faults.

## 2. Overall detection rates

| Method | Flagged beam-hours | Full-test rate |
|---|---:|---:|
{method_table}

The One-Class SVM result from the earlier W3 experiment was evaluated on a sampled subset of 12,000 test records. Its 4.60% rate must not be compared as if it were measured on the full test set.

## 3. Distribution evidence and the IQR-versus-3-sigma hypothesis

The earlier run marked {iqr_rate:.2%} of beam-hours with IQR and {sigma_rate:.2%} with 3-sigma (IQR rate / 3-sigma rate = {rate_ratio:.2f}×). A plausible methodological explanation is that the conventional IQR fences (Q1 − 1.5×IQR, Q3 + 1.5×IQR) are narrower than ±3 standard deviations for a Gaussian distribution. Non-Gaussian tails, skewness, beam-level heterogeneity, and train/test distribution shift can also affect the observed rates.

The following table tests whether the per-beam training distributions exhibit skewness/heavy tails and reports per-KPI flag rates. Shape statistics are calculated per beam over training hours; flag rates here are per KPI observation, not per beam-hour with any KPI flagged.

| KPI | Median train IQR/beam | Zero-IQR fallback distributions | Beams with |skewness| > 1 | Beams with excess kurtosis > 3 | 3-sigma KPI flag rate | IQR KPI flag rate |
|---|---:|---:|---:|---:|---:|---:|
{shape_table}

Interpretation must follow these observed values. If skewness/heavy tails are common, that supports a non-Gaussian-distribution explanation; if they are uncommon, the difference may instead be driven more by the relative fence widths or temporal distribution shift. A higher IQR rate alone does not prove any one explanation.

## 4. AND/OR consensus logic

The implemented rule is:

Consensus = IsolationForest AND (3-sigma OR IQR)

A record must be flagged by Isolation Forest and by at least one of the two statistical detectors. The rule is a screening compromise: the statistical detector supplies a second signal, while the AND with Isolation Forest reduces the candidate set. The OR between 3-sigma and IQR prevents the consensus from requiring both statistical rules to agree.

| Rule | Flagged beam-hours | Full-test rate |
|---|---:|---:|
{logic_table}

The table allows the final report to compare this rule against stricter IF AND 3-sigma AND IQR and looser IF OR 3-sigma OR IQR alternatives. These rates describe detector agreement, not fault-classification accuracy.

## 5. KPI correlations among consensus candidates

The correlation matrix uses all {n_candidates:,} consensus-candidate beam-hours. Confidence intervals are from a bootstrap that resamples base stations; p-values test a zero regression slope with base-station-clustered standard errors, followed by Holm correction across the six KPI pairs.

| KPI pair | n | Pearson r | 95% station-cluster bootstrap CI | Holm-adjusted cluster p |
|---|---:|---:|---:|---:|
{corr_table}

A statistically significant correlation indicates that KPI values co-vary within the selected candidate sample; it does not identify a causal root cause. Because candidate selection itself depends on the KPI-derived detectors, these correlations are exploratory and should be interpreted alongside the full-test distribution results and engineering context. There are only {len(np.unique(candidate_stations))} base-station clusters, so cluster-level inference remains approximate.

## 6. Figure captions and conclusions

**Figure 1. Per-KPI anomaly flag rates.** Percentage of individual KPI observations in the full test set flagged by the 3-sigma and IQR rules, shown separately for DLPRB, DLThpVol, DLThpTime, and MR_number. The denominator is the number of test beam-hours for that KPI. Compare the per-KPI rates rather than assuming one KPI accounts for the higher overall IQR rate.

**Figure 2. Training-distribution shape by beam.** Boxplots of per-beam skewness and excess kurtosis for each KPI, computed from the training period used to estimate thresholds. Large absolute skewness or positive excess kurtosis indicates departures from a symmetric Gaussian-like distribution. Use the corresponding summary table to quantify, rather than infer from the plot alone, how prevalent those departures are.

**Figure 3. KPI correlation matrix among consensus candidates.** Pearson correlation coefficients for the four KPI values in the selected candidate beam-hours. Asterisks mark Holm-adjusted, base-station-cluster-robust tests (* p < .05, ** p < .01, *** p < .001; ns, not significant); cell labels show r. Correlation strength is not evidence of causality.

## 7. Reproducibility and output files

- reports/w4_anomaly_test_flags.csv: full-test detector flags and KPI values.
- reports/w4_anomaly_logic_comparison.csv: counts and rates for alternative AND/OR rules.
- reports/w4_anomaly_distribution_summary.csv: pooled raw-KPI descriptive statistics for train and test.
- reports/w4_anomaly_threshold_shape_summary.csv: per-beam training-distribution shape and per-KPI flag rates.
- reports/w4_anomaly_kpi_correlation_matrix.csv: Pearson correlation matrix.
- reports/w4_anomaly_kpi_correlation_tests.csv: sample sizes, nominal and cluster-robust p-values, Holm correction, and cluster-bootstrap confidence intervals.
- reports/w4_anomaly_kpi_correlation_pvalues.csv and reports/w4_anomaly_kpi_correlation_pvalues_holm.csv: p-value matrices.
- figures/w4_anomaly_kpi_flag_rates.png
- figures/w4_anomaly_training_distribution_shape.png
- figures/w4_anomaly_kpi_correlation_significance.png

## 8. Limitations

- No verified fault labels; precision, recall, and F1 cannot be calculated.
- Correlation analysis is conditional on a detector-selected candidate set.
- Cluster-robust inference uses base station as the cluster unit; 30 clusters limit precision.
- Detection thresholds may be sensitive to temporal distribution shift.
- One-Class SVM remains a sampled-test result from the earlier experiment and is not re-estimated by this diagnostic script.
"""
    (reports / "w4_anomaly_diagnostics_report.md").write_text(
        report, encoding="utf-8"
    )

    print(f"Full-test beam-hours: {n_test:,}")
    print(f"Consensus candidates: {n_candidates:,} ({consensus.mean():.2%})")
    print(f"IQR rate: {iqr_rate:.2%}; 3-sigma rate: {sigma_rate:.2%}")
    print(f"Reports saved to: {reports.resolve()}")
    print(f"Figures saved to: {figures.resolve()}")


if __name__ == "__main__":
    main()
