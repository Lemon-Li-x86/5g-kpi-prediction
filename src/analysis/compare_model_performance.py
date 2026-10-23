#!/usr/bin/env python3
"""W3: compare existing traditional ML and deep learning KPI forecasts.

Usage (from any working directory):
    python -m src.analysis.compare_model_performance

Reads earlier model metrics without retraining either family. Reports use the
original KPI scale and the per-beam test metric, not hourly-mean curves.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from src.core.config import KPIS as SHARED_KPIS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KPIS = tuple(SHARED_KPIS)  # immutable order for historical comparison
METRICS = ("MAE", "RMSE", "R2")


def load_metrics(path: Path, family: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {family} metrics: {path}\n"
            "Run the relevant training script before comparing results."
        )
    df = pd.read_csv(path)
    if "target" in df.columns and "KPI" not in df.columns:
        df = df.rename(columns={"target": "KPI"})
    required = {"model", "KPI", *METRICS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    df = df.copy()
    df["model"] = df["model"].astype(str).str.strip()
    df["KPI"] = df["KPI"].astype(str).str.strip()
    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="raise")
        if not np.isfinite(df[metric]).all():
            raise ValueError(f"{path}: non-finite {metric} values")
    if (df[["MAE", "RMSE"]] < 0).any().any():
        raise ValueError(f"{path}: negative MAE/RMSE is invalid")
    if set(df["KPI"]) != set(KPIS):
        raise ValueError(f"{path}: KPIs do not match {KPIS}")
    if df.duplicated(["KPI", "model"]).any():
        raise ValueError(f"{path}: duplicate KPI/model results")

    # Traditional ML records test_rows; new DL records test_samples.
    if "test_samples" not in df.columns:
        if "test_rows" not in df.columns:
            raise ValueError(f"{path}: missing test_rows/test_samples")
        df["test_samples"] = df["test_rows"]
    df["test_samples"] = pd.to_numeric(df["test_samples"], errors="raise")
    if df["test_samples"].isna().any() or (df["test_samples"] <= 0).any():
        raise ValueError(f"{path}: invalid test sample count")

    if family == "DL":
        if "unit" not in df.columns or not df["unit"].eq("raw").all():
            raise ValueError("DL metrics must declare unit=raw")

    # A common test set implies an (approximately) common target variance:
    # R2 = 1 - RMSE^2 / variance(y_true). Detect obvious scoring mismatches.
    inferred_var = df["RMSE"].to_numpy() ** 2 / (1 - df["R2"].to_numpy())
    if np.any(~np.isfinite(inferred_var)) or np.any(inferred_var <= 0):
        raise ValueError(f"{path}: inconsistent RMSE/R2; check metric calculation")
    df["family"] = "Deep learning" if family == "DL" else "Traditional ML"
    return df[["KPI", "model", "family", *METRICS, "test_samples"]]


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_protocol(
    ml: pd.DataFrame, dl: pd.DataFrame, root: Path,
) -> list[str]:
    notes: list[str] = []
    sizes = set(pd.concat([ml, dl])["test_samples"].astype(int).tolist())
    if len(sizes) != 1:
        raise ValueError(f"Models used different test sample counts: {sorted(sizes)}")
    sample_count = sizes.pop()

    ml_meta = read_json(root / "reports" / "ml_feature_summary.json")
    dl_meta = read_json(root / "reports" / "w3_deep_learning_run_config.json")
    if ml_meta is not None and dl_meta is not None:
        start_ml = int(ml_meta["split"]["test_start_target_time"])
        first_dl, last_dl = map(int, dl_meta["test"])
        last_ml = int(ml_meta["dataset"]["target_time_max"])
        if (start_ml, last_ml) != (first_dl, last_dl):
            raise ValueError(
                f"Test hours mismatch: ML {start_ml}..{last_ml}, "
                f"DL {first_dl}..{last_dl}"
            )
        if not dl_meta.get("full_test", False):
            raise ValueError("DL run used a sampled test set; cannot compare fairly")
        if int(dl_meta["test_samples"]) != sample_count:
            raise ValueError("DL configuration and metrics have different test sizes")
        if int(ml_meta["split"]["test_rows"]) != sample_count:
            raise ValueError("ML configuration and metrics have different test sizes")
        notes.append(
            f"Checked metadata: identical held-out target hours "
            f"{start_ml}..{last_ml}, {sample_count:,} beam-hour observations."
        )
        notes.append(
            f"DL training windows: {dl_meta.get('train_samples', 'unknown')}; "
            f"ML training rows: {ml_meta['split'].get('train_rows', 'unknown')}."
        )
    else:
        notes.append(
            "WARNING: One or both run configuration files are missing. "
            "Equal test counts alone do not prove equal test hours/beam IDs; "
            "verify the original training protocols before drawing conclusions."
        )

    # Same KPI/time-level test set should have the same population variance.
    all_scores = pd.concat([ml, dl], ignore_index=True)
    for kpi, group in all_scores.groupby("KPI"):
        variance = group["RMSE"].to_numpy() ** 2 / (1 - group["R2"].to_numpy())
        if (variance.max() - variance.min()) / variance.mean() > 0.015:
            raise ValueError(
                f"{kpi}: incompatible RMSE/R2 implied test target variance. "
                "Check data split, target scaling, and scoring."
            )
    notes.append("MAE/RMSE: lower is better; R2: higher is better.")
    notes.append(
        "Comparison describes held-out accuracy, NOT equal-data/equal-compute "
        "architecture superiority: DL and ML have different training-sample "
        "counts, input representations and model-selection procedures."
    )
    notes.append(
        "Data were chronologically held out from files named 'train'; "
        "the evaluation is not an independent official test set."
    )
    notes.append(
        "Original DL StandardScaler was fitted on hours 0..671, including "
        "the internal DL validation period. This limitation should be disclosed."
    )
    return notes


def best_one(group: pd.DataFrame, metric: str) -> pd.Series:
    ascending = metric != "R2"
    return group.sort_values(
        [metric, "model"], ascending=[ascending, True], kind="stable"
    ).iloc[0]


def build_summary(all_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for kpi in KPIS:
        group = all_scores[all_scores["KPI"] == kpi]
        ml = group[group["family"] == "Traditional ML"]
        dl = group[group["family"] == "Deep learning"]
        for metric in METRICS:
            ml_best = best_one(ml, metric)
            dl_best = best_one(dl, metric)
            if metric == "R2":
                improvement = float(dl_best[metric] - ml_best[metric])
                unit = "R2 difference (DL - ML)"
            else:
                improvement = float(
                    100 * (ml_best[metric] - dl_best[metric]) / ml_best[metric]
                )
                unit = "error reduction % (positive = DL better)"
            rows.append({
                "KPI": kpi,
                "metric": metric,
                "ML_best_model": ml_best["model"],
                "ML_best_score": float(ml_best[metric]),
                "DL_best_model": dl_best["model"],
                "DL_best_score": float(dl_best[metric]),
                "DL_improvement": improvement,
                "improvement_unit": unit,
            })
    return pd.DataFrame(rows)


def make_figures(all_scores: pd.DataFrame, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    for kpi in KPIS:
        group = all_scores.loc[all_scores["KPI"] == kpi].copy()
        for metric in ("MAE", "RMSE"):
            group = group.sort_values(metric, ascending=True)
            labels = [f"{row.model} ({'DL' if row.family == 'Deep learning' else 'ML'})"
                      for row in group.itertuples(index=False)]
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh(labels, group[metric].to_numpy())
            ax.invert_yaxis()
            ax.set_xlabel(f"{metric} (original KPI scale, lower is better)")
            ax.set_title(f"{kpi} | traditional ML vs deep learning")
            ax.grid(axis="x", alpha=0.2)
            fig.tight_layout()
            fig.savefig(figures / f"w3_{kpi}_{metric.lower()}_comparison.png", dpi=170)
            plt.close(fig)


def make_report(
    summary: pd.DataFrame, notes: list[str], out: Path,
    ml_source: Path, dl_source: Path,
) -> None:
    lines = [
        "# 第三阶段：传统机器学习与深度学习 KPI 预测对比分析",
        "",
        "## 数据来源",
        "",
        f"- 传统机器学习：`{ml_source}`",
        f"- 深度学习：`{dl_source}`",
        "- 评价粒度：每个 beam × 目标小时，使用原始 KPI 尺度。",
        "",
        "## 三项指标的最优模型比较",
        "",
        "| KPI | 指标 | 传统 ML 最优 | DL 最优 | DL 相对变化 |",
        "|---|---|---|---|---:|",
    ]
    for r in summary.itertuples(index=False):
        delta = f"{r.DL_improvement:+.2f}%" if r.metric != "R2" else f"{r.DL_improvement:+.4f}"
        lines.append(
            f"| {r.KPI} | {r.metric} | {r.ML_best_model} "
            f"({r.ML_best_score:.6f}) | {r.DL_best_model} "
            f"({r.DL_best_score:.6f}) | {delta} |"
        )
    lines.extend(["", "## 实验解释及局限性", ""])
    lines.extend(f"- {note}" for note in notes)
    lines.extend([
        "", "## 下一步分析", "",
        "- 分析 DLThpVol 预测的误差分布，尤其是 MAE 和 RMSE 不同步的现象。",
        "- 在仅由训练时间段拟合标准化器的条件下重复 DL 实验。",
        "- 若要论证模型稳定性，增加多个时间窗口的滚动验证或重复种子试验。",
        "- 不把模型预测偏差直接当作真实 5G 网络故障标签。",
        "",
    ])
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ml", type=Path, default=PROJECT_ROOT / "reports" / "model_metrics.csv")
    parser.add_argument("--dl", type=Path, default=PROJECT_ROOT / "reports" / "w3_deep_learning_metrics.csv")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "reports" / "week3")
    parser.add_argument("--figures", type=Path, default=PROJECT_ROOT / "figures" / "week3")
    args = parser.parse_args()

    ml = load_metrics(args.ml, "ML")
    dl = load_metrics(args.dl, "DL")
    notes = validate_protocol(ml, dl, PROJECT_ROOT)
    scores = pd.concat([ml, dl], ignore_index=True)
    summary = build_summary(scores)

    args.out.mkdir(parents=True, exist_ok=True)
    scores.sort_values(["KPI", "family", "model"]).to_csv(
        args.out / "w3_ml_dl_all_metrics.csv", index=False
    )
    summary.to_csv(args.out / "w3_ml_dl_comparison_summary.csv", index=False)
    make_figures(scores, args.figures)
    make_report(
        summary, notes,
        args.out / "w3_ml_dl_comparison_report.md",
        args.ml, args.dl,
    )

    print("\n=== BEST TRADITIONAL ML vs BEST DL FOR EACH KPI ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("\nProtocol notes:")
    for note in notes:
        print(" -", note)
    print("\nSaved results:", args.out.resolve())
    print("Saved figures:", args.figures.resolve())


if __name__ == "__main__":
    main()
