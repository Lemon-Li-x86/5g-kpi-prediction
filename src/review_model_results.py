
"""Compare the existing W4 model results without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def compare(metrics: pd.DataFrame):
    required = {
        "target", "model", "MAE", "RMSE", "R2",
        "fit_seconds", "predict_seconds"
    }
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Missing metric columns: {sorted(missing)}")

    if metrics.duplicated(["target", "model"]).any():
        raise ValueError("Duplicate target/model rows")

    if metrics[["MAE", "RMSE", "R2"]].isna().any().any():
        raise ValueError("Missing evaluation results")

    metrics = metrics.copy()

    baseline = metrics.loc[
        metrics["model"] == "Persistence",
        ["target", "MAE", "RMSE"]
    ].rename(columns={
        "MAE": "baseline_MAE",
        "RMSE": "baseline_RMSE"
    })

    if baseline["target"].nunique() != metrics["target"].nunique():
        raise ValueError("Persistence baseline is incomplete")

    detail = metrics.merge(
        baseline, on="target", validate="many_to_one"
    )

    detail["MAE_improvement_pct"] = (
        100 * (1 - detail["MAE"] / detail["baseline_MAE"])
    )
    detail["RMSE_improvement_pct"] = (
        100 * (1 - detail["RMSE"] / detail["baseline_RMSE"])
    )

    learners = detail[
        detail["model"] != "Persistence"
    ].copy()

    for metric, ascending in [
        ("MAE", True),
        ("RMSE", True),
        ("R2", False)
    ]:
        learners[f"{metric}_rank"] = (
            learners.groupby("target")[metric]
            .rank(ascending=ascending, method="average")
        )

    summary = learners.groupby(
        "model", as_index=False
    ).agg(
        mean_MAE_rank=("MAE_rank", "mean"),
        mean_RMSE_rank=("RMSE_rank", "mean"),
        mean_R2_rank=("R2_rank", "mean"),
        mean_MAE_improvement_pct=(
            "MAE_improvement_pct", "mean"
        ),
        mean_RMSE_improvement_pct=(
            "RMSE_improvement_pct", "mean"
        ),
        total_fit_seconds=("fit_seconds", "sum"),
        total_predict_seconds=("predict_seconds", "sum"),
        targets=("target", "nunique")
    )

    rank_columns = [
        "mean_MAE_rank",
        "mean_RMSE_rank",
        "mean_R2_rank"
    ]

    summary["overall_equal_weight_rank"] = (
        summary[rank_columns].mean(axis=1)
    )

    summary = summary.sort_values(
        [
            "overall_equal_weight_rank",
            "mean_RMSE_rank",
            "total_fit_seconds",
            "model"
        ]
    )

    summary.insert(
        0, "descriptive_place",
        range(1, len(summary) + 1)
    )

    return detail, summary


def run(metrics_path: Path, output_dir: Path):
    metrics = pd.read_csv(metrics_path)
    detail, summary = compare(metrics)

    output_dir.mkdir(parents=True, exist_ok=True)

    detail.to_csv(
        output_dir / "model_comparison_detail.csv",
        index=False
    )
    summary.to_csv(
        output_dir / "model_comparison_summary.csv",
        index=False
    )

    wins = {}
    learners = detail[detail["model"] != "Persistence"]

    for metric, ascending in [
        ("MAE", True),
        ("RMSE", True),
        ("R2", False)
    ]:
        group = learners.groupby("target")[metric]
        indices = (
            group.idxmin() if ascending
            else group.idxmax()
        )
        wins[metric] = detail.loc[
            indices, ["target", "model", metric]
        ].to_dict(orient="records")

    context = {
        "ranking_rule": (
            "Within-KPI ranks of MAE, RMSE and R2; "
            "equal weighting across metrics and KPIs"
        ),
        "baseline": "Persistence",
        "best_by_metric": wins,
        "caveat": (
            "Descriptive comparison on one held-out period; "
            "not a significance test"
        )
    }

    with (
        output_dir / "model_comparison_context.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            context, f, indent=2, ensure_ascii=False
        )

    print("\nMODEL COMPARISON")
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=Path,
        default=ROOT / "reports" / "model_metrics.csv"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "validation"
    )
    args = parser.parse_args()
    run(args.metrics, args.out)
