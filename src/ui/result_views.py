"""Presentation-only views for persisted experiment results and session inference.

Model fitting and prediction live in src.models; alignment and correlations
live in src.analysis. This module never modifies trained weights or reports.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.core.config import KPIS
from src.analysis.session_kpi_analysis import candidate_kpi_correlation
from src.ui.dashboard_data import (
    available_prediction_models,
    load_anomaly_events,
    load_anomaly_hourly_rates,
    load_anomaly_summary,
    load_correlation_matrix,
    load_correlation_tests,
    load_model_metrics,
    load_prediction_series,
)


def download_csv(frame: pd.DataFrame, label: str, filename: str) -> None:
    if not frame.empty:
        st.download_button(
            label,
            frame.to_csv(index=False).encode("utf-8-sig"),
            file_name=filename,
            mime="text/csv",
        )


def render_forecast_evaluation(report_dir: Path | None, source_label: str) -> None:
    """Evaluate only the held-out test predictions belonging to this source."""
    st.caption(f"测试集评估来源：{source_label}。没有真值的单次推理不在此计算 R²。")
    metrics = load_model_metrics(report_dir)
    if metrics.empty:
        st.info("该实验没有可用的回归测试指标。")
        return

    available = [item for item in KPIS if item in set(metrics["target"])]
    if not available:
        st.info("该实验指标中没有可用的四项 KPI。")
        return
    kpi = st.selectbox("评估 KPI", available, key="evaluated_kpi")
    chosen = metrics.loc[metrics["target"] == kpi].copy()
    st.dataframe(chosen, use_container_width=True, hide_index=True)
    download_csv(chosen, "导出本次测试指标 CSV", "evaluation_metrics.csv")

    metric = st.selectbox("比较指标", ["R2", "MAE", "RMSE"], key="evaluated_metric")
    chart = chosen.dropna(subset=[metric])
    if not chart.empty:
        fig = px.bar(
            chart, x="model", y=metric,
            color="family" if "family" in chart.columns else None,
            title=f"{kpi} · {metric}（{source_label}）",
        )
        st.plotly_chart(fig, use_container_width=True)

    saved_models = available_prediction_models(report_dir)
    candidates = [
        m for m in chosen["model"].astype(str).unique()
        if m.casefold() in {name.casefold() for name in saved_models}
    ]
    if not candidates:
        st.info("该实验保存了指标，但没有对应的逐小时预测序列。")
        return
    model = st.selectbox("查看测试集真实值与预测值", candidates, key="evaluated_model")
    series = load_prediction_series(model, kpi, report_dir)
    row = chosen.loc[chosen["model"].str.casefold() == model.casefold()]
    if not row.empty:
        item = row.iloc[0]
        a, b, c = st.columns(3)
        a.metric("MAE", f"{item['MAE']:.4f}")
        b.metric("RMSE", f"{item['RMSE']:.4f}")
        c.metric("R²", f"{item['R2']:.4f}")
    if series.empty:
        st.info("未找到对应测试预测序列。")
        return
    fig = go.Figure()
    fig.add_scatter(x=series["Time"], y=series["actual"], name="真实值（波束小时均值）")
    fig.add_scatter(x=series["Time"], y=series["prediction"], name="预测值（波束小时均值）")
    fig.update_layout(
        title=f"{model} · {kpi}：测试集真实值与预测值",
        xaxis_title="测试时间索引（小时）",
        yaxis_title=f"{kpi}（原始数据尺度）",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("测试指标按原测试样本计算；上图按小时汇总所有波束，不是单条波束曲线。")
    download_csv(series, "导出测试集预测曲线 CSV", "test_prediction_series.csv")


def render_anomaly_training_results(report_dir: Path, source_label: str) -> None:
    st.caption(f"异常实验来源：{source_label}。候选标记不等于确认故障。")
    summary = load_anomaly_summary(report_dir)
    if summary.empty:
        st.info("该实验没有可用的异常统计报告。")
        return
    st.subheader("检测方法与标记比例")
    st.dataframe(summary, hide_index=True, use_container_width=True)
    rates = summary.copy()
    rates["标记率（%）"] = 100 * rates["flag_rate"]
    fig = px.bar(rates, x="method", y="标记率（%）", title="本次实验异常候选率")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("One-Class SVM 通常仅评估采样子集，其分母不等于其他全量检测方法。")
    download_csv(summary, "导出异常统计 CSV", "anomaly_training_summary.csv")

    hourly = load_anomaly_hourly_rates(report_dir)
    if not hourly.empty and "Time" in hourly.columns:
        st.subheader("逐小时异常标记率")
        st.line_chart(hourly.set_index("Time"))
    events = load_anomaly_events(report_dir)
    if not events.empty:
        st.subheader("保存的候选事件")
        st.dataframe(events.head(300), hide_index=True, use_container_width=True)
        st.caption("事件文件可能受训练参数 max_events 限制，不等于全部异常候选。")
        download_csv(events, "导出本次实验候选事件 CSV", "anomaly_training_events.csv")


def render_forecast_inference_result(saved: dict) -> None:
    frame = saved["frame"]
    path = Path(saved["signature"][0])
    st.caption(f"本次模型权重：{path.parent.name} / {path.name}")
    st.info("预测的是输入数据之后的下一小时。没有下一小时真值，因此不展示 MAE、RMSE、R²。")
    targets = [k for k in KPIS if k in frame.columns]
    if not targets:
        st.warning("模型没有产生可展示的 KPI 列。")
        return
    st.subheader("本次 KPI 预测结果")
    st.dataframe(frame, use_container_width=True, hide_index=True)
    chosen = st.selectbox("预测指标", targets, key="forecast_result_kpi")
    values = frame.copy()
    if len(values) == 1:
        a, b = st.columns(2)
        a.metric("预测目标小时 Time", str(values["Time"].iloc[0]))
        b.metric(chosen, f"{float(values[chosen].iloc[0]):.4f}")
    else:
        values["波束"] = values[["base_station", "cell", "beam"]].astype(str).agg("/".join, axis=1)
        fig = px.bar(values, x="波束", y=chosen, title=f"下一小时 {chosen} 波束预测结果")
        st.plotly_chart(fig, use_container_width=True)
    download_csv(frame, "导出预测结果 CSV", "kpi_forecast_result.csv")


def render_anomaly_inference_result(saved: dict) -> None:
    result = saved["frame"]
    path = Path(saved["signature"][0])
    st.caption(f"本次模型权重：{path.parent.name} / {path.name}")
    flags = ["sigma", "iqr", "isolation_forest", "one_class_svm", "consensus"]
    summary = pd.DataFrame({
        "方法": flags,
        "候选数量": [int(result[k].sum()) for k in flags],
        "标记率（%）": [100 * float(result[k].mean()) for k in flags],
    })
    st.subheader("本次异常候选检测")
    st.dataframe(summary, hide_index=True, use_container_width=True)
    st.plotly_chart(
        px.bar(summary, x="方法", y="标记率（%）", title="本次输入的异常标记率"),
        use_container_width=True,
    )
    if "Time" in result.columns:
        hourly = result.groupby("Time")[flags].mean().mul(100)
        st.subheader("逐小时异常率（%）")
        st.line_chart(hourly)
    st.dataframe(result.head(300), hide_index=True, use_container_width=True)
    st.caption("表格预览最多 300 行；CSV 导出包含所有输出记录。检测标记不是故障真值。")
    download_csv(result, "导出异常检测 CSV", "kpi_anomaly_result.csv")


def render_current_association(saved: dict) -> None:
    candidates, corr = candidate_kpi_correlation(saved["history"], saved["frame"])
    st.caption("只对本次输入中被共识规则标记的观测计算 Pearson 相关系数。"
               " 不复用过去实验的矩阵或显著性检验。")
    st.metric("本次共识候选 beam-hour", f"{len(candidates):,}")
    if corr.empty:
        st.warning("本次候选少于 3 条，无法可靠计算相关矩阵。")
        return
    st.plotly_chart(
        px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                  title="本次异常候选 KPI Pearson 相关矩阵", text_auto=".2f"),
        use_container_width=True,
    )
    st.dataframe(candidates.head(300), use_container_width=True, hide_index=True)
    download_csv(corr.reset_index().rename(columns={"index": "KPI"}),
                 "导出本次相关矩阵 CSV", "session_kpi_correlations.csv")
    download_csv(candidates, "导出用于分析的异常 KPI CSV", "session_anomaly_kpi_values.csv")
    st.warning("Pearson 相关性不是因果关系；未运行显著性检验，不报告 p 值或故障根因。")


def render_saved_association(report_dir: Path | None, label: str) -> None:
    corr, source = load_correlation_matrix(report_dir)
    if corr.empty:
        st.info("该实验没有保存 KPI 相关矩阵。")
        return
    corr = corr.apply(pd.to_numeric, errors="coerce")
    st.caption(f"结果来源：{label} · {source}")
    st.plotly_chart(
        px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                  text_auto=".2f", title="保存的 KPI 相关矩阵"),
        use_container_width=True,
    )
    download_csv(corr.reset_index().rename(columns={"index": "KPI"}),
                 "导出相关矩阵 CSV", "experiment_kpi_correlations.csv")
    tests = load_correlation_tests(report_dir)
    if not tests.empty:
        st.subheader("本实验对应的统计检验")
        st.dataframe(tests, hide_index=True, use_container_width=True)
        download_csv(tests, "导出统计检验 CSV", "kpi_correlation_tests.csv")
    else:
        st.info("该实验未保存显著性检验结果，不引用其他实验的 p 值。")
    st.caption("异常候选上的 KPI 相关性仅支持探索性分析，不能证明故障根因。")
