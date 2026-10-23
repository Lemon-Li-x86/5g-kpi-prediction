"""Streamlit dashboard for the 5G KPI prediction and anomaly-analysis project.

Run from the project root:
    streamlit run app.py

The app visualises saved experiment outputs; it does not train models or connect
 to a live free5GC/UERANSIM network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.dashboard_data import (  # noqa: E402
    FIGURE_DIR,
    KPIS,
    REPORT_DIR,
    available_prediction_models,
    load_anomaly_events,
    load_anomaly_hourly_rates,
    load_anomaly_summary,
    load_correlation_matrix,
    load_correlation_tests,
    load_experiment_config,
    load_model_metrics,
    load_prediction_series,
    load_preprocessing_summary,
)


st.set_page_config(
    page_title="5G KPI Analysis System",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 1. Cached data access
# ============================================================

@st.cache_data(show_spinner=False)
def get_model_metrics() -> pd.DataFrame:
    return load_model_metrics()


@st.cache_data(show_spinner=False)
def get_prediction_series(model: str, kpi: str) -> pd.DataFrame:
    return load_prediction_series(model, kpi)


@st.cache_data(show_spinner=False)
def get_anomaly_summary() -> pd.DataFrame:
    return load_anomaly_summary()


@st.cache_data(show_spinner=False)
def get_hourly_rates() -> pd.DataFrame:
    return load_anomaly_hourly_rates()


@st.cache_data(show_spinner=False)
def get_events() -> pd.DataFrame:
    return load_anomaly_events()


@st.cache_data(show_spinner=False)
def get_corr_matrix():
    return load_correlation_matrix()


@st.cache_data(show_spinner=False)
def get_corr_tests() -> pd.DataFrame:
    return load_correlation_tests()


@st.cache_data(show_spinner=False)
def get_data_summary() -> dict:
    return load_preprocessing_summary()


@st.cache_data(show_spinner=False)
def get_anomaly_config() -> dict:
    return load_experiment_config()


def format_rate(value) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "N/A"


def metric_row(metrics: pd.DataFrame, kpi: str, model: str) -> pd.Series | None:
    if metrics.empty:
        return None

    selected = metrics[
        (metrics["target"].astype(str) == kpi)
        & (metrics["model"].astype(str).str.casefold() == model.casefold())
    ]

    if selected.empty:
        return None
    return selected.iloc[0]


def render_metric_cards(row: pd.Series | None) -> None:
    if row is None:
        st.info("当前没有该模型与 KPI 组合的已保存评估指标。")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{row['MAE']:.4f}")
    col2.metric("RMSE", f"{row['RMSE']:.4f}")
    col3.metric("R²", f"{row['R2']:.4f}")


def add_download_button(frame: pd.DataFrame, label: str, filename: str) -> None:
    if frame.empty:
        return
    st.download_button(
        label=label,
        data=frame.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
    )


# ============================================================
# 2. Sidebar navigation
# ============================================================

st.sidebar.title("📡 5G KPI 分析系统")
st.sidebar.caption("离线实验结果交互式展示")
page = st.sidebar.radio(
    "功能模块",
    [
        "项目总览",
        "数据管理与预处理",
        "模型训练",
        "模型管理与推理",
        "KPI 预测",
        "模型评估",
        "异常检测",
        "KPI 关联分析",
    ],
)

st.sidebar.divider()
st.sidebar.caption("数据来源：仓库内已保存的实验结果文件")
st.sidebar.caption("当前版本不连接实时 free5GC/UERANSIM 网络。")


# ============================================================
# 3. Overview
# ============================================================

if page == "项目总览":
    st.title("5G 网络 KPI 预测与异常检测系统")
    st.write(
        "本系统整合传统机器学习、深度学习 KPI 预测，以及多方法异常筛查和 KPI 关联分析结果。"
    )

    summary = get_data_summary()
    dataset = summary.get("dataset", {})
    split = summary.get("split", {})
    metrics = get_model_metrics()
    anomaly_config = get_anomaly_config()

    train_range = split.get("train_time", [])
    test_range = split.get("test_time", [])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("数据记录数", f"{dataset.get('rows', 0):,}" if dataset else "N/A")
    c2.metric("Beam 数量", f"{dataset.get('beam_count', 0):,}" if dataset else "N/A")
    c3.metric("模型数量", f"{metrics['model'].nunique()}" if not metrics.empty else "N/A")
    c4.metric("联合异常候选数", f"{anomaly_config.get('consensus_candidates', 'N/A')}")

    st.subheader("实验数据与评估范围")
    info_cols = st.columns(2)
    with info_cols[0]:
        st.markdown("**数据集**")
        st.write(f"时间索引：{dataset.get('time_min', 'N/A')} 至 {dataset.get('time_max', 'N/A')}")
        st.write(f"训练区间：{train_range if train_range else 'N/A'}")
        st.write(f"测试区间：{test_range if test_range else 'N/A'}")
        st.write("KPI：DLPRB、DLThpVol、DLThpTime、MR_number")
    with info_cols[1]:
        st.markdown("**系统模块**")
        st.write("1. KPI 预测结果查看")
        st.write("2. 模型 MAE / RMSE / R² 对比")
        st.write("3. 异常标记率与候选事件筛选")
        st.write("4. KPI 相关性及可用的显著性结果")

    st.warning(
        "范围说明：当前 Web 应用展示的是基于开源 KPI 数据集生成的离线实验结果。"
        "它不会从 free5GC、UERANSIM 或实时基站接口采集 KPI，也不会在页面打开时重新训练模型。"
    )

    st.subheader("已有实验产物")
    artifact_rows = []
    expected = [
        ("传统 ML 模型指标", REPORT_DIR / "model_metrics.csv"),
        ("ML/DL 汇总指标", REPORT_DIR / "week3" / "w3_ml_dl_all_metrics.csv"),
        ("传统 ML 预测曲线数据", REPORT_DIR / "prediction_timeseries.csv"),
        ("GRU 预测序列", REPORT_DIR / "w3_gru_hourly_predictions.csv"),
        ("异常检测方法汇总", REPORT_DIR / "w3_anomaly_method_summary.csv"),
        ("异常候选事件", REPORT_DIR / "w3_anomaly_candidate_events.csv"),
        ("W4 相关性显著性结果", REPORT_DIR / "w4_anomaly_kpi_correlation_tests.csv"),
    ]
    for name, path in expected:
        artifact_rows.append({
            "文件用途": name,
            "状态": "可用" if path.exists() else "尚未生成",
            "路径": str(path.relative_to(PROJECT_ROOT)),
        })
    st.dataframe(pd.DataFrame(artifact_rows), use_container_width=True, hide_index=True)


# ============================================================
# 4. KPI prediction
# ============================================================

elif page == "KPI 预测":
    st.title("KPI 预测结果")
    st.caption(
        "曲线展示测试集内按小时聚合的 KPI 平均值；下方 MAE、RMSE 和 R² 来自模型保存的测试集评估指标。"
    )

    models = available_prediction_models()
    metrics = get_model_metrics()

    if not models:
        st.error("未找到预测结果文件。请先运行相应的传统 ML / 深度学习实验脚本。")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        kpi = st.selectbox("KPI 指标", KPIS, index=0)
    with col2:
        default_model = "GRU" if "GRU" in models else models[0]
        model = st.selectbox("预测模型", models, index=models.index(default_model))

    series = get_prediction_series(model, kpi)
    row = metric_row(metrics, kpi, model)
    render_metric_cards(row)

    if series.empty:
        st.warning("没有找到这个模型和 KPI 对应的预测序列。")
    else:
        time_min = int(series["Time"].min())
        time_max = int(series["Time"].max())
        if time_min < time_max:
            selected_range = st.slider(
                "显示时间索引范围",
                min_value=time_min,
                max_value=time_max,
                value=(time_min, time_max),
            )
            visible = series[
                series["Time"].between(selected_range[0], selected_range[1])
            ]
        else:
            visible = series

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=visible["Time"], y=visible["actual"],
            mode="lines", name="实际值（小时均值）",
        ))
        fig.add_trace(go.Scatter(
            x=visible["Time"], y=visible["prediction"],
            mode="lines", name="预测值（小时均值）",
        ))
        fig.update_layout(
            title=f"{kpi}：{model} 预测与实际值",
            xaxis_title="时间索引（小时）",
            yaxis_title=f"{kpi}（数据集数值尺度）",
            hovermode="x unified",
            legend_title="序列",
        )
        st.plotly_chart(fig, use_container_width=True)
        add_download_button(visible, "下载当前预测序列 CSV", f"{model}_{kpi}_prediction.csv")

        st.caption(
            "注意：横轴是数据集中的小时索引，不是实际日期时间；曲线为跨 Beam 聚合值，不代表单个基站或 Beam 的预测。"
        )


# ============================================================
# 5. Model evaluation
# ============================================================

elif page == "模型评估":
    st.title("模型性能比较")
    metrics = get_model_metrics()

    if metrics.empty:
        st.error("未找到可用的模型评估指标 CSV。")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        kpi = st.selectbox("KPI 指标", KPIS, key="metric_kpi")
    with col2:
        metric = st.selectbox("评价指标", ["MAE", "RMSE", "R2"], index=0)

    selected = metrics[metrics["target"] == kpi].copy()
    selected = selected.dropna(subset=[metric])

    if metric in {"MAE", "RMSE"}:
        selected = selected.sort_values(metric, ascending=True)
        direction = "数值越低越好"
    else:
        selected = selected.sort_values(metric, ascending=False)
        direction = "数值越高越好"

    st.caption(f"{metric}：{direction}。指标来自已保存的测试集结果，不在页面加载时重训模型。")

    fig = px.bar(
        selected,
        x="model",
        y=metric,
        color="family" if "family" in selected.columns else None,
        hover_data=[column for column in ["test_samples", "family"] if column in selected.columns],
        title=f"{kpi}：不同模型的 {metric} 比较",
    )
    fig.update_layout(
        xaxis_title="模型",
        yaxis_title=metric,
        xaxis_tickangle=-30,
        legend_title="模型类别",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(selected, use_container_width=True, hide_index=True)
    add_download_button(selected, "下载当前 KPI 的模型比较表", f"{kpi}_model_metrics.csv")


# ============================================================
# 6. Anomaly detection
# ============================================================

elif page == "异常检测":
    st.title("异常检测结果")
    summary = get_anomaly_summary()
    hourly = get_hourly_rates()
    events = get_events()

    if summary.empty:
        st.error("未找到异常检测方法汇总文件。")
        st.stop()

    st.subheader("检测方法标记率")
    display_summary = summary.copy()
    display_summary["标记率"] = display_summary["flag_rate"].map(format_rate)
    display_summary = display_summary.rename(columns={
        "method": "检测方法",
        "evaluated_samples": "评估样本数",
        "flagged_samples": "标记样本数",
        "evaluation_scope": "评估范围",
    })

    st.dataframe(
        display_summary[[
            "检测方法", "评估样本数", "标记样本数", "标记率", "评估范围"
        ]],
        use_container_width=True,
        hide_index=True,
    )

    if "OneClassSVM" in set(summary["method"].astype(str)):
        st.warning(
            "One-Class SVM 的结果来自抽样测试子集；其他主要方法使用完整测试集。"
            "因此其标记率不应被当作完全同条件的横向比较。"
        )

    plot_summary = summary.copy()
    plot_summary["flag_rate_pct"] = 100 * plot_summary["flag_rate"]
    fig = px.bar(
        plot_summary,
        x="method",
        y="flag_rate_pct",
        color="evaluation_scope" if "evaluation_scope" in plot_summary.columns else None,
        title="各检测方法的标记率",
        hover_data=[column for column in ["evaluated_samples", "flagged_samples"] if column in plot_summary.columns],
    )
    fig.update_layout(xaxis_title="检测方法", yaxis_title="标记率（%）")
    st.plotly_chart(fig, use_container_width=True)

    if not hourly.empty:
        st.subheader("测试期间的逐小时异常标记率")
        rate_columns = [column for column in hourly.columns if column != "Time"]
        hourly_long = hourly.melt(
            id_vars="Time",
            value_vars=rate_columns,
            var_name="检测方法",
            value_name="标记率",
        )
        hourly_long["标记率（%）"] = 100 * hourly_long["标记率"]
        fig = px.line(
            hourly_long,
            x="Time",
            y="标记率（%）",
            color="检测方法",
            title="逐小时异常标记率变化",
        )
        fig.update_layout(xaxis_title="时间索引（小时）", yaxis_title="标记率（%）")
        st.plotly_chart(fig, use_container_width=True)

    if not events.empty:
        st.subheader("异常候选事件")
        st.caption(
            "当前 CSV 只保存了按 Isolation Forest 分数排序的最多 2,000 条联合候选事件，"
            "不是全部候选事件。"
        )

        filtered = events.copy()
        filter_cols = st.columns(3)

        if "top_deviation_kpi" in filtered.columns:
            choices = ["全部"] + sorted(filtered["top_deviation_kpi"].dropna().astype(str).unique().tolist())
            selected_kpi = filter_cols[0].selectbox("主要偏离 KPI", choices)
            if selected_kpi != "全部":
                filtered = filtered[filtered["top_deviation_kpi"].astype(str) == selected_kpi]

        if "base_station" in filtered.columns:
            stations = sorted(filtered["base_station"].dropna().unique().tolist())
            station_choices = ["全部"] + [str(value) for value in stations]
            selected_station = filter_cols[1].selectbox("基站 ID", station_choices)
            if selected_station != "全部":
                filtered = filtered[filtered["base_station"].astype(str) == selected_station]

        if "Time" in filtered.columns and not filtered.empty:
            min_time = int(filtered["Time"].min())
            max_time = int(filtered["Time"].max())
            if min_time < max_time:
                time_range = filter_cols[2].slider(
                    "时间索引",
                    min_value=min_time,
                    max_value=max_time,
                    value=(min_time, max_time),
                    key="event_time_range",
                )
                filtered = filtered[filtered["Time"].between(time_range[0], time_range[1])]

        preferred_columns = [
            "Time", "base_station", "cell", "beam",
            "isolation_forest_score", "top_deviation_kpi",
            "n_sigma_kpis", "n_iqr_kpis",
            *KPIS,
        ]
        visible_columns = [column for column in preferred_columns if column in filtered.columns]
        st.dataframe(filtered[visible_columns], use_container_width=True, hide_index=True)
        add_download_button(filtered, "下载筛选后的候选事件", "filtered_anomaly_candidates.csv")
    else:
        st.info("没有找到候选事件 CSV。")


# ============================================================
# 7. KPI correlation and significance
# ============================================================

elif page == "KPI 关联分析":
    st.title("KPI 关联分析")
    corr, source_label = get_corr_matrix()
    tests = get_corr_tests()

    if corr.empty:
        st.info("未找到 KPI 相关性矩阵。")
        st.stop()

    st.caption(f"当前矩阵来源：{source_label}")

    p_lookup = {}
    n_values = set()
    if not tests.empty:
        p_col = "holm_adjusted_p_value" if "holm_adjusted_p_value" in tests.columns else (
            "holm_adjusted_cluster_p" if "holm_adjusted_cluster_p" in tests.columns else None
        )
        for row in tests.to_dict("records"):
            pair = tuple(sorted((str(row["KPI_1"]), str(row["KPI_2"]))))
            if p_col:
                p_lookup[pair] = row.get(p_col, np.nan)
            if "n_candidate_beam_hours" in row:
                n_values.add(int(row["n_candidate_beam_hours"]))

    corr = corr.apply(pd.to_numeric, errors="coerce")
    annotations = []

    for row_name in corr.index:
        labels = []
        for col_name in corr.columns:
            value = corr.loc[row_name, col_name]
            if not np.isfinite(value):
                labels.append("NA")
                continue

            label = f"{value:.2f}"
            pair = tuple(sorted((str(row_name), str(col_name))))
            p_adj = p_lookup.get(pair, np.nan)

            if row_name != col_name and np.isfinite(p_adj):
                if p_adj < 0.001:
                    label += " ***"
                elif p_adj < 0.01:
                    label += " **"
                elif p_adj < 0.05:
                    label += " *"
                else:
                    label += " ns"
            labels.append(label)
        annotations.append(labels)

    fig = go.Figure(data=go.Heatmap(
        z=corr.to_numpy(),
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        zmin=-1,
        zmax=1,
        colorscale="RdBu",
        reversescale=True,
        text=annotations,
        texttemplate="%{text}",
        hovertemplate="KPI X: %{x}<br>KPI Y: %{y}<br>Pearson r: %{z:.3f}<extra></extra>",
        colorbar_title="Pearson r",
    ))
    fig.update_layout(title="KPI Pearson 相关性矩阵", xaxis_title="KPI", yaxis_title="KPI")
    st.plotly_chart(fig, use_container_width=True)

    if tests.empty:
        st.warning(
            "当前仓库还没有 W4 相关性显著性检验结果。上图仅展示 W3 的探索性相关系数，"
            "不会显示或推断 p 值。运行 W4 统计诊断脚本后，刷新页面即可读取检验结果。"
        )
    else:
        st.markdown(
            "显著性标记：* p < 0.05，** p < 0.01，*** p < 0.001，ns 表示不显著。"
            "标记使用 Holm 校正后的基站聚类稳健 p 值。"
        )
        if n_values:
            st.caption(
                f"候选样本量：{min(n_values):,} 个 beam-hour；"
                "推断单位按基站聚类，置信区间和 p 值仍受基站簇数量限制。"
            )
        st.dataframe(tests, use_container_width=True, hide_index=True)
        add_download_button(tests, "下载相关性显著性检验结果", "kpi_correlation_tests.csv")

    st.info(
        "相关性描述 KPI 的共同变化，不等于因果关系。由于样本是由异常检测规则筛选出的候选事件，"
        "结果属于探索性关联分析，不应直接当作已确认的网络故障根因。"
    )


# ============================================================
# 8. Interactive workflow modules
# ============================================================
if page in ("数据管理与预处理", "模型训练", "模型管理与推理"):
    from src.ui.web_workflow import (
        render_data_page, render_training_page, render_model_page,
    )
    if page == "数据管理与预处理":
        render_data_page()
    elif page == "模型训练":
        render_training_page()
    else:
        render_model_page()


# ============================================================
# 8. Footer
# ============================================================

st.sidebar.divider()
st.sidebar.caption("W7 system integration prototype")
st.sidebar.caption("结果文件更新后，可使用页面菜单刷新缓存或重新启动应用。")
