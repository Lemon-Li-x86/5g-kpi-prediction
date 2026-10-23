"""5G KPI offline Web system: data, training, evaluation and inference.

Run from project root:
    python -m streamlit run app.py

This app uses local offline gotsf-ds data. It is not connected to a live
radio network and does not infer verified equipment faults.
"""
from __future__ import annotations

import streamlit as st

from src.ui.dashboard_data import list_experiment_runs, run_report_dir
from src.ui.result_views import render_current_association, render_saved_association
from src.ui.web_workflow import (
    render_data_page,
    render_training_page,
    render_model_page,
    render_inference_page,
)

st.set_page_config(
    page_title="5G KPI 预测与异常检测系统",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("📡 5G KPI 分析系统")
st.sidebar.caption("本地数据 → 训练评估 → 实际预测与异常检测")
page = st.sidebar.radio(
    "功能模块",
    [
        "数据管理与预处理",
        "模型训练",
        "模型管理与实验记录",
        "KPI 预测",
        "异常检测",
        "KPI 关联分析",
    ],
)
st.sidebar.divider()
st.sidebar.caption("离线分析工具 · 不连接实时 free5GC/UERANSIM 或基站")
st.sidebar.caption("历史研究成绩与新实验、当前推理严格分开。")

if page == "数据管理与预处理":
    render_data_page()
elif page == "模型训练":
    render_training_page()
elif page == "模型管理与实验记录":
    render_model_page()
elif page == "KPI 预测":
    render_inference_page("forecast")
elif page == "异常检测":
    render_inference_page("anomaly")
elif page == "KPI 关联分析":
    st.title("KPI 关联分析")
    st.caption("对当前异常检测产生的共识候选做探索性分析，或查看指定历史实验的保存结果。")
    saved = st.session_state.get("kpi_anomaly_result")
    runs = list_experiment_runs("anomaly")
    options = (["本次异常检测"] if saved is not None else []) + [
        f"异常训练实验：{run}" for run in runs
    ] + ["历史正式实验"]
    selected = st.selectbox("数据来源", options, key="association_source")
    if selected == "本次异常检测":
        try:
            render_current_association(saved)
        except (ValueError, TypeError, KeyError) as exc:
            st.error(f"本次异常关联分析无法计算：{exc}")
    elif selected == "历史正式实验":
        render_saved_association(None, "历史正式实验")
    else:
        run = selected.split("：", 1)[1]
        render_saved_association(run_report_dir(run), run)

st.divider()
st.caption("所有模型推理均基于用户选择的本机可信权重；异常标记只是候选，不等于确诊故障。")
