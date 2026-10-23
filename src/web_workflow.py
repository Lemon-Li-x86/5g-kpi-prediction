"""Interactive offline workflow pages for data preparation, training and inference."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from src.model_registry import ROOT, KPIS, archive_models, list_models

JOBS = ROOT / "resources" / "jobs"
STATE = JOBS / "current.json"
LOG = JOBS / "current.log"


def _busy() -> bool:
    if not STATE.is_file():
        return False
    try:
        state = json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return state.get("status") == "running"


def _launch(task: str, model: str = ""):
    if _busy():
        st.error("已有任务正在执行，请等待任务结束。")
        return
    JOBS.mkdir(parents=True, exist_ok=True)
    # No shell; all tasks and arguments are from fixed allowlists.
    STATE.write_text(json.dumps({"task": task, "model": model, "status": "running"}), encoding="utf-8")
    with LOG.open("w", encoding="utf-8") as log:
        subprocess.Popen(
            [sys.executable, "-m", "src.web_jobs", task, *([model] if model else [])],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, close_fds=True,
        )
    st.success("任务已提交。通过下方任务状态查看进展。")


def _status():
    st.subheader("任务状态")
    if not STATE.is_file():
        st.caption("没有执行记录。")
        return
    try:
        info = json.loads(STATE.read_text(encoding="utf-8"))
        st.json(info)
    except (OSError, ValueError) as exc:
        st.error(str(exc))
    if LOG.is_file():
        content = LOG.read_text(encoding="utf-8", errors="replace")
        st.code(content[-7000:] or "尚无输出", language="text")
    st.button("刷新任务状态", key="refresh_job")


def render_data_page():
    st.title("数据获取与预处理")
    st.write("当前流水线支持 gotsf-ds 固定格式的四项 KPI，尚不支持任意 CSV 自动适配。")
    raw = ROOT / "data" / "raw"
    processed = ROOT / "data" / "processed" / "kpi_timeseries.csv"
    from src.preprocess_kpi_data import FILES
    st.dataframe(pd.DataFrame([
        {"KPI": kpi, "文件": filename, "是否存在": (raw / filename).exists()}
        for kpi, filename in FILES.items()
    ]), hide_index=True, use_container_width=True)
    st.write("预处理输出：", "已存在" if processed.is_file() else "尚不存在")
    c1, c2, c3 = st.columns(3)
    if c1.button("下载原始数据", disabled=_busy()):
        _launch("download")
    if c2.button("审计并预处理", disabled=_busy() or not all((raw / x).exists() for x in FILES.values())):
        _launch("preprocess")
    if c3.button("生成传统 ML 特征", disabled=_busy() or not processed.is_file()):
        _launch("features")
    _status()


def render_training_page():
    st.title("模型训练")
    st.caption("训练任务在独立 Python 进程运行；开始训练前会归档已有旧权重，训练成功后再保存新版本。")
    choice = st.selectbox("训练任务", ["传统 ML", "LSTM", "GRU", "Transformer", "异常检测"])
    task, model = {
        "传统 ML": ("train_ml", ""),
        "LSTM": ("train_dl", "lstm"),
        "GRU": ("train_dl", "gru"),
        "Transformer": ("train_dl", "transformer"),
        "异常检测": ("train_anomaly", ""),
    }[choice]
    processed = ROOT / "data" / "processed" / "kpi_timeseries.csv"
    feats = ROOT / "data" / "processed" / "ml_features.pkl"
    ready = processed.is_file() and (task != "train_ml" or feats.is_file())
    if not ready:
        st.warning("请先完成数据预处理" + ("及传统 ML 特征构建。" if task == "train_ml" else "。"))
    if st.button("开始训练", disabled=_busy() or not ready):
        _launch(task, model)
    _status()


def render_model_page():
    st.title("模型版本管理与推理")
    st.caption("只扫描 resources/models/ 与 models/ 中的本地可信模型；不接受外部上传的 pickle。")
    if st.button("归档现有未版本化模型"):
        added = archive_models(ROOT, reason="manual_web_archive")
        st.success(f"已归档 {len(added)} 个模型文件。")
    models = list_models(ROOT)
    if not models:
        st.info("尚未发现本地权重。GitHub 中的历史指标 CSV 并不是模型权重，请先在本机运行训练。")
        return
    options = {f"{idx}: {item.label}": item for idx, item in enumerate(models)}
    selected_label = st.selectbox("选择本地权重版本", list(options))
    selected = options[selected_label]
    st.write({"类型": selected.family, "模型": selected.model, "目标": selected.target, "版本": selected.version, "路径": selected.path})
    if selected.family == "anomaly":
        st.info("异常模型的推理接口需要完整波束顺序和前一小时上下文；当前 UI 暂不直接开放此入口。")
        return
    st.warning("仅加载由本机训练生成的可信权重；推理要求历史输入具有与训练一致的四项 KPI 和波束 ID。")
    uploaded = st.file_uploader("上传待预测历史 CSV（不保存到磁盘）", type=["csv"])
    if uploaded is None:
        return
    if uploaded.size > 50 * 1024 * 1024:
        st.error("文件超过 50 MB 限制。")
        return
    try:
        data = pd.read_csv(uploaded)
        required = {"Time", "base_station", "cell", "beam", *KPIS}
        missing = required - set(data.columns)
        if missing:
            st.error(f"缺少必要字段：{sorted(missing)}")
            return
        st.write(f"输入样本：{len(data):,} 行")
        if st.button("运行下一小时预测"):
            from src.inference import predict_kpi
            with st.spinner("正在执行推理"):
                result = predict_kpi(data, selected.path)
            st.dataframe(result.head(200), use_container_width=True, hide_index=True)
            st.download_button("导出预测结果 CSV", result.to_csv(index=False).encode("utf-8-sig"),
                               "kpi_inference.csv", "text/csv")
    except Exception as exc:
        st.error(f"无法执行推理：{exc}")
