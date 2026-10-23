"""Data preparation, training and versioned inference pages with live job logs."""
from __future__ import annotations

import html
import hashlib
import io
import os
import json
import re
import secrets
import socket
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from src.services.model_registry import ROOT, KPIS, archive_models, list_models
from src.core.config import ML_HISTORY_HOURS
from src.services.process_env import python_job_env
from src.core.training_config import normalize_params, DL_DEFAULTS, ANOMALY_DEFAULTS
from src.ui.dashboard_data import list_experiment_runs, run_report_dir
from src.ui.result_views import (
    render_forecast_evaluation, render_anomaly_training_results,
    render_forecast_inference_result, render_anomaly_inference_result,
)

JOBS = ROOT / "resources" / "jobs"
STATE = JOBS / "current.json"
LOG = JOBS / "current.log"
SESSION = JOBS / "stream_session.json"
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _read_state() -> dict:
    if not STATE.exists():
        return {}
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # A worker might be in the middle of rewriting the state file.
        return {"status": "updating"}


def _busy() -> bool:
    return _read_state().get("status") in {"running", "updating"}


def _tail_log(limit: int = 150_000) -> str:
    if not LOG.is_file():
        return "等待日志文件生成..."
    try:
        with LOG.open("rb") as file:
            size = file.seek(0, 2)
            file.seek(max(0, size - limit))
            output = file.read().decode("utf-8", errors="replace")
    except OSError as exc:
        return f"暂时无法读取日志：{exc}"
    output = ANSI_ESCAPE.sub("", output).replace("\r", "\n")
    if size > limit:
        output = "[仅展示最近约 150 KB；完整日志保存在 resources/jobs/current.log]\n" + output
    return output if output.strip() else "任务正在运行，尚未有新的终端输出..."


def _read_session() -> dict:
    if not SESSION.is_file():
        return {}
    try:
        return json.loads(SESSION.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def _worker_alive(pid: object) -> bool:
    """Check a worker PID without sending a signal to it on Windows.

    os.kill(pid, 0) is intentionally used only on POSIX; Windows uses
    GetExitCodeProcess so this check cannot terminate the training worker.
    """
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        open_process.restype = wintypes.HANDLE
        get_exit_code = kernel32.GetExitCodeProcess
        get_exit_code.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        get_exit_code.restype = wintypes.BOOL
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        handle = open_process(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            if not get_exit_code(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == STILL_ACTIVE
        finally:
            close_handle(handle)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _launch(task: str, model: str = "", params: dict | None = None) -> None:
    if _busy():
        st.warning("已有任务正在执行，请勿重复启动。")
        return

    try:
        params = normalize_params(task, params, model)
    except ValueError as exc:
        st.error(f"训练参数无效：{exc}")
        return

    JOBS.mkdir(parents=True, exist_ok=True)
    # Obtain a localhost TCP port for this run; the supervisor binds it shortly.
    with socket.socket() as temp:
        temp.bind(("127.0.0.1", 0))
        port = temp.getsockname()[1]
    token = secrets.token_urlsafe(24)
    STATE.write_text(
        json.dumps({"task": task, "model": model, "parameters": params, "status": "running"}, ensure_ascii=False),
        encoding="utf-8",
    )

    env = python_job_env()
    try:
        with (JOBS / "stream_errors.log").open("w", encoding="utf-8") as diagnostics:
            worker = subprocess.Popen(
                [sys.executable, "-u", "-m", "src.services.web_stream_worker", task,
                 *([model] if model else []), "--port", str(port), "--token", token,
                 "--params-json", json.dumps(params, ensure_ascii=False)],
                cwd=ROOT,
                stdout=diagnostics,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
                close_fds=True,
                # A new Windows process group prevents stopping the Streamlit
                # console with Ctrl+C from also cancelling the training worker.
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
        # Record the real worker PID only after a successful process spawn.
        # It allows the UI to distinguish an expired/stale session from a live one.
        SESSION.write_text(
            json.dumps({"port": port, "token": token, "task": task,
                        "model": model, "pid": worker.pid}),
            encoding="utf-8",
        )
    except Exception as exc:
        STATE.write_text(
            json.dumps({"task": task, "model": model, "status": "failed", "error": str(exc)}, ensure_ascii=False),
            encoding="utf-8",
        )
        st.error(f"启动失败：{exc}")
        return
    st.success("任务已启动：终端输出通过 WebSocket 实时推送，日志在后台分批写入。")


def _terminal_html(content: str) -> str:
    escaped = html.escape(content)
    return f"""
    <!doctype html>
    <html><head><meta charset="utf-8"/>
    <style>
      html, body {{margin: 0; padding: 0; background: #101319;}}
      #terminal {{
        box-sizing: border-box; height: 360px; overflow-y: auto;
        background: #101319; color: #d4e6d2;
        padding: 14px 16px; border-radius: 9px;
        border: 1px solid #354035;
        font: 12.5px/1.55 Consolas, 'Cascadia Code', monospace;
      }}
      #terminal pre {{margin: 0; white-space: pre-wrap; overflow-wrap: anywhere;
        font: inherit;}}
      #terminal::-webkit-scrollbar {{width: 10px;}}
      #terminal::-webkit-scrollbar-thumb {{background: #516056; border-radius: 8px;}}
    </style></head>
    <body><div id="terminal"><pre>{escaped}</pre></div>
    <script>
      const pane = document.getElementById('terminal');
      pane.scrollTop = pane.scrollHeight;
    </script></body></html>
    """


def _stream_terminal_html(port: int, token: str, fallback: str, initial_status: str) -> str:
    """Read-only local WebSocket terminal. No log polling or command input."""
    endpoint = f"ws://127.0.0.1:{port}/?token={token}"
    template = r"""
<!doctype html>
<html><head><meta charset="utf-8"/>
<style>
 html,body{margin:0;padding:0;background:#11151b;color:#d1dfd4;font-family:Consolas,'Cascadia Code',monospace}
 #head{display:flex;align-items:center;justify-content:space-between;padding:9px 13px;background:#1a2420;border:1px solid #355043;border-bottom:0;border-radius:9px 9px 0 0;font-size:12px}
 #dot{color:#ffca68}
 #term{box-sizing:border-box;height:346px;overflow:auto;padding:13px 14px;background:#101319;border:1px solid #355043;border-radius:0 0 9px 9px}
 pre{margin:0;white-space:pre-wrap;overflow-wrap:anywhere;font:12.5px/1.55 Consolas,'Cascadia Code',monospace;color:#bbdfbd}
 #term::-webkit-scrollbar{width:10px}#term::-webkit-scrollbar-thumb{background:#465a4c;border-radius:7px}
 #note{font:11px/1.4 sans-serif;color:#a5b3ad;padding-top:6px}
</style></head>
<body>
 <div id="head"><span>● LOCAL TRAINING CONSOLE</span><span id="dot">连接中...</span></div>
 <div id="term"><pre id="text"></pre></div>
 <div id="note">↑ 滚动可查看旧记录；停在底部时自动跟随输出。仅显示日志，不接收命令输入。</div>
<script>
 const ENDPOINT=__WS_ENDPOINT__;
 const FALLBACK=__FALLBACK__;
 const INITIAL_STATUS=__INITIAL_STATUS__;
 const pane=document.getElementById('term');
 const text=document.getElementById('text');
 const dot=document.getElementById('dot');
 let terminalText='';
 let complete=false;
 let fallbackUsed=false;
 let retryCount=0;
 let queue='';
 let frameScheduled=false;
 const MAX_CHARS=200000;
 function flush(){
   frameScheduled=false;
   if(!queue)return;
   const follow=(pane.scrollHeight-pane.scrollTop-pane.clientHeight)<75;
   terminalText+=queue.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g,'').replace(/\r/g,'\n');
   queue='';
   if(terminalText.length>MAX_CHARS)terminalText=terminalText.slice(-MAX_CHARS);
   text.textContent=terminalText;
   if(follow)pane.scrollTop=pane.scrollHeight;
 }
 function push(s){queue+=s;if(!frameScheduled){frameScheduled=true;requestAnimationFrame(flush);}}
 function reset(s){terminalText='';queue='';text.textContent='';push(s);}
 function start(){
  if(complete)return;
  const ws=new WebSocket(ENDPOINT);
  ws.onopen=()=>{retryCount=0;dot.textContent='● 实时连接';dot.style.color='#7fe19d';};
  ws.onmessage=(event)=>{
   let msg;try{msg=JSON.parse(event.data)}catch(_){return}
   if(msg.type==='reset')reset(msg.data||'');
   else if(msg.type==='text')push(msg.data||'');
   else if(msg.type==='status'){
     complete=true;
     dot.textContent=(msg.status==='completed'?'✓ 训练完成':'✕ 任务结束：'+msg.status);
     dot.style.color=(msg.status==='completed'?'#7fe19d':'#ff8888');
   }
  };
  ws.onerror=()=>{};
  ws.onclose=()=>{
   if(complete)return;
   dot.textContent='● 正在重新连接...';dot.style.color='#ffca68';
   retryCount++;
   if(retryCount>=3 && (INITIAL_STATUS==='completed'||INITIAL_STATUS==='failed')){
     complete=true;
     if(!terminalText)reset(FALLBACK||'任务已结束，未能恢复历史输出。');
     dot.textContent=(INITIAL_STATUS==='completed'?'✓ 已完成（本地日志）':'✕ 已失败（本地日志）');
     dot.style.color=(INITIAL_STATUS==='completed'?'#7fe19d':'#ff8888');
     return;
   }
   if(retryCount>=10){
     complete=true;
     if(!terminalText)reset(FALLBACK||'实时服务暂时不可用，请查看 resources/jobs/stream_errors.log');
     dot.textContent='● 实时连接不可用';dot.style.color='#ff8888';
     return;
   }
   setTimeout(start,Math.min(4000,500+retryCount*500));
  };
 }
 start();
</script></body></html>
"""
    return (template
        .replace("__WS_ENDPOINT__", json.dumps(endpoint))
        .replace("__FALLBACK__", json.dumps(fallback, ensure_ascii=False))
        .replace("__INITIAL_STATUS__", json.dumps(initial_status))
    )


@st.fragment(run_every="2s")
def _legacy_status() -> None:
    """Compatibility with jobs launched before the live-stream change."""
    st.info("当前任务由旧版程序启动，仍使用每 2 秒日志轮询；下一次任务自动采用 WebSocket。")
    info = _read_state()
    st.caption(f"任务状态：{info.get('status', 'unknown')}")
    st.components.v1.html(_terminal_html(_tail_log()), height=370, scrolling=False)


def _status() -> None:
    st.subheader("任务状态 · 实时训练终端")
    info = _read_state()
    status = info.get("status", "idle")
    name = info.get("task", "无")
    model = info.get("model") or ""
    st.caption(f"任务：{name} {model} ｜ 当前状态：{status}")
    if info.get("error"):
        st.error(str(info["error"]))
    if info.get("output_dir"):
        st.caption(f"独立实验目录：{info['output_dir']}")
    if info.get("parameters"):
        with st.expander("查看本次任务实际参数"):
            st.json(info["parameters"])

    session = _read_session()
    session_matches = (
        bool(session.get("port") and session.get("token"))
        and (session.get("task"), session.get("model")) ==
            (info.get("task"), info.get("model") or "")
    )
    # The WebSocket worker intentionally exits after a short post-job grace
    # period. Once a job ends, show its durable logfile directly rather than
    # trying to reconnect to an already closed WebSocket server.
    if status != "running":
        st.components.v1.html(_terminal_html(_tail_log()), height=370, scrolling=False)
    elif session_matches and _worker_alive(session.get("pid")):
        st.components.v1.html(
            _stream_terminal_html(int(session["port"]), str(session["token"]), "", status),
            height=410,
            scrolling=False,
        )
    elif session_matches:
        st.warning(
            "实时日志工作器已退出，或这是旧版本留下的任务记录。"
            "以下显示已写入的日志。请确认没有训练进程在运行后，再标记该任务已中断。"
        )
        st.components.v1.html(_terminal_html(_tail_log()), height=370, scrolling=False)
        if st.button("确认任务已停止，清理失效状态", key="clear_stale_training"):
            STATE.write_text(
                json.dumps({**info, "status": "interrupted",
                            "error": "实时工作器已退出；任务是否完成请核对日志。"},
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            st.rerun()
    else:
        _legacy_status()

    st.caption("终端走实时消息流；current.log 在后台每 2 秒或累计 64 KB 后写入。")
    if LOG.is_file():
        st.download_button(
            "下载当前已写入的日志", LOG.read_bytes(),
            file_name="training.log", mime="text/plain",
            key="download_training_log",
        )
    st.caption("任务结束后如需更新按钮状态，可手动刷新页面。")


def render_data_page() -> None:
    st.title("数据获取与预处理")
    st.write("当前流水线支持 gotsf-ds 固定格式的四项 KPI，尚不支持任意 CSV 自动适配。")
    raw = ROOT / "data" / "raw"
    processed = ROOT / "data" / "processed" / "kpi_timeseries.csv"
    from src.data.preprocess_kpi_data import FILES
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


def render_training_page() -> None:
    st.title("模型训练")
    st.caption(
        "在 Web 中设置训练参数；保存本次配置并交给后台进程执行。"
        "已运行的任务不受输入框后续修改影响。"
    )
    choice = st.selectbox(
        "选择训练任务",
        ["传统 ML", "LSTM", "GRU", "Transformer", "异常检测"],
    )
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
        st.warning(
            "请先完成数据预处理"
            + ("及传统 ML 特征构建。" if task == "train_ml" else "。")
        )

    with st.form("training_parameters"):
        params = {}
        if task == "train_dl":
            d = DL_DEFAULTS
            st.subheader("训练参数")
            c1, c2, c3 = st.columns(3)
            params["epochs"] = int(c1.number_input(
                "最大训练轮次 (Epochs)", min_value=1, max_value=500,
                value=d["epochs"], step=1,
            ))
            params["batch_size"] = int(c2.number_input(
                "批量大小 (Batch Size)", min_value=1, max_value=4096,
                value=d["batch_size"], step=32,
            ))
            params["lr"] = float(c3.number_input(
                "学习率 (Learning Rate)", min_value=0.000001, max_value=0.1,
                value=d["lr"], step=0.0001, format="%.6f",
            ))
            c4, c5, c6 = st.columns(3)
            params["hidden"] = int(c4.number_input(
                "隐藏层维度 (Hidden Size)", min_value=4, max_value=1024,
                value=d["hidden"], step=4,
            ))
            params["patience"] = int(c5.number_input(
                "早停耐心 (Patience)", min_value=1, max_value=100,
                value=d["patience"], step=1,
            ))
            params["device"] = c6.selectbox(
                "计算设备", ["auto", "cpu", "cuda"], index=0,
                help="auto 优先选择可用的 CUDA；CUDA 不可用时选择 cuda 会报错。",
            )
            with st.expander("高级选项：时间窗口、样本量与随机种子"):
                a1, a2, a3 = st.columns(3)
                params["seq_len"] = int(a1.number_input(
                    "输入窗口（小时）", min_value=1, max_value=168,
                    value=d["seq_len"], step=1,
                ))
                params["seasonal_lag"] = int(a2.number_input(
                    "季节参考间隔（小时）", min_value=0, max_value=336,
                    value=d["seasonal_lag"], step=1,
                ))
                params["seed"] = int(a3.number_input(
                    "随机种子", min_value=0, max_value=2147483647,
                    value=d["seed"], step=1,
                ))
                a4, a5, a6 = st.columns(3)
                params["max_train"] = int(a4.number_input(
                    "训练样本上限", min_value=1, max_value=1000000,
                    value=d["max_train"], step=1000,
                ))
                params["max_val"] = int(a5.number_input(
                    "验证样本上限", min_value=1, max_value=1000000,
                    value=d["max_val"], step=1000,
                ))
                params["max_test"] = int(a6.number_input(
                    "测试样本上限（0=全部）", min_value=0, max_value=1000000,
                    value=d["max_test"], step=1000,
                ))
                st.caption(
                    "训练/验证/测试时间边界固定，不通过 Web 修改。"
                    "截取测试集可能使指标与完整测试集结果不可直接比较。"
                )
            if model == "transformer":
                st.caption("Transformer 的隐藏层维度必须能被 4 个注意力头整除。")
        elif task == "train_anomaly":
            d = ANOMALY_DEFAULTS
            st.subheader("异常检测参数")
            left, mid = st.columns(2)
            params["if_contamination"] = float(left.number_input(
                "Isolation Forest 预期异常占比", min_value=0.00001,
                max_value=0.5, value=d["if_contamination"],
                step=0.005, format="%.5f",
            ))
            params["if_train_sample"] = int(mid.number_input(
                "Isolation Forest 训练样本", min_value=100, max_value=1000000,
                value=d["if_train_sample"], step=1000,
            ))
            with st.expander("高级选项：SVM 抽样与报告"):
                a1, a2, a3 = st.columns(3)
                params["svm_train_sample"] = int(a1.number_input(
                    "SVM 训练样本", min_value=100, max_value=100000,
                    value=d["svm_train_sample"], step=100,
                ))
                params["svm_test_sample"] = int(a2.number_input(
                    "SVM 测试样本", min_value=100, max_value=1000000,
                    value=d["svm_test_sample"], step=1000,
                ))
                params["max_events"] = int(a3.number_input(
                    "最多保存异常事件", min_value=1, max_value=1000000,
                    value=d["max_events"], step=100,
                ))
                params["seed"] = int(st.number_input(
                    "随机种子", min_value=0, max_value=2147483647,
                    value=d["seed"], step=1,
                ))
            st.caption(
                "结果写入新的实验目录，历史异常检测图表不会自动切换到本次版本。"
            )
        else:
            st.subheader("传统 ML 训练参数")
            params["quick"] = st.checkbox(
                "快速验收模式（抽样训练与测试，成绩不用于正式实验报告）",
                value=True,
            )
            if params["quick"]:
                c1, c2 = st.columns(2)
                params["max_train"] = int(c1.number_input(
                    "最多训练样本", min_value=100, max_value=1000000,
                    value=512, step=100,
                ))
                params["max_test"] = int(c2.number_input(
                    "最多测试样本（0 = 全部）", min_value=0, max_value=1000000,
                    value=512, step=100,
                ))
                st.caption(
                    "快速模式用于验证 Web 后台、五种传统 ML 模型及保存链路，"
                    "抽样成绩不能与历史全量测试成绩比较。"
                )
            else:
                st.warning(
                    "正式模式将运行所有传统 ML 模型的全量训练与测试，"
                    "耗时和内存需求可能较高。"
                )
            st.info(
                "Web 会把报告、图表和权重保存到本次独立实验目录；"
                "不会覆盖 GitHub 中的历史 reports/、figures/ 或 models/。"
            )
        submitted = st.form_submit_button(
            "开始训练并保存配置", disabled=_busy() or not ready,
        )

    if submitted:
        try:
            checked = normalize_params(task, params, model)
        except ValueError as exc:
            st.error(f"参数检查失败：{exc}")
        else:
            _launch(task, model, checked)
    st.caption("训练期间保持配置不可更改；下次训练可以使用新的设置。")
    _status()

    # Read the durable experiment artifacts after the background worker has
    # completed; the training algorithm has already saved these files.
    info = _read_state()
    if info.get("status") == "completed" and info.get("task") in {
        "train_ml", "train_dl", "train_anomaly",
    }:
        output = Path(info.get("output_dir", ""))
        reports = output / "reports"
        if output.is_dir() and reports.is_dir():
            st.divider()
            st.subheader("最近一次训练的测试评估与保存结果")
            st.caption(f"实验版本：{output.name} · 已保存到：{reports}")
            st.info("快速训练只用于功能验收，不能替代此前完整训练的正式实验成绩。")
            if info["task"] == "train_anomaly":
                render_anomaly_training_results(reports, output.name)
            else:
                render_forecast_evaluation(reports, output.name)
    elif info.get("status") == "running":
        st.caption("训练结束后点击下方按钮刷新，即可在本页面查看本次测试指标、曲线及图表。")
    if st.button("刷新训练状态及评估结果", key="refresh_training_results"):
        st.rerun()


def render_model_page() -> None:
    """View local model versions and immutable historic/new experiment scores."""
    st.title("模型管理与实验记录")
    st.caption("模型权重只从本机可信目录枚举；正式历史对比不可被快速训练覆盖。")
    if st.button("归档现有未版本化模型", disabled=_busy()):
        saved = archive_models(ROOT, reason="manual_web_archive")
        st.success(f"已归档 {len(saved)} 个模型文件。")

    models = list_models(ROOT)
    if models:
        st.subheader("本地模型权重")
        frame = pd.DataFrame([{
            "版本": m.version, "算法": m.model, "类型": m.family,
            "预测目标": m.target, "文件": Path(m.path).name, "已归档": m.archived,
        } for m in models])
        st.dataframe(frame, hide_index=True, use_container_width=True)
        st.caption("在 KPI 预测或异常检测页面选择模型并进行推理。")
    else:
        st.info("本机尚未发现可用权重。历史报告并不包含模型权重。")

    st.divider()
    st.subheader("实验评价记录")
    versions = list_experiment_runs("forecast") + list_experiment_runs("anomaly")
    sources = ["历史正式实验（固定对比）"] + [f"新实验：{run}" for run in sorted(versions, reverse=True)]
    selected = st.selectbox("选择已有实验", sources, key="model_history_run")
    if selected.startswith("新实验："):
        run_id = selected.split("：", 1)[1]
        report_dir = run_report_dir(run_id)
        if run_id in list_experiment_runs("anomaly"):
            render_anomaly_training_results(report_dir, run_id)
        else:
            render_forecast_evaluation(report_dir, run_id)
    else:
        st.caption("以下仅是第 4～6 周已保存的正式历史成绩，不会随新模型训练而更新。")
        render_forecast_evaluation(None, "历史正式实验")


def _generate_demo_files(forecast_hours: int = ML_HISTORY_HOURS) -> None:
    """Reuse the same validated data preparation as the CLI demo generator."""
    from src.data.prepare_demo_inputs import COLUMNS, prepare_demo_frames
    source = ROOT / "data" / "processed" / "kpi_timeseries.csv"
    if not source.is_file():
        raise FileNotFoundError("请先在数据管理页面生成 data/processed/kpi_timeseries.csv")
    data = pd.read_csv(source, usecols=COLUMNS)
    forecast, anomaly = prepare_demo_frames(data, forecast_hours=forecast_hours)
    destination = ROOT / "resources" / "demo_inputs"
    destination.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(destination / "forecast_history.csv", index=False)
    anomaly.to_csv(destination / "anomaly_history.csv", index=False)


def render_inference_page(kind: str) -> None:
    """Real model inference; never substitute historical dashboard output."""
    if kind not in {"forecast", "anomaly"}:
        raise ValueError("Unknown inference kind")

    is_anomaly = kind == "anomaly"
    st.title("异常检测" if is_anomaly else "KPI 预测")
    st.caption(
        "选择本机实际权重和输入 CSV，立即执行计算并显示本次结果。"
        " 这里不使用历史实验指标替代推理。"
    )
    models = list_models(ROOT, family="anomaly") if is_anomaly else (
        list_models(ROOT, family="ml") + list_models(ROOT, family="dl")
    )
    if not models:
        st.warning("没有找到对应模型权重。请先到模型训练页面训练并发布模型。")
        return
    options = {f"{i} · {m.label}": m for i, m in enumerate(models)}
    chosen = st.selectbox("选择模型权重", list(options), key=f"{kind}_model")
    selected = options[chosen]
    st.caption(f"模型版本：{selected.version} ｜ 模型文件：{Path(selected.path).name}")

    st.subheader("输入 KPI 数据")
    input_mode = st.radio(
        "数据来源", ["系统演示数据", "上传 CSV"],
        horizontal=True, key=f"{kind}_input_mode",
    )
    payload = None
    if input_mode == "系统演示数据":
        demo_dir = ROOT / "resources" / "demo_inputs"
        target = demo_dir / ("anomaly_history.csv" if is_anomaly else "forecast_history.csv")
        hours = ML_HISTORY_HOURS
        if not is_anomaly:
            hours = int(st.number_input(
                "生成预测历史的小时数（传统 ML 至少 169）",
                min_value=ML_HISTORY_HOURS, max_value=1000, value=ML_HISTORY_HOURS,
                key="forecast_demo_hours",
            ))
        if st.button("从本机预处理数据生成演示 CSV", key=f"{kind}_generate_demo"):
            try:
                with st.spinner("正在从已处理 KPI 数据生成演示输入"):
                    _generate_demo_files(forecast_hours=hours)
                st.success("演示 CSV 已生成；不修改训练数据、历史实验或模型权重。")
            except Exception as exc:
                st.error(f"生成演示输入失败：{exc}")
        if target.is_file():
            payload = target.read_bytes()
            st.caption(f"当前文件：{target.relative_to(ROOT)}")
            st.download_button(
                "下载演示输入 CSV", payload,
                file_name=target.name, mime="text/csv",
                key=f"{kind}_demo_download",
            )
        else:
            st.info("当前没有演示 CSV，请先点击上面的生成按钮。")
    else:
        uploaded = st.file_uploader(
            "上传历史 KPI CSV", type=["csv"], key=f"{kind}_upload",
        )
        if uploaded is not None:
            payload = uploaded.getvalue()

    if payload is None:
        return
    if len(payload) > 50 * 1024 * 1024:
        st.error("输入 CSV 超过 50 MB 的当前上传限制。")
        return
    try:
        data = pd.read_csv(io.BytesIO(payload))
        required = {"Time", "base_station", "cell", "beam", *KPIS}
        missing = required - set(data.columns)
        if missing:
            st.error(f"缺少必要字段：{sorted(missing)}")
            return
        st.caption(f"输入行数：{len(data):,} ｜ 时间索引范围：{data['Time'].min()}～{data['Time'].max()}")
    except Exception as exc:
        st.error(f"无法读取 CSV：{exc}")
        return

    signature = (selected.path, hashlib.sha256(payload).hexdigest())
    key = "kpi_anomaly_result" if is_anomaly else "kpi_forecast_result"
    if st.button("运行异常检测" if is_anomaly else "预测下一小时 KPI",
                 key=f"{kind}_run_inference"):
        try:
            from src.models.inference import predict_kpi, detect_anomalies
            with st.spinner("正在运行模型推理"):
                result = (detect_anomalies(data, selected.path) if is_anomaly
                          else predict_kpi(data, selected.path))
            st.session_state[key] = {
                "signature": signature,
                "frame": result,
                "history": data if is_anomaly else None,
                "family": selected.family,
            }
            st.success("本次模型推理完成。")
        except Exception as exc:
            st.session_state.pop(key, None)
            st.error(f"推理失败：{exc}")
    saved = st.session_state.get(key)
    if saved is not None and saved.get("signature") == signature:
        st.divider()
        if is_anomaly:
            render_anomaly_inference_result(saved)
            st.caption("前往 KPI 关联分析页面可以直接分析本次共识异常候选。")
        else:
            render_forecast_inference_result(saved)
    elif saved is not None:
        st.info("模型或输入已切换，请点击运行以更新结果。")
    if is_anomaly:
        st.caption("本项目异常检测模型要求与训练时相同的全部波束 ID 和对齐的连续小时，第一小时作为差分上下文。")
