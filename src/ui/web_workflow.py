"""Data preparation, training and versioned inference pages with live job logs."""
from __future__ import annotations

import html
import hashlib
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
from src.services.process_env import python_job_env
from src.core.training_config import normalize_params, DL_DEFAULTS, ANOMALY_DEFAULTS

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


def render_model_page() -> None:
    """Select a trusted local checkpoint, run inference, and retain CSV exports."""
    st.title("模型版本管理与推理")
    st.caption("只扫描 resources/models/ 与 models/ 中的本地可信模型；不接受外部上传的 pickle。")
    if st.button("归档现有未版本化模型", disabled=_busy()):
        added = archive_models(ROOT, reason="manual_web_archive")
        st.success(f"已归档 {len(added)} 个模型文件。")
    models = list_models(ROOT)
    if not models:
        st.info("尚未发现本地权重。GitHub 中的历史指标 CSV 并不是模型权重，请先在本机运行训练。")
        return

    options = {f"{idx}: {item.label}": item for idx, item in enumerate(models)}
    selected_label = st.selectbox("选择本地权重版本", list(options))
    selected = options[selected_label]
    st.write({"类型": selected.family, "模型": selected.model,
              "目标": selected.target, "版本": selected.version, "路径": selected.path})
    if selected.family == "anomaly":
        st.info("异常检测必须使用和训练时相同的全部波束 ID、连续且对齐的小时轴；首小时只作差分上下文。检测结果是候选异常，不是已确认故障。")
    else:
        st.warning("仅加载本机可信权重；预测要求各波束的连续 KPI 历史序列。")

    uploaded = st.file_uploader("上传 KPI 历史 CSV（不保存到磁盘）", type=["csv"])
    if uploaded is None:
        return
    if uploaded.size > 50 * 1024 * 1024:
        st.error("文件超过 50 MB 限制。")
        return
    try:
        # Bind the remembered result to BOTH the model and precise CSV bytes.
        payload = uploaded.getvalue()
        signature = (selected.path, hashlib.sha256(payload).hexdigest())
        data = pd.read_csv(uploaded)
        required = {"Time", "base_station", "cell", "beam", *KPIS}
        missing = required - set(data.columns)
        if missing:
            st.error(f"缺少必要字段：{sorted(missing)}")
            return
        st.write(f"输入样本：{len(data):,} 行")

        if selected.family == "anomaly":
            label = "运行异常检测"
        else:
            label = "运行下一小时预测"
        if st.button(label):
            from src.models.inference import detect_anomalies, predict_kpi
            with st.spinner("正在执行推理"):
                result = (detect_anomalies(data, selected.path)
                          if selected.family == "anomaly"
                          else predict_kpi(data, selected.path))
            st.session_state["kpi_inference_result"] = {
                "signature": signature,
                "family": selected.family,
                "frame": result,
            }

        saved = st.session_state.get("kpi_inference_result")
        if saved and saved["signature"] == signature and saved["family"] == selected.family:
            result = saved["frame"]
            if selected.family == "anomaly":
                flags = ["sigma", "iqr", "isolation_forest", "one_class_svm", "consensus"]
                summary = pd.DataFrame({
                    "检测方法": flags,
                    "候选数量": [int(result[k].sum()) for k in flags],
                    "标记比例": [float(result[k].mean()) for k in flags],
                })
                st.subheader("本次输入数据的异常候选统计")
                st.dataframe(summary, hide_index=True, use_container_width=True)
                filename = "kpi_anomaly_candidates.csv"
                download_label = "导出异常标记 CSV"
            else:
                filename = "kpi_inference.csv"
                download_label = "导出预测结果 CSV"
            st.dataframe(result.head(200), hide_index=True, use_container_width=True)
            st.download_button(
                download_label, data=result.to_csv(index=False).encode("utf-8-sig"),
                file_name=filename, mime="text/csv",
            )
            st.caption("结果保存在当前浏览器会话中；下载或页面刷新不会清空结果。切换模型或上传文件后需重新运行推理。")
    except Exception as exc:
        st.error(f"无法执行推理：{exc}")
