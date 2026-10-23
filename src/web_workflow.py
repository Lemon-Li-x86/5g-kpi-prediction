"""Data preparation, training and versioned inference pages with live job logs."""
from __future__ import annotations

import html
import json
import os
import re
import secrets
import socket
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from src.model_registry import ROOT, KPIS, archive_models, list_models

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


def _launch(task: str, model: str = "") -> None:
    if _busy():
        st.warning("已有任务正在执行，请勿重复启动。")
        return

    JOBS.mkdir(parents=True, exist_ok=True)
    # Obtain a localhost TCP port for this run; the supervisor binds it shortly.
    with socket.socket() as temp:
        temp.bind(("127.0.0.1", 0))
        port = temp.getsockname()[1]
    token = secrets.token_urlsafe(24)
    # The session file is read by the UI. It is not committed to Git.
    SESSION.write_text(
        json.dumps({"port": port, "token": token, "task": task, "model": model}),
        encoding="utf-8",
    )
    STATE.write_text(
        json.dumps({"task": task, "model": model, "status": "running"}, ensure_ascii=False),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        with (JOBS / "stream_errors.log").open("w", encoding="utf-8") as diagnostics:
            subprocess.Popen(
                [sys.executable, "-u", "-m", "src.web_stream_worker", task,
                 *([model] if model else []), "--port", str(port), "--token", token],
                cwd=ROOT,
                stdout=diagnostics,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
                close_fds=True,
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

    session = _read_session()
    if session.get("port") and session.get("token") and (session.get("task"), session.get("model")) == (info.get("task"), info.get("model") or ""):
        # No @st.fragment; the browser receives push events directly.
        backup = ""
        if status != "running":
            backup = _tail_log()
        st.components.v1.html(
            _stream_terminal_html(int(session["port"]), str(session["token"]), backup, status),
            height=410,
            scrolling=False,
        )
    elif status == "running":
        _legacy_status()
    else:
        st.components.v1.html(_terminal_html(_tail_log()), height=370, scrolling=False)

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


def render_training_page() -> None:
    st.title("模型训练")
    st.caption("训练在独立进程中执行；训练前保存旧版本，成功后归档新版本。")
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


def render_model_page() -> None:
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
        st.info("异常模型的推理接口需要完整波束顺序和前一小时上下文；当前 UI 暂不直接开放此入口。")
        return
    st.warning("仅加载本机生成的可信权重；推理要求历史输入具有与训练一致的四项 KPI 和波束 ID。")
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
            st.download_button(
                "导出预测结果 CSV", result.to_csv(index=False).encode("utf-8-sig"),
                "kpi_inference.csv", "text/csv",
            )
    except Exception as exc:
        st.error(f"无法执行推理：{exc}")
