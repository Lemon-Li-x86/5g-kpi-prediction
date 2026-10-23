"""Safe subprocess runner for Streamlit's offline experiment workflow.

Usage: python -m src.web_jobs TASK [MODEL]
Tasks operate only on the repository's existing data pipeline.
"""
from __future__ import annotations

import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from src.model_registry import ROOT, archive_models

TASKS = {
    "download": [["src.download_kpi_data"]],
    "preprocess": [["src.audit_kpi_data"], ["src.preprocess_kpi_data"]],
    "features": [["src.build_ml_features"]],
    "train_ml": [["src.train_traditional_models"]],
    "train_anomaly": [["src.detect_kpi_anomalies"]],
    "diagnostics": [["src.analyze_anomaly_statistics"]],
}
DL_MODELS = {"lstm", "gru", "transformer"}


def _stamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run(task: str, model: str = "") -> int:
    state_path = ROOT / "resources" / "jobs" / "current.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if task == "train_dl":
        if model not in DL_MODELS:
            raise ValueError("Unsupported deep learning model")
        commands = [["src.train_deep_learning_models", "--models", model]]
    elif task in TASKS and not model:
        commands = TASKS[task]
    else:
        raise ValueError("Unsupported job specification")

    state = {"task": task, "model": model, "status": "running", "started": _stamp()}
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    try:
        if task.startswith("train_"):
            # Save pre-existing legacy weights before a training job can overwrite them.
            archive_models(ROOT, reason="before_" + task)
        for args in commands:
            cmd = [sys.executable, "-m", *args]
            print("Running:", " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=ROOT, check=True)
        if task.startswith("train_"):
            saved = archive_models(ROOT, reason="after_" + task)
            state["models_archived"] = len(saved)
        state["status"] = "completed"
        return 0
    except Exception as exc:
        traceback.print_exc()
        state["status"] = "failed"
        state["error"] = str(exc)
        return 1
    finally:
        state["finished"] = _stamp()
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=[*TASKS, "train_dl"])
    parser.add_argument("model", nargs="?", default="")
    args = parser.parse_args()
    raise SystemExit(run(args.task, args.model))
