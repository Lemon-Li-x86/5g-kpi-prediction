"""Run fixed 5G KPI tasks without overwriting previous DL experiments.

Run using `python -m src.services.web_jobs`.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.services.model_registry import ROOT, archive_models
from src.services.process_env import python_job_env

TASKS = {
    "download": [["src.data.download_kpi_data"]],
    "preprocess": [["src.data.audit_kpi_data"], ["src.data.preprocess_kpi_data"]],
    "features": [["src.data.build_ml_features"]],
    "train_ml": [["src.models.train_traditional_models"]],
    "train_anomaly": [["src.models.detect_kpi_anomalies"]],
    "diagnostics": [["src.analysis.analyze_anomaly_statistics"]],
}
DL_MODELS = {"lstm", "gru", "transformer"}


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run(task: str, model: str = "") -> int:
    state_path = ROOT / "resources" / "jobs" / "current.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)

    run_dir: Path | None = None
    run_id: str | None = None

    if task == "train_dl":
        if model not in DL_MODELS:
            raise ValueError("Unsupported deep learning model")
        run_id = f"{model}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:6]}"
        run_dir = ROOT / "resources" / "experiments" / run_id
        commands = [[
            "src.models.train_deep_learning_models", "--models", model,
            "--out", str(run_dir),
        ]]
    elif task in TASKS and not model:
        commands = TASKS[task]
    else:
        raise ValueError("Unsupported job specification")

    state = {
        "task": task,
        "model": model,
        "status": "running",
        "started": _stamp(),
    }
    if run_dir is not None:
        state["output_dir"] = str(run_dir)
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    try:
        # Legacy ML/anomaly trainers still write to fixed names, so protect them.
        if task in ("train_ml", "train_anomaly"):
            archive_models(ROOT, reason="before_" + task)

        for args in commands:
            cmd = [sys.executable, "-u", "-m", *args]
            print("Running:", " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=ROOT, check=True, env=python_job_env())

        if task == "train_dl":
            assert run_dir is not None and run_id is not None
            checkpoint = run_dir / "models" / f"w3_{model}.pt"
            metadata = checkpoint.with_suffix(".json")
            if not checkpoint.is_file() or not metadata.is_file():
                raise FileNotFoundError(
                    f"Training finished without matching checkpoint and metadata: {checkpoint}"
                )

            # Publish a copy for the UI; keep the originals and reports in run_dir.
            version_dir = ROOT / "resources" / "models" / run_id
            version_dir.mkdir(parents=True, exist_ok=False)
            saved_checkpoint = version_dir / checkpoint.name
            shutil.copy2(checkpoint, saved_checkpoint)
            shutil.copy2(metadata, saved_checkpoint.with_suffix(".json"))
            state["models_archived"] = 1
            state["model_path"] = str(saved_checkpoint)
            state["metrics_path"] = str(
                run_dir / "reports" / "w3_deep_learning_metrics.csv"
            )
            print("Published model:", saved_checkpoint, flush=True)
            print("New experiment results:", state["metrics_path"], flush=True)
        elif task in ("train_ml", "train_anomaly"):
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
        state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=[*TASKS, "train_dl"])
    parser.add_argument("model", nargs="?", default="")
    args = parser.parse_args()
    raise SystemExit(run(args.task, args.model))
