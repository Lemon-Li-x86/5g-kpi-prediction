"""Validated user-adjustable parameters for Web-launched experiments.

The CLI allowlist lives here, not in the Streamlit page. Parameters are
passed as subprocess argv (never through shell=True). Chronological split
boundaries stay fixed during ordinary Web tuning to avoid accidental leakage.
"""
from __future__ import annotations

from math import isfinite

DL_DEFAULTS = {
    "epochs": 12, "batch_size": 256, "lr": 0.001, "hidden": 32,
    "seq_len": 24, "seasonal_lag": 168, "patience": 3,
    "max_train": 60000, "max_val": 12000, "max_test": 0,
    "seed": 42, "device": "auto",
}
ANOMALY_DEFAULTS = {
    "if_train_sample": 80000, "if_contamination": 0.02,
    "svm_train_sample": 2500, "svm_test_sample": 12000,
    "max_events": 2000, "seed": 42,
}
ML_DEFAULTS = {"quick": False, "max_train": 200000, "max_test": 0}

SPECS = {
    "train_dl": {
        "epochs": (int, 1, 500),
        "batch_size": (int, 1, 4096),
        "lr": (float, 1e-6, 0.1),
        "hidden": (int, 4, 1024),
        "seq_len": (int, 1, 168),
        "seasonal_lag": (int, 0, 336),
        "patience": (int, 1, 100),
        "max_train": (int, 1, 1000000),
        "max_val": (int, 1, 1000000),
        "max_test": (int, 0, 1000000),
        "seed": (int, 0, 2147483647),
        "device": (str, ("auto", "cpu", "cuda")),
    },
    "train_anomaly": {
        "if_train_sample": (int, 100, 1000000),
        "if_contamination": (float, 0.00001, 0.5),
        "svm_train_sample": (int, 100, 100000),
        "svm_test_sample": (int, 100, 1000000),
        "max_events": (int, 1, 1000000),
        "seed": (int, 0, 2147483647),
    },
    "train_ml": {
        "quick": (bool,),
        "max_train": (int, 1, 1000000),
        "max_test": (int, 0, 1000000),
    },
}
DEFAULTS = {
    "train_dl": DL_DEFAULTS,
    "train_anomaly": ANOMALY_DEFAULTS,
    "train_ml": ML_DEFAULTS,
}
FLAGS = {
    "train_dl": {
        "epochs": "--epochs", "batch_size": "--batch-size", "lr": "--lr",
        "hidden": "--hidden", "seq_len": "--seq-len",
        "seasonal_lag": "--seasonal-lag", "patience": "--patience",
        "max_train": "--max-train", "max_val": "--max-val",
        "max_test": "--max-test", "seed": "--seed", "device": "--device",
    },
    "train_anomaly": {
        "if_train_sample": "--if-train-sample",
        "if_contamination": "--if-contamination",
        "svm_train_sample": "--svm-train-sample",
        "svm_test_sample": "--svm-test-sample",
        "max_events": "--max-events", "seed": "--seed",
    },
    "train_ml": {
        "quick": "--quick", "max_train": "--max-train", "max_test": "--max-test",
    },
}


def normalize_params(task: str, supplied: dict | None = None, model: str = "") -> dict:
    """Validate everything at the worker boundary, even if UI is bypassed."""
    supplied = {} if supplied is None else supplied
    if not isinstance(supplied, dict):
        raise ValueError("Training parameters must be a JSON object")
    if task not in DEFAULTS:
        if supplied:
            raise ValueError(f"Task {task} does not accept training parameters")
        return {}
    unknown = set(supplied) - set(SPECS[task])
    if unknown:
        raise ValueError(f"Unsupported parameters: {sorted(unknown)}")
    values = dict(DEFAULTS[task])
    for name, value in supplied.items():
        rule = SPECS[task][name]
        kind = rule[0]
        # bool is an int subclass in Python; require the exact type.
        if kind is bool:
            if type(value) is not bool:
                raise ValueError(f"{name} must be boolean")
        elif kind is int:
            if type(value) is not int or not rule[1] <= value <= rule[2]:
                raise ValueError(f"{name} must be an integer in [{rule[1]}, {rule[2]}]")
        elif kind is float:
            if type(value) not in (float, int) or not isfinite(value) or not rule[1] <= value <= rule[2]:
                raise ValueError(f"{name} must be a number in [{rule[1]}, {rule[2]}]")
            value = float(value)
        elif kind is str:
            if type(value) is not str or value not in rule[1]:
                raise ValueError(f"{name} must be one of {rule[1]}")
        values[name] = value

    if task == "train_dl":
        if model not in {"lstm", "gru", "transformer"}:
            raise ValueError("Unknown deep learning model")
        if model == "transformer" and values["hidden"] % 4 != 0:
            raise ValueError("Transformer hidden size must be divisible by 4 attention heads")
        if values["patience"] > values["epochs"]:
            raise ValueError("Early-stopping patience cannot exceed epochs")
        # The raw dataset is hourly; inference needs its selected history window.
        # Keep split hours fixed at validation=588 and test=672 in this UI.
    if task == "train_ml" and not values["quick"] and (
        values["max_train"] != ML_DEFAULTS["max_train"] or values["max_test"] != 0
    ):
        raise ValueError("Traditional ML sampling limits require quick mode")
    return values


def params_to_cli(task: str, supplied: dict | None = None, model: str = "") -> list[str]:
    values = normalize_params(task, supplied, model)
    args = []
    for name, flag in FLAGS.get(task, {}).items():
        if SPECS[task][name][0] is bool:
            if values[name]:
                args.append(flag)
        else:
            args.extend((flag, str(values[name])))
    return args
