#!/usr/bin/env python3
"""
Condensed W3 (original Week 5):
Beam-level multi-KPI forecasting with LSTM, GRU and Transformer.

Input:
    data/processed/kpi_timeseries.csv

Targets:
    DLPRB, DLThpVol, DLThpTime, MR_number

Task:
    Use historical KPI values to predict the next hour.

Evaluation:
    MAE, RMSE and R2 in original KPI units.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.core.config import KPIS, KEYS, DATASET_TEST_START_HOUR

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise SystemExit(
        "PyTorch is missing. Install it with: pip install torch"
    ) from exc


# ============================================================
# 1. Project configuration
# ============================================================

# Shared dataset schema; model-specific dimensions and settings remain local.
Z_COLS = [f"{k}_z" for k in KPIS]


# ============================================================
# 2. Load processed KPI data
# ============================================================

def load_tensor(csv_path: Path):

    required = [
        "Time",
        *KEYS,
        *KPIS,
        *Z_COLS,
    ]

    df = pd.read_csv(
        csv_path,
        usecols=required,
    )

    if df.empty or df[required].isna().any().any():
        raise ValueError(
            "Empty data or missing values in required columns"
        )

    df = df.sort_values(
        KEYS + ["Time"],
        kind="stable",
    )

    counts = (
        df.groupby(
            KEYS,
            sort=False,
            dropna=False,
        )
        .size()
        .to_numpy()
    )

    if len(counts) == 0 or not np.all(counts == counts[0]):
        raise ValueError(
            "All beams must have the same number of time points"
        )

    n_beams = len(counts)
    n_times = int(counts[0])

    time_grid = (
        df["Time"]
        .to_numpy()
        .reshape(n_beams, n_times)
    )

    timeline = time_grid[0].copy()

    if not np.array_equal(
        time_grid,
        np.broadcast_to(timeline, time_grid.shape),
    ):
        raise ValueError(
            "Misaligned time axes between beams"
        )

    if len(np.unique(timeline)) != len(timeline):
        raise ValueError("Duplicated times within a beam")

    data = (
        df[Z_COLS]
        .to_numpy(dtype=np.float32)
        .reshape(n_beams, n_times, len(KPIS))
    )

    raw = (
        df[KPIS]
        .to_numpy(dtype=np.float32)
        .reshape(n_beams, n_times, len(KPIS))
    )

    if not np.isfinite(data).all():
        raise ValueError(
            "NaN or infinite values in standardized KPI data"
        )

    if not np.isfinite(raw).all():
        raise ValueError(
            "NaN or infinite values in original KPI data"
        )

    return data, raw, timeline


# ============================================================
# 3. Sliding-window dataset
# ============================================================

class WindowDataset(Dataset):

    def __init__(
        self,
        data,
        target_indices,
        seq_len,
        seasonal_lag,
        sample_ids,
    ):

        self.data = data
        self.targets = target_indices.astype(np.int32)
        self.seq_len = seq_len
        self.seasonal_lag = seasonal_lag
        self.ids = sample_ids.astype(np.int64)
        self.n_targets = len(self.targets)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):

        item = int(self.ids[index])

        beam, target_pos = divmod(
            item,
            self.n_targets,
        )

        ti = int(self.targets[target_pos])

        # All input observations precede the target.
        seq = np.ascontiguousarray(
            self.data[
                beam,
                ti - self.seq_len:ti,
            ]
        )

        # Target is t+1.
        # Seasonal input corresponds to t-168.
        seasonal = np.ascontiguousarray(
            self.data[
                beam,
                ti - self.seasonal_lag - 1,
            ]
        )

        target = np.ascontiguousarray(
            self.data[beam, ti]
        )

        return (
            torch.from_numpy(seq),
            torch.from_numpy(seasonal),
            torch.from_numpy(target),
            beam,
            ti,
        )


def sample_ids(
    n_beams,
    n_times,
    limit,
    rng,
):

    total = n_beams * n_times

    if limit <= 0 or limit >= total:
        return np.arange(total, dtype=np.int64)

    return np.sort(
        rng.choice(
            total,
            size=limit,
            replace=False,
        )
    )


# ============================================================
# 4. LSTM / GRU / Transformer models
# ============================================================

from src.models.architectures import ForecastNet


# ============================================================
# 5. Validation loss
# ============================================================

def evaluate_loss(model, loader, device):

    model.eval()

    sse = 0.0
    total = 0

    with torch.no_grad():

        for seq, seasonal, target, _, _ in loader:

            seq = seq.to(device)
            seasonal = seasonal.to(device)
            target = target.to(device)

            pred = model(seq, seasonal)

            sse += float(
                torch.sum(
                    (pred - target) ** 2
                ).item()
            )

            total += target.numel()

    return sse / max(total, 1)


# ============================================================
# 6. Prediction
# ============================================================

def forecast(model, loader, device):

    model.eval()

    ys = []
    preds = []
    tids = []

    with torch.no_grad():

        for seq, seasonal, target, _, ti in loader:

            p = model(
                seq.to(device),
                seasonal.to(device),
            ).cpu().numpy()

            preds.append(p)
            ys.append(target.numpy())
            tids.append(ti.numpy())

    return (
        np.concatenate(ys),
        np.concatenate(preds),
        np.concatenate(tids),
    )


# ============================================================
# 7. Visualization
# ============================================================

def save_plots(
    history,
    y,
    pred,
    t_indices,
    timeline,
    model_name,
    figures,
):

    # Training and validation loss
    fig, ax = plt.subplots(figsize=(8, 4))

    epochs = np.arange(
        1,
        len(history["train"]) + 1,
    )

    ax.plot(
        epochs,
        history["train"],
        label="Train MSE",
    )

    ax.plot(
        epochs,
        history["val"],
        label="Validation MSE",
    )

    ax.set(
        xlabel="Epoch",
        ylabel="Mean squared error (scaled)",
        title=f"{model_name.upper()} training history",
    )

    ax.legend()
    fig.tight_layout()

    fig.savefig(
        figures / f"w3_{model_name}_loss.png",
        dpi=160,
    )

    plt.close(fig)

    # Aggregate predictions by target hour.
    # Metrics are still evaluated on individual beam-hours.
    aggregate = pd.DataFrame({
        "Time": timeline[t_indices],
    })

    for j, name in enumerate(KPIS):

        aggregate[f"{name}_actual"] = y[:, j]
        aggregate[f"{name}_pred"] = pred[:, j]

    by_hour = (
        aggregate.groupby(
            "Time",
            sort=True,
        )
        .mean(numeric_only=True)
    )

    for name in KPIS:

        fig, ax = plt.subplots(
            figsize=(9, 3.5)
        )

        ax.plot(
            by_hour.index,
            by_hour[f"{name}_actual"],
            label="Actual (beam mean)",
        )

        ax.plot(
            by_hour.index,
            by_hour[f"{name}_pred"],
            label="Predicted (beam mean)",
        )

        ax.set(
            xlabel="Target hour",
            ylabel="Original KPI units",
            title=(
                f"{model_name.upper()} {name} "
                "| hourly beam mean"
            ),
        )

        ax.legend()
        fig.tight_layout()

        fig.savefig(
            figures
            / f"w3_{model_name}_{name}_prediction.png",
            dpi=160,
        )

        plt.close(fig)

    by_hour.reset_index().to_csv(
        figures.parent
        / "reports"
        / f"w3_{model_name}_hourly_predictions.csv",
        index=False,
    )


# ============================================================
# 8. Main training pipeline
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "data/processed/kpi_timeseries.csv"
        ),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("."),
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=[
            "lstm",
            "gru",
            "transformer",
        ],
        default=[
            "lstm",
            "gru",
            "transformer",
        ],
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=24,
    )

    parser.add_argument(
        "--seasonal-lag",
        type=int,
        default=168,
    )

    parser.add_argument(
        "--min-target-hour",
        type=int,
        default=169,
    )

    parser.add_argument(
        "--validation-start",
        type=int,
        default=588,
    )

    parser.add_argument(
        "--test-start",
        type=int,
        default=DATASET_TEST_START_HOUR,
    )

    parser.add_argument(
        "--max-train",
        type=int,
        default=60000,
    )

    parser.add_argument(
        "--max-val",
        type=int,
        default=12000,
    )

    parser.add_argument(
        "--max-test",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--hidden",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
    )

    args = parser.parse_args()

    if (
        args.seq_len < 1
        or args.seasonal_lag < 0
        or args.epochs < 1
    ):
        parser.error(
            "Invalid sequence length, seasonal lag or epochs"
        )

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if (
        args.device == "cuda"
        and not torch.cuda.is_available()
    ):
        parser.error(
            "CUDA requested but unavailable"
        )

    if args.device == "auto":

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:
        device = torch.device(args.device)

    if device.type == "cpu":
        torch.set_num_threads(
            min(8, torch.get_num_threads())
        )

    # Output directories
    reports = args.out / "reports"
    figures = args.out / "figures"
    weights = args.out / "models"

    for directory in [
        reports,
        figures,
        weights,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # Load data
    data, raw, timeline = load_tensor(
        args.csv
    )

    # Recover original-scale values from StandardScaler.
    # The previous stage fitted its scaler on hours 0..671.
    scale_mask = timeline < args.validation_start

    scale_raw = raw[
        :,
        scale_mask,
        :,
    ].astype(np.float64)

    raw_mean = scale_raw.mean(
        axis=(0, 1)
    )

    raw_std = scale_raw.std(
        axis=(0, 1)
    )

    if np.any(raw_std <= 0):
        raise ValueError(
            "Constant KPI detected in training period"
        )

    # Verify consistency with previous preprocessing.
    rng_check = np.random.default_rng(
        args.seed
    )

    idx_b = rng_check.integers(
        0,
        len(raw),
        size=min(1000, raw.size // 4),
    )

    idx_t = rng_check.choice(
        np.flatnonzero(scale_mask),
        size=len(idx_b),
    )

    reconstructed = (
        raw[idx_b, idx_t, :]
        - raw_mean
    ) / raw_std

    max_err = float(
        np.max(
            np.abs(
                reconstructed
                - data[idx_b, idx_t, :]
            )
        )
    )

    # Refit from raw training prefix; never use validation/test scaler statistics.
    data = ((raw - raw_mean) / raw_std).astype(np.float32)

    del scale_raw

    # Chronological train / validation / test split
    min_idx = max(
        args.seq_len,
        args.seasonal_lag + 1,
    )

    positions = np.arange(
        len(timeline)
    )

    split_train = np.flatnonzero(
        (timeline >= args.min_target_hour)
        & (timeline < args.validation_start)
        & (positions >= min_idx)
    )

    split_val = np.flatnonzero(
        (timeline >= args.validation_start)
        & (timeline < args.test_start)
        & (positions >= min_idx)
    )

    split_test = np.flatnonzero(
        (timeline >= args.test_start)
        & (positions >= min_idx)
    )

    if min(
        len(split_train),
        len(split_val),
        len(split_test),
    ) == 0:
        raise ValueError(
            "Empty train, validation or test split"
        )

    rng = np.random.default_rng(
        args.seed
    )

    n_beams = data.shape[0]

    datasets = {}

    for name, indices, limit in [
        ("train", split_train, args.max_train),
        ("val", split_val, args.max_val),
        ("test", split_test, args.max_test),
    ]:

        ids = sample_ids(
            n_beams,
            len(indices),
            limit,
            rng,
        )

        datasets[name] = WindowDataset(
            data,
            indices,
            args.seq_len,
            args.seasonal_lag,
            ids,
        )

        print(
            f"{name}: {len(ids):,} samples; "
            f"hours {timeline[indices[0]]}"
            f"..{timeline[indices[-1]]}"
        )

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }

    loaders = {
        name: DataLoader(
            dataset,
            shuffle=(name == "train"),
            **loader_options,
        )
        for name, dataset in datasets.items()
    }

    metrics_rows = []
    run_infos = {}

    # Train selected models
    for model_name in args.models:

        print(
            f"\nTraining {model_name.upper()} "
            f"on {device}..."
        )

        model = ForecastNet(
            model_name,
            args.seq_len,
            args.hidden,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
        )

        criterion = nn.MSELoss()

        history = {
            "train": [],
            "val": [],
        }

        best = float("inf")
        patience_left = args.patience
        best_epoch = 0

        checkpoint = (
            weights / f"w3_{model_name}.pt"
        )

        for epoch in range(args.epochs):

            model.train()

            total_loss = 0.0
            total_elements = 0

            for seq, seasonal, y, _, _ in loaders["train"]:

                seq = seq.to(device)
                seasonal = seasonal.to(device)
                y = y.to(device)

                optimizer.zero_grad(
                    set_to_none=True
                )

                prediction = model(
                    seq,
                    seasonal,
                )

                loss = criterion(
                    prediction,
                    y,
                )

                loss.backward()
                optimizer.step()

                total_loss += (
                    float(loss.item())
                    * y.numel()
                )

                total_elements += y.numel()

            train_loss = (
                total_loss
                / max(1, total_elements)
            )

            val_loss = evaluate_loss(
                model,
                loaders["val"],
                device,
            )

            history["train"].append(
                train_loss
            )

            history["val"].append(
                val_loss
            )

            print(
                f"Epoch {epoch + 1}/{args.epochs}: "
                f"train MSE={train_loss:.5f}, "
                f"val MSE={val_loss:.5f}"
            )

            if val_loss < best - 1e-6:

                best = val_loss
                best_epoch = epoch + 1
                patience_left = args.patience

                torch.save(
                    model.state_dict(),
                    checkpoint,
                )

            else:

                patience_left -= 1

                if patience_left <= 0:
                    print("Early stopping")
                    break

        # Load best checkpoint
        model.load_state_dict(
            torch.load(
                checkpoint,
                map_location=device,
                weights_only=True,
            )
        )

        checkpoint.with_suffix('.json').write_text(json.dumps({
            'version': 1, 'family': 'dl', 'mode': model_name,
            'seq_len': args.seq_len, 'seasonal_lag': args.seasonal_lag,
            'hidden': args.hidden, 'mean': raw_mean.tolist(),
            'scale': raw_std.tolist(), 'scaler_fit_until': args.validation_start,
        }, indent=2), encoding='utf-8')

        # Predict on held-out test data
        truth_z, predicted_z, times = forecast(
            model,
            loaders["test"],
            device,
        )

        # Inverse transform to original KPI scale
        truth = (
            truth_z.astype(np.float64)
            * raw_std
            + raw_mean
        )

        predicted = (
            predicted_z.astype(np.float64)
            * raw_std
            + raw_mean
        )

        for j, name in enumerate(KPIS):

            metrics_rows.append({
                "model": model_name.upper(),
                "KPI": name,
                "MAE": mean_absolute_error(
                    truth[:, j],
                    predicted[:, j],
                ),
                "RMSE": math.sqrt(
                    mean_squared_error(
                        truth[:, j],
                        predicted[:, j],
                    )
                ),
                "R2": r2_score(
                    truth[:, j],
                    predicted[:, j],
                ),
                "test_samples": len(truth),
                "unit": "raw",
            })

        save_plots(
            history,
            truth,
            predicted,
            times,
            timeline,
            model_name,
            figures,
        )

        run_infos[model_name] = {
            "best_epoch": best_epoch,
            "best_val_mse": best,
            "num_parameters": sum(
                p.numel()
                for p in model.parameters()
            ),
            "epochs_run": len(history["train"]),
            "loss_history": history,
        }

    # Save metrics
    out_metrics = (
        reports
        / "w3_deep_learning_metrics.csv"
    )

    config_path = (
        reports
        / "w3_deep_learning_run_config.json"
    )

    config = {
        "dataset": str(args.csv),
        "seed": args.seed,
        "device": str(device),
        "seq_len": args.seq_len,
        "seasonal_lag": args.seasonal_lag,
        "target": (
            "t+1; four per-beam original-scale KPI "
            "targets (z normalized for training)"
        ),
        "metric_units": (
            "raw KPI units "
            "(same as earlier traditional ML)"
        ),
        "train_scale_mean": raw_mean.tolist(),
        "train_scale_std": raw_std.tolist(),
        "train_scaler_consistency_max_error": max_err,
        "training": [
            int(timeline[split_train[0]]),
            int(timeline[split_train[-1]]),
        ],
        "validation": [
            int(timeline[split_val[0]]),
            int(timeline[split_val[-1]]),
        ],
        "test": [
            int(timeline[split_test[0]]),
            int(timeline[split_test[-1]]),
        ],
        "train_samples": len(datasets["train"]),
        "validation_samples": len(datasets["val"]),
        "test_samples": len(datasets["test"]),
        "full_test": args.max_test == 0,
        "comparison_note": (
            "Same prediction target/test hours/raw target "
            "scale as earlier ML, but DL uses sampled "
            "training windows and separate validation."
        ),
        "validation_scaler_note": (
            "Previous scaler was fitted on hours 0..671, "
            "including the internal validation partition."
        ),
        "runs": run_infos,
    }

    # Preserve earlier model results when protocol matches.
    if (
        out_metrics.exists()
        and config_path.exists()
    ):

        previous = json.loads(
            config_path.read_text(
                encoding="utf-8"
            )
        )

        protocol_keys = [
            "dataset",
            "seed",
            "seq_len",
            "seasonal_lag",
            "target",
            "metric_units",
            "training",
            "validation",
            "test",
            "train_samples",
            "validation_samples",
            "test_samples",
            "full_test",
            "train_scale_mean",
            "train_scale_std",
        ]

        for key in protocol_keys:

            if previous.get(key) != config.get(key):

                raise ValueError(
                    f"Existing results use a different {key}. "
                    "Use a separate --out directory."
                )

        old_rows = pd.read_csv(
            out_metrics
        )

        new_rows = pd.DataFrame(
            metrics_rows
        )

        all_rows = pd.concat(
            [old_rows, new_rows],
            ignore_index=True,
        )

        all_rows = all_rows.drop_duplicates(
            ["model", "KPI"],
            keep="last",
        )

        config["runs"] = {
            **previous.get("runs", {}),
            **run_infos,
        }

    else:

        all_rows = pd.DataFrame(
            metrics_rows
        )

    all_rows.to_csv(
        out_metrics,
        index=False,
    )

    config_path.write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nMetrics saved:",
        out_metrics,
    )

    print(
        all_rows.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
