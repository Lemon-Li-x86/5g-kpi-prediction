
"""Time-series validation, tuning and significance testing."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import norm
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler

from w2_traditional_ml_kpi_prediction_training import (
    build_models,
    KPI_COLUMNS,
)


ROOT = Path(__file__).resolve().parents[1]

FEATURES_FILE = ROOT / "data" / "processed" / "ml_features.pkl"
FEATURE_META = ROOT / "reports" / "ml_feature_summary.json"
PREPROCESS_META = ROOT / "reports" / "preprocessing_summary.json"
REPORT_DIR = ROOT / "reports" / "validation"

SEED = 42
FOLD_STARTS = (384, 480, 576)
WINDOW_HOURS = 96
HOLDOUT_START = 672


# ============================================================
# 1. Load data and reconstruct original input values
# ============================================================

def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def feature_kpi(feature):
    for kpi in KPI_COLUMNS:
        if (
            feature == f"{kpi}_z_t"
            or feature.startswith(f"{kpi}_z_lag")
        ):
            return kpi

    raise ValueError(f"Unknown feature: {feature}")


def load_design(
    data_file=None,
    feature_meta=None,
    preprocess_meta=None,
):
    """
    Restore unstandardized lag features from the W3
    StandardScaler representation.

    This allows a NEW scaler to be fitted independently
    inside every cross-validation fold.
    """

    data_file = FEATURES_FILE if data_file is None else data_file
    feature_meta = FEATURE_META if feature_meta is None else feature_meta
    preprocess_meta = (
        PREPROCESS_META if preprocess_meta is None
        else preprocess_meta
    )

    feature_info = load_json(feature_meta)
    preprocessing = load_json(preprocess_meta)

    names = feature_info["features"]["feature_columns"]

    if len(names) != 20:
        raise ValueError(
            f"Expected 20 features, found {len(names)}"
        )

    stats = preprocessing["standardisation"]

    if stats["method"] != "StandardScaler":
        raise ValueError("Expected W3 StandardScaler data")

    df = pd.read_pickle(data_file)

    required = (
        names
        + ["feature_time", "target_time"]
        + [f"target_{kpi}" for kpi in KPI_COLUMNS]
        + [f"{kpi}_raw_t" for kpi in KPI_COLUMNS]
    )

    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    feature_times = df["feature_time"].to_numpy()
    target_times = df["target_time"].to_numpy()

    if not np.all(target_times - feature_times == 1):
        raise ValueError("Expected 1-hour forecast horizon")

    x_raw = df[names].to_numpy(
        dtype=np.float32, copy=True
    )

    for j, name in enumerate(names):
        kpi = feature_kpi(name)

        x_raw[:, j] = (
            x_raw[:, j] * np.float32(stats["scale"][kpi])
            + np.float32(stats["mean"][kpi])
        )

    if not np.isfinite(x_raw).all():
        raise ValueError("Invalid feature values detected")

    return df, x_raw, names


# ============================================================
# 2. Expanding-window cross-validation
# ============================================================

def iter_folds(
    target_times,
    starts=FOLD_STARTS,
    window=WINDOW_HOURS,
):
    """
    Fold 1: train 169..383, validate 384..479
    Fold 2: train 169..479, validate 480..575
    Fold 3: train 169..575, validate 576..671

    Final holdout 672..839 is not used here.
    """

    for fold, start in enumerate(starts, start=1):

        if start + window > HOLDOUT_START:
            raise ValueError(
                "Validation overlaps final holdout"
            )

        train = np.flatnonzero(
            target_times < start
        )

        valid = np.flatnonzero(
            (target_times >= start)
            & (target_times < start + window)
        )

        if not len(train) or not len(valid):
            raise ValueError(f"Empty fold: {fold}")

        if (
            target_times[train].max()
            >= target_times[valid].min()
        ):
            raise AssertionError("Time leakage detected")

        yield fold, start, train, valid


def fold_features(x_raw, names, train, valid):
    """
    Fit the scaler ONLY on the current fold's training
    observations. Apply each KPI's scale to all its lags.
    """

    current_positions = [
        names.index(f"{kpi}_z_t")
        for kpi in KPI_COLUMNS
    ]

    scaler = StandardScaler().fit(
        x_raw[np.ix_(train, current_positions)]
    )

    means = np.array(
        [
            scaler.mean_[
                KPI_COLUMNS.index(feature_kpi(name))
            ]
            for name in names
        ],
        dtype=np.float32,
    )

    scales = np.array(
        [
            scaler.scale_[
                KPI_COLUMNS.index(feature_kpi(name))
            ]
            for name in names
        ],
        dtype=np.float32,
    )

    scales[scales == 0] = 1

    x_train = (
        (x_raw[train] - means) / scales
    ).astype(np.float32)

    x_valid = (
        (x_raw[valid] - means) / scales
    ).astype(np.float32)

    return x_train, x_valid


# ============================================================
# 3. Model evaluation
# ============================================================

def score(y, pred):
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(
            np.sqrt(mean_squared_error(y, pred))
        ),
        "R2": float(r2_score(y, pred)),
    }


def selected_models(names, quick):
    built = build_models(quick=quick)

    allowed = {
        "Persistence": None,
        **built,
    }

    for name in names:
        if name not in allowed:
            raise ValueError(
                f"Unknown model {name}. "
                f"Available: {list(allowed)}"
            )

    return allowed


def evaluate_window(
    df,
    x_train,
    x_valid,
    train,
    valid,
    models,
    fold,
    save_loss=False,
):
    results = []
    loss_records = []

    times = df["target_time"].to_numpy(
        dtype=np.int32
    )[valid]

    unique_times, inverse = np.unique(
        times, return_inverse=True
    )

    counts = np.bincount(inverse)

    for kpi in KPI_COLUMNS:

        quick = "quick" in fold.lower()

        instantiated_models = (
            build_models(quick=quick)
            if any(m != "Persistence" for m in models)
            else {}
        )

        y_train = df[
            f"target_{kpi}"
        ].to_numpy(dtype=np.float32)[train]

        y_valid = df[
            f"target_{kpi}"
        ].to_numpy(dtype=np.float32)[valid]

        for name in models:

            start = time.perf_counter()

            if name == "Persistence":

                pred = df[
                    f"{kpi}_raw_t"
                ].to_numpy(dtype=np.float32)[valid]

                fit_seconds = 0.0
                predict_seconds = 0.0

            else:

                model = instantiated_models[name]

                model.fit(x_train, y_train)

                fit_seconds = (
                    time.perf_counter() - start
                )

                start = time.perf_counter()

                pred = np.asarray(
                    model.predict(x_valid),
                    dtype=np.float32,
                )

                predict_seconds = (
                    time.perf_counter() - start
                )

                del model

            metrics = score(y_valid, pred)

            results.append({
                "fold": fold,
                "target": kpi,
                "model": name,
                **metrics,
                "train_rows": len(train),
                "valid_rows": len(valid),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "train_max_target_time": int(
                    df["target_time"].iloc[train].max()
                ),
                "valid_min_target_time": int(
                    unique_times.min()
                ),
                "valid_max_target_time": int(
                    unique_times.max()
                ),
            })

            print(
                f"  {fold} | {kpi} | {name} | "
                f"RMSE={metrics['RMSE']:.6f}",
                flush=True,
            )

            if save_loss:

                error = (
                    y_valid.astype(np.float64) - pred
                )

                mean_abs = (
                    np.bincount(
                        inverse,
                        weights=np.abs(error),
                    )
                    / counts
                )

                mean_sq = (
                    np.bincount(
                        inverse,
                        weights=np.square(error),
                    )
                    / counts
                )

                for t, a, s in zip(
                    unique_times, mean_abs, mean_sq
                ):
                    loss_records.append({
                        "fold": fold,
                        "target": kpi,
                        "model": name,
                        "target_time": int(t),
                        "mae_loss": float(a),
                        "mse_loss": float(s),
                    })

    return results, loss_records


def evaluate(args, holdout=False):

    df, x_raw, names = load_design()

    chosen = list(dict.fromkeys(args.models))
    selected_models(chosen, args.quick)

    times = df["target_time"].to_numpy(
        dtype=np.int32
    )

    if holdout:
        folds = [
            (
                "holdout",
                HOLDOUT_START,
                np.flatnonzero(times < HOLDOUT_START),
                np.flatnonzero(times >= HOLDOUT_START),
            )
        ]
    else:
        folds = [
            (f"fold_{i}", start, train, valid)
            for i, start, train, valid in iter_folds(times)
        ]

    all_metrics = []
    all_loss = []

    for fold, start, train, valid in folds:

        print(
            f"\n{fold}: "
            f"train {times[train].min()}.."
            f"{times[train].max()}, "
            f"validation {times[valid].min()}.."
            f"{times[valid].max()}",
            flush=True,
        )

        if holdout:
            # Reproduce the original W4 input scaling.
            x_train = df[names].to_numpy(
                dtype=np.float32
            )[train]

            x_valid = df[names].to_numpy(
                dtype=np.float32
            )[valid]

        else:
            x_train, x_valid = fold_features(
                x_raw, names, train, valid
            )

        label = (
            f"{fold}_quick" if args.quick else fold
        )

        records, losses = evaluate_window(
            df,
            x_train,
            x_valid,
            train,
            valid,
            {model: None for model in chosen},
            label,
            save_loss=True,
        )

        all_metrics.extend(records)
        all_loss.extend(losses)

        del x_train, x_valid

    REPORT_DIR.mkdir(
        parents=True, exist_ok=True
    )

    prefix = (
        "holdout" if holdout
        else "walkforward"
    )

    metrics_df = pd.DataFrame(all_metrics)
    loss_df = pd.DataFrame(all_loss)

    metrics_df.to_csv(
        REPORT_DIR / f"{prefix}_fold_metrics.csv",
        index=False,
    )

    loss_df.to_csv(
        REPORT_DIR / f"{prefix}_hourly_losses.csv",
        index=False,
    )

    summary = metrics_df.groupby(
        ["target", "model"], as_index=False
    ).agg(
        mean_MAE=("MAE", "mean"),
        std_MAE=("MAE", "std"),
        mean_RMSE=("RMSE", "mean"),
        std_RMSE=("RMSE", "std"),
        mean_R2=("R2", "mean"),
        fold_count=("fold", "nunique"),
    )

    summary.to_csv(
        REPORT_DIR / f"{prefix}_summary.csv",
        index=False,
    )

    print("\nVALIDATION SUMMARY")
    print(summary.to_string(index=False))


# ============================================================
# 4. Statistical significance testing
# ============================================================

def hac_paired_test(
    reference,
    contender,
    bandwidth=24,
):
    """
    Paired loss comparison with Newey-West HAC variance.

    d(t) = contender_loss(t) - reference_loss(t)

    Positive d favors the reference model.
    Observations are hourly mean beam-level losses.
    """

    reference = np.asarray(
        reference, dtype=np.float64
    )
    contender = np.asarray(
        contender, dtype=np.float64
    )

    if reference.shape != contender.shape:
        raise ValueError("Paired series shape mismatch")

    d = contender - reference
    n = len(d)

    if n < 10 or not np.isfinite(d).all():
        raise ValueError(
            "At least 10 valid paired hours are required"
        )

    centered = d - d.mean()
    lag = min(bandwidth, n - 2)

    gamma0 = float(
        np.dot(centered, centered) / n
    )

    long_run_variance = gamma0

    for j in range(1, lag + 1):

        covariance = float(
            np.dot(
                centered[j:],
                centered[:-j],
            ) / n
        )

        weight = 1 - j / (lag + 1)

        long_run_variance += (
            2 * weight * covariance
        )

    if long_run_variance <= 1e-16:
        return {
            "time_points": n,
            "mean_loss_difference": float(d.mean()),
            "hac_se": 0.0,
            "z": None,
            "p_two_sided": None,
            "ci95_low": None,
            "ci95_high": None,
            "note": "Degenerate HAC variance",
        }

    se = float(
        np.sqrt(long_run_variance / n)
    )

    z = float(d.mean() / se)

    return {
        "time_points": n,
        "mean_loss_difference": float(d.mean()),
        "hac_se": se,
        "z": z,
        "p_two_sided": float(
            2 * norm.sf(abs(z))
        ),
        "ci95_low": float(d.mean() - 1.96 * se),
        "ci95_high": float(d.mean() + 1.96 * se),
        "note": "Newey-West HAC with Bartlett weights",
    }


def holm_adjust(pvalues):
    nonnull = pvalues.dropna().sort_values()

    m = len(nonnull)
    running = 0.0

    corrected = pd.Series(
        np.nan,
        index=pvalues.index,
        dtype=float,
    )

    for i, (idx, value) in enumerate(
        nonnull.items()
    ):

        running = max(
            running,
            min(1.0, (m - i) * float(value)),
        )

        corrected.loc[idx] = running

    return corrected


def significance(
    hourly_loss_path,
    reference,
    contenders,
    bandwidth=24,
):

    losses = pd.read_csv(hourly_loss_path)
    rows = []

    for target in KPI_COLUMNS:

        one_target = losses[
            losses["target"] == target
        ]

        for contender in contenders:

            if contender == reference:
                continue

            for loss_name in [
                "mae_loss",
                "mse_loss",
            ]:

                pivot = one_target.pivot(
                    index="target_time",
                    columns="model",
                    values=loss_name,
                )

                if (
                    reference not in pivot.columns
                    or contender not in pivot.columns
                ):
                    raise ValueError(
                        f"Missing predictions: "
                        f"{target}, {reference}, {contender}"
                    )

                paired = pivot[
                    [reference, contender]
                ].dropna().sort_index()

                if not np.all(
                    np.diff(paired.index.to_numpy()) == 1
                ):
                    raise ValueError(
                        "Non-consecutive target times"
                    )

                stat = hac_paired_test(
                    paired[reference].to_numpy(),
                    paired[contender].to_numpy(),
                    bandwidth,
                )

                rows.append({
                    "target": target,
                    "reference": reference,
                    "contender": contender,
                    "loss": loss_name,
                    "bandwidth": bandwidth,
                    **stat,
                })

    result = pd.DataFrame(rows)

    result["holm_p"] = result.groupby(
        ["reference", "contender", "loss"]
    )["p_two_sided"].transform(holm_adjust)

    REPORT_DIR.mkdir(
        parents=True, exist_ok=True
    )

    result.to_csv(
        REPORT_DIR / "significance_hac.csv",
        index=False,
    )

    print("\nSIGNIFICANCE TEST")
    print(result.to_string(index=False))

    print(
        "\nPositive mean_loss_difference "
        "favors the reference model."
    )

    return result


# ============================================================
# 5. Optuna hyperparameter tuning
# ============================================================

def tune(args):

    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "Install Optuna: pip install optuna"
        ) from exc

    from sklearn.base import clone
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor

    df, x_raw, names = load_design()

    if args.target not in KPI_COLUMNS:
        raise ValueError(
            f"Unknown KPI: {args.target}"
        )

    if args.model not in [
        "RandomForest",
        "XGBoost",
        "LightGBM",
    ]:
        raise ValueError(
            "Unsupported tuning model"
        )

    # Sample complete beam histories for faster search.
    beam_columns = [
        "base_station", "cell", "beam"
    ]

    beam_keys = pd.MultiIndex.from_frame(
        df[beam_columns]
    )

    distinct_beams = beam_keys.unique()

    if (
        args.max_beams
        and len(distinct_beams) > args.max_beams
    ):

        rng = np.random.default_rng(SEED)

        selected_indices = np.sort(
            rng.choice(
                len(distinct_beams),
                args.max_beams,
                replace=False,
            )
        )

        chosen = distinct_beams.take(
            selected_indices
        )

        beam_mask = beam_keys.isin(chosen)

    else:
        beam_mask = np.ones(
            len(df), dtype=bool
        )

    times = df[
        "target_time"
    ].to_numpy(dtype=np.int32)

    y = df[
        f"target_{args.target}"
    ].to_numpy(dtype=np.float32)

    fold_data = []

    for _, _, train, valid in iter_folds(times):

        train = train[beam_mask[train]]
        valid = valid[beam_mask[valid]]

        x_train, x_valid = fold_features(
            x_raw, names, train, valid
        )

        fold_data.append(
            (
                x_train,
                x_valid,
                y[train],
                y[valid],
            )
        )

    def objective(trial):

        if args.model == "RandomForest":

            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators", 40, 160
                ),
                "max_depth": trial.suggest_int(
                    "max_depth", 8, 24
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf", 5, 50
                ),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", 0.7, 1.0]
                ),
                "max_samples": trial.suggest_float(
                    "max_samples", 0.5, 0.9
                ),
                "random_state": SEED,
                "n_jobs": -1,
            }

            model = RandomForestRegressor(
                **params
            )

        elif args.model == "XGBoost":

            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators", 100, 450
                ),
                "max_depth": trial.suggest_int(
                    "max_depth", 3, 10
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    0.01, 0.15, log=True
                ),
                "subsample": trial.suggest_float(
                    "subsample", 0.65, 1.0
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.65, 1.0
                ),
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "n_jobs": -1,
                "random_state": SEED,
            }

            model = XGBRegressor(**params)

        else:

            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators", 100, 450
                ),
                "num_leaves": trial.suggest_int(
                    "num_leaves", 15, 127
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    0.01, 0.15, log=True
                ),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples", 10, 80
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.65, 1.0
                ),
                "subsample": trial.suggest_float(
                    "subsample", 0.65, 1.0
                ),
                "subsample_freq": 1,
                "n_jobs": -1,
                "random_state": SEED,
                "verbose": -1,
            }

            model = LGBMRegressor(**params)

        rmse_values = []

        for (
            x_train,
            x_valid,
            y_train,
            y_valid,
        ) in fold_data:

            fitted = clone(model).fit(
                x_train, y_train
            )

            predicted = fitted.predict(x_valid)

            rmse_values.append(
                np.sqrt(
                    mean_squared_error(
                        y_valid, predicted
                    )
                )
            )

        return float(np.mean(rmse_values))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=SEED
        ),
    )

    study.optimize(
        objective,
        n_trials=args.trials,
    )

    REPORT_DIR.mkdir(
        parents=True, exist_ok=True
    )

    label = (
        f"tune_{args.target}_{args.model}"
    )

    study.trials_dataframe().to_csv(
        REPORT_DIR / f"{label}_trials.csv",
        index=False,
    )

    best_result = {
        "target": args.target,
        "model": args.model,
        "method": "Optuna TPE",
        "fold_starts": list(FOLD_STARTS),
        "fold_hours": WINDOW_HOURS,
        "selection_data": (
            "Chronological training data before hour 672"
        ),
        "sampled_beams_limit": args.max_beams,
        "trials": args.trials,
        "best_cv_rmse": study.best_value,
        "best_params": study.best_params,
    }

    with (
        REPORT_DIR / f"{label}_best.json"
    ).open("w", encoding="utf-8") as f:

        json.dump(
            best_result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nOPTUNA BEST RESULT")
    print(
        "Best CV RMSE:",
        study.best_value,
    )
    print(
        "Best parameters:",
        study.best_params,
    )


# ============================================================
# 6. Command-line interface
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    for action in ["backtest", "holdout"]:

        command = commands.add_parser(action)

        command.add_argument(
            "--models",
            nargs="+",
            default=[
                "Persistence",
                "LinearRegression",
                "DecisionTree",
                "RandomForest",
                "XGBoost",
                "LightGBM",
            ],
        )

        command.add_argument(
            "--quick",
            action="store_true",
            help="Debugging only",
        )

    sig = commands.add_parser(
        "significance"
    )

    sig.add_argument(
        "--losses",
        type=Path,
        default=(
            REPORT_DIR / "holdout_hourly_losses.csv"
        ),
    )

    sig.add_argument(
        "--reference",
        default="RandomForest",
    )

    sig.add_argument(
        "--contenders",
        nargs="+",
        default=["XGBoost", "LightGBM"],
    )

    sig.add_argument(
        "--bandwidth",
        type=int,
        default=24,
    )

    tuning = commands.add_parser("tune")

    tuning.add_argument(
        "--model",
        choices=[
            "RandomForest",
            "XGBoost",
            "LightGBM",
        ],
        default="RandomForest",
    )

    tuning.add_argument(
        "--target",
        choices=KPI_COLUMNS,
        default="DLPRB",
    )

    tuning.add_argument(
        "--trials",
        type=int,
        default=12,
    )

    tuning.add_argument(
        "--max-beams",
        type=int,
        default=256,
    )

    args = parser.parse_args()

    if args.command == "backtest":

        evaluate(args, holdout=False)

    elif args.command == "holdout":

        evaluate(args, holdout=True)

    elif args.command == "significance":

        significance(
            args.losses,
            args.reference,
            args.contenders,
            args.bandwidth,
        )

    elif args.command == "tune":

        tune(args)


if __name__ == "__main__":
    main()
