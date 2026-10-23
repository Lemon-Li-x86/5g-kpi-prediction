"""Trusted local deployment artifacts; never load pickle files from strangers."""
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd

from src.core.config import ROOT, KPIS, KEYS, ML_LAGS, ML_HISTORY_HOURS


def save_ml(model, name, target, columns, output_dir=None):
    """Save a trusted model bundle; Web runs can isolate each experiment."""
    root = ROOT
    scaling = json.loads((root / 'reports/preprocessing_summary.json').read_text(encoding='utf-8'))['standardisation']
    bundle = dict(version=1, family='ml', model=model, target=target,
                  columns=columns, mean=[scaling['mean'][k] for k in KPIS],
                  scale=[scaling['scale'][k] for k in KPIS])
    directory = Path(output_dir) if output_dir is not None else root / 'models'
    path = directory / f'{name}_{target}.pkl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        pickle.dump(bundle, f)
    return path


def predict_kpi(history, artifact):
    """One next-hour forecast per beam from contiguous observed raw KPI history.

    artifact is an ML .pkl bundle or a DL checkpoint with .json sidecar.
    Returns original-unit predictions with beam identifiers and target Time.
    """
    path = Path(artifact)
    if path.suffix == '.pkl':
        with path.open('rb') as f:
            meta = pickle.load(f)
    else:
        meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    results = []
    for key, group in history.groupby(KEYS, sort=True):
        group = group.sort_values('Time')
        if group.Time.duplicated().any() or not np.all(np.diff(group.Time) == 1):
            raise ValueError('History must have unique contiguous hours per beam')
        raw = group[KPIS].to_numpy(dtype=np.float32)
        if not np.isfinite(raw).all():
            raise ValueError('Non-finite KPI input')
        z = ((raw - meta['mean']) / meta['scale']).astype(np.float32)
        row = dict(zip(KEYS, key))
        row['Time'] = int(group.Time.iloc[-1]) + 1
        if meta['family'] == 'ml':
            features = {}
            for j, k in enumerate(KPIS):
                features[f'{k}_z_t'] = z[-1, j]
                for lag in ML_LAGS:
                    if len(z) <= lag:
                        raise ValueError(f'ML requires {ML_HISTORY_HOURS} observed hours')
                    features[f'{k}_z_lag{lag}'] = z[-lag-1, j]
            row[meta['target']] = float(meta['model'].predict(np.array([[features[c] for c in meta['columns']]], dtype=np.float32))[0])
        else:
            import torch
            from src.models.architectures import ForecastNet
            if len(z) < max(meta['seq_len'], meta['seasonal_lag'] + 1):
                raise ValueError('Insufficient sequence/seasonal history')
            model = ForecastNet(meta['mode'], meta['seq_len'], meta['hidden'])
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            model.eval()
            with torch.no_grad():
                pred = model(torch.from_numpy(z[-meta['seq_len']:][None]), torch.from_numpy(z[-meta['seasonal_lag']-1][None])).numpy()[0]
            row.update(zip(KPIS, pred * meta['scale'] + meta['mean']))
        results.append(row)
    if not results:
        raise ValueError('Empty history')
    return pd.DataFrame(results)


def detect_anomalies(values, artifact):
    """Standardized [beam,time,4] values in the saved beam order.

    Include preceding context as the first hour; it is excluded from results.
    No estimator is fitted during detection. Candidates are not confirmed faults.
    """
    from src.models.detect_kpi_anomalies import detection_thresholds, features
    with Path(artifact).open('rb') as f:
        bundle = pickle.load(f)
    labels = None
    if isinstance(values, pd.DataFrame):
        ordered = values.sort_values(KEYS + ['Time'])
        groups = list(ordered.groupby(KEYS, sort=True))
        actual = ordered[KEYS].drop_duplicates().reset_index(drop=True)
        expected = bundle['keys'][KEYS].reset_index(drop=True)
        if not actual.equals(expected):
            raise ValueError('Unknown, missing, or mismatched beam identifiers')
        times = [g.Time.to_numpy() for _, g in groups]
        if any(not np.array_equal(t, times[0]) for t in times) or not np.all(np.diff(times[0]) == 1):
            raise ValueError('Expected aligned contiguous beam timelines')
        scaling = bundle['scaling']
        raw = ordered[KPIS].to_numpy(dtype=np.float32)
        values = ((raw - [scaling['mean'][k] for k in KPIS]) / [scaling['scale'][k] for k in KPIS]).reshape(len(groups),len(times[0]),4)
        labels = ordered.groupby(KEYS, sort=True).tail(len(times[0])-1)[KEYS+['Time']].reset_index(drop=True)
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3 or values.shape[1] < 2 or values.shape[2] != 4 or not np.isfinite(values).all():
        raise ValueError('Expected finite [beam, context+hours, 4] values')
    train = bundle['train']
    if values.shape[0] != train.shape[0]:
        raise ValueError('Beam count/order must match training artifact')
    sigma, iqr, _, _ = detection_thresholds(train, values[:, 1:])
    x = features(values)[:, 1:].reshape(-1, 8)
    forest = bundle['forest'].decision_function(x) < 0
    sigma = sigma.any(axis=2).reshape(-1)
    iqr = iqr.any(axis=2).reshape(-1)
    result = pd.DataFrame({'sigma': sigma, 'iqr': iqr, 'isolation_forest': forest,
                         'one_class_svm': bundle['svm'].predict(x) == -1,
                         'consensus': forest & (sigma | iqr)})

    return pd.concat([labels, result], axis=1) if labels is not None else result
