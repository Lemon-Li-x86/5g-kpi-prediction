"""Tests expose missing-hour windows and unsafe experiment output reuse."""
from pathlib import Path
import json
import pickle
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from src.core.config import KPIS, KEYS
from src.models.train_deep_learning_models import load_tensor, WindowDataset
from src.models.detect_kpi_anomalies import load_data


class AlgorithmAuditTests(unittest.TestCase):
    def frame(self, times):
        rows = []
        for beam in range(2):
            for t in times:
                row = dict(zip(KEYS, [0, 0, beam]), Time=t)
                for j, k in enumerate(KPIS):
                    row[k] = 10000 * beam + 100 * j + t
                    row[k + '_z'] = row[k] / 10
                rows.append(row)
        return pd.DataFrame(rows)

    def test_loaders_reject_joint_missing_hour(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'input.csv'
            self.frame([0, 1, 3, 4]).to_csv(p, index=False)
            for loader in (load_tensor, load_data):
                with self.subTest(loader=loader.__name__), self.assertRaisesRegex(ValueError, 'contiguous'):
                    loader(p)

    def test_anomaly_loader_rejects_duplicate_and_fractional_hours(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'input.csv'
            for times in ([0, 1, 1, 2], [.5, 1.5, 2.5]):
                self.frame(times).to_csv(p, index=False)
                with self.subTest(times=times), self.assertRaises(ValueError):
                    load_data(p)

    def test_dl_scaler_ignores_all_future_changes(self):
        from src.models.train_deep_learning_models import fit_training_prefix
        raw = np.arange(2 * 12 * 4, dtype=np.float32).reshape(2, 12, 4)
        data, mean, scale = fit_training_prefix(raw, np.arange(12), 8)
        changed = raw.copy(); changed[:, 8:] += 100000
        other, mean2, scale2 = fit_training_prefix(changed, np.arange(12), 8)
        np.testing.assert_array_equal(mean, mean2)
        np.testing.assert_array_equal(scale, scale2)
        np.testing.assert_array_equal(data[:, :8], other[:, :8])
        np.testing.assert_allclose(data * scale + mean, raw, rtol=1e-6, atol=1e-5)

    def test_window_uses_target_exclusive_history_and_origin_season(self):
        x = np.arange(2 * 175 * 4, dtype=np.float32).reshape(2, 175, 4)
        ds = WindowDataset(x, np.array([169, 170]), 24, 168, np.array([0, 3]))
        for i, beam, ti in [(0, 0, 169), (1, 1, 170)]:
            seq, seasonal, target, b, t = ds[i]
            np.testing.assert_array_equal(seq, x[beam, ti-24:ti])
            np.testing.assert_array_equal(seasonal, x[beam, ti-169])
            np.testing.assert_array_equal(target, x[beam, ti])
            self.assertEqual((b, t), (beam, ti))

    def test_forecast_rejects_fractional_hour_before_casting_target(self):
        from src.models.inference import predict_kpi
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'m.pkl'
            with p.open('wb') as f:
                pickle.dump(dict(family='ml',model=CaptureEstimator(),columns=[f'{k}_z_t' for k in KPIS],
                                 mean=[0]*4,scale=[1]*4,target=KPIS[0]),f)
            with self.assertRaisesRegex(ValueError,'integer'):
                predict_kpi(self.frame(np.arange(169)+.5),p)

    def test_existing_experiment_is_rejected_before_writes(self):
        from src.models.train_deep_learning_models import ensure_fresh_output
        with tempfile.TemporaryDirectory() as d:
            out = Path(d); (out / 'models').mkdir()
            original = out / 'models/w3_gru.pt'; original.write_bytes(b'old weights')
            with self.assertRaisesRegex(ValueError, 'separate --out'):
                ensure_fresh_output(out)
            self.assertEqual(original.read_bytes(), b'old weights')

    def test_twenty_features_match_inference_on_each_beam(self):
        from src.data.build_ml_features import build_features
        from src.models.inference import predict_kpi
        frame = self.frame(range(171))
        result = build_features(frame.copy())
        design, columns = result[0], result[1]
        self.assertEqual(len(columns), 20)
        for beam in range(2):
            row = design[(design.beam == beam) & (design.feature_time == 169)].iloc[0]
            for j, k in enumerate(KPIS):
                for lag, suffix in [(0, 't'), (1, 'lag1'), (2, 'lag2'), (24, 'lag24'), (168, 'lag168')]:
                    self.assertAlmostEqual(row[f'{k}_z_{suffix}'], (10000*beam+100*j+169-lag)/10, places=3)
                self.assertEqual(row[f'target_{k}'], 10000*beam+100*j+170)
            model = CaptureEstimator()
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / 'm.pkl'
                with p.open('wb') as f:
                    pickle.dump(dict(family='ml', model=model, columns=columns, mean=[0]*4,
                                     scale=[10]*4, target=KPIS[0]), f)
                hist = frame[(frame.beam == beam) & (frame.Time <= 169)]
                with patch.object(CaptureEstimator, 'predict', return_value=np.array([0.])) as call:
                    predict_kpi(hist, p)
                np.testing.assert_allclose(call.call_args.args[0][0], row[columns].to_numpy(dtype=float), rtol=1e-6)


class CaptureEstimator:
    def predict(self, x):
        return np.zeros(len(x))
