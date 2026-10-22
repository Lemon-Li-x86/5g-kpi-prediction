"""Deployment round-trip tests use synthetic data, not historical KPI scores."""
import json
import pickle
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.inference import predict_kpi, KPIS, KEYS

class DeploymentTests(unittest.TestCase):
    def history(self, n=169):
        frame = pd.DataFrame(np.arange(n*4).reshape(n,4)/100, columns=KPIS)
        frame['Time'] = np.arange(n)
        for k in KEYS:
            frame[k] = 1
        return frame

    def test_ml_roundtrip_and_history_validation(self):
        from sklearn.linear_model import LinearRegression
        columns = [f'{k}_z_t' for k in KPIS]
        x = np.arange(40).reshape(10,4)
        model = LinearRegression().fit(x, x[:,0]*2)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'model.pkl'
            with path.open('wb') as f:
                pickle.dump(dict(family='ml', model=model, columns=columns,
                                 mean=[0]*4, scale=[1]*4, target=KPIS[0]), f)
            history = self.history()
            result = predict_kpi(history, path)
            self.assertEqual(result.Time.iloc[0], 169)
            self.assertAlmostEqual(result.DLPRB.iloc[0], model.predict(history[KPIS].iloc[[-1]].to_numpy())[0], places=5)
            with self.assertRaises(ValueError):
                predict_kpi(history.iloc[:-1], path)
            with self.assertRaises(ValueError):
                predict_kpi(history.drop(index=4), path)

    def test_deep_trained_roundtrip(self):
        import torch
        from src.train_deep_learning_models import ForecastNet
        torch.set_num_threads(1)
        for mode in ['lstm', 'gru', 'transformer']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d:
                model = ForecastNet(mode, 4, 8)
                optimizer = torch.optim.Adam(model.parameters())
                seq = torch.randn(2,4,4)
                seasonal = torch.randn(2,4)
                loss = model(seq,seasonal).square().mean()
                loss.backward()
                optimizer.step()
                path = Path(d)/'model.pt'
                torch.save(model.state_dict(),path)
                path.with_suffix('.json').write_text(json.dumps(dict(family='dl', mode=mode, seq_len=4, hidden=8, seasonal_lag=2, mean=[0]*4, scale=[1]*4)))
                result = predict_kpi(self.history(6), path)
                model.eval()
                z = self.history(6)[KPIS].to_numpy(dtype=np.float32)
                with torch.no_grad():
                    expected = model(torch.from_numpy(z[-4:][None]),torch.from_numpy(z[-3][None])).numpy()[0]
                np.testing.assert_allclose(result[KPIS].to_numpy()[0],expected, rtol=1e-5)

if __name__ == '__main__':
    unittest.main()

class AnomalyTests(unittest.TestCase):
    def test_original_algorithms_roundtrip(self):
        from sklearn.ensemble import IsolationForest
        from sklearn.svm import OneClassSVM
        from src.detect_kpi_anomalies import features, detection_thresholds
        from src.inference import detect_anomalies
        rng = np.random.default_rng(42)
        train = rng.normal(size=(2,20,4)).astype(np.float32)
        values = rng.normal(size=(2,4,4)).astype(np.float32)
        x = features(train).reshape(-1,8)
        forest = IsolationForest(n_estimators=10,random_state=42).fit(x)
        svm = OneClassSVM(nu=.02).fit(x)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'anomalies.pkl'
            with path.open('wb') as f:
                pickle.dump(dict(train=train,forest=forest,svm=svm),f)
            result = detect_anomalies(values,path)
            sigma,iqr,_,_ = detection_thresholds(train,values[:,1:])
            np.testing.assert_array_equal(result.sigma,sigma.any(axis=2).reshape(-1))
            np.testing.assert_array_equal(result.iqr,iqr.any(axis=2).reshape(-1))
            np.testing.assert_array_equal(result.one_class_svm,svm.predict(features(values)[:,1:].reshape(-1,8)) == -1)
