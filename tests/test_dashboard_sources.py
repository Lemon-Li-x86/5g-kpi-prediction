"""Check dashboard version selection never mixes a new run with old reports."""
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.ui.dashboard_data import (
    available_prediction_models,
    list_experiment_runs,
    load_anomaly_summary,
    load_correlation_matrix,
    load_correlation_tests,
    load_model_metrics,
    load_prediction_series,
    run_report_dir,
)


class DashboardSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.history = self.root / "reports"
        self.history.mkdir()
        pd.DataFrame([{
            "target": "DLPRB", "model": "OldModel",
            "MAE": 99.0, "RMSE": 99.0, "R2": -99.0,
        }]).to_csv(self.history / "model_metrics.csv", index=False)
        pd.DataFrame([{
            "target": "DLPRB", "model": "OldModel",
            "target_time": 700, "actual_mean": 2, "prediction_mean": 3,
        }]).to_csv(self.history / "prediction_timeseries.csv", index=False)

    def run_dir(self, run_id, task):
        folder = self.root / "resources" / "experiments" / run_id
        reports = folder / "reports"
        reports.mkdir(parents=True)
        (folder / "requested_parameters.json").write_text(
            json.dumps({"task": task, "parameters": {"quick": True}}),
            encoding="utf-8",
        )
        return reports

    def test_ml_run_reads_only_its_own_scores_and_predictions(self):
        reports = self.run_dir("ml_20261023T075016Z_0e8e81", "train_ml")
        pd.DataFrame([{
            "target": "DLPRB", "model": "RandomForest",
            "MAE": 0.2, "RMSE": 0.3, "R2": 0.4,
        }]).to_csv(reports / "model_metrics.csv", index=False)
        pd.DataFrame([{
            "target": "DLPRB", "model": "RandomForest",
            "target_time": 672, "actual_mean": 1.0, "prediction_mean": 1.1,
        }]).to_csv(reports / "prediction_timeseries.csv", index=False)
        versions = list_experiment_runs("forecast", self.root)
        self.assertEqual(versions, ["ml_20261023T075016Z_0e8e81"])
        source = run_report_dir(versions[0], self.root)
        metrics = load_model_metrics(source)
        self.assertEqual(metrics["model"].tolist(), ["RandomForest"])
        self.assertAlmostEqual(metrics.iloc[0]["R2"], 0.4)
        self.assertEqual(available_prediction_models(source), ["RandomForest"])
        series = load_prediction_series("RandomForest", "DLPRB", source)
        self.assertEqual(series["prediction"].tolist(), [1.1])
        self.assertTrue(load_prediction_series("OldModel", "DLPRB", source).empty)

    def test_gru_run_uses_its_own_hourly_csv(self):
        reports = self.run_dir("gru_20261023T071632Z_9c47c2", "train_dl")
        pd.DataFrame([{
            "KPI": "DLPRB", "model": "GRU",
            "MAE": 0.5, "RMSE": 0.9, "R2": 0.23,
        }]).to_csv(reports / "w3_deep_learning_metrics.csv", index=False)
        pd.DataFrame([{
            "Time": 672, "DLPRB_actual": 4.5, "DLPRB_pred": 4.2,
        }]).to_csv(reports / "w3_gru_hourly_predictions.csv", index=False)
        source = run_report_dir("gru_20261023T071632Z_9c47c2", self.root)
        self.assertEqual(available_prediction_models(source), ["GRU"])
        self.assertEqual(load_model_metrics(source)["model"].tolist(), ["GRU"])
        self.assertEqual(
            load_prediction_series("GRU", "DLPRB", source)["prediction"].tolist(),
            [4.2],
        )

    def test_anomaly_run_never_borrows_historical_significance(self):
        reports = self.run_dir("anomaly_20261023T072714Z_3ed272", "train_anomaly")
        pd.DataFrame([{
            "method": "3sigma", "flagged_samples": 320, "flag_rate": 0.0556,
        }]).to_csv(reports / "w3_anomaly_method_summary.csv", index=False)
        matrix = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]],
                              columns=["DLPRB", "MR_number"],
                              index=["DLPRB", "MR_number"])
        matrix.to_csv(reports / "w3_anomaly_kpi_correlations.csv")
        (self.history / "w4_anomaly_kpi_correlation_tests.csv").write_text(
            "KPI_1,KPI_2,holm_adjusted_p_value\nDLPRB,MR_number,0.001\n",
            encoding="utf-8",
        )
        self.assertEqual(
            list_experiment_runs("anomaly", self.root),
            ["anomaly_20261023T072714Z_3ed272"],
        )
        self.assertEqual(list_experiment_runs("forecast", self.root), [])
        self.assertEqual(int(load_anomaly_summary(reports).iloc[0]["flagged_samples"]), 320)
        corr, label = load_correlation_matrix(reports)
        self.assertAlmostEqual(corr.loc["DLPRB", "MR_number"], 0.5)
        self.assertIn("exploratory", label)
        self.assertTrue(load_correlation_tests(reports).empty)

    def test_incomplete_runs_are_not_visible(self):
        reports = self.run_dir("ml_incomplete", "train_ml")
        # requested_parameters exists, but no metrics yet.
        self.assertEqual(list_experiment_runs("forecast", self.root), [])
        with self.assertRaises(ValueError):
            run_report_dir("ml_incomplete", self.root)
        with self.assertRaises(ValueError):
            run_report_dir("../../reports", self.root)


if __name__ == "__main__":
    unittest.main()
