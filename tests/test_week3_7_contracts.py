"""Source/artefact smoke contracts for original weeks 3 through 7.

Uses only Python standard library and tracked files. It intentionally does not
pretend to validate local downloaded datasets or model weights.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class WeekThreeToSevenContracts(unittest.TestCase):
    def test_each_week_has_source_code(self):
        required = {
            3: [
                "src/data/download_kpi_data.py", "src/data/audit_kpi_data.py",
                "src/data/preprocess_kpi_data.py", "src/data/build_ml_features.py",
            ],
            4: ["src/models/train_traditional_models.py", "src/models/inference.py"],
            5: ["src/models/train_deep_learning_models.py", "src/models/architectures.py"],
            6: [
                "src/models/detect_kpi_anomalies.py",
                "src/analysis/analyze_anomaly_statistics.py",
            ],
            7: [
                "app.py", "src/ui/dashboard_data.py", "src/ui/web_workflow.py",
                "src/services/model_registry.py", "src/services/web_jobs.py",
                "src/services/web_stream_worker.py",
            ],
        }
        for week, paths in required.items():
            for path in paths:
                with self.subTest(week=week, path=path):
                    self.assertTrue((ROOT / path).is_file())

    def test_tracked_historical_results_exist(self):
        required = [
            "reports/preprocessing_summary.json",
            "reports/raw_data_audit_summary.csv",
            "reports/ml_feature_summary.json",
            "reports/model_metrics.csv",
            "reports/prediction_timeseries.csv",
            "reports/w3_deep_learning_metrics.csv",
            "reports/week3/w3_ml_dl_all_metrics.csv",
            "reports/week3/w3_ml_dl_comparison_report.md",
            "reports/w3_anomaly_method_summary.csv",
            "reports/w3_anomaly_candidate_events.csv",
            "reports/w3_anomaly_root_cause_preliminary.md",
            "reports/w4_anomaly_diagnostics_report.md",
        ]
        for path in required:
            with self.subTest(path=path):
                self.assertTrue((ROOT / path).is_file())

    def test_historical_data_split_and_four_kpis(self):
        data = json.loads(
            (ROOT / "reports/preprocessing_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(data["dataset"]["rows"], 2419200)
        self.assertEqual(data["dataset"]["beam_count"], 2880)
        self.assertEqual(data["split"]["train_time"], [0, 671])
        self.assertEqual(data["split"]["test_time"], [672, 839])
        self.assertEqual(set(data["standardisation"]["kpi_columns"]),
                         {"DLPRB", "DLThpVol", "DLThpTime", "MR_number"})

    def test_ml_and_dl_metrics_have_expected_columns(self):
        for path in ("reports/model_metrics.csv", "reports/w3_deep_learning_metrics.csv"):
            with self.subTest(path=path), (ROOT / path).open(
                encoding="utf-8", newline=""
            ) as file:
                reader = csv.DictReader(file)
                self.assertTrue({"MAE", "RMSE", "R2"}.issubset(reader.fieldnames or ()))
                self.assertIsNotNone(next(reader, None))

    def test_cross_family_comparison_has_four_targets(self):
        with (ROOT / "reports/week3/w3_ml_dl_all_metrics.csv").open(
            encoding="utf-8", newline=""
        ) as file:
            data = list(csv.DictReader(file))
        self.assertEqual({row["KPI"] for row in data},
                         {"DLPRB", "DLThpVol", "DLThpTime", "MR_number"})
        self.assertTrue({"Traditional ML", "Deep learning"} <= {row["family"] for row in data})

    def test_documented_modules_and_manual_exist(self):
        for path in (
            "README.md", "USER_GUIDE.md", "requirements.txt",
            "docs/ARCHITECTURE.md", "docs/TECHNICAL_REPORT.md",
            "docs/WEEK3_7_AUDIT.md",
        ):
            with self.subTest(path=path):
                self.assertTrue((ROOT / path).is_file())


if __name__ == "__main__":
    unittest.main()
