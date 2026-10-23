"""Structural contracts for the reorganized W4 codebase."""
from pathlib import Path
import unittest

from src.core.config import ROOT, KPIS, KEYS


class ProjectLayoutTests(unittest.TestCase):
    def test_root_and_expected_fields(self):
        self.assertEqual(ROOT, Path(__file__).resolve().parents[1])
        self.assertEqual(KPIS, ["DLPRB", "DLThpVol", "DLThpTime", "MR_number"])
        self.assertEqual(KEYS, ["base_station", "cell", "beam"])

    def test_legacy_script_wrappers_resolve(self):
        mapping = {
            "download_kpi_data": "data",
            "preprocess_kpi_data": "data",
            "build_ml_features": "data",
            "train_traditional_models": "models",
            "train_deep_learning_models": "models",
            "detect_kpi_anomalies": "models",
            "inference": "models",
            "analyze_anomaly_statistics": "analysis",
            "model_registry": "services",
            "web_jobs": "services",
            "web_stream_worker": "services",
            "dashboard_data": "ui",
            "web_workflow": "ui",
        }
        for name, package in mapping.items():
            with self.subTest(name=name):
                implementation = ROOT / "src" / package / (name + ".py")
                wrapper = ROOT / "src" / (name + ".py")
                self.assertTrue(implementation.is_file())
                self.assertTrue(wrapper.is_file())
                self.assertIn(f"src.{package}.{name}", wrapper.read_text(encoding="utf-8"))

    def test_shared_model_architecture(self):
        from src.models.architectures import ForecastNet
        from src.train_deep_learning_models import ForecastNet as LegacyForecastNet
        self.assertIs(ForecastNet, LegacyForecastNet)

    def test_service_alias(self):
        from src.services.model_registry import list_models
        from src.model_registry import list_models as legacy_list_models
        self.assertIs(list_models, legacy_list_models)


if __name__ == "__main__":
    unittest.main()
