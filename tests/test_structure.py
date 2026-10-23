"""Verify the clean, layered W4 package layout without legacy wrappers."""
import ast
from pathlib import Path
import unittest

from src.core.config import ROOT, KPIS, KEYS

MODULES = {
    "data": ("download_kpi_data", "audit_kpi_data", "preprocess_kpi_data", "build_ml_features"),
    "models": ("architectures", "train_traditional_models", "train_deep_learning_models", "detect_kpi_anomalies", "inference"),
    "analysis": ("compare_model_performance", "analyze_anomaly_statistics", "validate_model_robustness", "review_model_results"),
    "services": ("model_registry", "web_jobs", "web_stream_worker"),
    "ui": ("dashboard_data", "web_workflow"),
}


class ProjectLayoutTests(unittest.TestCase):
    def test_project_root_and_fields(self):
        self.assertEqual(ROOT, Path(__file__).resolve().parents[1])
        self.assertEqual(KPIS, ["DLPRB", "DLThpVol", "DLThpTime", "MR_number"])
        self.assertEqual(KEYS, ["base_station", "cell", "beam"])

    def test_layered_implementations_exist_and_parse(self):
        for package, modules in MODULES.items():
            self.assertTrue((ROOT / "src" / package / "__init__.py").is_file())
            for name in modules:
                with self.subTest(name=f"{package}.{name}"):
                    path = ROOT / "src" / package / f"{name}.py"
                    self.assertTrue(path.is_file())
                    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_no_legacy_scripts_remain(self):
        self.assertEqual({p.name for p in (ROOT / "src").glob("*.py")}, {"__init__.py"})

    def test_background_jobs_use_new_modules(self):
        from src.services.web_jobs import TASKS
        for steps in TASKS.values():
            for args in steps:
                with self.subTest(target=args[0]):
                    self.assertGreaterEqual(args[0].count("."), 2)
        stream = (ROOT / "src/services/web_stream_worker.py").read_text(encoding="utf-8")
        ui = (ROOT / "src/ui/web_workflow.py").read_text(encoding="utf-8")
        self.assertIn("src.services.web_jobs", stream)
        self.assertIn("src.services.web_stream_worker", ui)

    def test_shared_architecture(self):
        from src.models.architectures import ForecastNet
        from src.models.train_deep_learning_models import ForecastNet as TrainingForecastNet
        self.assertIs(ForecastNet, TrainingForecastNet)


if __name__ == "__main__":
    unittest.main()
