"""Regression tests for versioned local artifact discovery and archiving."""
import json
import tempfile
import unittest
from pathlib import Path

from src.services.model_registry import archive_models, list_models


class RegistryTests(unittest.TestCase):
    def test_archive_keeps_original_and_discovers_versions(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            legacy = root / "models"
            legacy.mkdir()
            (legacy / "RandomForest_DLPRB.pkl").write_bytes(b"fake-test-content")
            (legacy / "w3_gru.pt").write_bytes(b"fake-test-weights")
            (legacy / "w3_gru.json").write_text(
                json.dumps({"family": "dl", "mode": "gru"}), encoding="utf-8"
            )
            before = list_models(root)
            self.assertEqual(len(before), 2)
            self.assertTrue(all(not x.archived for x in before))
            copied = archive_models(root, reason="unit_test")
            self.assertEqual(len(copied), 2)
            self.assertTrue(all(x.archived for x in copied))
            self.assertEqual(len(list_models(root)), 4)
            for item in copied:
                self.assertTrue(Path(item.path).exists())
                if item.family == "dl":
                    self.assertTrue(Path(item.path).with_suffix(".json").is_file())
            self.assertTrue((legacy / "RandomForest_DLPRB.pkl").exists())

    def test_invalid_torch_without_sidecar_not_listed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            legacy = root / "models"
            legacy.mkdir()
            (legacy / "w3_gru.pt").write_bytes(b"unknown")
            self.assertEqual(list_models(root), [])


if __name__ == "__main__":
    unittest.main()
