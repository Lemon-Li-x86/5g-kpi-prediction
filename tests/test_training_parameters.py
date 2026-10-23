"""Regression tests for editable Web training parameters and run isolation."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core.training_config import normalize_params, params_to_cli


class ParameterContractTests(unittest.TestCase):
    def test_dl_defaults_match_existing_cli(self):
        p = normalize_params("train_dl", {}, "gru")
        self.assertEqual((p["epochs"], p["batch_size"], p["lr"]), (12, 256, 0.001))
        self.assertEqual((p["seq_len"], p["seasonal_lag"], p["hidden"]), (24, 168, 32))

    def test_args_reach_training_program(self):
        args = params_to_cli(
            "train_dl", {"epochs": 6, "batch_size": 64, "lr": 0.0005, "device": "cpu"}, "lstm"
        )
        for key, value in (("--epochs", "6"), ("--batch-size", "64"),
                           ("--lr", "0.0005"), ("--device", "cpu")):
            self.assertEqual(args[args.index(key) + 1], value)

    def test_reject_bad_inputs_before_launch(self):
        invalid = [
            ("train_dl", {"epochs": 0}, "lstm"),
            ("train_dl", {"batch_size": "64"}, "lstm"),
            ("train_dl", {"unknown_flag": 1}, "gru"),
            ("train_dl", {"hidden": 30}, "transformer"),
            ("train_dl", {"patience": 20, "epochs": 4}, "gru"),
            ("train_dl", {"lr": float("nan")}, "lstm"),
            ("train_dl", {"device": "anywhere"}, "lstm"),
            ("train_anomaly", {"if_contamination": 0.8}, ""),
            ("download", {"epochs": 2}, ""),
        ]
        for task, params, model in invalid:
            with self.subTest(task=task, params=params), self.assertRaises(ValueError):
                normalize_params(task, params, model)

    def test_anomaly_cli_parameters(self):
        args = params_to_cli("train_anomaly", {"if_contamination": 0.05, "svm_train_sample": 500})
        self.assertEqual(args[args.index("--if-contamination") + 1], "0.05")
        self.assertEqual(args[args.index("--svm-train-sample") + 1], "500")

    def test_isolated_run_keeps_requested_parameters(self):
        from src.services import web_jobs
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            calls = []

            def fake_train(command, **kwargs):
                calls.append(command)
                out = Path(command[command.index("--out") + 1])
                checkpoint = out / "models" / "w3_gru.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(b"example-only-weight")
                checkpoint.with_suffix(".json").write_text(
                    json.dumps({"family": "dl", "mode": "gru"}), encoding="utf-8"
                )

            with patch.object(web_jobs, "ROOT", root), patch.object(
                web_jobs.subprocess, "run", side_effect=fake_train
            ):
                self.assertEqual(web_jobs.run("train_dl", "gru", {"epochs": 3, "batch_size": 64}), 0)
            output = json.loads((root / "resources/jobs/current.json").read_text(encoding="utf-8"))
            self.assertEqual(output["status"], "completed")
            self.assertEqual(output["parameters"]["epochs"], 3)
            self.assertEqual(output["parameters"]["batch_size"], 64)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][calls[0].index("--epochs") + 1], "3")
            self.assertEqual(calls[0][calls[0].index("--batch-size") + 1], "64")
            run = Path(output["output_dir"])
            request = json.loads((run / "requested_parameters.json").read_text(encoding="utf-8"))
            self.assertEqual(request["parameters"]["epochs"], 3)
            self.assertTrue(Path(output["model_path"]).is_file())
            saved_request = json.loads((Path(output["model_path"]).parent / "requested_parameters.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_request["parameters"]["batch_size"], 64)


if __name__ == "__main__":
    unittest.main()
