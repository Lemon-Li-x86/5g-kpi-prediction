"""Regression checks for shared data contracts and independent DL model settings.

Identifiers, temporal boundaries and ML lags have one source. Hyperparameters
with the same spelling may have model-specific meanings or valid ranges.
"""
import unittest

from src.core.config import (
    KPIS, KEYS, ML_LAGS, ML_HISTORY_HOURS, ML_FORECAST_HORIZON,
    DATASET_TEST_START_HOUR,
)
from src.core.training_config import normalize_params, params_to_cli


class SharedConfigContracts(unittest.TestCase):
    def test_four_kpis_and_beam_keys_have_canonical_order(self):
        self.assertEqual(KPIS, ["DLPRB", "DLThpVol", "DLThpTime", "MR_number"])
        self.assertEqual(KEYS, ["base_station", "cell", "beam"])

    def test_forecast_history_is_derived_from_lag_contract(self):
        self.assertEqual(ML_LAGS, (1, 2, 24, 168))
        self.assertEqual(ML_HISTORY_HOURS, max(ML_LAGS) + 1)
        self.assertEqual(ML_FORECAST_HORIZON, 1)
        self.assertEqual(DATASET_TEST_START_HOUR, 672)

    def test_independent_dl_models_allow_different_hidden_widths(self):
        # A GRU hidden-state width need not satisfy the Transformer's
        # number-of-heads divisibility rule.
        gru = normalize_params("train_dl", {"hidden": 17}, "gru")
        lstm = normalize_params("train_dl", {"hidden": 19}, "lstm")
        transformer = normalize_params("train_dl", {"hidden": 20}, "transformer")
        self.assertEqual((gru["hidden"], lstm["hidden"], transformer["hidden"]),
                         (17, 19, 20))
        with self.assertRaisesRegex(ValueError, "attention heads"):
            normalize_params("train_dl", {"hidden": 17}, "transformer")
        self.assertEqual(
            params_to_cli("train_dl", {"hidden": 17}, "gru")[
                params_to_cli("train_dl", {"hidden": 17}, "gru").index("--hidden") + 1
            ], "17"
        )

    def test_anomaly_and_regression_settings_are_scoped(self):
        anomaly = normalize_params("train_anomaly", {"seed": 9})
        gru = normalize_params("train_dl", {"seed": 7}, "gru")
        self.assertEqual((anomaly["seed"], gru["seed"]), (9, 7))
        with self.assertRaises(ValueError):
            normalize_params("train_dl", {"if_contamination": 0.02}, "gru")


if __name__ == "__main__":
    unittest.main()
