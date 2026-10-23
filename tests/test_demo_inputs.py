"""Synthetic fixture tests, without downloading full gotsf-ds."""
import unittest
import numpy as np
import pandas as pd

from src.core.config import KPIS, KEYS
from src.data.prepare_demo_inputs import prepare_demo_frames


class DemoInputTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for beam in range(3):
            for hour in range(180):
                row = dict(Time=hour, base_station=0, cell=0, beam=beam)
                row.update({k: float(hour + beam) for k in KPIS})
                rows.append(row)
        self.data = pd.DataFrame(rows)

    def test_forecast_and_anomaly_shapes(self):
        forecast, anomaly = prepare_demo_frames(self.data)
        self.assertEqual(len(forecast), 169)
        self.assertEqual(len(anomaly), 3 * 3)
        self.assertEqual(forecast.Time.iloc[-1], 179)
        self.assertEqual(sorted(anomaly.Time.unique()), [177, 178, 179])
        self.assertEqual(len(anomaly[KEYS].drop_duplicates()), 3)

    def test_detects_missing_beam_hour(self):
        incomplete = self.data.drop(
            self.data[(self.data.beam == 2) & (self.data.Time == 179)].index
        )
        with self.assertRaisesRegex(ValueError, "Missing hours"):
            prepare_demo_frames(incomplete)

    def test_rejects_short_history(self):
        with self.assertRaises(ValueError):
            prepare_demo_frames(self.data, forecast_hours=24)


if __name__ == "__main__":
    unittest.main()
