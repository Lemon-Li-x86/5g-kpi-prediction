"""Test association against the actual beam-hour data of anomaly inference."""
import unittest

import pandas as pd

from src.core.config import KPIS
from src.analysis.session_kpi_analysis import candidate_kpi_correlation


class CurrentAnomalyAssociationTests(unittest.TestCase):
    def setUp(self):
        histories = []
        detections = []
        for hour in range(10, 15):
            row = {"base_station": 2, "cell": 1, "beam": 5, "Time": hour}
            row.update({k: float(hour * (j + 1)) for j, k in enumerate(KPIS)})
            histories.append(row)
            if hour > 10:
                detections.append({**{c: row[c] for c in
                                       ["base_station", "cell", "beam", "Time"]},
                                   "sigma": True, "iqr": True,
                                   "isolation_forest": True,
                                   "one_class_svm": True,
                                   "consensus": hour > 11})
        self.history = pd.DataFrame(histories)
        self.flags = pd.DataFrame(detections)

    def test_only_flagged_beam_hours_are_correlated(self):
        candidates, corr = candidate_kpi_correlation(self.history, self.flags)
        self.assertEqual(candidates["Time"].tolist(), [12, 13, 14])
        self.assertEqual(len(candidates), 3)
        self.assertAlmostEqual(corr.loc["DLPRB", "MR_number"], 1.0)

    def test_shuffled_history_does_not_shift_labels(self):
        shuffled = self.history.sample(frac=1, random_state=42)
        candidates, corr = candidate_kpi_correlation(shuffled, self.flags)
        self.assertEqual(candidates["DLPRB"].tolist(), [12.0, 13.0, 14.0])

    def test_reject_duplicate_keys(self):
        bad = pd.concat([self.history, self.history.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            candidate_kpi_correlation(bad, self.flags)

    def test_reject_inconsistent_beam_timestamps(self):
        missing = self.history.loc[self.history["Time"] != 13]
        with self.assertRaisesRegex(ValueError, "absent"):
            candidate_kpi_correlation(missing, self.flags)

    def test_too_few_candidates_reports_no_correlation(self):
        flags = self.flags.copy()
        flags["consensus"] = [True, False, False, False]
        samples, corr = candidate_kpi_correlation(self.history, flags)
        self.assertEqual(len(samples), 1)
        self.assertTrue(corr.empty)


if __name__ == "__main__":
    unittest.main()
