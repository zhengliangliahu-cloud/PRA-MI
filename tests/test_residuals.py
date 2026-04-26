import unittest

from autoeegencoder.analysis.residuals import compute_residual_rows


class ResidualTests(unittest.TestCase):
    def test_median_reference_recomputed_within_group(self):
        rows = [
            {"dataset": "d", "protocol_tier": "P0", "method": "m", "test_subject": "1", "balanced_accuracy": "0.80"},
            {"dataset": "d", "protocol_tier": "P0", "method": "m", "test_subject": "2", "balanced_accuracy": "0.60"},
            {"dataset": "d", "protocol_tier": "P0", "method": "m", "test_subject": "3", "balanced_accuracy": "0.50"},
        ]
        out = compute_residual_rows(rows)
        by_subject = {row["test_subject"]: row for row in out}
        self.assertEqual(by_subject["1"]["error"], "0.200000")
        self.assertEqual(by_subject["1"]["median_other_subject_error"], "0.450000")
        self.assertEqual(by_subject["1"]["residual_error"], "-0.250000")


if __name__ == "__main__":
    unittest.main()

