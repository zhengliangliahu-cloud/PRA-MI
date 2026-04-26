import unittest

import numpy as np

from autoeegencoder.diagnostics.qri import build_qri_reference, compute_qri, derive_qri_signal


class QRITests(unittest.TestCase):
    def test_qri_outputs_p0_safe_local_components(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(4, 3, 32)).astype("float32")
        X[0, 2, :] = 0.0
        diag = compute_qri(X, sfreq=64)
        self.assertEqual(diag.trial_risk.shape, (4,))
        self.assertEqual(diag.channel_reliability.shape, (4, 3))
        self.assertEqual(diag.trial_availability, "D0b")
        self.assertTrue(all(value == "D0b" for value in diag.availability.values()))
        self.assertIn("qri_local_spatial_deviation_risk", diag.components)
        self.assertIn("qri_local_burst_risk", diag.components)
        self.assertGreater(diag.components["qri_local_flatline_risk"][0], 0.0)
        self.assertLess(diag.channel_reliability[0, 2], diag.channel_reliability[0, 0])

    def test_source_referenced_qri_marks_d0a_components(self):
        rng = np.random.default_rng(1)
        source = rng.normal(scale=0.5, size=(12, 4, 64)).astype("float32")
        source[:, 1, :] *= 1.2
        ref = build_qri_reference(source, sfreq=64)
        test = source[:2].copy()
        test[0, 2, :] *= 4.0
        diag = compute_qri(test, sfreq=64, reference=ref)
        self.assertEqual(diag.trial_availability, "D0a")
        self.assertEqual(diag.availability["qri_source_variance_deviation_risk"], "D0a")
        self.assertEqual(diag.availability["qri_source_high_frequency_deviation_risk"], "D0a")
        self.assertGreater(diag.trial_risk[0], diag.trial_risk[1])

    def test_qri_ablation_signals_are_available(self):
        rng = np.random.default_rng(2)
        source = rng.normal(scale=0.5, size=(10, 4, 64)).astype("float32")
        ref = build_qri_reference(source, sfreq=64)
        test = source[:3].copy()
        test[0, 0, :] *= 3.0
        diag = compute_qri(test, sfreq=64, reference=ref)
        local_trial_risk, local_rel = derive_qri_signal(diag, mode="local_only")
        source_trial_risk, source_rel = derive_qri_signal(diag, mode="source_only")
        self.assertEqual(local_trial_risk.shape, (3,))
        self.assertEqual(source_trial_risk.shape, (3,))
        self.assertEqual(local_rel.shape, (3, 4))
        self.assertEqual(source_rel.shape, (3, 4))
        self.assertGreater(source_trial_risk[0], source_trial_risk[1])


if __name__ == "__main__":
    unittest.main()
