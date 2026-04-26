import unittest

from autoeegencoder.data.splits import make_loso_split
from autoeegencoder.data.synthetic import make_synthetic_dataset
from autoeegencoder.protocols.transforms import prepare_method_arrays


class TransformTests(unittest.TestCase):
    def test_prepare_method_arrays_uses_mi_window_for_classification(self):
        cfg = {
            "seed": 2026,
            "dataset": {
                "name": "synthetic",
                "backend": "synthetic",
                "n_subjects": 4,
                "subjects": [1, 2, 3, 4],
                "n_trials_per_subject": 8,
                "n_channels": 4,
                "n_times": 256,
                "sfreq": 64,
                "baseline_window_s": [0.0, 1.0],
                "mi_window_s": [1.0, 4.0],
            },
        }
        arrays = make_synthetic_dataset(cfg, seed=2026)
        split = make_loso_split(arrays.subjects, 1, seed=2026)
        prepared = prepare_method_arrays(
            arrays,
            split,
            cfg,
            {
                "name": "p0_baseline_norm",
                "protocol_tier": "P0",
                "preprocessing": "baseline_source_zscore",
                "adapter": "none",
            },
        )
        self.assertEqual(prepared.train_X.shape[1], 4)
        self.assertEqual(prepared.train_X.shape[2], 192)
        self.assertEqual(prepared.val_X.shape[2], 192)
        self.assertEqual(prepared.test_X.shape[2], 192)

    def test_qca_ablation_features_are_deterministic(self):
        cfg = {
            "seed": 2026,
            "dataset": {
                "name": "synthetic",
                "backend": "synthetic",
                "n_subjects": 4,
                "subjects": [1, 2, 3, 4],
                "n_trials_per_subject": 8,
                "n_channels": 4,
                "n_times": 256,
                "sfreq": 64,
                "baseline_window_s": [0.0, 1.0],
                "mi_window_s": [1.0, 4.0],
            },
        }
        arrays = make_synthetic_dataset(cfg, seed=2026)
        split = make_loso_split(arrays.subjects, 1, seed=2026)
        method = {
            "name": "qca_random_reliability",
            "protocol_tier": "P0",
            "preprocessing": "qca_random_reliability_robust_norm",
            "adapter": "qca_random_reliability",
        }
        prepared_a = prepare_method_arrays(arrays, split, cfg, method, method_seed=77)
        prepared_b = prepare_method_arrays(arrays, split, cfg, method, method_seed=77)
        self.assertEqual(prepared_a.train_r.shape[1], arrays.X.shape[1] + 7)
        self.assertTrue((prepared_a.train_r == prepared_b.train_r).all())
        self.assertTrue((prepared_a.test_X == prepared_b.test_X).all())


if __name__ == "__main__":
    unittest.main()
