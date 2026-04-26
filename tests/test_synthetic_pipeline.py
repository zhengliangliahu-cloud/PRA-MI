import unittest

from autoeegencoder.data.splits import make_loso_split
from autoeegencoder.data.synthetic import make_synthetic_dataset
from autoeegencoder.models.numpy_baseline import run_numpy_method
from autoeegencoder.protocols.transforms import prepare_method_arrays


class SyntheticPipelineTests(unittest.TestCase):
    def test_synthetic_numpy_pipeline_runs(self):
        cfg = {
            "seed": 2026,
            "dataset": {
                "name": "synthetic",
                "backend": "synthetic",
                "n_subjects": 3,
                "n_trials_per_subject": 10,
                "n_channels": 4,
                "n_times": 128,
                "sfreq": 32,
                "baseline_window_s": [0.0, 1.0],
                "mi_window_s": [1.0, 4.0],
            },
        }
        arrays = make_synthetic_dataset(cfg, seed=2026)
        split = make_loso_split(arrays.subjects, 1, seed=2026)
        method = {
            "name": "p0_source",
            "protocol_tier": "P0",
            "preprocessing": "source_zscore",
            "adapter": "none",
        }
        prepared = prepare_method_arrays(arrays, split, cfg, method)
        metrics = run_numpy_method(prepared)
        self.assertIn("balanced_accuracy", metrics)
        self.assertGreaterEqual(metrics["balanced_accuracy"], 0.0)
        self.assertLessEqual(metrics["balanced_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()

