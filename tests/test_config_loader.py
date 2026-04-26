import unittest

from autoeegencoder.utils.config import load_config


class ConfigLoaderTests(unittest.TestCase):
    def test_default_config_loads_all_required_groups(self):
        cfg = load_config([])
        for key in ["dataset", "protocol", "model", "trainer", "experiment"]:
            self.assertIn(key, cfg)
        self.assertEqual(cfg["dataset"]["name"], "bnci2014_001")
        self.assertEqual(cfg["experiment"]["name"], "bnci_pra_mi_smoke")
        self.assertEqual(cfg["eval_subjects"], [1, 2])
        self.assertEqual(cfg["dataset"]["subjects"], [1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_legacy_subject_override_maps_to_eval_subjects(self):
        cfg = load_config(["experiment=bnci_pra_mi_smoke", "subjects=1,2"])
        self.assertEqual(cfg["dataset"]["name"], "bnci2014_001")
        self.assertEqual(cfg["eval_subjects"], [1, 2])
        self.assertEqual(cfg["subjects"], [1, 2])
        self.assertEqual(cfg["dataset"]["subjects"], [1, 2, 3, 4, 5, 6, 7, 8, 9])


if __name__ == "__main__":
    unittest.main()
