import unittest

from autoeegencoder.utils.random import derive_run_seed


class RandomnessTests(unittest.TestCase):
    def test_derive_run_seed_is_stable_and_order_independent(self):
        seed_a = derive_run_seed(2026, 3, "qca_enhanced")
        seed_b = derive_run_seed(2026, 3, "qca_enhanced")
        seed_c = derive_run_seed(2026, 3, "p0_baseline_robust")
        self.assertEqual(seed_a, seed_b)
        self.assertNotEqual(seed_a, seed_c)


if __name__ == "__main__":
    unittest.main()
