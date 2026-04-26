import unittest

import numpy as np

from autoeegencoder.data.splits import make_loso_split


class LOSOSplitTests(unittest.TestCase):
    def test_target_subject_not_in_source_splits(self):
        subjects = np.array([1, 1, 2, 2, 3, 3, 3])
        split = make_loso_split(subjects, 2, seed=7)
        self.assertTrue(np.all(subjects[split.test_idx] == 2))
        self.assertFalse(np.any(subjects[split.train_idx] == 2))
        self.assertFalse(np.any(subjects[split.val_idx] == 2))
        self.assertEqual(split.val_strategy, "source_subject_holdout")
        self.assertTrue(np.array_equal(np.unique(subjects[split.val_idx]), split.val_subjects))
        self.assertEqual(len(np.intersect1d(split.train_subjects, split.val_subjects)), 0)


if __name__ == "__main__":
    unittest.main()
