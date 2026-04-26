from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LOSOSplit:
    test_subject: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_subjects: np.ndarray
    val_subjects: np.ndarray
    val_strategy: str


def make_loso_split(
    subjects: np.ndarray,
    test_subject: int,
    *,
    val_subjects: int = 1,
    val_fraction: float = 0.2,
    seed: int = 2026,
) -> LOSOSplit:
    """Create a subject-independent LOSO split.

    Validation trials are sampled only from source subjects, never from the
    held-out target subject.
    """

    subjects = np.asarray(subjects)
    target_mask = subjects == test_subject
    if not np.any(target_mask):
        raise ValueError(f"test_subject={test_subject} is not present in subjects")

    source_mask = ~target_mask
    source_idx = np.flatnonzero(source_mask)
    test_idx = np.flatnonzero(target_mask)
    if source_idx.size == 0:
        raise ValueError("LOSO split requires at least one source trial")

    rng = np.random.default_rng(seed + int(test_subject))
    unique_source_subjects = np.unique(subjects[source_mask]).astype("int64")
    if unique_source_subjects.size >= 2:
        n_val_subjects = min(max(1, int(val_subjects)), unique_source_subjects.size - 1)
        val_subject_ids = np.sort(rng.choice(unique_source_subjects, size=n_val_subjects, replace=False))
        val_mask = np.isin(subjects, val_subject_ids) & source_mask
        val_idx = np.flatnonzero(val_mask)
        train_idx = np.flatnonzero(source_mask & ~val_mask)
        train_subject_ids = np.unique(subjects[train_idx]).astype("int64")
        strategy = "source_subject_holdout"
    else:
        shuffled = source_idx.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_fraction)))
        n_val = min(n_val, len(shuffled) - 1)
        val_idx = np.sort(shuffled[:n_val])
        train_idx = np.sort(shuffled[n_val:])
        val_subject_ids = np.unique(subjects[val_idx]).astype("int64")
        train_subject_ids = np.unique(subjects[train_idx]).astype("int64")
        strategy = "source_trial_fallback"

    if np.intersect1d(train_idx, test_idx).size or np.intersect1d(val_idx, test_idx).size:
        raise AssertionError("held-out target subject leaked into source split")

    return LOSOSplit(
        test_subject=int(test_subject),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=np.sort(test_idx),
        train_subjects=train_subject_ids,
        val_subjects=val_subject_ids,
        val_strategy=strategy,
    )
