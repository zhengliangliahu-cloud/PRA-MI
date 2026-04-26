from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EEGArrays:
    """In-memory EEG array bundle.

    X is expected to be shaped ``(n_trials, n_channels, n_times)``.
    y is binary for the first PRA-MI pilot: 0 = left hand, 1 = right hand.
    """

    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    sfreq: float
    ch_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.X.ndim != 3:
            raise ValueError(f"X must be 3D (trials, channels, times), got {self.X.shape}")
        n_trials = self.X.shape[0]
        if len(self.y) != n_trials or len(self.subjects) != n_trials:
            raise ValueError("X, y, and subjects must have the same first dimension")
        if len(self.ch_names) != self.X.shape[1]:
            raise ValueError("ch_names length must match X channel dimension")

