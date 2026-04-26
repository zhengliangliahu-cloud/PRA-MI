from __future__ import annotations

import numpy as np


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(y_true)
    recalls = []
    for label in labels:
        mask = y_true == label
        if np.any(mask):
            recalls.append(float(np.mean(y_pred[mask] == label)))
    return float(np.mean(recalls)) if recalls else 0.0

