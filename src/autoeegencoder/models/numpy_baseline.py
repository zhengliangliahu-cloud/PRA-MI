from __future__ import annotations

import numpy as np

from autoeegencoder.protocols.transforms import MethodArrays
from autoeegencoder.utils.metrics import balanced_accuracy


def _features(X: np.ndarray) -> np.ndarray:
    mean = np.mean(X, axis=2)
    std = np.std(X, axis=2)
    return np.concatenate([mean, std], axis=1)


def run_numpy_method(arrays: MethodArrays) -> dict[str, float]:
    """Dependency-light nearest-centroid smoke classifier."""

    train_f = _features(arrays.train_X)
    test_f = _features(arrays.test_X)
    centroids = {}
    for label in np.unique(arrays.train_y):
        centroids[int(label)] = train_f[arrays.train_y == label].mean(axis=0)
    labels = sorted(centroids)
    preds = []
    for row in test_f:
        distances = [np.linalg.norm(row - centroids[label]) for label in labels]
        preds.append(labels[int(np.argmin(distances))])
    y_pred = np.asarray(preds, dtype="int64")
    acc = float(np.mean(y_pred == arrays.test_y))
    bacc = balanced_accuracy(arrays.test_y, y_pred)
    return {"accuracy": acc, "balanced_accuracy": bacc}

