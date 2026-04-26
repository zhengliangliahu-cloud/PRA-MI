from __future__ import annotations

import numpy as np

from .arrays import EEGArrays


def make_synthetic_dataset(cfg: dict, seed: int) -> EEGArrays:
    """Generate a deterministic EEG-like binary MI toy dataset.

    The generator deliberately introduces subject-specific amplitude/style
    shifts and a few low-quality trials so protocol and diagnostic tests have
    non-trivial signals to audit.
    """

    ds = cfg["dataset"]
    rng = np.random.default_rng(seed)
    n_subjects = int(ds.get("n_subjects", 4))
    n_trials = int(ds.get("n_trials_per_subject", 32))
    n_channels = int(ds.get("n_channels", 8))
    n_times = int(ds.get("n_times", 256))
    sfreq = float(ds.get("sfreq", 64))

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    s_parts: list[np.ndarray] = []
    t = np.arange(n_times) / sfreq
    rhythm = np.sin(2 * np.pi * 10.0 * t)
    mi_start = int(ds["mi_window_s"][0] * sfreq)
    mi_stop = min(n_times, int(ds["mi_window_s"][1] * sfreq))

    for subject in range(1, n_subjects + 1):
        style = 0.8 + 0.15 * subject
        subject_noise = 0.15 + 0.03 * subject
        X = rng.normal(0.0, subject_noise, size=(n_trials, n_channels, n_times))
        y = np.arange(n_trials) % 2
        rng.shuffle(y)
        for i, label in enumerate(y):
            lateral = -1.0 if label == 0 else 1.0
            X[i, 0, mi_start:mi_stop] += lateral * 0.7 * rhythm[mi_start:mi_stop]
            X[i, 1, mi_start:mi_stop] -= lateral * 0.5 * rhythm[mi_start:mi_stop]
            X[i] *= style
            if i % 17 == 0:
                X[i, -1, :] = 0.0
            if i % 19 == 0:
                X[i, :, :] += rng.normal(0, 2.5, size=(n_channels, n_times))
        X_parts.append(X.astype("float32"))
        y_parts.append(y.astype("int64"))
        s_parts.append(np.full(n_trials, subject, dtype="int64"))

    arrays = EEGArrays(
        X=np.concatenate(X_parts, axis=0),
        y=np.concatenate(y_parts, axis=0),
        subjects=np.concatenate(s_parts, axis=0),
        sfreq=sfreq,
        ch_names=[f"Ch{i + 1}" for i in range(n_channels)],
        metadata={
            "backend": "synthetic",
            "notes": ds.get("harmonization_notes", []),
        },
    )
    arrays.validate()
    return arrays

