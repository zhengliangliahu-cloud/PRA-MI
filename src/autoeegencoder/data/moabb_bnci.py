from __future__ import annotations

from pathlib import Path

import numpy as np

from .arrays import EEGArrays


def load_bnci2014_001(cfg: dict, subjects: list[int]) -> EEGArrays:
    """Load BNCI2014_001 through MOABB.

    This function imports MOABB/MNE lazily so the rest of the package can be
    unit-tested without the full EEG stack installed.
    """

    try:
        from moabb.datasets import BNCI2014_001
        from moabb.paradigms import MotorImagery
        from moabb.utils import set_download_dir
    except ImportError as exc:
        raise RuntimeError(
            "MOABB/MNE are required for BNCI2014_001. Install requirements.txt "
            "or use dataset=synthetic for a no-download smoke test."
        ) from exc

    ds_cfg = cfg["dataset"]
    cache_dir = Path(cfg.get("data_root") or ds_cfg.get("cache_dir", "datasets/moabb"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    set_download_dir(str(cache_dir))

    dataset = BNCI2014_001()
    paradigm = MotorImagery(
        n_classes=2,
        events=["left_hand", "right_hand"],
        fmin=float(ds_cfg.get("fmin", 4.0)),
        fmax=float(ds_cfg.get("fmax", 45.0)),
        tmin=0.0,
        tmax=float(ds_cfg["mi_window_s"][1]),
        resample=float(ds_cfg.get("sfreq", 128)),
    )
    X, labels, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)
    label_map = {"left_hand": 0, "right_hand": 1}
    y = np.asarray([label_map[str(label)] for label in labels], dtype="int64")
    subject_ids = metadata["subject"].to_numpy(dtype="int64")
    ch_names = list(getattr(paradigm, "channels", []) or [f"Ch{i + 1}" for i in range(X.shape[1])])

    arrays = EEGArrays(
        X=np.asarray(X, dtype="float32"),
        y=y,
        subjects=subject_ids,
        sfreq=float(ds_cfg.get("sfreq", 128)),
        ch_names=ch_names,
        metadata={
            "backend": "moabb",
            "dataset": "BNCI2014_001",
            "notes": ds_cfg.get("harmonization_notes", []),
            "moabb_subjects_loaded": sorted(set(subject_ids.tolist())),
        },
    )
    arrays.validate()
    return arrays
