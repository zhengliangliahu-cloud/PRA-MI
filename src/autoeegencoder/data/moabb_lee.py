from __future__ import annotations

from pathlib import Path

import numpy as np

from .arrays import EEGArrays


def load_lee2019_mi(cfg: dict, subjects: list[int]) -> EEGArrays:
    try:
        from moabb.datasets import Lee2019_MI
        from moabb.paradigms import LeftRightImagery
        from moabb.utils import set_download_dir
    except ImportError as exc:
        raise RuntimeError(
            "MOABB/MNE are required for Lee2019_MI. Install the AEC environment first."
        ) from exc

    ds_cfg = cfg["dataset"]
    cache_dir = Path(cfg.get("data_root") or ds_cfg.get("cache_dir", "datasets/moabb"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    set_download_dir(str(cache_dir))

    dataset = Lee2019_MI()
    paradigm = LeftRightImagery(
        fmin=float(ds_cfg.get("fmin", 8.0)),
        fmax=float(ds_cfg.get("fmax", 32.0)),
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
            "dataset": "Lee2019_MI",
            "notes": ds_cfg.get("harmonization_notes", []),
            "moabb_subjects_loaded": sorted(set(subject_ids.tolist())),
        },
    )
    arrays.validate()
    return arrays
