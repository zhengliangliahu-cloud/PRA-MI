from __future__ import annotations

from .arrays import EEGArrays
from .moabb_bnci import load_bnci2014_001
from .moabb_cho import load_cho2017
from .moabb_lee import load_lee2019_mi
from .synthetic import make_synthetic_dataset


def load_dataset(cfg: dict) -> EEGArrays:
    backend = cfg["dataset"].get("backend")
    subjects = [int(s) for s in cfg["dataset"].get("subjects", [])]
    if backend == "synthetic":
        return make_synthetic_dataset(cfg, int(cfg.get("seed", 2026)))
    if backend == "moabb" and cfg["dataset"].get("moabb_id") == "BNCI2014_001":
        return load_bnci2014_001(cfg, subjects)
    if backend == "moabb" and cfg["dataset"].get("moabb_id") == "Cho2017":
        return load_cho2017(cfg, subjects)
    if backend == "moabb" and cfg["dataset"].get("moabb_id") == "Lee2019_MI":
        return load_lee2019_mi(cfg, subjects)
    raise NotImplementedError(f"Unsupported dataset backend: {backend}")
