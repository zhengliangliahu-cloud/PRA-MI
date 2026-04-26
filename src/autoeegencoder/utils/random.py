from __future__ import annotations

import hashlib
import random

import numpy as np


def derive_run_seed(base_seed: int, test_subject: int, method_name: str, repeat_index: int = 0) -> int:
    payload = f"{int(base_seed)}::{int(test_subject)}::{method_name}::{int(repeat_index)}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
