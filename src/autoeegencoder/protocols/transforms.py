from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from autoeegencoder.data.arrays import EEGArrays
from autoeegencoder.data.splits import LOSOSplit
from autoeegencoder.diagnostics.qri import (
    TrialDiagnostics,
    build_qri_reference,
    compute_qri,
    derive_qri_signal,
)

from .guards import ProtocolGuard


@dataclass(frozen=True)
class MethodArrays:
    train_X: np.ndarray
    train_y: np.ndarray
    train_r: np.ndarray
    val_X: np.ndarray
    val_y: np.ndarray
    val_r: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray
    test_r: np.ndarray
    test_diagnostics: TrialDiagnostics
    guard: ProtocolGuard


def _window_slice(window_s: list[float], sfreq: float, n_times: int) -> slice:
    start = max(0, int(round(float(window_s[0]) * sfreq)))
    stop = min(n_times, int(round(float(window_s[1]) * sfreq)))
    if stop <= start:
        raise ValueError(f"Invalid window {window_s} for sfreq={sfreq} n_times={n_times}")
    return slice(start, stop)


def _crop_window(X: np.ndarray, window: slice) -> np.ndarray:
    return np.asarray(X[:, :, window], dtype="float32")


def _source_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.mean(X, axis=(0, 2), keepdims=True)
    sigma = np.std(X, axis=(0, 2), keepdims=True)
    return mu, np.maximum(sigma, 1e-4)


def _robust_source_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.median(X, axis=(0, 2), keepdims=True)
    mad = np.median(np.abs(X - mu), axis=(0, 2), keepdims=True)
    sigma = 1.4826 * mad
    return mu, np.maximum(sigma, 1e-4)


def _apply_zscore(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return ((X - mu) / sigma).astype("float32")


def _baseline_reference(X: np.ndarray, baseline: slice) -> np.ndarray:
    return np.mean(X[:, :, baseline], axis=2, keepdims=True)


def _apply_baseline(X: np.ndarray, baseline: slice) -> np.ndarray:
    return (X - _baseline_reference(X, baseline)).astype("float32")


def _reliability_weighted_norm(
    X: np.ndarray,
    reliability: np.ndarray,
    trial_risk: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    mode: str = "minimal",
) -> np.ndarray:
    z = (X - mu) / sigma
    rel = reliability[:, :, None]
    if mode == "minimal":
        shrink = (X - mu) * 0.1 / sigma
        return (rel * z + (1.0 - rel) * shrink).astype("float32")
    effective_rel = np.clip(reliability * (1.0 - 0.50 * trial_risk[:, None]), 0.05, 1.0)[:, :, None]
    compressed = np.tanh(z / 2.5)
    return (effective_rel * z + (1.0 - effective_rel) * compressed).astype("float32")


def _component_or_zeros(diag: TrialDiagnostics, key: str) -> np.ndarray:
    value = diag.components.get(key)
    if value is None:
        return np.zeros_like(diag.trial_risk, dtype="float32")
    return value.astype("float32")


def _pack_qca_features(
    diag: TrialDiagnostics,
    *,
    channel_reliability: np.ndarray | None = None,
    trial_risk: np.ndarray | None = None,
    include_source: bool = True,
    include_local: bool = True,
) -> np.ndarray:
    if channel_reliability is None:
        channel_reliability = diag.channel_reliability
    if trial_risk is None:
        trial_risk = diag.trial_risk
    zeros = np.zeros_like(trial_risk, dtype="float32")
    global_features = np.column_stack(
        [
            1.0 - trial_risk,
            _component_or_zeros(diag, "qri_source_variance_deviation_risk") if include_source else zeros,
            _component_or_zeros(diag, "qri_source_mu_deviation_risk") if include_source else zeros,
            _component_or_zeros(diag, "qri_source_beta_deviation_risk") if include_source else zeros,
            _component_or_zeros(diag, "qri_source_high_frequency_deviation_risk") if include_source else zeros,
            _component_or_zeros(diag, "qri_local_spatial_deviation_risk") if include_local else zeros,
            _component_or_zeros(diag, "qri_local_burst_risk") if include_local else zeros,
        ]
    ).astype("float32")
    return np.concatenate([channel_reliability.astype("float32"), global_features], axis=1)


def _random_qca_features(shape: tuple[int, int], global_dim: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_trials, n_channels = shape
    trial_risk = rng.uniform(0.0, 1.0, size=n_trials).astype("float32")
    channel_reliability = rng.uniform(0.05, 1.0, size=(n_trials, n_channels)).astype("float32")
    global_features = rng.uniform(0.0, 1.0, size=(n_trials, global_dim)).astype("float32")
    return trial_risk, channel_reliability, np.concatenate([channel_reliability, global_features], axis=1)


def _generic_film_features(n_trials: int, n_channels: int, global_dim: int = 7) -> np.ndarray:
    return np.ones((n_trials, n_channels + global_dim), dtype="float32")


def _prepare_source_referenced_qri(
    train_raw: np.ndarray,
    val_raw: np.ndarray,
    test_raw: np.ndarray,
    baseline: slice,
    mi_window: slice,
    sfreq: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    TrialDiagnostics,
    TrialDiagnostics,
    TrialDiagnostics,
]:
    train_base = _crop_window(_apply_baseline(train_raw, baseline), mi_window)
    val_base = _crop_window(_apply_baseline(val_raw, baseline), mi_window)
    test_base = _crop_window(_apply_baseline(test_raw, baseline), mi_window)
    source_mu, source_sigma = _robust_source_stats(train_base)
    qri_reference = build_qri_reference(train_base, sfreq)
    train_diag = compute_qri(train_base, sfreq, reference=qri_reference)
    val_diag = compute_qri(val_base, sfreq, reference=qri_reference)
    test_diag = compute_qri(test_base, sfreq, reference=qri_reference)
    return train_base, val_base, test_base, source_mu, source_sigma, train_diag, val_diag, test_diag


def prepare_method_arrays(
    arrays: EEGArrays,
    split: LOSOSplit,
    cfg: dict,
    method: dict,
    method_seed: int | None = None,
) -> MethodArrays:
    ds_cfg = cfg["dataset"]
    preprocessing = method["preprocessing"]
    protocol_tier = method["protocol_tier"]
    guard = ProtocolGuard(protocol_tier)

    train_raw = arrays.X[split.train_idx]
    val_raw = arrays.X[split.val_idx]
    test_raw = arrays.X[split.test_idx]
    train_y = arrays.y[split.train_idx]
    val_y = arrays.y[split.val_idx]
    test_y = arrays.y[split.test_idx]
    baseline = _window_slice(ds_cfg["baseline_window_s"], arrays.sfreq, arrays.X.shape[-1])
    mi_window = _window_slice(ds_cfg["mi_window_s"], arrays.sfreq, arrays.X.shape[-1])

    train_mi_raw = _crop_window(train_raw, mi_window)
    val_mi_raw = _crop_window(val_raw, mi_window)
    test_mi_raw = _crop_window(test_raw, mi_window)

    train_diag = compute_qri(train_mi_raw, arrays.sfreq)
    val_diag = compute_qri(val_mi_raw, arrays.sfreq)
    test_diag = compute_qri(test_mi_raw, arrays.sfreq)
    ablation_rng = np.random.default_rng(0 if method_seed is None else int(method_seed))

    if preprocessing == "baseline_only":
        guard.record("trial_baseline_correction", "D0a", "trial pre-cue/rest baseline")
        train_X = _crop_window(_apply_baseline(train_raw, baseline), mi_window)
        val_X = _crop_window(_apply_baseline(val_raw, baseline), mi_window)
        test_X = _crop_window(_apply_baseline(test_raw, baseline), mi_window)
        train_r = train_diag.channel_reliability.astype("float32")
        val_r = val_diag.channel_reliability.astype("float32")
        test_r = test_diag.channel_reliability.astype("float32")
    elif preprocessing == "source_zscore":
        guard.record("source_zscore", "D0a", "source reference statistics only")
        source_mu, source_sigma = _source_stats(train_mi_raw)
        train_X = _apply_zscore(train_mi_raw, source_mu, source_sigma)
        val_X = _apply_zscore(val_mi_raw, source_mu, source_sigma)
        test_X = _apply_zscore(test_mi_raw, source_mu, source_sigma)
        train_r = train_diag.channel_reliability.astype("float32")
        val_r = val_diag.channel_reliability.astype("float32")
        test_r = test_diag.channel_reliability.astype("float32")
    elif preprocessing == "baseline_source_zscore":
        guard.record("trial_baseline_correction", "D0a", "trial pre-cue/rest baseline")
        train_base = _crop_window(_apply_baseline(train_raw, baseline), mi_window)
        val_base = _crop_window(_apply_baseline(val_raw, baseline), mi_window)
        test_base = _crop_window(_apply_baseline(test_raw, baseline), mi_window)
        source_mu, source_sigma = _source_stats(train_base)
        train_X = _apply_zscore(train_base, source_mu, source_sigma)
        val_X = _apply_zscore(val_base, source_mu, source_sigma)
        test_X = _apply_zscore(test_base, source_mu, source_sigma)
        train_r = train_diag.channel_reliability.astype("float32")
        val_r = val_diag.channel_reliability.astype("float32")
        test_r = test_diag.channel_reliability.astype("float32")
    elif preprocessing == "baseline_source_robust_zscore":
        guard.record("trial_baseline_correction", "D0a", "trial pre-cue/rest baseline")
        guard.record("source_robust_zscore", "D0a", "source median/MAD statistics on MI window")
        train_base = _crop_window(_apply_baseline(train_raw, baseline), mi_window)
        val_base = _crop_window(_apply_baseline(val_raw, baseline), mi_window)
        test_base = _crop_window(_apply_baseline(test_raw, baseline), mi_window)
        source_mu, source_sigma = _robust_source_stats(train_base)
        train_X = _apply_zscore(train_base, source_mu, source_sigma)
        val_X = _apply_zscore(val_base, source_mu, source_sigma)
        test_X = _apply_zscore(test_base, source_mu, source_sigma)
        train_r = train_diag.channel_reliability.astype("float32")
        val_r = val_diag.channel_reliability.astype("float32")
        test_r = test_diag.channel_reliability.astype("float32")
    elif preprocessing == "target_stat_zscore":
        guard.require_target_aggregate_allowed("target-subject unlabeled mean/std for P1")
        source_mu, source_sigma = _source_stats(train_mi_raw)
        target_mu, target_sigma = _source_stats(test_mi_raw)
        train_X = _apply_zscore(train_mi_raw, source_mu, source_sigma)
        val_X = _apply_zscore(val_mi_raw, source_mu, source_sigma)
        test_X = _apply_zscore(test_mi_raw, target_mu, target_sigma)
        train_r = train_diag.channel_reliability.astype("float32")
        val_r = val_diag.channel_reliability.astype("float32")
        test_r = test_diag.channel_reliability.astype("float32")
    elif preprocessing == "qca_reliability_norm":
        guard.record("trial_baseline_correction", "D0a", "trial pre-cue/rest baseline")
        guard.record("qri_reliability_weighted_norm", "D0b", "single-trial frozen QRI reliability on MI window")
        train_base = _crop_window(_apply_baseline(train_raw, baseline), mi_window)
        val_base = _crop_window(_apply_baseline(val_raw, baseline), mi_window)
        test_base = _crop_window(_apply_baseline(test_raw, baseline), mi_window)
        source_mu, source_sigma = _source_stats(train_base)
        train_X = _reliability_weighted_norm(
            train_base,
            train_diag.channel_reliability,
            train_diag.trial_risk,
            source_mu,
            source_sigma,
            mode="minimal",
        )
        val_X = _reliability_weighted_norm(
            val_base,
            val_diag.channel_reliability,
            val_diag.trial_risk,
            source_mu,
            source_sigma,
            mode="minimal",
        )
        test_X = _reliability_weighted_norm(
            test_base,
            test_diag.channel_reliability,
            test_diag.trial_risk,
            source_mu,
            source_sigma,
            mode="minimal",
        )
        train_r = train_diag.channel_reliability.astype("float32")
        val_r = val_diag.channel_reliability.astype("float32")
        test_r = test_diag.channel_reliability.astype("float32")
    elif preprocessing == "qca_reliability_robust_norm":
        guard.record("trial_baseline_correction", "D0a", "trial pre-cue/rest baseline")
        guard.record("source_robust_zscore", "D0a", "source median/MAD statistics on MI window")
        guard.record("qri_source_reference_profile", "D0a", "source-only QRI reference on baseline-corrected MI window")
        guard.record("qri_reliability_weighted_robust_norm", "D0a", "source-referenced P0-safe QRI with robust normalization")
        train_base = _crop_window(_apply_baseline(train_raw, baseline), mi_window)
        val_base = _crop_window(_apply_baseline(val_raw, baseline), mi_window)
        test_base = _crop_window(_apply_baseline(test_raw, baseline), mi_window)
        source_mu, source_sigma = _robust_source_stats(train_base)
        qri_reference = build_qri_reference(train_base, arrays.sfreq)
        train_diag = compute_qri(train_base, arrays.sfreq, reference=qri_reference)
        val_diag = compute_qri(val_base, arrays.sfreq, reference=qri_reference)
        test_diag = compute_qri(test_base, arrays.sfreq, reference=qri_reference)
        train_X = _reliability_weighted_norm(
            train_base,
            train_diag.channel_reliability,
            train_diag.trial_risk,
            source_mu,
            source_sigma,
            mode="enhanced",
        )
        val_X = _reliability_weighted_norm(
            val_base,
            val_diag.channel_reliability,
            val_diag.trial_risk,
            source_mu,
            source_sigma,
            mode="enhanced",
        )
        test_X = _reliability_weighted_norm(
            test_base,
            test_diag.channel_reliability,
            test_diag.trial_risk,
            source_mu,
            source_sigma,
            mode="enhanced",
        )
        train_r = _pack_qca_features(train_diag)
        val_r = _pack_qca_features(val_diag)
        test_r = _pack_qca_features(test_diag)
    elif preprocessing == "qca_local_only_robust_norm":
        guard.record("trial_baseline_correction", "D0a", "trial pre-cue/rest baseline")
        guard.record("source_robust_zscore", "D0a", "source median/MAD statistics on MI window")
        guard.record("qri_local_only_profile", "D0b", "single-trial local QRI ablation on baseline-corrected MI window")
        train_base = _crop_window(_apply_baseline(train_raw, baseline), mi_window)
        val_base = _crop_window(_apply_baseline(val_raw, baseline), mi_window)
        test_base = _crop_window(_apply_baseline(test_raw, baseline), mi_window)
        source_mu, source_sigma = _robust_source_stats(train_base)
        train_diag = compute_qri(train_base, arrays.sfreq)
        val_diag = compute_qri(val_base, arrays.sfreq)
        test_diag = compute_qri(test_base, arrays.sfreq)
        train_trial_risk, train_channel_rel = derive_qri_signal(train_diag, mode="local_only")
        val_trial_risk, val_channel_rel = derive_qri_signal(val_diag, mode="local_only")
        test_trial_risk, test_channel_rel = derive_qri_signal(test_diag, mode="local_only")
        train_X = _reliability_weighted_norm(train_base, train_channel_rel, train_trial_risk, source_mu, source_sigma, mode="enhanced")
        val_X = _reliability_weighted_norm(val_base, val_channel_rel, val_trial_risk, source_mu, source_sigma, mode="enhanced")
        test_X = _reliability_weighted_norm(test_base, test_channel_rel, test_trial_risk, source_mu, source_sigma, mode="enhanced")
        train_r = _pack_qca_features(train_diag, channel_reliability=train_channel_rel, trial_risk=train_trial_risk, include_source=False, include_local=True)
        val_r = _pack_qca_features(val_diag, channel_reliability=val_channel_rel, trial_risk=val_trial_risk, include_source=False, include_local=True)
        test_r = _pack_qca_features(test_diag, channel_reliability=test_channel_rel, trial_risk=test_trial_risk, include_source=False, include_local=True)
    elif preprocessing == "qca_source_only_robust_norm":
        guard.record("trial_baseline_correction", "D0a", "trial pre-cue/rest baseline")
        guard.record("source_robust_zscore", "D0a", "source median/MAD statistics on MI window")
        guard.record("qri_source_reference_profile", "D0a", "source-only QRI reference on baseline-corrected MI window")
        guard.record("qri_source_only_ablation", "D0a", "source-referenced-only QRI ablation without local trial cues")
        train_base = _crop_window(_apply_baseline(train_raw, baseline), mi_window)
        val_base = _crop_window(_apply_baseline(val_raw, baseline), mi_window)
        test_base = _crop_window(_apply_baseline(test_raw, baseline), mi_window)
        source_mu, source_sigma = _robust_source_stats(train_base)
        qri_reference = build_qri_reference(train_base, arrays.sfreq)
        train_diag = compute_qri(train_base, arrays.sfreq, reference=qri_reference)
        val_diag = compute_qri(val_base, arrays.sfreq, reference=qri_reference)
        test_diag = compute_qri(test_base, arrays.sfreq, reference=qri_reference)
        train_trial_risk, train_channel_rel = derive_qri_signal(train_diag, mode="source_only")
        val_trial_risk, val_channel_rel = derive_qri_signal(val_diag, mode="source_only")
        test_trial_risk, test_channel_rel = derive_qri_signal(test_diag, mode="source_only")
        train_X = _reliability_weighted_norm(train_base, train_channel_rel, train_trial_risk, source_mu, source_sigma, mode="enhanced")
        val_X = _reliability_weighted_norm(val_base, val_channel_rel, val_trial_risk, source_mu, source_sigma, mode="enhanced")
        test_X = _reliability_weighted_norm(test_base, test_channel_rel, test_trial_risk, source_mu, source_sigma, mode="enhanced")
        train_r = _pack_qca_features(train_diag, channel_reliability=train_channel_rel, trial_risk=train_trial_risk, include_source=True, include_local=False)
        val_r = _pack_qca_features(val_diag, channel_reliability=val_channel_rel, trial_risk=val_trial_risk, include_source=True, include_local=False)
        test_r = _pack_qca_features(test_diag, channel_reliability=test_channel_rel, trial_risk=test_trial_risk, include_source=True, include_local=False)
    elif preprocessing == "qca_random_reliability_robust_norm":
        guard.record("trial_baseline_correction", "D0a", "trial pre-cue/rest baseline")
        guard.record("source_robust_zscore", "D0a", "source median/MAD statistics on MI window")
        guard.record("qri_random_reliability_ablation", "D0a", "random reliability control on robust normalized MI window")
        train_base = _crop_window(_apply_baseline(train_raw, baseline), mi_window)
        val_base = _crop_window(_apply_baseline(val_raw, baseline), mi_window)
        test_base = _crop_window(_apply_baseline(test_raw, baseline), mi_window)
        source_mu, source_sigma = _robust_source_stats(train_base)
        qri_reference = build_qri_reference(train_base, arrays.sfreq)
        train_diag = compute_qri(train_base, arrays.sfreq, reference=qri_reference)
        val_diag = compute_qri(val_base, arrays.sfreq, reference=qri_reference)
        test_diag = compute_qri(test_base, arrays.sfreq, reference=qri_reference)
        train_trial_risk, train_channel_rel, train_r = _random_qca_features(train_diag.channel_reliability.shape, 7, ablation_rng)
        val_trial_risk, val_channel_rel, val_r = _random_qca_features(val_diag.channel_reliability.shape, 7, ablation_rng)
        test_trial_risk, test_channel_rel, test_r = _random_qca_features(test_diag.channel_reliability.shape, 7, ablation_rng)
        train_X = _reliability_weighted_norm(train_base, train_channel_rel, train_trial_risk, source_mu, source_sigma, mode="enhanced")
        val_X = _reliability_weighted_norm(val_base, val_channel_rel, val_trial_risk, source_mu, source_sigma, mode="enhanced")
        test_X = _reliability_weighted_norm(test_base, test_channel_rel, test_trial_risk, source_mu, source_sigma, mode="enhanced")
    elif preprocessing == "qca_shuffled_reliability_robust_norm":
        guard.record("trial_baseline_correction", "D0a", "trial pre-cue/rest baseline")
        guard.record("source_robust_zscore", "D0a", "source median/MAD statistics on MI window")
        guard.record("qri_source_reference_profile", "D0a", "source-only QRI reference on baseline-corrected MI window")
        guard.record("qri_shuffled_reliability_ablation", "D0a", "trial-shuffled reliability control on robust normalized MI window")
        train_base = _crop_window(_apply_baseline(train_raw, baseline), mi_window)
        val_base = _crop_window(_apply_baseline(val_raw, baseline), mi_window)
        test_base = _crop_window(_apply_baseline(test_raw, baseline), mi_window)
        source_mu, source_sigma = _robust_source_stats(train_base)
        qri_reference = build_qri_reference(train_base, arrays.sfreq)
        train_diag = compute_qri(train_base, arrays.sfreq, reference=qri_reference)
        val_diag = compute_qri(val_base, arrays.sfreq, reference=qri_reference)
        test_diag = compute_qri(test_base, arrays.sfreq, reference=qri_reference)
        train_perm = ablation_rng.permutation(train_diag.channel_reliability.shape[0])
        val_perm = ablation_rng.permutation(val_diag.channel_reliability.shape[0])
        test_perm = ablation_rng.permutation(test_diag.channel_reliability.shape[0])
        train_trial_risk = train_diag.trial_risk[train_perm]
        val_trial_risk = val_diag.trial_risk[val_perm]
        test_trial_risk = test_diag.trial_risk[test_perm]
        train_channel_rel = train_diag.channel_reliability[train_perm]
        val_channel_rel = val_diag.channel_reliability[val_perm]
        test_channel_rel = test_diag.channel_reliability[test_perm]
        train_X = _reliability_weighted_norm(train_base, train_channel_rel, train_trial_risk, source_mu, source_sigma, mode="enhanced")
        val_X = _reliability_weighted_norm(val_base, val_channel_rel, val_trial_risk, source_mu, source_sigma, mode="enhanced")
        test_X = _reliability_weighted_norm(test_base, test_channel_rel, test_trial_risk, source_mu, source_sigma, mode="enhanced")
        train_r = _pack_qca_features(train_diag)[train_perm]
        val_r = _pack_qca_features(val_diag)[val_perm]
        test_r = _pack_qca_features(test_diag)[test_perm]
    elif preprocessing == "qca_adapter_only_robust_zscore":
        guard.record("trial_baseline_correction", "D0a", "trial pre-cue/rest baseline")
        guard.record("source_robust_zscore", "D0a", "source median/MAD statistics on MI window")
        guard.record("qri_source_reference_profile", "D0a", "source-only QRI reference used by adapter only")
        (
            train_base,
            val_base,
            test_base,
            source_mu,
            source_sigma,
            train_diag,
            val_diag,
            test_diag,
        ) = _prepare_source_referenced_qri(train_raw, val_raw, test_raw, baseline, mi_window, arrays.sfreq)
        train_X = _apply_zscore(train_base, source_mu, source_sigma)
        val_X = _apply_zscore(val_base, source_mu, source_sigma)
        test_X = _apply_zscore(test_base, source_mu, source_sigma)
        train_r = _pack_qca_features(train_diag)
        val_r = _pack_qca_features(val_diag)
        test_r = _pack_qca_features(test_diag)
    elif preprocessing == "qca_norm_only_robust_norm":
        guard.record("trial_baseline_correction", "D0a", "trial pre-cue/rest baseline")
        guard.record("source_robust_zscore", "D0a", "source median/MAD statistics on MI window")
        guard.record("qri_norm_only_ablation", "D0a", "source-referenced QRI affects normalization only")
        (
            train_base,
            val_base,
            test_base,
            source_mu,
            source_sigma,
            train_diag,
            val_diag,
            test_diag,
        ) = _prepare_source_referenced_qri(train_raw, val_raw, test_raw, baseline, mi_window, arrays.sfreq)
        train_X = _reliability_weighted_norm(
            train_base, train_diag.channel_reliability, train_diag.trial_risk, source_mu, source_sigma, mode="enhanced"
        )
        val_X = _reliability_weighted_norm(
            val_base, val_diag.channel_reliability, val_diag.trial_risk, source_mu, source_sigma, mode="enhanced"
        )
        test_X = _reliability_weighted_norm(
            test_base, test_diag.channel_reliability, test_diag.trial_risk, source_mu, source_sigma, mode="enhanced"
        )
        train_r = train_diag.channel_reliability.astype("float32")
        val_r = val_diag.channel_reliability.astype("float32")
        test_r = test_diag.channel_reliability.astype("float32")
    elif preprocessing == "qca_shuffle_adapter_only_robust_norm":
        guard.record("trial_baseline_correction", "D0a", "trial pre-cue/rest baseline")
        guard.record("source_robust_zscore", "D0a", "source median/MAD statistics on MI window")
        guard.record("qri_shuffle_adapter_only_ablation", "D0a", "structured QRI normalization with shuffled adapter input")
        (
            train_base,
            val_base,
            test_base,
            source_mu,
            source_sigma,
            train_diag,
            val_diag,
            test_diag,
        ) = _prepare_source_referenced_qri(train_raw, val_raw, test_raw, baseline, mi_window, arrays.sfreq)
        train_X = _reliability_weighted_norm(
            train_base, train_diag.channel_reliability, train_diag.trial_risk, source_mu, source_sigma, mode="enhanced"
        )
        val_X = _reliability_weighted_norm(
            val_base, val_diag.channel_reliability, val_diag.trial_risk, source_mu, source_sigma, mode="enhanced"
        )
        test_X = _reliability_weighted_norm(
            test_base, test_diag.channel_reliability, test_diag.trial_risk, source_mu, source_sigma, mode="enhanced"
        )
        train_r = _pack_qca_features(train_diag)[ablation_rng.permutation(train_diag.channel_reliability.shape[0])]
        val_r = _pack_qca_features(val_diag)[ablation_rng.permutation(val_diag.channel_reliability.shape[0])]
        test_r = _pack_qca_features(test_diag)[ablation_rng.permutation(test_diag.channel_reliability.shape[0])]
    elif preprocessing == "qca_shuffle_norm_only_robust_norm":
        guard.record("trial_baseline_correction", "D0a", "trial pre-cue/rest baseline")
        guard.record("source_robust_zscore", "D0a", "source median/MAD statistics on MI window")
        guard.record("qri_shuffle_norm_only_ablation", "D0a", "shuffled QRI normalization with structured adapter input")
        (
            train_base,
            val_base,
            test_base,
            source_mu,
            source_sigma,
            train_diag,
            val_diag,
            test_diag,
        ) = _prepare_source_referenced_qri(train_raw, val_raw, test_raw, baseline, mi_window, arrays.sfreq)
        train_norm_perm = ablation_rng.permutation(train_diag.channel_reliability.shape[0])
        val_norm_perm = ablation_rng.permutation(val_diag.channel_reliability.shape[0])
        test_norm_perm = ablation_rng.permutation(test_diag.channel_reliability.shape[0])
        train_X = _reliability_weighted_norm(
            train_base,
            train_diag.channel_reliability[train_norm_perm],
            train_diag.trial_risk[train_norm_perm],
            source_mu,
            source_sigma,
            mode="enhanced",
        )
        val_X = _reliability_weighted_norm(
            val_base,
            val_diag.channel_reliability[val_norm_perm],
            val_diag.trial_risk[val_norm_perm],
            source_mu,
            source_sigma,
            mode="enhanced",
        )
        test_X = _reliability_weighted_norm(
            test_base,
            test_diag.channel_reliability[test_norm_perm],
            test_diag.trial_risk[test_norm_perm],
            source_mu,
            source_sigma,
            mode="enhanced",
        )
        train_r = _pack_qca_features(train_diag)
        val_r = _pack_qca_features(val_diag)
        test_r = _pack_qca_features(test_diag)
    elif preprocessing == "generic_film_robust_zscore":
        guard.record("trial_baseline_correction", "D0a", "trial pre-cue/rest baseline")
        guard.record("source_robust_zscore", "D0a", "source median/MAD statistics on MI window")
        guard.record("generic_film_adapter", "D0a", "constant same-capacity adapter control without reliability semantics")
        (
            train_base,
            val_base,
            test_base,
            source_mu,
            source_sigma,
            train_diag,
            val_diag,
            test_diag,
        ) = _prepare_source_referenced_qri(train_raw, val_raw, test_raw, baseline, mi_window, arrays.sfreq)
        train_X = _apply_zscore(train_base, source_mu, source_sigma)
        val_X = _apply_zscore(val_base, source_mu, source_sigma)
        test_X = _apply_zscore(test_base, source_mu, source_sigma)
        train_r = _generic_film_features(train_X.shape[0], train_X.shape[1])
        val_r = _generic_film_features(val_X.shape[0], val_X.shape[1])
        test_r = _generic_film_features(test_X.shape[0], test_X.shape[1])
    else:
        raise NotImplementedError(f"Unknown preprocessing: {preprocessing}")

    return MethodArrays(
        train_X=train_X,
        train_y=train_y,
        train_r=train_r,
        val_X=val_X,
        val_y=val_y,
        val_r=val_r,
        test_X=test_X,
        test_y=test_y,
        test_r=test_r,
        test_diagnostics=test_diag,
        guard=guard,
    )
