from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LOCAL_CHANNEL_COMPONENTS = (
    "qri_local_spatial_deviation_risk",
    "qri_local_amplitude_deviation_risk",
    "qri_local_burst_risk",
    "qri_local_high_frequency_spread_risk",
    "qri_local_flatline_risk",
)
SOURCE_CHANNEL_COMPONENTS = (
    "qri_source_variance_deviation_risk",
    "qri_source_mu_deviation_risk",
    "qri_source_beta_deviation_risk",
    "qri_source_high_frequency_deviation_risk",
)


@dataclass(frozen=True)
class QRIReference:
    log_var_center: np.ndarray
    log_var_scale: np.ndarray
    log_mu_center: np.ndarray
    log_mu_scale: np.ndarray
    log_beta_center: np.ndarray
    log_beta_scale: np.ndarray
    log_hf_ratio_center: np.ndarray
    log_hf_ratio_scale: np.ndarray


@dataclass(frozen=True)
class TrialDiagnostics:
    trial_risk: np.ndarray
    channel_reliability: np.ndarray
    components: dict[str, np.ndarray]
    channel_components: dict[str, np.ndarray]
    availability: dict[str, str]
    trial_availability: str = "D0b"


def _robust_minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype="float64")
    lo, hi = np.nanpercentile(values, [5, 95])
    if not np.isfinite(hi - lo) or hi <= lo:
        return np.zeros_like(values, dtype="float64")
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _robust_trial_deviation(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype="float64")
    median = np.median(values, axis=1, keepdims=True)
    mad = np.median(np.abs(values - median), axis=1, keepdims=True)
    scale = np.maximum(1.4826 * mad, 1e-6)
    return np.clip(np.abs(values - median) / (4.0 * scale), 0.0, 1.0)


def _robust_center_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype="float64")
    center = np.median(values, axis=0)
    mad = np.median(np.abs(values - center[None, ...]), axis=0)
    scale = np.maximum(1.4826 * mad, 1e-6)
    return center, scale


def _reference_deviation(values: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype="float64")
    return np.clip(np.abs(values - center[None, :]) / (4.0 * scale[None, :]), 0.0, 1.0)


def _spectral_features(X: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = X - np.mean(X, axis=-1, keepdims=True)
    diff_energy = np.mean(np.diff(centered, axis=-1) ** 2, axis=-1)
    channel_var = np.var(centered, axis=-1)
    diff_ratio = np.log(diff_energy / (channel_var + 1e-8) + 1e-8)
    if centered.shape[-1] < 16 or sfreq <= 0:
        log_var = np.log(channel_var + 1e-8)
        return log_var, log_var, diff_ratio

    spec = np.fft.rfft(centered, axis=-1)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(centered.shape[-1], d=1.0 / sfreq)
    mu_hi = min(12.0, freqs.max())
    beta_hi = min(30.0, freqs.max())
    hf_lo = min(30.0, freqs.max())
    hf_hi = min(45.0, freqs.max())
    mu_band = (freqs >= 8.0) & (freqs <= mu_hi)
    beta_band = (freqs >= 13.0) & (freqs <= beta_hi)
    hf_band = (freqs >= hf_lo) & (freqs <= hf_hi)
    if mu_band.sum() < 1 or beta_band.sum() < 1 or hf_band.sum() < 1:
        log_var = np.log(channel_var + 1e-8)
        return log_var, log_var, diff_ratio
    mu_power = np.mean(power[..., mu_band], axis=-1)
    beta_power = np.mean(power[..., beta_band], axis=-1)
    hf_power = np.mean(power[..., hf_band], axis=-1)
    return (
        np.log(mu_power + 1e-8),
        np.log(beta_power + 1e-8),
        np.log(hf_power / (mu_power + beta_power + 1e-8) + 1e-8),
    )


def build_qri_reference(X: np.ndarray, sfreq: float) -> QRIReference:
    X = np.asarray(X, dtype="float64")
    if X.ndim != 3:
        raise ValueError(f"Expected X shape (trials, channels, times), got {X.shape}")
    channel_var = np.var(X, axis=-1)
    log_var = np.log(channel_var + 1e-8)
    log_mu, log_beta, log_hf_ratio = _spectral_features(X, sfreq)
    log_var_center, log_var_scale = _robust_center_scale(log_var)
    log_mu_center, log_mu_scale = _robust_center_scale(log_mu)
    log_beta_center, log_beta_scale = _robust_center_scale(log_beta)
    log_hf_ratio_center, log_hf_ratio_scale = _robust_center_scale(log_hf_ratio)
    return QRIReference(
        log_var_center=log_var_center.astype("float32"),
        log_var_scale=log_var_scale.astype("float32"),
        log_mu_center=log_mu_center.astype("float32"),
        log_mu_scale=log_mu_scale.astype("float32"),
        log_beta_center=log_beta_center.astype("float32"),
        log_beta_scale=log_beta_scale.astype("float32"),
        log_hf_ratio_center=log_hf_ratio_center.astype("float32"),
        log_hf_ratio_scale=log_hf_ratio_scale.astype("float32"),
    )


def _compose_channel_risk(channel_components: dict[str, np.ndarray], mode: str) -> np.ndarray:
    if mode == "local_only":
        weights = {
            "qri_local_spatial_deviation_risk": 0.35,
            "qri_local_burst_risk": 0.20,
            "qri_local_amplitude_deviation_risk": 0.20,
            "qri_local_high_frequency_spread_risk": 0.15,
            "qri_local_flatline_risk": 0.10,
        }
    elif mode == "source_only":
        weights = {
            "qri_source_variance_deviation_risk": 0.25,
            "qri_source_mu_deviation_risk": 0.25,
            "qri_source_beta_deviation_risk": 0.25,
            "qri_source_high_frequency_deviation_risk": 0.25,
        }
    elif mode == "combined":
        weights = {
            "qri_source_variance_deviation_risk": 0.15,
            "qri_source_mu_deviation_risk": 0.15,
            "qri_source_beta_deviation_risk": 0.15,
            "qri_source_high_frequency_deviation_risk": 0.15,
            "qri_local_spatial_deviation_risk": 0.12,
            "qri_local_burst_risk": 0.10,
            "qri_local_amplitude_deviation_risk": 0.08,
            "qri_local_high_frequency_spread_risk": 0.05,
            "qri_local_flatline_risk": 0.05,
        }
    else:
        raise ValueError(f"Unknown QRI composition mode: {mode}")

    shape = next(iter(channel_components.values())).shape
    channel_risk = np.zeros(shape, dtype="float64")
    for key, weight in weights.items():
        value = channel_components.get(key)
        if value is None:
            raise ValueError(f"Missing channel-level QRI component '{key}' for mode={mode}")
        channel_risk += weight * value
    return np.clip(channel_risk, 0.0, 1.0)


def derive_qri_signal(
    diag: TrialDiagnostics,
    mode: str = "combined",
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "random":
        if rng is None:
            raise ValueError("random QRI ablation requires an RNG")
        trial_risk = rng.uniform(0.0, 1.0, size=diag.trial_risk.shape)
        channel_reliability = rng.uniform(0.05, 1.0, size=diag.channel_reliability.shape)
        return trial_risk.astype("float32"), channel_reliability.astype("float32")

    if mode == "shuffled":
        if rng is None:
            raise ValueError("shuffled QRI ablation requires an RNG")
        perm = rng.permutation(diag.channel_reliability.shape[0])
        return diag.trial_risk[perm].astype("float32"), diag.channel_reliability[perm].astype("float32")

    channel_risk = _compose_channel_risk(diag.channel_components, mode)
    trial_risk = np.clip(
        0.70 * np.mean(channel_risk, axis=1) + 0.30 * np.percentile(channel_risk, 90, axis=1),
        0.0,
        1.0,
    )
    trial_modifier = 1.0 - 0.25 * trial_risk[:, None]
    channel_reliability = np.clip((1.0 - channel_risk) * trial_modifier, 0.05, 1.0)
    return trial_risk.astype("float32"), channel_reliability.astype("float32")


def compute_qri(X: np.ndarray, sfreq: float, reference: QRIReference | None = None) -> TrialDiagnostics:
    """Compute rule-based, label-free QRI diagnostics.

    P0-safe means method-side QRI may only use:
    - D0b single-trial internal structure
    - D0a source-only reference statistics
    """

    X = np.asarray(X, dtype="float64")
    if X.ndim != 3:
        raise ValueError(f"Expected X shape (trials, channels, times), got {X.shape}")

    channel_var = np.var(X, axis=-1)
    log_var = np.log(channel_var + 1e-8)
    flatline_risk = (channel_var < 1e-7).astype("float64")

    amplitude = np.max(np.abs(X), axis=-1)
    log_amplitude = np.log(amplitude + 1e-8)
    local_amplitude_risk = _robust_trial_deviation(log_amplitude)
    local_spatial_deviation_risk = _robust_trial_deviation(log_var)
    peak_to_rms = np.log(amplitude / np.sqrt(channel_var + 1e-8) + 1e-8)
    local_burst_risk = _robust_trial_deviation(peak_to_rms)
    log_mu, log_beta, log_hf_ratio = _spectral_features(X, sfreq)
    local_high_frequency_spread_risk = _robust_trial_deviation(log_hf_ratio)

    components = {
        "qri_local_spatial_deviation_risk": np.mean(local_spatial_deviation_risk, axis=1),
        "qri_local_amplitude_deviation_risk": np.mean(local_amplitude_risk, axis=1),
        "qri_local_burst_risk": np.mean(local_burst_risk, axis=1),
        "qri_local_high_frequency_spread_risk": np.mean(local_high_frequency_spread_risk, axis=1),
        "qri_local_flatline_risk": np.mean(flatline_risk, axis=1),
    }
    channel_components = {
        "qri_local_spatial_deviation_risk": local_spatial_deviation_risk,
        "qri_local_amplitude_deviation_risk": local_amplitude_risk,
        "qri_local_burst_risk": local_burst_risk,
        "qri_local_high_frequency_spread_risk": local_high_frequency_spread_risk,
        "qri_local_flatline_risk": flatline_risk,
    }
    availability = {
        "qri_local_spatial_deviation_risk": "D0b",
        "qri_local_amplitude_deviation_risk": "D0b",
        "qri_local_burst_risk": "D0b",
        "qri_local_high_frequency_spread_risk": "D0b",
        "qri_local_flatline_risk": "D0b",
    }

    if reference is None:
        channel_risk = np.clip(
            0.35 * local_spatial_deviation_risk
            + 0.20 * local_burst_risk
            + 0.20 * local_amplitude_risk
            + 0.15 * local_high_frequency_spread_risk
            + 0.10 * flatline_risk,
            0.0,
            1.0,
        )
        trial_availability = "D0b"
    else:
        source_variance_deviation_risk = _reference_deviation(
            log_var,
            reference.log_var_center,
            reference.log_var_scale,
        )
        source_mu_deviation_risk = _reference_deviation(
            log_mu,
            reference.log_mu_center,
            reference.log_mu_scale,
        )
        source_beta_deviation_risk = _reference_deviation(
            log_beta,
            reference.log_beta_center,
            reference.log_beta_scale,
        )
        source_high_frequency_deviation_risk = _reference_deviation(
            log_hf_ratio,
            reference.log_hf_ratio_center,
            reference.log_hf_ratio_scale,
        )
        components.update(
            {
                "qri_source_variance_deviation_risk": np.mean(source_variance_deviation_risk, axis=1),
                "qri_source_mu_deviation_risk": np.mean(source_mu_deviation_risk, axis=1),
                "qri_source_beta_deviation_risk": np.mean(source_beta_deviation_risk, axis=1),
                "qri_source_high_frequency_deviation_risk": np.mean(source_high_frequency_deviation_risk, axis=1),
            }
        )
        channel_components.update(
            {
                "qri_source_variance_deviation_risk": source_variance_deviation_risk,
                "qri_source_mu_deviation_risk": source_mu_deviation_risk,
                "qri_source_beta_deviation_risk": source_beta_deviation_risk,
                "qri_source_high_frequency_deviation_risk": source_high_frequency_deviation_risk,
            }
        )
        availability.update(
            {
                "qri_source_variance_deviation_risk": "D0a",
                "qri_source_mu_deviation_risk": "D0a",
                "qri_source_beta_deviation_risk": "D0a",
                "qri_source_high_frequency_deviation_risk": "D0a",
            }
        )
        channel_risk = np.clip(
            0.15 * source_variance_deviation_risk
            + 0.15 * source_mu_deviation_risk
            + 0.15 * source_beta_deviation_risk
            + 0.15 * source_high_frequency_deviation_risk
            + 0.12 * local_spatial_deviation_risk
            + 0.10 * local_burst_risk
            + 0.08 * local_amplitude_risk
            + 0.05 * local_high_frequency_spread_risk
            + 0.05 * flatline_risk,
            0.0,
            1.0,
        )
        trial_availability = "D0a"

    trial_risk = np.clip(
        0.70 * np.mean(channel_risk, axis=1) + 0.30 * np.percentile(channel_risk, 90, axis=1),
        0.0,
        1.0,
    )
    trial_modifier = 1.0 - 0.25 * trial_risk[:, None]
    channel_reliability = np.clip((1.0 - channel_risk) * trial_modifier, 0.05, 1.0)

    return TrialDiagnostics(
        trial_risk=trial_risk.astype("float32"),
        channel_reliability=channel_reliability.astype("float32"),
        components={key: value.astype("float32") for key, value in components.items()},
        channel_components={key: value.astype("float32") for key, value in channel_components.items()},
        availability=availability,
        trial_availability=trial_availability,
    )
