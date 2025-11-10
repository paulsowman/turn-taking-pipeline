"""
Drift correction using windowed cross-correlation.

Estimates time-varying offset between audio streams to handle clock drift.
"""

import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
from typing import Dict, Tuple, List

from utils.logging_setup import get_logger

logger = get_logger(__name__)


def estimate_drift(
    meg_env: np.ndarray,
    ext_env: np.ndarray,
    sr: float,
    initial_offset_s: float,
    config: Dict,
) -> Tuple[Dict, List[Dict]]:
    """
    Estimate time drift using sliding window cross-correlation.

    Parameters
    ----------
    meg_env : np.ndarray
        MEG envelope signal.
    ext_env : np.ndarray
        External envelope signal (same sampling rate as MEG).
    sr : float
        Sampling rate.
    initial_offset_s : float
        Initial offset estimate.
    config : dict
        Configuration with window_size_s and window_overlap.

    Returns
    -------
    drift_coeffs : dict
        Linear drift model coefficients {a: slope, b: intercept}.
    window_stats : list of dict
        Per-window statistics.
    """
    window_size_s = config.get("window_size_s", 90)
    overlap = config.get("window_overlap", 0.5)

    window_size = int(window_size_s * sr)
    hop_size = int(window_size * (1 - overlap))

    # Apply initial offset to ext_env
    offset_samples = int(initial_offset_s * sr)
    if offset_samples > 0:
        ext_env_shifted = ext_env[offset_samples:]
        meg_env_aligned = meg_env[:len(ext_env_shifted)]
    else:
        meg_env_aligned = meg_env[-offset_samples:]
        ext_env_shifted = ext_env[:len(meg_env_aligned)]

    # Sliding window cross-correlation
    window_centers = []
    window_offsets = []
    window_stats = []

    n_samples = min(len(meg_env_aligned), len(ext_env_shifted))
    n_windows = (n_samples - window_size) // hop_size + 1

    logger.info(f"Computing drift over {n_windows} windows...")

    for i in range(n_windows):
        start_idx = i * hop_size
        end_idx = start_idx + window_size

        if end_idx > n_samples:
            break

        # Extract windows
        meg_window = meg_env_aligned[start_idx:end_idx]
        ext_window = ext_env_shifted[start_idx:end_idx]

        # Window center time
        center_time_s = (start_idx + window_size / 2) / sr

        # Local cross-correlation
        offset_local_s, xcorr_peak, alignment_error = _window_cross_correlation(
            meg_window, ext_window, sr
        )

        window_centers.append(center_time_s)
        window_offsets.append(offset_local_s)

        window_stats.append({
            "center_time_s": float(center_time_s),
            "offset_s": float(offset_local_s),
            "xcorr_peak": float(xcorr_peak),
            "alignment_error_ms": float(alignment_error * 1000),
        })

    # Fit linear drift model
    if len(window_centers) > 1:
        drift_coeffs = _fit_drift_model(
            np.array(window_centers),
            np.array(window_offsets),
            initial_offset_s,
        )
    else:
        logger.warning("Not enough windows for drift estimation")
        drift_coeffs = {"a": 0.0, "b": initial_offset_s}

    return drift_coeffs, window_stats


def _window_cross_correlation(
    meg_window: np.ndarray,
    ext_window: np.ndarray,
    sr: float,
    max_lag_s: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Compute cross-correlation for a single window.

    Returns
    -------
    offset_s : float
        Local time offset (seconds).
    xcorr_peak : float
        Normalized cross-correlation peak.
    alignment_error : float
        Alignment error (seconds).
    """
    max_lag_samples = int(max_lag_s * sr)

    # Normalize
    meg_norm = (meg_window - np.mean(meg_window)) / (np.std(meg_window) + 1e-8)
    ext_norm = (ext_window - np.mean(ext_window)) / (np.std(ext_window) + 1e-8)

    # Cross-correlation
    xcorr = signal.correlate(meg_norm, ext_norm, mode="same", method="auto")

    # Find peak near zero lag
    center = len(xcorr) // 2
    search_start = max(0, center - max_lag_samples)
    search_end = min(len(xcorr), center + max_lag_samples)

    search_region = xcorr[search_start:search_end]
    peak_idx_local = np.argmax(np.abs(search_region))
    peak_idx = search_start + peak_idx_local

    # Offset
    lag_samples = peak_idx - center
    offset_s = lag_samples / sr
    xcorr_peak = xcorr[peak_idx] / len(meg_norm)

    # Alignment error (absolute offset from zero)
    alignment_error = abs(offset_s)

    return offset_s, xcorr_peak, alignment_error


def _fit_drift_model(
    times: np.ndarray,
    offsets: np.ndarray,
    initial_offset: float,
) -> Dict:
    """
    Fit linear drift model to offset estimates.

    Model: offset(t) = a * t + b

    Parameters
    ----------
    times : np.ndarray
        Window center times.
    offsets : np.ndarray
        Estimated offsets.
    initial_offset : float
        Initial offset (used as constraint).

    Returns
    -------
    coeffs : dict
        {a: slope, b: intercept}
    """
    # Add initial offset point at t=0
    times_full = np.concatenate([[0], times])
    offsets_full = np.concatenate([[initial_offset], offsets])

    # Linear regression
    def linear_model(t, a, b):
        return a * t + b

    try:
        popt, _ = curve_fit(linear_model, times_full, offsets_full)
        a, b = popt
    except Exception as e:
        logger.warning(f"Drift model fitting failed: {e}. Using simple mean.")
        a = 0.0
        b = np.mean(offsets_full)

    return {"a": float(a), "b": float(b)}


def apply_drift_correction(
    timestamps: np.ndarray,
    drift_coeffs: Dict,
) -> np.ndarray:
    """
    Apply drift correction to timestamps.

    Parameters
    ----------
    timestamps : np.ndarray
        Original timestamps (seconds).
    drift_coeffs : dict
        Drift model coefficients.

    Returns
    -------
    corrected_timestamps : np.ndarray
        Drift-corrected timestamps.
    """
    a = drift_coeffs["a"]
    b = drift_coeffs["b"]

    # Correction: t_corrected = t - offset(t)
    offset = a * timestamps + b
    corrected = timestamps - offset

    return corrected
