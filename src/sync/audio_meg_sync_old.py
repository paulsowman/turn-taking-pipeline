"""
Audio-MEG synchronization with drift correction.

This module implements robust synchronization between external high-rate audio
(e.g., 44.1 kHz WAV) and low-rate auxiliary audio recorded in MEG (~1 kHz).
"""

import numpy as np
import mne
from scipy import signal
from scipy.signal import hilbert, correlate
from typing import Dict, Tuple, Optional, Callable
import librosa
from pathlib import Path

from utils.logging_setup import get_logger

logger = get_logger(__name__)


def synchronize_audio_meg(
    meg_raw: mne.io.Raw,
    external_audio_path: Path,
    config: Dict,
    aux_channel: str = "MISC007",
) -> Dict:
    """
    Synchronize external audio with MEG auxiliary channel.

    This function:
    1. Extracts MEG auxiliary audio channel
    2. Loads and preprocesses external audio
    3. Computes initial offset via cross-correlation
    4. Estimates drift with windowed cross-correlation
    5. Returns synchronization parameters and mapping function

    Parameters
    ----------
    meg_raw : mne.io.Raw
        MEG raw data containing auxiliary audio channel.
    external_audio_path : Path
        Path to external WAV audio file.
    config : dict
        Configuration dictionary with sync parameters.
    aux_channel : str
        Name of MEG auxiliary audio channel.

    Returns
    -------
    sync_params : dict
        Synchronization parameters containing:
        - initial_offset_s : float
            Initial time offset (seconds) from external to MEG
        - drift_model : str
            "linear" or "piecewise"
        - drift_coefficients : dict
            {a: slope, b: intercept} for linear model
        - window_stats : list of dict
            Per-window sync statistics
        - qc_metrics : dict
            Quality control metrics
        - meg_sfreq : float
            MEG sampling frequency
        - ext_sfreq : float
            External audio sampling frequency
    """
    logger.info("Starting audio-MEG synchronization...")

    sync_cfg = config.get("sync", {})

    # 1. Extract MEG auxiliary audio
    logger.info(f"Extracting MEG auxiliary channel: {aux_channel}")
    meg_audio, meg_sfreq = _extract_meg_audio(meg_raw, aux_channel, sync_cfg)

    # 2. Load external audio
    logger.info(f"Loading external audio: {external_audio_path}")
    ext_audio, ext_sfreq = _load_external_audio(external_audio_path, sync_cfg)

    # 3. Preprocess both signals
    logger.info("Preprocessing audio signals...")
    meg_audio_prep = _preprocess_audio(meg_audio, meg_sfreq, sync_cfg)
    ext_audio_prep = _preprocess_audio(ext_audio, ext_sfreq, sync_cfg, target_sr=meg_sfreq)

    # 4. Compute envelope for cross-correlation
    logger.info("Computing amplitude envelopes...")
    meg_env = _compute_envelope(meg_audio_prep, meg_sfreq, sync_cfg)
    ext_env = _compute_envelope(ext_audio_prep, meg_sfreq, sync_cfg)

    # 5. Find initial offset
    logger.info("Computing initial offset via cross-correlation...")
    initial_offset_s, xcorr_peak = _find_initial_offset(
        meg_env, ext_env, meg_sfreq, sync_cfg
    )
    logger.info(f"Initial offset: {initial_offset_s:.4f} s (xcorr peak: {xcorr_peak:.4f})")

    # 6. Estimate drift (if enabled)
    drift_coeffs = {"a": 0.0, "b": initial_offset_s}
    window_stats = []

    if sync_cfg.get("use_windowed_sync", True):
        logger.info("Estimating drift with windowed cross-correlation...")
        drift_coeffs, window_stats = _estimate_drift_windowed(
            meg_env, ext_env, meg_sfreq, initial_offset_s, sync_cfg
        )
        logger.info(
            f"Drift model: offset(t) = {drift_coeffs['a']:.6e} * t + {drift_coeffs['b']:.4f}"
        )

    # 7. Quality control
    logger.info("Computing QC metrics...")
    qc_metrics = _compute_qc_metrics(
        meg_env,
        ext_env,
        meg_sfreq,
        drift_coeffs,
        window_stats,
        sync_cfg,
    )

    # 8. Package results
    sync_params = {
        "initial_offset_s": float(initial_offset_s),
        "drift_model": sync_cfg.get("drift_model", "linear"),
        "drift_coefficients": drift_coeffs,
        "window_stats": window_stats,
        "qc_metrics": qc_metrics,
        "meg_sfreq": float(meg_sfreq),
        "ext_sfreq": float(ext_sfreq),
        "aux_channel": aux_channel,
        "meg_duration_s": float(len(meg_audio) / meg_sfreq),
        "ext_duration_s": float(len(ext_audio) / ext_sfreq),
    }

    logger.info("Synchronization complete!")
    logger.info(f"Median alignment error: {qc_metrics['median_error_ms']:.2f} ms")

    return sync_params


def ext_to_meg_time(
    t_ext: np.ndarray,
    sync_params: Dict,
) -> np.ndarray:
    """
    Convert external audio timestamps to MEG timebase.

    Parameters
    ----------
    t_ext : array-like
        External audio timestamps (seconds).
    sync_params : dict
        Synchronization parameters from synchronize_audio_meg().

    Returns
    -------
    t_meg : np.ndarray
        Corresponding MEG timestamps (seconds).
    """
    t_ext = np.asarray(t_ext)
    coeffs = sync_params["drift_coefficients"]

    # Apply drift model: t_meg = t_ext - offset(t_ext)
    # where offset(t) = a * t + b
    offset = coeffs["a"] * t_ext + coeffs["b"]
    t_meg = t_ext - offset

    return t_meg


# ============================================================================
# Helper functions
# ============================================================================


def _extract_meg_audio(
    raw: mne.io.Raw,
    channel: str,
    config: Dict,
) -> Tuple[np.ndarray, float]:
    """Extract audio channel from MEG raw data."""
    if channel not in raw.ch_names:
        available = [ch for ch in raw.ch_names if "MISC" in ch or "STI" in ch]
        raise ValueError(
            f"Channel {channel} not found in MEG data. "
            f"Available auxiliary channels: {available}"
        )

    # Pick channel
    picks = mne.pick_channels(raw.ch_names, include=[channel])
    audio_data, times = raw[picks, :]
    audio = audio_data.flatten()
    sfreq = raw.info["sfreq"]

    return audio, sfreq


def _load_external_audio(
    audio_path: Path,
    config: Dict,
) -> Tuple[np.ndarray, float]:
    """Load external audio file."""
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load with librosa (handles various formats)
    audio, sr = librosa.load(audio_path, sr=None, mono=True)

    return audio, sr


def _preprocess_audio(
    audio: np.ndarray,
    sr: float,
    config: Dict,
    target_sr: Optional[float] = None,
) -> np.ndarray:
    """
    Preprocess audio: low-pass filter and optionally resample.

    Parameters
    ----------
    audio : np.ndarray
        Audio signal.
    sr : float
        Sampling rate.
    config : dict
        Sync configuration.
    target_sr : float, optional
        Target sampling rate for resampling.

    Returns
    -------
    audio_processed : np.ndarray
        Preprocessed audio.
    """
    # Low-pass filter to prevent aliasing
    lowpass_freq = config.get("lowpass_freq", 400)
    if lowpass_freq < sr / 2:
        nyq = sr / 2
        sos = signal.butter(4, lowpass_freq / nyq, btype="low", output="sos")
        audio = signal.sosfilt(sos, audio)

    # Resample if needed
    if target_sr is not None and target_sr != sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    return audio


def _compute_envelope(
    audio: np.ndarray,
    sr: float,
    config: Dict,
) -> np.ndarray:
    """
    Compute amplitude envelope via Hilbert transform.

    Parameters
    ----------
    audio : np.ndarray
        Audio signal.
    sr : float
        Sampling rate.
    config : dict
        Sync configuration.

    Returns
    -------
    envelope : np.ndarray
        Amplitude envelope.
    """
    # Hilbert transform
    analytic = hilbert(audio)
    envelope = np.abs(analytic)

    # Low-pass filter envelope
    env_lowpass = config.get("envelope_lowpass", 15)
    nyq = sr / 2
    if env_lowpass < nyq:
        sos = signal.butter(3, env_lowpass / nyq, btype="low", output="sos")
        envelope = signal.sosfilt(sos, envelope)

    return envelope


def _find_initial_offset(
    meg_env: np.ndarray,
    ext_env: np.ndarray,
    sr: float,
    config: Dict,
) -> Tuple[float, float]:
    """
    Find initial time offset via cross-correlation.

    Parameters
    ----------
    meg_env : np.ndarray
        MEG envelope.
    ext_env : np.ndarray
        External audio envelope.
    sr : float
        Sampling rate.
    config : dict
        Sync configuration.

    Returns
    -------
    offset_s : float
        Time offset (seconds).
    peak_value : float
        Normalized cross-correlation peak value.
    """
    max_lag_s = config.get("max_lag_s", 10.0)
    max_lag_samples = int(max_lag_s * sr)

    # Normalize signals
    meg_norm = (meg_env - np.mean(meg_env)) / np.std(meg_env)
    ext_norm = (ext_env - np.mean(ext_env)) / np.std(ext_env)

    # Cross-correlate
    xcorr = correlate(meg_norm, ext_norm, mode="valid", method="fft")

    # Find peak within reasonable lag range
    # Restrict search to central region
    n_xcorr = len(xcorr)
    center = n_xcorr // 2
    search_start = max(0, center - max_lag_samples)
    search_end = min(n_xcorr, center + max_lag_samples)

    search_region = xcorr[search_start:search_end]
    peak_idx = np.argmax(np.abs(search_region))
    peak_idx_global = search_start + peak_idx

    # Convert to time offset
    lag_samples = peak_idx_global - (len(ext_env) - 1)
    offset_s = lag_samples / sr
    peak_value = xcorr[peak_idx_global] / len(ext_norm)

    return offset_s, peak_value


def _estimate_drift_windowed(
    meg_env: np.ndarray,
    ext_env: np.ndarray,
    sr: float,
    initial_offset_s: float,
    config: Dict,
) -> Tuple[Dict, list]:
    """
    Estimate time drift using windowed cross-correlation.

    Parameters
    ----------
    meg_env : np.ndarray
        MEG envelope.
    ext_env : np.ndarray
        External envelope (resampled to MEG rate).
    sr : float
        Sampling rate.
    initial_offset_s : float
        Initial offset estimate.
    config : dict
        Sync configuration.

    Returns
    -------
    drift_coeffs : dict
        {a: slope, b: intercept} for linear drift model.
    window_stats : list of dict
        Statistics for each window.
    """
    from .drift_correction import estimate_drift

    drift_coeffs, window_stats = estimate_drift(
        meg_env, ext_env, sr, initial_offset_s, config
    )

    return drift_coeffs, window_stats


def _compute_qc_metrics(
    meg_env: np.ndarray,
    ext_env: np.ndarray,
    sr: float,
    drift_coeffs: Dict,
    window_stats: list,
    config: Dict,
) -> Dict:
    """
    Compute quality control metrics for synchronization.

    Returns
    -------
    qc_metrics : dict
        - median_error_ms : Median alignment error
        - max_error_ms : Maximum alignment error
        - drift_rate_ppm : Drift rate in parts per million
        - sync_quality : "pass" or "fail"
    """
    if not window_stats:
        # No windowed analysis, estimate from overall drift
        median_error_ms = 0.0
        max_error_ms = 0.0
    else:
        # Extract errors from windows
        errors_ms = [w["alignment_error_ms"] for w in window_stats]
        median_error_ms = float(np.median(errors_ms))
        max_error_ms = float(np.max(errors_ms))

    # Drift rate in parts per million (ppm)
    drift_slope = drift_coeffs.get("a", 0.0)
    drift_rate_ppm = abs(drift_slope) * 1e6

    # QC pass/fail
    target_error = config.get("target_alignment_error_ms", 10.0)
    max_acceptable = config.get("max_acceptable_error_ms", 50.0)

    if median_error_ms < target_error:
        sync_quality = "excellent"
    elif median_error_ms < max_acceptable:
        sync_quality = "pass"
    else:
        sync_quality = "fail"

    return {
        "median_error_ms": median_error_ms,
        "max_error_ms": max_error_ms,
        "drift_rate_ppm": drift_rate_ppm,
        "sync_quality": sync_quality,
        "n_windows": len(window_stats),
    }
