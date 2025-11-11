"""
Audio-MEG synchronization - CORRECTED VERSION

Uses normalized correlation coefficient with sliding window approach,
adapted from robust_alignment.py
"""

import numpy as np
import mne
from scipy import signal as scipy_signal
from typing import Dict, Tuple, Optional
import librosa
from pathlib import Path

from utils.logging_setup import get_logger

logger = get_logger(__name__)


def synchronize_audio_meg(
    meg_raw: mne.io.Raw,
    external_audio_path: Path,
    config: Dict,
    aux_channel: str = "MISC 007",
) -> Dict:
    """
    Synchronize external audio with MEG auxiliary channel using robust correlation.

    Uses multiple alignment strategies (energy envelope, filtered signals, normalized)
    with normalized correlation coefficient for robustness to amplitude differences.

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
        Synchronization parameters.
    """
    logger.info("Starting robust audio-MEG synchronization...")

    sync_cfg = config.get("sync", {})

    # 1. Extract MEG auxiliary audio
    logger.info(f"Extracting MEG auxiliary channel: {aux_channel}")
    meg_audio, meg_sfreq = _extract_meg_audio(meg_raw, aux_channel)

    # 2. Load external audio
    # IMPORTANT: Use channel=0 (left) for console_mic files to avoid participant leakage
    # Console mic has interviewer on left (ch 0) and participant on right (ch 1)
    logger.info(f"Loading external audio: {external_audio_path}")
    from utils.io import load_audio
    audio_channel = sync_cfg.get("audio_channel", 0)  # Default to left channel
    ext_audio, ext_sfreq = load_audio(external_audio_path, sr=None, channel=audio_channel)
    logger.info(f"  Using channel {audio_channel} (0=left/interviewer, 1=right/participant)")

    logger.info(f"MEG audio: {len(meg_audio)/meg_sfreq:.1f}s at {meg_sfreq}Hz")
    logger.info(f"External audio: {len(ext_audio)/ext_sfreq:.1f}s at {ext_sfreq}Hz")

    # 3. Resample to common sample rate
    target_sr = min(meg_sfreq, ext_sfreq)
    logger.info(f"Using common sample rate: {target_sr} Hz")

    if meg_sfreq != target_sr:
        meg_resampled = librosa.resample(meg_audio, orig_sr=meg_sfreq, target_sr=target_sr)
    else:
        meg_resampled = meg_audio.copy()

    if ext_sfreq != target_sr:
        ext_resampled = librosa.resample(ext_audio, orig_sr=ext_sfreq, target_sr=target_sr)
    else:
        ext_resampled = ext_audio.copy()

    # 4. Try multiple alignment strategies - HYBRID APPROACH
    # First, try full-signal alignment. Only use chunking if correlation < 0.7
    logger.info("HYBRID APPROACH: Testing full-signal alignment first...")

    strategies = [
        ("Energy envelopes",
         _get_energy_envelope(meg_resampled, target_sr),
         _get_energy_envelope(ext_resampled, target_sr)),
        ("Bandpass filtered (200-4000 Hz)",
         _apply_speech_filter(meg_resampled, target_sr),
         _apply_speech_filter(ext_resampled, target_sr)),
        ("Normalized signals",
         _normalize_signal(meg_resampled),
         _normalize_signal(ext_resampled)),
        ("Raw signals",
         meg_resampled,
         ext_resampled),
    ]

    best_corr = 0
    best_result = None
    full_signal_best_corr = 0

    # STEP 1: Try full-signal alignment for all strategies
    for strategy_name, meg_proc, ext_proc in strategies:
        logger.info(f"  Strategy (full-signal): {strategy_name}")

        # Use full-signal sliding window approach
        result = _find_best_alignment_sliding(
            meg_proc, ext_proc, target_sr, config, chunk_offset_samples=0
        )

        if result:
            logger.info(f"    Full-signal correlation: {result['peak_correlation']:.4f}")
            logger.info(f"    Offset: {result['offset_seconds']:.2f}s")

            if abs(result['peak_correlation']) > abs(best_corr):
                best_corr = result['peak_correlation']
                best_result = result
                best_result['strategy'] = strategy_name
                best_result['method'] = 'full-signal'

    full_signal_best_corr = best_corr

    # STEP 2: If full-signal correlation < 0.7, try chunking
    CHUNKING_THRESHOLD = 0.7
    if abs(full_signal_best_corr) < CHUNKING_THRESHOLD:
        logger.info(f"Full-signal correlation ({full_signal_best_corr:.4f}) < {CHUNKING_THRESHOLD}")
        logger.info("Trying chunked approach to avoid dead time/artifacts...")

        for strategy_name, meg_proc, ext_proc in strategies:
            logger.info(f"  Strategy (chunked): {strategy_name}")

            # Use chunked approach to avoid dead time and artifacts
            result = _find_best_alignment_chunked(
                meg_proc, ext_proc, target_sr, config
            )

            if result:
                logger.info(f"    Chunked correlation: {result['peak_correlation']:.4f}")
                logger.info(f"    Offset: {result['offset_seconds']:.2f}s")
                logger.info(f"    Selected chunk: {result.get('selected_chunk', 'N/A')}")

                if abs(result['peak_correlation']) > abs(best_corr):
                    best_corr = result['peak_correlation']
                    best_result = result
                    best_result['strategy'] = strategy_name
                    best_result['method'] = 'chunked'
    else:
        logger.info(f"Full-signal correlation ({full_signal_best_corr:.4f}) >= {CHUNKING_THRESHOLD}")
        logger.info("Using full-signal alignment (chunking not needed)")

    if best_result is None:
        raise ValueError("No reliable alignment found")

    logger.info(f"Best strategy: {best_result['strategy']}")
    logger.info(f"Method: {best_result.get('method', 'unknown')}")
    logger.info(f"Final correlation: {best_result['peak_correlation']:.4f}")
    logger.info(f"Final offset: {best_result['offset_seconds']:.2f}s")

    # 5. Package results
    sync_params = {
        "initial_offset_s": float(best_result['offset_seconds']),
        "drift_model": "none",  # No windowed drift for now
        "drift_coefficients": {"a": 0.0, "b": best_result['offset_seconds']},
        "window_stats": [],
        "qc_metrics": {
            "median_error_ms": 0.0,  # Not computed in this version
            "max_error_ms": 0.0,
            "drift_rate_ppm": 0.0,
            "sync_quality": _assess_quality(best_result['peak_correlation']),
            "n_windows": 0,
            "peak_correlation": float(best_result['peak_correlation']),
            "alignment_strategy": best_result['strategy'],
            "alignment_method": best_result.get('method', 'unknown'),
            "full_signal_correlation": float(full_signal_best_corr),
        },
        "meg_sfreq": float(meg_sfreq),
        "ext_sfreq": float(ext_sfreq),
        "aux_channel": aux_channel,
        "meg_duration_s": float(len(meg_audio) / meg_sfreq),
        "ext_duration_s": float(len(ext_audio) / ext_sfreq),
    }

    logger.info("Synchronization complete!")
    logger.info(f"Quality: {sync_params['qc_metrics']['sync_quality']}")

    return sync_params


def ext_to_meg_time(t_ext: np.ndarray, sync_params: Dict) -> np.ndarray:
    """Convert external audio timestamps to MEG timebase."""
    t_ext = np.asarray(t_ext)
    offset_s = sync_params["initial_offset_s"]
    # External time + offset = MEG time
    # Positive offset means external starts AFTER MEG
    return t_ext + offset_s


def _extract_meg_audio(raw: mne.io.Raw, channel: str) -> Tuple[np.ndarray, float]:
    """Extract audio channel from MEG raw data."""
    if channel not in raw.ch_names:
        available = [ch for ch in raw.ch_names if 'MISC' in ch or 'STI' in ch]
        raise ValueError(
            f"Channel {channel} not found. Available: {available}"
        )

    picks = mne.pick_channels(raw.ch_names, include=[channel])
    audio_data, times = raw[picks, :]
    audio = audio_data.flatten()
    sfreq = raw.info["sfreq"]

    return audio, sfreq


def _find_best_alignment_chunked(
    meg_signal: np.ndarray,
    ext_signal: np.ndarray,
    sr: float,
    config: Dict,
) -> Optional[Dict]:
    """
    Find best alignment using chunked approach to avoid dead time and artifacts.

    Splits signals into 4 chunks and tests chunks 2 and 3 (middle 50% of recording)
    to avoid dead time at beginning/end and select the cleanest portion.
    """
    logger.info("Using chunked synchronization strategy")

    # Split both signals into 4 equal chunks
    shorter_len = min(len(meg_signal), len(ext_signal))
    chunk_size = shorter_len // 4

    # Define chunks 2 and 3 (indices 1 and 2 in 0-indexed array)
    chunks_to_test = [
        ("chunk_2", chunk_size, 2 * chunk_size),
        ("chunk_3", 2 * chunk_size, 3 * chunk_size),
    ]

    best_chunk_result = None
    best_chunk_corr = 0

    for chunk_name, start_idx, end_idx in chunks_to_test:
        logger.info(f"  Testing {chunk_name}: {start_idx/sr:.1f}s to {end_idx/sr:.1f}s")

        # Extract chunk from both signals
        meg_chunk = meg_signal[start_idx:end_idx]
        ext_chunk = ext_signal[start_idx:end_idx]

        # Find alignment for this chunk
        chunk_result = _find_best_alignment_sliding(
            meg_chunk, ext_chunk, sr, config, chunk_offset_samples=start_idx
        )

        if chunk_result:
            logger.info(f"    {chunk_name} correlation: {chunk_result['peak_correlation']:.4f}, offset: {chunk_result['offset_seconds']:.2f}s")

            if abs(chunk_result['peak_correlation']) > abs(best_chunk_corr):
                best_chunk_corr = chunk_result['peak_correlation']
                best_chunk_result = chunk_result
                best_chunk_result['selected_chunk'] = chunk_name

    if best_chunk_result:
        logger.info(f"  Selected {best_chunk_result['selected_chunk']} with correlation {best_chunk_corr:.4f}")

    return best_chunk_result


def _find_best_alignment_sliding(
    meg_signal: np.ndarray,
    ext_signal: np.ndarray,
    sr: float,
    config: Dict,
    chunk_offset_samples: int = 0,
) -> Optional[Dict]:
    """
    Find best alignment using sliding window with normalized correlation.

    This uses correlation coefficient (np.corrcoef) which is invariant
    to amplitude scaling, unlike raw cross-correlation.

    Parameters
    ----------
    chunk_offset_samples : int
        Offset of this chunk from the start of the full signal (for chunked sync).
    """
    # Adaptive window size - use 10 minutes or 80% of shorter signal, whichever is less
    window_duration_s = min(10 * 60, 0.8 * min(len(meg_signal), len(ext_signal)) / sr)
    window_samples = int(window_duration_s * sr)

    # Search window: ±5 minutes (from robust_alignment.py default)
    search_duration_s = 5 * 60
    search_samples = int(search_duration_s * sr)

    logger.debug(f"MEG signal length: {len(meg_signal)/sr:.1f}s")
    logger.debug(f"Ext signal length: {len(ext_signal)/sr:.1f}s")
    logger.debug(f"Window duration: {window_duration_s:.1f}s")
    logger.debug(f"Search range: ±{search_duration_s:.1f}s")

    # Determine which signal is shorter
    if len(meg_signal) < len(ext_signal):
        template = meg_signal
        target = ext_signal
        template_is_meg = True
        logger.debug("Using MEG as template (shorter signal)")
    else:
        template = ext_signal
        target = meg_signal
        template_is_meg = False
        logger.debug("Using External audio as template (shorter signal)")

    # Use window from middle of template
    template_start = max(0, (len(template) - window_samples) // 2)
    template_end = min(len(template), template_start + window_samples)
    template_window = template[template_start:template_end]

    logger.debug(f"Template window: {template_start/sr:.1f}s to {template_end/sr:.1f}s ({len(template_window)/sr:.1f}s)")

    if len(template_window) < 30 * sr:  # Minimum 30 seconds for reliable correlation
        logger.warning(f"Template too short: {len(template_window)/sr:.1f}s")
        return None

    # Search range in target - allow wider search
    target_center = (len(target) - len(template_window)) // 2
    target_start = max(0, target_center - search_samples)
    target_end = min(len(target) - len(template_window), target_center + search_samples)

    logger.debug(f"Target search range: {target_start/sr:.1f}s to {(target_end + len(template_window))/sr:.1f}s")

    # Slide template across target
    step_size = max(1, int(sr / 10))  # 0.1 second steps
    best_corr_abs = 0
    best_corr = 0
    best_offset = 0
    all_corrs = []
    all_offsets = []

    n_positions = 0
    for target_pos in range(target_start, target_end + 1, step_size):
        target_window = target[target_pos:target_pos + len(template_window)]

        if len(target_window) == len(template_window):
            # Normalized correlation coefficient
            corr = np.corrcoef(template_window, target_window)[0, 1]

            if not np.isnan(corr):
                # Calculate offset - positive offset means external audio starts AFTER MEG
                if template_is_meg:
                    # MEG is template, ext is target
                    # If target_pos > template_start: external starts later (positive offset)
                    offset_s = (target_pos - template_start) / sr
                else:
                    # Ext is template, MEG is target
                    # If target_pos > template_start: MEG starts later, ext starts earlier (negative offset)
                    offset_s = (template_start - target_pos) / sr

                # NOTE: chunk_offset_samples is intentionally NOT added here
                # The signals passed to this function are already chunks, and their
                # relative positions encode the offset. Adding chunk_offset would double-count.

                all_corrs.append(corr)
                all_offsets.append(offset_s)
                n_positions += 1

                if abs(corr) > best_corr_abs:
                    best_corr_abs = abs(corr)
                    best_corr = corr
                    best_offset = offset_s

    logger.debug(f"Searched {n_positions} positions")
    logger.debug(f"Best correlation: {best_corr:.4f} (|corr|={best_corr_abs:.4f})")
    logger.debug(f"Best offset: {best_offset:.2f}s")
    if all_offsets:
        logger.debug(f"Offset range: [{min(all_offsets):.1f}s, {max(all_offsets):.1f}s]")

    if best_corr_abs < 0.1:
        logger.warning(f"Poor correlation: {best_corr:.4f}")
        return None

    return {
        "peak_correlation": best_corr,
        "offset_seconds": best_offset,
        "all_correlations": all_corrs,
        "all_offsets": all_offsets,
    }


def _get_energy_envelope(signal: np.ndarray, sr: float, window_ms: float = 50) -> np.ndarray:
    """Calculate energy envelope using RMS smoothing."""
    window_samples = max(1, int(window_ms * sr / 1000))
    energy = np.convolve(
        signal ** 2,
        np.ones(window_samples) / window_samples,
        mode="same"
    )
    return np.sqrt(energy)


def _apply_speech_filter(signal: np.ndarray, sr: float) -> np.ndarray:
    """Apply bandpass filter optimized for speech (200-4000 Hz)."""
    try:
        nyquist = sr / 2
        low = 200 / nyquist
        high = min(4000 / nyquist, 0.99)

        sos = scipy_signal.butter(4, [low, high], btype="band", output="sos")
        filtered = scipy_signal.sosfilt(sos, signal)
        return filtered
    except:
        logger.warning("Speech filter failed, using raw signal")
        return signal


def _normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Normalize signal to unit RMS."""
    rms = np.sqrt(np.mean(signal ** 2))
    if rms > 1e-10:
        return signal / rms
    else:
        return signal


def _assess_quality(correlation: float) -> str:
    """Assess alignment quality based on correlation coefficient."""
    abs_corr = abs(correlation)
    if abs_corr > 0.7:
        return "excellent"
    elif abs_corr > 0.5:
        return "good"
    elif abs_corr > 0.3:
        return "fair"
    else:
        return "poor"


# Helper functions for QC plotting (called by test scripts)
def _load_external_audio(audio_path: Path, config: Dict = None) -> Tuple[np.ndarray, float]:
    """Load external audio file for QC plotting."""
    from utils.io import load_audio
    # Use channel 0 (left) for console_mic files to avoid participant leakage
    audio_channel = 0
    if config and "sync" in config:
        audio_channel = config["sync"].get("audio_channel", 0)
    audio, sr = load_audio(audio_path, sr=None, channel=audio_channel)
    return audio, sr


def _preprocess_audio(audio: np.ndarray, sr: float, config: Dict = None, target_sr: float = None) -> np.ndarray:
    """Preprocess audio for QC plotting (basic resampling and normalization)."""
    import librosa

    # Resample if needed
    if target_sr is not None and sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    # Normalize to unit RMS
    return _normalize_signal(audio)


def _compute_envelope(audio: np.ndarray, sr: float, config: Dict = None) -> np.ndarray:
    """Compute envelope for QC plotting."""
    return _get_energy_envelope(audio, sr)
