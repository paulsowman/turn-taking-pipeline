"""
TRF Predictor Generation Functions

Creates predictor time series for Temporal Response Function (TRF) analysis.
All predictors are generated at MEG sampling rate (1000 Hz) in MEG timebase.
"""

import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from scipy.ndimage import gaussian_filter1d
import warnings

from utils.logging_setup import get_logger

logger = get_logger(__name__)


def create_envelope_predictor(
    audio: np.ndarray,
    sr: int,
    meg_times: np.ndarray,
    sync_offset: float,
    hop_length_ms: float = 10,
    smoothing_sigma_ms: float = 10,
) -> np.ndarray:
    """
    Create audio envelope predictor at MEG sampling rate.

    Parameters
    ----------
    audio : np.ndarray
        Audio signal (mono)
    sr : int
        Audio sampling rate
    meg_times : np.ndarray
        MEG time points (seconds)
    sync_offset : float
        Sync offset for audio->MEG conversion (meg_time = audio_time - offset)
    hop_length_ms : float
        Hop length for envelope computation (milliseconds)
    smoothing_sigma_ms : float
        Gaussian smoothing width (milliseconds)

    Returns
    -------
    envelope_meg : np.ndarray
        Envelope at MEG time points
    """
    # Compute envelope in audio timebase
    hop_length = int(hop_length_ms * sr / 1000)

    envelope = librosa.feature.rms(
        y=audio,
        frame_length=int(0.025 * sr),  # 25ms window
        hop_length=hop_length
    )[0]

    envelope_times_audio = np.arange(len(envelope)) * hop_length / sr

    # Convert to MEG timebase
    envelope_times_meg = envelope_times_audio - sync_offset

    # Interpolate to MEG sampling rate
    envelope_meg = np.interp(meg_times, envelope_times_meg, envelope)

    # Smooth if requested
    if smoothing_sigma_ms > 0:
        meg_sfreq = 1.0 / np.median(np.diff(meg_times))
        sigma_samples = smoothing_sigma_ms * meg_sfreq / 1000
        envelope_meg = gaussian_filter1d(envelope_meg, sigma_samples)

    logger.info(f"Created envelope predictor: {len(envelope_meg)} samples, range={envelope_meg.min():.4f} to {envelope_meg.max():.4f}")

    return envelope_meg


def create_meg_audio_envelope(
    meg_raw: 'mne.io.Raw',
    channel_name: str,
    smoothing_sigma_ms: float = 10,
    normalize: bool = True,
) -> np.ndarray:
    """
    Create envelope from MEG auxiliary audio channel.

    Parameters
    ----------
    meg_raw : mne.io.Raw
        MEG raw data
    channel_name : str
        Name of MEG audio channel (e.g., 'MISC 007', 'MISC 008')
    smoothing_sigma_ms : float
        Gaussian smoothing width (milliseconds)
    normalize : bool
        Normalize envelope to 0-1 range for comparison with external audio

    Returns
    -------
    envelope_meg : np.ndarray
        Envelope of MEG audio channel

    Notes
    -----
    This extracts the envelope from audio recorded directly to the MEG file
    (e.g., from microphones connected to MEG auxiliary inputs).
    Useful for verifying synchronization with external audio.

    The envelope is normalized by default so it can be visually compared with
    external audio envelopes on the same scale.
    """
    if channel_name not in meg_raw.ch_names:
        raise ValueError(f"Channel {channel_name} not found in MEG data")

    # Get channel data
    audio_data = meg_raw[channel_name][0][0]

    # Compute envelope (absolute value)
    envelope_meg = np.abs(audio_data)

    # Smooth if requested
    if smoothing_sigma_ms > 0:
        meg_sfreq = meg_raw.info['sfreq']
        sigma_samples = smoothing_sigma_ms * meg_sfreq / 1000
        envelope_meg = gaussian_filter1d(envelope_meg, sigma_samples)

    # Normalize to 0-1 range for visual comparison with external audio
    if normalize:
        env_min = envelope_meg.min()
        env_max = envelope_meg.max()
        if env_max > env_min:
            envelope_meg = (envelope_meg - env_min) / (env_max - env_min)
        logger.info(f"Created MEG audio envelope ({channel_name}): normalized to 0-1 range")
    else:
        logger.info(f"Created MEG audio envelope ({channel_name}): range={envelope_meg.min():.4f} to {envelope_meg.max():.4f}")

    return envelope_meg


def create_f0_predictor(
    audio: np.ndarray,
    sr: int,
    meg_times: np.ndarray,
    sync_offset: float,
    fmin: float = 80,
    fmax: float = 400,
    hop_length_ms: float = 10,
    fill_unvoiced: float = 0.0,
) -> np.ndarray:
    """
    Create F0 (pitch) predictor at MEG sampling rate.

    Parameters
    ----------
    audio : np.ndarray
        Audio signal (mono)
    sr : int
        Audio sampling rate
    meg_times : np.ndarray
        MEG time points (seconds)
    sync_offset : float
        Sync offset for audio->MEG conversion
    fmin : float
        Minimum F0 (Hz)
    fmax : float
        Maximum F0 (Hz)
    hop_length_ms : float
        Hop length for F0 computation (milliseconds)
    fill_unvoiced : float
        Value to use for unvoiced segments (0.0 or np.nan)

    Returns
    -------
    f0_meg : np.ndarray
        F0 at MEG time points
    """
    hop_length = int(hop_length_ms * sr / 1000)

    # Compute F0 in audio timebase
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            sr=sr,
            fmin=fmin,
            fmax=fmax,
            hop_length=hop_length,
            frame_length=int(0.025 * sr),
            fill_na=fill_unvoiced
        )

    f0_times_audio = np.arange(len(f0)) * hop_length / sr

    # Convert to MEG timebase
    f0_times_meg = f0_times_audio - sync_offset

    # Interpolate to MEG sampling rate
    f0_meg = np.interp(meg_times, f0_times_meg, f0)

    n_voiced = np.sum(f0_meg > 0)
    pct_voiced = 100 * n_voiced / len(f0_meg)
    logger.info(f"Created F0 predictor: {len(f0_meg)} samples, {pct_voiced:.1f}% voiced")

    return f0_meg


def create_word_onset_predictor(
    word_times_meg: np.ndarray,
    meg_times: np.ndarray,
) -> np.ndarray:
    """
    Create word onset predictor (delta/stick functions).

    Parameters
    ----------
    word_times_meg : np.ndarray
        Word onset times in MEG timebase (seconds)
    meg_times : np.ndarray
        MEG time points (seconds)

    Returns
    -------
    onsets : np.ndarray
        Delta functions at word onset times (1 at onset, 0 elsewhere)
    """
    onsets = np.zeros(len(meg_times))

    # For each word, find nearest MEG sample
    for word_time in word_times_meg:
        idx = np.argmin(np.abs(meg_times - word_time))
        onsets[idx] = 1.0

    logger.info(f"Created word onset predictor: {len(word_times_meg)} onsets")

    return onsets


def create_surprisal_predictor(
    word_times_meg: np.ndarray,
    surprisal_values: np.ndarray,
    meg_times: np.ndarray,
) -> np.ndarray:
    """
    Create surprisal predictor (delta functions weighted by surprisal).

    Parameters
    ----------
    word_times_meg : np.ndarray
        Word onset times in MEG timebase (seconds)
    surprisal_values : np.ndarray
        Surprisal value for each word
    meg_times : np.ndarray
        MEG time points (seconds)

    Returns
    -------
    surprisal : np.ndarray
        Delta functions weighted by surprisal
    """
    surprisal = np.zeros(len(meg_times))

    for word_time, surp_val in zip(word_times_meg, surprisal_values):
        idx = np.argmin(np.abs(meg_times - word_time))
        surprisal[idx] = surp_val

    logger.info(f"Created surprisal predictor: mean={np.mean(surprisal_values):.2f}, range={surprisal_values.min():.2f} to {surprisal_values.max():.2f}")

    return surprisal


def create_duration_predictor(
    word_times_meg: np.ndarray,
    word_durations: np.ndarray,
    meg_times: np.ndarray,
) -> np.ndarray:
    """
    Create word duration predictor (delta functions weighted by duration).

    Parameters
    ----------
    word_times_meg : np.ndarray
        Word onset times in MEG timebase (seconds)
    word_durations : np.ndarray
        Duration of each word (seconds)
    meg_times : np.ndarray
        MEG time points (seconds)

    Returns
    -------
    duration : np.ndarray
        Delta functions weighted by word duration
    """
    duration = np.zeros(len(meg_times))

    for word_time, dur in zip(word_times_meg, word_durations):
        idx = np.argmin(np.abs(meg_times - word_time))
        duration[idx] = dur

    logger.info(f"Created duration predictor: mean={np.mean(word_durations)*1000:.1f}ms, range={word_durations.min()*1000:.1f} to {word_durations.max()*1000:.1f}ms")

    return duration


def create_speaker_predictor(
    env_interviewer: np.ndarray,
    env_participant: np.ndarray,
    threshold_db: float = 6.0,
    silence_threshold: float = 0.01,
    min_duration_samples: int = 50,
) -> np.ndarray:
    """
    Create speaker predictor based on envelope energy.

    Parameters
    ----------
    env_interviewer : np.ndarray
        Interviewer envelope at MEG sampling rate
    env_participant : np.ndarray
        Participant envelope at MEG sampling rate
    threshold_db : float
        Energy difference threshold (dB) for single-speaker detection
    silence_threshold : float
        Energy threshold for silence detection (RMS amplitude)
    min_duration_samples : int
        Minimum duration for speaker segments (samples)

    Returns
    -------
    speaker : np.ndarray
        Speaker labels: 0=silence, 1=interviewer, 2=participant, 3=overlap
    """
    # Compute energy ratio in dB (for single-speaker detection)
    ratio_db = 10 * np.log10((env_interviewer + 1e-10) / (env_participant + 1e-10))

    # Detect silence (both channels low)
    is_silence = (env_interviewer < silence_threshold) & (env_participant < silence_threshold)

    # Initial classification
    speaker = np.zeros(len(ratio_db), dtype=int)

    # Silence
    speaker[is_silence] = 0

    # Single speakers (when one is clearly louder)
    speaker[(~is_silence) & (ratio_db > threshold_db)] = 1  # Interviewer
    speaker[(~is_silence) & (ratio_db < -threshold_db)] = 2  # Participant

    # Overlap (both speaking: not silence, and ratio within threshold)
    speaker[(~is_silence) & (np.abs(ratio_db) <= threshold_db)] = 3

    # Remove brief segments (morphological opening)
    if min_duration_samples > 1:
        from scipy.ndimage import binary_opening
        kernel = np.ones(min_duration_samples)

        # Process each speaker category separately
        for label in [1, 2, 3]:
            mask = speaker == label
            mask_cleaned = binary_opening(mask, structure=kernel)
            # Remove brief segments - revert to silence
            speaker[mask & ~mask_cleaned] = 0

    # Statistics
    n_silence = np.sum(speaker == 0)
    n_interviewer = np.sum(speaker == 1)
    n_participant = np.sum(speaker == 2)
    n_overlap = np.sum(speaker == 3)
    total = len(speaker)

    logger.info(f"Created speaker predictor:")
    logger.info(f"  Silence:      {100*n_silence/total:5.1f}% ({n_silence} samples)")
    logger.info(f"  Interviewer:  {100*n_interviewer/total:5.1f}% ({n_interviewer} samples)")
    logger.info(f"  Participant:  {100*n_participant/total:5.1f}% ({n_participant} samples)")
    logger.info(f"  Overlap:      {100*n_overlap/total:5.1f}% ({n_overlap} samples)")

    return speaker


def compute_word_surprisal(
    words: List[str],
    context_window: int = 50,
    model_name: str = "gpt2",
    use_cache: bool = True,
) -> np.ndarray:
    """
    Compute word surprisal using GPT-2 language model.

    Surprisal is defined as -log P(word | context), measuring how unexpected
    a word is given the preceding context.

    Parameters
    ----------
    words : list of str
        List of words in order
    context_window : int
        Number of previous words to use as context
    model_name : str
        GPT-2 model variant ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')
    use_cache : bool
        Whether to cache model (saves memory if processing multiple subjects)

    Returns
    -------
    surprisal : np.ndarray
        Surprisal value for each word (nats)

    Notes
    -----
    Requires: pip install transformers torch
    First run will download ~500MB model from HuggingFace.
    """
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        import torch
    except ImportError:
        logger.error("GPT-2 surprisal requires: pip install transformers torch")
        raise

    logger.info(f"Computing surprisal for {len(words)} words using {model_name}...")

    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    logger.info(f"  Using device: {device}")

    surprisal = np.zeros(len(words))

    with torch.no_grad():
        for i, word in enumerate(words):
            # Get context (previous words)
            start_idx = max(0, i - context_window)
            context = words[start_idx:i] if i > 0 else []

            # Prepare input
            if context:
                context_text = " " + " ".join(context) + " " + word
            else:
                context_text = word

            # Tokenize
            input_ids = tokenizer.encode(context_text, return_tensors='pt').to(device)

            # Get logits
            outputs = model(input_ids)
            logits = outputs.logits

            # Get target token (last token in sequence)
            target_token_id = input_ids[0, -1]

            # Get log probability of target token
            log_probs = torch.nn.functional.log_softmax(logits[0, -2, :], dim=0)
            log_prob = log_probs[target_token_id].item()

            # Surprisal = -log P(word | context)
            surprisal[i] = -log_prob

            if (i + 1) % 100 == 0:
                logger.info(f"  Processed {i + 1}/{len(words)} words")

    logger.info(f"  Surprisal: mean={np.mean(surprisal):.2f}, std={np.std(surprisal):.2f}, range={surprisal.min():.2f} to {surprisal.max():.2f}")

    # Clean up to save memory
    if not use_cache:
        del model
        del tokenizer
        if device == 'cuda':
            torch.cuda.empty_cache()

    return surprisal
