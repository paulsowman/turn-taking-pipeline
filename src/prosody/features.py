"""
Prosodic feature extraction for turn-taking prediction.

Uses Parselmouth (Praat) for F0 extraction and librosa for energy/spectral features.
"""

import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
import librosa
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy import signal as scipy_signal

from utils.logging_setup import get_logger

logger = get_logger(__name__)


def extract_prosody_features(
    audio_path: Path,
    sr: Optional[int] = None,
    frame_shift: float = 0.01,  # 10ms frames
    f0_min: float = 75.0,
    f0_max: float = 500.0,
    channel: Optional[int] = 0,
) -> pd.DataFrame:
    """
    Extract comprehensive prosodic features from audio.

    Parameters
    ----------
    audio_path : Path
        Path to audio file.
    sr : int, optional
        Sampling rate. If None, uses file's native rate.
    frame_shift : float
        Time step between frames (seconds). Default 10ms.
    f0_min, f0_max : float
        F0 (pitch) search range in Hz.
    channel : int, optional
        Channel to load for stereo files:
        - 0: Left channel (default, interviewer for console_mic)
        - 1: Right channel (participant for console_mic)
        - None: Mix to mono (NOT RECOMMENDED for dual-mic recordings)

    Returns
    -------
    features : pd.DataFrame
        DataFrame with columns:
        - time: Center time of frame (seconds)
        - f0: Fundamental frequency (Hz), NaN for unvoiced
        - f0_smooth: Smoothed F0 (interpolated through unvoiced)
        - intensity: Sound intensity (dB)
        - energy: RMS energy
        - zcr: Zero-crossing rate
        - spectral_centroid: Spectral centroid (Hz)
        - spectral_flux: Spectral flux (change from previous frame)
        - is_voiced: Boolean, True if F0 detected

    Notes
    -----
    - Uses Praat's autocorrelation method for robust F0 extraction
    - Frame-by-frame features for alignment with MEG data
    - Suitable for TRF modeling and turn-taking prediction
    - For dual-mic recordings: ALWAYS specify channel to avoid cross-talk
    """
    audio_path = Path(audio_path)
    logger.info(f"Extracting prosody from: {audio_path.name}")
    if channel is not None:
        logger.info(f"  Using channel {channel} (0=left/interviewer, 1=right/participant)")

    # Load audio with proper channel selection
    from utils.io import load_audio
    y, sr_librosa = load_audio(audio_path, sr=sr, channel=channel)

    # Create Parselmouth Sound object from audio array
    # Parselmouth.Sound can be created from numpy array
    snd = parselmouth.Sound(y, sampling_frequency=sr_librosa)
    if sr is not None and snd.sampling_frequency != sr:
        snd = snd.resample(sr)
        sr_librosa = sr

    # === F0 (Pitch) Extraction ===
    logger.info("Extracting F0...")
    pitch = snd.to_pitch(time_step=frame_shift, pitch_floor=f0_min, pitch_ceiling=f0_max)

    # Extract F0 values
    times = pitch.xs()
    f0_values = []
    for t in times:
        f0 = pitch.get_value_at_time(t)
        f0_values.append(f0 if f0 > 0 else np.nan)

    # Smooth F0 (interpolate through unvoiced regions)
    f0_smooth = pd.Series(f0_values).interpolate(method='linear', limit=5).values

    # === Intensity ===
    logger.info("Extracting intensity...")
    intensity = snd.to_intensity(time_step=frame_shift)
    intensity_values = [call(intensity, "Get value at time", t, "Linear") for t in times]

    # === Energy and Spectral Features ===
    logger.info("Extracting spectral features...")

    # Frame parameters
    hop_length = int(frame_shift * sr_librosa)
    n_fft = 2048

    # RMS Energy
    rms = librosa.feature.rms(y=y, hop_length=hop_length, frame_length=n_fft)[0]

    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length, frame_length=n_fft)[0]

    # Spectral centroid
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr_librosa, hop_length=hop_length, n_fft=n_fft)[0]

    # Spectral flux (change between frames)
    spec = np.abs(librosa.stft(y, hop_length=hop_length, n_fft=n_fft))
    spec_flux = np.sqrt(np.sum(np.diff(spec, axis=1)**2, axis=0))
    spec_flux = np.concatenate([[0], spec_flux])  # Prepend 0 for first frame

    # Time axis for librosa features
    times_librosa = librosa.frames_to_time(np.arange(len(rms)), sr=sr_librosa, hop_length=hop_length)

    # === Align all features to common time grid ===
    # Use Praat times as reference (more consistent with F0)
    n_frames = len(times)

    # Interpolate librosa features to Praat time grid
    rms_interp = np.interp(times, times_librosa, rms)
    zcr_interp = np.interp(times, times_librosa, zcr)
    spec_cent_interp = np.interp(times, times_librosa, spec_cent)
    spec_flux_interp = np.interp(times, times_librosa, spec_flux)

    # === Create DataFrame ===
    features = pd.DataFrame({
        "time": times,
        "f0": f0_values,
        "f0_smooth": f0_smooth,
        "intensity": intensity_values,
        "energy": rms_interp,
        "zcr": zcr_interp,
        "spectral_centroid": spec_cent_interp,
        "spectral_flux": spec_flux_interp,
        "is_voiced": ~np.isnan(f0_values),
    })

    logger.info(f"Extracted {len(features)} frames ({features['time'].max():.1f}s)")
    logger.info(f"Voiced frames: {features['is_voiced'].sum()} ({features['is_voiced'].mean()*100:.1f}%)")

    return features


def detect_pauses(
    prosody: pd.DataFrame,
    energy_threshold: Optional[float] = None,
    min_pause_duration: float = 0.2,
) -> pd.DataFrame:
    """
    Detect pauses in speech from prosodic features.

    Parameters
    ----------
    prosody : pd.DataFrame
        Prosody features from extract_prosody_features().
    energy_threshold : float, optional
        Energy threshold for pause detection (RMS).
        If None, uses median - 1.5*MAD.
    min_pause_duration : float
        Minimum pause duration to include (seconds).

    Returns
    -------
    pauses : pd.DataFrame
        DataFrame with columns:
        - pause_id: Pause number
        - start: Pause start time (seconds)
        - end: Pause end time (seconds)
        - duration: Pause duration
        - pre_pause_f0: Mean F0 before pause (for boundary tone analysis)
        - post_pause_f0: Mean F0 after pause

    Notes
    -----
    Pauses indicate potential turn boundaries (TRPs).
    Pre-pause F0 patterns (rising/falling) signal turn-yielding.
    """
    if energy_threshold is None:
        # Robust threshold: median - 1.5 * MAD
        median_energy = prosody["energy"].median()
        mad = np.median(np.abs(prosody["energy"] - median_energy))
        energy_threshold = median_energy - 1.5 * mad
        logger.info(f"Auto-detected energy threshold: {energy_threshold:.4f}")

    # Detect low-energy frames
    is_pause = prosody["energy"] < energy_threshold

    # Find pause boundaries
    pause_starts = np.where(np.diff(is_pause.astype(int)) == 1)[0] + 1
    pause_ends = np.where(np.diff(is_pause.astype(int)) == -1)[0] + 1

    # Handle edge cases
    if is_pause.iloc[0]:
        pause_starts = np.concatenate([[0], pause_starts])
    if is_pause.iloc[-1]:
        pause_ends = np.concatenate([pause_ends, [len(prosody)]])

    # Build pause DataFrame
    pauses = []
    for i, (start_idx, end_idx) in enumerate(zip(pause_starts, pause_ends)):
        start_time = prosody.iloc[start_idx]["time"]
        end_time = prosody.iloc[end_idx - 1]["time"]
        duration = end_time - start_time

        if duration >= min_pause_duration:
            # Get F0 before and after pause
            pre_window = prosody.iloc[max(0, start_idx - 10):start_idx]
            post_window = prosody.iloc[end_idx:min(len(prosody), end_idx + 10)]

            pre_f0 = pre_window["f0"].median() if len(pre_window) > 0 else np.nan
            post_f0 = post_window["f0"].median() if len(post_window) > 0 else np.nan

            pauses.append({
                "pause_id": i,
                "start": start_time,
                "end": end_time,
                "duration": duration,
                "pre_pause_f0": pre_f0,
                "post_pause_f0": post_f0,
            })

    pauses_df = pd.DataFrame(pauses)

    logger.info(f"Detected {len(pauses_df)} pauses (threshold={energy_threshold:.4f}, min_dur={min_pause_duration}s)")

    return pauses_df


def calculate_speech_rate(
    transcript: pd.DataFrame,
    window_size: float = 5.0,
) -> pd.DataFrame:
    """
    Calculate local speech rate from transcript.

    Parameters
    ----------
    transcript : pd.DataFrame
        Transcript with word-level timestamps (from ASR module).
    window_size : float
        Window size for local speech rate (seconds).

    Returns
    -------
    speech_rate : pd.DataFrame
        DataFrame with columns:
        - time: Center of window (seconds)
        - words_per_second: Speech rate (words/sec)
        - syllables_per_second: Estimated syllable rate (if available)

    Notes
    -----
    Speech rate acceleration often precedes turn-yielding.
    Deceleration can signal turn-holding or completion.
    """
    # Extract word times
    word_times = []
    for _, seg in transcript.iterrows():
        words = seg.get("words", [])
        for word in words:
            word_times.append({
                "time": (word["start"] + word["end"]) / 2,
                "word": word.get("word", ""),
            })

    if len(word_times) == 0:
        logger.warning("No words found in transcript")
        return pd.DataFrame(columns=["time", "words_per_second"])

    word_df = pd.DataFrame(word_times)

    # Calculate speech rate in sliding windows
    min_time = word_df["time"].min()
    max_time = word_df["time"].max()

    step = window_size / 2  # 50% overlap
    times = np.arange(min_time, max_time, step)

    speech_rates = []
    for t in times:
        window_start = t - window_size / 2
        window_end = t + window_size / 2

        words_in_window = word_df[(word_df["time"] >= window_start) & (word_df["time"] < window_end)]

        if len(words_in_window) > 0:
            rate = len(words_in_window) / window_size
            speech_rates.append({
                "time": t,
                "words_per_second": rate,
            })

    speech_rate_df = pd.DataFrame(speech_rates)

    logger.info(f"Calculated speech rate at {len(speech_rate_df)} timepoints "
                f"(window={window_size}s)")

    return speech_rate_df


def extract_trp_features(
    prosody: pd.DataFrame,
    pauses: pd.DataFrame,
    speech_rate: pd.DataFrame,
    transcript: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract turn-transition relevance place (TRP) features.

    Parameters
    ----------
    prosody : pd.DataFrame
        Prosodic features from extract_prosody_features().
    pauses : pd.DataFrame
        Pause detections from detect_pauses().
    speech_rate : pd.DataFrame
        Speech rate from calculate_speech_rate().
    transcript : pd.DataFrame
        Transcript from ASR module.

    Returns
    -------
    trp_features : pd.DataFrame
        DataFrame with TRP candidate features:
        - time: TRP time (pause start)
        - duration: Pause duration
        - pre_f0: Mean F0 before pause
        - f0_slope: F0 slope in 500ms before pause
        - pre_intensity: Intensity before pause
        - speech_rate_before: Speech rate before pause
        - is_segment_final: Is this after segment-final word?
        - following_gap: Duration until next speech

    Notes
    -----
    TRPs are potential turn boundaries where:
    - Prosodic cues (falling F0, low intensity, pause) co-occur
    - Syntactic completion (segment-final position)
    - Pragmatic relevance (LLM features to be added later)
    """
    trp_rows = []

    for _, pause in pauses.iterrows():
        pause_start = pause["start"]
        pause_end = pause["end"]

        # Get prosody in pre-pause window (500ms)
        pre_window = prosody[(prosody["time"] >= pause_start - 0.5) &
                              (prosody["time"] < pause_start)]

        if len(pre_window) == 0:
            continue

        # F0 slope (linear regression)
        if pre_window["f0"].notna().sum() > 2:
            voiced = pre_window[pre_window["f0"].notna()]
            if len(voiced) > 1:
                f0_slope = np.polyfit(voiced["time"], voiced["f0"], 1)[0]
            else:
                f0_slope = 0.0
        else:
            f0_slope = 0.0

        # Speech rate before pause
        rate_before = speech_rate[speech_rate["time"] < pause_start]
        if len(rate_before) > 0:
            rate_val = rate_before.iloc[-1]["words_per_second"]
        else:
            rate_val = np.nan

        # Check if pause follows segment-final word
        is_segment_final = False
        for _, seg in transcript.iterrows():
            if abs(seg["end"] - pause_start) < 0.1:  # Within 100ms
                is_segment_final = True
                break

        # Following gap (until next speech)
        post_speech = prosody[(prosody["time"] > pause_end) & (prosody["is_voiced"])]
        if len(post_speech) > 0:
            following_gap = post_speech.iloc[0]["time"] - pause_end
        else:
            following_gap = np.nan

        trp_rows.append({
            "time": pause_start,
            "duration": pause["duration"],
            "pre_f0": pause["pre_pause_f0"],
            "f0_slope": f0_slope,
            "pre_intensity": pre_window["intensity"].mean(),
            "speech_rate_before": rate_val,
            "is_segment_final": is_segment_final,
            "following_gap": following_gap,
        })

    trp_df = pd.DataFrame(trp_rows)

    logger.info(f"Extracted TRP features for {len(trp_df)} candidates")

    return trp_df
