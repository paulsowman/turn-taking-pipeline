#!/usr/bin/env python3
"""
Test synchronization with corrected channel mapping:
- console_mic audio files should correlate with MISC 007 (console/interviewer mic)
- subject_mic audio files should correlate with MISC 008 (subject mic)

This tests a few subjects with poor quality to see if channel correction helps.
"""

import numpy as np
import mne
import librosa
from scipy import signal
from scipy.signal import hilbert, correlate
from pathlib import Path
import pandas as pd

# Subject to group mapping
SUBJECT_TO_GROUP = {
    'sub-01': 'G01', 'sub-02': 'G02', 'sub-03': 'G03', 'sub-04': 'G04',
    'sub-05': 'G05', 'sub-06': 'G06', 'sub-07': 'G07', 'sub-08': 'G08',
    'sub-09': 'G09', 'sub-10': 'G10', 'sub-11': 'G11', 'sub-13': 'G13',
    'sub-14': 'G14', 'sub-15': 'G15', 'sub-16': 'G16', 'sub-17': 'G17',
    'sub-18': 'G18', 'sub-19': 'G19', 'sub-21': 'G21', 'sub-22': 'G22',
    'sub-23': 'G23', 'sub-24': 'G24', 'sub-25': 'G25', 'sub-26': 'G26',
    'sub-27': 'G27', 'sub-29': 'G29', 'sub-31': 'G31', 'sub-32': 'G32',
}

MEG_BASE = Path("/Users/em18033/Library/CloudStorage/OneDrive-SharedLibraries-MacquarieUniversity/Natural_Conversations_study - Documents/analysis/natural-conversations-bids/derivatives/mne-bids-pipeline-20240628")
AUDIO_BASE = Path("/Users/em18033/Library/CloudStorage/OneDrive-AUTUniversity/Projects/Conversational_AI/Archive/data/audios")


def compute_sync_offset(meg_raw, ext_audio_path, aux_channel="MISC 007", max_lag_s=30.0):
    """
    Compute synchronization offset using envelope cross-correlation.

    Returns:
        dict with 'offset_s' and 'correlation' keys, or error dict
    """
    try:
        # Extract MEG audio
        meg_sfreq = meg_raw.info['sfreq']
        meg_audio = meg_raw.copy().pick_channels([aux_channel]).get_data()[0]

        # Load external audio
        ext_audio, ext_sfreq = librosa.load(str(ext_audio_path), sr=None, mono=True)

        # Preprocessing parameters
        target_sr = 1000.0
        lowpass_freq = 400
        env_lowpass = 15

        # Filter MEG
        meg_nyq = meg_sfreq / 2
        sos = signal.butter(4, lowpass_freq / meg_nyq, btype="low", output="sos")
        meg_filtered = signal.sosfilt(sos, meg_audio)

        # Filter and resample external
        ext_nyq = ext_sfreq / 2
        sos = signal.butter(4, lowpass_freq / ext_nyq, btype="low", output="sos")
        ext_filtered = signal.sosfilt(sos, ext_audio)
        ext_resampled = librosa.resample(ext_filtered, orig_sr=ext_sfreq, target_sr=target_sr)

        # Compute envelopes
        meg_env = np.abs(hilbert(meg_filtered))
        ext_env = np.abs(hilbert(ext_resampled))

        # Lowpass filter envelopes
        nyq = target_sr / 2
        sos_env = signal.butter(3, env_lowpass / nyq, btype="low", output="sos")
        meg_env_filt = signal.sosfilt(sos_env, meg_env)
        ext_env_filt = signal.sosfilt(sos_env, ext_env)

        # Normalize
        meg_norm = (meg_env_filt - np.mean(meg_env_filt)) / np.std(meg_env_filt)
        ext_norm = (ext_env_filt - np.mean(ext_env_filt)) / np.std(ext_env_filt)

        # Cross-correlation with realistic search window
        xcorr = correlate(meg_norm, ext_norm, mode="full", method="fft")
        lags = np.arange(len(xcorr)) - (len(ext_norm) - 1)
        lag_times = lags / target_sr

        # Search in ±max_lag_s window
        max_lag_samples = int(max_lag_s * target_sr)
        zero_lag_idx = len(ext_norm) - 1

        search_start = max(0, zero_lag_idx - max_lag_samples)
        search_end = min(len(xcorr), zero_lag_idx + max_lag_samples + 1)

        search_region = xcorr[search_start:search_end]
        peak_idx_local = np.argmax(search_region)
        peak_idx_global = search_start + peak_idx_local

        offset_s = lag_times[peak_idx_global]
        correlation = xcorr[peak_idx_global] / len(ext_norm)

        return {
            'offset_s': float(offset_s),
            'correlation': float(correlation),
        }
    except Exception as e:
        return {'error': str(e)}


def test_subject_run(subject, run):
    """Test a subject/run with corrected channel mapping."""
    group = SUBJECT_TO_GROUP.get(subject)
    if not group:
        return None

    meg_file = MEG_BASE / subject / "meg" / f"{subject}_task-conversation_run-{run:02d}_proc-clean_raw.fif"
    console_audio = AUDIO_BASE / group / f"console_mic_B{run}.wav"

    # Check files exist
    if not meg_file.exists() or not console_audio.exists():
        return None

    # Load MEG
    try:
        meg_raw = mne.io.read_raw_fif(meg_file, preload=True, verbose=False)
    except:
        return None

    # CORRECTED: console_mic audio correlates with MISC 007 (console/interviewer mic)
    result_007 = compute_sync_offset(meg_raw, console_audio, aux_channel="MISC 007")

    # Original (incorrect): was testing console_mic against MISC 008
    result_008 = compute_sync_offset(meg_raw, console_audio, aux_channel="MISC 008")

    return {
        'subject': subject,
        'run': run,
        'MISC_007_correlation': result_007.get('correlation', np.nan),
        'MISC_007_offset': result_007.get('offset_s', np.nan),
        'MISC_008_correlation': result_008.get('correlation', np.nan),
        'MISC_008_offset': result_008.get('offset_s', np.nan),
    }


# Test subjects with poor quality from dual-source run
test_cases = [
    ('sub-11', 1), ('sub-11', 2), ('sub-11', 3),
    ('sub-14', 1), ('sub-14', 2),
    ('sub-15', 1), ('sub-15', 2), ('sub-15', 3),
    ('sub-16', 2),  # This was the only one where MISC 008 won
    ('sub-17', 1), ('sub-17', 2),
    ('sub-05', 1),  # Good control case
]

print("=" * 80)
print("TESTING CORRECTED CHANNEL MAPPING")
print("=" * 80)
print("Channel mapping (from main.py):")
print("  MISC 007 = console/interviewer mic")
print("  MISC 008 = subject mic")
print()
print("Testing: console_mic audio files → should correlate with MISC 007")
print("=" * 80)
print()

results = []
for subject, run in test_cases:
    print(f"{subject} run-{run}:", end=" ", flush=True)
    result = test_subject_run(subject, run)
    if result:
        results.append(result)
        corr_007 = result['MISC_007_correlation']
        corr_008 = result['MISC_008_correlation']

        if np.isnan(corr_007):
            print("ERROR")
        else:
            winner = "MISC 007" if abs(corr_007) >= abs(corr_008) else "MISC 008"
            marker_007 = "→" if winner == "MISC 007" else " "
            marker_008 = "→" if winner == "MISC 008" else " "

            print(f"{marker_007} MISC 007: {corr_007:.3f}  {marker_008} MISC 008: {corr_008:.3f}")
    else:
        print("SKIP")

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

df = pd.DataFrame(results)

# Compare
print(f"\nMean MISC 007 correlation: {df['MISC_007_correlation'].mean():.3f}")
print(f"Mean MISC 008 correlation: {df['MISC_008_correlation'].mean():.3f}")
print()

print("Cases where MISC 007 wins (expected - correct channel):")
wins_007 = df[df['MISC_007_correlation'].abs() >= df['MISC_008_correlation'].abs()]
print(f"  Count: {len(wins_007)} / {len(df)}")
print()

print("Cases where MISC 008 wins (unexpected - wrong channel):")
wins_008 = df[df['MISC_007_correlation'].abs() < df['MISC_008_correlation'].abs()]
if len(wins_008) > 0:
    print(wins_008[['subject', 'run', 'MISC_007_correlation', 'MISC_008_correlation']].to_string(index=False))
else:
    print("  None")

print()
print("=" * 80)
