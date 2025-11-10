#!/usr/bin/env python3
"""
Batch script to generate audio-MEG synchronization parameters for all subjects and runs.
Uses the corrected cross-correlation approach with realistic search window.
"""

import numpy as np
import mne
import librosa
import soundfile as sf
from scipy import signal
from scipy.signal import hilbert, correlate
from pathlib import Path
import json
import argparse
from datetime import datetime

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
OUTPUT_BASE = Path("outputs/sync")


def compute_sync_offset(meg_raw, ext_audio_path, aux_channel="MISC 007", max_lag_s=30.0):
    """
    Compute synchronization offset using envelope cross-correlation.

    Returns:
        dict with 'offset_s' and 'correlation' keys, or None if failed
    """
    # Extract MEG audio
    meg_sfreq = meg_raw.info['sfreq']
    meg_audio = meg_raw.copy().pick_channels([aux_channel]).get_data()[0]

    # Load external audio - use soundfile to preserve stereo
    ext_audio_data, ext_sfreq = sf.read(str(ext_audio_path))

    # Select channel with most power (usually Ch0, but check to be safe)
    if len(ext_audio_data.shape) > 1 and ext_audio_data.shape[1] == 2:
        # Stereo - select channel with higher RMS
        rms_ch0 = np.sqrt(np.mean(ext_audio_data[:, 0]**2))
        rms_ch1 = np.sqrt(np.mean(ext_audio_data[:, 1]**2))

        if rms_ch0 >= rms_ch1:
            ext_audio = ext_audio_data[:, 0]
            selected_channel = 0
        else:
            ext_audio = ext_audio_data[:, 1]
            selected_channel = 1
    else:
        # Mono
        ext_audio = ext_audio_data if ext_audio_data.ndim == 1 else ext_audio_data[:, 0]
        selected_channel = 0

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
        'meg_duration_s': float(len(meg_audio) / meg_sfreq),
        'ext_duration_s': float(len(ext_audio) / ext_sfreq),
        'meg_sfreq': float(meg_sfreq),
        'ext_sfreq': float(ext_sfreq),
        'selected_audio_channel': int(selected_channel),
    }


def generate_sync_params(subject, run, force=False):
    """
    Generate sync parameters for a given subject and run.

    Returns:
        tuple of (success: bool, message: str, correlation: float)
    """
    # Get paths
    group = SUBJECT_TO_GROUP.get(subject)
    if not group:
        return False, f"Unknown subject {subject}", 0.0

    meg_file = MEG_BASE / subject / "meg" / f"{subject}_task-conversation_run-{run:02d}_proc-clean_raw.fif"
    audio_file = AUDIO_BASE / group / f"console_mic_B{run}.wav"
    output_dir = OUTPUT_BASE / subject / f"run-{run:02d}"
    output_file = output_dir / "sync_params.json"

    # Check if already exists
    if output_file.exists() and not force:
        return False, f"Already exists (use --force to overwrite)", 0.0

    # Check if files exist
    if not meg_file.exists():
        return False, f"MEG file not found: {meg_file}", 0.0
    if not audio_file.exists():
        return False, f"Audio file not found: {audio_file}", 0.0

    # Load MEG
    try:
        meg_raw = mne.io.read_raw_fif(meg_file, preload=True, verbose=False)
    except Exception as e:
        return False, f"Failed to load MEG: {e}", 0.0

    # Compute sync
    try:
        result = compute_sync_offset(meg_raw, audio_file)
    except Exception as e:
        return False, f"Sync computation failed: {e}", 0.0

    # Assess quality
    correlation = result['correlation']
    if abs(correlation) > 0.7:
        quality = "excellent"
    elif abs(correlation) > 0.5:
        quality = "good"
    elif abs(correlation) > 0.3:
        quality = "moderate"
    else:
        quality = "poor"

    # Create sync params structure
    sync_params = {
        "initial_offset_s": result['offset_s'],
        "drift_model": "linear",
        "drift_coefficients": {
            "a": 0.0,
            "b": result['offset_s']
        },
        "window_stats": [],
        "qc_metrics": {
            "median_error_ms": 0.0,
            "max_error_ms": 0.0,
            "drift_rate_ppm": 0.0,
            "sync_quality": quality,
            "n_windows": 0,
            "peak_correlation": correlation,
            "alignment_strategy": "Envelope cross-correlation (power-based channel selection)",
            "selected_audio_channel": result['selected_audio_channel']
        },
        "meg_sfreq": result['meg_sfreq'],
        "ext_sfreq": result['ext_sfreq'],
        "aux_channel": "MISC 007",
        "meg_duration_s": result['meg_duration_s'],
        "ext_duration_s": result['ext_duration_s'],
        "generated_at": datetime.now().isoformat(),
    }

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(sync_params, f, indent=2)

    return True, f"Success ({quality}, corr={correlation:.3f})", correlation


def main():
    parser = argparse.ArgumentParser(description="Batch generate sync parameters")
    parser.add_argument('--subjects', nargs='+', help='Subjects to process (default: all)')
    parser.add_argument('--runs', nargs='+', type=int, default=[1, 2, 3, 4, 5], help='Runs to process')
    parser.add_argument('--force', action='store_true', help='Overwrite existing files')
    parser.add_argument('--min-correlation', type=float, default=0.7, help='Minimum correlation threshold for warning')
    args = parser.parse_args()

    # Determine subjects
    if args.subjects:
        subjects = args.subjects
    else:
        subjects = sorted(SUBJECT_TO_GROUP.keys())

    print("=" * 80)
    print("BATCH SYNC PARAMETER GENERATION")
    print("=" * 80)
    print(f"Subjects: {len(subjects)}")
    print(f"Runs: {args.runs}")
    print(f"Total tasks: {len(subjects) * len(args.runs)}")
    print(f"Force overwrite: {args.force}")
    print(f"Min correlation: {args.min_correlation}")
    print("=" * 80)
    print()

    # Process each subject/run
    total = 0
    success_count = 0
    skip_count = 0
    fail_count = 0
    poor_quality = []

    for subject in subjects:
        print(f"\n{subject}:")
        for run in args.runs:
            total += 1
            success, message, correlation = generate_sync_params(subject, run, force=args.force)

            status = "✓" if success else "✗" if "not found" not in message.lower() and "already exists" not in message.lower() else "○"

            if success:
                success_count += 1
                if abs(correlation) < args.min_correlation:
                    poor_quality.append((subject, run, correlation))
                    print(f"  Run {run}: {status} {message} ⚠️  LOW QUALITY")
                else:
                    print(f"  Run {run}: {status} {message}")
            elif "already exists" in message.lower():
                skip_count += 1
                print(f"  Run {run}: {status} {message}")
            else:
                fail_count += 1
                print(f"  Run {run}: {status} {message}")

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total tasks:      {total}")
    print(f"Successful:       {success_count}")
    print(f"Skipped:          {skip_count}")
    print(f"Failed:           {fail_count}")
    print(f"Poor quality:     {len(poor_quality)}")

    if poor_quality:
        print()
        print("Poor quality syncs (correlation < {:.2f}):".format(args.min_correlation))
        for subject, run, corr in poor_quality:
            print(f"  {subject} run-{run}: corr={corr:.3f}")

    print("=" * 80)


if __name__ == "__main__":
    main()
