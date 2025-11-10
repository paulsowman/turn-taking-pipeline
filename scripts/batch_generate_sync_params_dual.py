#!/usr/bin/env python3
"""
Batch script to generate audio-MEG synchronization parameters using BOTH audio sources.
Tries both subject mic (MISC 007) and console mic (MISC 008), selects the best correlation.
Creates a comparison table showing both results.
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
import pandas as pd
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
    try:
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
    except Exception as e:
        return {'error': str(e)}


def generate_sync_params_dual(subject, run, force=False):
    """
    Generate sync parameters trying both subject mic and console mic.

    Returns:
        dict with results for both sources
    """
    # Get paths
    group = SUBJECT_TO_GROUP.get(subject)
    if not group:
        return {'error': f"Unknown subject {subject}"}

    meg_file = MEG_BASE / subject / "meg" / f"{subject}_task-conversation_run-{run:02d}_proc-clean_raw.fif"
    console_audio_file = AUDIO_BASE / group / f"console_mic_B{run}.wav"
    subject_audio_file = AUDIO_BASE / group / f"subject_mic_B{run}.wav"
    output_dir = OUTPUT_BASE / subject / f"run-{run:02d}"
    output_file = output_dir / "sync_params.json"

    result = {
        'subject': subject,
        'run': run,
        'console_meg_console_audio': {},  # MISC 007 + console_mic
        'subject_meg_subject_audio': {},  # MISC 008 + subject_mic
        'selected': None,
        'output_file': str(output_file)
    }

    # Check if already exists
    if output_file.exists() and not force:
        result['skipped'] = True
        return result

    # Check if files exist
    if not meg_file.exists():
        result['error'] = f"MEG file not found"
        return result
    if not console_audio_file.exists():
        result['error'] = f"Console audio file not found"
        return result
    if not subject_audio_file.exists():
        result['error'] = f"Subject audio file not found"
        return result

    # Load MEG
    try:
        meg_raw = mne.io.read_raw_fif(meg_file, preload=True, verbose=False)
    except Exception as e:
        result['error'] = f"Failed to load MEG: {e}"
        return result

    # Try CORRECT pairing 1: MISC 007 (console/interviewer MEG) + console_mic audio
    print(f"    Trying MISC 007 (console/interviewer) + console_mic audio...")
    console_meg_result = compute_sync_offset(meg_raw, console_audio_file, aux_channel="MISC 007")
    result['console_meg_console_audio'] = console_meg_result

    # Try CORRECT pairing 2: MISC 008 (subject MEG) + subject_mic audio
    print(f"    Trying MISC 008 (subject) + subject_mic audio...")
    subject_meg_result = compute_sync_offset(meg_raw, subject_audio_file, aux_channel="MISC 008")
    result['subject_meg_subject_audio'] = subject_meg_result

    # Select best one based on correlation
    if 'error' in console_meg_result and 'error' in subject_meg_result:
        result['error'] = "Both pairings failed"
        return result
    elif 'error' in console_meg_result:
        result['selected'] = 'subject_meg_subject_audio'
        best_result = subject_meg_result
        best_channel = "MISC 008"
    elif 'error' in subject_meg_result:
        result['selected'] = 'console_meg_console_audio'
        best_result = console_meg_result
        best_channel = "MISC 007"
    else:
        # Compare correlations
        console_corr = abs(console_meg_result['correlation'])
        subject_corr = abs(subject_meg_result['correlation'])

        if console_corr >= subject_corr:
            result['selected'] = 'console_meg_console_audio'
            best_result = console_meg_result
            best_channel = "MISC 007"
        else:
            result['selected'] = 'subject_meg_subject_audio'
            best_result = subject_meg_result
            best_channel = "MISC 008"

    # Assess quality
    correlation = best_result['correlation']
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
        "initial_offset_s": best_result['offset_s'],
        "drift_model": "linear",
        "drift_coefficients": {
            "a": 0.0,
            "b": best_result['offset_s']
        },
        "window_stats": [],
        "qc_metrics": {
            "median_error_ms": 0.0,
            "max_error_ms": 0.0,
            "drift_rate_ppm": 0.0,
            "sync_quality": quality,
            "n_windows": 0,
            "peak_correlation": correlation,
            "alignment_strategy": "Envelope cross-correlation (dual source, correct pairings, power-based channel selection)",
            "selected_channel": best_channel,
            "console_meg_console_audio_correlation": console_meg_result.get('correlation', None),
            "subject_meg_subject_audio_correlation": subject_meg_result.get('correlation', None),
            "selected_audio_channel": best_result.get('selected_audio_channel', 0),
        },
        "meg_sfreq": best_result['meg_sfreq'],
        "ext_sfreq": best_result['ext_sfreq'],
        "aux_channel": best_channel,
        "meg_duration_s": best_result['meg_duration_s'],
        "ext_duration_s": best_result['ext_duration_s'],
        "generated_at": datetime.now().isoformat(),
    }

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(sync_params, f, indent=2)

    result['saved'] = True
    result['quality'] = quality

    return result


def main():
    parser = argparse.ArgumentParser(description="Batch generate sync parameters with dual source comparison")
    parser.add_argument('--subjects', nargs='+', help='Subjects to process (default: all)')
    parser.add_argument('--runs', nargs='+', type=int, default=[1, 2, 3, 4, 5], help='Runs to process')
    parser.add_argument('--force', action='store_true', help='Overwrite existing files')
    parser.add_argument('--output-table', default='outputs/sync_comparison_table.csv', help='Output CSV table path')
    args = parser.parse_args()

    # Determine subjects
    if args.subjects:
        subjects = args.subjects
    else:
        subjects = sorted(SUBJECT_TO_GROUP.keys())

    print("=" * 80)
    print("BATCH SYNC PARAMETER GENERATION - DUAL SOURCE")
    print("=" * 80)
    print(f"Subjects: {len(subjects)}")
    print(f"Runs: {args.runs}")
    print(f"Total tasks: {len(subjects) * len(args.runs)}")
    print(f"Force overwrite: {args.force}")
    print("=" * 80)
    print()

    # Process each subject/run
    results = []

    for subject in subjects:
        print(f"\n{subject}:")
        for run in args.runs:
            print(f"  Run {run}:")
            result = generate_sync_params_dual(subject, run, force=args.force)
            results.append(result)

            if 'skipped' in result:
                print(f"    ○ Skipped (already exists)")
            elif 'error' in result:
                print(f"    ✗ Error: {result['error']}")
            elif 'saved' in result:
                console_corr = result['console_meg_console_audio'].get('correlation', 'N/A')
                subject_corr = result['subject_meg_subject_audio'].get('correlation', 'N/A')
                selected = result['selected']

                if isinstance(console_corr, float):
                    console_str = f"{console_corr:.3f}"
                else:
                    console_str = "ERROR"

                if isinstance(subject_corr, float):
                    subject_str = f"{subject_corr:.3f}"
                else:
                    subject_str = "ERROR"

                marker = "→" if selected == 'console_meg_console_audio' else " "
                print(f"    {marker} MISC 007 + console_mic: {console_str}")
                marker = "→" if selected == 'subject_meg_subject_audio' else " "
                print(f"    {marker} MISC 008 + subject_mic: {subject_str}")
                print(f"    ✓ Selected: {selected} ({result['quality']})")

    # Create comparison table
    print()
    print("=" * 80)
    print("Creating comparison table...")
    print("=" * 80)

    table_data = []
    for r in results:
        if 'error' in r or 'skipped' in r:
            continue

        row = {
            'Subject': r['subject'],
            'Run': r['run'],
            'Console_MEG_Console_Audio_Correlation': r['console_meg_console_audio'].get('correlation', np.nan),
            'Console_MEG_Console_Audio_Offset_s': r['console_meg_console_audio'].get('offset_s', np.nan),
            'Subject_MEG_Subject_Audio_Correlation': r['subject_meg_subject_audio'].get('correlation', np.nan),
            'Subject_MEG_Subject_Audio_Offset_s': r['subject_meg_subject_audio'].get('offset_s', np.nan),
            'Selected_Pairing': r['selected'],
            'Final_Correlation': r['console_meg_console_audio'].get('correlation') if r['selected'] == 'console_meg_console_audio' else r['subject_meg_subject_audio'].get('correlation'),
            'Final_Offset_s': r['console_meg_console_audio'].get('offset_s') if r['selected'] == 'console_meg_console_audio' else r['subject_meg_subject_audio'].get('offset_s'),
            'Quality': r.get('quality', 'unknown')
        }
        table_data.append(row)

    df = pd.DataFrame(table_data)

    # Save table
    output_table = Path(args.output_table)
    output_table.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_table, index=False)

    print(f"✓ Comparison table saved to: {output_table}")
    print()

    # Print summary statistics
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total processed: {len(table_data)}")
    print(f"MISC 007 + console_mic selected: {(df['Selected_Pairing'] == 'console_meg_console_audio').sum()}")
    print(f"MISC 008 + subject_mic selected: {(df['Selected_Pairing'] == 'subject_meg_subject_audio').sum()}")
    print()
    print("Quality breakdown:")
    print(df['Quality'].value_counts().to_string())
    print()
    print("Mean correlations:")
    print(f"  MISC 007 + console_mic: {df['Console_MEG_Console_Audio_Correlation'].mean():.3f}")
    print(f"  MISC 008 + subject_mic: {df['Subject_MEG_Subject_Audio_Correlation'].mean():.3f}")
    print(f"  Final (best): {df['Final_Correlation'].mean():.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
