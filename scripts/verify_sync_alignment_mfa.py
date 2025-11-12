#!/usr/bin/env python3
"""
Verify Audio-MEG Synchronization Alignment Using MFA Timing

Same as verify_sync_alignment.py but uses MFA word onsets for improved accuracy.

Creates a multi-panel plot showing:
- Panel 1: Audio envelope with MFA word onset markers
- Panel 2: F0 contour with MFA word onset markers
- Panel 3: MEG auxiliary channel envelope vs Audio envelope (alignment check)
- Panel 4: Zoomed view of a speech segment with MFA word onsets

MFA provides ±10-20ms timing accuracy vs Whisper's ±50-200ms.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import librosa
import mne
import json
import argparse

# Add src to path
try:
    from utils.config import load_config, get_subject_paths
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from utils.config import load_config, get_subject_paths


def load_mfa_word_onsets(subject, run, base_dir):
    """Load MFA word onset times."""
    feature_dir = base_dir / "outputs" / "features" / subject / f"run-{run:02d}"
    mfa_file = feature_dir / "transcript_mfa.csv"

    if not mfa_file.exists():
        raise FileNotFoundError(
            f"MFA transcript not found: {mfa_file}\n"
            f"Run: python scripts/run_mfa_alignment.py --subject {subject} --run {run}"
        )

    mfa_df = pd.read_csv(mfa_file)

    # MFA CSV has columns: start, end, duration, word
    # And if MEG-aligned: start_meg, end_meg, duration_meg
    words_list = []
    for _, row in mfa_df.iterrows():
        words_list.append({
            'time_audio': row['start'],
            'time_meg': row.get('start_meg', row['start']),  # Use MEG time if available
            'word': row['word']
        })

    return pd.DataFrame(words_list)


def compute_envelope_f0(audio, sr, hop_length_ms=10):
    """Compute envelope and F0 in audio timebase."""
    hop_length = int(hop_length_ms * sr / 1000)

    # Envelope (RMS)
    envelope = librosa.feature.rms(
        y=audio,
        frame_length=int(0.025 * sr),
        hop_length=hop_length
    )[0]
    envelope_times = np.arange(len(envelope)) * hop_length / sr

    # F0
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        sr=sr,
        fmin=80,
        fmax=400,
        hop_length=hop_length,
        frame_length=int(0.025 * sr),
        fill_na=0.0
    )
    f0_times = np.arange(len(f0)) * hop_length / sr

    return envelope, envelope_times, f0, f0_times


def verify_sync_alignment_mfa(subject, run, base_dir, time_window=None, zoom_window=None):
    """
    Create diagnostic visualization using MFA word timing.

    Parameters
    ----------
    subject : str
        Subject ID (e.g., 'sub-01')
    run : int
        Run number (1-5)
    base_dir : Path
        Base directory of project
    time_window : tuple, optional
        (start, end) time in seconds to plot (MEG time). Default: first 60s
    zoom_window : tuple, optional
        (start, end) time in seconds for zoomed panel. Default: auto-select speech segment
    """
    print(f"\n{'='*70}")
    print(f"VERIFYING SYNC ALIGNMENT (MFA): {subject} run-{run:02d}")
    print(f"{'='*70}\n")

    # Load config and paths
    config = load_config()
    paths = get_subject_paths(subject, run, config)

    # Load sync parameters
    sync_dir = base_dir / "outputs" / "sync" / subject / f"run-{run:02d}"
    sync_params_file = sync_dir / "sync_params.json"

    if not sync_params_file.exists():
        raise FileNotFoundError(f"Sync params not found: {sync_params_file}")

    with open(sync_params_file, 'r') as f:
        sync_params = json.load(f)

    sync_offset = sync_params['initial_offset_s']
    print(f"Sync offset: {sync_offset:.4f} s")
    print(f"  Formula: meg_time = audio_time - offset")
    print(f"           meg_time = audio_time - ({sync_offset:.3f})\n")

    # Load MEG data
    print("Loading MEG data...")
    meg_raw = mne.io.read_raw_fif(paths['meg_raw'], preload=True, verbose=False)
    meg_sfreq = meg_raw.info['sfreq']
    meg_times = meg_raw.times

    # Get auxiliary channel
    aux_channel = config["sync"].get("aux_channel_name", "MISC 007")
    if aux_channel in meg_raw.ch_names:
        aux_data = meg_raw[aux_channel][0][0]
        # Compute envelope from MEG aux channel for comparison
        aux_envelope = np.abs(aux_data)
        # Smooth with same window as audio envelope (~10ms)
        from scipy.ndimage import gaussian_filter1d
        sigma = int(0.01 * meg_sfreq)  # 10ms in samples
        aux_envelope_smooth = gaussian_filter1d(aux_envelope, sigma)
    else:
        print(f"Warning: Aux channel {aux_channel} not found")
        aux_data = np.zeros_like(meg_times)
        aux_envelope_smooth = np.zeros_like(meg_times)

    # Load audio
    print("Loading audio...")
    audio_file = paths['external_audio_interviewer']
    # Use channel 0 (left) for console_mic files - interviewer only
    sys.path.insert(0, str(base_dir / "src"))
    from utils.io import load_audio
    audio, sr = load_audio(audio_file, sr=None, channel=0)

    # Compute audio features in audio timebase
    print("Computing audio features...")
    envelope, envelope_times_audio, f0, f0_times_audio = compute_envelope_f0(audio, sr)

    # Convert to MEG timebase
    envelope_times_meg = envelope_times_audio - sync_offset
    f0_times_meg = f0_times_audio - sync_offset

    print(f"  Audio timebase: 0 to {envelope_times_audio[-1]:.2f}s")
    print(f"  MEG timebase:   {envelope_times_meg[0]:.2f} to {envelope_times_meg[-1]:.2f}s")

    # Load MFA word onsets
    print("Loading MFA word onsets...")
    words_df = load_mfa_word_onsets(subject, run, base_dir)
    print(f"  Found {len(words_df)} words")
    print(f"  ✓ Using MFA timing (±10-20ms accuracy)\n")

    # Determine time window
    if time_window is None:
        time_window = (0, min(60, meg_times[-1]))

    t_start, t_end = time_window

    # Auto-select zoom window if not specified
    if zoom_window is None:
        # Find a segment with multiple words
        words_in_window = words_df[
            (words_df['time_meg'] >= t_start + 10) &
            (words_df['time_meg'] <= t_start + 20)
        ]
        if len(words_in_window) > 0:
            first_word = words_in_window.iloc[0]['time_meg']
            zoom_window = (first_word - 0.5, first_word + 2.5)
        else:
            zoom_window = (t_start + 10, t_start + 13)

    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    fig.suptitle(f'Sync Alignment Verification (MFA): {subject} run-{run:02d}\n' +
                 f'Sync offset = {sync_offset:.4f}s | MFA timing ±10-20ms accuracy',
                 fontsize=14, fontweight='bold')

    # Panel 1: Envelope with word onsets
    ax = axes[0]
    mask = (envelope_times_meg >= t_start) & (envelope_times_meg <= t_end)
    ax.plot(envelope_times_meg[mask], envelope[mask], 'b-', linewidth=0.5, label='Envelope')

    words_in_range = words_df[(words_df['time_meg'] >= t_start) & (words_df['time_meg'] <= t_end)]
    for _, word in words_in_range.iterrows():
        ax.axvline(word['time_meg'], color='red', alpha=0.3, linewidth=0.8)

    ax.set_ylabel('Envelope\n(RMS)', fontsize=10)
    ax.set_xlim(t_start, t_end)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_title('Audio Envelope + MFA Word Onsets (red lines)', fontsize=10)

    # Panel 2: F0 with word onsets
    ax = axes[1]
    mask = (f0_times_meg >= t_start) & (f0_times_meg <= t_end)
    f0_plot = f0[mask].copy()
    f0_plot[f0_plot == 0] = np.nan  # Don't plot unvoiced segments
    ax.plot(f0_times_meg[mask], f0_plot, 'g-', linewidth=0.8, label='F0')

    for _, word in words_in_range.iterrows():
        ax.axvline(word['time_meg'], color='red', alpha=0.3, linewidth=0.8)

    ax.set_ylabel('F0 (Hz)', fontsize=10)
    ax.set_xlim(t_start, t_end)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_title('F0 Contour + MFA Word Onsets (red lines)', fontsize=10)

    # Panel 3: MEG auxiliary channel envelope vs Audio envelope (alignment check)
    ax = axes[2]

    # Plot MEG aux envelope (already in MEG time)
    mask_meg = (meg_times >= t_start) & (meg_times <= t_end)
    ax.plot(meg_times[mask_meg], aux_envelope_smooth[mask_meg],
            'k-', linewidth=1, alpha=0.7, label='MEG Aux Envelope')

    # Plot audio envelope (converted to MEG time)
    mask_audio = (envelope_times_meg >= t_start) & (envelope_times_meg <= t_end)
    # Normalize to match scale for comparison
    if len(envelope[mask_audio]) > 0 and len(aux_envelope_smooth[mask_meg]) > 0:
        env_scaled = envelope[mask_audio] / np.max(envelope[mask_audio]) * np.max(aux_envelope_smooth[mask_meg])
        ax.plot(envelope_times_meg[mask_audio], env_scaled,
                'b-', linewidth=1, alpha=0.5, label='Audio Envelope (scaled)')

    for _, word in words_in_range.iterrows():
        ax.axvline(word['time_meg'], color='red', alpha=0.3, linewidth=0.8)

    ax.set_ylabel('Envelope\n(normalized)', fontsize=10)
    ax.set_xlim(t_start, t_end)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f'Alignment Check: MEG Aux vs Audio Envelope (offset={sync_offset:.3f}s)', fontsize=10)

    # Panel 4: Zoomed view with word labels
    ax = axes[3]
    z_start, z_end = zoom_window

    # Plot envelope
    mask = (envelope_times_meg >= z_start) & (envelope_times_meg <= z_end)
    ax.plot(envelope_times_meg[mask], envelope[mask], 'b-', linewidth=1, label='Envelope')

    # Plot F0 (scaled to fit)
    mask = (f0_times_meg >= z_start) & (f0_times_meg <= z_end)
    f0_plot = f0[mask].copy()
    f0_plot[f0_plot == 0] = np.nan
    if not np.all(np.isnan(f0_plot)):
        f0_scaled = (f0_plot - np.nanmin(f0_plot)) / (np.nanmax(f0_plot) - np.nanmin(f0_plot) + 1e-6)
        mask_env = (envelope_times_meg >= z_start) & (envelope_times_meg <= z_end)
        if len(envelope[mask_env]) > 0:
            f0_scaled *= envelope[mask_env].max() * 0.8
            ax.plot(f0_times_meg[mask], f0_scaled, 'g-', linewidth=1, alpha=0.7, label='F0 (scaled)')

    # Plot word onsets with labels
    words_in_zoom = words_df[(words_df['time_meg'] >= z_start) & (words_df['time_meg'] <= z_end)]
    mask_env = (envelope_times_meg >= z_start) & (envelope_times_meg <= z_end)
    y_max = envelope[mask_env].max() if len(envelope[mask_env]) > 0 else 1

    for idx, (_, word) in enumerate(words_in_zoom.iterrows()):
        ax.axvline(word['time_meg'], color='red', alpha=0.5, linewidth=1.5)
        # Alternate label heights
        y_pos = y_max * (0.9 if idx % 2 == 0 else 0.7)
        ax.text(word['time_meg'], y_pos, word['word'],
                rotation=90, fontsize=8, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    ax.set_ylabel('Amplitude', fontsize=10)
    ax.set_xlabel('Time (s, MEG timebase)', fontsize=10)
    ax.set_xlim(z_start, z_end)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_title(f'Zoomed View (MFA): {z_start:.2f}s - {z_end:.2f}s (MEG time)', fontsize=10)

    plt.tight_layout()

    # Save figure
    output_dir = base_dir / "outputs" / "sync" / subject / f"run-{run:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "sync_verification_mfa.png"

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")

    # Print diagnostic info
    print(f"\n{'='*70}")
    print("DIAGNOSTIC CHECKS (MFA):")
    print(f"{'='*70}")

    # Check 1: Time range coverage
    print(f"\n1. Time Range Coverage:")
    print(f"   MEG duration:      {meg_times[-1]:.2f}s")
    print(f"   Audio duration:    {len(audio)/sr:.2f}s")
    print(f"   Envelope range (MEG time):  {envelope_times_meg[0]:.2f}s to {envelope_times_meg[-1]:.2f}s")
    print(f"   MFA word range (MEG time):  {words_df['time_meg'].min():.2f}s to {words_df['time_meg'].max():.2f}s")

    # Check 2: Word onset alignment
    print(f"\n2. MFA Word Onset Alignment:")
    print(f"   Total words: {len(words_df)}")
    print(f"   Timing accuracy: ±10-20ms (vs Whisper ±50-200ms)")

    # Sample a few words and check if they align with envelope peaks
    sample_words = words_df.sample(min(5, len(words_df)))
    print(f"\n   Sample MFA word onset checks:")
    for _, word in sample_words.iterrows():
        t_meg = word['time_meg']
        t_audio = word['time_audio']

        # Find nearest envelope sample
        idx = np.argmin(np.abs(envelope_times_meg - t_meg))
        env_val = envelope[idx]

        print(f"   '{word['word'][:15]:15s}' @ t_audio={t_audio:6.2f}s, t_meg={t_meg:6.2f}s, env={env_val:.4f}")

    # Check 3: Verify formula consistency
    print(f"\n3. Formula Verification:")
    print(f"   Sync offset: {sync_offset:.4f}s")
    sample_word = words_df.iloc[0]
    t_audio = sample_word['time_audio']
    t_meg = sample_word['time_meg']
    calculated_t_meg = t_audio - sync_offset
    error = abs(t_meg - calculated_t_meg)

    print(f"   First word: '{sample_word['word']}'")
    print(f"   Audio time (MFA): {t_audio:.4f}s")
    print(f"   MEG time (file):  {t_meg:.4f}s")
    print(f"   MEG time (calc):  {calculated_t_meg:.4f}s  [= {t_audio:.4f} - ({sync_offset:.4f})]")
    print(f"   Error:            {error:.6f}s ({error*1000:.3f}ms)")

    if error < 0.001:
        print(f"   ✓ Formula consistent (error < 1ms)")
    else:
        print(f"   ✗ WARNING: Formula mismatch (error = {error*1000:.2f}ms)")

    print(f"\n{'='*70}\n")

    return fig


def main():
    parser = argparse.ArgumentParser(description='Verify audio-MEG synchronization using MFA timing')
    parser.add_argument('--subject', type=str, required=True, help='Subject ID (e.g., sub-01)')
    parser.add_argument('--run', type=int, required=True, help='Run number (1-5)')
    parser.add_argument('--start', type=float, default=None, help='Start time (MEG time, seconds)')
    parser.add_argument('--end', type=float, default=None, help='End time (MEG time, seconds)')
    parser.add_argument('--zoom-start', type=float, default=None, help='Zoom window start (MEG time)')
    parser.add_argument('--zoom-end', type=float, default=None, help='Zoom window end (MEG time)')

    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent

    time_window = None
    if args.start is not None and args.end is not None:
        time_window = (args.start, args.end)

    zoom_window = None
    if args.zoom_start is not None and args.zoom_end is not None:
        zoom_window = (args.zoom_start, args.zoom_end)

    fig = verify_sync_alignment_mfa(
        args.subject,
        args.run,
        base_dir,
        time_window=time_window,
        zoom_window=zoom_window
    )

    plt.show()


if __name__ == '__main__':
    main()
