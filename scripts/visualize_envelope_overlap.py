#!/usr/bin/env python3
"""
Visualize overlapping speech envelopes (amplitude) to identify speaker periods.
Cleaner than using raw dB levels.
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import json
from pathlib import Path
import argparse
from scipy.ndimage import gaussian_filter1d

# Configuration
PIPELINE_DIR = Path("/Users/em18033/Library/CloudStorage/OneDrive-AUTUniversity/Projects/turn-taking-pipeline")


def load_audio_channels(subject, run):
    """Load both audio channels for a given subject and run."""
    from utils.config import load_config, get_subject_paths

    config = load_config()
    paths = get_subject_paths(subject, run, config)

    # Load both audio channels
    interviewer_audio, sr = librosa.load(paths['external_audio_interviewer'], sr=None)
    participant_audio, _ = librosa.load(paths['external_audio_participant'], sr=sr)

    return interviewer_audio, participant_audio, sr


def compute_smooth_envelope(audio, sr, hop_length_ms=10, smooth_ms=50):
    """
    Compute smooth RMS envelope.

    Parameters
    ----------
    audio : np.ndarray
        Audio signal
    sr : int
        Sample rate
    hop_length_ms : float
        Hop length in milliseconds for RMS computation
    smooth_ms : float
        Gaussian smoothing window in milliseconds

    Returns
    -------
    envelope : np.ndarray
        Smooth RMS envelope
    times : np.ndarray
        Time axis in seconds
    """
    hop_length = int(hop_length_ms * sr / 1000)
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]

    # Apply Gaussian smoothing to reduce noise
    if smooth_ms > 0:
        sigma_samples = smooth_ms / hop_length_ms
        rms = gaussian_filter1d(rms, sigma=sigma_samples)

    # Time axis
    times = np.arange(len(rms)) * hop_length / sr

    return rms, times


def identify_speaker_from_envelopes(interviewer_env, participant_env, method='ratio', threshold=2.0):
    """
    Identify who is speaking based on envelope comparison.

    Parameters
    ----------
    interviewer_env : np.ndarray
        Interviewer RMS envelope
    participant_env : np.ndarray
        Participant RMS envelope
    method : str
        Method for classification:
        - 'ratio': Use ratio of envelopes (interviewer_env / participant_env > threshold)
        - 'difference': Use difference (interviewer_env - participant_env > threshold * median_env)
    threshold : float
        Threshold for classification (default: 2.0 for ratio, meaning 2x louder)

    Returns
    -------
    dict with boolean masks for listener, speaker, overlap, silence periods
    """
    # Add small epsilon to avoid division by zero
    eps = 1e-10

    # Determine overall noise floor (median of quieter moments)
    combined = np.maximum(interviewer_env, participant_env)
    noise_floor = np.percentile(combined, 10)  # 10th percentile as noise floor

    if method == 'ratio':
        # Ratio method: who is louder by factor of threshold
        ratio_int = interviewer_env / (participant_env + eps)
        ratio_part = participant_env / (interviewer_env + eps)

        # Also require minimum absolute level above noise floor
        int_active = (interviewer_env > noise_floor * 2)
        part_active = (participant_env > noise_floor * 2)

        listener_periods = (ratio_int > threshold) & int_active
        speaker_periods = (ratio_part > threshold) & part_active

    elif method == 'difference':
        # Difference method: absolute difference relative to noise floor
        diff_threshold = threshold * noise_floor

        int_active = (interviewer_env > noise_floor * 2)
        part_active = (participant_env > noise_floor * 2)

        listener_periods = (interviewer_env - participant_env > diff_threshold) & int_active
        speaker_periods = (participant_env - interviewer_env > diff_threshold) & part_active

    # Both speaking if both above noise floor but neither dominant
    both_active = (interviewer_env > noise_floor * 2) & (participant_env > noise_floor * 2)
    overlap_periods = both_active & ~listener_periods & ~speaker_periods

    # Silence if both below threshold
    silence_periods = ~listener_periods & ~speaker_periods & ~overlap_periods

    return {
        'listener': listener_periods,
        'speaker': speaker_periods,
        'overlap': overlap_periods,
        'silence': silence_periods,
        'noise_floor': noise_floor
    }


def plot_envelope_overlay(interviewer_env, participant_env, times,
                          start_time=0, duration=60,
                          threshold=2.0, method='ratio',
                          output_file=None):
    """Create detailed envelope overlay plot."""

    # Extract segment
    mask = (times >= start_time) & (times < start_time + duration)
    times_seg = times[mask]
    int_env = interviewer_env[mask]
    part_env = participant_env[mask]

    # Identify periods
    periods = identify_speaker_from_envelopes(int_env, part_env, method=method, threshold=threshold)
    noise_floor = periods['noise_floor']

    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)

    # 1. Envelope overlay
    ax = axes[0]
    ax.fill_between(times_seg, 0, int_env, alpha=0.4, color='blue', label='Interviewer envelope')
    ax.fill_between(times_seg, 0, part_env, alpha=0.4, color='red', label='Participant envelope')
    ax.plot(times_seg, int_env, 'b-', linewidth=1, alpha=0.8)
    ax.plot(times_seg, part_env, 'r-', linewidth=1, alpha=0.8)
    ax.axhline(noise_floor, color='gray', linestyle='--', linewidth=1.5,
               alpha=0.6, label=f'Noise floor: {noise_floor:.4f}')
    ax.axhline(noise_floor * 2, color='gray', linestyle=':', linewidth=1.5,
               alpha=0.4, label=f'Activity threshold: {noise_floor*2:.4f}')

    ax.set_ylabel('RMS Amplitude', fontsize=11)
    ax.set_title(f'Speech Envelope Overlap (Method: {method}, Threshold: {threshold})',
                fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(int_env.max(), part_env.max()) * 1.1)

    # 2. Ratio/Difference
    ax = axes[1]
    if method == 'ratio':
        eps = 1e-10
        ratio = interviewer_env / (participant_env + eps)
        ratio_seg = ratio[mask]
        ax.plot(times_seg, ratio_seg, 'purple', linewidth=1.5)
        ax.axhline(threshold, color='blue', linestyle='--', linewidth=2, alpha=0.6,
                  label=f'Interviewer dominant (ratio > {threshold})')
        ax.axhline(1/threshold, color='red', linestyle='--', linewidth=2, alpha=0.6,
                  label=f'Participant dominant (ratio < {1/threshold:.2f})')
        ax.axhline(1, color='black', linestyle='-', linewidth=1)
        ax.set_ylabel('Interviewer / Participant', fontsize=11)
        ax.set_title('Envelope Ratio', fontsize=12)
        ax.set_yscale('log')
        ax.set_ylim(0.1, 10)
    else:
        diff = interviewer_env - participant_env
        diff_seg = diff[mask]
        ax.plot(times_seg, diff_seg, 'purple', linewidth=1.5)
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        diff_threshold = threshold * noise_floor
        ax.axhline(diff_threshold, color='blue', linestyle='--', linewidth=2, alpha=0.6,
                  label=f'Interviewer dominant (diff > {diff_threshold:.4f})')
        ax.axhline(-diff_threshold, color='red', linestyle='--', linewidth=2, alpha=0.6,
                  label=f'Participant dominant (diff < {-diff_threshold:.4f})')
        ax.set_ylabel('Interviewer - Participant', fontsize=11)
        ax.set_title('Envelope Difference', fontsize=12)

    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. Speaker classification
    ax = axes[2]

    # Create color-coded background
    for i in range(len(times_seg)):
        if periods['listener'][i]:
            ax.axvspan(times_seg[i], times_seg[min(i+1, len(times_seg)-1)],
                      color='blue', alpha=0.4)
        elif periods['speaker'][i]:
            ax.axvspan(times_seg[i], times_seg[min(i+1, len(times_seg)-1)],
                      color='red', alpha=0.4)
        elif periods['overlap'][i]:
            ax.axvspan(times_seg[i], times_seg[min(i+1, len(times_seg)-1)],
                      color='purple', alpha=0.4)

    # Calculate percentages
    listen_pct = 100 * np.sum(periods['listener']) / len(int_env)
    speak_pct = 100 * np.sum(periods['speaker']) / len(part_env)
    overlap_pct = 100 * np.sum(periods['overlap']) / len(int_env)
    silence_pct = 100 * np.sum(periods['silence']) / len(int_env)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.4, label=f'Interviewer only: {listen_pct:.1f}%'),
        Patch(facecolor='red', alpha=0.4, label=f'Participant only: {speak_pct:.1f}%'),
        Patch(facecolor='purple', alpha=0.4, label=f'Both speaking: {overlap_pct:.1f}%'),
        Patch(facecolor='white', label=f'Silence: {silence_pct:.1f}%')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel('Speaker', fontsize=11)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_title('Speaker Classification', fontsize=12)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved envelope overlay: {output_file}")

    return fig


def compare_parameters(interviewer_env, participant_env, times, output_file):
    """Compare different threshold and method combinations."""

    # Show first 2 minutes
    mask = times < 120
    times_seg = times[mask]
    int_env = interviewer_env[mask]
    part_env = participant_env[mask]

    # Test different parameters
    configs = [
        ('ratio', 1.5, 'Ratio > 1.5'),
        ('ratio', 2.0, 'Ratio > 2.0'),
        ('ratio', 3.0, 'Ratio > 3.0'),
        ('difference', 1.0, 'Diff > 1.0 × noise'),
        ('difference', 2.0, 'Diff > 2.0 × noise'),
        ('difference', 3.0, 'Diff > 3.0 × noise'),
    ]

    fig, axes = plt.subplots(len(configs), 1, figsize=(16, 2.5*len(configs)), sharex=True)

    for idx, (method, threshold, label) in enumerate(configs):
        ax = axes[idx]

        # Identify periods
        periods = identify_speaker_from_envelopes(int_env, part_env, method=method, threshold=threshold)

        # Plot envelopes
        ax.fill_between(times_seg, 0, int_env, alpha=0.3, color='blue')
        ax.fill_between(times_seg, 0, part_env, alpha=0.3, color='red')
        ax.plot(times_seg, int_env, 'b-', linewidth=0.8, alpha=0.6, label='Interviewer')
        ax.plot(times_seg, part_env, 'r-', linewidth=0.8, alpha=0.6, label='Participant')

        # Calculate percentages
        listen_pct = 100 * np.sum(periods['listener']) / len(int_env)
        speak_pct = 100 * np.sum(periods['speaker']) / len(part_env)
        overlap_pct = 100 * np.sum(periods['overlap']) / len(int_env)
        silence_pct = 100 * np.sum(periods['silence']) / len(int_env)

        ax.set_ylabel('RMS Amp', fontsize=9)
        title = f'{label}  |  Listen: {listen_pct:.1f}%  |  Speak: {speak_pct:.1f}%  |  Overlap: {overlap_pct:.1f}%  |  Silence: {silence_pct:.1f}%'
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(loc='upper right', fontsize=9)

    axes[-1].set_xlabel('Time (s)', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved parameter comparison: {output_file}")

    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize envelope overlap for speaker detection')
    parser.add_argument('--subject', type=str, default='sub-01',
                       help='Subject ID (default: sub-01)')
    parser.add_argument('--run', type=int, default=1,
                       help='Run number (default: 1)')
    parser.add_argument('--method', type=str, default='ratio', choices=['ratio', 'difference'],
                       help='Detection method: ratio or difference (default: ratio)')
    parser.add_argument('--threshold', type=float, default=2.0,
                       help='Threshold for speaker detection (default: 2.0)')
    parser.add_argument('--smooth-ms', type=float, default=50,
                       help='Gaussian smoothing window in ms (default: 50)')
    parser.add_argument('--start-time', type=float, default=0,
                       help='Start time for detailed plot (seconds, default: 0)')
    parser.add_argument('--duration', type=float, default=60,
                       help='Duration for detailed plot (seconds, default: 60)')

    args = parser.parse_args()

    print("="*70)
    print("ENVELOPE OVERLAP VISUALIZATION")
    print("="*70)
    print(f"Subject: {args.subject}")
    print(f"Run: {args.run}")
    print(f"Method: {args.method}")
    print(f"Threshold: {args.threshold}")
    print(f"Smoothing: {args.smooth_ms}ms")
    print()

    # Load audio
    print("Loading audio channels...")
    interviewer_audio, participant_audio, sr = load_audio_channels(args.subject, args.run)
    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {len(interviewer_audio) / sr:.1f}s")
    print()

    # Compute smooth envelopes
    print("Computing smooth envelopes...")
    interviewer_env, times = compute_smooth_envelope(interviewer_audio, sr, smooth_ms=args.smooth_ms)
    participant_env, _ = compute_smooth_envelope(participant_audio, sr, smooth_ms=args.smooth_ms)
    print(f"  Frames: {len(interviewer_env)}")
    print(f"  Smoothing: {args.smooth_ms}ms Gaussian")
    print()

    # Create output directory
    output_dir = PIPELINE_DIR / "outputs" / "diagnostics" / args.subject / f"run-{args.run:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate overall statistics
    print("Calculating speaker statistics...")
    periods = identify_speaker_from_envelopes(interviewer_env, participant_env,
                                              method=args.method, threshold=args.threshold)

    listen_pct = 100 * np.sum(periods['listener']) / len(interviewer_env)
    speak_pct = 100 * np.sum(periods['speaker']) / len(participant_env)
    overlap_pct = 100 * np.sum(periods['overlap']) / len(interviewer_env)
    silence_pct = 100 * np.sum(periods['silence']) / len(interviewer_env)

    print(f"\nSpeaker distribution ({args.method}, threshold={args.threshold}):")
    print(f"  Interviewer only (listening): {listen_pct:.1f}%")
    print(f"  Participant only (speaking):  {speak_pct:.1f}%")
    print(f"  Both speaking (overlap):      {overlap_pct:.1f}%")
    print(f"  Silence:                      {silence_pct:.1f}%")
    print(f"  Noise floor: {periods['noise_floor']:.6f}")
    print()

    # Create detailed overlay plot
    print(f"Creating detailed envelope overlay ({args.start_time}s - {args.start_time + args.duration}s)...")
    overlay_file = output_dir / f"envelope_overlay_{args.start_time:.0f}s-{args.start_time + args.duration:.0f}s.png"
    plot_envelope_overlay(interviewer_env, participant_env, times,
                         start_time=args.start_time, duration=args.duration,
                         threshold=args.threshold, method=args.method,
                         output_file=overlay_file)

    # Create parameter comparison
    print("\nCreating parameter comparison plots...")
    comparison_file = output_dir / "envelope_parameter_comparison.png"
    compare_parameters(interviewer_env, participant_env, times, comparison_file)

    print(f"\n✓ Visualization complete! See: {output_dir}")
    print()


if __name__ == '__main__':
    main()
