#!/usr/bin/env python3
"""
Visualize dual-microphone channels and speaker turn-taking patterns.
Shows overlay of both channels to identify crossover points.
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import json
from pathlib import Path
import argparse

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


def compute_rms_db(audio, sr, hop_length_ms=10):
    """Compute RMS energy in dB."""
    hop_length = int(hop_length_ms * sr / 1000)
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]

    # Convert to dB
    db = 20 * np.log10(rms + 1e-10)

    # Time axis
    times = np.arange(len(rms)) * hop_length / sr

    return rms, db, times


def identify_speaker_periods(interviewer_db, participant_db, threshold_db=-40):
    """Identify who is speaking at each time point."""
    interviewer_speaking = interviewer_db > threshold_db
    participant_speaking = participant_db > threshold_db

    # Classify each moment
    listener_periods = interviewer_speaking & ~participant_speaking  # Interviewer only
    speaker_periods = participant_speaking & ~interviewer_speaking   # Participant only
    overlap_periods = interviewer_speaking & participant_speaking    # Both
    silence_periods = ~interviewer_speaking & ~participant_speaking  # Neither

    return {
        'listener': listener_periods,
        'speaker': speaker_periods,
        'overlap': overlap_periods,
        'silence': silence_periods
    }


def find_turn_transitions(interviewer_db, participant_db, threshold_db=-40):
    """Find points where the speaker changes."""
    periods = identify_speaker_periods(interviewer_db, participant_db, threshold_db)

    # Find transitions
    listener_to_speaker = np.diff(periods['speaker'].astype(int)) == 1  # Participant starts speaking
    speaker_to_listener = np.diff(periods['listener'].astype(int)) == 1  # Interviewer starts speaking

    return {
        'participant_turns': np.where(listener_to_speaker)[0],
        'interviewer_turns': np.where(speaker_to_listener)[0]
    }


def plot_microphone_overlay(interviewer_db, participant_db, times, threshold_db,
                            start_time=0, duration=60, output_file=None):
    """
    Create detailed overlay plot of microphone channels.

    Parameters
    ----------
    interviewer_db : np.ndarray
        Interviewer energy in dB
    participant_db : np.ndarray
        Participant energy in dB
    times : np.ndarray
        Time axis in seconds
    threshold_db : float
        Speech detection threshold
    start_time : float
        Start time for visualization (seconds)
    duration : float
        Duration to visualize (seconds)
    output_file : Path or None
        Where to save the plot
    """
    # Extract segment
    mask = (times >= start_time) & (times < start_time + duration)
    times_seg = times[mask]
    int_seg = interviewer_db[mask]
    part_seg = participant_db[mask]

    # Identify periods
    periods = identify_speaker_periods(int_seg, part_seg, threshold_db)

    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)

    # 1. Overlay plot
    ax = axes[0]
    ax.plot(times_seg, int_seg, 'b-', linewidth=1.5, label='Interviewer', alpha=0.7)
    ax.plot(times_seg, part_seg, 'r-', linewidth=1.5, label='Participant', alpha=0.7)
    ax.axhline(threshold_db, color='green', linestyle='--', linewidth=2,
               label=f'Threshold ({threshold_db} dB)', alpha=0.5)

    # Mark crossover points
    relative_energy = int_seg - part_seg
    crossovers = np.where(np.diff(np.sign(relative_energy)))[0]
    ax.scatter(times_seg[crossovers], int_seg[crossovers],
              color='purple', s=50, zorder=10, alpha=0.6,
              label='Crossover points', marker='x')

    ax.set_ylabel('Energy (dB)', fontsize=11)
    ax.set_title(f'Microphone Channel Overlay (Threshold: {threshold_db} dB)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-80, -10)

    # 2. Relative energy
    ax = axes[1]
    ax.plot(times_seg, relative_energy, 'purple', linewidth=1.5)
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.fill_between(times_seg, 0, relative_energy, where=(relative_energy > 0),
                     alpha=0.3, color='blue', label='Interviewer louder')
    ax.fill_between(times_seg, 0, relative_energy, where=(relative_energy < 0),
                     alpha=0.3, color='red', label='Participant louder')
    ax.set_ylabel('Relative Energy (dB)', fontsize=11)
    ax.set_title('Interviewer - Participant Energy Difference', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 3. Speaker classification
    ax = axes[2]

    # Create color-coded background
    y_max = 1.0
    for i in range(len(times_seg)):
        if periods['listener'][i]:
            ax.axvspan(times_seg[i], times_seg[min(i+1, len(times_seg)-1)],
                      color='blue', alpha=0.3)
        elif periods['speaker'][i]:
            ax.axvspan(times_seg[i], times_seg[min(i+1, len(times_seg)-1)],
                      color='red', alpha=0.3)
        elif periods['overlap'][i]:
            ax.axvspan(times_seg[i], times_seg[min(i+1, len(times_seg)-1)],
                      color='purple', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.3, label='Interviewer only (listening)'),
        Patch(facecolor='red', alpha=0.3, label='Participant only (speaking)'),
        Patch(facecolor='purple', alpha=0.3, label='Both speaking (overlap)'),
        Patch(facecolor='white', label='Silence')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_ylim(0, y_max)
    ax.set_yticks([])
    ax.set_ylabel('Speaker', fontsize=11)
    ax.set_title('Speaker Classification', fontsize=12)

    # 4. Turn-taking events
    ax = axes[3]

    # Find turn transitions in this segment
    listener_starts = np.diff(periods['listener'].astype(int)) == 1
    speaker_starts = np.diff(periods['speaker'].astype(int)) == 1

    for idx in np.where(listener_starts)[0]:
        ax.axvline(times_seg[idx], color='blue', linestyle='-', alpha=0.5, linewidth=1)
    for idx in np.where(speaker_starts)[0]:
        ax.axvline(times_seg[idx], color='red', linestyle='-', alpha=0.5, linewidth=1)

    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel('Turn Events', fontsize=11)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_title('Turn-Taking Events', fontsize=12)

    legend_elements = [
        plt.Line2D([0], [0], color='blue', linewidth=2, label='Interviewer turn start'),
        plt.Line2D([0], [0], color='red', linewidth=2, label='Participant turn start')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved overlay plot: {output_file}")

    return fig


def create_summary_plots(interviewer_db, participant_db, times, thresholds, output_file):
    """Create summary figure showing multiple threshold comparisons."""
    fig, axes = plt.subplots(len(thresholds), 1, figsize=(16, 3*len(thresholds)), sharex=True)

    if len(thresholds) == 1:
        axes = [axes]

    # Show first 2 minutes for overview
    mask = times < 120
    times_seg = times[mask]
    int_seg = interviewer_db[mask]
    part_seg = participant_db[mask]

    for idx, thresh in enumerate(thresholds):
        ax = axes[idx]

        # Plot channels
        ax.plot(times_seg, int_seg, 'b-', linewidth=0.8, label='Interviewer', alpha=0.6)
        ax.plot(times_seg, part_seg, 'r-', linewidth=0.8, label='Participant', alpha=0.6)
        ax.axhline(thresh, color='green', linestyle='--', linewidth=2, alpha=0.5)

        # Calculate percentages
        periods = identify_speaker_periods(int_seg, part_seg, thresh)
        listen_pct = 100 * np.sum(periods['listener']) / len(int_seg)
        speak_pct = 100 * np.sum(periods['speaker']) / len(int_seg)
        overlap_pct = 100 * np.sum(periods['overlap']) / len(int_seg)
        silence_pct = 100 * np.sum(periods['silence']) / len(int_seg)

        ax.set_ylabel('Energy (dB)', fontsize=10)
        ax.set_title(f'Threshold: {thresh} dB  |  Listening: {listen_pct:.1f}%  |  Speaking: {speak_pct:.1f}%  |  Overlap: {overlap_pct:.1f}%  |  Silence: {silence_pct:.1f}%',
                    fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-80, -10)

        if idx == 0:
            ax.legend(loc='upper right')

    axes[-1].set_xlabel('Time (s)', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved threshold comparison: {output_file}")

    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize speaker turns and microphone crossovers')
    parser.add_argument('--subject', type=str, default='sub-01',
                       help='Subject ID (default: sub-01)')
    parser.add_argument('--run', type=int, default=1,
                       help='Run number (default: 1)')
    parser.add_argument('--threshold', type=float, default=-40,
                       help='Speech detection threshold in dB (default: -40)')
    parser.add_argument('--start-time', type=float, default=0,
                       help='Start time for detailed plot (seconds, default: 0)')
    parser.add_argument('--duration', type=float, default=60,
                       help='Duration for detailed plot (seconds, default: 60)')

    args = parser.parse_args()

    print("="*70)
    print("SPEAKER TURN-TAKING VISUALIZATION")
    print("="*70)
    print(f"Subject: {args.subject}")
    print(f"Run: {args.run}")
    print(f"Threshold: {args.threshold} dB")
    print()

    # Load audio
    print("Loading audio channels...")
    interviewer_audio, participant_audio, sr = load_audio_channels(args.subject, args.run)
    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {len(interviewer_audio) / sr:.1f}s")
    print()

    # Compute RMS and dB
    print("Computing RMS energy...")
    interviewer_rms, interviewer_db, times = compute_rms_db(interviewer_audio, sr)
    participant_rms, participant_db, _ = compute_rms_db(participant_audio, sr)
    print(f"  Frames: {len(interviewer_rms)}")
    print()

    # Create output directory
    output_dir = PIPELINE_DIR / "outputs" / "diagnostics" / args.subject / f"run-{args.run:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate overall statistics
    print("Calculating speaker statistics...")
    periods = identify_speaker_periods(interviewer_db, participant_db, args.threshold)

    listen_pct = 100 * np.sum(periods['listener']) / len(interviewer_db)
    speak_pct = 100 * np.sum(periods['speaker']) / len(participant_db)
    overlap_pct = 100 * np.sum(periods['overlap']) / len(interviewer_db)
    silence_pct = 100 * np.sum(periods['silence']) / len(interviewer_db)

    print(f"\nSpeaker distribution (threshold: {args.threshold} dB):")
    print(f"  Interviewer only (listening): {listen_pct:.1f}%")
    print(f"  Participant only (speaking):  {speak_pct:.1f}%")
    print(f"  Both speaking (overlap):      {overlap_pct:.1f}%")
    print(f"  Silence:                      {silence_pct:.1f}%")

    # Find turn transitions
    transitions = find_turn_transitions(interviewer_db, participant_db, args.threshold)
    print(f"\nTurn-taking events:")
    print(f"  Interviewer turn starts: {len(transitions['interviewer_turns'])}")
    print(f"  Participant turn starts: {len(transitions['participant_turns'])}")
    print()

    # Create detailed overlay plot for specified time segment
    print(f"Creating detailed overlay plot ({args.start_time}s - {args.start_time + args.duration}s)...")
    overlay_file = output_dir / f"speaker_overlay_{args.start_time:.0f}s-{args.start_time + args.duration:.0f}s.png"
    plot_microphone_overlay(interviewer_db, participant_db, times, args.threshold,
                           start_time=args.start_time, duration=args.duration,
                           output_file=overlay_file)

    # Create threshold comparison plots
    print("\nCreating threshold comparison plots...")
    comparison_thresholds = [-60, -54, -50, -45, -40, -35, -30]
    comparison_file = output_dir / "threshold_comparison_overlay.png"
    create_summary_plots(interviewer_db, participant_db, times,
                        comparison_thresholds, comparison_file)

    print(f"\n✓ Visualization complete! See: {output_dir}")
    print()


if __name__ == '__main__':
    main()
