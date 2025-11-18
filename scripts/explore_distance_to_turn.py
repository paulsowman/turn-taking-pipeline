#!/usr/bin/env python3
"""
Explore Turn-Taking Predictors

Visualize distance-to-turn and proportion-through-turn predictors
to guide stratification analysis (far vs. near turn boundaries).

Usage:
    python scripts/explore_distance_to_turn.py --subject sub-01 --condition conversation --speaker interviewer
"""

import argparse
import numpy as np
import mne
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Explore distance-to-turn predictor')
    parser.add_argument('--subject', required=True, help='Subject ID (e.g., sub-01)')
    parser.add_argument('--condition', required=True, choices=['conversation', 'nursery_rhyme'],
                       help='Condition')
    parser.add_argument('--speaker', required=True, choices=['participant', 'interviewer'],
                       help='Which speaker to analyze')
    parser.add_argument('--window', type=float, nargs=2, default=[0, 60],
                       help='Time window to visualize (start end in seconds)')

    args = parser.parse_args()

    # Load combined FIF file
    fif_file = Path(f'outputs/combined_runs/{args.subject}/{args.condition}_combined.fif')

    if not fif_file.exists():
        print(f"ERROR: Combined FIF file not found: {fif_file}")
        print("Run combine_runs.py first")
        return 1

    print(f"Loading: {fif_file}")
    raw = mne.io.read_raw_fif(fif_file, preload=True, verbose=False)

    # Get distance-to-turn channel
    distance_channel = f'MISC_distance_to_turn_{args.speaker}'

    if distance_channel not in raw.ch_names:
        print(f"ERROR: Channel {distance_channel} not found")
        print(f"Available MISC channels: {[ch for ch in raw.ch_names if 'MISC' in ch]}")
        return 1

    # Extract distance predictor
    ch_idx = raw.ch_names.index(distance_channel)
    distance_data, times = raw[ch_idx, :]
    distance_data = distance_data[0]  # Shape: (1, n_times) -> (n_times,)

    # Get proportion-through-turn channel
    proportion_channel = f'MISC_proportion_through_turn_{args.speaker}'

    if proportion_channel not in raw.ch_names:
        print(f"WARNING: Channel {proportion_channel} not found")
        print("This predictor may not have been created yet. Run create_trf_fif.py with latest code.")
        proportion_data = None
    else:
        ch_idx_prop = raw.ch_names.index(proportion_channel)
        proportion_data, _ = raw[ch_idx_prop, :]
        proportion_data = proportion_data[0]

    # Get MEG sample rate
    sfreq = raw.info['sfreq']

    # Extract window
    window_start, window_end = args.window
    time_mask = (times >= window_start) & (times <= window_end)
    times_window = times[time_mask]
    distance_window = distance_data[time_mask]

    # Compute statistics
    print("\n" + "="*70)
    print("TURN-TAKING PREDICTOR STATISTICS")
    print("="*70)
    print(f"Speaker: {args.speaker}")
    print(f"Condition: {args.condition}")
    print(f"Sampling rate: {sfreq} Hz")
    print(f"Total duration: {times[-1]:.1f} seconds")

    print(f"\n--- DISTANCE-TO-TURN ---")
    print(f"  Mean: {np.mean(distance_data):.2f} seconds")
    print(f"  Median: {np.median(distance_data):.2f} seconds")
    print(f"  Min: {np.min(distance_data):.2f} seconds")
    print(f"  Max: {np.max(distance_data):.2f} seconds")
    print(f"  Std: {np.std(distance_data):.2f} seconds")

    # Suggested binning thresholds
    print(f"\nFixed-threshold stratification (using distance):")
    print(f"  Close to boundary: distance < 1.0s ({np.sum(distance_data < 1.0) / len(distance_data) * 100:.1f}% of data)")
    print(f"  Far from boundary: distance > 2.0s ({np.sum(distance_data > 2.0) / len(distance_data) * 100:.1f}% of data)")
    print(f"  Medium distance: 1.0s < distance < 2.0s ({np.sum((distance_data >= 1.0) & (distance_data <= 2.0)) / len(distance_data) * 100:.1f}% of data)")

    if proportion_data is not None:
        # Filter out NaN values (periods when not listening)
        valid_proportion = proportion_data[~np.isnan(proportion_data)]

        print(f"\n--- PROPORTION-THROUGH-TURN (RECOMMENDED) ---")
        print(f"  Valid samples (during listening): {len(valid_proportion)} / {len(proportion_data)} ({len(valid_proportion)/len(proportion_data)*100:.1f}%)")
        print(f"  Mean: {np.mean(valid_proportion):.3f}")
        print(f"  Median: {np.median(valid_proportion):.3f}")
        print(f"  Min: {np.min(valid_proportion):.3f}")
        print(f"  Max: {np.max(valid_proportion):.3f}")
        print(f"  Std: {np.std(valid_proportion):.3f}")

        print(f"\nTurn-length normalized stratification (using proportion):")
        print(f"  First half of turns: proportion < 0.5 ({np.sum(valid_proportion < 0.5) / len(valid_proportion) * 100:.1f}% of listening)")
        print(f"  Second half of turns: proportion >= 0.5 ({np.sum(valid_proportion >= 0.5) / len(valid_proportion) * 100:.1f}% of listening)")

        proportion_window = proportion_data[time_mask]
    else:
        proportion_window = None

    # Create visualization
    if proportion_data is not None:
        # Two-column layout: distance (left) vs proportion (right)
        fig, axes = plt.subplots(3, 2, figsize=(16, 10))

        # LEFT COLUMN: Distance-to-turn
        # Panel 1a: Distance over time
        ax = axes[0, 0]
        ax.plot(times_window, distance_window, linewidth=0.5, color='blue', alpha=0.7)
        ax.axhline(1.0, color='red', linestyle='--', label='Close threshold (1.0s)', linewidth=2)
        ax.axhline(2.0, color='green', linestyle='--', label='Far threshold (2.0s)', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Distance to turn (s)')
        ax.set_title(f'Distance-to-Turn Over Time ({args.speaker})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        # Panel 2a: Histogram
        ax = axes[1, 0]
        ax.hist(distance_data, bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax.axvline(1.0, color='red', linestyle='--', label='Close (1.0s)', linewidth=2)
        ax.axvline(2.0, color='green', linestyle='--', label='Far (2.0s)', linewidth=2)
        ax.set_xlabel('Distance to turn (s)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Distance-to-Turn')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Panel 3a: Cumulative distribution
        ax = axes[2, 0]
        sorted_dist = np.sort(distance_data)
        cumulative = np.arange(1, len(sorted_dist) + 1) / len(sorted_dist) * 100
        ax.plot(sorted_dist, cumulative, linewidth=2, color='blue')
        ax.axvline(1.0, color='red', linestyle='--', label='Close (1.0s)', linewidth=2)
        ax.axvline(2.0, color='green', linestyle='--', label='Far (2.0s)', linewidth=2)
        ax.set_xlabel('Distance to turn (s)')
        ax.set_ylabel('Cumulative percentage (%)')
        ax.set_title('Cumulative Distribution of Distance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])

        # RIGHT COLUMN: Proportion-through-turn
        valid_proportion = proportion_data[~np.isnan(proportion_data)]

        # Panel 1b: Proportion over time
        ax = axes[0, 1]
        ax.plot(times_window, proportion_window, linewidth=0.5, color='purple', alpha=0.7)
        ax.axhline(0.5, color='orange', linestyle='--', label='Halfway through turn', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Proportion through turn')
        ax.set_title(f'Proportion-Through-Turn Over Time ({args.speaker})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        # Panel 2b: Histogram
        ax = axes[1, 1]
        ax.hist(valid_proportion, bins=50, color='purple', alpha=0.7, edgecolor='black')
        ax.axvline(0.5, color='orange', linestyle='--', label='Halfway (0.5)', linewidth=2)
        ax.set_xlabel('Proportion through turn')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Proportion-Through-Turn')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Panel 3b: Cumulative distribution
        ax = axes[2, 1]
        sorted_prop = np.sort(valid_proportion)
        cumulative = np.arange(1, len(sorted_prop) + 1) / len(sorted_prop) * 100
        ax.plot(sorted_prop, cumulative, linewidth=2, color='purple')
        ax.axvline(0.5, color='orange', linestyle='--', label='Halfway (0.5)', linewidth=2)
        ax.set_xlabel('Proportion through turn')
        ax.set_ylabel('Cumulative percentage (%)')
        ax.set_title('Cumulative Distribution of Proportion')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])
        ax.set_xlim([0, 1])

        plt.tight_layout()
    else:
        # Fallback: distance only (3x1 layout)
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # Panel 1: Distance over time
        ax = axes[0]
        ax.plot(times_window, distance_window, linewidth=0.5, color='blue', alpha=0.7)
        ax.axhline(1.0, color='red', linestyle='--', label='Close threshold (1.0s)', linewidth=2)
        ax.axhline(2.0, color='green', linestyle='--', label='Far threshold (2.0s)', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Distance to turn (s)')
        ax.set_title(f'Distance-to-Turn Over Time ({args.speaker} - {args.condition})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        # Panel 2: Histogram
        ax = axes[1]
        ax.hist(distance_data, bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax.axvline(1.0, color='red', linestyle='--', label='Close threshold', linewidth=2)
        ax.axvline(2.0, color='green', linestyle='--', label='Far threshold', linewidth=2)
        ax.set_xlabel('Distance to turn (s)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Distance-to-Turn')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Panel 3: Cumulative distribution
        ax = axes[2]
        sorted_dist = np.sort(distance_data)
        cumulative = np.arange(1, len(sorted_dist) + 1) / len(sorted_dist) * 100
        ax.plot(sorted_dist, cumulative, linewidth=2, color='blue')
        ax.axvline(1.0, color='red', linestyle='--', label='Close threshold', linewidth=2)
        ax.axvline(2.0, color='green', linestyle='--', label='Far threshold', linewidth=2)
        ax.set_xlabel('Distance to turn (s)')
        ax.set_ylabel('Cumulative percentage (%)')
        ax.set_title('Cumulative Distribution of Distance-to-Turn')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])

        plt.tight_layout()

    # Save plot
    output_dir = Path(f'outputs/diagnostics')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{args.subject}_{args.condition}_{args.speaker}_distance_to_turn.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved plot: {output_file}")

    plt.close()

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\nTo test your hypothesis (surprisal sensitivity decreases near boundaries):")

    if proportion_data is not None:
        print("\nRECOMMENDED: Use proportion-through-turn for stratification")
        print("\n1. Run stratified TRF analysis using proportion predictor:")
        print(f"   - FAR from boundary: proportion < 0.5 (first half of turns)")
        print(f"   - CLOSE to boundary: proportion >= 0.5 (second half of turns)")
        print("\n2. For each condition, fit separate TRF models:")
        print("   - Load combined FIF file")
        print("   - Create boolean masks: far_mask = (proportion < 0.5)")
        print("   - Fit TRF on far_mask, then on ~far_mask")
        print("\n3. Compare surprisal TRF kernels between far and close")
        print("\n4. Look for:")
        print("   - Smaller surprisal TRF amplitude when close to boundary")
        print("   - Earlier/later surprisal TRF peak latency when close vs. far")
        print("   - Changes in topographic distribution")
        print("\n5. Statistical test: Paired t-test on surprisal TRF amplitudes")
        print("\nADVANTAGES of proportion-based stratification:")
        print("  - Turn-length normalized (fair comparison across short/long turns)")
        print("  - Balanced data split (~50/50)")
        print("  - Accounts for natural turn duration variability")
    else:
        print("\nWARNING: Proportion predictor not found in data.")
        print("Run: python scripts/create_trf_fif.py --subject <sub> --runs 1 2 3 4 5 --downsample 100")
        print("\nFallback: Use distance-based thresholds")
        print("\n1. Run TRF analysis with these thresholds:")
        print(f"   - Close: distance < 1.0s")
        print(f"   - Far: distance > 2.0s")
        print("\n2. Compare surprisal TRF kernels between close and far conditions")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
