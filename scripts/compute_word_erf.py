#!/usr/bin/env python3
"""
Compute ERF (Event-Related Field) for word onsets

Quick sanity check to validate timing and preprocessing.
Should show classic M50/M100/N400 components.

Usage:
    python scripts/compute_word_erf.py sub-01 --condition conversation
    python scripts/compute_word_erf.py sub-01 --condition conversation --split-by-surprisal
"""
import mne
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def compute_erf(subject, condition, speaker='participant', split_by_surprisal=False):
    """
    Compute ERF for word onsets.

    Parameters
    ----------
    subject : str
        Subject ID
    condition : str
        'conversation' or 'nursery_rhyme'
    speaker : str
        'participant' or 'interviewer'
    split_by_surprisal : bool
        Whether to split into high vs low surprisal words
    """
    print(f"\n{'='*70}")
    print(f"WORD-ONSET ERF: {subject} - {condition.upper()} - {speaker.upper()}")
    print(f"{'='*70}\n")

    # Load combined FIF file
    fif_file = f'outputs/trf_combined/{subject}/{subject}_{condition}_trf_raw.fif'
    raw = mne.io.read_raw_fif(fif_file, preload=True, verbose=False)

    print(f"Loaded: {fif_file}")
    print(f"Duration: {raw.times[-1]:.1f}s")
    print(f"Sampling rate: {raw.info['sfreq']} Hz")

    # Get word onset times from MISC channel
    word_onset_ch = f'MISC_word_onsets_{speaker}'
    if word_onset_ch not in raw.ch_names:
        print(f"✗ ERROR: {word_onset_ch} not found")
        return None

    # Extract word onset samples (where value = 1)
    onset_data = raw[raw.ch_names.index(word_onset_ch), :][0][0]
    onset_samples = np.where(onset_data > 0.5)[0]
    n_words = len(onset_samples)

    print(f"\nFound {n_words} word onsets")

    # Get surprisal values if splitting
    if split_by_surprisal:
        surprisal_ch = f'MISC_surprisal_{speaker}'
        if surprisal_ch not in raw.ch_names:
            print(f"✗ WARNING: {surprisal_ch} not found, cannot split by surprisal")
            split_by_surprisal = False
        else:
            surprisal_data = raw[raw.ch_names.index(surprisal_ch), :][0][0]
            surprisal_values = surprisal_data[onset_samples]

            # Split at median
            median_surprisal = np.median(surprisal_values[surprisal_values > 0])
            high_surprisal_mask = surprisal_values > median_surprisal
            low_surprisal_mask = (surprisal_values > 0) & (surprisal_values <= median_surprisal)

            print(f"Median surprisal: {median_surprisal:.2f} nats")
            print(f"High surprisal words: {high_surprisal_mask.sum()}")
            print(f"Low surprisal words: {low_surprisal_mask.sum()}")

    # Create events array for MNE epoching
    # Format: [sample, 0, event_id]
    if split_by_surprisal:
        events_high = np.column_stack([
            onset_samples[high_surprisal_mask],
            np.zeros(high_surprisal_mask.sum(), dtype=int),
            np.ones(high_surprisal_mask.sum(), dtype=int)  # event_id = 1
        ])
        events_low = np.column_stack([
            onset_samples[low_surprisal_mask],
            np.zeros(low_surprisal_mask.sum(), dtype=int),
            np.ones(low_surprisal_mask.sum(), dtype=int) * 2  # event_id = 2
        ])
        events = np.vstack([events_high, events_low])
        events = events[events[:, 0].argsort()]  # Sort by time
        event_id = {'high_surprisal': 1, 'low_surprisal': 2}
    else:
        events = np.column_stack([
            onset_samples,
            np.zeros(n_words, dtype=int),
            np.ones(n_words, dtype=int)
        ])
        event_id = {'word_onset': 1}

    print(f"\nCreating epochs: -100ms to +600ms around word onsets...")

    # Create epochs
    picks = mne.pick_types(raw.info, meg=True, exclude=[])
    epochs = mne.Epochs(
        raw,
        events,
        event_id,
        tmin=-0.1,
        tmax=0.6,
        baseline=(-0.1, 0),
        picks=picks,
        preload=True,
        reject_by_annotation=True,  # Exclude BAD segments
        verbose=False
    )

    print(f"Epochs created: {len(epochs)} epochs")
    print(f"Dropped: {epochs.drop_log.count(['IGNORED'])} due to BAD annotations")

    # Compute ERF (average)
    if split_by_surprisal:
        evoked_high = epochs['high_surprisal'].average()
        evoked_low = epochs['low_surprisal'].average()
        print(f"\nHigh surprisal ERF: {len(epochs['high_surprisal'])} epochs averaged")
        print(f"Low surprisal ERF: {len(epochs['low_surprisal'])} epochs averaged")
    else:
        evoked = epochs.average()
        print(f"\nERF computed: {len(epochs)} epochs averaged")

    # Save results
    output_dir = Path('outputs/erf_analysis') / subject / condition / speaker
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving results to: {output_dir}")

    # Save evoked data
    if split_by_surprisal:
        evoked_high.save(output_dir / 'high_surprisal-ave.fif', overwrite=True)
        evoked_low.save(output_dir / 'low_surprisal-ave.fif', overwrite=True)
        print(f"  ✓ Saved evoked responses")
    else:
        evoked.save(output_dir / 'word_onset-ave.fif', overwrite=True)
        print(f"  ✓ Saved evoked response")

    # Generate plots
    print("\nGenerating plots...")

    # 1. Butterfly plot
    fig, axes = plt.subplots(1, 1 if not split_by_surprisal else 2, figsize=(12, 6))
    if not split_by_surprisal:
        axes = [axes]

    if split_by_surprisal:
        # Plot high surprisal
        for ch_idx in range(len(picks)):
            axes[0].plot(evoked_high.times, evoked_high.data[ch_idx, :] * 1e15,
                        alpha=0.1, color='red')
        axes[0].plot(evoked_high.times, evoked_high.data.mean(axis=0) * 1e15,
                    linewidth=2, color='darkred', label='Mean')
        axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
        axes[0].axvline(0, color='k', linestyle='--', alpha=0.3, label='Word onset')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude (fT)')
        axes[0].set_title(f'High Surprisal ERF (n={len(epochs["high_surprisal"])})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot low surprisal
        for ch_idx in range(len(picks)):
            axes[1].plot(evoked_low.times, evoked_low.data[ch_idx, :] * 1e15,
                        alpha=0.1, color='blue')
        axes[1].plot(evoked_low.times, evoked_low.data.mean(axis=0) * 1e15,
                    linewidth=2, color='darkblue', label='Mean')
        axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
        axes[1].axvline(0, color='k', linestyle='--', alpha=0.3, label='Word onset')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Amplitude (fT)')
        axes[1].set_title(f'Low Surprisal ERF (n={len(epochs["low_surprisal"])})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        # Plot single ERF
        for ch_idx in range(len(picks)):
            axes[0].plot(evoked.times, evoked.data[ch_idx, :] * 1e15,
                        alpha=0.1, color='gray')
        axes[0].plot(evoked.times, evoked.data.mean(axis=0) * 1e15,
                    linewidth=2, color='red', label='Mean')
        axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
        axes[0].axvline(0, color='k', linestyle='--', alpha=0.3, label='Word onset')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude (fT)')
        axes[0].set_title(f'Word Onset ERF (n={len(epochs)})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'erf_butterfly.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: erf_butterfly.png")

    # 2. Topographic plots at key time points
    if split_by_surprisal:
        # Difference wave (high - low surprisal)
        diff_evoked = mne.combine_evoked([evoked_high, evoked_low], weights=[1, -1])

        fig = diff_evoked.plot_topomap(
            times=[0.05, 0.1, 0.2, 0.3, 0.4],
            ch_type='mag',
            size=3,
            show=False
        )
        fig.savefig(output_dir / 'erf_topomap_difference.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: erf_topomap_difference.png (high - low surprisal)")
    else:
        fig = evoked.plot_topomap(
            times=[0.05, 0.1, 0.2, 0.3, 0.4],
            ch_type='mag',
            size=3,
            show=False
        )
        fig.savefig(output_dir / 'erf_topomap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: erf_topomap.png")

    # 3. Find and report peak latencies
    print("\nERF Peak Latencies:")

    def find_peaks(evoked_obj, name):
        """Find M50, M100, N400-like components."""
        mean_signal = evoked_obj.data.mean(axis=0)
        times = evoked_obj.times

        # M50: 40-70ms, positive peak
        m50_mask = (times >= 0.04) & (times <= 0.07)
        if m50_mask.any():
            m50_idx = np.argmax(mean_signal[m50_mask])
            m50_time = times[m50_mask][m50_idx]
            m50_amp = mean_signal[m50_mask][m50_idx]
            print(f"  {name} M50:  {m50_time*1000:6.1f}ms (amplitude: {m50_amp*1e15:.2f} fT)")

        # M100: 80-130ms, negative or positive peak (depends on sensor orientation)
        m100_mask = (times >= 0.08) & (times <= 0.13)
        if m100_mask.any():
            m100_idx = np.argmax(np.abs(mean_signal[m100_mask]))
            m100_time = times[m100_mask][m100_idx]
            m100_amp = mean_signal[m100_mask][m100_idx]
            print(f"  {name} M100: {m100_time*1000:6.1f}ms (amplitude: {m100_amp*1e15:.2f} fT)")

        # N400-like: 300-500ms
        n400_mask = (times >= 0.3) & (times <= 0.5)
        if n400_mask.any():
            n400_idx = np.argmax(np.abs(mean_signal[n400_mask]))
            n400_time = times[n400_mask][n400_idx]
            n400_amp = mean_signal[n400_mask][n400_idx]
            print(f"  {name} N400: {n400_time*1000:6.1f}ms (amplitude: {n400_amp*1e15:.2f} fT)")

    if split_by_surprisal:
        find_peaks(evoked_high, "High surprisal")
        find_peaks(evoked_low, "Low surprisal")

        # Compare N400 effect
        high_mean = evoked_high.data.mean(axis=0)
        low_mean = evoked_low.data.mean(axis=0)
        diff = high_mean - low_mean
        n400_mask = (evoked_high.times >= 0.3) & (evoked_high.times <= 0.5)
        n400_effect = np.abs(diff[n400_mask]).max()
        print(f"\n  N400 effect (high-low): {n400_effect*1e15:.2f} fT")
    else:
        find_peaks(evoked, "Word onset")

    print(f"\n✓ ERF analysis complete!")
    print(f"Results saved to: {output_dir}")

    return evoked if not split_by_surprisal else (evoked_high, evoked_low)


def main():
    parser = argparse.ArgumentParser(description='Compute word-onset ERF')
    parser.add_argument('subject', help='Subject ID (e.g., sub-01)')
    parser.add_argument('--condition', required=True,
                       choices=['conversation', 'nursery_rhyme'],
                       help='Condition to analyze')
    parser.add_argument('--speaker', default='participant',
                       choices=['participant', 'interviewer'],
                       help='Which speaker to analyze (default: participant)')
    parser.add_argument('--split-by-surprisal', action='store_true',
                       help='Split into high vs low surprisal words')

    args = parser.parse_args()

    evoked = compute_erf(
        args.subject,
        args.condition,
        speaker=args.speaker,
        split_by_surprisal=args.split_by_surprisal
    )

    if evoked is not None:
        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("\nCheck ERF plots:")
        print("  - Look for M100 peak around 80-130ms")
        print("  - If M100 is clear and at expected latency, timing is correct")
        print("  - If split by surprisal, look for larger N400 for high surprisal")
        print("\nIf timing looks good:")
        print("  - TRF negative peaks might be real anticipatory effects")
        print("  - Or artifacts from predictor correlations")

    return 0


if __name__ == '__main__':
    sys.exit(main())
