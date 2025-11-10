#!/usr/bin/env python3
"""
Diagnostic script to visualize RMS energy distributions and determine
optimal threshold for listener-only period detection.
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

    # Convert to dB (current approach)
    db = 20 * np.log10(rms + 1e-10)

    # Also compute dBFS (dB relative to full scale)
    peak = np.max(np.abs(audio))
    dbfs = 20 * np.log10(rms / peak + 1e-10)

    return rms, db, dbfs


def analyze_thresholds(interviewer_db, participant_db, thresholds_db):
    """Analyze what percentage of time would be classified at different thresholds."""
    results = []

    for thresh in thresholds_db:
        interviewer_speaking = interviewer_db > thresh
        participant_speaking = participant_db > thresh

        listener_mask = interviewer_speaking & ~participant_speaking

        stats = {
            'threshold_db': thresh,
            'interviewer_only_pct': 100 * np.sum(listener_mask) / len(interviewer_db),
            'participant_only_pct': 100 * np.sum(participant_speaking & ~interviewer_speaking) / len(participant_db),
            'both_speaking_pct': 100 * np.sum(interviewer_speaking & participant_speaking) / len(interviewer_db),
            'silence_pct': 100 * np.sum(~interviewer_speaking & ~participant_speaking) / len(interviewer_db)
        }
        results.append(stats)

    return results


def analyze_relative_threshold(interviewer_db, participant_db, relative_db_values):
    """Analyze using relative energy (interviewer must be X dB louder than participant)."""
    results = []

    for rel_db in relative_db_values:
        # Listener when interviewer is rel_db louder than participant
        listener_mask = (interviewer_db - participant_db) > rel_db
        # Also require interviewer above some minimum (e.g., -50 dB to exclude pure silence)
        listener_mask = listener_mask & (interviewer_db > -50)

        participant_mask = (participant_db - interviewer_db) > rel_db
        participant_mask = participant_mask & (participant_db > -50)

        stats = {
            'relative_db': rel_db,
            'interviewer_louder_pct': 100 * np.sum(listener_mask) / len(interviewer_db),
            'participant_louder_pct': 100 * np.sum(participant_mask) / len(participant_db),
            'similar_energy_pct': 100 * np.sum(~listener_mask & ~participant_mask) / len(interviewer_db)
        }
        results.append(stats)

    return results


def plot_diagnostics(interviewer_rms, participant_rms, interviewer_db, participant_db,
                     interviewer_dbfs, participant_dbfs, threshold_results,
                     relative_results, output_file):
    """Create comprehensive diagnostic plots."""
    fig = plt.figure(figsize=(16, 12))

    # 1. RMS distributions (linear scale)
    ax1 = plt.subplot(3, 3, 1)
    ax1.hist(interviewer_rms, bins=100, alpha=0.5, label='Interviewer', color='blue', density=True)
    ax1.hist(participant_rms, bins=100, alpha=0.5, label='Participant', color='red', density=True)
    ax1.set_xlabel('RMS (linear)')
    ax1.set_ylabel('Density')
    ax1.set_title('RMS Energy Distribution (Linear)')
    ax1.legend()
    ax1.set_xlim(0, 0.1)  # Focus on typical range
    ax1.grid(True, alpha=0.3)

    # 2. dB distributions (current method)
    ax2 = plt.subplot(3, 3, 2)
    ax2.hist(interviewer_db, bins=100, alpha=0.5, label='Interviewer', color='blue', density=True)
    ax2.hist(participant_db, bins=100, alpha=0.5, label='Participant', color='red', density=True)
    ax2.set_xlabel('dB (current method)')
    ax2.set_ylabel('Density')
    ax2.set_title('Energy Distribution (dB - Current)')
    ax2.legend()
    ax2.axvline(-40, color='green', linestyle='--', linewidth=2, label='Current threshold (-40 dB)')
    ax2.grid(True, alpha=0.3)

    # 3. dBFS distributions
    ax3 = plt.subplot(3, 3, 3)
    ax3.hist(interviewer_dbfs, bins=100, alpha=0.5, label='Interviewer', color='blue', density=True)
    ax3.hist(participant_dbfs, bins=100, alpha=0.5, label='Participant', color='red', density=True)
    ax3.set_xlabel('dBFS (relative to peak)')
    ax3.set_ylabel('Density')
    ax3.set_title('Energy Distribution (dBFS)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Relative energy distribution
    ax4 = plt.subplot(3, 3, 4)
    relative_energy = interviewer_db - participant_db
    ax4.hist(relative_energy, bins=100, color='purple', density=True)
    ax4.set_xlabel('Interviewer dB - Participant dB')
    ax4.set_ylabel('Density')
    ax4.set_title('Relative Energy Distribution')
    ax4.axvline(0, color='black', linestyle='-', linewidth=1)
    ax4.axvline(10, color='green', linestyle='--', linewidth=2, label='10 dB difference')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Time percentages vs absolute threshold
    ax5 = plt.subplot(3, 3, 5)
    thresholds = [r['threshold_db'] for r in threshold_results]
    ax5.plot(thresholds, [r['interviewer_only_pct'] for r in threshold_results],
             'o-', label='Interviewer only (listening)', linewidth=2)
    ax5.plot(thresholds, [r['participant_only_pct'] for r in threshold_results],
             's-', label='Participant only (speaking)', linewidth=2)
    ax5.plot(thresholds, [r['silence_pct'] for r in threshold_results],
             '^-', label='Silence', linewidth=2)
    ax5.plot(thresholds, [r['both_speaking_pct'] for r in threshold_results],
             'd-', label='Both speaking', linewidth=2)
    ax5.set_xlabel('Threshold (dB)')
    ax5.set_ylabel('Percentage of time (%)')
    ax5.set_title('Time Distribution vs Absolute Threshold')
    ax5.axvline(-40, color='red', linestyle='--', alpha=0.5, label='Current (-40 dB)')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # 6. Time percentages vs relative threshold
    ax6 = plt.subplot(3, 3, 6)
    rel_thresholds = [r['relative_db'] for r in relative_results]
    ax6.plot(rel_thresholds, [r['interviewer_louder_pct'] for r in relative_results],
             'o-', label='Interviewer louder (listening)', linewidth=2)
    ax6.plot(rel_thresholds, [r['participant_louder_pct'] for r in relative_results],
             's-', label='Participant louder (speaking)', linewidth=2)
    ax6.plot(rel_thresholds, [r['similar_energy_pct'] for r in relative_results],
             '^-', label='Similar energy', linewidth=2)
    ax6.set_xlabel('Relative threshold (dB difference)')
    ax6.set_ylabel('Percentage of time (%)')
    ax6.set_title('Time Distribution vs Relative Threshold')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    # 7. Percentiles
    ax7 = plt.subplot(3, 3, 7)
    percentiles = np.arange(0, 101, 5)
    int_percentiles = np.percentile(interviewer_db, percentiles)
    part_percentiles = np.percentile(participant_db, percentiles)
    ax7.plot(percentiles, int_percentiles, 'o-', label='Interviewer', linewidth=2)
    ax7.plot(percentiles, part_percentiles, 's-', label='Participant', linewidth=2)
    ax7.set_xlabel('Percentile')
    ax7.set_ylabel('dB')
    ax7.set_title('Energy Percentiles')
    ax7.axhline(-40, color='red', linestyle='--', alpha=0.5, label='Current threshold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Scatter plot: interviewer vs participant energy
    ax8 = plt.subplot(3, 3, 8)
    # Subsample for visualization
    subsample = np.random.choice(len(interviewer_db), min(10000, len(interviewer_db)), replace=False)
    ax8.scatter(interviewer_db[subsample], participant_db[subsample], alpha=0.1, s=1)
    ax8.set_xlabel('Interviewer dB')
    ax8.set_ylabel('Participant dB')
    ax8.set_title('Energy Scatter Plot (10k samples)')
    ax8.axhline(-40, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax8.axvline(-40, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax8.plot([-100, 0], [-100, 0], 'g--', alpha=0.5, linewidth=1, label='Equal energy')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.set_xlim(-80, -10)
    ax8.set_ylim(-80, -10)

    # 9. Summary statistics text
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')

    # Find current threshold result
    current_result = next((r for r in threshold_results if r['threshold_db'] == -40), threshold_results[0])

    summary_text = f"""
SUMMARY STATISTICS

RMS (linear):
  Interviewer: {np.mean(interviewer_rms):.4f} ± {np.std(interviewer_rms):.4f}
  Participant: {np.mean(participant_rms):.4f} ± {np.std(participant_rms):.4f}

dB (current method):
  Interviewer: {np.mean(interviewer_db):.1f} ± {np.std(interviewer_db):.1f} dB
  Participant: {np.mean(participant_db):.1f} ± {np.std(participant_db):.1f} dB

CURRENT THRESHOLD (-40 dB):
  Listening: {current_result['interviewer_only_pct']:.1f}%
  Speaking: {current_result['participant_only_pct']:.1f}%
  Silence: {current_result['silence_pct']:.1f}%
  Overlap: {current_result['both_speaking_pct']:.1f}%

RECOMMENDATIONS:
  - Absolute threshold: Try -50 to -45 dB
  - Relative threshold: Try 5-10 dB difference
  - Target ~30-40% listening time
    """

    ax9.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
             verticalalignment='center')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved diagnostic plot: {output_file}")

    return fig


def main():
    parser = argparse.ArgumentParser(description='Diagnose listener threshold')
    parser.add_argument('--subject', type=str, default='sub-01',
                       help='Subject ID (default: sub-01)')
    parser.add_argument('--run', type=int, default=1,
                       help='Run number (default: 1)')

    args = parser.parse_args()

    print("="*70)
    print("LISTENER THRESHOLD DIAGNOSTIC")
    print("="*70)
    print(f"Subject: {args.subject}")
    print(f"Run: {args.run}")
    print()

    # Load audio
    print("Loading audio channels...")
    interviewer_audio, participant_audio, sr = load_audio_channels(args.subject, args.run)
    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {len(interviewer_audio) / sr:.1f}s")
    print()

    # Compute RMS and dB
    print("Computing RMS energy...")
    interviewer_rms, interviewer_db, interviewer_dbfs = compute_rms_db(interviewer_audio, sr)
    participant_rms, participant_db, participant_dbfs = compute_rms_db(participant_audio, sr)
    print(f"  Interviewer: {len(interviewer_rms)} frames")
    print(f"  Participant: {len(participant_rms)} frames")
    print()

    # Test absolute thresholds
    print("Testing absolute thresholds...")
    test_thresholds = np.arange(-60, -20, 2)  # -60 to -20 dB in 2 dB steps
    threshold_results = analyze_thresholds(interviewer_db, participant_db, test_thresholds)

    # Test relative thresholds
    print("Testing relative thresholds...")
    test_relative = np.arange(0, 21, 2)  # 0 to 20 dB difference in 2 dB steps
    relative_results = analyze_relative_threshold(interviewer_db, participant_db, test_relative)
    print()

    # Create output directory
    output_dir = PIPELINE_DIR / "outputs" / "diagnostics" / args.subject / f"run-{args.run:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot
    print("Creating diagnostic plots...")
    output_file = output_dir / "listener_threshold_diagnostic.png"
    plot_diagnostics(interviewer_rms, participant_rms,
                    interviewer_db, participant_db,
                    interviewer_dbfs, participant_dbfs,
                    threshold_results, relative_results,
                    output_file)

    # Save numerical results
    print("\nSaving numerical results...")
    results_file = output_dir / "threshold_analysis.json"
    results = {
        'subject': args.subject,
        'run': args.run,
        'sample_rate': sr,
        'duration_s': len(interviewer_audio) / sr,
        'rms_stats': {
            'interviewer': {
                'mean': float(np.mean(interviewer_rms)),
                'std': float(np.std(interviewer_rms)),
                'median': float(np.median(interviewer_rms)),
                'percentiles': {int(p): float(np.percentile(interviewer_rms, p))
                               for p in [10, 25, 50, 75, 90]}
            },
            'participant': {
                'mean': float(np.mean(participant_rms)),
                'std': float(np.std(participant_rms)),
                'median': float(np.median(participant_rms)),
                'percentiles': {int(p): float(np.percentile(participant_rms, p))
                               for p in [10, 25, 50, 75, 90]}
            }
        },
        'db_stats': {
            'interviewer': {
                'mean': float(np.mean(interviewer_db)),
                'std': float(np.std(interviewer_db)),
                'median': float(np.median(interviewer_db)),
                'percentiles': {int(p): float(np.percentile(interviewer_db, p))
                               for p in [10, 25, 50, 75, 90]}
            },
            'participant': {
                'mean': float(np.mean(participant_db)),
                'std': float(np.std(participant_db)),
                'median': float(np.median(participant_db)),
                'percentiles': {int(p): float(np.percentile(participant_db, p))
                               for p in [10, 25, 50, 75, 90]}
            }
        },
        'absolute_threshold_results': [{k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                                        for k, v in r.items()} for r in threshold_results],
        'relative_threshold_results': [{k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                                       for k, v in r.items()} for r in relative_results]
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved: {results_file}")

    # Print recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    # Find threshold that gives ~30-40% listening time
    target_pct = 35
    best_abs = min(threshold_results,
                   key=lambda r: abs(r['interviewer_only_pct'] - target_pct))
    best_rel = min(relative_results,
                   key=lambda r: abs(r['interviewer_louder_pct'] - target_pct))

    print(f"\nFor ~{target_pct}% listening time:")
    print(f"  Absolute threshold: {best_abs['threshold_db']:.0f} dB")
    print(f"    → Listening: {best_abs['interviewer_only_pct']:.1f}%")
    print(f"    → Speaking: {best_abs['participant_only_pct']:.1f}%")
    print(f"    → Silence: {best_abs['silence_pct']:.1f}%")
    print()
    print(f"  Relative threshold: {best_rel['relative_db']:.0f} dB difference")
    print(f"    → Interviewer louder: {best_rel['interviewer_louder_pct']:.1f}%")
    print(f"    → Participant louder: {best_rel['participant_louder_pct']:.1f}%")

    print(f"\n✓ Diagnostic complete! See: {output_dir}")
    print()


if __name__ == '__main__':
    main()
