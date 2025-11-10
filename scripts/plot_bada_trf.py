#!/usr/bin/env python3
"""
Plot TRF results from ba-da localizer analysis using matplotlib only.
No wxPython required.
"""

import argparse
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import mne


def plot_trf_results(trf_file, evoked_file, output_dir):
    """
    Plot TRF results using matplotlib.

    Parameters
    ----------
    trf_file : Path
        Path to TRF model pickle file
    evoked_file : Path
        Path to MNE evoked file
    output_dir : Path
        Output directory for plots
    """
    # Load TRF model
    print(f"Loading TRF model from {trf_file}...")
    with open(trf_file, 'rb') as f:
        trf = pickle.load(f)

    print(f"TRF correlation: mean={trf.r.mean():.3f}, max={trf.r.max():.3f}")

    # Load ERF
    print(f"Loading ERF from {evoked_file}...")
    evoked = mne.read_evokeds(evoked_file)[0]

    # Create comparison plot
    fig = plt.figure(figsize=(16, 10))

    # 1. TRF correlation across sensors
    ax1 = plt.subplot(2, 3, 1)
    correlations = trf.r.x  # Extract correlation values
    ax1.hist(correlations, bins=30, edgecolor='black', alpha=0.7)
    ax1.axvline(correlations.mean(), color='r', linestyle='--',
                label=f'Mean: {correlations.mean():.3f}')
    ax1.axvline(correlations.max(), color='g', linestyle='--',
                label=f'Max: {correlations.max():.3f}')
    ax1.set_xlabel('Correlation')
    ax1.set_ylabel('Number of sensors')
    ax1.set_title('TRF Model Correlation Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Top 10 sensors by correlation
    ax2 = plt.subplot(2, 3, 2)
    top_10_idx = np.argsort(correlations)[-10:]
    sensor_names = [f"Sensor {i}" for i in top_10_idx]
    ax2.barh(range(10), correlations[top_10_idx])
    ax2.set_yticks(range(10))
    ax2.set_yticklabels(sensor_names)
    ax2.set_xlabel('Correlation')
    ax2.set_title('Top 10 Sensors by TRF Correlation')
    ax2.grid(True, alpha=0.3, axis='x')

    # 3. TRF kernel - average across sensors
    ax3 = plt.subplot(2, 3, 3)
    try:
        h_data = trf.h if not isinstance(trf.h, tuple) else trf.h[0]
        # h_data is (sensor, time) - average across sensors
        h_avg = h_data.x.mean(axis=0)
        h_times = h_data.time.times * 1000  # Convert to ms

        ax3.plot(h_times, h_avg, linewidth=2)
        ax3.axhline(0, color='k', linestyle='-', alpha=0.3)
        ax3.axvline(0, color='r', linestyle='--', alpha=0.5, label='Stimulus onset')
        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel('TRF amplitude (normalized)')
        ax3.set_title('TRF Kernel (averaged across sensors)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    except Exception as e:
        ax3.text(0.5, 0.5, f'Error plotting TRF kernel:\n{e}',
                ha='center', va='center', transform=ax3.transAxes)

    # 4. ERF - Global Field Power
    ax4 = plt.subplot(2, 3, 4)
    gfp = np.sqrt(np.mean(evoked.data ** 2, axis=0))
    times = evoked.times * 1000

    ax4.plot(times, gfp, linewidth=2, label='ERF GFP')
    ax4.axvline(0, color='r', linestyle='--', alpha=0.5, label='Stimulus onset')

    # Mark M100
    m100_window = (evoked.times >= 0.08) & (evoked.times <= 0.12)
    if m100_window.any():
        m100_idx = np.argmax(gfp[m100_window])
        m100_time = times[m100_window][m100_idx]
        ax4.axvline(m100_time, color='g', linestyle='--', alpha=0.5,
                   label=f'M100 ({m100_time:.0f}ms)')

    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Global Field Power')
    ax4.set_title('ERF Global Field Power')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. ERF - Strongest sensor
    ax5 = plt.subplot(2, 3, 5)
    peak_sensor_idx = np.argmax(np.abs(evoked.data).max(axis=1))
    peak_sensor_data = evoked.data[peak_sensor_idx, :]
    peak_sensor_name = evoked.ch_names[peak_sensor_idx]

    ax5.plot(times, peak_sensor_data * 1e15, linewidth=2)  # Convert to fT
    ax5.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax5.axvline(0, color='r', linestyle='--', alpha=0.5, label='Stimulus onset')
    ax5.set_xlabel('Time (ms)')
    ax5.set_ylabel('Amplitude (fT)')
    ax5.set_title(f'ERF at peak sensor: {peak_sensor_name}')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Correlation vs time in TRF kernel
    ax6 = plt.subplot(2, 3, 6)
    try:
        # Show which time points in TRF kernel have highest predictive power
        h_data = trf.h if not isinstance(trf.h, tuple) else trf.h[0]
        h_times = h_data.time.times * 1000

        # For each time point, compute average absolute weight across sensors
        h_power = np.abs(h_data.x).mean(axis=0)

        ax6.plot(h_times, h_power, linewidth=2)
        ax6.axvline(0, color='r', linestyle='--', alpha=0.5, label='Stimulus onset')

        # Mark peak time
        peak_time = h_times[np.argmax(h_power)]
        ax6.axvline(peak_time, color='g', linestyle='--', alpha=0.5,
                   label=f'Peak TRF ({peak_time:.0f}ms)')

        ax6.set_xlabel('Time (ms)')
        ax6.set_ylabel('Mean |TRF weight|')
        ax6.set_title('TRF Kernel Power Over Time')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    except Exception as e:
        ax6.text(0.5, 0.5, f'Error:\n{e}',
                ha='center', va='center', transform=ax6.transAxes)

    plt.suptitle('Ba-Da Localizer: ERF vs TRF Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save figure
    output_file = output_dir / 'erf_trf_detailed_comparison.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")

    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot TRF results without wxPython")
    parser.add_argument("--subject", default="sub-01", help="Subject ID")
    parser.add_argument("--base-dir", type=Path,
                       default=Path("/Users/em18033/Library/CloudStorage/OneDrive-AUTUniversity/Projects/turn-taking-pipeline"),
                       help="Pipeline base directory")

    args = parser.parse_args()

    # Find files
    bada_dir = args.base_dir / "outputs" / "bada_localizer" / args.subject
    trf_file = bada_dir / "trf_model.pkl"
    evoked_file = bada_dir / "evoked-ave.fif"

    if not trf_file.exists():
        print(f"Error: TRF file not found: {trf_file}")
        return

    if not evoked_file.exists():
        print(f"Error: Evoked file not found: {evoked_file}")
        return

    print("="*70)
    print("PLOTTING BA-DA TRF RESULTS")
    print("="*70)

    # Create plots
    fig = plot_trf_results(trf_file, evoked_file, bada_dir)

    print("\n" + "="*70)
    print("PLOTTING COMPLETE!")
    print("="*70)
    print(f"\nOutput directory: {bada_dir}")


if __name__ == "__main__":
    main()
