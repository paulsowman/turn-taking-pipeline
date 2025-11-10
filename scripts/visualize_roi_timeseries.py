#!/usr/bin/env python3
"""
Visualize time series for BADA localizer ROI sensors/electrodes.

Shows the evoked response for each sensor/electrode in the ROI
to help understand what signals were selected.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mne


def load_roi_sensors(roi_json_path):
    """Load ROI sensor information from JSON file."""
    with open(roi_json_path, 'r') as f:
        roi_data = json.load(f)

    left_sensors = roi_data['left_hemisphere']['names']
    right_sensors = roi_data['right_hemisphere']['names']

    return left_sensors, right_sensors


def compute_evoked(meg_file):
    """
    Compute evoked response from BADA localizer data.

    Uses same approach as bada_localizer_roi.py
    """
    print(f"\nLoading MEG file...")
    raw = mne.io.read_raw_fif(meg_file, preload=True, verbose=False)

    print(f"  Detecting syllable onsets from annotations...")
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    if len(events) == 0:
        raise ValueError("No events found in annotations")

    print(f"  Found {len(events)} syllable onsets")

    # Epoch around syllable onsets
    tmin, tmax = -0.1, 0.5
    print(f"  Creating epochs ({tmin} to {tmax}s)...")

    epochs = mne.Epochs(
        raw, events,
        tmin=tmin, tmax=tmax,
        baseline=(tmin, 0),
        preload=True,
        verbose=False
    )

    # Average to create evoked response
    evoked = epochs.average()

    return evoked


def plot_roi_timeseries(evoked, left_sensors, right_sensors, sensor_type, output_path):
    """
    Plot time series for all ROI sensors/electrodes.

    Creates figure with:
    - Left hemisphere sensors in one subplot
    - Right hemisphere sensors in another subplot
    - Highlights the 50-200ms window used for ROI selection
    """
    print(f"\nCreating time series visualization...")

    # Get channel names
    all_ch_names = evoked.ch_names

    # Find indices
    left_indices = [all_ch_names.index(ch) for ch in left_sensors]
    right_indices = [all_ch_names.index(ch) for ch in right_sensors]

    # Get data
    times = evoked.times
    data = evoked.data

    # Create figure
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6))

    # === LEFT HEMISPHERE ===
    for i, (idx, ch_name) in enumerate(zip(left_indices, left_sensors)):
        # Compute RMS in 50-200ms window
        time_mask = (times >= 0.05) & (times <= 0.20)
        rms = np.sqrt(np.mean(data[idx, time_mask] ** 2))

        # Plot
        ax_left.plot(times, data[idx], label=f'{ch_name} (RMS={rms:.2e})', alpha=0.7)

    # Highlight ROI selection window
    ax_left.axvspan(0.05, 0.20, alpha=0.1, color='yellow', label='ROI window (50-200ms)')
    ax_left.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax_left.axvline(0, color='k', linestyle='--', linewidth=0.5)

    ax_left.set_xlabel('Time (s)', fontsize=12)
    ax_left.set_ylabel('Amplitude', fontsize=12)
    ax_left.set_title(f'Left Hemisphere ROI ({len(left_sensors)} {sensor_type})',
                      fontsize=14, fontweight='bold')
    ax_left.legend(loc='upper right', fontsize=8, ncol=1)
    ax_left.grid(True, alpha=0.3)

    # === RIGHT HEMISPHERE ===
    for i, (idx, ch_name) in enumerate(zip(right_indices, right_sensors)):
        # Compute RMS in 50-200ms window
        time_mask = (times >= 0.05) & (times <= 0.20)
        rms = np.sqrt(np.mean(data[idx, time_mask] ** 2))

        # Plot
        ax_right.plot(times, data[idx], label=f'{ch_name} (RMS={rms:.2e})', alpha=0.7)

    # Highlight ROI selection window
    ax_right.axvspan(0.05, 0.20, alpha=0.1, color='yellow', label='ROI window (50-200ms)')
    ax_right.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax_right.axvline(0, color='k', linestyle='--', linewidth=0.5)

    ax_right.set_xlabel('Time (s)', fontsize=12)
    ax_right.set_ylabel('Amplitude', fontsize=12)
    ax_right.set_title(f'Right Hemisphere ROI ({len(right_sensors)} {sensor_type})',
                       fontsize=14, fontweight='bold')
    ax_right.legend(loc='upper right', fontsize=8, ncol=1)
    ax_right.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize time series for BADA localizer ROI sensors'
    )
    parser.add_argument('--subject', type=str, default='sub-01', help='Subject ID')
    parser.add_argument('--meg-file', type=str, required=True,
                       help='MEG file for BADA localizer (run 6)')
    parser.add_argument('--roi-json', type=str, required=True,
                       help='ROI JSON file (from bada_localizer_roi.py)')
    parser.add_argument('--sensor-type', type=str, default='mag',
                       choices=['mag', 'grad', 'eeg'], help='Sensor type')
    parser.add_argument('--base-dir', type=Path, default=None, help='Base directory')

    args = parser.parse_args()

    # Set base directory
    if args.base_dir is None:
        args.base_dir = Path(__file__).resolve().parent.parent
    else:
        args.base_dir = Path(args.base_dir)

    print("="*70)
    print("BADA LOCALIZER ROI TIME SERIES VISUALIZATION")
    print("="*70)
    print(f"\nSubject: {args.subject}")
    print(f"MEG file: {args.meg_file}")
    print(f"ROI JSON: {args.roi_json}")
    print(f"Sensor type: {args.sensor_type}")

    # Load ROI sensors
    print(f"\nLoading ROI sensors...")
    left_sensors, right_sensors = load_roi_sensors(args.roi_json)
    print(f"  Left hemisphere: {len(left_sensors)} sensors")
    print(f"  Right hemisphere: {len(right_sensors)} sensors")

    # Compute evoked response
    evoked = compute_evoked(args.meg_file)

    # Plot time series
    output_dir = args.base_dir / "outputs" / "bada_localizer" / args.subject
    output_path = output_dir / f'roi_timeseries_{args.sensor_type}.png'

    plot_roi_timeseries(evoked, left_sensors, right_sensors, args.sensor_type, output_path)

    print("\n" + "="*70)
    print("ROI TIME SERIES VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"\nOutput: {output_path}\n")


if __name__ == '__main__':
    main()
