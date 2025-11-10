#!/usr/bin/env python
"""
BADA Localizer ROI Selection

Uses the BADA syllable discrimination task (run 6) to define subject-specific,
functionally-defined ROIs for speech processing.

Creates bilateral, spatially contiguous ROIs based on:
- Option 1: ERF/ERP responses to speech sounds
- Option 2: Word-level TRF model performance

Outputs:
- ROI sensor lists (JSON)
- Visualization plots for sanity checking
- Modified TRF scripts can use --roi-sensors parameter
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from scipy import ndimage
import json


def compute_erf_erp_contrast(meg_file, sensor_type, tmin=-0.1, tmax=0.5):
    """
    Compute ERF/ERP contrast for speech sounds.

    Uses all speech events (ba + da combined) vs baseline.

    Parameters
    ----------
    meg_file : str
        Path to MEG file (run 6 - BADA localizer)
    sensor_type : str
        'mag', 'grad', or 'eeg'
    tmin, tmax : float
        Time window for epochs (s)

    Returns
    -------
    evoked : mne.Evoked
        Averaged evoked response
    contrast : np.ndarray
        Contrast values per sensor (peak-to-peak amplitude in 50-250ms window)
    """
    print(f"\n{'='*70}")
    print("COMPUTING ERF/ERP CONTRAST")
    print(f"{'='*70}\n")

    # Load MEG data
    print(f"Loading MEG file...")
    raw = mne.io.read_raw_fif(meg_file, preload=True, verbose=False)

    # Pick sensors
    if sensor_type == 'eeg':
        picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude='bads')
    else:
        picks = mne.pick_types(raw.info, meg=sensor_type, eeg=False, exclude='bads')

    print(f"  Sensors: {len(picks)} {sensor_type}")
    print(f"  Sampling rate: {raw.info['sfreq']} Hz")

    # Find events from annotations (BADA task uses annotations)
    print(f"\n  Detecting syllable onsets from annotations...")
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    if len(events) == 0:
        raise ValueError("No events found in annotations")

    event_times = events[:, 0] / raw.info['sfreq']

    print(f"  Found {len(events)} syllable onsets")
    print(f"  Event types: {event_id}")
    print(f"  Time range: {event_times.min():.1f}s - {event_times.max():.1f}s")
    print(f"  Mean ISI: {np.diff(event_times).mean():.3f}s")

    epochs = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=tmin, tmax=tmax,
        picks=picks,
        baseline=(None, 0),
        preload=True,
        verbose=False
    )

    print(f"  Created {len(epochs)} epochs")
    print(f"  Time window: {tmin} to {tmax}s")
    print(f"  Baseline: {tmin} to 0s")

    # Average to create evoked response
    evoked = epochs.average()

    print(f"\n  Computing contrast (RMS amplitude 50-200ms)...")

    # Extract time window for contrast (50-200ms - typical speech response)
    times = evoked.times
    time_mask = (times >= 0.05) & (times <= 0.20)

    # Get data
    data = evoked.data[:, time_mask]

    # Compute RMS amplitude per sensor (captures magnitude regardless of polarity)
    contrast = np.sqrt(np.mean(data ** 2, axis=1))

    print(f"  Mean contrast: {contrast.mean():.2e}")
    print(f"  Max contrast: {contrast.max():.2e}")

    return evoked, contrast, raw.info


def identify_bilateral_clusters(contrast, info, sensor_type, n_sensors_per_hemi=20,
                                 distance_threshold=0.04):
    """
    Identify bilateral ROIs by selecting top N sensors per hemisphere.

    Selects sensors with highest RMS amplitude in each hemisphere.
    No spatial contiguity constraint (avoids issues with gradiometer source/sink patterns).

    Parameters
    ----------
    contrast : np.ndarray
        Contrast value per sensor
    info : mne.Info
        MNE info object with sensor positions
    sensor_type : str
        'mag', 'grad', or 'eeg'
    n_sensors_per_hemi : int
        Target number of sensors per hemisphere (default: 20)
    distance_threshold : float
        Unused (kept for API compatibility)

    Returns
    -------
    left_roi : list
        Sensor indices for left hemisphere ROI
    right_roi : list
        Sensor indices for right hemisphere ROI
    """
    print(f"\n{'='*70}")
    print("IDENTIFYING BILATERAL ROIS")
    print(f"{'='*70}\n")

    # Get sensor positions
    if sensor_type == 'eeg':
        picks = mne.pick_types(info, meg=False, eeg=True, exclude='bads')
    else:
        picks = mne.pick_types(info, meg=sensor_type, eeg=False, exclude='bads')

    pos = np.array([info['chs'][i]['loc'][:3] for i in picks])

    print(f"  Total sensors: {len(picks)}")
    print(f"  Target per hemisphere: {n_sensors_per_hemi}")

    # Separate hemispheres based on x-coordinate (left = negative, right = positive)
    left_mask = pos[:, 0] < 0
    right_mask = pos[:, 0] > 0

    print(f"  Left hemisphere: {np.sum(left_mask)} sensors")
    print(f"  Right hemisphere: {np.sum(right_mask)} sensors")

    # For each hemisphere, select top N sensors by RMS amplitude
    def select_top_sensors(hemi_mask, n_target):
        """
        Select top N sensors by RMS amplitude.

        For EEG: excludes midline electrodes (|x| < 0.02m) to avoid frontal artifacts.
        For MEG: simple ranking by RMS amplitude.
        """
        # Get sensors in this hemisphere
        hemi_indices = np.where(hemi_mask)[0]
        hemi_contrast = contrast[hemi_indices]
        hemi_pos = pos[hemi_indices]

        # For EEG, exclude electrodes too close to midline (frontal blob artifact)
        if sensor_type == 'eeg':
            # Exclude electrodes within 2cm of midline
            midline_threshold = 0.02  # meters
            lateral_mask = np.abs(hemi_pos[:, 0]) >= midline_threshold

            # Filter to lateral electrodes only
            lateral_indices = hemi_indices[lateral_mask]
            lateral_contrast = hemi_contrast[lateral_mask]

            if len(lateral_indices) < n_target:
                print(f"    Warning: Only {len(lateral_indices)} lateral electrodes available, using all")
                sorted_indices = np.argsort(lateral_contrast)[::-1]
            else:
                sorted_indices = np.argsort(lateral_contrast)[::-1][:n_target]

            return lateral_indices[sorted_indices].tolist()
        else:
            # For MEG: simple ranking by RMS amplitude
            sorted_indices = np.argsort(hemi_contrast)[::-1]
            top_n = min(n_target, len(sorted_indices))
            return hemi_indices[sorted_indices[:top_n]].tolist()

    left_roi = select_top_sensors(left_mask, n_sensors_per_hemi)
    right_roi = select_top_sensors(right_mask, n_sensors_per_hemi)

    print(f"\n  Left ROI: {len(left_roi)} sensors")
    print(f"  Right ROI: {len(right_roi)} sensors")
    print(f"  Total ROI: {len(left_roi) + len(right_roi)} sensors")

    # Compute average contrast in ROIs
    left_contrast_mean = contrast[left_roi].mean()
    right_contrast_mean = contrast[right_roi].mean()
    all_contrast_mean = contrast.mean()

    print(f"\n  ROI contrast enrichment:")
    print(f"    Left ROI:  {left_contrast_mean:.2e} ({left_contrast_mean/all_contrast_mean:.2f}× overall mean)")
    print(f"    Right ROI: {right_contrast_mean:.2e} ({right_contrast_mean/all_contrast_mean:.2f}× overall mean)")

    return left_roi, right_roi


def visualize_roi(contrast, info, sensor_type, left_roi, right_roi, evoked, output_path):
    """
    Create comprehensive ROI visualization for sanity checking.

    Generates:
    1. Topography with ROI overlay
    2. 3D sensor positions with ROI highlighted
    3. Contrast distribution (ROI vs non-ROI)
    4. Time series for best sensor in each hemisphere
    """
    print(f"\n{'='*70}")
    print("CREATING ROI VISUALIZATION")
    print(f"{'='*70}\n")

    # Pick sensors
    if sensor_type == 'eeg':
        picks = mne.pick_types(info, meg=False, eeg=True, exclude='bads')
    else:
        picks = mne.pick_types(info, meg=sensor_type, eeg=False, exclude='bads')

    ch_info = mne.pick_info(info, picks)

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 6))

    # === 1. Topography with ROI overlay ===
    print("  1. Topography with ROI overlay...")
    ax1 = plt.subplot(1, 3, 1)

    # Create mask for ROI sensors
    roi_mask = np.zeros(len(picks), dtype=bool)
    roi_mask[left_roi + right_roi] = True

    # Plot topography
    im, cn = mne.viz.plot_topomap(
        contrast,
        ch_info,
        axes=ax1,
        show=False,
        cmap='hot',
        vlim=(contrast.min(), contrast.max()),
        contours=6,
        sensors=True,
        names=None,
        mask=None,
        res=128
    )

    # Overlay ROI sensors as larger markers
    pos = np.array([ch_info['chs'][i]['loc'][:3] for i in range(len(picks))])
    pos_2d = mne.channels.layout._find_topomap_coords(ch_info, picks=np.arange(len(picks)))

    # Plot ROI sensors
    ax1.scatter(pos_2d[left_roi, 0], pos_2d[left_roi, 1],
                c='cyan', s=100, marker='o', edgecolors='blue', linewidths=2,
                label='Left ROI', zorder=10)
    ax1.scatter(pos_2d[right_roi, 0], pos_2d[right_roi, 1],
                c='lime', s=100, marker='o', edgecolors='green', linewidths=2,
                label='Right ROI', zorder=10)

    ax1.set_title(f'ROI Selection - {sensor_type.upper()}\n'
                  f'Left: {len(left_roi)}, Right: {len(right_roi)}',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Contrast (50-250ms)', rotation=270, labelpad=20)

    # === 2. 3D sensor positions ===
    print("  2. 3D sensor positions...")
    ax2 = plt.subplot(1, 3, 2, projection='3d')

    # All sensors (gray)
    ax2.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                c='gray', s=20, alpha=0.3, label='Non-ROI')

    # ROI sensors (colored)
    ax2.scatter(pos[left_roi, 0], pos[left_roi, 1], pos[left_roi, 2],
                c='cyan', s=80, marker='o', edgecolors='blue', linewidths=2,
                label='Left ROI')
    ax2.scatter(pos[right_roi, 0], pos[right_roi, 1], pos[right_roi, 2],
                c='lime', s=80, marker='o', edgecolors='green', linewidths=2,
                label='Right ROI')

    ax2.set_xlabel('X (Left-Right)')
    ax2.set_ylabel('Y (Anterior-Posterior)')
    ax2.set_zlabel('Z (Inferior-Superior)')
    ax2.set_title('3D Sensor Positions', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)

    # === 3. Contrast distribution ===
    print("  3. Contrast distribution...")
    ax3 = plt.subplot(1, 3, 3)

    non_roi_mask = ~roi_mask

    # Histograms
    ax3.hist(contrast[non_roi_mask], bins=30, alpha=0.5, label='Non-ROI',
             color='gray', edgecolor='black')
    ax3.hist(contrast[left_roi], bins=15, alpha=0.7, label='Left ROI',
             color='cyan', edgecolor='blue')
    ax3.hist(contrast[right_roi], bins=15, alpha=0.7, label='Right ROI',
             color='lime', edgecolor='green')

    # Mean lines
    ax3.axvline(contrast[non_roi_mask].mean(), color='gray', linestyle='--',
                linewidth=2, alpha=0.7)
    ax3.axvline(contrast[left_roi].mean(), color='cyan', linestyle='--',
                linewidth=2, alpha=0.7)
    ax3.axvline(contrast[right_roi].mean(), color='lime', linestyle='--',
                linewidth=2, alpha=0.7)

    ax3.set_xlabel('Contrast (50-250ms)', fontsize=12)
    ax3.set_ylabel('Number of Sensors', fontsize=12)
    ax3.set_title('Contrast Distribution', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Saved: {output_path}")


def save_roi_sensors(left_roi, right_roi, sensor_names, output_dir, sensor_type):
    """Save ROI sensor lists to JSON."""
    print(f"\n{'='*70}")
    print("SAVING ROI SENSOR LISTS")
    print(f"{'='*70}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dictionary with sensor indices and names
    roi_data = {
        'sensor_type': sensor_type,
        'left_hemisphere': {
            'indices': [int(i) for i in left_roi],
            'names': [sensor_names[i] for i in left_roi],
            'n_sensors': len(left_roi)
        },
        'right_hemisphere': {
            'indices': [int(i) for i in right_roi],
            'names': [sensor_names[i] for i in right_roi],
            'n_sensors': len(right_roi)
        },
        'combined': {
            'indices': [int(i) for i in (left_roi + right_roi)],
            'names': [sensor_names[i] for i in (left_roi + right_roi)],
            'n_sensors': len(left_roi) + len(right_roi)
        }
    }

    # Save as JSON
    output_file = output_dir / f'roi_sensors_{sensor_type}.json'
    with open(output_file, 'w') as f:
        json.dump(roi_data, f, indent=2)

    print(f"✓ Saved: {output_file}")

    # Also save as simple text list (easier for command-line use)
    output_file_txt = output_dir / f'roi_sensors_{sensor_type}.txt'
    with open(output_file_txt, 'w') as f:
        f.write('\n'.join(roi_data['combined']['names']))

    print(f"✓ Saved: {output_file_txt}")

    return roi_data


def main():
    parser = argparse.ArgumentParser(description="BADA localizer ROI selection")
    parser.add_argument('--subject', type=str, default='sub-01', help='Subject ID')
    parser.add_argument('--meg-file', type=str, required=True,
                       help='MEG file for BADA localizer (run 6)')
    parser.add_argument('--sensor-type', type=str, default='mag',
                       choices=['mag', 'grad', 'eeg'], help='Sensor type')
    parser.add_argument('--n-sensors-per-hemi', type=int, default=20,
                       help='Target number of sensors per hemisphere (default: 20)')
    parser.add_argument('--distance-threshold', type=float, default=0.04,
                       help='Max distance (m) between neighbors (default: 0.04m = 4cm)')
    parser.add_argument('--base-dir', type=Path, default=None, help='Base directory')

    args = parser.parse_args()

    base_dir = args.base_dir or Path(__file__).parent.parent

    print("="*70)
    print("BADA LOCALIZER ROI SELECTION")
    print("="*70)
    print(f"\nSubject: {args.subject}")
    print(f"MEG file: {args.meg_file}")
    print(f"Sensor type: {args.sensor_type}")
    print(f"\nROI parameters:")
    print(f"  Sensors per hemisphere: {args.n_sensors_per_hemi}")
    print(f"  Distance threshold: {args.distance_threshold*100:.1f}cm")

    # Compute ERF/ERP contrast
    evoked, contrast, info = compute_erf_erp_contrast(args.meg_file, args.sensor_type)

    # Identify bilateral clusters
    left_roi, right_roi = identify_bilateral_clusters(
        contrast, info, args.sensor_type,
        n_sensors_per_hemi=args.n_sensors_per_hemi,
        distance_threshold=args.distance_threshold
    )

    # Get sensor names
    if args.sensor_type == 'eeg':
        picks = mne.pick_types(info, meg=False, eeg=True, exclude='bads')
    else:
        picks = mne.pick_types(info, meg=args.sensor_type, eeg=False, exclude='bads')
    sensor_names = [info['ch_names'][i] for i in picks]

    # Save ROI
    output_dir = base_dir / "outputs" / "bada_localizer" / args.subject
    roi_data = save_roi_sensors(left_roi, right_roi, sensor_names, output_dir, args.sensor_type)

    # Visualize
    output_plot = output_dir / f'roi_visualization_{args.sensor_type}.png'
    visualize_roi(contrast, info, args.sensor_type, left_roi, right_roi, evoked, output_plot)

    print(f"\n{'='*70}")
    print("BADA LOCALIZER ROI SELECTION COMPLETE!")
    print(f"{'='*70}\n")

    print(f"Results: {output_dir}\n")
    print(f"ROI Summary:")
    print(f"  Left hemisphere:  {len(left_roi)} sensors")
    print(f"  Right hemisphere: {len(right_roi)} sensors")
    print(f"  Total:            {len(left_roi) + len(right_roi)} sensors")

    print(f"\n✓ ROI selection complete!")
    print(f"\nTo use ROI in TRF analysis, add:")
    print(f"  --roi-sensors {output_dir}/roi_sensors_{args.sensor_type}.json")


if __name__ == "__main__":
    main()
