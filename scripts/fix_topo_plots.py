#!/usr/bin/env python
"""
Fix Topography Plots

Regenerate proper MEG/EEG topography plots from saved TRF models.
Creates real head maps instead of scatter plots.
"""

import numpy as np
import mne
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import argparse


def create_proper_topomap(trf_model_path, meg_file, sensor_type, output_path):
    """
    Create proper topography plot from saved TRF model.

    Parameters
    ----------
    trf_model_path : Path
        Path to saved TRF model pickle file
    meg_file : str
        Path to MEG file (to get sensor positions)
    sensor_type : str
        'mag', 'grad', or 'eeg'
    output_path : Path
        Where to save the fixed topography plot
    """
    print(f"\nLoading TRF model from: {trf_model_path}")
    with open(trf_model_path, 'rb') as f:
        trf = pickle.load(f)

    # Extract correlation values
    if hasattr(trf, 'r'):
        if hasattr(trf.r, 'x'):
            corr_values = trf.r.x
        else:
            corr_values = np.array(trf.r)
    else:
        raise ValueError("Cannot extract correlation values from TRF model")

    print(f"  Correlation values: mean={corr_values.mean():.4f}, max={corr_values.max():.4f}")
    print(f"  Number of sensors: {len(corr_values)}")

    # Load MEG file to get sensor positions
    print(f"\nLoading MEG file: {meg_file}")
    raw = mne.io.read_raw_fif(meg_file, preload=False, verbose=False)

    # Pick sensors
    if sensor_type == 'eeg':
        picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude='bads')
    else:
        picks = mne.pick_types(raw.info, meg=sensor_type, eeg=False, exclude='bads')

    print(f"  Found {len(picks)} {sensor_type} sensors")

    # Create info object with only the selected sensors
    info = mne.pick_info(raw.info, picks)

    # Verify dimensions match
    if len(corr_values) != len(picks):
        print(f"WARNING: Correlation values ({len(corr_values)}) != picks ({len(picks)})")
        print("  Using minimum length...")
        min_len = min(len(corr_values), len(picks))
        corr_values = corr_values[:min_len]
        picks = picks[:min_len]
        info = mne.pick_info(raw.info, picks)

    # Create topography plot
    print(f"\nCreating topography plot...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Determine colormap limits
    vmax = max(abs(corr_values.min()), corr_values.max())

    # Plot topography
    im, cn = mne.viz.plot_topomap(
        corr_values,
        info,
        axes=ax,
        show=False,
        cmap='RdBu_r',
        vlim=(corr_values.min(), corr_values.max()),
        contours=6,
        sensors=True,
        names=None,
        mask=None,
        res=64
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('TRF Correlation', rotation=270, labelpad=20)

    # Title
    ax.set_title(
        f'TRF Model Correlation - {sensor_type.upper()}\n'
        f'Mean: {corr_values.mean():.4f}, Max: {corr_values.max():.4f}',
        fontsize=14, pad=20
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fix topography plots from saved TRF models")
    parser.add_argument('--trf-dir', type=Path, required=True, help='Directory containing TRF results')
    parser.add_argument('--meg-file', type=str, required=True, help='MEG file (for sensor positions)')
    parser.add_argument('--sensor-type', type=str, required=True, choices=['mag', 'grad', 'eeg'],
                       help='Sensor type')

    args = parser.parse_args()

    # Find TRF model file
    trf_model_path = args.trf_dir / 'trf_model.pkl'

    if not trf_model_path.exists():
        raise FileNotFoundError(f"TRF model not found: {trf_model_path}")

    # Output path
    output_path = args.trf_dir / 'trf_correlation_topo_fixed.png'

    # Create proper topography
    create_proper_topomap(trf_model_path, args.meg_file, args.sensor_type, output_path)

    print(f"\n{'='*70}")
    print("TOPOGRAPHY FIX COMPLETE!")
    print(f"{'='*70}")
    print(f"Fixed topography saved to: {output_path}")


if __name__ == '__main__':
    main()
