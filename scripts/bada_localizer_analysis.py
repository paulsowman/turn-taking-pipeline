#!/usr/bin/env python3
"""
Ba-Da Localizer ERF/TRF Analysis

This script analyzes the ba-da localizer (run 6) using both:
1. Traditional ERF (evoked response) analysis
2. TRF analysis

The TRF should approximate the ERF for simple syllable onsets.
This serves as a validation that the TRF approach is working correctly.

Expected results:
- M100 (~100ms) auditory response
- M200 (~200ms) response
- Strong correlation between TRF and ERF
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import mne
import eelbrain
import matplotlib.pyplot as plt


def detect_bada_onsets(meg_raw):
    """
    Detect ba-da syllable onsets from annotations.

    Parameters
    ----------
    meg_raw : mne.io.Raw
        MEG data with annotations

    Returns
    -------
    events : np.ndarray
        Events array (n_events, 3) with columns [sample, prev_value, event_id]
    event_id : dict
        Mapping from event names to IDs
    event_times : np.ndarray
        Event onset times in seconds
    """
    # Get events from annotations (ba-da localizer uses annotations)
    events, event_id = mne.events_from_annotations(meg_raw, verbose=False)

    if len(events) == 0:
        raise ValueError("Could not find any events in annotations")

    event_times = events[:, 0] / meg_raw.info['sfreq']

    print(f"  Found {len(events)} syllable onsets")
    print(f"  Event types: {event_id}")
    print(f"  Time range: {event_times.min():.1f}s - {event_times.max():.1f}s")
    print(f"  Mean ISI: {np.diff(event_times).mean():.3f}s")

    return events, event_id, event_times


def compute_erf(meg_raw, events, event_id, sensor_type='mag', tmin=-0.2, tmax=0.5):
    """
    Compute traditional ERF (event-related field).

    Parameters
    ----------
    meg_raw : mne.io.Raw
        MEG data
    events : np.ndarray
        Events array from mne.events_from_annotations
    event_id : dict
        Mapping from event names to IDs (e.g., {'ba': 1, 'da': 2})
    sensor_type : str
        'mag', 'grad', or 'eeg'
    tmin, tmax : float
        Time window around events (seconds)

    Returns
    -------
    evoked : mne.Evoked
        Averaged evoked response (combined ba+da)
    epochs : mne.Epochs
        Individual epochs
    """
    print("\nComputing ERF...")

    # Create epochs
    if sensor_type == 'eeg':
        picks = mne.pick_types(meg_raw.info, meg=False, eeg=True, exclude='bads')
    else:
        picks = mne.pick_types(meg_raw.info, meg=sensor_type, eeg=False, exclude='bads')

    epochs = mne.Epochs(
        meg_raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        picks=picks,
        baseline=(None, 0),
        preload=True,
        verbose=False
    )

    print(f"  Epochs: {len(epochs)} trials")
    print(f"  Event counts: {dict(zip(event_id.keys(), [len(epochs[k]) for k in event_id.keys()]))}")

    # Equalize event counts (ba vs da)
    epochs.equalize_event_counts(event_id)
    print(f"  After equalization: {len(epochs)} trials")

    # Compute evoked (average across all event types)
    evoked = epochs.average()

    peak_ch, peak_time = evoked.get_peak()
    print(f"  Peak latency: {peak_time * 1000:.0f}ms")
    print(f"  Peak channel: {peak_ch}")

    return evoked, epochs


def compute_trf_localizer(meg_raw, event_times, sensor_type='mag'):
    """
    Compute TRF for ba-da localizer.

    Uses simple impulse predictor at syllable onsets.
    TRF should approximate ERF.
    """
    print("\nComputing TRF...")

    # Get MEG data
    if sensor_type == 'eeg':
        picks = mne.pick_types(meg_raw.info, meg=False, eeg=True, exclude='bads')
    else:
        picks = mne.pick_types(meg_raw.info, meg=sensor_type, eeg=False, exclude='bads')
    meg_data, meg_times = meg_raw[picks, :]
    sfreq = meg_raw.info['sfreq']

    # Create impulse predictor
    impulse = np.zeros(len(meg_times))
    for event_time in event_times:
        idx = np.argmin(np.abs(meg_times - event_time))
        impulse[idx] = 1.0

    print(f"  Impulses: {int(impulse.sum())}")

    # Convert to eelbrain
    time_dim = eelbrain.UTS(0, 1/sfreq, meg_data.shape[1])

    ch_info = mne.pick_info(meg_raw.info, picks)
    try:
        sensor_dim = eelbrain.load.mne.sensor_dim(ch_info)
    except (AttributeError, TypeError):
        sensor_dim = eelbrain.Case

    meg_ndvar = eelbrain.NDVar(
        meg_data,
        dims=(sensor_dim, time_dim),
        name='MEG'
    )

    impulse_ndvar = eelbrain.NDVar(
        impulse,
        dims=(time_dim,),
        name='Syllable'
    )

    # Fit TRF
    trf = eelbrain.boosting(
        meg_ndvar,
        impulse_ndvar,
        tstart=-0.2,
        tstop=0.5,
        scale_data=True,
        delta=0.005,
        mindelta=0.0005,
        error='l1',
    )

    print(f"  TRF correlation: {trf.r.mean():.3f}")
    print(f"  Max correlation: {trf.r.max():.3f}")

    return trf, meg_ndvar, impulse_ndvar


def compare_erf_trf(evoked, trf, output_dir):
    """
    Compare ERF and TRF to validate TRF approach.

    TRF kernel should resemble ERF waveform for simple impulse predictor.
    """
    print("\nComparing ERF and TRF...")

    # Create separate plots (eelbrain plots don't work well with subplots)

    # 1. Plot ERF using MNE
    fig_erf = evoked.plot(spatial_colors=True, gfp=True, show=False)
    fig_erf.savefig(output_dir / 'erf_evoked.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir / 'erf_evoked.png'}")
    plt.close(fig_erf)

    # 2. Plot TRF correlation topography
    try:
        fig_trf_corr = eelbrain.plot.Topomap(trf.r, vmax=0.2, cmap='RdYlGn')
        fig_trf_corr.set_title('TRF Model Correlation')
        fig_trf_corr.figure.savefig(output_dir / 'trf_correlation.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_dir / 'trf_correlation.png'}")
        plt.close(fig_trf_corr.figure)
    except Exception as e:
        print(f"  Warning: Could not plot TRF correlation: {e}")

    # 3. Plot TRF kernel (response function)
    try:
        if hasattr(trf, 'h'):
            h_data = trf.h if not isinstance(trf.h, tuple) else trf.h[0]
            fig_trf_kernel = eelbrain.plot.TopoButterfly(h_data, vmax=0.01, cmap='xpolar')
            fig_trf_kernel.set_title('TRF Kernel (Impulse Response)')
            fig_trf_kernel.figure.savefig(output_dir / 'trf_kernel.png', dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved: {output_dir / 'trf_kernel.png'}")
            plt.close(fig_trf_kernel.figure)
    except Exception as e:
        print(f"  Warning: Could not plot TRF kernel: {e}")

    # 4. Plot GFP comparison
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Global field power for ERF
        gfp_erf = np.sqrt(np.mean(evoked.data ** 2, axis=0))
        times_erf = evoked.times

        ax.plot(times_erf * 1000, gfp_erf, label='ERF GFP', linewidth=2)

        # Mark M100 and M200
        m100_window = (times_erf >= 0.08) & (times_erf <= 0.12)
        m200_window = (times_erf >= 0.15) & (times_erf <= 0.25)

        if m100_window.any():
            m100_idx = np.argmax(gfp_erf[m100_window])
            m100_time = times_erf[m100_window][m100_idx] * 1000
            ax.axvline(m100_time, color='r', linestyle='--', alpha=0.5, label=f'M100 ({m100_time:.0f}ms)')

        if m200_window.any():
            m200_idx = np.argmax(gfp_erf[m200_window])
            m200_time = times_erf[m200_window][m200_idx] * 1000
            ax.axvline(m200_time, color='b', linestyle='--', alpha=0.5, label=f'M200 ({m200_time:.0f}ms)')

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Global Field Power (T)')
        ax.set_title('ERF Global Field Power - Ba-Da Localizer')
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.savefig(output_dir / 'erf_gfp.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_dir / 'erf_gfp.png'}")
        plt.close(fig)
    except Exception as e:
        print(f"  Warning: Could not plot GFP: {e}")


def main():
    parser = argparse.ArgumentParser(description="Ba-Da localizer ERF/TRF analysis")
    parser.add_argument("--subject", required=True, help="Subject ID")
    parser.add_argument("--meg-file", required=True, help="Path to MEG file (run-06)")
    parser.add_argument("--sensor-type", default="mag",
                       choices=['mag', 'grad', 'eeg'],
                       help="Sensor type")
    parser.add_argument("--base-dir", type=Path,
                       default=Path("/Users/em18033/Library/CloudStorage/OneDrive-AUTUniversity/Projects/turn-taking-pipeline"),
                       help="Pipeline base directory")

    args = parser.parse_args()

    print("="*70)
    print("BA-DA LOCALIZER ERF/TRF ANALYSIS")
    print("="*70)

    # Load MEG
    print(f"\nLoading MEG data: {args.meg_file}")
    meg_raw = mne.io.read_raw_fif(args.meg_file, preload=True, verbose=False)
    print(f"  Duration: {meg_raw.times[-1]:.1f}s")
    print(f"  Sampling rate: {meg_raw.info['sfreq']:.0f} Hz")

    # Detect syllable onsets
    print(f"\nDetecting syllable onsets from annotations...")
    events, event_id, event_times = detect_bada_onsets(meg_raw)

    # Compute ERF
    evoked, epochs = compute_erf(meg_raw, events, event_id, args.sensor_type)

    # Compute TRF
    trf, meg_ndvar, impulse_ndvar = compute_trf_localizer(
        meg_raw, event_times, args.sensor_type
    )

    # Save results
    output_dir = args.base_dir / "outputs" / "bada_localizer" / args.subject
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    # Save evoked
    evoked.save(output_dir / 'evoked-ave.fif', overwrite=True)
    print(f"✓ Saved: {output_dir / 'evoked-ave.fif'}")

    # Save TRF
    import pickle
    with open(output_dir / 'trf_model.pkl', 'wb') as f:
        pickle.dump(trf, f)
    print(f"✓ Saved: {output_dir / 'trf_model.pkl'}")

    # Save events
    np.save(output_dir / 'events.npy', events)
    print(f"✓ Saved: {output_dir / 'events.npy'}")

    # Compare ERF and TRF
    compare_erf_trf(evoked, trf, output_dir)

    # Save summary
    peak_ch, peak_time = evoked.get_peak()
    summary = {
        'subject': args.subject,
        'sensor_type': args.sensor_type,
        'n_events': len(events),
        'erf_peak_latency_ms': float(peak_time * 1000),
        'erf_peak_channel': peak_ch,
        'trf_correlation_mean': float(trf.r.mean()),
        'trf_correlation_max': float(trf.r.max()),
    }

    import json
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved: {output_dir / 'summary.json'}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults: {output_dir}")
    print(f"\nERF:")
    print(f"  Peak latency: {summary['erf_peak_latency_ms']:.0f}ms")
    print(f"  Peak channel: {summary['erf_peak_channel']}")
    print(f"\nTRF:")
    print(f"  Mean correlation: {summary['trf_correlation_mean']:.3f}")
    print(f"  Max correlation: {summary['trf_correlation_max']:.3f}")

    if summary['trf_correlation_max'] > 0.1:
        print("\n✓ TRF shows good correlation - method validated!")
    else:
        print("\n⚠ Warning: Low TRF correlation - check data/parameters")


if __name__ == "__main__":
    main()
