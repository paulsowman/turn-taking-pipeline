#!/usr/bin/env python3
"""
TRF Analysis on Combined Conversation Data

Fits Temporal Response Function (TRF) models using eelbrain's boosted regression
to relate linguistic/prosodic predictors to MEG responses.

Features:
- Automatic polarity alignment using M100 window (80-150ms)
- 3-panel visualization with separate y-axes for clarity
- Handles opposite sensor polarities that would otherwise average to near-zero
- Edge artifact removal: fits -200 to +800ms, reports -100 to +600ms
- Multi-predictor model: each kernel shows unique contribution

Visualization Panels:
  Panel 1: Polarity-aligned mean across sensors (red)
  Panel 2: Top 5 sensors by RMS (individual sensor waveforms)
  Panel 3: RMS magnitude across sensors (polarity-independent)

Usage:
    # Analyze single condition with subset of predictors
    python scripts/analyze_trf_combined.py sub-01 --condition conversation --predictors envelope word_onsets surprisal

    # Compare both conditions
    python scripts/analyze_trf_combined.py sub-01 --compare

    # Analyze both speakers
    python scripts/analyze_trf_combined.py sub-01 --compare --speaker both

Outputs:
    trf_{predictor}.png        - Eelbrain TopoButterfly plot (if wxPython available)
    trf_{predictor}_dual.png   - 3-panel plot with separate y-axes
    trf_model.pickle           - Fitted TRF model (edges already cropped)
"""
import eelbrain
import mne
import numpy as np
from pathlib import Path
import argparse
import sys
import time
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def align_sensor_polarities(kernel_data, times, m100_window=(0.08, 0.15)):
    """
    Align sensor polarities by flipping those with negative peaks in M100 window.

    Uses a principled window around the expected M100 auditory response (80-150ms)
    rather than searching the entire time range, which avoids being influenced by
    late artifacts or edge effects.

    Parameters
    ----------
    kernel_data : ndarray
        Shape (n_sensors, n_times)
    times : ndarray
        Time axis in seconds
    m100_window : tuple
        (start, end) time window in seconds for finding peak (default: 0.08-0.15s)

    Returns
    -------
    aligned_mean : ndarray
        Polarity-aligned mean across sensors, shape (n_times,)
    n_flipped : int
        Number of sensors that were flipped
    """
    data = kernel_data.copy()
    n_sensors = data.shape[0]
    n_flipped = 0

    # Find indices for M100 window
    m100_mask = (times >= m100_window[0]) & (times <= m100_window[1])

    for i in range(n_sensors):
        # Find peak only within M100 window
        m100_data = data[i, m100_mask]
        if len(m100_data) == 0:
            continue  # Skip if window is empty

        peak_idx_in_window = np.argmax(np.abs(m100_data))
        peak_value = m100_data[peak_idx_in_window]

        # If peak is negative, flip the entire sensor
        if peak_value < 0:
            data[i, :] *= -1
            n_flipped += 1

    return np.mean(data, axis=0), n_flipped


def compute_rms_across_sensors(kernel_data):
    """
    Compute RMS across sensors at each timepoint.

    Parameters
    ----------
    kernel_data : ndarray
        Shape (n_sensors, n_times)

    Returns
    -------
    ndarray
        RMS values, shape (n_times,)
    """
    return np.sqrt(np.mean(kernel_data**2, axis=0))


def analyze_condition(subject, condition, speaker='participant', save_plots=True, predictors_to_use=None,
                     tstart=-0.2, tstop=0.8, crop_start=-0.1, crop_stop=0.6):
    """
    Analyze TRF for one condition.

    Parameters
    ----------
    subject : str
        Subject ID (e.g., 'sub-01')
    condition : str
        'conversation' or 'nursery_rhyme'
    speaker : str
        'participant', 'interviewer', or 'both'
    tstart : float
        TRF fitting window start in seconds (default: -0.2)
    tstop : float
        TRF fitting window end in seconds (default: 0.8)
    crop_start : float
        Cropped/saved window start in seconds (default: -0.1)
    crop_stop : float
        Cropped/saved window end in seconds (default: 0.6)
    save_plots : bool
        Whether to save TRF plots
    predictors_to_use : list of str, optional
        Which predictors to include. Options: 'envelope', 'word_onsets', 'surprisal',
        'f0_deviation', 'duration_deviation', 'pause'.
        If None, uses all available predictors.

    Returns
    -------
    trf : eelbrain TRF object
    """
    start_time = time.time()

    print(f"\n{'='*70}")
    print(f"TRF ANALYSIS: {subject} - {condition.upper()} - {speaker.upper()}")
    print(f"{'='*70}\n")

    # Load combined FIF file
    fif_file = f'outputs/trf_combined/{subject}/{subject}_{condition}_trf_raw.fif'

    # Load with MNE and preload data
    raw = mne.io.read_raw_fif(fif_file, preload=True, verbose=False)
    print(f"Loaded: {fif_file}")
    print(f"Duration: {raw.times[-1]:.1f}s")
    print(f"Total channels: {len(raw.ch_names)}")
    print(f"Sampling rate: {raw.info['sfreq']} Hz")
    print(f"BAD annotations: {len(raw.annotations)}")

    # Convert to eelbrain NDVar for continuous data
    # Note: BAD annotations will be automatically excluded during boosting
    print("\nConverting to eelbrain format...")

    # Get MEG channel indices
    meg_picks = mne.pick_types(raw.info, meg=True, exclude=[])
    meg_ch_names = [raw.ch_names[i] for i in meg_picks]
    print(f"Found {len(meg_ch_names)} MEG channels")

    # Extract MEG data and convert to eelbrain NDVar
    meg_data_array, times = raw[meg_picks, :]

    # Create time dimension (following existing TRF code pattern)
    time_dim = eelbrain.UTS(0, 1.0/raw.info['sfreq'], meg_data_array.shape[1])

    # Create sensor dimension from MNE info (following existing TRF code)
    ch_info = mne.pick_info(raw.info, meg_picks)
    try:
        sensor_dim = eelbrain.load.mne.sensor_dim(ch_info)
        print("  Using eelbrain sensor dimension")
    except (AttributeError, TypeError):
        sensor_dim = eelbrain.Case
        print("  Using Case dimension (fallback)")

    # Create NDVar
    meg = eelbrain.NDVar(meg_data_array, dims=(sensor_dim, time_dim), name='meg')
    print(f"MEG data shape: {meg.x.shape}")

    # Extract predictors
    print("\nExtracting predictors...")
    predictors = {}

    # TODO: Add "closeness to turn boundary" predictor
    # Compute distance to nearest turn boundary (min of time since last turn, time until next turn)
    # This would capture boundary-proximal vs. boundary-distal processing effects
    # Could test if surprisal/prosodic effects are modulated by turn position
    # Implementation:
    #   - Extract turn boundaries from MFA data (speaker switches)
    #   - For each sample: closeness = min(time_since_last_boundary, time_until_next_boundary)
    #   - Add as continuous predictor: MISC_closeness_to_turn
    #   - Consider interaction terms: surprisal × closeness, f0_deviation × closeness

    # Define all available predictors based on speaker
    if speaker == 'both':
        all_predictor_channels = {
            'envelope_int': 'MISC_envelope_interviewer',
            'word_onsets_int': 'MISC_word_onsets_interviewer',
            'surprisal_int': 'MISC_surprisal_interviewer',
            'f0_deviation_int': 'MISC_f0_deviation_interviewer',
            'duration_deviation_int': 'MISC_duration_deviation_interviewer',
            'pause_int': 'MISC_pause_interviewer',
            'envelope_part': 'MISC_envelope_participant',
            'word_onsets_part': 'MISC_word_onsets_participant',
            'surprisal_part': 'MISC_surprisal_participant',
            'f0_deviation_part': 'MISC_f0_deviation_participant',
            'duration_deviation_part': 'MISC_duration_deviation_participant',
            'pause_part': 'MISC_pause_participant',
        }
    elif speaker == 'interviewer':
        all_predictor_channels = {
            'envelope': 'MISC_envelope_interviewer',
            'word_onsets': 'MISC_word_onsets_interviewer',
            'surprisal': 'MISC_surprisal_interviewer',
            'f0_deviation': 'MISC_f0_deviation_interviewer',
            'duration_deviation': 'MISC_duration_deviation_interviewer',
            'pause': 'MISC_pause_interviewer',
        }
    else:  # participant
        all_predictor_channels = {
            'envelope': 'MISC_envelope_participant',
            'word_onsets': 'MISC_word_onsets_participant',
            'surprisal': 'MISC_surprisal_participant',
            'f0_deviation': 'MISC_f0_deviation_participant',
            'duration_deviation': 'MISC_duration_deviation_participant',
            'pause': 'MISC_pause_participant',
        }

    # Filter predictors if specified
    if predictors_to_use is not None:
        print(f"Using subset of predictors: {', '.join(predictors_to_use)}")
        predictor_channels = {}
        for name, ch in all_predictor_channels.items():
            # Extract base predictor name (strip _int/_part suffix for 'both' speaker)
            base_name = name.replace('_int', '').replace('_part', '')
            if base_name in predictors_to_use:
                predictor_channels[name] = ch
    else:
        predictor_channels = all_predictor_channels

    # Get available channel names from raw file
    available_channels = raw.ch_names

    for name, ch in predictor_channels.items():
        if ch in available_channels:
            # Extract this channel data as numpy array
            ch_idx = raw.ch_names.index(ch)
            pred_data, _ = raw[ch_idx, :]

            # Convert to eelbrain NDVar (1D time series)
            predictors[name] = eelbrain.NDVar(pred_data[0], dims=(time_dim,), name=name)

            # Check non-zero values
            n_nonzero = np.sum(pred_data != 0)
            print(f"  {name:25s}: {n_nonzero:6d} non-zero samples")
        else:
            print(f"  WARNING: {ch} not found")

    if len(predictors) == 0:
        print("\n✗ ERROR: No predictors found!")
        return None

    # Convert predictors dict to tuple (eelbrain expects tuple/list)
    predictor_names = list(predictors.keys())
    predictor_ndvars = tuple(predictors.values())

    # Fit TRF model
    print("\n" + "="*70)
    print("FITTING TRF MODEL")
    print("="*70)

    # Estimate time based on data duration and sampling rate
    # Empirically: ~0.08 min/min at 1000 Hz, scales linearly with sampling rate
    data_duration_min = meg_data_array.shape[1] / raw.info['sfreq'] / 60
    time_per_min_factor = 0.08 * (raw.info['sfreq'] / 1000)  # Scale with sampling rate
    estimated_time_min = data_duration_min * time_per_min_factor

    print(f"Data duration: {data_duration_min:.1f} minutes")
    print(f"Sampling rate: {raw.info['sfreq']:.0f} Hz")
    if estimated_time_min < 1:
        print(f"Estimated fitting time: <1 minute")
    else:
        print(f"Estimated fitting time: ~{estimated_time_min:.1f} minutes")
    print("(Progress updates will appear below)")
    print("\nParameters:")
    print(f"  - Fitting window: {tstart*1000:.0f}ms to {tstop*1000:.0f}ms")
    print(f"  - Saved window: {crop_start*1000:.0f}ms to {crop_stop*1000:.0f}ms (edges cropped)")
    print("  - Basis function width: 50ms")
    print("  - Cross-validation: 5-fold")
    print("  - Error metric: L1 (robust to outliers)")
    print("  - Selective stopping: True")

    try:
        trf = eelbrain.boosting(
            meg,
            predictor_ndvars,  # Pass as tuple, not dict
            tstart=tstart,  # User-configurable fitting window start
            tstop=tstop,    # User-configurable fitting window end
            basis=0.050,    # 50ms basis function
            error='l1',     # L1 error (robust)
            partitions=5,   # 5-fold cross-validation
            selective_stopping=True,
        )
    except Exception as e:
        print(f"\n✗ ERROR fitting TRF: {e}")
        import traceback
        traceback.print_exc()
        return None

    print("\n✓ TRF model fitted!")

    # Diagnostic: Check time range BEFORE cropping
    h_before = trf.h[0] if isinstance(trf.h, tuple) else trf.h
    times_before = h_before.time.times if hasattr(h_before.time, 'times') else h_before.time
    print(f"\n  BEFORE cropping:")
    print(f"    Requested fit window: {tstart*1000:.1f} to {tstop*1000:.1f} ms")
    print(f"    Actual time range: {times_before[0]*1000:.1f} to {times_before[-1]*1000:.1f} ms")
    print(f"    N time points: {len(times_before)}")
    if len(times_before) > 1:
        basis_spacing = times_before[1] - times_before[0]
        print(f"    Basis spacing: {basis_spacing*1000:.1f} ms (basis width: 50ms)")
    print(f"    Note: Basis functions are centered at discrete intervals (50ms)")

    # Crop TRF kernels to remove edge artifacts
    print("\n  Cropping edge artifacts...")
    print(f"    Fit window: {tstart*1000:.0f} to {tstop*1000:.0f}ms")
    print(f"    Saved/visualization window: {crop_start*1000:.0f} to {crop_stop*1000:.0f}ms")

    # Use saved window for visualization (no additional buffering)
    viz_start = crop_start
    viz_stop = crop_stop

    # Crop to user-specified window
    if isinstance(trf.h, tuple):
        h_list_cropped = []
        for i, h in enumerate(trf.h):
            print(f"    Cropping predictor {i}...")
            h_cropped = h.sub(time=(crop_start, crop_stop))
            h_list_cropped.append(h_cropped)
        # Replace with cropped versions
        trf.h = tuple(h_list_cropped)
    else:
        h_cropped = trf.h.sub(time=(crop_start, crop_stop))
        trf.h = h_cropped

    # Diagnostic: Check time range AFTER cropping
    h_after = trf.h[0] if isinstance(trf.h, tuple) else trf.h
    times_after = h_after.time.times if hasattr(h_after.time, 'times') else h_after.time
    print(f"\n  AFTER cropping:")
    print(f"    Time range: {times_after[0]*1000:.1f} to {times_after[-1]*1000:.1f} ms")
    print(f"    N time points: {len(times_after)}")
    print(f"  ✓ Edge artifacts removed")

    # Get mean correlation across sensors
    r_mean = trf.r.mean() if hasattr(trf.r, 'mean') else float(trf.r)
    r_max = trf.r.max() if hasattr(trf.r, 'max') else float(trf.r)

    print(f"  Cross-validated r (mean): {r_mean:.4f}")
    print(f"  Cross-validated r (max):  {r_max:.4f}")
    print(f"  Cross-validated r² (mean): {r_mean**2:.4f}")

    # Save results
    output_dir = Path('outputs/trf_analysis') / subject / condition / speaker
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving results to: {output_dir}")

    # Save TRF model
    with open(output_dir / 'trf_model.pickle', 'wb') as f:
        pickle.dump(trf, f)
    print(f"  ✓ Saved TRF model")

    # Find and print peak latencies
    print("\nTRF Peak Latencies:")
    # When multiple predictors passed as tuple, trf.h is a tuple indexed by position
    if isinstance(trf.h, tuple):
        h_list = trf.h
    else:
        h_list = [trf.h]

    for i, (pred_name, h) in enumerate(zip(predictor_names, h_list)):
        # Diagnostic: show time range of this kernel
        h_times = h.time.times if hasattr(h.time, 'times') else h.time
        print(f"\n  {pred_name}:")
        print(f"    Kernel time range: {h_times[0]*1000:.1f} to {h_times[-1]*1000:.1f} ms")

        # Find peak across all sensors and time points
        # For each time point, find max absolute value across sensors
        max_across_sensors = np.abs(h.x).max(axis=0)
        # Find which time point has the highest value
        peak_idx = np.argmax(max_across_sensors)
        peak_time = h_times[peak_idx]
        peak_val = max_across_sensors[peak_idx]
        print(f"    Global peak: {peak_time*1000:6.1f}ms (amplitude: {peak_val:.4f})")

        # Also check M100 window (80-150ms) if it exists
        m100_mask = (h_times >= 0.08) & (h_times <= 0.15)
        if np.any(m100_mask):
            m100_max = max_across_sensors[m100_mask].max()
            m100_idx = np.where(m100_mask)[0][np.argmax(max_across_sensors[m100_mask])]
            m100_time = h_times[m100_idx]
            print(f"    M100 window peak: {m100_time*1000:.1f}ms (amplitude: {m100_max:.4f})")

    # Generate plots
    if save_plots:
        print("\nGenerating TRF plots...")
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt

        for i, (pred_name, h) in enumerate(zip(predictor_names, h_list)):
            try:
                # Use matplotlib backend for static plots
                p = eelbrain.plot.TopoButterfly(h, vmax=None)
                p.save(output_dir / f'trf_{pred_name}.png', dpi=300)
                print(f"  ✓ Saved: trf_{pred_name}.png")
                p.close()
            except Exception as e:
                print(f"  ✗ Error plotting {pred_name}: {e}")

            # Always create 3-panel plot (polarity-aligned + best sensor + RMS)
            try:
                fig, axes = plt.subplots(3, 1, figsize=(12, 14))
                times = h.time.times if hasattr(h.time, 'times') else h.time
                kernel_data = h.x  # Shape: (n_sensors, n_times)

                # Crop visualization window to avoid edge artifacts
                # Use calculated viz window (removes edges from saved window)
                viz_mask = (times >= viz_start) & (times <= viz_stop)
                times_viz = times[viz_mask]
                kernel_data_viz = kernel_data[:, viz_mask]

                # Find best sensors (top 5 by RMS across visualization window)
                sensor_rms = np.sqrt(np.mean(kernel_data_viz**2, axis=1))
                best_sensor_indices = np.argsort(sensor_rms)[-5:][::-1]  # Top 5 in descending order
                best_sensor_idx = best_sensor_indices[0]  # Best sensor

                # Panel 1: Polarity-aligned mean
                ax = axes[0]
                aligned_mean, n_flipped = align_sensor_polarities(kernel_data_viz, times_viz)

                # Plot ONLY polarity-aligned mean (no individual sensors to avoid y-axis scaling issues)
                ax.plot(times_viz, aligned_mean, linewidth=2.5, color='red', label='Polarity-aligned mean', zorder=3)

                ax.axhline(0, color='k', linestyle='--', alpha=0.3)
                ax.axvline(0, color='k', linestyle='--', alpha=0.3)
                ax.set_ylabel('TRF amplitude')
                ax.set_title(f'{pred_name} - Polarity-Aligned Mean\n({n_flipped}/{kernel_data_viz.shape[0]} sensors flipped based on M100 window)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                # Set y-axis limits based on aligned mean (with 20% padding)
                mean_range = np.max(np.abs(aligned_mean))
                ax.set_ylim(-mean_range * 1.2, mean_range * 1.2)

                # Panel 2: Best sensors
                ax = axes[1]
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Distinct colors
                for i, sensor_idx in enumerate(best_sensor_indices):
                    ax.plot(times_viz, kernel_data_viz[sensor_idx, :],
                           linewidth=2, color=colors[i],
                           label=f'Sensor #{sensor_idx} (rank {i+1})',
                           alpha=0.8, zorder=5-i)

                ax.axhline(0, color='k', linestyle='--', alpha=0.3)
                ax.axvline(0, color='k', linestyle='--', alpha=0.3)
                ax.set_ylabel('TRF amplitude')
                ax.set_title(f'{pred_name} - Top 5 Sensors by RMS')
                ax.legend(loc='best', fontsize=9)
                ax.grid(True, alpha=0.3)

                # Panel 3: RMS magnitude (polarity-independent)
                ax = axes[2]
                rms = compute_rms_across_sensors(kernel_data_viz)

                ax.plot(times_viz, rms, linewidth=2, color='purple', label='RMS across sensors')
                ax.axvline(0, color='k', linestyle='--', alpha=0.3)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('RMS amplitude')
                ax.set_title(f'{pred_name} - RMS Magnitude (polarity-independent)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_ylim(bottom=0)  # RMS is always non-negative

                plt.tight_layout()
                fig.savefig(output_dir / f'trf_{pred_name}_dual.png', dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"  ✓ Saved 3-panel plot: trf_{pred_name}_dual.png (viz: {viz_start*1000:.0f} to {viz_stop*1000:.0f}ms, {n_flipped}/{kernel_data_viz.shape[0]} sensors flipped)")
            except Exception as e2:
                print(f"  ✗ 3-panel plot failed: {e2}")

    # Report timing
    elapsed_time = time.time() - start_time
    print(f"\n⏱ Analysis completed in {elapsed_time/60:.1f} minutes ({elapsed_time:.0f}s)")

    return trf


def compare_conditions(subject, speaker='participant', predictors_to_use=None,
                      tstart=-0.2, tstop=0.8, crop_start=-0.1, crop_stop=0.6):
    """
    Compare TRF between conversation and nursery rhyme.

    Parameters
    ----------
    subject : str
        Subject ID
    speaker : str
        'participant', 'interviewer', or 'both'
    predictors_to_use : list of str, optional
        Subset of predictors to use
    tstart : float
        TRF fitting window start in seconds (default: -0.2)
    tstop : float
        TRF fitting window end in seconds (default: 0.8)
    crop_start : float
        Cropped/saved window start in seconds (default: -0.1)
    crop_stop : float
        Cropped/saved window end in seconds (default: 0.6)

    Returns
    -------
    trf_conv, trf_nursery : tuple of TRF objects
    """
    start_time = time.time()

    print(f"\n{'='*70}")
    print(f"COMPARING CONDITIONS - {speaker.upper()}")
    print(f"{'='*70}\n")

    # Analyze both conditions
    trf_conv = analyze_condition(subject, 'conversation', speaker=speaker, predictors_to_use=predictors_to_use,
                                tstart=tstart, tstop=tstop, crop_start=crop_start, crop_stop=crop_stop)
    trf_nursery = analyze_condition(subject, 'nursery_rhyme', speaker=speaker, predictors_to_use=predictors_to_use,
                                   tstart=tstart, tstop=tstop, crop_start=crop_start, crop_stop=crop_stop)

    if trf_conv is None or trf_nursery is None:
        print("\n✗ ERROR: Failed to fit one or both conditions")
        return None, None

    # Compare model performance
    print("\n" + "="*70)
    print("CONDITION COMPARISON")
    print("="*70)

    # Get mean r values
    r_conv = trf_conv.r.mean() if hasattr(trf_conv.r, 'mean') else float(trf_conv.r)
    r_nursery = trf_nursery.r.mean() if hasattr(trf_nursery.r, 'mean') else float(trf_nursery.r)

    print(f"\nModel Performance:")
    print(f"  Conversation r²:    {r_conv**2:.4f}")
    print(f"  Nursery Rhyme r²:   {r_nursery**2:.4f}")
    print(f"  Difference:         {(r_conv**2 - r_nursery**2):.4f}")

    # Compare predictor effects
    # Predictors are: word_onsets, surprisal, f0_deviation, duration_deviation, pause
    # Index 1 is surprisal, index 2 is f0_deviation

    # Get TRF kernels as lists (handle both tuple and single)
    conv_h = trf_conv.h if isinstance(trf_conv.h, tuple) else [trf_conv.h]
    nursery_h = trf_nursery.h if isinstance(trf_nursery.h, tuple) else [trf_nursery.h]

    # Compare surprisal effects (index 1)
    if len(conv_h) > 1 and len(nursery_h) > 1:
        conv_peak = np.abs(conv_h[1].x).max()  # surprisal is index 1
        nursery_peak = np.abs(nursery_h[1].x).max()

        print(f"\nSurprisal Effect Amplitude:")
        print(f"  Conversation:       {conv_peak:.4f}")
        print(f"  Nursery Rhyme:      {nursery_peak:.4f}")
        print(f"  Ratio (Conv/Nurs):  {conv_peak/nursery_peak:.2f}x")

        # Expected: Conversation should have stronger surprisal effects
        if conv_peak > nursery_peak:
            print(f"  ✓ As expected: stronger surprisal in conversation")
        else:
            print(f"  ⚠ Unexpected: stronger surprisal in nursery rhyme")

    # Compare F0 deviation effects (index 2)
    if len(conv_h) > 2 and len(nursery_h) > 2:
        conv_f0 = np.abs(conv_h[2].x).max()  # f0_deviation is index 2
        nursery_f0 = np.abs(nursery_h[2].x).max()

        print(f"\nF0 Deviation Effect Amplitude:")
        print(f"  Conversation:       {conv_f0:.4f}")
        print(f"  Nursery Rhyme:      {nursery_f0:.4f}")
        print(f"  Ratio (Conv/Nurs):  {conv_f0/nursery_f0:.2f}x")

    # Report total timing
    total_time = time.time() - start_time
    print(f"\n⏱ Total comparison completed in {total_time/60:.1f} minutes ({total_time:.0f}s)")

    return trf_conv, trf_nursery


def main():
    parser = argparse.ArgumentParser(
        description='TRF analysis on combined conversation data'
    )
    parser.add_argument(
        'subject',
        help='Subject ID (e.g., sub-01)'
    )
    parser.add_argument(
        '--condition',
        choices=['conversation', 'nursery_rhyme'],
        help='Condition to analyze (if not comparing)'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare both conditions'
    )
    parser.add_argument(
        '--speaker',
        choices=['participant', 'interviewer', 'both'],
        default='participant',
        help='Which speaker to analyze (default: participant)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip saving TRF plots'
    )
    parser.add_argument(
        '--predictors',
        nargs='+',
        choices=['envelope', 'word_onsets', 'surprisal', 'f0_deviation', 'duration_deviation', 'pause'],
        help='Subset of predictors to use (default: all available)'
    )
    parser.add_argument(
        '--tstart',
        type=float,
        default=-0.2,
        help='TRF fitting window start in seconds (default: -0.2 = -200ms)'
    )
    parser.add_argument(
        '--tstop',
        type=float,
        default=0.8,
        help='TRF fitting window end in seconds (default: 0.8 = +800ms)'
    )
    parser.add_argument(
        '--crop-start',
        type=float,
        default=-0.1,
        help='Cropped/saved window start in seconds (default: -0.1 = -100ms)'
    )
    parser.add_argument(
        '--crop-stop',
        type=float,
        default=0.6,
        help='Cropped/saved window end in seconds (default: 0.6 = +600ms)'
    )

    args = parser.parse_args()

    overall_start = time.time()

    if args.compare:
        # Compare both conditions
        trf_conv, trf_nursery = compare_conditions(
            args.subject,
            speaker=args.speaker,
            predictors_to_use=args.predictors,
            tstart=args.tstart,
            tstop=args.tstop,
            crop_start=args.crop_start,
            crop_stop=args.crop_stop
        )

        if trf_conv is not None and trf_nursery is not None:
            print("\n" + "="*70)
            print("COMPARISON COMPLETE")
            print("="*70)
            print(f"\nResults saved to: outputs/trf_analysis/{args.subject}/")

    elif args.condition:
        # Analyze single condition
        trf = analyze_condition(
            args.subject,
            args.condition,
            speaker=args.speaker,
            save_plots=not args.no_plots,
            predictors_to_use=args.predictors,
            tstart=args.tstart,
            tstop=args.tstop,
            crop_start=args.crop_start,
            crop_stop=args.crop_stop
        )

        if trf is not None:
            print("\n" + "="*70)
            print("ANALYSIS COMPLETE")
            print("="*70)
            print(f"\nResults saved to: outputs/trf_analysis/{args.subject}/{args.condition}/{args.speaker}/")

    else:
        print("ERROR: Must specify either --condition or --compare")
        parser.print_help()
        return 1

    print("\nNext steps:")
    print("  1. Check TRF plots in outputs/trf_analysis/{subject}/{condition}/{speaker}/")
    print("  2. Look for expected patterns:")
    print("     - Word onsets: ~50-150ms (auditory N1/P2)")
    print("     - Surprisal: ~200-400ms (N400-like)")
    print("     - F0 deviation: ~100-200ms (auditory)")
    print("     - Stronger effects in conversation vs. nursery rhyme")

    # Final timing report
    total_elapsed = time.time() - overall_start
    print(f"\n{'='*70}")
    print(f"⏱ TOTAL RUNTIME: {total_elapsed/60:.1f} minutes ({total_elapsed:.0f}s)")
    print(f"{'='*70}")

    # Estimate for all subjects
    if args.compare:
        print(f"\n📊 Estimated time for all subjects:")
        for n_subjects in [5, 10, 20]:
            estimated = (total_elapsed * n_subjects) / 60
            print(f"  {n_subjects} subjects: ~{estimated:.0f} minutes ({estimated/60:.1f} hours)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
