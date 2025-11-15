#!/usr/bin/env python3
"""
TRF Analysis on Combined Conversation Data

Fits Temporal Response Function (TRF) models using eelbrain's boosted regression
to relate linguistic/prosodic predictors to MEG responses.

Usage:
    # Analyze single condition
    python scripts/analyze_trf_combined.py sub-01 --condition conversation

    # Analyze both conditions and compare
    python scripts/analyze_trf_combined.py sub-01 --compare

    # Analyze both speakers
    python scripts/analyze_trf_combined.py sub-01 --compare --speaker both
"""
import eelbrain
import mne
import numpy as np
from pathlib import Path
import argparse
import sys
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def analyze_condition(subject, condition, speaker='participant', save_plots=True):
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
    save_plots : bool
        Whether to save TRF plots

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

    # Define which predictors to use based on speaker
    if speaker == 'both':
        predictor_channels = {
            'word_onsets_int': 'MISC_word_onsets_interviewer',
            'surprisal_int': 'MISC_surprisal_interviewer',
            'f0_deviation_int': 'MISC_f0_deviation_interviewer',
            'duration_deviation_int': 'MISC_duration_deviation_interviewer',
            'pause_int': 'MISC_pause_interviewer',
            'word_onsets_part': 'MISC_word_onsets_participant',
            'surprisal_part': 'MISC_surprisal_participant',
            'f0_deviation_part': 'MISC_f0_deviation_participant',
            'duration_deviation_part': 'MISC_duration_deviation_participant',
            'pause_part': 'MISC_pause_participant',
        }
    elif speaker == 'interviewer':
        predictor_channels = {
            'word_onsets': 'MISC_word_onsets_interviewer',
            'surprisal': 'MISC_surprisal_interviewer',
            'f0_deviation': 'MISC_f0_deviation_interviewer',
            'duration_deviation': 'MISC_duration_deviation_interviewer',
            'pause': 'MISC_pause_interviewer',
        }
    else:  # participant
        predictor_channels = {
            'word_onsets': 'MISC_word_onsets_participant',
            'surprisal': 'MISC_surprisal_participant',
            'f0_deviation': 'MISC_f0_deviation_participant',
            'duration_deviation': 'MISC_duration_deviation_participant',
            'pause': 'MISC_pause_participant',
        }

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
    print("This may take 5-15 minutes depending on data length...")
    print("\nParameters:")
    print("  - Time window: -100ms to +600ms")
    print("  - Basis function width: 50ms")
    print("  - Cross-validation: 5-fold")
    print("  - Error metric: L1 (robust to outliers)")
    print("  - Selective stopping: True")

    try:
        trf = eelbrain.boosting(
            meg,
            predictor_ndvars,  # Pass as tuple, not dict
            tstart=-0.100,  # Start 100ms before predictor
            tstop=0.600,    # End 600ms after predictor
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
    trf.save(output_dir / 'trf_model.pickle')
    print(f"  ✓ Saved TRF model")

    # Find and print peak latencies
    print("\nTRF Peak Latencies:")
    # When multiple predictors passed as tuple, trf.h is a tuple indexed by position
    if isinstance(trf.h, tuple):
        h_list = trf.h
    else:
        h_list = [trf.h]

    for i, (pred_name, h) in enumerate(zip(predictor_names, h_list)):
        # Find peak across all sensors
        peak_idx = np.argmax(np.abs(h.x).max(axis=1))
        peak_time = h.time.times[peak_idx] if hasattr(h.time, 'times') else h.time[peak_idx]
        peak_val = np.abs(h.x).max()
        print(f"  {pred_name:25s}: {peak_time*1000:6.1f}ms (amplitude: {peak_val:.4f})")

    # Generate plots
    if save_plots:
        print("\nGenerating TRF plots...")
        for i, (pred_name, h) in enumerate(zip(predictor_names, h_list)):
            try:
                fig = eelbrain.plot.TopoButterfly(h)
                fig.save(output_dir / f'trf_{pred_name}.png', dpi=300)
                print(f"  ✓ Saved: trf_{pred_name}.png")
                fig.close()
            except Exception as e:
                print(f"  ✗ Error plotting {pred_name}: {e}")

    # Report timing
    elapsed_time = time.time() - start_time
    print(f"\n⏱ Analysis completed in {elapsed_time/60:.1f} minutes ({elapsed_time:.0f}s)")

    return trf


def compare_conditions(subject, speaker='participant'):
    """
    Compare TRF between conversation and nursery rhyme.

    Parameters
    ----------
    subject : str
        Subject ID
    speaker : str
        'participant', 'interviewer', or 'both'

    Returns
    -------
    trf_conv, trf_nursery : tuple of TRF objects
    """
    start_time = time.time()

    print(f"\n{'='*70}")
    print(f"COMPARING CONDITIONS - {speaker.upper()}")
    print(f"{'='*70}\n")

    # Analyze both conditions
    trf_conv = analyze_condition(subject, 'conversation', speaker=speaker)
    trf_nursery = analyze_condition(subject, 'nursery_rhyme', speaker=speaker)

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

    args = parser.parse_args()

    overall_start = time.time()

    if args.compare:
        # Compare both conditions
        trf_conv, trf_nursery = compare_conditions(args.subject, speaker=args.speaker)

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
            save_plots=not args.no_plots
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
