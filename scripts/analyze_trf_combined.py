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
            predictors,
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
    print(f"  Cross-validated r: {trf.r:.4f}")
    print(f"  Cross-validated r²: {trf.r**2:.4f}")

    # Save results
    output_dir = Path('outputs/trf_analysis') / subject / condition / speaker
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving results to: {output_dir}")

    # Save TRF model
    trf.save(output_dir / 'trf_model.pickle')
    print(f"  ✓ Saved TRF model")

    # Find and print peak latencies
    print("\nTRF Peak Latencies:")
    for pred_name in predictors.keys():
        if pred_name in trf.h_scaled:
            h = trf.h_scaled[pred_name]
            # Find peak across all sensors
            peak_idx = np.argmax(np.abs(h.x).max(axis=1))
            peak_time = h.time[peak_idx]
            peak_val = np.abs(h.x).max()
            print(f"  {pred_name:25s}: {peak_time*1000:6.1f}ms (amplitude: {peak_val:.4f})")

    # Generate plots
    if save_plots:
        print("\nGenerating TRF plots...")
        for pred_name in predictors.keys():
            if pred_name in trf.h:
                try:
                    fig = eelbrain.plot.TopoButterfly(trf.h[pred_name])
                    fig.save(output_dir / f'trf_{pred_name}.png', dpi=300)
                    print(f"  ✓ Saved: trf_{pred_name}.png")
                    fig.close()
                except Exception as e:
                    print(f"  ✗ Error plotting {pred_name}: {e}")

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
    print(f"\nModel Performance:")
    print(f"  Conversation r²:    {trf_conv.r**2:.4f}")
    print(f"  Nursery Rhyme r²:   {trf_nursery.r**2:.4f}")
    print(f"  Difference:         {(trf_conv.r**2 - trf_nursery.r**2):.4f}")

    # Compare surprisal effects
    surprisal_key = 'surprisal' if speaker != 'both' else 'surprisal_part'

    if surprisal_key in trf_conv.h_scaled and surprisal_key in trf_nursery.h_scaled:
        conv_peak = np.abs(trf_conv.h_scaled[surprisal_key].x).max()
        nursery_peak = np.abs(trf_nursery.h_scaled[surprisal_key].x).max()

        print(f"\nSurprisal Effect Amplitude:")
        print(f"  Conversation:       {conv_peak:.4f}")
        print(f"  Nursery Rhyme:      {nursery_peak:.4f}")
        print(f"  Ratio (Conv/Nurs):  {conv_peak/nursery_peak:.2f}x")

        # Expected: Conversation should have stronger surprisal effects
        if conv_peak > nursery_peak:
            print(f"  ✓ As expected: stronger surprisal in conversation")
        else:
            print(f"  ⚠ Unexpected: stronger surprisal in nursery rhyme")

    # Compare prosodic deviation effects
    f0_key = 'f0_deviation' if speaker != 'both' else 'f0_deviation_part'

    if f0_key in trf_conv.h_scaled and f0_key in trf_nursery.h_scaled:
        conv_f0 = np.abs(trf_conv.h_scaled[f0_key].x).max()
        nursery_f0 = np.abs(trf_nursery.h_scaled[f0_key].x).max()

        print(f"\nF0 Deviation Effect Amplitude:")
        print(f"  Conversation:       {conv_f0:.4f}")
        print(f"  Nursery Rhyme:      {nursery_f0:.4f}")
        print(f"  Ratio (Conv/Nurs):  {conv_f0/nursery_f0:.2f}x")

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

    return 0


if __name__ == '__main__':
    sys.exit(main())
