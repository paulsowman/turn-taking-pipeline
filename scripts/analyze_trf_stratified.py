#!/usr/bin/env python3
"""
Stratified TRF Analysis by Turn Position

Fits separate TRF models for different positions within turns to test
the hypothesis that surprisal sensitivity decreases as listeners prepare
to take their turn.

Stratification:
  - FAR from boundary: proportion < 0.5 (first half of turns)
  - CLOSE to boundary: proportion >= 0.5 (second half of turns)

Usage:
    python scripts/analyze_trf_stratified.py sub-01 --condition conversation --speaker interviewer

Outputs:
    trf_stratified_{predictor}_comparison.png  - Side-by-side kernel comparison
    trf_stratified_far_model.pickle           - TRF model for FAR condition
    trf_stratified_close_model.pickle         - TRF model for CLOSE condition
"""
import eelbrain
import mne
import numpy as np
from pathlib import Path
import argparse
import sys
import pickle
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def create_stratification_masks(proportion_data, meg_times):
    """
    Create boolean masks for stratifying data by turn position.

    Parameters
    ----------
    proportion_data : ndarray
        Proportion-through-turn predictor (0-1 during listening, NaN elsewhere)
    meg_times : ndarray
        MEG time axis

    Returns
    -------
    far_mask : ndarray
        Boolean mask for first half of turns (proportion < 0.5)
    close_mask : ndarray
        Boolean mask for second half of turns (proportion >= 0.5)
    """
    # Filter out NaN values (periods when not listening)
    valid_listening = ~np.isnan(proportion_data)

    # Create masks for stratification
    far_mask = valid_listening & (proportion_data < 0.5)
    close_mask = valid_listening & (proportion_data >= 0.5)

    return far_mask, close_mask


def analyze_stratified_trf(
    subject,
    condition,
    speaker='interviewer',
    predictors_to_use=None,
    tstart=-0.2,
    tstop=0.6,
):
    """
    Fit stratified TRF models for far vs. close to turn boundaries.

    Parameters
    ----------
    subject : str
        Subject ID (e.g., 'sub-01')
    condition : str
        Condition ('conversation' or 'nursery_rhyme')
    speaker : str
        Which speaker to analyze ('interviewer' or 'participant')
    predictors_to_use : list or None
        List of predictor names to use (None = all available)
    tstart : float
        TRF fitting window start in seconds
    tstop : float
        TRF fitting window end in seconds
    """
    print("="*70)
    print("STRATIFIED TRF ANALYSIS")
    print("="*70)
    print(f"Subject: {subject}")
    print(f"Condition: {condition}")
    print(f"Speaker: {speaker}")
    print(f"Stratification: proportion-through-turn (< 0.5 vs >= 0.5)")

    # Load combined FIF file
    fif_file = Path(f'outputs/trf_combined/{subject}/{subject}_{condition}_trf_raw.fif')

    if not fif_file.exists():
        print(f"\n✗ ERROR: Combined FIF file not found: {fif_file}")
        print("Run combine_runs.py first")
        return None

    print(f"\nLoading: {fif_file}")
    raw = mne.io.read_raw_fif(fif_file, preload=True, verbose=False)

    # Get proportion-through-turn predictor
    proportion_channel = f'MISC_proportion_through_turn_{speaker}'

    if proportion_channel not in raw.ch_names:
        print(f"\n✗ ERROR: Channel {proportion_channel} not found")
        print("Regenerate data with: python scripts/create_trf_fif.py --subject <sub> --runs 1 2 3 4 5")
        return None

    # Extract proportion predictor
    print(f"\nExtracting stratification predictor: {proportion_channel}")
    ch_idx = raw.ch_names.index(proportion_channel)
    proportion_data, meg_times_array = raw[ch_idx, :]
    proportion_data = proportion_data[0]  # Shape: (1, n_times) -> (n_times,)
    meg_times_array = meg_times_array[0] if len(meg_times_array.shape) > 1 else meg_times_array

    # Create stratification masks
    far_mask, close_mask = create_stratification_masks(proportion_data, meg_times_array)

    # Report stratification statistics
    n_total = len(proportion_data)
    n_far = np.sum(far_mask)
    n_close = np.sum(close_mask)
    duration_far = n_far / raw.info['sfreq']
    duration_close = n_close / raw.info['sfreq']

    print("\n" + "="*70)
    print("STRATIFICATION STATISTICS")
    print("="*70)
    print(f"Total samples: {n_total}")
    print(f"  FAR (first half):   {n_far:6d} samples ({duration_far:6.1f}s) - {n_far/n_total*100:5.1f}%")
    print(f"  CLOSE (second half): {n_close:6d} samples ({duration_close:6.1f}s) - {n_close/n_total*100:5.1f}%")

    # Define predictors based on speaker
    if speaker == 'interviewer':
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
        print(f"\nUsing subset of predictors: {', '.join(predictors_to_use)}")
        predictor_channels = {name: ch for name, ch in all_predictor_channels.items()
                             if name in predictors_to_use}
    else:
        predictor_channels = all_predictor_channels

    # Fit TRF for each stratification level
    results = {}

    for stratum_name, mask in [('far', far_mask), ('close', close_mask)]:
        print("\n" + "="*70)
        print(f"FITTING TRF: {stratum_name.upper()} FROM BOUNDARY")
        print("="*70)

        # Create masked raw object (set non-mask periods to zero)
        raw_masked = raw.copy()

        # Zero out data outside the mask for all channels
        for ch_idx in range(len(raw_masked.ch_names)):
            ch_data = raw_masked.get_data(picks=[ch_idx])
            ch_data[:, ~mask] = 0
            raw_masked._data[ch_idx] = ch_data[0]

        # Extract MEG data
        meg_picks = mne.pick_types(raw_masked.info, meg=True)
        meg_data_array, _ = raw_masked[meg_picks, :]

        # Convert to eelbrain NDVar
        time_dim = eelbrain.UTS(0, 1.0/raw_masked.info['sfreq'], meg_data_array.shape[1])
        sensor_dim = eelbrain.Sensor(
            locs=np.array([raw_masked.info['chs'][i]['loc'][:3] for i in meg_picks]),
            names=[raw_masked.info['ch_names'][i] for i in meg_picks]
        )
        meg = eelbrain.NDVar(meg_data_array, dims=(sensor_dim, time_dim), name='MEG')

        # Extract predictors
        predictors = {}
        print("\nPredictors:")
        for name, ch in predictor_channels.items():
            if ch in raw_masked.ch_names:
                ch_idx = raw_masked.ch_names.index(ch)
                pred_data, _ = raw_masked[ch_idx, :]
                predictors[name] = eelbrain.NDVar(pred_data[0], dims=(time_dim,), name=name)

                # Count non-zero in masked region only
                n_nonzero = np.sum((pred_data[0] != 0) & mask)
                print(f"  {name:25s}: {n_nonzero:6d} non-zero samples in {stratum_name}")
            else:
                print(f"  WARNING: {ch} not found")

        if len(predictors) == 0:
            print(f"\n✗ ERROR: No predictors found for {stratum_name}!")
            continue

        predictor_names = list(predictors.keys())
        predictor_ndvars = tuple(predictors.values())

        # Fit TRF
        print(f"\nFitting TRF for {stratum_name} condition...")
        print(f"  Window: {tstart*1000:.0f}ms to {tstop*1000:.0f}ms")
        print(f"  Data duration: {duration_far if stratum_name == 'far' else duration_close:.1f}s")

        try:
            trf = eelbrain.boosting(
                meg,
                predictor_ndvars,
                tstart=tstart,
                tstop=tstop,
                basis=0.050,
                error='l1',
                partitions=5,
                selective_stopping=True,
            )
            print(f"✓ TRF fitted for {stratum_name}")

            results[stratum_name] = {
                'trf': trf,
                'predictor_names': predictor_names,
                'n_samples': n_far if stratum_name == 'far' else n_close,
                'duration': duration_far if stratum_name == 'far' else duration_close,
            }

        except Exception as e:
            print(f"✗ ERROR fitting TRF for {stratum_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Compare results
    if 'far' in results and 'close' in results:
        print("\n" + "="*70)
        print("COMPARISON: FAR vs CLOSE TO BOUNDARY")
        print("="*70)

        # Create comparison plots
        output_dir = Path(f'outputs/trf_analysis/{subject}/{condition}_stratified_{speaker}')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save models
        for stratum_name, result in results.items():
            model_file = output_dir / f'trf_{stratum_name}_model.pickle'
            with open(model_file, 'wb') as f:
                pickle.dump(result['trf'], f)
            print(f"✓ Saved {stratum_name} model: {model_file}")

        # Create comparison plots for each predictor
        predictor_names = results['far']['predictor_names']

        for pred_idx, pred_name in enumerate(predictor_names):
            print(f"\nComparing {pred_name}...")

            # Extract kernels
            h_far = results['far']['trf'].h[pred_idx] if isinstance(results['far']['trf'].h, tuple) else results['far']['trf'].h
            h_close = results['close']['trf'].h[pred_idx] if isinstance(results['close']['trf'].h, tuple) else results['close']['trf'].h

            # Get times
            times_far = h_far.time.times if hasattr(h_far.time, 'times') else h_far.time
            times_close = h_close.time.times if hasattr(h_close.time, 'times') else h_close.time

            # Extract data (average across sensors for visualization)
            data_far = np.mean(h_far.x, axis=0) if len(h_far.x.shape) > 1 else h_far.x
            data_close = np.mean(h_close.x, axis=0) if len(h_close.x.shape) > 1 else h_close.x

            # Create comparison plot
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            # Panel 1: Overlay
            ax = axes[0]
            ax.plot(times_far * 1000, data_far, color='blue', linewidth=2, label='FAR (first half)', alpha=0.8)
            ax.plot(times_close * 1000, data_close, color='red', linewidth=2, label='CLOSE (second half)', alpha=0.8)
            ax.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.axvline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_ylabel('TRF Amplitude (sensor average)')
            ax.set_title(f'{pred_name.upper()} TRF: Far vs Close to Turn Boundary')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Panel 2: Difference (far - close)
            ax = axes[1]
            # Interpolate to common time axis if needed
            if len(times_far) == len(times_close) and np.allclose(times_far, times_close):
                diff = data_far - data_close
                times_diff = times_far
            else:
                # Use shorter time axis
                min_len = min(len(times_far), len(times_close))
                diff = data_far[:min_len] - data_close[:min_len]
                times_diff = times_far[:min_len]

            ax.plot(times_diff * 1000, diff, color='purple', linewidth=2, label='Difference (far - close)')
            ax.fill_between(times_diff * 1000, 0, diff, color='purple', alpha=0.3)
            ax.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.axvline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Difference in Amplitude')
            ax.set_title('Difference: FAR minus CLOSE')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save plot
            plot_file = output_dir / f'trf_{pred_name}_comparison.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: {plot_file}")

        print("\n" + "="*70)
        print("HYPOTHESIS TEST: SURPRISAL SENSITIVITY")
        print("="*70)

        if 'surprisal' in predictor_names:
            h_far = results['far']['trf'].h[predictor_names.index('surprisal')]
            h_close = results['close']['trf'].h[predictor_names.index('surprisal')]

            # Compute peak amplitudes
            data_far = np.mean(h_far.x, axis=0) if len(h_far.x.shape) > 1 else h_far.x
            data_close = np.mean(h_close.x, axis=0) if len(h_close.x.shape) > 1 else h_close.x

            peak_far = np.max(np.abs(data_far))
            peak_close = np.max(np.abs(data_close))

            times = h_far.time.times if hasattr(h_far.time, 'times') else h_far.time
            peak_time_far = times[np.argmax(np.abs(data_far))] * 1000
            peak_time_close = times[np.argmax(np.abs(data_close))] * 1000

            print(f"\nSurprisal TRF comparison:")
            print(f"  FAR (first half):   Peak = {peak_far:.2e} at {peak_time_far:.0f}ms")
            print(f"  CLOSE (second half): Peak = {peak_close:.2e} at {peak_time_close:.0f}ms")
            print(f"  Reduction: {(1 - peak_close/peak_far)*100:.1f}%")

            if peak_close < peak_far:
                print(f"\n✓ HYPOTHESIS SUPPORTED: Surprisal sensitivity is reduced when close to boundary")
            else:
                print(f"\n✗ HYPOTHESIS NOT SUPPORTED: Surprisal sensitivity is NOT reduced when close to boundary")
        else:
            print("Surprisal predictor not included in analysis")

        print("\n✓ Stratified analysis complete!")
        print(f"Output directory: {output_dir}")

    else:
        print("\n✗ ERROR: Could not fit both stratification levels")
        return None

    return results


def main():
    parser = argparse.ArgumentParser(description='Stratified TRF analysis by turn position')
    parser.add_argument('subject', help='Subject ID (e.g., sub-01)')
    parser.add_argument('--condition', default='conversation',
                       choices=['conversation', 'nursery_rhyme'],
                       help='Condition to analyze')
    parser.add_argument('--speaker', default='interviewer',
                       choices=['interviewer', 'participant'],
                       help='Which speaker to analyze')
    parser.add_argument('--predictors', nargs='+',
                       help='Subset of predictors to use (default: all)')
    parser.add_argument('--tstart', type=float, default=-0.2,
                       help='TRF fitting window start (default: -0.2)')
    parser.add_argument('--tstop', type=float, default=0.6,
                       help='TRF fitting window end (default: 0.6)')

    args = parser.parse_args()

    results = analyze_stratified_trf(
        subject=args.subject,
        condition=args.condition,
        speaker=args.speaker,
        predictors_to_use=args.predictors,
        tstart=args.tstart,
        tstop=args.tstop,
    )

    if results is None:
        sys.exit(1)

    return 0


if __name__ == '__main__':
    sys.exit(main())
