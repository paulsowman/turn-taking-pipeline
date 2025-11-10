#!/usr/bin/env python3
"""
Word-Level TRF Analysis - Combined Conversation Runs

Combines runs 1, 3, 5 (conversation condition) for better statistical power.
~2700 word events instead of ~900.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import mne
import eelbrain
from scipy.interpolate import interp1d
import sys
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils.conditions import CONVERSATION_RUNS


def load_and_concatenate_features(subject: str, runs: list, base_dir: Path):
    """
    Load features from multiple runs and concatenate.

    Parameters
    ----------
    subject : str
        Subject ID
    runs : list
        List of run numbers to combine
    base_dir : Path
        Pipeline base directory

    Returns
    -------
    combined_words : pd.DataFrame
        Concatenated word features with run offset applied
    run_boundaries : dict
        Time boundaries for each run
    """
    all_words = []
    run_boundaries = {}
    cumulative_time = 0.0

    for run in runs:
        feature_dir = base_dir / "outputs" / "features" / subject / f"run-{run:02d}"

        # Load transcript and prosody
        transcript = pd.read_csv(feature_dir / "transcript.csv")
        prosody = pd.read_csv(feature_dir / "prosody.csv")
        turns = pd.read_csv(feature_dir / "turns.csv")

        print(f"\nRun {run}:")
        print(f"  Segments: {len(transcript)}")
        print(f"  Turns: {len(turns)}")

        # Extract words with features
        words_list = []

        for _, seg in transcript.iterrows():
            if 'words' not in seg or pd.isna(seg['words']):
                continue

            words_data = eval(seg['words']) if isinstance(seg['words'], str) else seg['words']
            seg_start_meg = seg['start_meg']
            seg_start_orig = seg['start']

            for w in words_data:
                word_time_orig = w['start']
                word_time_meg = seg_start_meg + (word_time_orig - seg_start_orig)

                # Add cumulative offset for run concatenation
                word_time_combined = word_time_meg + cumulative_time

                words_list.append({
                    'time_meg': word_time_meg,
                    'time_combined': word_time_combined,
                    'word': w['word'],
                    'duration': w['end'] - w['start'],
                    'segment_id': seg['segment_id'],
                    'run': run
                })

        run_words = pd.DataFrame(words_list)
        print(f"  Words: {len(run_words)}")

        # Add prosodic features
        prosody_times = prosody['time'].values

        for feature in ['f0_smooth', 'intensity', 'energy']:
            valid_mask = ~prosody[feature].isna()
            if valid_mask.sum() > 1:
                interp_func = interp1d(
                    prosody_times[valid_mask],
                    prosody[feature].values[valid_mask],
                    kind='nearest',
                    bounds_error=False,
                    fill_value=np.nan
                )
                run_words[feature] = interp_func(run_words['time_meg'].values)
            else:
                run_words[feature] = np.nan

        # Fill NaN
        run_words['f0_smooth'] = run_words['f0_smooth'].fillna(0)
        for feature in ['intensity', 'energy']:
            run_words[feature] = run_words[feature].fillna(run_words[feature].mean())

        # Add turn features
        run_words['turn_idx'] = -1
        run_words['position_in_turn'] = 0.0
        run_words['is_turn_initial'] = False

        for turn_idx, turn in turns.iterrows():
            # Turns.csv already has MEG-aligned times (same as transcript)
            turn_start = turn['start']
            turn_end = turn['end']

            in_turn = (run_words['time_meg'] >= turn_start) & (run_words['time_meg'] <= turn_end)
            run_words.loc[in_turn, 'turn_idx'] = turn_idx

            turn_words = run_words[in_turn]
            if len(turn_words) > 0:
                positions = (turn_words['time_meg'] - turn_start) / turn['duration']
                run_words.loc[in_turn, 'position_in_turn'] = positions

                # Mark first word in turn
                first_word_idx = turn_words.index[0]
                run_words.loc[first_word_idx, 'is_turn_initial'] = True

        # Store run boundary
        run_duration = run_words['time_meg'].max() + 1.0  # Add 1s buffer
        run_boundaries[run] = {
            'start': cumulative_time,
            'end': cumulative_time + run_duration,
            'duration': run_duration
        }

        all_words.append(run_words)
        cumulative_time += run_duration

    # Concatenate all runs
    combined_words = pd.concat(all_words, ignore_index=True)

    print(f"\n{'='*70}")
    print(f"COMBINED DATASET:")
    print(f"  Total runs: {len(runs)}")
    print(f"  Total words: {len(combined_words)}")
    print(f"  Total duration: {cumulative_time:.1f}s")
    print(f"  Runs: {runs}")
    print(f"{'='*70}")

    return combined_words, run_boundaries


def load_roi_sensors(roi_json_path):
    """Load ROI sensor names from JSON file."""
    with open(roi_json_path, 'r') as f:
        roi_data = json.load(f)

    left_sensors = roi_data['left_hemisphere']['names']
    right_sensors = roi_data['right_hemisphere']['names']

    return left_sensors + right_sensors


def load_and_concatenate_meg(subject: str, runs: list, meg_files: list, sensor_type: str = 'mag', roi_sensors_json: str = None):
    """
    Load and concatenate MEG data from multiple runs.

    Parameters
    ----------
    subject : str
        Subject ID
    runs : list
        List of run numbers
    meg_files : list
        List of MEG file paths (same order as runs)
    sensor_type : str
        'mag', 'grad', or 'eeg'

    Returns
    -------
    meg_data_combined : np.ndarray
        Concatenated MEG data (n_sensors, n_samples)
    meg_times_combined : np.ndarray
        Combined time vector
    picks : np.ndarray
        Sensor indices
    ch_info : mne.Info
        Channel info
    sfreq : float
        Sampling frequency
    """
    all_meg_data = []
    cumulative_time = 0.0

    # Load ROI sensors if specified
    roi_sensor_names = None
    if roi_sensors_json is not None:
        roi_sensor_names = load_roi_sensors(roi_sensors_json)
        print(f"\nLoaded ROI: {len(roi_sensor_names)} sensors from {roi_sensors_json}")

    print(f"\nLoading and concatenating MEG data...")

    for i, (run, meg_file) in enumerate(zip(runs, meg_files)):
        print(f"\nRun {run}: {meg_file}")
        meg_raw = mne.io.read_raw_fif(meg_file, preload=True, verbose=False)

        picks = mne.pick_types(meg_raw.info, meg=sensor_type, eeg=(sensor_type=='eeg'),
                               exclude='bads')

        # Apply ROI sensor selection if specified
        if roi_sensor_names is not None:
            # Get channel names for picked sensors
            ch_names = [meg_raw.ch_names[p] for p in picks]
            # Filter to only ROI sensors
            roi_picks = [p for p, ch in zip(picks, ch_names) if ch in roi_sensor_names]
            picks = np.array(roi_picks)
            print(f"  Applied ROI filter: {len(picks)} sensors")

        meg_data, meg_times = meg_raw[picks, :]
        sfreq = meg_raw.info['sfreq']

        print(f"  Sensors: {len(picks)} {sensor_type}")
        print(f"  Duration: {meg_times.max():.1f}s")
        print(f"  Samples: {len(meg_times)}")

        all_meg_data.append(meg_data)
        cumulative_time += meg_times.max()

        if i == 0:
            ch_info = mne.pick_info(meg_raw.info, picks)

    # Concatenate MEG data
    meg_data_combined = np.concatenate(all_meg_data, axis=1)
    meg_times_combined = np.arange(meg_data_combined.shape[1]) / sfreq

    print(f"\nCombined MEG:")
    print(f"  Total duration: {meg_times_combined.max():.1f}s")
    print(f"  Total samples: {len(meg_times_combined)}")

    return meg_data_combined, meg_times_combined, picks, ch_info, sfreq


def create_combined_predictors(words_df, meg_times, sfreq):
    """Create word-level predictors for combined runs."""
    n_samples = len(meg_times)
    predictors = {}

    # Initialize predictor arrays
    f0_predictor = np.zeros(n_samples)
    intensity_predictor = np.zeros(n_samples)
    duration_predictor = np.zeros(n_samples)
    turn_initial_predictor = np.zeros(n_samples)

    print(f"\nCreating predictors for {len(words_df)} words...")

    for _, word in words_df.iterrows():
        # Use combined time
        onset_idx = np.argmin(np.abs(meg_times - word['time_combined']))

        if onset_idx < n_samples:
            f0_predictor[onset_idx] += word['f0_smooth']
            intensity_predictor[onset_idx] += word['intensity']
            duration_predictor[onset_idx] += word['duration']

            if word['is_turn_initial']:
                turn_initial_predictor[onset_idx] = 1.0

    # Normalize
    if f0_predictor.max() > 0:
        f0_predictor = f0_predictor / f0_predictor.max()
    if intensity_predictor.max() > 0:
        intensity_predictor = intensity_predictor / intensity_predictor.max()
    if duration_predictor.max() > 0:
        duration_predictor = duration_predictor / duration_predictor.max()

    predictors['f0'] = f0_predictor
    predictors['intensity'] = intensity_predictor
    predictors['duration'] = duration_predictor
    predictors['turn_initial'] = turn_initial_predictor

    print(f"  F0 impulses: {np.sum(f0_predictor > 0)}")
    print(f"  Intensity impulses: {np.sum(intensity_predictor > 0)}")
    print(f"  Turn-initial impulses: {np.sum(turn_initial_predictor > 0)}")

    return predictors


def main():
    parser = argparse.ArgumentParser(description="Word-level TRF on combined conversation runs")
    parser.add_argument("--subject", required=True, help="Subject ID")
    parser.add_argument("--runs", nargs='+', type=int, default=CONVERSATION_RUNS,
                       help=f"Runs to combine (default: {CONVERSATION_RUNS})")
    parser.add_argument("--meg-files", nargs='+', required=True,
                       help="Paths to MEG files (same order as runs)")
    parser.add_argument("--sensor-type", default="mag",
                       choices=['mag', 'grad', 'eeg'],
                       help="Sensor type")
    parser.add_argument("--roi-sensors", type=str, default=None,
                       help="Path to ROI sensors JSON file (from BADA localizer)")
    parser.add_argument("--base-dir", type=Path,
                       default=Path("/Users/em18033/Library/CloudStorage/OneDrive-AUTUniversity/Projects/turn-taking-pipeline"),
                       help="Pipeline base directory")

    args = parser.parse_args()

    if len(args.meg_files) != len(args.runs):
        raise ValueError(f"Number of MEG files ({len(args.meg_files)}) must match number of runs ({len(args.runs)})")

    print("="*70)
    print("WORD-LEVEL TRF - COMBINED CONVERSATION RUNS")
    print("="*70)
    print(f"\nSubject: {args.subject}")
    print(f"Runs: {args.runs}")
    print(f"Sensor type: {args.sensor_type}")

    # 1. Load and combine features
    words_df, run_boundaries = load_and_concatenate_features(
        args.subject, args.runs, args.base_dir
    )

    # 2. Load and combine MEG
    meg_data, meg_times, picks, ch_info, sfreq = load_and_concatenate_meg(
        args.subject, args.runs, args.meg_files, args.sensor_type, args.roi_sensors
    )

    # 3. Create predictors
    predictors = create_combined_predictors(words_df, meg_times, sfreq)

    # 4. Convert to eelbrain
    print("\n" + "="*70)
    print("CONVERTING TO EELBRAIN FORMAT")
    print("="*70)

    time_dim = eelbrain.UTS(0, 1/sfreq, meg_data.shape[1])

    try:
        sensor_dim = eelbrain.load.mne.sensor_dim(ch_info)
    except (AttributeError, TypeError):
        print("  Using simple sensor indexing...")
        sensor_dim = eelbrain.Case

    meg_ndvar = eelbrain.NDVar(
        meg_data,
        dims=(sensor_dim, time_dim),
        info={'subject': args.subject},
        name='MEG'
    )

    pred_ndvars = {}
    for name, data in predictors.items():
        pred_ndvars[name] = eelbrain.NDVar(
            data,
            dims=(time_dim,),
            name=name.replace('_', ' ').title()
        )

    print(f"  MEG shape: {meg_ndvar.shape}")
    print(f"  Predictors: {list(pred_ndvars.keys())}")

    # 5. Fit TRF
    print("\n" + "="*70)
    print("FITTING TRF MODEL")
    print("="*70)

    selected_predictors = ['f0', 'intensity', 'turn_initial']
    print(f"\nUsing predictors: {selected_predictors}")

    print("\nFitting TRF (this may take 5-10 minutes for combined data)...")
    trf = eelbrain.boosting(
        meg_ndvar,
        [pred_ndvars[p] for p in selected_predictors],
        tstart=-0.1,
        tstop=0.5,
        scale_data=True,
        delta=0.005,
        mindelta=0.0005,
        partitions=5,
        test=1,
        selective_stopping=True,
        error='l1',
    )

    print("\n✓ TRF fitting complete!")
    print(f"  Model correlation: {trf.r.mean():.3f}")
    print(f"  Max correlation: {trf.r.max():.3f}")

    # 6. Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    runs_str = '+'.join(map(str, args.runs))
    output_dir = args.base_dir / "outputs" / "trf_word_combined" / args.subject / f"runs-{runs_str}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save TRF model
    import pickle
    with open(output_dir / "trf_model.pkl", 'wb') as f:
        pickle.dump(trf, f)
    print(f"✓ Saved: {output_dir / 'trf_model.pkl'}")

    # Save word features
    words_df.to_csv(output_dir / "word_features_combined.csv", index=False)
    print(f"✓ Saved: {output_dir / 'word_features_combined.csv'}")

    # Save run boundaries
    import json
    with open(output_dir / "run_boundaries.json", 'w') as f:
        json.dump(run_boundaries, f, indent=2)
    print(f"✓ Saved: {output_dir / 'run_boundaries.json'}")

    # Save summary
    summary = {
        'subject': args.subject,
        'runs': args.runs,
        'sensor_type': args.sensor_type,
        'n_sensors': len(picks),
        'n_words': len(words_df),
        'n_runs': len(args.runs),
        'total_duration_s': float(meg_times.max()),
        'model_correlation_mean': float(trf.r.mean()),
        'model_correlation_max': float(trf.r.max()),
        'predictors': selected_predictors,
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved: {output_dir / 'summary.json'}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults: {output_dir}")
    print(f"\nCombined dataset:")
    print(f"  Runs: {args.runs}")
    print(f"  Words: {len(words_df)}")
    print(f"  Duration: {meg_times.max():.1f}s")
    print(f"\nModel performance:")
    print(f"  Mean correlation: {trf.r.mean():.3f}")
    print(f"  Max correlation: {trf.r.max():.3f}")

    return summary


if __name__ == "__main__":
    main()
