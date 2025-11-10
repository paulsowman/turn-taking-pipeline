#!/usr/bin/env python3
"""
Word-Level TRF Analysis - CORRECTED EVENT STRUCTURE

This implements TRF using word onsets as events, with scalar predictors
at each word onset. This is the standard approach for psycholinguistic TRF.

Key differences from previous (incorrect) sensor-space TRF:
1. Events = word onsets (~899 per run vs 85 turn onsets)
2. Predictors = scalar values AT each word onset
3. Proper event-predictor structure for TRF
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import mne
import eelbrain
from scipy.interpolate import interp1d


def load_word_features(subject: str, run: int, base_dir: Path):
    """
    Load word-level features from pipeline outputs.

    Returns DataFrame with one row per word containing:
    - time_meg: word onset time in MEG coordinates
    - word: word text
    - duration: word duration
    - f0: F0 at word onset
    - intensity: intensity at word onset
    - position_in_turn: relative position (0=start, 1=end)
    - turn_position: which turn this word is in
    """
    feature_dir = base_dir / "outputs" / "features" / subject / f"run-{run:02d}"

    # Load transcript and prosody
    transcript = pd.read_csv(feature_dir / "transcript.csv")
    prosody = pd.read_csv(feature_dir / "prosody.csv")
    turns = pd.read_csv(feature_dir / "turns.csv")

    # Extract words
    words_list = []

    for _, seg in transcript.iterrows():
        if 'words' not in seg or pd.isna(seg['words']):
            continue

        words_data = eval(seg['words']) if isinstance(seg['words'], str) else seg['words']
        seg_start_meg = seg['start_meg']
        seg_start_orig = seg['start']

        for w in words_data:
            # Word time in MEG coordinates
            word_time_orig = w['start']
            word_time_meg = seg_start_meg + (word_time_orig - seg_start_orig)

            words_list.append({
                'time_meg': word_time_meg,
                'word': w['word'],
                'duration': w['end'] - w['start'],
                'segment_id': seg['segment_id']
            })

    words_df = pd.DataFrame(words_list)

    # Add prosodic features at word onsets
    prosody_times = prosody['time'].values

    for feature in ['f0_smooth', 'intensity', 'energy']:
        # Interpolate to word onset times
        valid_mask = ~prosody[feature].isna()
        if valid_mask.sum() > 1:
            interp_func = interp1d(
                prosody_times[valid_mask],
                prosody[feature].values[valid_mask],
                kind='nearest',
                bounds_error=False,
                fill_value=np.nan
            )
            words_df[feature] = interp_func(words_df['time_meg'].values)
        else:
            words_df[feature] = np.nan

    # Fill NaN in F0 with 0 (unvoiced)
    words_df['f0_smooth'] = words_df['f0_smooth'].fillna(0)

    # Fill NaN in intensity/energy with mean
    for feature in ['intensity', 'energy']:
        words_df[feature] = words_df[feature].fillna(words_df[feature].mean())

    # Add turn-level features
    # Find which turn each word belongs to
    words_df['turn_idx'] = -1
    words_df['position_in_turn'] = 0.0

    for turn_idx, turn in turns.iterrows():
        # Turns.csv already has MEG-aligned times (same as transcript)
        turn_start = turn['start']
        turn_end = turn['end']

        in_turn = (words_df['time_meg'] >= turn_start) & (words_df['time_meg'] <= turn_end)
        words_df.loc[in_turn, 'turn_idx'] = turn_idx

        # Calculate relative position in turn (0 to 1)
        turn_words = words_df[in_turn]
        if len(turn_words) > 0:
            positions = (turn_words['time_meg'] - turn_start) / turn['duration']
            words_df.loc[in_turn, 'position_in_turn'] = positions

    # Mark turn-initial words
    words_df['is_turn_initial'] = False
    for turn_idx in words_df['turn_idx'].unique():
        if turn_idx >= 0:
            turn_words = words_df[words_df['turn_idx'] == turn_idx]
            if len(turn_words) > 0:
                first_word_idx = turn_words.index[0]
                words_df.loc[first_word_idx, 'is_turn_initial'] = True

    return words_df


def create_word_level_predictors(words_df, meg_times, meg_sfreq):
    """
    Create event-based predictors for word-level TRF.

    Instead of continuous predictors, we create:
    1. Impulse train at word onsets
    2. Each impulse weighted by predictor value

    This is the proper structure for event-related TRF.
    """
    n_samples = len(meg_times)

    predictors = {}

    # F0: impulse at word onset, scaled by F0 value
    f0_predictor = np.zeros(n_samples)
    intensity_predictor = np.zeros(n_samples)
    duration_predictor = np.zeros(n_samples)
    turn_initial_predictor = np.zeros(n_samples)

    for _, word in words_df.iterrows():
        # Find closest MEG sample to word onset
        onset_idx = np.argmin(np.abs(meg_times - word['time_meg']))

        # Create impulse scaled by feature value
        f0_predictor[onset_idx] += word['f0_smooth']
        intensity_predictor[onset_idx] += word['intensity']
        duration_predictor[onset_idx] += word['duration']

        # Binary predictor for turn-initial words
        if word['is_turn_initial']:
            turn_initial_predictor[onset_idx] = 1.0

    # Normalize continuous predictors
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

    # Also create simple word rate predictor (words per second in sliding window)
    word_rate = np.zeros(n_samples)
    window_size = int(1.0 * meg_sfreq)  # 1-second window

    for i in range(len(meg_times)):
        window_start = max(0, i - window_size // 2)
        window_end = min(n_samples - 1, i + window_size // 2)  # Fix: -1 to stay within bounds

        # Count word onsets in window
        window_time_start = meg_times[window_start]
        window_time_end = meg_times[window_end]

        n_words = ((words_df['time_meg'] >= window_time_start) &
                   (words_df['time_meg'] <= window_time_end)).sum()

        window_duration = window_time_end - window_time_start
        if window_duration > 0:
            word_rate[i] = n_words / window_duration

    predictors['word_rate'] = word_rate

    return predictors


def word_level_trf_analysis(subject: str, run: int, meg_raw: mne.io.Raw,
                             base_dir: Path, sensor_type: str = "mag"):
    """
    Fit word-level TRF model.

    Uses word onsets as events with scalar predictors at each onset.
    """
    print("="*70)
    print("WORD-LEVEL TRF ANALYSIS")
    print("="*70)

    # 1. Load word features
    words_df = load_word_features(subject, run, base_dir)
    print(f"\n✓ Loaded {len(words_df)} words")
    print(f"  Time range: {words_df['time_meg'].min():.1f}s - {words_df['time_meg'].max():.1f}s")
    print(f"  Turns: {words_df['turn_idx'].max() + 1}")
    print(f"  Turn-initial words: {words_df['is_turn_initial'].sum()}")

    # 2. Prepare MEG data
    print(f"\nPreparing MEG sensor data ({sensor_type})...")
    picks = mne.pick_types(meg_raw.info, meg=sensor_type, eeg=(sensor_type=='eeg'),
                           exclude='bads')
    meg_data, meg_times = meg_raw[picks, :]
    sfreq = meg_raw.info['sfreq']

    print(f"  Sensors: {len(picks)} {sensor_type}")
    print(f"  Duration: {meg_times.max():.1f}s")
    print(f"  Sampling rate: {sfreq:.0f} Hz")

    # 3. Create predictors
    print("\nCreating word-level predictors...")
    predictors = create_word_level_predictors(words_df, meg_times, sfreq)

    print(f"  F0 impulses: {np.sum(predictors['f0'] > 0)}")
    print(f"  Intensity impulses: {np.sum(predictors['intensity'] > 0)}")
    print(f"  Turn-initial impulses: {np.sum(predictors['turn_initial'] > 0)}")
    print(f"  Mean word rate: {predictors['word_rate'].mean():.2f} words/s")

    # 4. Convert to eelbrain format
    print("\nConverting to eelbrain NDVars...")

    # Time dimension
    time_dim = eelbrain.UTS(0, 1/sfreq, meg_data.shape[1])

    # Sensor dimension
    ch_names = [meg_raw.ch_names[i] for i in picks]
    ch_info = mne.pick_info(meg_raw.info, picks)

    try:
        sensor_dim = eelbrain.load.mne.sensor_dim(ch_info)
    except (AttributeError, TypeError):
        print("  Using simple sensor indexing...")
        sensor_dim = eelbrain.Case

    # MEG NDVar
    meg_ndvar = eelbrain.NDVar(
        meg_data,
        dims=(sensor_dim, time_dim),
        info={'subject': subject},
        name='MEG'
    )

    # Predictor NDVars
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

    print(f"\nFitting TRF with {len(pred_ndvars)} predictors:")
    for name in pred_ndvars.keys():
        print(f"  - {name}")

    # Use subset of predictors for initial test
    selected_predictors = ['f0', 'intensity', 'turn_initial']

    trf = eelbrain.boosting(
        meg_ndvar,
        [pred_ndvars[p] for p in selected_predictors],
        tstart=-0.1,      # Start 100ms before event
        tstop=0.5,        # End 500ms after event
        scale_data=True,  # Z-score MEG and predictors
        delta=0.005,      # 5ms TRF resolution
        mindelta=0.0005,  # Min delta for convergence
        partitions=5,     # 5-fold cross-validation
        test=1,           # Use partition 1 for testing
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

    output_dir = base_dir / "outputs" / "trf_word_level" / subject / f"run-{run:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save TRF model
    import pickle
    with open(output_dir / "trf_model.pkl", 'wb') as f:
        pickle.dump(trf, f)
    print(f"✓ Saved: {output_dir / 'trf_model.pkl'}")

    # Save word features
    words_df.to_csv(output_dir / "word_features.csv", index=False)
    print(f"✓ Saved: {output_dir / 'word_features.csv'}")

    # Try plotting
    try:
        import matplotlib.pyplot as plt

        if hasattr(trf, 'r'):
            fig = eelbrain.plot.Topomap(trf.r, vmax=0.1, cmap='RdYlGn')
            fig.set_title(f'Word-Level TRF Correlation - {subject} Run {run}')
            fig.figure.savefig(output_dir / "trf_correlation.png", dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {output_dir / 'trf_correlation.png'}")
            plt.close(fig.figure)

        # Plot TRF kernels for each predictor
        for i, pred_name in enumerate(selected_predictors):
            try:
                if isinstance(trf.h_scaled, dict):
                    h_data = trf.h_scaled[pred_name]
                else:
                    h_data = trf.h_scaled[i]

                fig = eelbrain.plot.TopoButterfly(h_data, vmax=0.05, cmap='xpolar')
                fig.set_title(f'Word TRF: {pred_name}')
                fig.figure.savefig(output_dir / f"trf_{pred_name}.png", dpi=150, bbox_inches='tight')
                print(f"✓ Saved: {output_dir / f'trf_{pred_name}.png'}")
                plt.close(fig.figure)
            except Exception as e:
                print(f"  Warning: Could not plot {pred_name}: {e}")

    except Exception as e:
        print(f"  Warning: Plotting failed: {e}")

    # Save summary
    summary = {
        'subject': subject,
        'run': run,
        'sensor_type': sensor_type,
        'n_sensors': len(picks),
        'n_words': len(words_df),
        'n_turns': int(words_df['turn_idx'].max() + 1),
        'model_correlation_mean': float(trf.r.mean()),
        'model_correlation_max': float(trf.r.max()),
        'predictors': selected_predictors,
    }

    import json
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved: {output_dir / 'summary.json'}")

    print("\n" + "="*70)
    print("WORD-LEVEL TRF COMPLETE!")
    print("="*70)
    print(f"\nResults: {output_dir}")
    print(f"\nModel performance:")
    print(f"  Mean correlation: {trf.r.mean():.3f}")
    print(f"  Max correlation: {trf.r.max():.3f}")

    return {
        'trf': trf,
        'summary': summary,
        'words_df': words_df,
        'meg_ndvar': meg_ndvar,
        'predictors': pred_ndvars,
    }


def main():
    parser = argparse.ArgumentParser(description="Word-level TRF analysis")
    parser.add_argument("--subject", required=True, help="Subject ID")
    parser.add_argument("--run", type=int, required=True, help="Run number")
    parser.add_argument("--meg-file", required=True, help="Path to MEG file")
    parser.add_argument("--sensor-type", default="mag",
                       choices=['mag', 'grad', 'eeg'],
                       help="Sensor type")
    parser.add_argument("--base-dir", type=Path,
                       default=Path("/Users/em18033/Library/CloudStorage/OneDrive-AUTUniversity/Projects/turn-taking-pipeline"),
                       help="Pipeline base directory")

    args = parser.parse_args()

    # Load MEG data
    print(f"Loading MEG data from: {args.meg_file}")
    meg_raw = mne.io.read_raw_fif(args.meg_file, preload=True, verbose=False)

    # Run analysis
    results = word_level_trf_analysis(
        args.subject,
        args.run,
        meg_raw,
        args.base_dir,
        args.sensor_type
    )

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
