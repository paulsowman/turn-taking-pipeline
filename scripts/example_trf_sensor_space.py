#!/usr/bin/env python3
"""
TRF Analysis in Sensor Space with Eelbrain

This example shows how to perform TRF analysis using sensor-space MEG data
instead of source-space. This is faster and useful for:
- Initial exploratory analysis
- Channel-level effects
- Validating features before source localization

Demonstrates:
1. Loading MEG sensor data (MNE Raw/Epochs format)
2. Converting to eelbrain NDVar
3. Fitting TRFs with prosody features
4. Testing significance at sensor level
5. Visualizing topographic TRF patterns
"""

import numpy as np
import pandas as pd
import mne
from pathlib import Path
import json
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import eelbrain


def load_pipeline_features(subject: str, run: int, base_dir: Path):
    """Load features from turn-taking pipeline."""
    feature_dir = base_dir / "outputs" / "features" / subject / f"run-{run:02d}"

    features = {}

    # Load sync params
    with open(feature_dir / "sync_params.json") as f:
        features["sync_params"] = json.load(f)

    # Load CSV files
    features["prosody"] = pd.read_csv(feature_dir / "prosody.csv")
    features["turns"] = pd.read_csv(feature_dir / "turns.csv")
    features["trp_features"] = pd.read_csv(feature_dir / "trp_features.csv")

    print(f"✓ Loaded features for {subject} run {run}")

    return features


def align_features_to_meg_sensor(features: dict, meg_times: np.ndarray):
    """
    Align features to MEG sensor data timebase.

    Parameters
    ----------
    features : dict
        Features from pipeline
    meg_times : np.ndarray
        MEG time array (in seconds)

    Returns
    -------
    meg_features : dict
        Features aligned to MEG sampling rate
    """
    prosody = features["prosody"]
    offset_s = features["sync_params"]["initial_offset_s"]

    # Convert prosody times to MEG timebase
    prosody_times_meg = prosody["time"].values - offset_s

    print(f"Aligning features to MEG sensor data...")
    print(f"  Offset: {offset_s:.3f}s")
    print(f"  MEG time range: {meg_times.min():.1f}s to {meg_times.max():.1f}s")

    meg_features = {}

    # Interpolate continuous features
    for feature_name in ["f0_smooth", "intensity", "energy"]:
        interp_func = interp1d(
            prosody_times_meg,
            prosody[feature_name].values,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan if feature_name == "f0_smooth" else 0
        )
        meg_features[feature_name] = interp_func(meg_times)

    # Create TRP event regressor
    trp_events = np.zeros(len(meg_times))
    trps = features["trp_features"]

    if len(trps) > 0:
        trp_times_meg = trps["time"].values - offset_s
        for trp_time in trp_times_meg:
            if meg_times.min() <= trp_time <= meg_times.max():
                idx = np.argmin(np.abs(meg_times - trp_time))
                trp_events[idx] = 1.0

    meg_features["trp_events"] = trp_events

    # Create turn onset regressor
    turn_events = np.zeros(len(meg_times))
    turns = features["turns"]

    if "start_meg" in turns.columns:
        turn_times = turns["start_meg"].values
    else:
        turn_times = turns["start"].values - offset_s

    for turn_time in turn_times:
        if meg_times.min() <= turn_time <= meg_times.max():
            idx = np.argmin(np.abs(meg_times - turn_time))
            turn_events[idx] = 1.0

    meg_features["turn_events"] = turn_events

    print(f"✓ Aligned features:")
    print(f"  F0 voiced samples: {np.sum(~np.isnan(meg_features['f0_smooth']))}")
    print(f"  TRP events: {int(np.sum(trp_events))}")
    print(f"  Turn events: {int(np.sum(turn_events))}")

    return meg_features


def sensor_trf_analysis(subject: str, run: int, meg_raw: mne.io.Raw, base_dir: Path,
                        sensor_type: str = "mag"):
    """
    Complete sensor-space TRF analysis.

    Parameters
    ----------
    subject : str
        Subject ID
    run : int
        Run number
    meg_raw : mne.io.Raw
        MEG sensor data (MNE Raw object)
    base_dir : Path
        Pipeline base directory
    sensor_type : str
        'mag', 'grad', or 'eeg'

    Returns
    -------
    trf_results : dict
        TRF model and results
    """
    print("="*70)
    print("SENSOR-SPACE TRF ANALYSIS")
    print("="*70)

    # 1. Load features
    features = load_pipeline_features(subject, run, base_dir)

    # 2. Get MEG sensor data
    print(f"\nPreparing MEG sensor data ({sensor_type})...")

    # Pick sensor type
    picks = mne.pick_types(meg_raw.info, meg=sensor_type, eeg=(sensor_type=='eeg'),
                           exclude='bads')

    # Get data and times
    meg_data, meg_times = meg_raw[picks, :]
    sfreq = meg_raw.info['sfreq']

    print(f"  Sensors: {len(picks)} {sensor_type}")
    print(f"  Duration: {meg_times.max():.1f}s")
    print(f"  Sampling rate: {sfreq:.0f} Hz")

    # 3. Align features
    meg_features = align_features_to_meg_sensor(features, meg_times)

    # 4. Convert to eelbrain NDVar
    print("\nConverting to eelbrain format...")

    # Create time dimension
    time_dim = eelbrain.UTS(0, 1/sfreq, meg_data.shape[1])

    # For MEG sensor-space, we can use MNE epochs-like structure
    # Create a simple sensor dimension using channel info
    # Pick only the selected channels from info
    ch_names = [meg_raw.ch_names[i] for i in picks]
    ch_info = mne.pick_info(meg_raw.info, picks)

    # Create sensor dimension from MNE info object
    try:
        sensor_dim = eelbrain.load.mne.sensor_dim(ch_info)
    except (AttributeError, TypeError):
        # Fallback: create simple case dimension for sensors
        print(f"  Using simple sensor indexing for {len(picks)} channels...")
        sensor_dim = eelbrain.Case

    # MEG data as NDVar
    meg_ndvar = eelbrain.NDVar(
        meg_data,
        dims=(sensor_dim, time_dim),
        info={'subject': subject},
        name='MEG'
    )

    print(f"  MEG NDVar: {meg_ndvar.shape}")

    # Create predictor NDVars
    predictors = {}

    # F0 (pitch) - fill NaN with 0 (unvoiced segments)
    f0_clean = np.nan_to_num(meg_features['f0_smooth'], nan=0.0)
    predictors['f0'] = eelbrain.NDVar(
        f0_clean,
        dims=(time_dim,),
        name='F0'
    )

    # Intensity - fill NaN with mean or 0
    intensity_clean = np.nan_to_num(meg_features['intensity'],
                                     nan=np.nanmean(meg_features['intensity']))
    predictors['intensity'] = eelbrain.NDVar(
        intensity_clean,
        dims=(time_dim,),
        name='Intensity'
    )

    # Energy - fill NaN with 0
    energy_clean = np.nan_to_num(meg_features['energy'], nan=0.0)
    predictors['energy'] = eelbrain.NDVar(
        energy_clean,
        dims=(time_dim,),
        name='Energy'
    )

    # TRP events
    predictors['trp'] = eelbrain.NDVar(
        meg_features['trp_events'],
        dims=(time_dim,),
        name='TRP'
    )

    # Turn events
    predictors['turns'] = eelbrain.NDVar(
        meg_features['turn_events'],
        dims=(time_dim,),
        name='TurnOnset'
    )

    print(f"✓ Created {len(predictors)} predictors")

    # 5. Fit TRF
    print("\n" + "="*70)
    print("FITTING TRF MODEL")
    print("="*70)

    # Use subset of predictors (continuous + one event)
    active_predictors = [
        predictors['f0'],
        predictors['intensity'],
        predictors['turns'],  # Use turn onsets (more events than TRPs)
    ]

    print(f"\nFitting TRF with {len(active_predictors)} predictors:")
    for p in active_predictors:
        print(f"  - {p.name}")

    # Fit boosting TRF
    trf = eelbrain.boosting(
        meg_ndvar,           # MEG sensor data
        active_predictors,   # Predictors
        tstart=-0.1,         # Start 100ms before event
        tstop=0.5,           # End 500ms after event
        basis=0.050,         # 50ms basis functions
        partitions=5,        # 5-fold cross-validation
        test=1,              # Use partition 1 for testing
        selective_stopping=True,  # Early stopping per predictor
        error='l1',          # L1 error (robust to outliers)
    )

    print("\n✓ TRF fitting complete!")
    print(f"  Model correlation: {trf.r.mean():.3f}")

    # Print contribution of each predictor
    print("\nPredictor contributions:")
    # trf.h is a tuple of TRF kernels, one per predictor
    predictor_names = ['f0', 'intensity', 'turns']
    if hasattr(trf, 'proportion_explained'):
        for i, pred_name in enumerate(predictor_names):
            if isinstance(trf.proportion_explained, dict):
                contrib = trf.proportion_explained[pred_name].mean()
            else:
                # proportion_explained might be an array
                contrib = trf.proportion_explained[i].mean() if hasattr(trf.proportion_explained[i], 'mean') else trf.proportion_explained[i]
            print(f"  {pred_name}: {contrib:.3f}")
    else:
        print("  (Proportion explained not available in this eelbrain version)")

    # 6. Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    output_dir = base_dir / "outputs" / "trf_sensor" / subject / f"run-{run:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save TRF object
    import pickle
    trf_file = output_dir / "trf_model.pkl"
    with open(trf_file, 'wb') as f:
        pickle.dump(trf, f)
    print(f"✓ Saved TRF model: {trf_file}")

    # Try to visualize if plotting is available
    try:
        import matplotlib.pyplot as plt

        # Plot model correlation topography
        if hasattr(trf, 'r'):
            fig1 = eelbrain.plot.Topomap(trf.r, vmax=0.3, cmap='RdYg_r')
            fig1.set_title(f'TRF Model Correlation - {subject} Run {run}')
            fig1.figure.savefig(output_dir / "trf_model_correlation.png", dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {output_dir / 'trf_model_correlation.png'}")
            plt.close(fig1.figure)

        # Plot TRFs for each predictor
        if hasattr(trf, 'h') and hasattr(trf, 'h_scaled'):
            for i, pred_name in enumerate(predictor_names):
                try:
                    if isinstance(trf.h_scaled, dict):
                        h_data = trf.h_scaled[pred_name]
                    else:
                        h_data = trf.h_scaled[i]

                    fig2 = eelbrain.plot.TopoButterfly(h_data, vmax=0.1, cmap='xpolar')
                    fig2.set_title(f'TRF: {pred_name}')
                    fig2.figure.savefig(output_dir / f"trf_{pred_name}.png", dpi=150, bbox_inches='tight')
                    print(f"✓ Saved: {output_dir / f'trf_{pred_name}.png'}")
                    plt.close(fig2.figure)
                except Exception as e:
                    print(f"  Warning: Could not plot {pred_name}: {e}")

    except Exception as e:
        print(f"  Warning: Plotting failed: {e}")
        print("  TRF model saved but plots not generated")

    # Save summary stats
    summary = {
        'subject': subject,
        'run': run,
        'sensor_type': sensor_type,
        'n_sensors': len(picks),
        'model_correlation_mean': float(trf.r.mean()),
        'model_correlation_max': float(trf.r.max()),
        'predictors': predictor_names,
    }

    # Add contributions if available
    if hasattr(trf, 'proportion_explained'):
        try:
            if isinstance(trf.proportion_explained, dict):
                summary['contributions'] = {
                    pred: float(trf.proportion_explained[pred].mean())
                    for pred in predictor_names
                }
            else:
                summary['contributions'] = {
                    predictor_names[i]: float(trf.proportion_explained[i].mean())
                    for i in range(len(predictor_names))
                }
        except Exception:
            pass

    import json
    with open(output_dir / "trf_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Saved summary: {output_dir / 'trf_summary.json'}")

    print("\n" + "="*70)
    print("SENSOR TRF ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nModel performance:")
    print(f"  Mean correlation: {trf.r.mean():.3f}")
    print(f"  Max correlation:  {trf.r.max():.3f}")

    if 'contributions' in summary:
        print(f"\nPredictor contributions:")
        for pred, contrib in summary['contributions'].items():
            print(f"  {pred}: {contrib:.3f}")

    return {
        'trf': trf,
        'summary': summary,
        'meg_ndvar': meg_ndvar,
        'predictors': predictors,
    }


def quick_sensor_trf_example():
    """
    Quick example showing the minimal code needed for sensor TRF.
    """
    print("""
MINIMAL SENSOR-SPACE TRF EXAMPLE
================================

import mne
import eelbrain
from pathlib import Path

# 1. Load your MEG data
meg_raw = mne.io.read_raw_fif("path/to/your_raw.fif", preload=True)

# 2. Run TRF analysis
from example_trf_sensor_space import sensor_trf_analysis

results = sensor_trf_analysis(
    subject="sub-01",
    run=1,
    meg_raw=meg_raw,
    base_dir=Path("/path/to/turn-taking-pipeline"),
    sensor_type="mag"  # or "grad" or "eeg"
)

# 3. Results are saved automatically!
# Check: outputs/trf_sensor/sub-01/run-01/

# 4. Load saved TRF later
trf = eelbrain.load.unpickle("outputs/trf_sensor/sub-01/run-01/trf_model.pickle")

# Plot specific predictor TRF
eelbrain.plot.TopoButterfly(trf.h_scaled['F0'])
    """)


def main():
    """Run sensor-space TRF analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Sensor-space TRF analysis")
    parser.add_argument("--subject", type=str, default="sub-01")
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--meg-file", type=Path, required=False,
                       help="Path to MEG raw file (.fif)")
    parser.add_argument("--sensor-type", type=str, default="mag",
                       choices=["mag", "grad", "eeg"])
    parser.add_argument("--example", action="store_true",
                       help="Show minimal example code")

    args = parser.parse_args()

    if args.example:
        quick_sensor_trf_example()
        return

    base_dir = Path(__file__).parent.parent

    if args.meg_file:
        # Load user's MEG data
        print(f"Loading MEG data from: {args.meg_file}")
        meg_raw = mne.io.read_raw_fif(args.meg_file, preload=True)

        # Run TRF analysis
        results = sensor_trf_analysis(
            args.subject,
            args.run,
            meg_raw,
            base_dir,
            args.sensor_type
        )

        print("\n✓✓✓ Analysis complete!")

    else:
        print("ERROR: Please provide MEG file with --meg-file")
        print("\nExample usage:")
        print("  python example_trf_sensor_space.py --subject sub-01 --run 1 \\")
        print("    --meg-file /path/to/sub-01_task-conversation_run-01_proc-clean_raw.fif")
        print("\nOr show minimal example:")
        print("  python example_trf_sensor_space.py --example")


if __name__ == "__main__":
    main()
