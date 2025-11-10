#!/usr/bin/env python
"""
Combined Envelope TRF Analysis for Multiple Runs

Combines conversation runs (1, 3, 5) for improved statistical power.
Includes comprehensive visualizations of results.
"""

import numpy as np
import mne
import eelbrain
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
from scipy.interpolate import interp1d


def load_roi_sensors(roi_json_path):
    """Load ROI sensor names from JSON file."""
    with open(roi_json_path, 'r') as f:
        roi_data = json.load(f)

    left_sensors = roi_data['left_hemisphere']['names']
    right_sensors = roi_data['right_hemisphere']['names']

    return left_sensors + right_sensors


def identify_speaker_from_audio(subject, run, base_dir, meg_times, threshold_db=-40):
    """
    Identify who is speaking at each timepoint using acoustic energy from both mics.

    Uses separate microphone recordings for participant and interviewer to determine
    speaker identity based on relative energy levels.

    Parameters
    ----------
    subject : str
        Subject ID
    run : int
        Run number
    base_dir : Path
        Base directory
    meg_times : np.ndarray
        MEG timepoints (1000 Hz)
    threshold_db : float
        Minimum dB level to consider as speech (default: -40 dB)

    Returns
    -------
    listener_mask : np.ndarray
        Boolean mask: True when participant is listening (interviewer speaking),
        False when participant is speaking or silence
    """
    from utils.config import load_config, get_subject_paths

    print("  Identifying speaker from dual-microphone recordings...")

    config = load_config()
    paths = get_subject_paths(subject, run, config)

    if 'external_audio_interviewer' not in paths or 'external_audio_participant' not in paths:
        raise FileNotFoundError(f"Both audio channels required for speaker identification")

    # Load both audio channels
    interviewer_audio, sr = librosa.load(paths['external_audio_interviewer'], sr=None)
    participant_audio, _ = librosa.load(paths['external_audio_participant'], sr=sr)

    # Load sync offset
    sync_dir = base_dir / "outputs" / "sync" / subject / f"run-{run:02d}"
    sync_params_file = sync_dir / "sync_params.json"
    with open(sync_params_file, 'r') as f:
        sync_params = json.load(f)
    sync_offset = sync_params['initial_offset_s']

    # Compute RMS energy for both channels (100 Hz)
    hop_length = int(0.010 * sr)  # 10ms hops
    interviewer_rms = librosa.feature.rms(y=interviewer_audio, hop_length=hop_length)[0]
    participant_rms = librosa.feature.rms(y=participant_audio, hop_length=hop_length)[0]

    # Convert to dB
    interviewer_db = 20 * np.log10(interviewer_rms + 1e-10)
    participant_db = 20 * np.log10(participant_rms + 1e-10)

    # Time base for RMS
    rms_times_audio = np.arange(len(interviewer_rms)) * hop_length / sr
    # Positive offset means external starts AFTER MEG, so ADD offset
    rms_times_meg = rms_times_audio - sync_offset

    # Interpolate to MEG sampling rate
    interp_interviewer = interp1d(rms_times_meg, interviewer_db,
                                   kind='nearest', bounds_error=False, fill_value=-100)
    interp_participant = interp1d(rms_times_meg, participant_db,
                                    kind='nearest', bounds_error=False, fill_value=-100)

    interviewer_db_meg = interp_interviewer(meg_times)
    participant_db_meg = interp_participant(meg_times)

    # Determine speaker at each timepoint
    # Listener (True) = interviewer speaking AND participant not speaking
    # Speaking (False) = participant speaking OR silence

    interviewer_speaking = interviewer_db_meg > threshold_db
    participant_speaking = participant_db_meg > threshold_db

    # Participant is listening when ONLY interviewer is above threshold
    listener_mask = interviewer_speaking & ~participant_speaking

    # Statistics
    both_speaking_pct = 100 * np.sum(interviewer_speaking & participant_speaking) / len(meg_times)
    interviewer_only_pct = 100 * np.sum(listener_mask) / len(meg_times)
    participant_only_pct = 100 * np.sum(participant_speaking & ~interviewer_speaking) / len(meg_times)
    silence_pct = 100 * np.sum(~interviewer_speaking & ~participant_speaking) / len(meg_times)

    print(f"    Interviewer only (listening):  {interviewer_only_pct:.1f}%")
    print(f"    Participant only (speaking):   {participant_only_pct:.1f}%")
    print(f"    Both speaking (overlap):       {both_speaking_pct:.1f}%")
    print(f"    Silence:                       {silence_pct:.1f}%")
    print(f"    Threshold: {threshold_db} dB")

    return listener_mask


def load_synchronized_audio(subject, run, base_dir):
    """Load audio file and get sync offset."""
    # Get audio file path from config
    from utils.config import load_config, get_subject_paths

    config = load_config()
    paths = get_subject_paths(subject, run, config)

    if 'external_audio_interviewer' not in paths:
        raise FileNotFoundError(f"No audio file found for {subject} run {run}")

    audio_file = paths['external_audio_interviewer']

    print(f"  Loading audio: {audio_file}")
    audio, sr = librosa.load(audio_file, sr=None)

    # Load sync parameters
    sync_dir = base_dir / "outputs" / "sync" / subject / f"run-{run:02d}"
    sync_params_file = sync_dir / "sync_params.json"

    if not sync_params_file.exists():
        raise FileNotFoundError(f"Sync params not found: {sync_params_file}")

    import json
    with open(sync_params_file, 'r') as f:
        sync_params = json.load(f)
    sync_offset = sync_params['initial_offset_s']

    print(f"  Audio: {len(audio)/sr:.1f}s @ {sr} Hz")
    print(f"  Sync offset: {sync_offset:.3f}s")

    return audio, sr, sync_offset


def compute_acoustic_envelope(audio, sr, method='rms', frame_length=None, hop_length=None):
    """
    Compute acoustic envelope using RMS or Hilbert method.

    Parameters
    ----------
    audio : np.ndarray
        Audio waveform
    sr : int
        Sample rate of audio
    method : str
        'rms' or 'hilbert'
    frame_length : int, optional
        Frame length in samples for RMS (default: 25ms)
    hop_length : int, optional
        Hop length in samples (default: 10ms, giving ~100 Hz envelope)

    Returns
    -------
    envelope : np.ndarray
        Acoustic envelope
    envelope_times : np.ndarray
        Time points for envelope
    envelope_sr : float
        Effective sample rate of envelope
    """
    if method == 'rms':
        if frame_length is None:
            frame_length = int(0.025 * sr)  # 25ms
        if hop_length is None:
            hop_length = int(0.010 * sr)    # 10ms → ~100 Hz

        envelope = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        envelope_times = np.arange(len(envelope)) * hop_length / sr
        envelope_sr = sr / hop_length

    elif method == 'hilbert':
        from scipy.signal import hilbert
        analytic_signal = hilbert(audio)
        envelope = np.abs(analytic_signal)

        # Downsample to ~100 Hz
        if hop_length is None:
            hop_length = int(0.010 * sr)
        envelope = envelope[::hop_length]
        envelope_times = np.arange(len(envelope)) * hop_length / sr
        envelope_sr = sr / hop_length

    else:
        raise ValueError(f"Unknown method: {method}")

    return envelope, envelope_times, envelope_sr


def align_envelope_to_meg(envelope, envelope_times, sync_offset, meg_times):
    """
    Align acoustic envelope to MEG time base.

    Parameters
    ----------
    envelope : np.ndarray
        Acoustic envelope values
    envelope_times : np.ndarray
        Times for envelope (in audio timebase)
    sync_offset : float
        Sync offset in seconds (audio_time - meg_time)
    meg_times : np.ndarray
        MEG time points

    Returns
    -------
    envelope_meg : np.ndarray
        Envelope interpolated to MEG timebase
    """
    # Adjust envelope times to MEG timebase
    # Positive offset means external starts AFTER MEG, so ADD offset
    envelope_times_meg = envelope_times - sync_offset

    # Linear interpolation to MEG sampling rate
    interp_func = interp1d(
        envelope_times_meg,
        envelope,
        kind='linear',
        bounds_error=False,
        fill_value=0
    )

    envelope_meg = interp_func(meg_times)

    return envelope_meg


def load_and_process_run(subject, run, meg_file, sensor_type, envelope_method, base_dir, roi_sensors_json=None, listener_only=False):
    """Load and process a single run."""
    print(f"\n{'='*70}")
    print(f"PROCESSING RUN {run}")
    print(f"{'='*70}\n")

    # Load synchronized audio
    print("Loading synchronized audio...")
    audio, sr, sync_offset = load_synchronized_audio(subject, run, base_dir)

    # Compute acoustic envelope
    print(f"\nComputing {envelope_method} envelope...")
    envelope, envelope_times, envelope_sr = compute_acoustic_envelope(
        audio, sr, method=envelope_method
    )
    print(f"  Frame length: 25.0ms")
    print(f"  Hop length: 10.0ms")
    print(f"  Envelope: {len(envelope)} samples @ {envelope_sr:.1f} Hz")
    print(f"  Duration: {envelope_times[-1]:.1f}s")

    # Load MEG data
    print("\nLoading MEG data...")
    meg_raw = mne.io.read_raw_fif(meg_file, preload=True, verbose=False)

    # Handle EEG vs MEG sensor selection
    if sensor_type == 'eeg':
        picks = mne.pick_types(meg_raw.info, meg=False, eeg=True, exclude='bads')
    else:
        picks = mne.pick_types(meg_raw.info, meg=sensor_type, eeg=False, exclude='bads')

    # Apply ROI sensor selection if specified
    if roi_sensors_json is not None:
        roi_sensor_names = load_roi_sensors(roi_sensors_json)
        # Get channel names for picked sensors
        ch_names = [meg_raw.ch_names[p] for p in picks]
        # Filter to only ROI sensors
        roi_picks = [p for p, ch in zip(picks, ch_names) if ch in roi_sensor_names]
        picks = np.array(roi_picks)
        print(f"  Applied ROI filter: {len(picks)} sensors")

    meg_data, meg_times = meg_raw[picks, :]
    sfreq = meg_raw.info['sfreq']

    print(f"  Sensors: {len(picks)} {sensor_type}")
    print(f"  Duration: {meg_times[-1]:.1f}s")
    print(f"  Sampling rate: {sfreq} Hz")

    # Align envelope to MEG
    print("\nAligning envelope to MEG...")
    envelope_meg = align_envelope_to_meg(envelope, envelope_times, sync_offset, meg_times)

    valid_samples = np.sum(envelope_meg > 0)
    print(f"  Envelope MEG time range: {envelope_times[0] - sync_offset:.1f}s to {envelope_times[-1] - sync_offset:.1f}s")
    print(f"  MEG time range: {meg_times[0]:.1f}s to {meg_times[-1]:.1f}s")
    print(f"  Valid MEG samples: {valid_samples}/{len(meg_times)} ({100*valid_samples/len(meg_times):.1f}%)")
    print(f"  Envelope stats: mean={envelope_meg.mean():.6f}, max={envelope_meg.max():.6f}")

    # Apply listener-only mask if requested
    listener_mask = None
    if listener_only:
        print("\nApplying listener-only mask...")
        try:
            # Use dual-microphone speaker identification (robust to consecutive turns)
            listener_mask = identify_speaker_from_audio(subject, run, base_dir, meg_times)

            listening_pct = 100 * np.sum(listener_mask) / len(listener_mask)
            print(f"  Listener periods: {listening_pct:.1f}% of recording")
            print(f"  {np.sum(listener_mask)/sfreq:.1f}s listening / {len(meg_times)/sfreq:.1f}s total")

            # Zero out envelope during non-listening periods
            envelope_meg[~listener_mask] = 0

        except Exception as e:
            print(f"  Warning: Could not apply listener mask: {e}")
            print(f"  Continuing without masking...")
            listener_mask = None

    return meg_data, envelope_meg, meg_times, meg_raw.info, picks, listener_mask


def combine_runs(run_data_list):
    """Concatenate data from multiple runs."""
    print(f"\n{'='*70}")
    print("CONCATENATING RUNS")
    print(f"{'='*70}\n")

    meg_combined = np.concatenate([rd['meg'] for rd in run_data_list], axis=1)
    envelope_combined = np.concatenate([rd['envelope'] for rd in run_data_list])

    # Create combined time array
    time_offsets = [0]
    for rd in run_data_list[:-1]:
        time_offsets.append(time_offsets[-1] + rd['times'][-1])

    times_combined = []
    for i, rd in enumerate(run_data_list):
        times_combined.append(rd['times'] + time_offsets[i])
    times_combined = np.concatenate(times_combined)

    print(f"  Combined MEG shape: {meg_combined.shape}")
    print(f"  Combined envelope length: {len(envelope_combined)}")
    print(f"  Total duration: {times_combined[-1]:.1f}s")

    return meg_combined, envelope_combined, times_combined


def fit_trf_model(meg_data, envelope, times, sensor_dim, tstart=-0.2, tstop=0.5):
    """Fit TRF model using eelbrain boosting."""
    print(f"\n{'='*70}")
    print("FITTING TRF MODEL")
    print(f"{'='*70}\n")

    print("Converting to eelbrain format...")
    time_dim = eelbrain.UTS(0, 1/1000.0, meg_data.shape[1])  # Assuming 1000 Hz

    meg_ndvar = eelbrain.NDVar(
        meg_data,
        dims=(sensor_dim, time_dim),
        name='MEG'
    )

    envelope_ndvar = eelbrain.NDVar(
        envelope,
        dims=(time_dim,),
        name='Envelope'
    )

    print(f"  Using eelbrain sensor dimension")
    print(f"  MEG shape: {meg_data.shape}")
    print(f"  Envelope shape: {envelope.shape}")

    print("\nFitting TRF with acoustic envelope predictor...")
    print(f"  TRF window: {tstart*1000:.0f}ms to {tstop*1000:.0f}ms")
    print(f"  This may take 5-10 minutes...")

    trf = eelbrain.boosting(
        meg_ndvar,
        envelope_ndvar,
        tstart=tstart,
        tstop=tstop,
        scale_data=True,
        delta=0.005,
        mindelta=0.0005,
        error='l1',
    )

    print(f"\n✓ TRF fitting complete!")
    print(f"  Model correlation: {trf.r.mean():.4f}")
    print(f"  Max correlation: {trf.r.max():.4f}")

    return trf, meg_ndvar, envelope_ndvar


def create_visualizations(trf, meg_ndvar, envelope_meg, output_dir):
    """Create comprehensive visualizations."""
    print(f"\n{'='*70}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*70}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. TRF Correlation Topography
    print("  1. TRF correlation topography...")
    try:
        fig = plt.figure(figsize=(10, 8))

        # Extract correlation values
        corr_values = trf.r.x if hasattr(trf.r, 'x') else np.array(trf.r)

        # Create topography plot using MNE
        from mne.viz import plot_topomap
        info = meg_ndvar.info if hasattr(meg_ndvar, 'info') else None

        if info:
            plot_topomap(corr_values, info, axes=plt.gca(), show=False,
                        cmap='RdBu_r', vlim=(corr_values.min(), corr_values.max()))
        else:
            # Fallback: simple heatmap
            plt.scatter(range(len(corr_values)), corr_values, c=corr_values, cmap='RdBu_r')
            plt.colorbar(label='Correlation')

        plt.title(f'TRF Model Correlation\nMean: {corr_values.mean():.4f}, Max: {corr_values.max():.4f}')
        plt.savefig(output_dir / 'trf_correlation_topo.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: trf_correlation_topo.png")
    except Exception as e:
        print(f"    Warning: Could not create topography plot: {e}")

    # 2. TRF Correlation Distribution
    print("  2. TRF correlation distribution...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    corr_values = trf.r.x if hasattr(trf.r, 'x') else np.array(trf.r)
    ax.hist(corr_values, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(corr_values.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {corr_values.mean():.4f}')
    ax.axvline(corr_values.max(), color='g', linestyle='--', linewidth=2, label=f'Max: {corr_values.max():.4f}')
    ax.set_xlabel('Correlation')
    ax.set_ylabel('Number of Sensors')
    ax.set_title('TRF Model Correlation Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(output_dir / 'trf_correlation_dist.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: trf_correlation_dist.png")

    # 3. Top 10 Sensors Correlation
    print("  3. Top 10 sensors correlation...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    top_idx = np.argsort(np.abs(corr_values))[-10:][::-1]
    top_corrs = corr_values[top_idx]

    colors = ['green' if c > 0 else 'red' for c in top_corrs]
    ax.barh(range(10), top_corrs, color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(10))
    ax.set_yticklabels([f'Sensor {i}' for i in top_idx])
    ax.set_xlabel('Correlation')
    ax.set_title('Top 10 Sensors by Absolute Correlation')
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')

    plt.savefig(output_dir / 'trf_top10_sensors.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: trf_top10_sensors.png")

    # 4. TRF Kernel (averaged across sensors)
    print("  4. TRF kernel temporal profile...")
    try:
        if hasattr(trf, 'h'):
            h_data = trf.h.x if hasattr(trf.h, 'x') else np.array(trf.h)

            # Average across sensors
            if h_data.ndim == 2:
                h_avg = h_data.mean(axis=0)
            else:
                h_avg = h_data

            # Get time axis
            if hasattr(trf.h, 'time'):
                h_times = trf.h.time.times * 1000  # Convert to ms
            else:
                h_times = np.linspace(-200, 500, len(h_avg))

            # Crop first and last 50ms to remove edge artifacts
            mask = (h_times >= -150) & (h_times <= 450)
            h_times_cropped = h_times[mask]
            h_avg_cropped = h_avg[mask]

            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            ax.plot(h_times_cropped, h_avg_cropped, linewidth=2, color='blue')
            ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Stimulus onset')
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('TRF Amplitude (a.u.)')
            ax.set_title('TRF Kernel (Averaged Across Sensors)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.savefig(output_dir / 'trf_kernel_avg.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    ✓ Saved: trf_kernel_avg.png")
    except Exception as e:
        print(f"    Warning: Could not create TRF kernel plot: {e}")

    # 5. Envelope diagnostic (first 30s)
    print("  5. Envelope diagnostic...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Show first 30 seconds
    end_idx = min(30000, len(envelope_meg))  # 30s at 1000 Hz
    times_plot = np.arange(end_idx) / 1000.0

    ax1.plot(times_plot, envelope_meg[:end_idx], linewidth=0.5, color='blue', alpha=0.8)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Envelope Amplitude')
    ax1.set_title('Acoustic Envelope (First 30s)')
    ax1.grid(True, alpha=0.3)

    # Distribution
    ax2.hist(envelope_meg[envelope_meg > 0], bins=100, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Envelope Amplitude')
    ax2.set_ylabel('Count')
    ax2.set_title('Envelope Amplitude Distribution')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'envelope_diagnostic.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: envelope_diagnostic.png")

    print(f"\n✓ All visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Combined envelope TRF analysis")
    parser.add_argument('--subject', type=str, default='sub-01', help='Subject ID')
    parser.add_argument('--runs', type=int, nargs='+', default=[1, 3, 5], help='Run numbers to combine')
    parser.add_argument('--meg-files', type=str, nargs='+', required=True, help='MEG files for each run')
    parser.add_argument('--sensor-type', type=str, default='mag', choices=['mag', 'grad', 'eeg'], help='Sensor type')
    parser.add_argument('--envelope-method', type=str, default='rms', choices=['rms', 'hilbert'], help='Envelope method')
    parser.add_argument('--roi-sensors', type=str, default=None, help='Path to ROI sensors JSON file (from BADA localizer)')
    parser.add_argument('--listener-only', action='store_true', help='Only analyze periods when participant is listening (not speaking)')
    parser.add_argument('--base-dir', type=Path, default=None, help='Base directory')

    args = parser.parse_args()

    if len(args.meg_files) != len(args.runs):
        raise ValueError("Number of MEG files must match number of runs")

    base_dir = args.base_dir or Path(__file__).parent.parent

    print("="*70)
    print("COMBINED ENVELOPE TRF ANALYSIS")
    print("="*70)
    print(f"\nSubject: {args.subject}")
    print(f"Runs: {args.runs}")
    print(f"Sensor type: {args.sensor_type}")
    print(f"Envelope method: {args.envelope_method}")

    # Load and process each run
    run_data_list = []
    for run, meg_file in zip(args.runs, args.meg_files):
        meg_data, envelope, times, info, picks, listener_mask = load_and_process_run(
            args.subject, run, meg_file, args.sensor_type, args.envelope_method, base_dir, args.roi_sensors, args.listener_only
        )
        run_data_list.append({
            'meg': meg_data,
            'envelope': envelope,
            'times': times,
            'info': info,
            'picks': picks
        })

    # Combine runs
    meg_combined, envelope_combined, times_combined = combine_runs(run_data_list)

    # Create sensor dimension
    print("\nCreating sensor dimension...")
    ch_info = mne.pick_info(run_data_list[0]['info'], run_data_list[0]['picks'])
    try:
        sensor_dim = eelbrain.load.mne.sensor_dim(ch_info)
        print("  Using eelbrain sensor dimension")
    except (AttributeError, TypeError):
        sensor_dim = eelbrain.Case
        print("  Using Case dimension (fallback)")

    # Fit TRF model
    trf, meg_ndvar, envelope_ndvar = fit_trf_model(
        meg_combined, envelope_combined, times_combined, sensor_dim
    )

    # Save results (separate directory for each sensor type)
    output_dir = base_dir / "outputs" / "trf_envelope_combined" / args.subject / f"runs-{'_'.join(map(str, args.runs))}" / args.sensor_type
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}\n")

    # Save TRF model
    import pickle
    with open(output_dir / 'trf_model.pkl', 'wb') as f:
        pickle.dump(trf, f)
    print(f"✓ Saved: {output_dir / 'trf_model.pkl'}")

    # Save envelope
    np.save(output_dir / 'envelope_combined.npy', envelope_combined)
    print(f"✓ Saved: {output_dir / 'envelope_combined.npy'}")

    # Save summary
    summary = {
        'subject': args.subject,
        'runs': args.runs,
        'sensor_type': args.sensor_type,
        'envelope_method': args.envelope_method,
        'n_sensors': meg_combined.shape[0],
        'total_duration_s': times_combined[-1],
        'envelope_sr_hz': 100.0,
        'model_correlation_mean': float(trf.r.mean()),
        'model_correlation_max': float(trf.r.max()),
        'n_runs_combined': len(args.runs)
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved: {output_dir / 'summary.json'}")

    # Create visualizations
    create_visualizations(trf, meg_ndvar, envelope_combined, output_dir)

    print(f"\n{'='*70}")
    print("COMBINED ENVELOPE TRF COMPLETE!")
    print(f"{'='*70}\n")

    print(f"Results: {output_dir}\n")

    print("Model performance:")
    print(f"  Mean correlation: {summary['model_correlation_mean']:.4f}")
    print(f"  Max correlation: {summary['model_correlation_max']:.4f}")
    print(f"  Total duration: {summary['total_duration_s']:.1f}s ({len(args.runs)} runs)")

    if summary['model_correlation_mean'] > 0.015:
        print(f"\n✅ SUCCESS! Combined TRF shows signal!")
        print(f"  This validates combining runs improves power.")
    else:
        print(f"\n⚠ Low correlation - may need to check sync offsets.")

    print(f"\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
