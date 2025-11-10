#!/usr/bin/env python3
"""
Group-Level TRF Analysis Script

Computes group averages across all subjects for:
- TRF kernels (envelope and F0)
- Model correlations
- Statistical testing
- Visualization

Creates grand average plots and saves group-level statistics.
"""

import pickle
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import argparse

# Configuration
PIPELINE_DIR = Path("/Users/em18033/Library/CloudStorage/OneDrive-AUTUniversity/Projects/turn-taking-pipeline")

SUBJECTS = [
    'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08',
    'sub-09', 'sub-10', 'sub-11', 'sub-13', 'sub-14', 'sub-15', 'sub-16', 'sub-17',
    'sub-18', 'sub-19', 'sub-21', 'sub-22', 'sub-23', 'sub-24', 'sub-25', 'sub-26',
    'sub-27', 'sub-29', 'sub-31', 'sub-32'
]


def load_subject_data(subject, runs, sensor_type):
    """Load TRF models and summary for a subject."""
    trf_dir = PIPELINE_DIR / f"outputs/trf_multipredictor/{subject}/runs-{runs}/{sensor_type}"

    # Load TRF models
    model_file = trf_dir / "trf_models.pkl"
    if not model_file.exists():
        print(f"  WARNING: Missing TRF models for {subject}")
        return None

    with open(model_file, 'rb') as f:
        models = pickle.load(f)

    # Load summary
    summary_file = trf_dir / "summary.json"
    with open(summary_file, 'r') as f:
        summary = json.load(f)

    return {
        'models': models,
        'summary': summary,
        'subject': subject
    }


def extract_trf_kernels(data_list, model_name='envelope_only'):
    """Extract TRF kernels from all subjects."""
    kernels = []
    subjects = []

    for data in data_list:
        if data is None:
            continue

        model = data['models'][model_name]
        if hasattr(model, 'h') and model.h is not None:
            # Extract kernel data - shape should be (n_sensors, n_times)
            kernel = model.h.x  # Get underlying numpy array
            kernels.append(kernel)
            subjects.append(data['subject'])

    if len(kernels) == 0:
        return None, None

    # Stack kernels - shape: (n_subjects, n_sensors, n_times)
    kernels_array = np.stack(kernels, axis=0)

    return kernels_array, subjects


def extract_correlations(data_list, model_name='full'):
    """Extract correlations from all subjects."""
    correlations = []
    subjects = []

    for data in data_list:
        if data is None:
            continue

        # Get correlation from summary
        corr = data['summary']['model_correlations'][model_name]
        correlations.append(corr)
        subjects.append(data['subject'])

    return np.array(correlations), subjects


def align_polarities(kernels_array):
    """
    Align polarities across subjects for each sensor independently.

    Uses the sign of the peak (max absolute value) to determine polarity.
    Flips kernels where needed so all subjects have consistent polarity per sensor.

    Parameters
    ----------
    kernels_array : ndarray
        Shape (n_subjects, n_sensors, n_times)

    Returns
    -------
    aligned_kernels : ndarray
        Polarity-aligned kernels with same shape
    flip_matrix : ndarray
        Shape (n_subjects, n_sensors) - binary matrix indicating which were flipped
    """
    n_subjects, n_sensors, n_times = kernels_array.shape
    aligned_kernels = kernels_array.copy()
    flip_matrix = np.zeros((n_subjects, n_sensors), dtype=bool)

    # For each sensor independently
    for sensor_idx in range(n_sensors):
        # Get all subjects' kernels for this sensor
        sensor_kernels = kernels_array[:, sensor_idx, :]  # (n_subjects, n_times)

        # Find peak (max absolute value) for each subject
        peaks = sensor_kernels[np.arange(n_subjects), np.abs(sensor_kernels).argmax(axis=1)]

        # Determine reference polarity (use majority vote)
        n_positive = np.sum(peaks > 0)
        n_negative = np.sum(peaks < 0)
        reference_positive = n_positive >= n_negative

        # Flip subjects that don't match reference polarity
        for subj_idx in range(n_subjects):
            peak_positive = peaks[subj_idx] > 0
            if peak_positive != reference_positive:
                aligned_kernels[subj_idx, sensor_idx, :] *= -1
                flip_matrix[subj_idx, sensor_idx] = True

    return aligned_kernels, flip_matrix


def smooth_trf_temporal(kernels_array, sigma_ms=10, sampling_rate_hz=200):
    """
    Apply temporal smoothing to TRF kernels using Gaussian filter.

    Parameters
    ----------
    kernels_array : ndarray
        Shape (n_subjects, n_sensors, n_times) or (n_sensors, n_times)
    sigma_ms : float
        Gaussian kernel width in milliseconds (default: 10ms)
    sampling_rate_hz : float
        Sampling rate in Hz (default: 200 Hz = 5ms resolution after downsampling)

    Returns
    -------
    smoothed : ndarray
        Temporally smoothed kernels with same shape
    """
    # Convert sigma from ms to samples
    sigma_samples = sigma_ms * (sampling_rate_hz / 1000.0)

    # Apply Gaussian filter along time axis (last dimension)
    smoothed = gaussian_filter1d(kernels_array, sigma=sigma_samples, axis=-1)

    return smoothed


def compute_group_statistics(kernels_array, align_polarity=True, smooth_sigma_ms=10):
    """
    Compute group-level statistics for TRF kernels.

    Parameters
    ----------
    kernels_array : ndarray
        Shape (n_subjects, n_sensors, n_times)
    align_polarity : bool
        If True, align polarities across subjects before averaging
    smooth_sigma_ms : float or None
        If provided, apply temporal Gaussian smoothing with this width in ms.
        Typical values: 5-20ms. Set to None to disable smoothing.

    Returns
    -------
    stats : dict
        Dictionary containing mean, sem, CI, t-stats, p-values, etc.
    """
    # Align polarities if requested
    if align_polarity:
        aligned_kernels, flip_matrix = align_polarities(kernels_array)
        n_flipped = np.sum(flip_matrix, axis=0)  # Count flips per sensor
    else:
        aligned_kernels = kernels_array
        flip_matrix = None
        n_flipped = None

    # Apply temporal smoothing if requested
    if smooth_sigma_ms is not None and smooth_sigma_ms > 0:
        smoothed_kernels = smooth_trf_temporal(aligned_kernels, sigma_ms=smooth_sigma_ms)
    else:
        smoothed_kernels = aligned_kernels

    # Mean across subjects
    mean_kernel = np.mean(smoothed_kernels, axis=0)  # (n_sensors, n_times)

    # Standard error
    sem_kernel = stats.sem(smoothed_kernels, axis=0)  # (n_sensors, n_times)

    # 95% confidence intervals
    ci_kernel = 1.96 * sem_kernel

    # One-sample t-test against zero at each time point
    t_stats, p_values = stats.ttest_1samp(smoothed_kernels, 0, axis=0)

    result = {
        'mean': mean_kernel,
        'sem': sem_kernel,
        'ci_95': ci_kernel,
        't_stats': t_stats,
        'p_values': p_values,
        'n_subjects': smoothed_kernels.shape[0],
        'polarity_aligned': align_polarity,
        'temporal_smoothing_ms': smooth_sigma_ms
    }

    if align_polarity:
        result['flip_matrix'] = flip_matrix
        result['n_flipped_per_sensor'] = n_flipped
        result['total_flips'] = int(np.sum(flip_matrix))
        result['pct_flipped'] = 100 * np.sum(flip_matrix) / flip_matrix.size

    return result


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
    ndarray
        Polarity-aligned mean across sensors, shape (n_times,)
    """
    data = kernel_data.copy()
    n_sensors = data.shape[0]

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

    return np.mean(data, axis=0)  # Average across sensors


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


def plot_group_trf_kernels(envelope_stats, f0_stats, times, output_file):
    """Plot group-average TRF kernels with dual visualization: polarity-aligned and RMS."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Crop to -0.1 to 0.4s to avoid edge artifacts
    time_mask = (times >= -0.1) & (times <= 0.4)
    times_cropped = times[time_mask]

    # Compute polarity-aligned averages across sensors (using M100 window)
    envelope_mean_aligned = align_sensor_polarities(envelope_stats['mean'], times)
    envelope_ci_aligned = align_sensor_polarities(envelope_stats['ci_95'], times)

    f0_mean_aligned = align_sensor_polarities(f0_stats['mean'], times)
    f0_ci_aligned = align_sensor_polarities(f0_stats['ci_95'], times)

    # Compute RMS across sensors
    envelope_mean_rms = compute_rms_across_sensors(envelope_stats['mean'])
    envelope_ci_rms = compute_rms_across_sensors(envelope_stats['ci_95'])

    f0_mean_rms = compute_rms_across_sensors(f0_stats['mean'])
    f0_ci_rms = compute_rms_across_sensors(f0_stats['ci_95'])

    # Apply time mask
    envelope_mean_aligned = envelope_mean_aligned[time_mask]
    envelope_ci_aligned = envelope_ci_aligned[time_mask]
    f0_mean_aligned = f0_mean_aligned[time_mask]
    f0_ci_aligned = f0_ci_aligned[time_mask]

    envelope_mean_rms = envelope_mean_rms[time_mask]
    envelope_ci_rms = envelope_ci_rms[time_mask]
    f0_mean_rms = f0_mean_rms[time_mask]
    f0_ci_rms = f0_ci_rms[time_mask]

    # TOP ROW: Polarity-aligned (signed mean)
    # Envelope TRF (polarity-aligned)
    axes[0, 0].plot(times_cropped, envelope_mean_aligned, 'b-', linewidth=2, label='Group Mean')
    axes[0, 0].fill_between(times_cropped,
                         envelope_mean_aligned - envelope_ci_aligned,
                         envelope_mean_aligned + envelope_ci_aligned,
                         alpha=0.3, color='b', label='95% CI')
    axes[0, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].axvline(0, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].set_xlabel('Time (s)', fontsize=11)
    axes[0, 0].set_ylabel('TRF Amplitude', fontsize=11)
    axes[0, 0].set_title(f'Envelope TRF - Polarity-Aligned\n(N={envelope_stats["n_subjects"]})',
                     fontsize=12, fontweight='bold')
    axes[0, 0].set_xlim(-0.1, 0.4)
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    # F0 TRF (polarity-aligned)
    axes[0, 1].plot(times_cropped, f0_mean_aligned, 'r-', linewidth=2, label='Group Mean')
    axes[0, 1].fill_between(times_cropped,
                         f0_mean_aligned - f0_ci_aligned,
                         f0_mean_aligned + f0_ci_aligned,
                         alpha=0.3, color='r', label='95% CI')
    axes[0, 1].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].axvline(0, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].set_xlabel('Time (s)', fontsize=11)
    axes[0, 1].set_ylabel('TRF Amplitude', fontsize=11)
    axes[0, 1].set_title(f'F0 TRF - Polarity-Aligned\n(N={f0_stats["n_subjects"]})',
                     fontsize=12, fontweight='bold')
    axes[0, 1].set_xlim(-0.1, 0.4)
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    # BOTTOM ROW: RMS across sensors (magnitude-only)
    # Envelope TRF (RMS)
    axes[1, 0].plot(times_cropped, envelope_mean_rms, 'b-', linewidth=2, label='RMS Mean')
    axes[1, 0].fill_between(times_cropped,
                         np.maximum(0, envelope_mean_rms - envelope_ci_rms),
                         envelope_mean_rms + envelope_ci_rms,
                         alpha=0.3, color='b', label='95% CI')
    axes[1, 0].axvline(0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].set_xlabel('Time (s)', fontsize=11)
    axes[1, 0].set_ylabel('TRF Magnitude (RMS)', fontsize=11)
    axes[1, 0].set_title(f'Envelope TRF - RMS Across Sensors\n(N={envelope_stats["n_subjects"]})',
                     fontsize=12, fontweight='bold')
    axes[1, 0].set_xlim(-0.1, 0.4)
    axes[1, 0].set_ylim(bottom=0)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    # F0 TRF (RMS)
    axes[1, 1].plot(times_cropped, f0_mean_rms, 'r-', linewidth=2, label='RMS Mean')
    axes[1, 1].fill_between(times_cropped,
                         np.maximum(0, f0_mean_rms - f0_ci_rms),
                         f0_mean_rms + f0_ci_rms,
                         alpha=0.3, color='r', label='95% CI')
    axes[1, 1].axvline(0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].set_xlabel('Time (s)', fontsize=11)
    axes[1, 1].set_ylabel('TRF Magnitude (RMS)', fontsize=11)
    axes[1, 1].set_title(f'F0 TRF - RMS Across Sensors\n(N={f0_stats["n_subjects"]})',
                     fontsize=12, fontweight='bold')
    axes[1, 1].set_xlim(-0.1, 0.4)
    axes[1, 1].set_ylim(bottom=0)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_group_correlations(data_list, output_file):
    """Plot distribution of correlations across subjects."""
    # Extract correlations for all models
    full_corrs, subjects = extract_correlations(data_list, 'full')
    env_corrs, _ = extract_correlations(data_list, 'envelope_only')
    f0_corrs, _ = extract_correlations(data_list, 'f0_only')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Full model
    axes[0].hist(full_corrs, bins=15, alpha=0.7, color='purple', edgecolor='black')
    axes[0].axvline(np.mean(full_corrs), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(full_corrs):.4f}')
    axes[0].set_xlabel('Correlation', fontsize=12)
    axes[0].set_ylabel('Number of Subjects', fontsize=12)
    axes[0].set_title(f'Full Model (Env+F0)\nN={len(full_corrs)}', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Envelope only
    axes[1].hist(env_corrs, bins=15, alpha=0.7, color='blue', edgecolor='black')
    axes[1].axvline(np.mean(env_corrs), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(env_corrs):.4f}')
    axes[1].set_xlabel('Correlation', fontsize=12)
    axes[1].set_ylabel('Number of Subjects', fontsize=12)
    axes[1].set_title(f'Envelope Only\nN={len(env_corrs)}', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F0 only
    axes[2].hist(f0_corrs, bins=15, alpha=0.7, color='red', edgecolor='black')
    axes[2].axvline(np.mean(f0_corrs), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(f0_corrs):.4f}')
    axes[2].set_xlabel('Correlation', fontsize=12)
    axes[2].set_ylabel('Number of Subjects', fontsize=12)
    axes[2].set_title(f'F0 Only\nN={len(f0_corrs)}', fontsize=12, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Group-Level Model Correlations', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    return {
        'full': {'mean': np.mean(full_corrs), 'std': np.std(full_corrs), 'sem': stats.sem(full_corrs)},
        'envelope_only': {'mean': np.mean(env_corrs), 'std': np.std(env_corrs), 'sem': stats.sem(env_corrs)},
        'f0_only': {'mean': np.mean(f0_corrs), 'std': np.std(f0_corrs), 'sem': stats.sem(f0_corrs)}
    }


def plot_subject_comparison(data_list, output_file):
    """Plot per-subject correlations for comparison."""
    full_corrs, subjects = extract_correlations(data_list, 'full')
    env_corrs, _ = extract_correlations(data_list, 'envelope_only')
    f0_corrs, _ = extract_correlations(data_list, 'f0_only')

    fig, ax = plt.subplots(figsize=(16, 8))

    x = np.arange(len(subjects))
    width = 0.25

    ax.bar(x - width, full_corrs, width, label='Full (Env+F0)', color='purple', alpha=0.7)
    ax.bar(x, env_corrs, width, label='Envelope Only', color='blue', alpha=0.7)
    ax.bar(x + width, f0_corrs, width, label='F0 Only', color='red', alpha=0.7)

    ax.set_xlabel('Subject', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_title('Model Correlations by Subject', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def save_group_statistics(envelope_stats, f0_stats, corr_stats, output_file):
    """Save group statistics to JSON."""
    stats_dict = {
        'envelope_trf': {
            'n_subjects': int(envelope_stats['n_subjects']),
            'mean_amplitude_range': [float(envelope_stats['mean'].min()),
                                    float(envelope_stats['mean'].max())],
            'polarity_aligned': envelope_stats['polarity_aligned'],
        },
        'f0_trf': {
            'n_subjects': int(f0_stats['n_subjects']),
            'mean_amplitude_range': [float(f0_stats['mean'].min()),
                                    float(f0_stats['mean'].max())],
            'polarity_aligned': f0_stats['polarity_aligned'],
        },
        'correlations': corr_stats
    }

    # Add polarity alignment details if available
    if envelope_stats['polarity_aligned']:
        stats_dict['envelope_trf']['polarity_alignment'] = {
            'total_flips': envelope_stats['total_flips'],
            'pct_flipped': float(envelope_stats['pct_flipped']),
            'n_flipped_per_sensor': envelope_stats['n_flipped_per_sensor'].tolist()
        }

    if f0_stats['polarity_aligned']:
        stats_dict['f0_trf']['polarity_alignment'] = {
            'total_flips': f0_stats['total_flips'],
            'pct_flipped': float(f0_stats['pct_flipped']),
            'n_flipped_per_sensor': f0_stats['n_flipped_per_sensor'].tolist()
        }

    with open(output_file, 'w') as f:
        json.dump(stats_dict, f, indent=2)

    print(f"  Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Group-level TRF analysis')
    parser.add_argument('--runs', type=str, default='1_3_5',
                       help='Run combination (e.g., "1_3_5" or "2_4")')
    parser.add_argument('--sensor-type', type=str, default='mag',
                       choices=['mag', 'eeg'],
                       help='Sensor type (mag or eeg)')
    parser.add_argument('--subjects', nargs='+', default=None,
                       help='Specific subjects to include (default: all)')
    parser.add_argument('--smooth', type=float, default=10,
                       help='Temporal smoothing width in ms (default: 10, use 0 to disable)')

    args = parser.parse_args()

    subjects = args.subjects if args.subjects else SUBJECTS

    print("="*70)
    print(f"GROUP-LEVEL TRF ANALYSIS")
    print("="*70)
    print(f"Runs: {args.runs}")
    print(f"Sensor type: {args.sensor_type}")
    print(f"Number of subjects: {len(subjects)}")
    print(f"Temporal smoothing: {args.smooth}ms" if args.smooth > 0 else "Temporal smoothing: disabled")
    print()

    # Create output directory
    output_dir = PIPELINE_DIR / f"outputs/group_trf/runs-{args.runs}/{args.sensor_type}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data from all subjects
    print("Loading subject data...")
    data_list = []
    for subject in subjects:
        print(f"  Loading {subject}...")
        data = load_subject_data(subject, args.runs, args.sensor_type)
        data_list.append(data)

    # Remove None entries
    data_list = [d for d in data_list if d is not None]
    print(f"\nSuccessfully loaded {len(data_list)} subjects\n")

    # Extract TRF kernels
    print("Extracting TRF kernels...")
    envelope_kernels, env_subjects = extract_trf_kernels(data_list, 'envelope_only')
    f0_kernels, f0_subjects = extract_trf_kernels(data_list, 'f0_only')

    # Compute group statistics
    print("Computing group statistics...")
    smooth_sigma = args.smooth if args.smooth > 0 else None
    envelope_stats = compute_group_statistics(envelope_kernels, smooth_sigma_ms=smooth_sigma)
    f0_stats = compute_group_statistics(f0_kernels, smooth_sigma_ms=smooth_sigma)

    # Get time axis (assuming -0.2 to 0.5s at 1ms resolution)
    n_times = envelope_kernels.shape[2]
    times = np.linspace(-0.2, 0.5, n_times)

    # Save group-average kernels
    print("\nSaving group-average data...")
    np.save(output_dir / 'envelope_group_mean.npy', envelope_stats['mean'])
    np.save(output_dir / 'envelope_group_sem.npy', envelope_stats['sem'])
    np.save(output_dir / 'f0_group_mean.npy', f0_stats['mean'])
    np.save(output_dir / 'f0_group_sem.npy', f0_stats['sem'])
    np.save(output_dir / 'times.npy', times)
    print(f"  Saved group-average kernel data")

    # Generate plots
    print("\nGenerating plots...")
    plot_group_trf_kernels(envelope_stats, f0_stats, times,
                          output_dir / 'group_trf_kernels.png')

    corr_stats = plot_group_correlations(data_list,
                                         output_dir / 'group_correlations.png')

    plot_subject_comparison(data_list,
                           output_dir / 'subject_correlations.png')

    # Save statistics
    print("\nSaving group statistics...")
    save_group_statistics(envelope_stats, f0_stats, corr_stats,
                         output_dir / 'group_statistics.json')

    # Print summary
    print("\n" + "="*70)
    print("GROUP STATISTICS SUMMARY")
    print("="*70)
    print(f"\nEnvelope TRF:")
    print(f"  N subjects: {envelope_stats['n_subjects']}")
    print(f"  Mean amplitude range: [{envelope_stats['mean'].min():.4f}, {envelope_stats['mean'].max():.4f}]")
    if envelope_stats['polarity_aligned']:
        print(f"  Polarity alignment: {envelope_stats['total_flips']} flips ({envelope_stats['pct_flipped']:.1f}%)")
        print(f"  Flips per sensor: {envelope_stats['n_flipped_per_sensor']}")

    print(f"\nF0 TRF:")
    print(f"  N subjects: {f0_stats['n_subjects']}")
    print(f"  Mean amplitude range: [{f0_stats['mean'].min():.4f}, {f0_stats['mean'].max():.4f}]")
    if f0_stats['polarity_aligned']:
        print(f"  Polarity alignment: {f0_stats['total_flips']} flips ({f0_stats['pct_flipped']:.1f}%)")
        print(f"  Flips per sensor: {f0_stats['n_flipped_per_sensor']}")

    print(f"\nModel Correlations:")
    print(f"  Full model: {corr_stats['full']['mean']:.4f} ± {corr_stats['full']['sem']:.4f}")
    print(f"  Envelope only: {corr_stats['envelope_only']['mean']:.4f} ± {corr_stats['envelope_only']['sem']:.4f}")
    print(f"  F0 only: {corr_stats['f0_only']['mean']:.4f} ± {corr_stats['f0_only']['sem']:.4f}")

    print(f"\n" + "="*70)
    print(f"RESULTS SAVED TO:")
    print(f"{output_dir}")
    print("="*70)
    print("\nGenerated files:")
    print(f"  - envelope_group_mean.npy")
    print(f"  - envelope_group_sem.npy")
    print(f"  - f0_group_mean.npy")
    print(f"  - f0_group_sem.npy")
    print(f"  - times.npy")
    print(f"  - group_trf_kernels.png")
    print(f"  - group_correlations.png")
    print(f"  - subject_correlations.png")
    print(f"  - group_statistics.json")
    print()


if __name__ == '__main__':
    main()
