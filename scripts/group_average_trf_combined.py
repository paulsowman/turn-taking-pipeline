#!/usr/bin/env python3
"""
Group-Level TRF Analysis for Combined Pipeline

Computes group averages across all subjects for:
- TRF kernels for each predictor
- Polarity-aligned sensor means
- Condition comparisons (conversation vs nursery rhyme)
- Statistical testing
- Grand average visualization

Compatible with outputs from analyze_trf_combined.py
"""

import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
import argparse
import sys

# Use non-interactive backend
matplotlib.use('Agg')


def align_sensor_polarities(kernel_data, times, m100_window=(0.08, 0.15)):
    """
    Align sensor polarities by flipping those with negative peaks in M100 window.

    Parameters
    ----------
    kernel_data : ndarray
        Shape (n_sensors, n_times)
    times : ndarray
        Time points in seconds
    m100_window : tuple
        (start, end) in seconds for M100 window

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

    if not np.any(m100_mask):
        print(f"  WARNING: No time points in M100 window {m100_window}")
        return np.mean(data, axis=0), 0

    for i in range(n_sensors):
        m100_data = data[i, m100_mask]
        if len(m100_data) == 0:
            continue

        # Find peak in M100 window
        peak_idx_in_window = np.argmax(np.abs(m100_data))
        peak_value = m100_data[peak_idx_in_window]

        # Flip if negative
        if peak_value < 0:
            data[i, :] *= -1
            n_flipped += 1

    return np.mean(data, axis=0), n_flipped


def load_subject_trf(subject, condition, speaker='participant'):
    """
    Load TRF model for one subject/condition.

    Parameters
    ----------
    subject : str
        e.g., 'sub-01'
    condition : str
        'conversation' or 'nursery_rhyme'
    speaker : str
        'participant', 'interviewer', or 'both'

    Returns
    -------
    trf : TRF object or None
    """
    trf_file = Path('outputs/trf_analysis') / subject / condition / speaker / 'trf_model.pickle'

    if not trf_file.exists():
        return None

    try:
        with open(trf_file, 'rb') as f:
            trf = pickle.load(f)
        return trf
    except Exception as e:
        print(f"  ERROR loading {subject}/{condition}: {e}")
        return None


def extract_predictor_kernels(trf, predictor_idx=0):
    """
    Extract kernel data for a specific predictor.

    Parameters
    ----------
    trf : TRF object
        From eelbrain boosting
    predictor_idx : int
        Index of predictor (0=first predictor, etc.)

    Returns
    -------
    kernel_data : ndarray
        Shape (n_sensors, n_times)
    times : ndarray
        Time points in seconds
    """
    if isinstance(trf.h, tuple):
        h = trf.h[predictor_idx]
    else:
        h = trf.h

    times = h.time.times if hasattr(h.time, 'times') else h.time
    kernel_data = h.x

    return kernel_data, times


def compute_group_average(subjects, condition, speaker='participant', predictor_idx=0,
                         predictor_name='envelope'):
    """
    Compute group average TRF for one predictor.

    Parameters
    ----------
    subjects : list of str
        Subject IDs
    condition : str
        'conversation' or 'nursery_rhyme'
    speaker : str
        Which speaker's predictors
    predictor_idx : int
        Which predictor to extract
    predictor_name : str
        Name for display

    Returns
    -------
    dict with keys:
        - group_mean: polarity-aligned group average (n_times,)
        - times: time points
        - n_subjects: number of subjects included
        - subject_means: individual subject means (n_subjects, n_times)
        - sem: standard error of mean across subjects
    """
    subject_means = []
    times_ref = None
    subjects_included = []

    print(f"\nLoading {condition} - {predictor_name}:")

    for subject in subjects:
        trf = load_subject_trf(subject, condition, speaker)

        if trf is None:
            print(f"  {subject}: MISSING")
            continue

        try:
            kernel_data, times = extract_predictor_kernels(trf, predictor_idx)

            # Align polarities within subject
            aligned_mean, n_flipped = align_sensor_polarities(kernel_data, times)

            subject_means.append(aligned_mean)
            subjects_included.append(subject)

            if times_ref is None:
                times_ref = times

            print(f"  {subject}: ✓ ({n_flipped} sensors flipped)")

        except Exception as e:
            print(f"  {subject}: ERROR - {e}")
            continue

    if len(subject_means) == 0:
        print(f"  ERROR: No subjects loaded for {condition}/{predictor_name}")
        return None

    # Stack subject means
    subject_means = np.array(subject_means)  # Shape: (n_subjects, n_times)

    # Compute group statistics
    group_mean = np.mean(subject_means, axis=0)
    sem = stats.sem(subject_means, axis=0)

    print(f"  Group: {len(subjects_included)} subjects included")

    return {
        'group_mean': group_mean,
        'times': times_ref,
        'n_subjects': len(subjects_included),
        'subject_means': subject_means,
        'sem': sem,
        'subjects_included': subjects_included
    }


def plot_group_trf(group_data, predictor_name, condition, output_dir):
    """
    Create group average TRF plot with SEM shading.

    Parameters
    ----------
    group_data : dict
        From compute_group_average
    predictor_name : str
        For title/filename
    condition : str
        'conversation' or 'nursery_rhyme'
    output_dir : Path
        Where to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    times = group_data['times']
    mean = group_data['group_mean']
    sem = group_data['sem']
    n = group_data['n_subjects']

    # Plot mean
    ax.plot(times, mean, linewidth=2.5, color='red', label=f'Group mean (n={n})')

    # SEM shading
    ax.fill_between(times, mean - sem, mean + sem, alpha=0.3, color='red', label='±SEM')

    # Reference lines
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(0, color='k', linestyle='--', alpha=0.3)

    # Formatting
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('TRF amplitude (polarity-aligned)', fontsize=12)
    ax.set_title(f'{predictor_name} - {condition.replace("_", " ").title()}\nGroup Average (n={n})',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f'group_{predictor_name}_{condition}.png'
    fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"  ✓ Saved: {filename}")


def plot_condition_comparison(conv_data, nursery_data, predictor_name, output_dir):
    """
    Compare conversation vs nursery rhyme on same plot.

    Parameters
    ----------
    conv_data : dict
        Group data for conversation
    nursery_data : dict
        Group data for nursery rhyme
    predictor_name : str
        For title/filename
    output_dir : Path
        Where to save plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Conversation
    times_conv = conv_data['times']
    mean_conv = conv_data['group_mean']
    sem_conv = conv_data['sem']
    n_conv = conv_data['n_subjects']

    ax.plot(times_conv, mean_conv, linewidth=2.5, color='blue',
            label=f'Conversation (n={n_conv})')
    ax.fill_between(times_conv, mean_conv - sem_conv, mean_conv + sem_conv,
                    alpha=0.2, color='blue')

    # Nursery rhyme
    times_nurs = nursery_data['times']
    mean_nurs = nursery_data['group_mean']
    sem_nurs = nursery_data['sem']
    n_nurs = nursery_data['n_subjects']

    ax.plot(times_nurs, mean_nurs, linewidth=2.5, color='green',
            label=f'Nursery Rhyme (n={n_nurs})')
    ax.fill_between(times_nurs, mean_nurs - sem_nurs, mean_nurs + sem_nurs,
                    alpha=0.2, color='green')

    # Reference lines
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(0, color='k', linestyle='--', alpha=0.3)

    # Formatting
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('TRF amplitude (polarity-aligned)', fontsize=12)
    ax.set_title(f'{predictor_name} - Condition Comparison',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f'group_{predictor_name}_comparison.png'
    fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"  ✓ Saved: {filename}")


def run_statistics(conv_data, nursery_data, predictor_name):
    """
    Run statistical tests comparing conditions.

    Parameters
    ----------
    conv_data : dict
        Group data for conversation
    nursery_data : dict
        Group data for nursery rhyme
    predictor_name : str
        For display

    Returns
    -------
    dict with statistical results
    """
    print(f"\nStatistics for {predictor_name}:")

    # Check if times match
    if not np.allclose(conv_data['times'], nursery_data['times']):
        print("  WARNING: Time points don't match between conditions")
        return None

    times = conv_data['times']
    conv_means = conv_data['subject_means']  # (n_subjects, n_times)
    nurs_means = nursery_data['subject_means']

    # Paired t-test at each time point
    n_times = len(times)
    t_values = np.zeros(n_times)
    p_values = np.zeros(n_times)

    for t in range(n_times):
        t_val, p_val = stats.ttest_rel(conv_means[:, t], nurs_means[:, t])
        t_values[t] = t_val
        p_values[t] = p_val

    # Find significant time points (uncorrected p < 0.05)
    sig_mask = p_values < 0.05
    n_sig = np.sum(sig_mask)

    print(f"  Paired t-test (n={conv_data['n_subjects']} subjects)")
    print(f"  Significant time points (p < 0.05, uncorrected): {n_sig}/{n_times}")

    if n_sig > 0:
        sig_times = times[sig_mask]
        print(f"  Time range of significance: {sig_times[0]*1000:.1f} to {sig_times[-1]*1000:.1f} ms")

    # Peak amplitude comparison
    conv_peak = np.max(np.abs(conv_data['group_mean']))
    nurs_peak = np.max(np.abs(nursery_data['group_mean']))

    print(f"  Peak amplitude:")
    print(f"    Conversation: {conv_peak:.4f}")
    print(f"    Nursery Rhyme: {nurs_peak:.4f}")
    print(f"    Ratio: {conv_peak/nurs_peak:.2f}x")

    return {
        'times': times,
        't_values': t_values,
        'p_values': p_values,
        'sig_mask': sig_mask,
        'n_sig': n_sig,
        'conv_peak': conv_peak,
        'nurs_peak': nurs_peak
    }


def main():
    parser = argparse.ArgumentParser(
        description='Group-level TRF analysis across subjects'
    )
    parser.add_argument(
        '--subjects',
        nargs='+',
        help='Subject IDs (default: sub-01 through sub-10)'
    )
    parser.add_argument(
        '--speaker',
        choices=['participant', 'interviewer', 'both'],
        default='participant',
        help='Which speaker to analyze (default: participant)'
    )
    parser.add_argument(
        '--predictors',
        nargs='+',
        default=['envelope', 'word_onsets', 'surprisal', 'f0_deviation', 'duration_deviation'],
        help='Predictor names (must match order in analyze_trf_combined.py)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/group_trf_combined',
        help='Output directory for group plots'
    )

    args = parser.parse_args()

    # Default subjects if not specified
    if args.subjects is None:
        args.subjects = [f'sub-{i:02d}' for i in range(1, 11)]  # sub-01 to sub-10

    output_dir = Path(args.output_dir) / args.speaker

    print("="*70)
    print("GROUP-LEVEL TRF ANALYSIS")
    print("="*70)
    print(f"Subjects: {len(args.subjects)}")
    print(f"Speaker: {args.speaker}")
    print(f"Predictors: {args.predictors}")
    print(f"Output: {output_dir}")
    print("")

    # Process each predictor
    for pred_idx, pred_name in enumerate(args.predictors):
        print("\n" + "="*70)
        print(f"PREDICTOR: {pred_name}")
        print("="*70)

        # Load conversation
        conv_data = compute_group_average(
            args.subjects,
            'conversation',
            args.speaker,
            pred_idx,
            pred_name
        )

        # Load nursery rhyme
        nursery_data = compute_group_average(
            args.subjects,
            'nursery_rhyme',
            args.speaker,
            pred_idx,
            pred_name
        )

        if conv_data is None or nursery_data is None:
            print(f"  SKIPPING {pred_name} (missing data)")
            continue

        # Plot individual conditions
        print("\nCreating plots:")
        plot_group_trf(conv_data, pred_name, 'conversation', output_dir)
        plot_group_trf(nursery_data, pred_name, 'nursery_rhyme', output_dir)

        # Plot comparison
        plot_condition_comparison(conv_data, nursery_data, pred_name, output_dir)

        # Statistics
        stats_results = run_statistics(conv_data, nursery_data, pred_name)

        # Save statistics
        if stats_results is not None:
            stats_file = output_dir / f'statistics_{pred_name}.npz'
            np.savez(
                stats_file,
                times=stats_results['times'],
                t_values=stats_results['t_values'],
                p_values=stats_results['p_values'],
                sig_mask=stats_results['sig_mask'],
                conv_subjects=conv_data['subjects_included'],
                nurs_subjects=nursery_data['subjects_included']
            )
            print(f"  ✓ Saved statistics: statistics_{pred_name}.npz")

    print("\n" + "="*70)
    print("GROUP ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Check group plots in outputs/group_trf_combined/")
    print("  2. Review statistics files (.npz)")
    print("  3. Look for condition differences (conversation vs nursery rhyme)")


if __name__ == '__main__':
    main()
