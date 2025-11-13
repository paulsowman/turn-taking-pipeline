#!/usr/bin/env python3
"""
Combine Multiple Runs into Condition-Specific FIF Files

Combines runs for each subject based on experimental condition:
- Condition 1 (conversation): Runs 1, 3, 5 - Natural free conversation
- Condition 2 (nursery_rhyme): Runs 2, 4 - Structured nursery rhyme repetition

Adds gap annotations at concatenation boundaries to avoid edge artifacts
in TRF analysis.

Usage:
    # Single subject, both conditions
    python scripts/combine_runs.py --subject sub-01

    # All subjects
    python scripts/combine_runs.py --all

    # Specific condition only
    python scripts/combine_runs.py --subject sub-01 --condition conversation
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import mne
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import load_config
from utils.logging_setup import setup_logging

# All available subjects
ALL_SUBJECTS = [
    'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05',
    'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10',
    'sub-11', 'sub-13', 'sub-14', 'sub-15', 'sub-16',
    'sub-17', 'sub-18', 'sub-19', 'sub-21', 'sub-22',
    'sub-23', 'sub-24', 'sub-25', 'sub-26', 'sub-27',
    'sub-29', 'sub-31', 'sub-32'
]

# Experimental design
CONDITION_RUNS = {
    'conversation': [1, 3, 5],  # Natural free conversation
    'nursery_rhyme': [2, 4],    # Structured nursery rhyme repetition (control)
}


def combine_runs(
    subject: str,
    condition: str,
    base_dir: Path,
    gap_duration: float = 2.0,
    overwrite: bool = False,
) -> bool:
    """
    Combine runs for one subject/condition.

    Parameters
    ----------
    subject : str
        Subject ID
    condition : str
        Condition name ('conversation' or 'nursery_rhyme')
    base_dir : Path
        Base directory
    gap_duration : float
        Duration of gap (seconds) to insert between runs (default: 2.0)
    overwrite : bool
        Whether to overwrite existing combined file

    Returns
    -------
    success : bool
    """
    print(f"\n{'='*70}")
    print(f"COMBINING RUNS: {subject} - {condition.upper()}")
    print(f"{'='*70}\n")

    runs = CONDITION_RUNS[condition]
    print(f"  Condition: {condition}")
    print(f"  Runs to combine: {runs}")
    print(f"  Gap between runs: {gap_duration}s\n")

    # Check output
    output_dir = base_dir / "outputs" / "trf_combined" / subject
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{subject}_{condition}_trf_raw.fif"

    if output_file.exists() and not overwrite:
        print(f"✓ Combined file already exists: {output_file}")
        print(f"  Use --overwrite to regenerate")
        return True

    # Load individual run files
    raw_files = []
    missing_runs = []

    for run in runs:
        trf_file = base_dir / "outputs" / "trf" / subject / f"run-{run:02d}" / f"{subject}_run-{run:02d}_trf_raw.fif"

        if not trf_file.exists():
            print(f"  ✗ Missing run {run}: {trf_file}")
            missing_runs.append(run)
            continue

        print(f"  Loading run {run}: {trf_file.name}")
        raw = mne.io.read_raw_fif(trf_file, preload=True, verbose=False)
        raw_files.append((run, raw))

    if missing_runs:
        print(f"\n✗ Cannot combine - missing runs: {missing_runs}")
        print(f"  Run create_trf_fif.py for these runs first")
        return False

    if len(raw_files) < 2:
        print(f"\n✗ Need at least 2 runs to combine, found {len(raw_files)}")
        return False

    # Get sampling frequency
    sfreq = raw_files[0][1].info['sfreq']
    gap_samples = int(gap_duration * sfreq)

    print(f"\n  Sampling rate: {sfreq} Hz")
    print(f"  Gap samples: {gap_samples}")

    # Concatenate with MNE
    print(f"\n  Concatenating {len(raw_files)} runs...")
    raws_only = [raw for _, raw in raw_files]

    # Concatenate
    raw_combined = mne.concatenate_raws(raws_only, preload=True, verbose=False)

    print(f"  ✓ Combined duration: {raw_combined.times[-1]:.1f}s")
    print(f"  ✓ Total samples: {len(raw_combined.times)}")

    # Add BAD annotations at concatenation boundaries
    # MNE automatically adds boundary events, we'll mark regions around them as BAD
    print(f"\n  Adding BAD annotations at concatenation boundaries...")

    # Calculate cumulative durations to find boundaries
    cumulative_duration = 0.0
    boundary_times = []

    for i, (run, raw) in enumerate(raw_files[:-1]):  # All except last
        run_duration = raw.times[-1]
        cumulative_duration += run_duration
        boundary_times.append(cumulative_duration)
        print(f"    Boundary after run {run}: {cumulative_duration:.2f}s")

    # Create BAD annotations around boundaries
    onsets = []
    durations = []
    descriptions = []

    for boundary_time in boundary_times:
        # Mark region from -gap_duration/2 to +gap_duration/2 around boundary
        onset = boundary_time - gap_duration / 2
        if onset < 0:
            onset = 0

        onsets.append(onset)
        durations.append(gap_duration)
        descriptions.append('BAD_concatenation_boundary')

    # Add annotations
    if onsets:
        bad_annot = mne.Annotations(
            onset=onsets,
            duration=durations,
            description=descriptions,
            orig_time=raw_combined.info['meas_date']
        )

        # Merge with any existing annotations
        if raw_combined.annotations is not None:
            raw_combined.set_annotations(raw_combined.annotations + bad_annot)
        else:
            raw_combined.set_annotations(bad_annot)

        print(f"  ✓ Added {len(onsets)} BAD annotations")

    # Save combined file
    print(f"\n  Saving combined file...")
    raw_combined.save(output_file, overwrite=True, verbose=False)

    print(f"✓ Saved: {output_file}")
    print(f"  Total channels: {len(raw_combined.ch_names)}")
    print(f"  MEG channels: {len(mne.pick_types(raw_combined.info, meg=True))}")
    print(f"  MISC channels: {len(mne.pick_types(raw_combined.info, misc=True))}")

    # Save metadata
    metadata_file = output_dir / f"{subject}_{condition}_metadata.json"

    # Calculate statistics across runs
    total_words_interviewer = 0
    total_words_participant = 0

    for run, _ in raw_files:
        metadata_path = base_dir / "outputs" / "trf" / subject / f"run-{run:02d}" / f"{subject}_run-{run:02d}_predictor_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                run_meta = json.load(f)
                total_words_interviewer += run_meta.get('n_words_interviewer', 0)
                total_words_participant += run_meta.get('n_words_participant', 0)

    metadata = {
        'subject': subject,
        'condition': condition,
        'condition_description': 'Natural free conversation' if condition == 'conversation' else 'Structured nursery rhyme repetition',
        'runs_combined': runs,
        'n_runs': len(runs),
        'gap_duration_s': gap_duration,
        'concatenation_boundaries_s': boundary_times,
        'total_duration_s': float(raw_combined.times[-1]),
        'sfreq': sfreq,
        'n_words_interviewer': total_words_interviewer,
        'n_words_participant': total_words_participant,
        'n_words_total': total_words_interviewer + total_words_participant,
        'n_meg_channels': len(mne.pick_types(raw_combined.info, meg=True)),
        'n_misc_channels': len(mne.pick_types(raw_combined.info, misc=True)),
        'bad_annotations': len(onsets),
    }

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved metadata: {metadata_file}")
    print(f"\n  Summary:")
    print(f"    Condition: {metadata['condition_description']}")
    print(f"    Total words: {total_words_interviewer + total_words_participant}")
    print(f"    Interviewer: {total_words_interviewer} words")
    print(f"    Participant: {total_words_participant} words")
    print(f"    Duration: {raw_combined.times[-1]:.1f}s")
    print()

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Combine runs into condition-specific FIF files'
    )
    parser.add_argument(
        '--subject',
        nargs='+',
        help='Subject IDs (e.g., sub-01 sub-02)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all subjects'
    )
    parser.add_argument(
        '--condition',
        choices=['conversation', 'nursery_rhyme', 'both'],
        default='both',
        help='Which condition to combine (default: both)'
    )
    parser.add_argument(
        '--gap',
        type=float,
        default=2.0,
        help='Gap duration between runs in seconds (default: 2.0)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing combined files'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging("combine_runs")

    # Base directory
    base_dir = Path(__file__).parent.parent

    # Load config
    config = load_config()

    # Determine subjects
    if args.all:
        subjects = ALL_SUBJECTS
    else:
        if not args.subject:
            print("Error: Must specify --subject or --all")
            sys.exit(1)
        subjects = args.subject

    # Determine conditions
    if args.condition == 'both':
        conditions = ['conversation', 'nursery_rhyme']
    else:
        conditions = [args.condition]

    print("="*70)
    print("COMBINE RUNS BY CONDITION")
    print("="*70)
    print(f"\nSubjects: {', '.join(subjects)}")
    print(f"Conditions: {', '.join(conditions)}")
    print(f"Gap between runs: {args.gap}s")
    print(f"Overwrite: {args.overwrite}")
    print(f"\nCondition design:")
    for cond in conditions:
        runs = CONDITION_RUNS[cond]
        print(f"  {cond}: runs {runs}")
    print(f"\nTotal combinations: {len(subjects)} × {len(conditions)} = {len(subjects) * len(conditions)}")
    print()

    # Process each subject/condition
    results = []
    for subject in subjects:
        for condition in conditions:
            try:
                success = combine_runs(
                    subject=subject,
                    condition=condition,
                    base_dir=base_dir,
                    gap_duration=args.gap,
                    overwrite=args.overwrite,
                )
                results.append({
                    'subject': subject,
                    'condition': condition,
                    'success': success
                })
            except Exception as e:
                print(f"\n✗ Error processing {subject} {condition}: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'subject': subject,
                    'condition': condition,
                    'success': False
                })

    # Summary
    print("\n" + "="*70)
    print("COMBINATION SUMMARY")
    print("="*70)

    n_success = sum(1 for r in results if r['success'])
    n_fail = sum(1 for r in results if not r['success'])

    print(f"\nTotal: {len(results)}")
    print(f"  ✓ Success: {n_success}")
    print(f"  ✗ Failed: {n_fail}")

    if n_fail > 0:
        print(f"\nFailed combinations:")
        for r in results:
            if not r['success']:
                print(f"  - {r['subject']} {r['condition']}")

    print(f"\n{'='*70}")
    print("OUTPUT STRUCTURE")
    print(f"{'='*70}")
    print(f"\nCombined files saved to:")
    print(f"  outputs/trf_combined/{{subject}}/{{subject}}_{{condition}}_trf_raw.fif")
    print(f"\nMetadata saved to:")
    print(f"  outputs/trf_combined/{{subject}}/{{subject}}_{{condition}}_metadata.json")
    print(f"\nNext steps:")
    print(f"  1. Load combined files for TRF analysis")
    print(f"  2. Use BAD annotations to exclude concatenation boundaries")
    print(f"  3. Compare conversation vs nursery_rhyme conditions")
    print()


if __name__ == "__main__":
    main()
