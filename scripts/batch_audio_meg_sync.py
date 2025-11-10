#!/usr/bin/env python3
"""
Batch audio-MEG synchronization for all subjects and runs.
Generates sync parameters required for TRF analyses.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import argparse

# Pipeline paths
PIPELINE_DIR = Path(__file__).parent.parent
VENV_PYTHON = PIPELINE_DIR / "venv/bin/python"
SYNC_SCRIPT = PIPELINE_DIR / "scripts/run_audio_meg_sync.py"
MEG_BASE = Path.home() / "Library/CloudStorage/OneDrive-SharedLibraries-MacquarieUniversity/Natural_Conversations_study - Documents/analysis/natural-conversations-bids/derivatives/mne-bids-pipeline-20240628"

# All subjects with MEG data
ALL_SUBJECTS = [
    'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08',
    'sub-09', 'sub-10', 'sub-11', 'sub-13', 'sub-14', 'sub-15', 'sub-16', 'sub-17',
    'sub-18', 'sub-19', 'sub-21', 'sub-22', 'sub-23', 'sub-24', 'sub-25', 'sub-26',
    'sub-27', 'sub-29', 'sub-31', 'sub-32'
]

# Runs for conversation task
CONVERSATION_RUNS = [1, 2, 3, 4, 5]


def check_sync_exists(subject, run):
    """Check if sync params already exist for this subject/run."""
    sync_file = PIPELINE_DIR / f"outputs/sync/{subject}/run-{run:02d}/sync_params.json"
    return sync_file.exists()


def run_sync_for_subject_run(subject, run, force=False):
    """Run synchronization for a single subject and run."""

    # Check if already exists
    if not force and check_sync_exists(subject, run):
        print(f"  ✓ {subject} run-{run:02d}: Sync params already exist (skipping)")
        return True, "exists"

    # Build MEG file path
    meg_file = MEG_BASE / subject / "meg" / f"{subject}_task-conversation_run-{run:02d}_proc-clean_raw.fif"

    if not meg_file.exists():
        print(f"  ✗ {subject} run-{run:02d}: MEG file not found: {meg_file}")
        return False, "meg_missing"

    # Run sync script
    cmd = [
        str(VENV_PYTHON),
        str(SYNC_SCRIPT),
        "--subject", subject,
        "--run", str(run),
        "--meg-file", str(meg_file)
    ]

    print(f"  → {subject} run-{run:02d}: Running sync...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print(f"  ✓ {subject} run-{run:02d}: Sync complete")
            return True, "success"
        else:
            print(f"  ✗ {subject} run-{run:02d}: Sync failed")
            print(f"    Error: {result.stderr[:200]}")
            return False, "failed"

    except subprocess.TimeoutExpired:
        print(f"  ✗ {subject} run-{run:02d}: Sync timed out (> 5 min)")
        return False, "timeout"
    except Exception as e:
        print(f"  ✗ {subject} run-{run:02d}: Error: {e}")
        return False, "error"


def main():
    parser = argparse.ArgumentParser(description='Batch audio-MEG synchronization')
    parser.add_argument('--subjects', nargs='+', help='Specific subjects to process (default: all)')
    parser.add_argument('--runs', nargs='+', type=int, help='Specific runs to process (default: 1-5)')
    parser.add_argument('--force', action='store_true', help='Re-run sync even if params exist')
    parser.add_argument('--skip-existing', action='store_true', help='Skip subjects that already have all sync params')

    args = parser.parse_args()

    # Determine subjects and runs to process
    subjects = args.subjects if args.subjects else ALL_SUBJECTS
    runs = args.runs if args.runs else CONVERSATION_RUNS

    print("=" * 70)
    print(f"BATCH AUDIO-MEG SYNCHRONIZATION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"Subjects: {len(subjects)}")
    print(f"Runs: {runs}")
    print(f"Force re-sync: {args.force}")
    print("=" * 70)
    print()

    # Track statistics
    total_tasks = len(subjects) * len(runs)
    completed = 0
    skipped = 0
    failed = 0
    errors = []

    # Process each subject
    for i, subject in enumerate(subjects, 1):
        print(f"\n[{i}/{len(subjects)}] Processing {subject}")
        print("-" * 70)

        # Check if we should skip this subject
        if args.skip_existing:
            all_exist = all(check_sync_exists(subject, run) for run in runs)
            if all_exist:
                print(f"  ✓ {subject}: All sync params exist (skipping subject)")
                skipped += len(runs)
                continue

        # Process each run for this subject
        for run in runs:
            success, status = run_sync_for_subject_run(subject, run, args.force)

            if status == "exists":
                skipped += 1
            elif success:
                completed += 1
            else:
                failed += 1
                errors.append(f"{subject} run-{run:02d}: {status}")

    # Print summary
    print("\n" + "=" * 70)
    print("BATCH SYNCHRONIZATION COMPLETE")
    print("=" * 70)
    print(f"Total tasks: {total_tasks}")
    print(f"Completed: {completed}")
    print(f"Skipped (already exist): {skipped}")
    print(f"Failed: {failed}")
    print()

    if errors:
        print(f"Errors ({len(errors)}):")
        for error in errors[:20]:  # Show first 20 errors
            print(f"  - {error}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
    else:
        print("No errors!")

    print()
    print(f"Sync params location: {PIPELINE_DIR / 'outputs/sync'}")
    print("=" * 70)

    # Return exit code
    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
