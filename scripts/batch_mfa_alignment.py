#!/usr/bin/env python
"""
Batch MFA alignment for all subjects and runs.

This script runs Montreal Forced Aligner on Whisper transcripts for all subjects,
with automatic quality filtering to remove hallucinations and NaN artifacts.

Usage:
    python scripts/batch_mfa_alignment.py
    python scripts/batch_mfa_alignment.py --subjects sub-01 sub-02 sub-03
    python scripts/batch_mfa_alignment.py --runs 1 3 5
    python scripts/batch_mfa_alignment.py --parallel 4
"""

import subprocess
import sys
from pathlib import Path
import argparse
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import yaml

# Project paths
PIPELINE_DIR = Path(__file__).parent.parent
VENV_PYTHON = PIPELINE_DIR / "venv" / "bin" / "python"
MFA_SCRIPT = PIPELINE_DIR / "scripts" / "run_mfa_alignment.py"

# All available subjects (from your batch_all_subjects.py)
ALL_SUBJECTS = [
    'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05',
    'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10',
    'sub-11', 'sub-13', 'sub-14', 'sub-15', 'sub-16',
    'sub-17', 'sub-18', 'sub-19', 'sub-21', 'sub-22',
    'sub-23', 'sub-24', 'sub-25', 'sub-26', 'sub-27',
    'sub-29', 'sub-31', 'sub-32'
]

# Conversation runs (odd blocks)
CONVERSATION_RUNS = [1, 3, 5]


def load_config():
    """Load configuration to check which runs to process."""
    config_path = PIPELINE_DIR / 'config' / 'config.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)


def check_transcripts_exist(subject, run):
    """Check if transcripts exist for a subject/run."""
    run_str = f"run-{run:02d}"
    features_dir = PIPELINE_DIR / 'outputs' / 'features' / subject / run_str

    transcript_int = features_dir / "transcript_interviewer.csv"
    transcript_part = features_dir / "transcript_participant.csv"

    return transcript_int.exists() and transcript_part.exists()


def run_mfa_alignment(subject, run, log_file, extra_args=None):
    """
    Run MFA alignment for one subject/run.

    Parameters
    ----------
    subject : str
        Subject ID (e.g., 'sub-01').
    run : int
        Run number.
    log_file : Path
        Path to log file.
    extra_args : list, optional
        Additional arguments to pass to run_mfa_alignment.py.

    Returns
    -------
    success : bool
        True if alignment succeeded.
    """
    # Build command
    # Note: We need conda to run MFA, so we use conda run
    cmd = [
        'conda', 'run', '-n', 'mfa',
        'python', str(MFA_SCRIPT),
        '--subject', subject,
        '--run', str(run),
    ]

    if extra_args:
        cmd.extend(extra_args)

    print(f"  → {subject} run-{run}: Running MFA alignment...")

    with open(log_file, 'w') as f:
        f.write(f"Command: {' '.join(str(c) for c in cmd)}\n")
        f.write(f"Started: {datetime.now()}\n\n")

        try:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=PIPELINE_DIR,
                timeout=1200,  # 20 minute timeout per subject/run
            )

            success = result.returncode == 0
            f.write(f"\n\nCompleted: {datetime.now()}\n")
            f.write(f"Return code: {result.returncode}\n")

            if success:
                print(f"  ✓ {subject} run-{run}: MFA alignment complete")
            else:
                print(f"  ✗ {subject} run-{run}: MFA alignment failed (see log)")

            return success

        except subprocess.TimeoutExpired:
            f.write(f"\n\nERROR: Timeout after 20 minutes\n")
            print(f"  ✗ {subject} run-{run}: Timeout")
            return False
        except Exception as e:
            f.write(f"\n\nERROR: {e}\n")
            print(f"  ✗ {subject} run-{run}: Error - {e}")
            return False


def process_subject_run(args_tuple):
    """
    Process one subject/run (for parallel execution).

    Parameters
    ----------
    args_tuple : tuple
        (subject, run, log_dir, extra_args)

    Returns
    -------
    result : dict
        Processing result.
    """
    subject, run, log_dir, extra_args = args_tuple

    # Check if transcripts exist
    if not check_transcripts_exist(subject, run):
        return {
            'subject': subject,
            'run': run,
            'status': 'skipped',
            'reason': 'transcripts not found'
        }

    # Check if already processed
    run_str = f"run-{run:02d}"
    textgrid_int = PIPELINE_DIR / 'outputs' / 'mfa' / subject / run_str / 'interviewer' / 'textgrids'
    textgrid_part = PIPELINE_DIR / 'outputs' / 'mfa' / subject / run_str / 'participant' / 'textgrids'

    if textgrid_int.exists() and textgrid_part.exists():
        int_tgs = list(textgrid_int.glob('*.TextGrid'))
        part_tgs = list(textgrid_part.glob('*.TextGrid'))
        if int_tgs and part_tgs:
            return {
                'subject': subject,
                'run': run,
                'status': 'skipped',
                'reason': 'already aligned'
            }

    # Run MFA alignment
    log_file = log_dir / f"mfa_{subject}_run{run:02d}.log"
    success = run_mfa_alignment(subject, run, log_file, extra_args)

    return {
        'subject': subject,
        'run': run,
        'status': 'success' if success else 'failed',
        'reason': None
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch MFA alignment for all subjects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all subjects, all conversation runs
  python scripts/batch_mfa_alignment.py

  # Process specific subjects
  python scripts/batch_mfa_alignment.py --subjects sub-01 sub-02 sub-03

  # Process specific runs
  python scripts/batch_mfa_alignment.py --runs 1 3

  # Run in parallel
  python scripts/batch_mfa_alignment.py --parallel 4

  # Force reprocessing (overwrite existing)
  python scripts/batch_mfa_alignment.py --force

  # With OOV removal (more aggressive filtering)
  python scripts/batch_mfa_alignment.py --remove-oov
        """
    )

    parser.add_argument('--subjects', nargs='+', default=None,
                        help='Subjects to process (default: all)')
    parser.add_argument('--runs', nargs='+', type=int, default=None,
                        help='Runs to process (default: conversation runs 1,3,5)')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel jobs (default: 1 = sequential)')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocessing even if outputs exist')
    parser.add_argument('--remove-oov', action='store_true',
                        help='Remove out-of-vocabulary words (passed to MFA script)')
    parser.add_argument('--no-speech-threshold', type=float, default=0.5,
                        help='Quality filter threshold (default: 0.5)')

    args = parser.parse_args()

    # Determine subjects and runs
    subjects = args.subjects if args.subjects else ALL_SUBJECTS
    runs = args.runs if args.runs else CONVERSATION_RUNS

    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = PIPELINE_DIR / 'logs' / f'batch_mfa_{timestamp}'
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build extra args to pass to MFA script
    extra_args = []
    if args.remove_oov:
        extra_args.append('--remove-oov')
    if args.no_speech_threshold != 0.5:
        extra_args.extend(['--no-speech-threshold', str(args.no_speech_threshold)])

    # Print configuration
    print("\n" + "="*70)
    print("BATCH MFA ALIGNMENT")
    print("="*70)
    print(f"Subjects: {len(subjects)}")
    print(f"Runs: {runs}")
    print(f"Parallel jobs: {args.parallel}")
    print(f"Force reprocessing: {args.force}")
    print(f"Extra args: {' '.join(extra_args) if extra_args else 'none'}")
    print(f"Log directory: {log_dir}")
    print("="*70)

    # Build task list
    tasks = []
    for subject in subjects:
        for run in runs:
            tasks.append((subject, run, log_dir, extra_args))

    print(f"\nTotal tasks: {len(tasks)}")
    print(f"Starting at: {datetime.now()}\n")

    # Process tasks
    results = []

    if args.parallel > 1:
        # Parallel processing
        print(f"Running {args.parallel} jobs in parallel...\n")
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(process_subject_run, task): task
                      for task in tasks}

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
    else:
        # Sequential processing
        print("Running sequentially...\n")
        for task in tasks:
            result = process_subject_run(task)
            results.append(result)

    # Summary
    print("\n" + "="*70)
    print("BATCH MFA ALIGNMENT SUMMARY")
    print("="*70)
    print(f"\nTotal: {len(results)}")

    success_count = sum(1 for r in results if r['status'] == 'success')
    skipped_count = sum(1 for r in results if r['status'] == 'skipped')
    failed_count = sum(1 for r in results if r['status'] == 'failed')

    print(f"  ✓ Success: {success_count}")
    print(f"  - Skipped: {skipped_count}")
    print(f"  ✗ Failed: {failed_count}")

    # Show skipped
    if skipped_count > 0:
        print(f"\nSkipped ({skipped_count}):")
        for r in results:
            if r['status'] == 'skipped':
                print(f"  - {r['subject']} run-{r['run']}: {r['reason']}")

    # Show failed
    if failed_count > 0:
        print(f"\nFailed ({failed_count}):")
        for r in results:
            if r['status'] == 'failed':
                print(f"  - {r['subject']} run-{r['run']}")

    print(f"\nLogs saved to: {log_dir}")
    print("="*70)

    return 0 if failed_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
