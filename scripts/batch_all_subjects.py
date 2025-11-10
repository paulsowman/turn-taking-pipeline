#!/usr/bin/env python
"""
Batch process all subjects for multi-predictor TRF analysis.
This script:
1. Runs BADA localizer for MAG and EEG sensors
2. Runs multi-predictor TRF analysis (runs 1+3+5 and 2+4) for both sensor types
"""

import subprocess
import sys
from pathlib import Path
import json
from datetime import datetime
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Project paths
PIPELINE_DIR = Path("/Users/em18033/Library/CloudStorage/OneDrive-AUTUniversity/Projects/turn-taking-pipeline")
BIDS_DIR = Path("/Users/em18033/Library/CloudStorage/OneDrive-SharedLibraries-MacquarieUniversity/Natural_Conversations_study - Documents/analysis/natural-conversations-bids/derivatives/mne-bids-pipeline-20240628")
VENV_PYTHON = PIPELINE_DIR / "venv" / "bin" / "python"

# All available subjects (excluding those that don't exist)
ALL_SUBJECTS = [
    'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05',
    'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10',
    'sub-11', 'sub-13', 'sub-14', 'sub-15', 'sub-16',
    'sub-17', 'sub-18', 'sub-19', 'sub-21', 'sub-22',
    'sub-23', 'sub-24', 'sub-25', 'sub-26', 'sub-27',
    'sub-29', 'sub-31', 'sub-32'
]


def run_command(cmd, log_file, description):
    """Run a command and log output."""
    print(f"  {description}...")

    with open(log_file, 'w') as f:
        f.write(f"Command: {' '.join(str(c) for c in cmd)}\n")
        f.write(f"Started: {datetime.now()}\n\n")

        try:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=PIPELINE_DIR
            )

            success = result.returncode == 0
            f.write(f"\n\nCompleted: {datetime.now()}\n")
            f.write(f"Return code: {result.returncode}\n")

            return success

        except Exception as e:
            f.write(f"\n\nERROR: {e}\n")
            return False


def check_file_exists(file_path):
    """Check if a file exists."""
    return Path(file_path).exists()


def process_subject(subject, log_dir, skip_localizer=False, skip_trf=False):
    """Process a single subject."""
    print(f"\n{'='*70}")
    print(f"PROCESSING {subject} - {datetime.now()}")
    print(f"{'='*70}")

    subject_meg_dir = BIDS_DIR / subject / "meg"

    # Check if subject directory exists
    if not subject_meg_dir.exists():
        print(f"  WARNING: MEG directory not found for {subject}")
        return False

    results = {
        'subject': subject,
        'localizer_mag': False,
        'localizer_eeg': False,
        'trf_135_mag': False,
        'trf_135_eeg': False,
        'trf_24_mag': False,
        'trf_24_eeg': False,
    }

    # Step 1: BADA Localizer
    if not skip_localizer:
        print("\nStep 1: BADA Localizer ROI Selection")

        localizer_file = subject_meg_dir / f"{subject}_task-conversation_run-06_proc-clean_raw.fif"
        roi_mag = PIPELINE_DIR / f"outputs/bada_localizer/{subject}/roi_sensors_mag.json"
        roi_eeg = PIPELINE_DIR / f"outputs/bada_localizer/{subject}/roi_sensors_eeg.json"

        if not localizer_file.exists():
            print(f"  WARNING: Localizer file not found: {localizer_file}")
        else:
            # MAG localizer + ROI selection
            if not roi_mag.exists():
                # Run ROI selection (automatically runs localizer analysis first via dependencies)
                cmd = [
                    str(VENV_PYTHON),
                    str(PIPELINE_DIR / "scripts/bada_localizer_roi.py"),
                    "--subject", subject,
                    "--meg-file", str(localizer_file),
                    "--sensor-type", "mag",
                    "--n-sensors-per-hemi", "10"
                ]
                results['localizer_mag'] = run_command(
                    cmd,
                    log_dir / f"{subject}_localizer_mag.log",
                    "Running MAG localizer + ROI selection"
                )
            else:
                print("  MAG ROI already exists, skipping")
                results['localizer_mag'] = True

            # EEG localizer + ROI selection
            if not roi_eeg.exists():
                cmd = [
                    str(VENV_PYTHON),
                    str(PIPELINE_DIR / "scripts/bada_localizer_roi.py"),
                    "--subject", subject,
                    "--meg-file", str(localizer_file),
                    "--sensor-type", "eeg",
                    "--n-sensors-per-hemi", "5"
                ]
                results['localizer_eeg'] = run_command(
                    cmd,
                    log_dir / f"{subject}_localizer_eeg.log",
                    "Running EEG localizer + ROI selection"
                )
            else:
                print("  EEG ROI already exists, skipping")
                results['localizer_eeg'] = True

    # Step 2: Multi-predictor TRF
    if not skip_trf:
        print("\nStep 2: Multi-predictor TRF Analysis")

        roi_mag = PIPELINE_DIR / f"outputs/bada_localizer/{subject}/roi_sensors_mag.json"
        roi_eeg = PIPELINE_DIR / f"outputs/bada_localizer/{subject}/roi_sensors_eeg.json"

        if not roi_mag.exists() or not roi_eeg.exists():
            print("  WARNING: ROI files not found, skipping TRF")
        else:
            # Define MEG files
            meg_run_01 = subject_meg_dir / f"{subject}_task-conversation_run-01_proc-clean_raw.fif"
            meg_run_02 = subject_meg_dir / f"{subject}_task-conversation_run-02_proc-clean_raw.fif"
            meg_run_03 = subject_meg_dir / f"{subject}_task-conversation_run-03_proc-clean_raw.fif"
            meg_run_04 = subject_meg_dir / f"{subject}_task-conversation_run-04_proc-clean_raw.fif"
            meg_run_05 = subject_meg_dir / f"{subject}_task-conversation_run-05_proc-clean_raw.fif"

            # Check files exist
            files_exist = all([
                meg_run_01.exists(),
                meg_run_02.exists(),
                meg_run_03.exists(),
                meg_run_04.exists(),
                meg_run_05.exists()
            ])

            if not files_exist:
                print("  WARNING: Not all MEG files found")
            else:
                # Runs 1+3+5 - MAG
                summary_135_mag = PIPELINE_DIR / f"outputs/trf_multipredictor/{subject}/runs-1_3_5/mag/summary.json"
                if not summary_135_mag.exists():
                    cmd = [
                        str(VENV_PYTHON),
                        str(PIPELINE_DIR / "scripts/multipredictor_trf_combined.py"),
                        "--subject", subject,
                        "--runs", "1", "3", "5",
                        "--meg-files", str(meg_run_01), str(meg_run_03), str(meg_run_05),
                        "--sensor-type", "mag"
                    ]
                    # Add ROI sensors if available
                    if roi_mag.exists():
                        cmd.extend(["--roi-sensors", str(roi_mag)])
                    results['trf_135_mag'] = run_command(
                        cmd,
                        log_dir / f"{subject}_trf_135_mag.log",
                        "Running TRF runs 1+3+5 MAG"
                    )
                else:
                    print("  TRF runs 1+3+5 MAG already exists, skipping")
                    results['trf_135_mag'] = True

                # Runs 1+3+5 - EEG
                summary_135_eeg = PIPELINE_DIR / f"outputs/trf_multipredictor/{subject}/runs-1_3_5/eeg/summary.json"
                if not summary_135_eeg.exists():
                    cmd = [
                        str(VENV_PYTHON),
                        str(PIPELINE_DIR / "scripts/multipredictor_trf_combined.py"),
                        "--subject", subject,
                        "--runs", "1", "3", "5",
                        "--meg-files", str(meg_run_01), str(meg_run_03), str(meg_run_05),
                        "--sensor-type", "eeg",
                        "--roi-sensors", str(roi_eeg)
                    ]
                    results['trf_135_eeg'] = run_command(
                        cmd,
                        log_dir / f"{subject}_trf_135_eeg.log",
                        "Running TRF runs 1+3+5 EEG"
                    )
                else:
                    print("  TRF runs 1+3+5 EEG already exists, skipping")
                    results['trf_135_eeg'] = True

                # Runs 2+4 - MAG
                summary_24_mag = PIPELINE_DIR / f"outputs/trf_multipredictor/{subject}/runs-2_4/mag/summary.json"
                if not summary_24_mag.exists():
                    cmd = [
                        str(VENV_PYTHON),
                        str(PIPELINE_DIR / "scripts/multipredictor_trf_combined.py"),
                        "--subject", subject,
                        "--runs", "2", "4",
                        "--meg-files", str(meg_run_02), str(meg_run_04),
                        "--sensor-type", "mag",
                        "--roi-sensors", str(roi_mag)
                    ]
                    results['trf_24_mag'] = run_command(
                        cmd,
                        log_dir / f"{subject}_trf_24_mag.log",
                        "Running TRF runs 2+4 MAG"
                    )
                else:
                    print("  TRF runs 2+4 MAG already exists, skipping")
                    results['trf_24_mag'] = True

                # Runs 2+4 - EEG
                summary_24_eeg = PIPELINE_DIR / f"outputs/trf_multipredictor/{subject}/runs-2_4/eeg/summary.json"
                if not summary_24_eeg.exists():
                    cmd = [
                        str(VENV_PYTHON),
                        str(PIPELINE_DIR / "scripts/multipredictor_trf_combined.py"),
                        "--subject", subject,
                        "--runs", "2", "4",
                        "--meg-files", str(meg_run_02), str(meg_run_04),
                        "--sensor-type", "eeg",
                        "--roi-sensors", str(roi_eeg)
                    ]
                    results['trf_24_eeg'] = run_command(
                        cmd,
                        log_dir / f"{subject}_trf_24_eeg.log",
                        "Running TRF runs 2+4 EEG"
                    )
                else:
                    print("  TRF runs 2+4 EEG already exists, skipping")
                    results['trf_24_eeg'] = True

    print(f"\n{'='*70}")
    print(f"COMPLETED {subject} - {datetime.now()}")
    print(f"{'='*70}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description='Batch process all subjects for multi-predictor TRF analysis')
    parser.add_argument('--subjects', nargs='+', default=None, help='Specific subjects to process')
    parser.add_argument('--skip-localizer', action='store_true', help='Skip BADA localizer')
    parser.add_argument('--skip-trf', action='store_true', help='Skip TRF analysis')
    parser.add_argument('--max-parallel', type=int, default=3, help='Maximum parallel jobs')

    args = parser.parse_args()

    # Determine which subjects to process
    subjects = args.subjects if args.subjects else ALL_SUBJECTS

    print("="*70)
    print("BATCH MULTI-PREDICTOR TRF ANALYSIS")
    print("="*70)
    print(f"\nSubjects to process: {', '.join(subjects)}")
    print(f"Skip localizer: {args.skip_localizer}")
    print(f"Skip TRF: {args.skip_trf}")
    print(f"Max parallel jobs: {args.max_parallel}")
    print()

    # Create log directory
    log_dir = PIPELINE_DIR / "logs" / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logs will be saved to: {log_dir}\n")

    # Process subjects sequentially (simpler and more reliable)
    all_results = []

    for subject in subjects:
        try:
            result = process_subject(subject, log_dir, args.skip_localizer, args.skip_trf)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR processing {subject}: {e}")
            all_results.append({
                'subject': subject,
                'error': str(e)
            })

        # Small delay between subjects
        time.sleep(2)

    # Print summary
    print("\n" + "="*70)
    print("BATCH PROCESSING COMPLETE")
    print("="*70)
    print("\nSummary:")

    for result in all_results:
        if 'error' in result:
            print(f"  {result['subject']}: ERROR - {result['error']}")
        else:
            subject = result['subject']
            loc_mag = "✓" if result.get('localizer_mag') else "✗"
            loc_eeg = "✓" if result.get('localizer_eeg') else "✗"
            trf_135_mag = "✓" if result.get('trf_135_mag') else "✗"
            trf_135_eeg = "✓" if result.get('trf_135_eeg') else "✗"
            trf_24_mag = "✓" if result.get('trf_24_mag') else "✗"
            trf_24_eeg = "✓" if result.get('trf_24_eeg') else "✗"

            print(f"  {subject}: LOC[MAG:{loc_mag} EEG:{loc_eeg}] TRF[135-MAG:{trf_135_mag} 135-EEG:{trf_135_eeg} 24-MAG:{trf_24_mag} 24-EEG:{trf_24_eeg}]")

    # Save results to JSON
    results_file = log_dir / "batch_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"Logs saved to: {log_dir}")
    print("\nAll done!")


if __name__ == "__main__":
    main()
