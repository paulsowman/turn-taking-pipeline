#!/usr/bin/env python3
"""
Parallel Batch Multipredictor TRF Analysis Script
Processes all subjects with parallelization:
  1. Sync analysis for all conversation runs (parallel per subject)
  2. BADA localizer ROI selection (parallel: MEG & EEG)
  3. Multipredictor TRF (parallel: all 4 combinations per subject)
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
PIPELINE_DIR = Path("/Users/em18033/Library/CloudStorage/OneDrive-AUTUniversity/Projects/turn-taking-pipeline")
BIDS_DIR = Path("/Users/em18033/Library/CloudStorage/OneDrive-SharedLibraries-MacquarieUniversity/Natural_Conversations_study - Documents/analysis/natural-conversations-bids/derivatives/mne-bids-pipeline-20240628")
AUDIO_BASE_DIR = Path("/Users/em18033/Library/CloudStorage/OneDrive-AUTUniversity/Projects/Conversational_AI/Archive/data/audios")

# Subject mapping (subject ID to audio group)
SUBJECT_TO_GROUP = {
    'sub-01': 'G01', 'sub-02': 'G02', 'sub-03': 'G03', 'sub-04': 'G04',
    'sub-05': 'G05', 'sub-06': 'G06', 'sub-07': 'G07', 'sub-08': 'G08',
    'sub-09': 'G09', 'sub-10': 'G10', 'sub-11': 'G11', 'sub-13': 'G13',
    'sub-14': 'G14', 'sub-15': 'G15', 'sub-16': 'G16', 'sub-17': 'G17',
    'sub-18': 'G18', 'sub-19': 'G19', 'sub-21': 'G21', 'sub-22': 'G22',
    'sub-23': 'G23', 'sub-24': 'G24', 'sub-25': 'G25', 'sub-26': 'G26',
    'sub-27': 'G27', 'sub-29': 'G29', 'sub-31': 'G31', 'sub-32': 'G32',
}


def run_command(cmd, description, log_file=None):
    """Run a command and return success status"""
    print(f"  {description}...")
    try:
        if log_file:
            with open(log_file, 'w') as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode == 0:
            print(f"  ✓ {description} complete")
            return True
        else:
            print(f"  ✗ {description} failed (exit code {result.returncode})")
            return False
    except Exception as e:
        print(f"  ✗ {description} failed: {e}")
        return False


def process_sync_analysis(subject, subject_meg_dir, log_dir, skip_sync):
    """Process sync analysis for all runs of a subject"""
    if skip_sync:
        print("  Step 1: Synchronization Analysis - SKIPPED")
        return True

    print(f"  Step 1: Synchronization Analysis (runs 1-5) - {datetime.now().strftime('%H:%M:%S')}")

    sync_jobs = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        for run in [1, 2, 3, 4, 5]:
            run_padded = f"{run:02d}"
            meg_file = subject_meg_dir / f"{subject}_task-conversation_run-{run_padded}_proc-clean_raw.fif"

            if not meg_file.exists():
                print(f"    WARNING: MEG file not found for run {run}")
                continue

            sync_params = PIPELINE_DIR / f"outputs/sync/{subject}/run-{run_padded}/sync_params.json"
            if sync_params.exists():
                print(f"    Run {run}: Sync params already exist, skipping")
                continue

            log_file = log_dir / f"{subject}_sync_run{run}.log"
            cmd = [
                'python', 'scripts/run_audio_meg_sync.py',
                '--subject', subject,
                '--run', str(run),
                '--meg-file', str(meg_file)
            ]

            future = executor.submit(run_command, cmd, f"Run {run} sync", log_file)
            sync_jobs.append((run, future))

    # Wait for all sync jobs
    if sync_jobs:
        print(f"    Waiting for {len(sync_jobs)} sync jobs...")
        for run, future in sync_jobs:
            future.result()
        print(f"    All sync jobs complete - {datetime.now().strftime('%H:%M:%S')}")

    return True


def process_localizer(subject, subject_meg_dir, log_dir, skip_localizer):
    """Process BADA localizer ROI selection"""
    if skip_localizer:
        print("  BADA Localizer: SKIPPED")
        return True

    print(f"  Step 2: BADA Localizer ROI Selection - {datetime.now().strftime('%H:%M:%S')}")

    localizer_file = subject_meg_dir / f"{subject}_task-conversation_run-06_proc-clean_raw.fif"
    if not localizer_file.exists():
        print(f"    WARNING: Localizer file not found: {localizer_file}")
        return False

    roi_mag = PIPELINE_DIR / f"outputs/bada_localizer/{subject}/roi_sensors_mag.json"
    roi_eeg = PIPELINE_DIR / f"outputs/bada_localizer/{subject}/roi_sensors_eeg.json"

    if roi_mag.exists() and roi_eeg.exists():
        print("    ROI files already exist, skipping")
        return True

    # Run MEG and EEG localizer in parallel
    localizer_jobs = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        if not roi_mag.exists():
            log_file = log_dir / f"{subject}_localizer_mag.log"
            cmd = [
                'python', 'scripts/bada_localizer_analysis.py',
                '--subject', subject,
                '--meg-file', str(localizer_file),
                '--sensor-type', 'mag'
            ]
            future = executor.submit(run_command, cmd, "MEG localizer", log_file)
            localizer_jobs.append(future)

        if not roi_eeg.exists():
            log_file = log_dir / f"{subject}_localizer_eeg.log"
            cmd = [
                'python', 'scripts/bada_localizer_analysis.py',
                '--subject', subject,
                '--meg-file', str(localizer_file),
                '--sensor-type', 'eeg'
            ]
            future = executor.submit(run_command, cmd, "EEG localizer", log_file)
            localizer_jobs.append(future)

    # Wait for localizer jobs
    if localizer_jobs:
        print(f"    Waiting for {len(localizer_jobs)} localizer jobs...")
        results = [future.result() for future in localizer_jobs]
        print(f"    Localizer complete - {datetime.now().strftime('%H:%M:%S')}")
        return all(results)

    return True


def process_trf_analysis(subject, subject_meg_dir, log_dir, skip_trf):
    """Process multipredictor TRF analysis"""
    if skip_trf:
        print("  Multipredictor TRF: SKIPPED")
        return True

    print(f"  Step 3: Multipredictor TRF Analysis - {datetime.now().strftime('%H:%M:%S')}")

    roi_mag = PIPELINE_DIR / f"outputs/bada_localizer/{subject}/roi_sensors_mag.json"
    roi_eeg = PIPELINE_DIR / f"outputs/bada_localizer/{subject}/roi_sensors_eeg.json"

    if not roi_mag.exists() or not roi_eeg.exists():
        print(f"    WARNING: ROI files not found")
        return False

    # Define MEG files
    meg_files = {
        1: subject_meg_dir / f"{subject}_task-conversation_run-01_proc-clean_raw.fif",
        2: subject_meg_dir / f"{subject}_task-conversation_run-02_proc-clean_raw.fif",
        3: subject_meg_dir / f"{subject}_task-conversation_run-03_proc-clean_raw.fif",
        4: subject_meg_dir / f"{subject}_task-conversation_run-04_proc-clean_raw.fif",
        5: subject_meg_dir / f"{subject}_task-conversation_run-05_proc-clean_raw.fif",
    }

    # Check files exist
    for run, filepath in meg_files.items():
        if not filepath.exists():
            print(f"    WARNING: MEG file not found for run {run}")
            return False

    # Run 2 TRF combinations in parallel (each can use 3-5 GB RAM)
    # With 32GB RAM, running 2 at a time is safe
    trf_jobs = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Runs 1+3+5 - MEG
        output_dir = PIPELINE_DIR / f"outputs/trf_multipredictor/{subject}/runs-1_3_5/mag"
        if not (output_dir / "summary.json").exists():
            log_file = log_dir / f"{subject}_trf_135_mag.log"
            cmd = [
                'python', 'scripts/multipredictor_trf_combined.py',
                '--subject', subject,
                '--runs', '1', '3', '5',
                '--meg-files', str(meg_files[1]), str(meg_files[3]), str(meg_files[5]),
                '--sensor-type', 'mag',
                '--roi-sensors', str(roi_mag),
                '--listener-only'
            ]
            future = executor.submit(run_command, cmd, "Runs 1+3+5 MEG", log_file)
            trf_jobs.append(('1+3+5 MEG', future))
        else:
            print("    Runs 1+3+5 MEG already complete")

        # Runs 1+3+5 - EEG
        output_dir = PIPELINE_DIR / f"outputs/trf_multipredictor/{subject}/runs-1_3_5/eeg"
        if not (output_dir / "summary.json").exists():
            log_file = log_dir / f"{subject}_trf_135_eeg.log"
            cmd = [
                'python', 'scripts/multipredictor_trf_combined.py',
                '--subject', subject,
                '--runs', '1', '3', '5',
                '--meg-files', str(meg_files[1]), str(meg_files[3]), str(meg_files[5]),
                '--sensor-type', 'eeg',
                '--roi-sensors', str(roi_eeg)
            ]
            future = executor.submit(run_command, cmd, "Runs 1+3+5 EEG", log_file)
            trf_jobs.append(('1+3+5 EEG', future))
        else:
            print("    Runs 1+3+5 EEG already complete")

        # Runs 2+4 - MEG
        output_dir = PIPELINE_DIR / f"outputs/trf_multipredictor/{subject}/runs-2_4/mag"
        if not (output_dir / "summary.json").exists():
            log_file = log_dir / f"{subject}_trf_24_mag.log"
            cmd = [
                'python', 'scripts/multipredictor_trf_combined.py',
                '--subject', subject,
                '--runs', '2', '4',
                '--meg-files', str(meg_files[2]), str(meg_files[4]),
                '--sensor-type', 'mag',
                '--roi-sensors', str(roi_mag)
            ]
            future = executor.submit(run_command, cmd, "Runs 2+4 MEG", log_file)
            trf_jobs.append(('2+4 MEG', future))
        else:
            print("    Runs 2+4 MEG already complete")

        # Runs 2+4 - EEG
        output_dir = PIPELINE_DIR / f"outputs/trf_multipredictor/{subject}/runs-2_4/eeg"
        if not (output_dir / "summary.json").exists():
            log_file = log_dir / f"{subject}_trf_24_eeg.log"
            cmd = [
                'python', 'scripts/multipredictor_trf_combined.py',
                '--subject', subject,
                '--runs', '2', '4',
                '--meg-files', str(meg_files[2]), str(meg_files[4]),
                '--sensor-type', 'eeg',
                '--roi-sensors', str(roi_eeg)
            ]
            future = executor.submit(run_command, cmd, "Runs 2+4 EEG", log_file)
            trf_jobs.append(('2+4 EEG', future))
        else:
            print("    Runs 2+4 EEG already complete")

    # Wait for all TRF jobs
    if trf_jobs:
        print(f"    Waiting for {len(trf_jobs)} TRF jobs...")
        for name, future in trf_jobs:
            result = future.result()
            if not result:
                print(f"      WARNING: {name} failed")
        print(f"    All TRF jobs complete - {datetime.now().strftime('%H:%M:%S')}")

    return True


def process_subject(subject, log_dir, skip_sync, skip_localizer, skip_trf):
    """Process a single subject"""
    print(f"\n{'='*70}")
    print(f"PROCESSING {subject} - {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}")

    # Get audio group
    audio_group = SUBJECT_TO_GROUP.get(subject)
    if not audio_group:
        print(f"ERROR: No audio group mapping for {subject}")
        return False

    subject_meg_dir = BIDS_DIR / subject / "meg"
    audio_dir = AUDIO_BASE_DIR / audio_group

    # Check directories exist
    if not subject_meg_dir.exists():
        print(f"WARNING: MEG directory not found: {subject_meg_dir}")
        return False

    if not audio_dir.exists():
        print(f"WARNING: Audio directory not found: {audio_dir}")
        return False

    # Process each step
    process_sync_analysis(subject, subject_meg_dir, log_dir, skip_sync)
    process_localizer(subject, subject_meg_dir, log_dir, skip_localizer)
    process_trf_analysis(subject, subject_meg_dir, log_dir, skip_trf)

    print(f"\n{'='*70}")
    print(f"COMPLETED {subject} - {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Parallel batch multipredictor TRF analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --subjects sub-01 sub-02 --max-parallel 8
  %(prog)s --skip-sync --max-parallel 6
  %(prog)s  # Process all subjects with default settings
        """
    )

    parser.add_argument('--subjects', nargs='+', default=None,
                       help='Process only specified subjects (e.g., sub-01 sub-02)')
    parser.add_argument('--skip-sync', action='store_true',
                       help='Skip synchronization analysis')
    parser.add_argument('--skip-localizer', action='store_true',
                       help='Skip BADA localizer ROI selection')
    parser.add_argument('--skip-trf', action='store_true',
                       help='Skip multipredictor TRF analysis')
    parser.add_argument('--max-parallel', type=int, default=2,
                       help='Maximum parallel subjects (default: 2, safe for 32GB RAM)')

    args = parser.parse_args()

    # Determine subjects to process
    if args.subjects:
        subjects = args.subjects
    else:
        subjects = sorted(SUBJECT_TO_GROUP.keys())

    print("="*70)
    print("PARALLEL BATCH MULTIPREDICTOR TRF ANALYSIS")
    print("="*70)
    print(f"\nSubjects to process: {', '.join(subjects)}")
    print(f"Skip sync: {args.skip_sync}")
    print(f"Skip localizer: {args.skip_localizer}")
    print(f"Skip TRF: {args.skip_trf}")
    print(f"Max parallel subjects: {args.max_parallel}")
    print()

    # Create log directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = PIPELINE_DIR / f"logs/batch_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logs will be saved to: {log_dir}\n")

    # Change to pipeline directory
    import os
    os.chdir(PIPELINE_DIR)

    # Process subjects in parallel
    print("="*70)
    print("STARTING PARALLEL PROCESSING")
    print("="*70)

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {
            executor.submit(process_subject, subject, log_dir,
                          args.skip_sync, args.skip_localizer, args.skip_trf): subject
            for subject in subjects
        }

        for future in as_completed(futures):
            subject = futures[future]
            try:
                result = future.result()
                if result:
                    print(f"\n✓ {subject} completed successfully")
                else:
                    print(f"\n✗ {subject} completed with warnings")
            except Exception as e:
                print(f"\n✗ {subject} failed with exception: {e}")

    elapsed = time.time() - start_time

    # Final summary
    print("\n" + "="*70)
    print(f"BATCH PROCESSING COMPLETE - {datetime.now().strftime('%H:%M:%S')}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print("="*70)
    print("\nSummary of processed subjects:")

    for subject in subjects:
        summary_135_mag = PIPELINE_DIR / f"outputs/trf_multipredictor/{subject}/runs-1_3_5/mag/summary.json"
        summary_135_eeg = PIPELINE_DIR / f"outputs/trf_multipredictor/{subject}/runs-1_3_5/eeg/summary.json"
        summary_24_mag = PIPELINE_DIR / f"outputs/trf_multipredictor/{subject}/runs-2_4/mag/summary.json"
        summary_24_eeg = PIPELINE_DIR / f"outputs/trf_multipredictor/{subject}/runs-2_4/eeg/summary.json"

        status = f"  {subject}:"
        status += " 1+3+5-MAG✓" if summary_135_mag.exists() else " 1+3+5-MAG✗"
        status += " 1+3+5-EEG✓" if summary_135_eeg.exists() else " 1+3+5-EEG✗"
        status += " 2+4-MAG✓" if summary_24_mag.exists() else " 2+4-MAG✗"
        status += " 2+4-EEG✓" if summary_24_eeg.exists() else " 2+4-EEG✗"

        print(status)

    print(f"\nLogs saved to: {log_dir}")
    print("All done!")


if __name__ == '__main__':
    main()
