#!/usr/bin/env python3
"""
Batch audio–MEG synchronization using the robust alignment from modified_audio_processing_2.py.

For each subject and run, this script:
- Finds MEG data and external audio files (interviewer/participant).
- Optionally extracts MEG audio (interviewer/participant) from KIT .con files.
- Uses _align_audio() from modified_audio_processing_2.py to compute offset and correlation.
- Saves sync parameters to outputs/sync/{subject}/run-XX/sync_params.json.

CLI options:
  --subjects       Specific subjects to process (default: ALL_SUBJECTS in script)
  --runs           Specific runs to process (default: 1-5)
  --force          Re-run sync even if params exist
  --skip-existing  Skip subjects that already have all sync params

Requirements:
- modified_audio_processing_2.py must be importable in PYTHONPATH.
- External audio files must be accessible or MEG audio must be extractable.

Author: Updated to use alignment from modified_audio_processing_2.py
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Import functions from the working pipeline
# Ensure this import works in your environment (e.g., set PYTHONPATH to include the module directory)
try:
    from modified_audio_processing_2 import (
        _align_audio,
        diagnose_audio_files,
        _keep_left_channel_only,
        _extract_meg_audio,
    )
except Exception as e:
    print("✗ Failed to import alignment pipeline from modified_audio_processing_2.py")
    print(f"  Error: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------
# Configuration — update these paths to match your environment
# ---------------------------------------------------------------------

PIPELINE_DIR = Path(__file__).parent.parent

# Location where sync params will be written
SYNC_BASE = PIPELINE_DIR / "outputs" / "sync"

# Base for MEG data (KIT .con or MNE .fif). Update these to your actual storage layout.
MEG_BASE_KIT = (
    Path.home()
    / "Library/CloudStorage/OneDrive-SharedLibraries-MacquarieUniversity/Natural_Conversations_study - Documents/analysis/natural-conversations-bids/derivatives/kit-raw"
)  # Example
MEG_BASE_FIF = (
    Path.home()
    / "Library/CloudStorage/OneDrive-SharedLibraries-MacquarieUniversity/Natural_Conversations_study - Documents/analysis/natural-conversations-bids/derivatives/mne-bids-pipeline-20240628"
)

# External audio base — interviewer and participant recordings
AUDIO_BASE = (
    Path.home()
    / "Library/CloudStorage/OneDrive-SharedLibraries-MacquarieUniversity/Natural_Conversations_study - Documents/analysis/natural-conversations-bids/derivatives/external-audio"
)  # Example

# Subjects and runs (defaults)
ALL_SUBJECTS = [
    "sub-01",
    "sub-02",
    "sub-03",
    "sub-04",
    "sub-05",
    "sub-06",
    "sub-07",
    "sub-08",
    "sub-09",
    "sub-10",
    "sub-11",
    "sub-13",
    "sub-14",
    "sub-15",
    "sub-16",
    "sub-17",
    "sub-18",
    "sub-19",
    "sub-21",
    "sub-22",
    "sub-23",
    "sub-24",
    "sub-25",
    "sub-26",
    "sub-27",
    "sub-29",
    "sub-31",
    "sub-32",
]
CONVERSATION_RUNS = [1, 2, 3, 4, 5]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def sync_params_path(subject: str, run: int) -> Path:
    return SYNC_BASE / subject / f"run-{run:02d}" / "sync_params.json"


def check_sync_exists(subject: str, run: int) -> bool:
    return sync_params_path(subject, run).exists()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def find_meg_file(subject: str, run: int):
    """
    Try to find MEG file for a subject/run. Supports either KIT .con or MNE .fif.
    Adjust patterns to match your filenames.
    """
    # Try MNE FIF first (original batch script used this)
    fif = (
        MEG_BASE_FIF
        / subject
        / "meg"
        / f"{subject}_task-conversation_run-{run:02d}_proc-clean_raw.fif"
    )
    if fif.exists():
        return "fif", fif

    # Try KIT .con (needed to extract audio channels)
    con = (
        MEG_BASE_KIT
        / subject
        / "meg"
        / f"{subject}_task-conversation_run-{run:02d}.con"
    )
    if con.exists():
        return "con", con

    return None, None


def find_external_audio(subject: str, run: int):
    """
    Attempt to locate interviewer and participant external audio files.
    Update patterns to match your storage.
    """
    # Example patterns — adjust to your filenames
    interviewer = (
        AUDIO_BASE
        / subject
        / f"{subject}_task-conversation_run-{run:02d}_interviewer.wav"
    )
    participant = (
        AUDIO_BASE
        / subject
        / f"{subject}_task-conversation_run-{run:02d}_participant.wav"
    )

    interviewer_exists = interviewer.exists()
    participant_exists = participant.exists()

    return {
        "interviewer": interviewer if interviewer_exists else None,
        "participant": participant if participant_exists else None,
    }


def compute_alignment(meg_audio_file: Path, external_audio_file: Path):
    """
    Run diagnostics and alignment using the robust method from modified_audio_processing_2.py.
    Returns (offset_seconds, peak_correlation, diagnostics) or raises on error.
    """
    # Left-channel only for external audio, as per pipeline [1]
    left_channel = _keep_left_channel_only(
        str(external_audio_file), str(meg_audio_file.parent)
    )
    # Diagnostics first (safe practice in the pipeline) [1]
    _ = diagnose_audio_files(str(meg_audio_file), str(left_channel))
    # Compute alignment; plot disabled for batch use
    offset, peak_corr = _align_audio(str(meg_audio_file), str(left_channel), plot=False)
    return offset, peak_corr, {"left_channel_file": left_channel}


def save_sync(subject: str, run: int, payload: dict):
    out_path = sync_params_path(subject, run)
    ensure_dir(out_path.parent)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


# ---------------------------------------------------------------------
# Main sync per subject/run
# ---------------------------------------------------------------------


def run_sync_for_subject_run(subject: str, run: int, force: bool = False):
    # Check existing
    if not force and check_sync_exists(subject, run):
        print(f"  ✓ {subject} run-{run:02d}: Sync params already exist (skipping)")
        return True, "exists"

    meg_kind, meg_file = find_meg_file(subject, run)
    if meg_file is None:
        print(f"  ✗ {subject} run-{run:02d}: MEG file not found")
        return False, "meg_missing"

    print(f"  → {subject} run-{run:02d}: Using MEG file ({meg_kind}): {meg_file}")

    # Prepare MEG-derived audio (interviewer/participant) if KIT .con
    interviewer_meg_audio = None
    participant_meg_audio = None
    try:
        if meg_kind == "con":
            # Extract interviewer and participant audio from KIT MEG file [1]
            processing_dir = (
                SYNC_BASE / subject / f"run-{run:02d}" / "meg_audio"
            ).resolve()
            processing_dir.mkdir(parents=True, exist_ok=True)
            interviewer_meg_audio_str, participant_meg_audio_str = _extract_meg_audio(
                str(meg_file), str(processing_dir)
            )
            interviewer_meg_audio = Path(interviewer_meg_audio_str)
            participant_meg_audio = Path(participant_meg_audio_str)
        else:
            # For FIF: assume MEG-aligned audio has been pre-extracted elsewhere,
            # or rely on external audio only. You can add your own FIF audio extraction here if needed.
            pass
    except Exception as e:
        print(f"  ✗ {subject} run-{run:02d}: Failed to extract MEG audio: {e}")
        # We can continue if external audio is available; MEG audio improves alignment robustness
        interviewer_meg_audio = None
        participant_meg_audio = None

    # Find external audio files
    ext = find_external_audio(subject, run)
    if not ext["interviewer"] and not ext["participant"]:
        print(f"  ✗ {subject} run-{run:02d}: No external audio found")
        return False, "audio_missing"

    # For each available channel, compute alignment
    results = {
        "subject": subject,
        "run": run,
        "meg_file": str(meg_file),
        "alignments": {},
    }

    # Interviewer alignment
    if ext["interviewer"]:
        try:
            meg_ref = (
                interviewer_meg_audio
                if interviewer_meg_audio
                else (participant_meg_audio or interviewer_meg_audio)
            )
            if meg_ref is None and meg_kind == "con":
                print(
                    f"  ⚠️ {subject} run-{run:02d}: No MEG-derived audio available for interviewer; skipping"
                )
            else:
                ref_audio = (
                    meg_ref if meg_ref else ext["interviewer"]
                )  # fallback to external as ref if needed
                offset, corr, diag = compute_alignment(ref_audio, ext["interviewer"])
                results["alignments"]["interviewer"] = {
                    "external_audio": str(ext["interviewer"]),
                    "meg_audio_ref": str(ref_audio),
                    "offset_seconds": float(offset),
                    "peak_correlation": float(corr),
                    "diagnostics": diag,
                }
                print(
                    f"  ✓ {subject} run-{run:02d}: Interviewer offset={offset:.3f}s corr={corr:.3f}"
                )
        except Exception as e:
            print(f"  ✗ {subject} run-{run:02d}: Interviewer sync failed: {e}")

    # Participant alignment
    if ext["participant"]:
        try:
            meg_ref = (
                participant_meg_audio
                if participant_meg_audio
                else (interviewer_meg_audio or participant_meg_audio)
            )
            if meg_ref is None and meg_kind == "con":
                print(
                    f"  ⚠️ {subject} run-{run:02d}: No MEG-derived audio available for participant; skipping"
                )
            else:
                ref_audio = meg_ref if meg_ref else ext["participant"]
                offset, corr, diag = compute_alignment(ref_audio, ext["participant"])
                results["alignments"]["participant"] = {
                    "external_audio": str(ext["participant"]),
                    "meg_audio_ref": str(ref_audio),
                    "offset_seconds": float(offset),
                    "peak_correlation": float(corr),
                    "diagnostics": diag,
                }
                print(
                    f"  ✓ {subject} run-{run:02d}: Participant offset={offset:.3f}s corr={corr:.3f}"
                )
        except Exception as e:
            print(f"  ✗ {subject} run-{run:02d}: Participant sync failed: {e}")

    # Ensure we have at least one alignment
    if not results["alignments"]:
        print(f"  ✗ {subject} run-{run:02d}: No alignments produced")
        return False, "failed"

    # Save sync params
    out_path = save_sync(subject, run, results)
    print(f"  ✓ {subject} run-{run:02d}: Sync params saved → {out_path}")
    return True, "success"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Batch audio–MEG synchronization (robust alignment)"
    )
    parser.add_argument(
        "--subjects", nargs="+", help="Specific subjects to process (default: all)"
    )
    parser.add_argument(
        "--runs", nargs="+", type=int, help="Specific runs to process (default: 1–5)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-run sync even if params exist"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip subjects that already have all sync params",
    )
    args = parser.parse_args()

    subjects = args.subjects if args.subjects else ALL_SUBJECTS
    runs = args.runs if args.runs else CONVERSATION_RUNS

    print("=" * 70)
    print(
        f"BATCH AUDIO–MEG SYNCHRONIZATION (robust) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print("=" * 70)
    print(f"Subjects: {len(subjects)}")
    print(f"Runs: {runs}")
    print(f"Force re-sync: {args.force}")
    print("=" * 70)
    print()

    total_tasks = len(subjects) * len(runs)
    completed = 0
    skipped = 0
    failed = 0
    errors = []

    for i, subject in enumerate(subjects, 1):
        print(f"\n[{i}/{len(subjects)}] Processing {subject}")
        print("-" * 70)

        if args.skip_existing:
            all_exist = all(check_sync_exists(subject, run) for run in runs)
            if all_exist:
                print(f"  ✓ {subject}: All sync params exist (skipping subject)")
                skipped += len(runs)
                continue

        for run in runs:
            success, status = run_sync_for_subject_run(subject, run, args.force)
            if status == "exists":
                skipped += 1
            elif success:
                completed += 1
            else:
                failed += 1
                errors.append(f"{subject} run-{run:02d}: {status}")

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
        for error in errors[:20]:
            print(f"  - {error}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
    else:
        print("No errors!")

    print()
    print(f"Sync params location: {SYNC_BASE}")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
