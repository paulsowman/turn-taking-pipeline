#!/usr/bin/env python3
"""
Batch audio–MEG synchronization using robust alignment from modified_audio_processing_2.py,
with direct support for MNE .fif MEG files.

For each subject/run:
- Loads the MEG .fif file.
- Extracts the MEG auxiliary audio channel (e.g., 'MISC 007') and writes it to WAV.
- Aligns external interviewer/participant audio to this MEG audio using _align_audio().
- Saves sync parameters to outputs/sync/{subject}/run-XX/sync_params.json.

CLI:
  --subjects, --runs, --force, --skip-existing

Author: FIF-based alignment using modified_audio_processing_2.py
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import mne
import soundfile
import numpy as np

# Import robust alignment utilities
try:
    from modified_audio_processing_2 import (
        _align_audio,
        diagnose_audio_files,
        _keep_left_channel_only,
    )
except Exception as e:
    print("✗ Failed to import from modified_audio_processing_2.py")
    print(f"  Error: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------
# Configuration — update these to your environment
# ---------------------------------------------------------------------

PIPELINE_DIR = Path(__file__).parent.parent

# Where sync params get written
SYNC_BASE = PIPELINE_DIR / "outputs" / "sync"

# MEG base (FIF)
MEG_BASE_FIF = (
    Path.home()
    / "Library/CloudStorage/OneDrive-SharedLibraries-MacquarieUniversity/Natural_Conversations_study - Documents/analysis/natural-conversations-bids/derivatives/mne-bids-pipeline-20240628"
)  # [2]

# External audio base — adjust if needed
AUDIO_BASE = (
    Path.home()
    / "Library/CloudStorage/OneDrive-AUTUniversity/Projects/Conversational_AI/Archive/data/audios"
)  # [3]

# Auxiliary audio channel name in FIF (from your config)
AUX_CHANNEL_NAME = "MISC 007"  # [3]

# Defaults
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


def fif_path(subject: str, run: int) -> Path:
    return (
        MEG_BASE_FIF
        / subject
        / "meg"
        / f"{subject}_task-conversation_run-{run:02d}_proc-clean_raw.fif"
    )  # [2]


def find_external_audio(subject: str, run: int):
    """
    Locate external interviewer and participant audio based on your config’s naming convention.
    Assumes files are under AUDIO_BASE/G{subject_num}/ and named console_mic_B{block}.wav and subject_mic_B{block}.wav [3].
    """
    # Derive G{XX} from subject (sub-01 -> G01)
    try:
        subject_num = int(subject.split("-")[1])
    except Exception:
        subject_num = None

    g_dir = (
        AUDIO_BASE / f"G{subject_num:02d}" if subject_num is not None else AUDIO_BASE
    )

    # Map run to block label (B1..B5). If your mapping differs, adjust here.
    block_label = f"B{run}"

    interviewer = g_dir / f"console_mic_{block_label}.wav"  # [3]
    participant = g_dir / f"subject_mic_{block_label}.wav"  # [3]

    return {
        "interviewer": interviewer if interviewer.exists() else None,
        "participant": participant if participant.exists() else None,
    }


def extract_aux_audio_from_fif(
    fif_file: Path, out_dir: Path, aux_name: str = AUX_CHANNEL_NAME
) -> Path:
    """
    Extract the auxiliary audio channel from a .fif file and save to WAV.
    Uses the raw channel name (e.g., 'MISC 007') defined in your config [3].
    """
    ensure_dir(out_dir)
    raw = mne.io.read_raw_fif(str(fif_file), preload=True)
    sfreq = int(raw.info["sfreq"])

    # Find channel by name (case-sensitive match)
    if aux_name in raw.ch_names:
        idx = raw.ch_names.index(aux_name)
    else:
        # Fallback: try to find a MISC channel that looks like audio
        misc_indices = [i for i, ch in enumerate(raw.ch_names) if ch.startswith("MISC")]
        if len(misc_indices) == 0:
            raise RuntimeError(
                f"No MISC channels found in {fif_file}. Please set AUX_CHANNEL_NAME correctly."
            )
        # Heuristic: pick the first MISC channel
        idx = misc_indices[0]
        print(f"  ⚠️ Using fallback MISC channel: {raw.ch_names[idx]}")

    data = raw.get_data(picks=[idx]).squeeze()  # shape (n_times,)
    # Scale to reasonable WAV range
    rms = np.sqrt(np.mean(data**2))
    scale = 0.5 / (rms + 1e-12)
    data_scaled = np.clip(data * scale, -1.0, 1.0)

    out_path = out_dir / (
        fif_file.stem.replace("_proc-clean_raw", "") + "_meg_aux_audio.wav"
    )
    soundfile.write(str(out_path), data_scaled, sfreq)
    return out_path


def compute_alignment(meg_audio_file: Path, external_audio_file: Path, work_dir: Path):
    """
    Run diagnostics and alignment using robust methods from modified_audio_processing_2.py [1].
    Returns offset_seconds, peak_correlation, diagnostics dict.
    """
    ensure_dir(work_dir)
    # Ensure external audio is left-channel only (consistent with pipeline) [1]
    left_channel = _keep_left_channel_only(str(external_audio_file), str(work_dir))
    # Diagnostics first [1]
    _ = diagnose_audio_files(str(meg_audio_file), str(left_channel))
    # Alignment
    offset, peak_corr = _align_audio(str(meg_audio_file), str(left_channel), plot=False)
    return offset, peak_corr, {"left_channel_file": left_channel}


def save_sync(subject: str, run: int, payload: dict):
    out_path = sync_params_path(subject, run)
    ensure_dir(out_path.parent)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


# ---------------------------------------------------------------------
# Per subject/run
# ---------------------------------------------------------------------


def run_sync_for_subject_run(subject: str, run: int, force: bool = False):
    if not force and check_sync_exists(subject, run):
        print(f"  ✓ {subject} run-{run:02d}: Sync params already exist (skipping)")
        return True, "exists"

    fif = fif_path(subject, run)
    if not fif.exists():
        print(f"  ✗ {subject} run-{run:02d}: MEG file not found: {fif}")
        return False, "meg_missing"

    print(f"  → {subject} run-{run:02d}: Using MEG .fif: {fif}")

    # Extract MEG aux audio from .fif [3]
    work_dir = SYNC_BASE / subject / f"run-{run:02d}" / "meg_audio"
    try:
        meg_aux_wav = extract_aux_audio_from_fif(fif, work_dir, AUX_CHANNEL_NAME)
    except Exception as e:
        print(f"  ✗ {subject} run-{run:02d}: Failed to extract aux audio: {e}")
        return False, "aux_extraction_failed"

    # Locate external audio [3]
    ext = find_external_audio(subject, run)
    if not ext["interviewer"] and not ext["participant"]:
        print(f"  ✗ {subject} run-{run:02d}: No external audio found")
        return False, "audio_missing"

    results = {
        "subject": subject,
        "run": run,
        "meg_file": str(fif),
        "meg_aux_audio": str(meg_aux_wav),
        "alignments": {},
    }

    # Align interviewer
    if ext["interviewer"]:
        try:
            offset, corr, diag = compute_alignment(
                meg_aux_wav, ext["interviewer"], work_dir
            )
            results["alignments"]["interviewer"] = {
                "external_audio": str(ext["interviewer"]),
                "offset_seconds": float(offset),
                "peak_correlation": float(corr),
                "diagnostics": diag,
            }
            print(
                f"  ✓ {subject} run-{run:02d}: Interviewer offset={offset:.3f}s corr={corr:.3f}"
            )
        except Exception as e:
            print(f"  ✗ {subject} run-{run:02d}: Interviewer sync failed: {e}")

    # Align participant
    if ext["participant"]:
        try:
            offset, corr, diag = compute_alignment(
                meg_aux_wav, ext["participant"], work_dir
            )
            results["alignments"]["participant"] = {
                "external_audio": str(ext["participant"]),
                "offset_seconds": float(offset),
                "peak_correlation": float(corr),
                "diagnostics": diag,
            }
            print(
                f"  ✓ {subject} run-{run:02d}: Participant offset={offset:.3f}s corr={corr:.3f}"
            )
        except Exception as e:
            print(f"  ✗ {subject} run-{run:02d}: Participant sync failed: {e}")

    if not results["alignments"]:
        print(f"  ✗ {subject} run-{run:02d}: No alignments produced")
        return False, "failed"

    out_path = save_sync(subject, run, results)
    print(f"  ✓ {subject} run-{run:02d}: Sync params saved → {out_path}")
    return True, "success"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Batch audio–MEG synchronization (.fif)"
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
        f"BATCH AUDIO–MEG SYNCHRONIZATION (.fif robust) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
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
