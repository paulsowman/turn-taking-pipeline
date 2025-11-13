#!/usr/bin/env python3
"""
Convert Whisper Transcripts to MFA Format

Takes existing Whisper transcript CSVs and converts them to plain text files
that MFA can use for forced alignment.

Usage:
    # Single subject
    python scripts/whisper_to_mfa.py --subject sub-01 --run 1

    # Multiple subjects
    python scripts/whisper_to_mfa.py --subject sub-01 sub-02 sub-03 --runs 1 2 3 4 5

    # All subjects
    python scripts/whisper_to_mfa.py --all
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import load_config, get_subject_paths
from utils.logging_setup import setup_logging
from asr.mfa_alignment import whisper_to_mfa_text

# All available subjects
ALL_SUBJECTS = [
    'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05',
    'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10',
    'sub-11', 'sub-13', 'sub-14', 'sub-15', 'sub-16',
    'sub-17', 'sub-18', 'sub-19', 'sub-21', 'sub-22',
    'sub-23', 'sub-24', 'sub-25', 'sub-26', 'sub-27',
    'sub-29', 'sub-31', 'sub-32'
]


def convert_subject_run(subject: str, run: int, base_dir: Path, config: dict) -> bool:
    """
    Convert one subject/run Whisper transcript to MFA format.
    Creates separate MFA inputs for both interviewer and participant.

    Parameters
    ----------
    subject : str
        Subject ID (e.g., 'sub-01')
    run : int
        Run number (1-5)
    base_dir : Path
        Base directory of pipeline
    config : dict
        Configuration dictionary

    Returns
    -------
    success : bool
    """
    print(f"\n{'='*70}")
    print(f"Converting {subject} run-{run:02d} (DUAL-SPEAKER)")
    print(f"{'='*70}\n")

    # Load transcripts
    features_dir = base_dir / "outputs" / "features" / subject / f"run-{run:02d}"
    transcript_interviewer_path = features_dir / "transcript_interviewer.csv"
    transcript_participant_path = features_dir / "transcript_participant.csv"

    if not transcript_interviewer_path.exists():
        print(f"  ✗ Interviewer transcript not found: {transcript_interviewer_path}")
        return False

    if not transcript_participant_path.exists():
        print(f"  ✗ Participant transcript not found: {transcript_participant_path}")
        return False

    transcript_interviewer = pd.read_csv(transcript_interviewer_path)
    transcript_participant = pd.read_csv(transcript_participant_path)

    print(f"  Interviewer:")
    print(f"    Segments: {len(transcript_interviewer)}")
    print(f"    Total words: {sum(len(eval(seg['words']) if isinstance(seg['words'], str) else seg['words']) for _, seg in transcript_interviewer.iterrows() if not pd.isna(seg['words']))}")

    print(f"  Participant:")
    print(f"    Segments: {len(transcript_participant)}")
    print(f"    Total words: {sum(len(eval(seg['words']) if isinstance(seg['words'], str) else seg['words']) for _, seg in transcript_participant.iterrows() if not pd.isna(seg['words']))}")

    # Prepare output directories for both speakers
    mfa_dir_interviewer = base_dir / "outputs" / "mfa" / subject / f"run-{run:02d}" / "interviewer"
    mfa_dir_participant = base_dir / "outputs" / "mfa" / subject / f"run-{run:02d}" / "participant"
    mfa_dir_interviewer.mkdir(parents=True, exist_ok=True)
    mfa_dir_participant.mkdir(parents=True, exist_ok=True)

    # Get audio paths
    paths = get_subject_paths(subject, run, config)
    audio_interviewer_src = paths['external_audio_interviewer']
    audio_participant_src = paths['external_audio_participant']

    # --- INTERVIEWER ---
    print(f"\n  Processing INTERVIEWER:")

    # Convert to MFA text format
    text_file_interviewer = mfa_dir_interviewer / f"{subject}_run-{run:02d}_interviewer.txt"
    whisper_to_mfa_text(transcript_interviewer, text_file_interviewer, segments_only=False)

    # Link/copy audio file
    if audio_interviewer_src.exists():
        audio_dst_interviewer = mfa_dir_interviewer / f"{subject}_run-{run:02d}_interviewer.wav"

        if audio_dst_interviewer.exists() or audio_dst_interviewer.is_symlink():
            audio_dst_interviewer.unlink()

        try:
            audio_dst_interviewer.symlink_to(audio_interviewer_src.resolve())
            print(f"    ✓ Linked audio: {audio_dst_interviewer.name} → {audio_interviewer_src}")
        except OSError:
            import shutil
            shutil.copy2(audio_interviewer_src, audio_dst_interviewer)
            print(f"    ✓ Copied audio: {audio_dst_interviewer.name}")
    else:
        print(f"    ✗ Audio file not found: {audio_interviewer_src}")
        return False

    # --- PARTICIPANT ---
    print(f"\n  Processing PARTICIPANT:")

    # Convert to MFA text format
    text_file_participant = mfa_dir_participant / f"{subject}_run-{run:02d}_participant.txt"
    whisper_to_mfa_text(transcript_participant, text_file_participant, segments_only=False)

    # Link/copy audio file
    if audio_participant_src.exists():
        audio_dst_participant = mfa_dir_participant / f"{subject}_run-{run:02d}_participant.wav"

        if audio_dst_participant.exists() or audio_dst_participant.is_symlink():
            audio_dst_participant.unlink()

        try:
            audio_dst_participant.symlink_to(audio_participant_src.resolve())
            print(f"    ✓ Linked audio: {audio_dst_participant.name} → {audio_participant_src}")
        except OSError:
            import shutil
            shutil.copy2(audio_participant_src, audio_dst_participant)
            print(f"    ✓ Copied audio: {audio_dst_participant.name}")
    else:
        print(f"    ✗ Audio file not found: {audio_participant_src}")
        return False

    print(f"\n  ✓ MFA input prepared for BOTH speakers:")
    print(f"    Interviewer: {mfa_dir_interviewer}")
    print(f"    Participant: {mfa_dir_participant}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert Whisper transcripts to MFA format'
    )
    parser.add_argument(
        '--subject',
        nargs='+',
        help='Subject IDs (e.g., sub-01 sub-02)'
    )
    parser.add_argument(
        '--runs',
        nargs='+',
        type=int,
        help='Run numbers (e.g., 1 2 3)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Convert all subjects and runs'
    )
    parser.add_argument(
        '--config',
        type=Path,
        help='Path to config file'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging("whisper_to_mfa")

    # Base directory
    base_dir = Path(__file__).parent.parent

    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = load_config()

    # Determine subjects and runs
    if args.all:
        subjects = ALL_SUBJECTS
        runs = [1, 2, 3, 4, 5]
    else:
        if not args.subject:
            print("Error: Must specify --subject or --all")
            sys.exit(1)
        subjects = args.subject
        runs = args.runs if args.runs else [1]

    print("="*70)
    print("WHISPER → MFA CONVERSION")
    print("="*70)
    print(f"\nSubjects: {', '.join(subjects)}")
    print(f"Runs: {', '.join(map(str, runs))}")
    print(f"\nTotal conversions: {len(subjects)} × {len(runs)} = {len(subjects) * len(runs)}")
    print()

    # Convert each subject/run
    results = []
    for subject in subjects:
        for run in runs:
            try:
                success = convert_subject_run(subject, run, base_dir, config)
                results.append({
                    'subject': subject,
                    'run': run,
                    'success': success
                })
            except Exception as e:
                print(f"✗ Error converting {subject} run-{run}: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'subject': subject,
                    'run': run,
                    'success': False
                })

    # Summary
    print("\n" + "="*70)
    print("CONVERSION SUMMARY")
    print("="*70)

    n_success = sum(1 for r in results if r['success'])
    n_fail = sum(1 for r in results if not r['success'])

    print(f"\nTotal: {len(results)}")
    print(f"  ✓ Success: {n_success}")
    print(f"  ✗ Failed: {n_fail}")

    if n_fail > 0:
        print(f"\nFailed conversions:")
        for r in results:
            if not r['success']:
                print(f"  - {r['subject']} run-{r['run']}")

    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print(f"{'='*70}")
    print(f"\n1. Install MFA (if not already):")
    print(f"   pip install montreal-forced-aligner")
    print(f"")
    print(f"2. Download models:")
    print(f"   mfa model download acoustic english_us_arpa")
    print(f"   mfa model download dictionary english_us_arpa")
    print(f"")
    print(f"3. Run MFA alignment:")
    print(f"   python scripts/run_mfa_alignment.py --subject {subjects[0]} --run {runs[0]}")
    print(f"   # Or for all:")
    print(f"   python scripts/run_mfa_alignment.py --all")
    print()


if __name__ == "__main__":
    main()
