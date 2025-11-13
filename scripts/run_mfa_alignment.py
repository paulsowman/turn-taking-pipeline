#!/usr/bin/env python3
"""
Run Montreal Forced Aligner

Runs MFA on Whisper transcripts to get precise word and phone boundaries.

Prerequisites:
    1. pip install montreal-forced-aligner
    2. mfa model download acoustic english_us_arpa
    3. mfa model download dictionary english_us_arpa
    4. Run whisper_to_mfa.py first to prepare input files

Usage:
    # Single subject
    python scripts/run_mfa_alignment.py --subject sub-01 --run 1

    # Multiple runs for one subject
    python scripts/run_mfa_alignment.py --subject sub-01 --runs 1 2 3

    # All subjects
    python scripts/run_mfa_alignment.py --all

    # Custom models
    python scripts/run_mfa_alignment.py --subject sub-01 --run 1 \\
        --acoustic english_mfa --dictionary english_us_mfa
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.logging_setup import setup_logging
from asr.mfa_alignment import (
    run_mfa_alignment,
    check_mfa_installed,
    check_mfa_models,
    parse_textgrid,
    align_mfa_to_meg,
    compare_whisper_mfa_timing,
)

# All available subjects
ALL_SUBJECTS = [
    'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05',
    'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10',
    'sub-11', 'sub-13', 'sub-14', 'sub-15', 'sub-16',
    'sub-17', 'sub-18', 'sub-19', 'sub-21', 'sub-22',
    'sub-23', 'sub-24', 'sub-25', 'sub-26', 'sub-27',
    'sub-29', 'sub-31', 'sub-32'
]


def process_speaker_mfa(
    speaker: str,
    mfa_speaker_dir: Path,
    acoustic_model: str,
    dictionary: str,
    num_jobs: int,
) -> tuple:
    """
    Run MFA alignment for one speaker.

    Parameters
    ----------
    speaker : str
        'interviewer' or 'participant'
    mfa_speaker_dir : Path
        Directory with speaker-specific audio and text files
    acoustic_model : str
        MFA acoustic model name
    dictionary : str
        MFA dictionary name
    num_jobs : int
        Number of parallel jobs

    Returns
    -------
    words_df, phones_df : tuple of DataFrames
        Word and phone alignments (audio timebase)
    """
    print(f"\n  === {speaker.upper()} ===")

    # Check for input files
    audio_files = list(mfa_speaker_dir.glob("*.wav"))
    text_files = list(mfa_speaker_dir.glob("*.txt"))

    if len(audio_files) == 0:
        raise ValueError(f"No audio files in {mfa_speaker_dir}")

    if len(text_files) == 0:
        raise ValueError(f"No text files in {mfa_speaker_dir}")

    print(f"  Found {len(audio_files)} audio files, {len(text_files)} text files")

    # Run MFA alignment
    output_dir = mfa_speaker_dir / "textgrids"
    output_dir.mkdir(exist_ok=True)

    success = run_mfa_alignment(
        audio_dir=mfa_speaker_dir,
        text_dir=mfa_speaker_dir,
        output_dir=output_dir,
        acoustic_model=acoustic_model,
        dictionary=dictionary,
        num_jobs=num_jobs,
        clean=True,
    )

    if not success:
        raise RuntimeError(f"MFA alignment failed for {speaker}")

    # Parse TextGrid output
    textgrid_files = list(output_dir.glob("*.TextGrid"))
    if len(textgrid_files) == 0:
        raise ValueError(f"No TextGrid files generated for {speaker}")

    textgrid_file = textgrid_files[0]
    print(f"  Parsing TextGrid: {textgrid_file.name}")

    words_df, phones_df = parse_textgrid(textgrid_file)
    print(f"  ✓ {len(words_df)} words, {len(phones_df)} phones")

    return words_df, phones_df


def process_subject_run(
    subject: str,
    run: int,
    base_dir: Path,
    acoustic_model: str,
    dictionary: str,
    num_jobs: int,
    compare_timing: bool = True,
) -> bool:
    """
    Run MFA alignment for one subject/run on both speakers.

    Parameters
    ----------
    subject : str
        Subject ID
    run : int
        Run number
    base_dir : Path
        Base directory
    acoustic_model : str
        MFA acoustic model name
    dictionary : str
        MFA dictionary name
    num_jobs : int
        Number of parallel jobs
    compare_timing : bool
        Compare Whisper vs MFA timing

    Returns
    -------
    success : bool
    """
    print(f"\n{'='*70}")
    print(f"MFA ALIGNMENT (DUAL-SPEAKER): {subject} run-{run:02d}")
    print(f"{'='*70}\n")

    mfa_dir = base_dir / "outputs" / "mfa" / subject / f"run-{run:02d}"
    mfa_dir_interviewer = mfa_dir / "interviewer"
    mfa_dir_participant = mfa_dir / "participant"

    if not mfa_dir_interviewer.exists():
        print(f"✗ Interviewer MFA directory not found: {mfa_dir_interviewer}")
        print(f"  Run whisper_to_mfa.py first")
        return False

    if not mfa_dir_participant.exists():
        print(f"✗ Participant MFA directory not found: {mfa_dir_participant}")
        print(f"  Run whisper_to_mfa.py first")
        return False

    # Process both speakers
    try:
        words_interviewer, phones_interviewer = process_speaker_mfa(
            "interviewer",
            mfa_dir_interviewer,
            acoustic_model,
            dictionary,
            num_jobs,
        )

        words_participant, phones_participant = process_speaker_mfa(
            "participant",
            mfa_dir_participant,
            acoustic_model,
            dictionary,
            num_jobs,
        )
    except Exception as e:
        print(f"✗ Error during MFA alignment: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Align to MEG timebase
    sync_dir = base_dir / "outputs" / "sync" / subject / f"run-{run:02d}"
    sync_params_file = sync_dir / "sync_params.json"

    if sync_params_file.exists():
        with open(sync_params_file) as f:
            sync_params = json.load(f)

        print(f"\nAligning to MEG timebase...")
        words_interviewer_meg = align_mfa_to_meg(words_interviewer, sync_params)
        phones_interviewer_meg = align_mfa_to_meg(phones_interviewer, sync_params)
        words_participant_meg = align_mfa_to_meg(words_participant, sync_params)
        phones_participant_meg = align_mfa_to_meg(phones_participant, sync_params)
    else:
        print(f"⚠ Sync params not found: {sync_params_file}")
        print(f"  Saving audio times only (not MEG-aligned)")
        words_interviewer_meg = words_interviewer
        phones_interviewer_meg = phones_interviewer
        words_participant_meg = words_participant
        phones_participant_meg = phones_participant

    # Save output
    features_dir = base_dir / "outputs" / "features" / subject / f"run-{run:02d}"
    features_dir.mkdir(parents=True, exist_ok=True)

    # Save interviewer
    words_interviewer_output = features_dir / "transcript_mfa_interviewer.csv"
    phones_interviewer_output = features_dir / "phones_mfa_interviewer.csv"
    words_interviewer_meg.to_csv(words_interviewer_output, index=False)
    phones_interviewer_meg.to_csv(phones_interviewer_output, index=False)

    # Save participant
    words_participant_output = features_dir / "transcript_mfa_participant.csv"
    phones_participant_output = features_dir / "phones_mfa_participant.csv"
    words_participant_meg.to_csv(words_participant_output, index=False)
    phones_participant_meg.to_csv(phones_participant_output, index=False)

    print(f"\n✓ Saved MFA output:")
    print(f"  Interviewer words: {words_interviewer_output}")
    print(f"  Interviewer phones: {phones_interviewer_output}")
    print(f"  Participant words: {words_participant_output}")
    print(f"  Participant phones: {phones_participant_output}")

    # Compare with Whisper timing
    if compare_timing:
        print(f"\nComparing Whisper vs MFA timing...")

        # Interviewer comparison
        whisper_interviewer = features_dir / "transcript_interviewer.csv"
        if whisper_interviewer.exists():
            transcript_df = pd.read_csv(whisper_interviewer)
            whisper_words = []
            for _, seg in transcript_df.iterrows():
                if 'words' not in seg or pd.isna(seg['words']):
                    continue
                words_data = eval(seg['words']) if isinstance(seg['words'], str) else seg['words']
                for w in words_data:
                    whisper_words.append({
                        'start': w['start'],
                        'word': w['word'].strip()
                    })
            whisper_words_df = pd.DataFrame(whisper_words)
            comparison = compare_whisper_mfa_timing(whisper_words_df, words_interviewer)
            comparison_output = features_dir / "timing_comparison_interviewer.csv"
            comparison.to_csv(comparison_output, index=False)
            print(f"  Saved interviewer comparison: {comparison_output}")

        # Participant comparison
        whisper_participant = features_dir / "transcript_participant.csv"
        if whisper_participant.exists():
            transcript_df = pd.read_csv(whisper_participant)
            whisper_words = []
            for _, seg in transcript_df.iterrows():
                if 'words' not in seg or pd.isna(seg['words']):
                    continue
                words_data = eval(seg['words']) if isinstance(seg['words'], str) else seg['words']
                for w in words_data:
                    whisper_words.append({
                        'start': w['start'],
                        'word': w['word'].strip()
                    })
            whisper_words_df = pd.DataFrame(whisper_words)
            comparison = compare_whisper_mfa_timing(whisper_words_df, words_participant)
            comparison_output = features_dir / "timing_comparison_participant.csv"
            comparison.to_csv(comparison_output, index=False)
            print(f"  Saved participant comparison: {comparison_output}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Run Montreal Forced Aligner on Whisper transcripts'
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
        help='Process all subjects and runs'
    )
    parser.add_argument(
        '--acoustic',
        default='english_us_arpa',
        help='Acoustic model name (default: english_us_arpa)'
    )
    parser.add_argument(
        '--dictionary',
        default='english_us_arpa',
        help='Dictionary name (default: english_us_arpa)'
    )
    parser.add_argument(
        '--num-jobs',
        type=int,
        default=4,
        help='Number of parallel jobs (default: 4)'
    )
    parser.add_argument(
        '--no-compare',
        action='store_true',
        help='Skip Whisper vs MFA timing comparison'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging("mfa_alignment")

    # Base directory
    base_dir = Path(__file__).parent.parent

    # Check MFA installation
    print("Checking MFA installation...")
    if not check_mfa_installed():
        print("\n✗ MFA not installed")
        print("\nInstall with:")
        print("  pip install montreal-forced-aligner")
        sys.exit(1)

    # Check models
    acoustic_ok, dict_ok = check_mfa_models(args.acoustic, args.dictionary)
    if not (acoustic_ok and dict_ok):
        print("\n✗ Required models not found")
        print("\nDownload with:")
        print(f"  mfa model download acoustic {args.acoustic}")
        print(f"  mfa model download dictionary {args.dictionary}")
        sys.exit(1)

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

    print("\n" + "="*70)
    print("MONTREAL FORCED ALIGNER - BATCH PROCESSING")
    print("="*70)
    print(f"\nSubjects: {', '.join(subjects)}")
    print(f"Runs: {', '.join(map(str, runs))}")
    print(f"Acoustic model: {args.acoustic}")
    print(f"Dictionary: {args.dictionary}")
    print(f"Parallel jobs: {args.num_jobs}")
    print(f"\nTotal alignments: {len(subjects)} × {len(runs)} = {len(subjects) * len(runs)}")

    # Process each subject/run
    results = []
    for subject in subjects:
        for run in runs:
            try:
                success = process_subject_run(
                    subject=subject,
                    run=run,
                    base_dir=base_dir,
                    acoustic_model=args.acoustic,
                    dictionary=args.dictionary,
                    num_jobs=args.num_jobs,
                    compare_timing=not args.no_compare,
                )
                results.append({
                    'subject': subject,
                    'run': run,
                    'success': success
                })
            except Exception as e:
                print(f"\n✗ Error processing {subject} run-{run}: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'subject': subject,
                    'run': run,
                    'success': False
                })

    # Summary
    print("\n" + "="*70)
    print("MFA ALIGNMENT SUMMARY")
    print("="*70)

    n_success = sum(1 for r in results if r['success'])
    n_fail = sum(1 for r in results if not r['success'])

    print(f"\nTotal: {len(results)}")
    print(f"  ✓ Success: {n_success}")
    print(f"  ✗ Failed: {n_fail}")

    if n_fail > 0:
        print(f"\nFailed alignments:")
        for r in results:
            if not r['success']:
                print(f"  - {r['subject']} run-{r['run']}")

    print()


if __name__ == "__main__":
    main()
