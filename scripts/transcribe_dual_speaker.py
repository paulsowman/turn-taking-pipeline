#!/usr/bin/env python3
"""
Transcribe Both Speakers Separately

Runs Whisper transcription on both audio channels separately to create
speaker-specific transcripts for dual-speaker MFA alignment.

Usage:
    python scripts/transcribe_dual_speaker.py --subject sub-01 --run 1
"""

import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import load_config, get_subject_paths
from utils.logging_setup import setup_logging
from asr.whisper_asr import transcribe_audio, align_to_meg_time
import json

# All available subjects
ALL_SUBJECTS = [
    'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05',
    'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10',
    'sub-11', 'sub-13', 'sub-14', 'sub-15', 'sub-16',
    'sub-17', 'sub-18', 'sub-19', 'sub-21', 'sub-22',
    'sub-23', 'sub-24', 'sub-25', 'sub-26', 'sub-27',
    'sub-29', 'sub-31', 'sub-32'
]


def transcribe_speaker(
    subject: str,
    run: int,
    base_dir: Path,
    config: dict,
    whisper_model: str = "base",
) -> bool:
    """
    Transcribe both speakers separately.

    Parameters
    ----------
    subject : str
        Subject ID
    run : int
        Run number
    base_dir : Path
        Base directory
    config : dict
        Configuration
    whisper_model : str
        Whisper model size

    Returns
    -------
    success : bool
    """
    print(f"\n{'='*70}")
    print(f"DUAL-SPEAKER TRANSCRIPTION: {subject} run-{run:02d}")
    print(f"{'='*70}\n")

    # Get paths
    paths = get_subject_paths(subject, run, config)
    audio_interviewer = paths['external_audio_interviewer']
    audio_participant = paths.get('external_audio_participant', audio_interviewer)

    # Output directory
    feature_dir = base_dir / "outputs" / "features" / subject / f"run-{run:02d}"
    feature_dir.mkdir(parents=True, exist_ok=True)

    # Load sync params (if they exist)
    sync_dir = base_dir / "outputs" / "sync" / subject / f"run-{run:02d}"
    sync_params_file = sync_dir / "sync_params.json"

    if sync_params_file.exists():
        with open(sync_params_file) as f:
            sync_params = json.load(f)
        print(f"  Sync offset: {sync_params['initial_offset_s']:.4f}s")
    else:
        print(f"  ⚠ Warning: Sync params not found, will save audio times only")
        sync_params = None

    # Transcribe INTERVIEWER (channel 0)
    print(f"\n  === INTERVIEWER (channel 0) ===")
    print(f"  Audio: {audio_interviewer.name}")
    print(f"  Model: {whisper_model}")

    transcript_interviewer = transcribe_audio(
        audio_interviewer,
        model_name=whisper_model,
        language="en",
        channel=0,  # Left channel / interviewer
    )

    print(f"  Segments: {len(transcript_interviewer)}")
    n_words_interviewer = sum(
        len(seg.get('words', [])) for _, seg in transcript_interviewer.iterrows()
    )
    print(f"  Words: {n_words_interviewer}")

    # Align to MEG time if sync params available
    if sync_params:
        transcript_interviewer = align_to_meg_time(transcript_interviewer, sync_params)

    # Save interviewer transcript
    output_interviewer = feature_dir / "transcript_interviewer.csv"
    transcript_interviewer.to_csv(output_interviewer, index=False)
    print(f"  ✓ Saved: {output_interviewer}")

    # Transcribe PARTICIPANT
    # Determine channel based on file
    if audio_participant != audio_interviewer:
        # Separate file (subject_mic) - use channel 0
        participant_channel = 0
        channel_desc = "channel 0 (separate file)"
    else:
        # Same file (console_mic) - use channel 1
        participant_channel = 1
        channel_desc = "channel 1 (right)"

    print(f"\n  === PARTICIPANT ({channel_desc}) ===")
    print(f"  Audio: {audio_participant.name}")
    print(f"  Model: {whisper_model}")

    transcript_participant = transcribe_audio(
        audio_participant,
        model_name=whisper_model,
        language="en",
        channel=participant_channel,
    )

    print(f"  Segments: {len(transcript_participant)}")
    n_words_participant = sum(
        len(seg.get('words', [])) for _, seg in transcript_participant.iterrows()
    )
    print(f"  Words: {n_words_participant}")

    # Align to MEG time if sync params available
    if sync_params:
        transcript_participant = align_to_meg_time(transcript_participant, sync_params)

    # Save participant transcript
    output_participant = feature_dir / "transcript_participant.csv"
    transcript_participant.to_csv(output_participant, index=False)
    print(f"  ✓ Saved: {output_participant}")

    print(f"\n  ✓ Dual-speaker transcription complete:")
    print(f"    Interviewer: {n_words_interviewer} words")
    print(f"    Participant: {n_words_participant} words")
    print(f"    Total: {n_words_interviewer + n_words_participant} words")
    print()

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Transcribe both speakers separately for dual-speaker MFA'
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
        '--model',
        default='base',
        help='Whisper model (default: base)'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging("transcribe_dual_speaker")

    # Base directory
    base_dir = Path(__file__).parent.parent

    # Load config
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
    print("DUAL-SPEAKER TRANSCRIPTION")
    print("="*70)
    print(f"\nSubjects: {', '.join(subjects)}")
    print(f"Runs: {', '.join(map(str, runs))}")
    print(f"Model: {args.model}")
    print(f"\nTotal: {len(subjects)} × {len(runs)} = {len(subjects) * len(runs)}")
    print()

    # Process each subject/run
    results = []
    for subject in subjects:
        for run in runs:
            try:
                success = transcribe_speaker(
                    subject=subject,
                    run=run,
                    base_dir=base_dir,
                    config=config,
                    whisper_model=args.model,
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
    print("TRANSCRIPTION SUMMARY")
    print("="*70)

    n_success = sum(1 for r in results if r['success'])
    n_fail = sum(1 for r in results if not r['success'])

    print(f"\nTotal: {len(results)}")
    print(f"  ✓ Success: {n_success}")
    print(f"  ✗ Failed: {n_fail}")

    if n_fail > 0:
        print(f"\nFailed transcriptions:")
        for r in results:
            if not r['success']:
                print(f"  - {r['subject']} run-{r['run']}")

    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print(f"{'='*70}")
    print(f"\n1. Convert to MFA format:")
    print(f"   python scripts/whisper_to_mfa.py --subject {subjects[0]} --run {runs[0]}")
    print(f"")
    print(f"2. Run MFA alignment:")
    print(f"   python scripts/run_mfa_alignment.py --subject {subjects[0]} --run {runs[0]}")
    print(f"")
    print(f"3. Create TRF FIF file:")
    print(f"   python scripts/create_trf_fif.py --subject {subjects[0]} --run {runs[0]}")
    print()


if __name__ == "__main__":
    main()
