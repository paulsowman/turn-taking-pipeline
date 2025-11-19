#!/usr/bin/env python
"""
Run Montreal Forced Aligner (MFA) on Whisper transcripts.

This script:
1. Loads Whisper transcripts
2. Filters low-quality segments (hallucinations, NaN text, etc.)
3. Prepares MFA-compatible text files
4. Runs MFA alignment
5. Extracts word and phone-level alignments

Usage:
    python scripts/run_mfa_alignment.py --subject sub-01 --run 1
    python scripts/run_mfa_alignment.py --subject sub-01 --run 1 --remove-oov
"""

import argparse
import sys
from pathlib import Path
import subprocess
import pandas as pd
import shutil
import yaml
from typing import Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from asr.mfa_utils import prepare_mfa_text
from utils.logging_setup import get_logger

logger = get_logger(__name__)


def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)


def process_speaker_mfa(
    speaker: str,
    transcript_path: Path,
    audio_path: Path,
    mfa_dir: Path,
    acoustic_model: str = "english_us_arpa",
    dictionary: str = "english_us_arpa",
    no_speech_threshold: float = 0.5,
    min_avg_logprob: float = -1.0,
    remove_oov: bool = False,
    num_jobs: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run MFA alignment for one speaker.

    Parameters
    ----------
    speaker : str
        Speaker identifier ('interviewer' or 'participant').
    transcript_path : Path
        Path to Whisper transcript CSV.
    audio_path : Path
        Path to audio file.
    mfa_dir : Path
        Directory for MFA working files.
    acoustic_model : str
        MFA acoustic model name.
    dictionary : str
        MFA dictionary name.
    no_speech_threshold : float
        Quality filter threshold.
    min_avg_logprob : float
        Minimum confidence threshold.
    remove_oov : bool
        Remove out-of-vocabulary words before alignment.
    num_jobs : int
        Number of parallel jobs for MFA.

    Returns
    -------
    words_df : pd.DataFrame
        Word-level alignments.
    phones_df : pd.DataFrame
        Phone-level alignments.
    """
    logger.info(f"\n# === {speaker.upper()} ===")

    # Create speaker MFA directory
    speaker_dir = mfa_dir / speaker
    speaker_dir.mkdir(parents=True, exist_ok=True)

    # Load transcript
    logger.info(f"Loading transcript: {transcript_path}")
    transcript = pd.read_csv(transcript_path)

    # Prepare MFA text file
    text_path = speaker_dir / f"{mfa_dir.parent.name}_{speaker}.txt"
    logger.info(f"Preparing MFA text file...")

    text_path, stats = prepare_mfa_text(
        transcript=transcript,
        output_path=text_path,
        speaker=speaker,
        no_speech_threshold=no_speech_threshold,
        min_avg_logprob=min_avg_logprob,
        dictionary_name=dictionary,
        remove_oov=remove_oov,
    )

    logger.info(f"  Found {stats['n_segments_original']} segments")
    logger.info(f"  Filtered to {stats['n_segments_filtered']} high-quality segments")
    logger.info(f"  Wrote {stats['n_segments_written']} segments to text file")
    if stats['n_oov_words'] > 0:
        logger.info(f"  Removed {stats['n_oov_words']} out-of-vocabulary words")

    # Create symlink to audio file (MFA expects audio + text in same dir)
    audio_link = speaker_dir / f"{mfa_dir.parent.name}_{speaker}{audio_path.suffix}"
    if audio_link.exists():
        audio_link.unlink()
    audio_link.symlink_to(audio_path)

    # Create output directory for TextGrids
    textgrid_dir = speaker_dir / "textgrids"
    textgrid_dir.mkdir(exist_ok=True)

    # Run MFA alignment
    logger.info(f"Running MFA alignment...")
    logger.info(f"  Audio: {audio_path.name}")
    logger.info(f"  Text: {text_path.name}")
    logger.info(f"  Model: {acoustic_model}")
    logger.info(f"  Dictionary: {dictionary}")

    try:
        result = subprocess.run(
            [
                'mfa', 'align',
                '--clean',  # Clean previous runs
                '--num_jobs', str(num_jobs),
                '--output_format', 'long_textgrid',
                str(speaker_dir),
                dictionary,
                acoustic_model,
                str(textgrid_dir),
            ],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode != 0:
            logger.error(f"✗ MFA alignment failed with return code {result.returncode}")
            logger.error(f"STDOUT:\n{result.stdout}")
            logger.error(f"STDERR:\n{result.stderr}")
            raise RuntimeError(f"MFA alignment failed for {speaker}")

        logger.info(f"✓ MFA alignment complete")

    except subprocess.TimeoutExpired:
        logger.error(f"✗ MFA alignment timed out after 10 minutes")
        raise RuntimeError(f"MFA alignment timeout for {speaker}")

    # Parse TextGrid outputs
    textgrid_files = list(textgrid_dir.glob("*.TextGrid"))
    if not textgrid_files:
        logger.error(f"✗ No TextGrid files generated")
        raise RuntimeError(f"No TextGrid output for {speaker}")

    logger.info(f"  Generated {len(textgrid_files)} TextGrid file(s)")

    # Extract word and phone alignments from TextGrid
    # For now, return empty DataFrames - you can implement TextGrid parsing
    # using libraries like praatio or textgrid
    words_df = pd.DataFrame()
    phones_df = pd.DataFrame()

    logger.info(f"✓ {speaker} alignment complete")

    return words_df, phones_df


def process_subject_run(
    subject: str,
    run: int,
    config: dict,
    no_speech_threshold: float = 0.5,
    min_avg_logprob: float = -1.0,
    remove_oov: bool = False,
    acoustic_model: str = "english_us_arpa",
    dictionary: str = "english_us_arpa",
    num_jobs: int = 4,
):
    """
    Process MFA alignment for one subject/run.

    Parameters
    ----------
    subject : str
        Subject ID (e.g., 'sub-01').
    run : int
        Run number.
    config : dict
        Configuration dictionary.
    no_speech_threshold : float
        Filter segments with no_speech_prob > this value.
    min_avg_logprob : float
        Filter segments with avg_logprob < this value.
    remove_oov : bool
        Remove out-of-vocabulary words before alignment.
    acoustic_model : str
        MFA acoustic model.
    dictionary : str
        MFA pronunciation dictionary.
    num_jobs : int
        Parallel jobs for MFA.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"MFA ALIGNMENT: {subject} run-{run:02d}")
    logger.info(f"{'='*70}")

    # Get paths from config
    run_str = f"run-{run:02d}"
    features_dir = Path(config['data']['output_dir']) / 'features' / subject / run_str
    mfa_base_dir = Path(config['data']['output_dir']) / 'mfa' / subject / run_str

    # Check if transcripts exist
    transcript_int = features_dir / f"transcript_interviewer.csv"
    transcript_part = features_dir / f"transcript_participant.csv"

    if not transcript_int.exists():
        logger.error(f"✗ Interviewer transcript not found: {transcript_int}")
        raise FileNotFoundError(f"Missing transcript: {transcript_int}")

    if not transcript_part.exists():
        logger.error(f"✗ Participant transcript not found: {transcript_part}")
        raise FileNotFoundError(f"Missing transcript: {transcript_part}")

    # Get audio paths
    # Subject number for audio paths
    subj_num = int(subject.split('-')[1])
    group = f"G{subj_num:02d}"

    audio_base = Path(config['data']['external_audio_base_dir'])
    audio_int = audio_base / group / f"console_mic_B{run}.wav"
    audio_part = audio_base / group / f"subject_mic_B{run}.wav"

    if not audio_int.exists():
        logger.error(f"✗ Interviewer audio not found: {audio_int}")
        raise FileNotFoundError(f"Missing audio: {audio_int}")

    if not audio_part.exists():
        logger.error(f"✗ Participant audio not found: {audio_part}")
        raise FileNotFoundError(f"Missing audio: {audio_part}")

    # Process interviewer
    try:
        words_interviewer, phones_interviewer = process_speaker_mfa(
            "interviewer",
            transcript_int,
            audio_int,
            mfa_base_dir,
            acoustic_model,
            dictionary,
            no_speech_threshold,
            min_avg_logprob,
            remove_oov,
            num_jobs,
        )
    except Exception as e:
        logger.error(f"✗ Error during MFA alignment: {e}")
        raise

    # Process participant
    try:
        words_participant, phones_participant = process_speaker_mfa(
            "participant",
            transcript_part,
            audio_part,
            mfa_base_dir,
            acoustic_model,
            dictionary,
            no_speech_threshold,
            min_avg_logprob,
            remove_oov,
            num_jobs,
        )
    except Exception as e:
        logger.error(f"✗ Error during MFA alignment: {e}")
        raise

    logger.info(f"\n✓ MFA alignment complete for {subject} {run_str}")


def main():
    parser = argparse.ArgumentParser(description="Run MFA alignment on Whisper transcripts")
    parser.add_argument('--subject', type=str, required=True, help='Subject ID (e.g., sub-01)')
    parser.add_argument('--run', type=int, required=True, help='Run number')
    parser.add_argument('--no-speech-threshold', type=float, default=0.5,
                        help='Filter segments with no_speech_prob > this (default: 0.5)')
    parser.add_argument('--min-avg-logprob', type=float, default=-1.0,
                        help='Filter segments with avg_logprob < this (default: -1.0)')
    parser.add_argument('--remove-oov', action='store_true',
                        help='Remove out-of-vocabulary words before alignment')
    parser.add_argument('--acoustic-model', type=str, default='english_us_arpa',
                        help='MFA acoustic model (default: english_us_arpa)')
    parser.add_argument('--dictionary', type=str, default='english_us_arpa',
                        help='MFA dictionary (default: english_us_arpa)')
    parser.add_argument('--num-jobs', type=int, default=4,
                        help='Number of parallel jobs (default: 4)')

    args = parser.parse_args()

    # Load config
    config = load_config()

    # Process
    try:
        process_subject_run(
            subject=args.subject,
            run=args.run,
            config=config,
            no_speech_threshold=args.no_speech_threshold,
            min_avg_logprob=args.min_avg_logprob,
            remove_oov=args.remove_oov,
            acoustic_model=args.acoustic_model,
            dictionary=args.dictionary,
            num_jobs=args.num_jobs,
        )
        logger.info("\n✓ SUCCESS")
        return 0

    except Exception as e:
        logger.error(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
