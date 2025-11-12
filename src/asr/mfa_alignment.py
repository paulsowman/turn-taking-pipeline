"""
Montreal Forced Aligner (MFA) Integration

Provides precise word and phone-level timing by forced alignment of
transcripts to audio. Improves timing accuracy from ±50-200ms (Whisper)
to ±10-20ms (MFA).

Installation:
    pip install montreal-forced-aligner

Download models:
    mfa model download acoustic english_us_arpa
    mfa model download dictionary english_us_arpa

Usage:
    1. Get Whisper transcripts (approximate timing)
    2. Convert to MFA format (plain text)
    3. Run MFA alignment (creates TextGrid files)
    4. Parse TextGrids to get precise word/phone times
"""

import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import shutil

from utils.logging_setup import get_logger

logger = get_logger(__name__)


def whisper_to_mfa_text(
    transcript_df: pd.DataFrame,
    output_path: Union[str, Path],
    segments_only: bool = False
) -> None:
    """
    Convert Whisper transcript to MFA text format.

    MFA expects plain text files (one per audio file) with the same basename
    as the audio file.

    Parameters
    ----------
    transcript_df : pd.DataFrame
        Whisper transcript with 'text' column
    output_path : Path
        Output text file path
    segments_only : bool
        If True, write one line per segment. If False, concatenate all text.
        Default: False (single line)

    Notes
    -----
    MFA text format:
    - Plain text file (.txt)
    - Same basename as audio file (e.g., audio.wav → audio.txt)
    - Can be single line or multiple lines
    - MFA treats entire file as one utterance unless you split files
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if segments_only:
        # One line per segment
        text = "\n".join(transcript_df['text'].astype(str).tolist())
    else:
        # Single line (MFA will find word boundaries)
        text = " ".join(transcript_df['text'].astype(str).tolist())

    # Clean up text
    text = text.strip()
    text = " ".join(text.split())  # Normalize whitespace

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

    logger.info(f"Saved MFA text: {output_path}")
    logger.info(f"  Words: {len(text.split())}")
    logger.info(f"  Characters: {len(text)}")


def check_mfa_installed() -> bool:
    """Check if MFA is installed and accessible."""
    try:
        result = subprocess.run(
            ['mfa', 'version'],
            capture_output=True,
            text=True,
            timeout=30  # MFA v3.x can be slow to start on macOS
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            logger.info(f"MFA version: {version}")
            return True
        else:
            logger.error("MFA command failed")
            return False
    except FileNotFoundError:
        logger.error("MFA not found. Install with: pip install montreal-forced-aligner")
        return False
    except Exception as e:
        logger.error(f"Error checking MFA: {e}")
        return False


def check_mfa_models(
    acoustic_model: str = "english_us_arpa",
    dictionary: str = "english_us_arpa"
) -> Tuple[bool, bool]:
    """
    Check if required MFA models are downloaded.

    Returns
    -------
    acoustic_available : bool
    dictionary_available : bool
    """
    try:
        # Check acoustic model
        result = subprocess.run(
            ['mfa', 'model', 'inspect', 'acoustic', acoustic_model],
            capture_output=True,
            text=True,
            timeout=30  # MFA v3.x can be slow to start on macOS
        )
        acoustic_available = result.returncode == 0

        # Check dictionary
        result = subprocess.run(
            ['mfa', 'model', 'inspect', 'dictionary', dictionary],
            capture_output=True,
            text=True,
            timeout=30  # MFA v3.x can be slow to start on macOS
        )
        dictionary_available = result.returncode == 0

        if acoustic_available and dictionary_available:
            logger.info(f"✓ MFA models available: {acoustic_model}, {dictionary}")
        else:
            if not acoustic_available:
                logger.warning(f"✗ Acoustic model not found: {acoustic_model}")
                logger.warning(f"  Download with: mfa model download acoustic {acoustic_model}")
            if not dictionary_available:
                logger.warning(f"✗ Dictionary not found: {dictionary}")
                logger.warning(f"  Download with: mfa model download dictionary {dictionary}")

        return acoustic_available, dictionary_available

    except Exception as e:
        logger.error(f"Error checking MFA models: {e}")
        return False, False


def run_mfa_alignment(
    audio_dir: Union[str, Path],
    text_dir: Union[str, Path],
    output_dir: Union[str, Path],
    acoustic_model: str = "english_us_arpa",
    dictionary: str = "english_us_arpa",
    num_jobs: int = 4,
    clean: bool = True,
    speaker_characters: int = 0,
) -> bool:
    """
    Run Montreal Forced Aligner.

    Parameters
    ----------
    audio_dir : Path
        Directory containing audio files (.wav)
    text_dir : Path
        Directory containing transcript files (.txt)
        Must have same basename as audio files
    output_dir : Path
        Output directory for TextGrid files
    acoustic_model : str
        Name of acoustic model (default: english_us_arpa)
    dictionary : str
        Name of pronunciation dictionary (default: english_us_arpa)
    num_jobs : int
        Number of parallel jobs (default: 4)
    clean : bool
        Clean up temporary files after alignment (default: True)
    speaker_characters : int
        Number of characters in filename that identify speaker.
        0 = single speaker (default)

    Returns
    -------
    success : bool

    Notes
    -----
    MFA expects:
    - audio_dir/file.wav
    - text_dir/file.txt
    - Creates: output_dir/file.TextGrid

    Processing time: ~1-2 minutes per hour of audio
    """
    audio_dir = Path(audio_dir)
    text_dir = Path(text_dir)
    output_dir = Path(output_dir)

    logger.info("=" * 70)
    logger.info("MONTREAL FORCED ALIGNER")
    logger.info("=" * 70)

    # Check MFA installation
    if not check_mfa_installed():
        logger.error("MFA not installed. Install with: pip install montreal-forced-aligner")
        return False

    # Check models
    acoustic_ok, dict_ok = check_mfa_models(acoustic_model, dictionary)
    if not (acoustic_ok and dict_ok):
        logger.error("Required MFA models not found")
        return False

    # Validate directories
    if not audio_dir.exists():
        logger.error(f"Audio directory not found: {audio_dir}")
        return False

    if not text_dir.exists():
        logger.error(f"Text directory not found: {text_dir}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    # Count files
    audio_files = list(audio_dir.glob("*.wav"))
    text_files = list(text_dir.glob("*.txt"))

    logger.info(f"Audio directory: {audio_dir}")
    logger.info(f"  Audio files: {len(audio_files)}")
    logger.info(f"Text directory: {text_dir}")
    logger.info(f"  Text files: {len(text_files)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Acoustic model: {acoustic_model}")
    logger.info(f"Dictionary: {dictionary}")
    logger.info(f"Parallel jobs: {num_jobs}")
    logger.info("")

    if len(audio_files) == 0:
        logger.error("No audio files found")
        return False

    if len(text_files) == 0:
        logger.error("No text files found")
        return False

    # Build MFA command
    cmd = [
        'mfa', 'align',
        str(audio_dir),
        str(dictionary),
        str(acoustic_model),
        str(output_dir),
        '--num_jobs', str(num_jobs),
        '--output_format', 'long_textgrid',  # Readable TextGrid format
    ]

    if clean:
        cmd.append('--clean')

    if speaker_characters > 0:
        cmd.extend(['--speaker_characters', str(speaker_characters)])

    logger.info(f"Running MFA command:")
    logger.info(f"  {' '.join(cmd)}")
    logger.info("")

    # Run MFA
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour max
        )

        # Log output
        if result.stdout:
            logger.info("MFA stdout:")
            logger.info(result.stdout)

        if result.stderr:
            logger.info("MFA stderr:")
            logger.info(result.stderr)

        if result.returncode == 0:
            # Check output files
            textgrid_files = list(output_dir.glob("*.TextGrid"))
            logger.info("")
            logger.info(f"✓ MFA alignment complete")
            logger.info(f"  Output TextGrids: {len(textgrid_files)}")

            if len(textgrid_files) == 0:
                logger.warning("No TextGrid files generated!")
                return False

            return True
        else:
            logger.error(f"✗ MFA alignment failed with return code {result.returncode}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("MFA alignment timed out (>1 hour)")
        return False
    except Exception as e:
        logger.error(f"Error running MFA: {e}")
        return False


def parse_textgrid(
    textgrid_path: Union[str, Path],
    word_tier_name: str = "words",
    phone_tier_name: str = "phones",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse MFA TextGrid output to extract word and phone boundaries.

    Parameters
    ----------
    textgrid_path : Path
        Path to TextGrid file
    word_tier_name : str
        Name of word tier (default: "words")
    phone_tier_name : str
        Name of phone tier (default: "phones")

    Returns
    -------
    words_df : pd.DataFrame
        Word boundaries with columns:
        - start: Start time (seconds)
        - end: End time (seconds)
        - duration: Duration (seconds)
        - word: Word text
    phones_df : pd.DataFrame
        Phone boundaries with columns:
        - start: Start time (seconds)
        - end: End time (seconds)
        - duration: Duration (seconds)
        - phone: Phone symbol

    Notes
    -----
    Requires textgrid package: pip install praat-textgrids
    """
    try:
        import textgrids
    except ImportError:
        logger.error("textgrids package not installed. Install with: pip install praat-textgrids")
        raise

    textgrid_path = Path(textgrid_path)
    if not textgrid_path.exists():
        raise FileNotFoundError(f"TextGrid not found: {textgrid_path}")

    # Parse TextGrid
    tg = textgrids.TextGrid(str(textgrid_path))

    # Extract words
    words = []
    if word_tier_name in tg:
        word_tier = tg[word_tier_name]
        for interval in word_tier:
            if interval.text.strip() and interval.text.strip() != "":
                words.append({
                    'start': interval.xmin,
                    'end': interval.xmax,
                    'duration': interval.xmax - interval.xmin,
                    'word': interval.text.strip()
                })

    words_df = pd.DataFrame(words)

    # Extract phones
    phones = []
    if phone_tier_name in tg:
        phone_tier = tg[phone_tier_name]
        for interval in phone_tier:
            if interval.text.strip() and interval.text.strip() != "":
                phones.append({
                    'start': interval.xmin,
                    'end': interval.xmax,
                    'duration': interval.xmax - interval.xmin,
                    'phone': interval.text.strip()
                })

    phones_df = pd.DataFrame(phones)

    logger.info(f"Parsed TextGrid: {textgrid_path.name}")
    logger.info(f"  Words: {len(words_df)}")
    logger.info(f"  Phones: {len(phones_df)}")

    return words_df, phones_df


def align_mfa_to_meg(
    mfa_words: pd.DataFrame,
    sync_params: Dict,
) -> pd.DataFrame:
    """
    Convert MFA word times from audio timebase to MEG timebase.

    Parameters
    ----------
    mfa_words : pd.DataFrame
        MFA word boundaries with 'start' and 'end' columns
    sync_params : dict
        Synchronization parameters with 'initial_offset_s'

    Returns
    -------
    mfa_words_meg : pd.DataFrame
        Words with added columns: start_meg, end_meg, duration_meg

    Notes
    -----
    Uses the CORRECT formula: meg_time = audio_time - offset
    (accounting for reversed sign convention in sync function)
    """
    offset_s = sync_params['initial_offset_s']

    mfa_words_meg = mfa_words.copy()

    # Convert times to MEG timebase
    # NOTE: Sync function uses REVERSED sign convention!
    # Formula: meg_time = audio_time - offset
    mfa_words_meg['start_meg'] = mfa_words['start'] - offset_s
    mfa_words_meg['end_meg'] = mfa_words['end'] - offset_s
    mfa_words_meg['duration_meg'] = mfa_words_meg['end_meg'] - mfa_words_meg['start_meg']

    logger.info(f"Aligned {len(mfa_words_meg)} words to MEG timebase")
    logger.info(f"  Offset: {offset_s:.4f}s")
    logger.info(f"  Audio time range: {mfa_words['start'].min():.2f} - {mfa_words['end'].max():.2f}s")
    logger.info(f"  MEG time range: {mfa_words_meg['start_meg'].min():.2f} - {mfa_words_meg['end_meg'].max():.2f}s")

    return mfa_words_meg


def compare_whisper_mfa_timing(
    whisper_words: pd.DataFrame,
    mfa_words: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare Whisper and MFA word timing to quantify improvement.

    Parameters
    ----------
    whisper_words : pd.DataFrame
        Whisper word times with 'start' and 'word' columns
    mfa_words : pd.DataFrame
        MFA word times with 'start' and 'word' columns

    Returns
    -------
    comparison : pd.DataFrame
        Comparison with columns:
        - word: Word text
        - whisper_start: Whisper onset time
        - mfa_start: MFA onset time
        - time_diff: Difference (Whisper - MFA)
        - abs_diff: Absolute difference

    Notes
    -----
    Words are matched by order (assumes same word sequence).
    Time difference shows how much Whisper timing differs from MFA.
    """
    # Match words by position (assume same order)
    n_words = min(len(whisper_words), len(mfa_words))

    comparison = pd.DataFrame({
        'word': mfa_words['word'].iloc[:n_words].values,
        'whisper_start': whisper_words['start'].iloc[:n_words].values,
        'mfa_start': mfa_words['start'].iloc[:n_words].values,
    })

    comparison['time_diff'] = comparison['whisper_start'] - comparison['mfa_start']
    comparison['abs_diff'] = comparison['time_diff'].abs()

    # Summary statistics
    logger.info("=" * 70)
    logger.info("WHISPER vs MFA TIMING COMPARISON")
    logger.info("=" * 70)
    logger.info(f"Words compared: {len(comparison)}")
    logger.info(f"Mean difference: {comparison['time_diff'].mean()*1000:.1f} ms")
    logger.info(f"Std difference: {comparison['time_diff'].std()*1000:.1f} ms")
    logger.info(f"Mean absolute difference: {comparison['abs_diff'].mean()*1000:.1f} ms")
    logger.info(f"Median absolute difference: {comparison['abs_diff'].median()*1000:.1f} ms")
    logger.info(f"Max absolute difference: {comparison['abs_diff'].max()*1000:.1f} ms")
    logger.info(f"95th percentile: {comparison['abs_diff'].quantile(0.95)*1000:.1f} ms")

    # Count large errors
    large_errors_50ms = (comparison['abs_diff'] > 0.05).sum()
    large_errors_100ms = (comparison['abs_diff'] > 0.1).sum()
    large_errors_200ms = (comparison['abs_diff'] > 0.2).sum()

    logger.info(f"")
    logger.info(f"Errors > 50ms: {large_errors_50ms} ({100*large_errors_50ms/len(comparison):.1f}%)")
    logger.info(f"Errors > 100ms: {large_errors_100ms} ({100*large_errors_100ms/len(comparison):.1f}%)")
    logger.info(f"Errors > 200ms: {large_errors_200ms} ({100*large_errors_200ms/len(comparison):.1f}%)")
    logger.info("")

    return comparison
