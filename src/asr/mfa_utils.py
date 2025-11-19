"""
Utilities for Montreal Forced Aligner (MFA) integration.

Handles:
- Quality filtering of Whisper transcripts before MFA alignment
- OOV (out-of-vocabulary) word handling
- MFA text file preparation
- Dictionary validation
"""

import pandas as pd
import numpy as np
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import re

from utils.logging_setup import get_logger

logger = get_logger(__name__)


def filter_low_quality_segments(
    transcript: pd.DataFrame,
    no_speech_threshold: float = 0.5,
    min_avg_logprob: float = -1.0,
    remove_empty: bool = True,
) -> pd.DataFrame:
    """
    Filter out low-quality Whisper segments before MFA alignment.

    Parameters
    ----------
    transcript : pd.DataFrame
        Whisper transcript with quality metrics.
    no_speech_threshold : float
        Maximum acceptable no_speech_prob. Segments above this are filtered.
        Default 0.5 (50% confidence that it's not speech).
    min_avg_logprob : float
        Minimum average log probability. More negative = less confident.
        Default -1.0 filters very low confidence segments.
    remove_empty : bool
        Remove segments with empty or NaN text.

    Returns
    -------
    filtered : pd.DataFrame
        Filtered transcript with only high-quality segments.

    Notes
    -----
    Common issues this filters:
    - Whisper hallucinations at end of audio (high no_speech_prob > 0.5)
    - Very low confidence transcriptions (avg_logprob < -1.0)
    - Empty/NaN text - IMPORTANT: pandas NaN values appear as "nan" string when
      written to text files, which MFA cannot align as it's not in the dictionary
      (even though "nan" can be a valid informal word for grandmother)
    - Segments with identical timestamps (processing artifacts)
    - Zero-duration segments
    """
    initial_count = len(transcript)
    filtered = transcript.copy()

    # Track filtering reasons for logging
    filters_applied = []

    # Filter 1: Remove empty or NaN text
    if remove_empty:
        before = len(filtered)
        filtered = filtered[filtered['text'].notna()]
        filtered = filtered[filtered['text'].str.strip() != '']
        removed = before - len(filtered)
        if removed > 0:
            filters_applied.append(f"{removed} empty/NaN text")

    # Filter 2: Remove high no_speech_prob
    if 'no_speech_prob' in filtered.columns:
        before = len(filtered)
        filtered = filtered[filtered['no_speech_prob'] <= no_speech_threshold]
        removed = before - len(filtered)
        if removed > 0:
            filters_applied.append(f"{removed} high no_speech_prob (>{no_speech_threshold})")

    # Filter 3: Remove very low confidence
    if 'avg_logprob' in filtered.columns:
        before = len(filtered)
        filtered = filtered[filtered['avg_logprob'] >= min_avg_logprob]
        removed = before - len(filtered)
        if removed > 0:
            filters_applied.append(f"{removed} low confidence (<{min_avg_logprob})")

    # Filter 4: Remove segments with zero duration or identical start/end
    if 'start' in filtered.columns and 'end' in filtered.columns:
        before = len(filtered)
        filtered = filtered[filtered['end'] > filtered['start']]
        removed = before - len(filtered)
        if removed > 0:
            filters_applied.append(f"{removed} zero duration")

    # Filter 5: Remove segments where all words have identical timestamps
    if 'words' in filtered.columns:
        def has_valid_word_times(words_list):
            if not words_list or not isinstance(words_list, list):
                return True  # Keep if no words list
            if len(words_list) <= 1:
                return True  # Keep single words
            # Check if all words have the same start AND end time
            times = [(w.get('start'), w.get('end')) for w in words_list]
            unique_times = set(times)
            return len(unique_times) > 1  # Keep if there's variation

        before = len(filtered)
        # Safe evaluation of words column (might be string representation of list)
        try:
            import ast
            if filtered['words'].dtype == 'object':
                # Try to parse if it's a string representation
                filtered['words_parsed'] = filtered['words'].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )
                filtered = filtered[filtered['words_parsed'].apply(has_valid_word_times)]
                filtered = filtered.drop(columns=['words_parsed'])
            else:
                filtered = filtered[filtered['words'].apply(has_valid_word_times)]
        except Exception as e:
            logger.warning(f"Could not validate word timestamps: {e}")

        removed = before - len(filtered)
        if removed > 0:
            filters_applied.append(f"{removed} invalid word timestamps")

    filtered_count = len(filtered)
    removed_total = initial_count - filtered_count

    if removed_total > 0:
        logger.info(f"Filtered {removed_total}/{initial_count} segments: {', '.join(filters_applied)}")
    else:
        logger.info(f"All {initial_count} segments passed quality filters")

    return filtered.reset_index(drop=True)


def get_mfa_dictionary_words(dictionary_name: str = "english_us_arpa") -> Optional[Set[str]]:
    """
    Get list of words in MFA dictionary.

    Parameters
    ----------
    dictionary_name : str
        Name of MFA dictionary (e.g., 'english_us_arpa').

    Returns
    -------
    words : set of str, or None
        Set of words in dictionary (lowercase), or None if dictionary not accessible.
    """
    try:
        # Run mfa model inspect
        result = subprocess.run(
            ['mfa', 'model', 'inspect', 'dictionary', dictionary_name],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            logger.warning(f"Could not inspect MFA dictionary: {result.stderr}")
            return None

        # Parse output to extract words
        # The output format varies, so we'll use a simple heuristic
        words = set()
        for line in result.stdout.split('\n'):
            # Skip header lines and empty lines
            line = line.strip()
            if not line or line.startswith('-') or 'phone' in line.lower():
                continue
            # Extract first word (before any whitespace or tab)
            parts = re.split(r'\s+', line)
            if parts:
                word = parts[0].lower()
                if word and not word.startswith('#'):
                    words.add(word)

        logger.info(f"Loaded {len(words)} words from MFA dictionary '{dictionary_name}'")
        return words

    except subprocess.TimeoutExpired:
        logger.warning("MFA dictionary inspection timed out")
        return None
    except Exception as e:
        logger.warning(f"Could not get MFA dictionary words: {e}")
        return None


def handle_oov_words(
    text: str,
    dictionary_words: Optional[Set[str]] = None,
    remove_oov: bool = True,
) -> str:
    """
    Handle out-of-vocabulary (OOV) words in text.

    Parameters
    ----------
    text : str
        Input text.
    dictionary_words : set of str, optional
        Set of valid dictionary words. If None, OOV handling is skipped.
    remove_oov : bool
        If True, remove OOV words. If False, keep them (MFA will handle).

    Returns
    -------
    cleaned_text : str
        Text with OOV words handled.

    Notes
    -----
    Special cases handled:
    - 'nan' -> ALWAYS removed (pandas NaN artifact, not in MFA dictionary)
      Note: While "nan"/"nana" can be an informal word for grandmother,
      it's not in the english_us_arpa dictionary, and in this context
      it's always a pandas NaN->string conversion artifact
    - Numbers -> kept (MFA can handle)
    - Contractions -> kept
    - Punctuation -> normalized
    """
    # Always remove pandas NaN artifacts (appears as string "nan")
    # This is NOT the informal word for grandmother - that would be "nana"
    # In our context, "nan" only appears from pandas NaN values
    text = re.sub(r'\bnan\b', '', text, flags=re.IGNORECASE)

    # If no dictionary provided, just clean and return
    if dictionary_words is None:
        return ' '.join(text.split())  # Normalize whitespace

    # Tokenize (simple whitespace + punctuation handling)
    words = text.split()
    cleaned_words = []

    for word in words:
        # Extract core word (remove punctuation for matching)
        core_word = re.sub(r'[^\w\'-]', '', word).lower()

        # Skip empty
        if not core_word:
            continue

        # Always keep numbers
        if core_word.replace('.', '').replace(',', '').isdigit():
            cleaned_words.append(word)
            continue

        # Check if in dictionary
        if core_word in dictionary_words:
            cleaned_words.append(word)
        elif not remove_oov:
            # Keep OOV word - MFA will handle
            cleaned_words.append(word)
        else:
            logger.debug(f"Removing OOV word: '{word}'")

    return ' '.join(cleaned_words)


def prepare_mfa_text(
    transcript: pd.DataFrame,
    output_path: Path,
    speaker: str = "speaker",
    no_speech_threshold: float = 0.5,
    min_avg_logprob: float = -1.0,
    dictionary_name: str = "english_us_arpa",
    remove_oov: bool = False,
) -> Tuple[Path, Dict]:
    """
    Prepare MFA-compatible text file from Whisper transcript.

    Parameters
    ----------
    transcript : pd.DataFrame
        Whisper transcript with columns: segment_id, start, end, text, etc.
    output_path : Path
        Output path for .txt file.
    speaker : str
        Speaker identifier for logging.
    no_speech_threshold : float
        Quality filter: max no_speech_prob to keep segment.
    min_avg_logprob : float
        Quality filter: min avg_logprob to keep segment.
    dictionary_name : str
        MFA dictionary name for OOV checking.
    remove_oov : bool
        If True, remove out-of-vocabulary words. If False, keep them.

    Returns
    -------
    text_path : Path
        Path to created text file.
    stats : dict
        Statistics about the preparation:
        - n_segments_original: Original segment count
        - n_segments_filtered: Segments after quality filtering
        - n_segments_written: Segments written to file
        - n_oov_words: Out-of-vocabulary words found (if checked)

    Notes
    -----
    MFA expects a plain text file with:
    - One utterance per file (for single-file alignment), or
    - All text concatenated with spaces (which we use)

    This function:
    1. Filters low-quality segments (hallucinations, artifacts)
    2. Optionally checks for OOV words
    3. Writes clean text file for MFA
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        'n_segments_original': len(transcript),
        'n_segments_filtered': 0,
        'n_segments_written': 0,
        'n_oov_words': 0,
    }

    # Step 1: Quality filtering
    filtered = filter_low_quality_segments(
        transcript,
        no_speech_threshold=no_speech_threshold,
        min_avg_logprob=min_avg_logprob,
    )
    stats['n_segments_filtered'] = len(filtered)

    if len(filtered) == 0:
        logger.warning(f"No segments passed quality filters for {speaker}")
        # Write empty file
        output_path.write_text("")
        return output_path, stats

    # Step 2: Optional OOV handling
    dictionary_words = None
    if remove_oov:
        dictionary_words = get_mfa_dictionary_words(dictionary_name)
        if dictionary_words is None:
            logger.warning("Could not load dictionary, skipping OOV removal")

    # Step 3: Combine text from all segments
    texts = []
    for _, row in filtered.iterrows():
        text = str(row['text']).strip()
        if not text or text == 'nan':
            continue

        # Handle OOV if requested
        if dictionary_words is not None:
            original_words = set(re.findall(r'\b\w+\b', text.lower()))
            text = handle_oov_words(text, dictionary_words, remove_oov=True)
            cleaned_words = set(re.findall(r'\b\w+\b', text.lower()))
            oov = original_words - cleaned_words - {'nan'}  # Exclude 'nan'
            stats['n_oov_words'] += len(oov)
            if oov:
                logger.debug(f"OOV words removed: {oov}")
        else:
            # Always remove 'nan' even without dictionary
            text = handle_oov_words(text, None, remove_oov=False)

        if text:  # Only add non-empty text
            texts.append(text)

    stats['n_segments_written'] = len(texts)

    # Step 4: Write to file
    combined_text = ' '.join(texts)
    output_path.write_text(combined_text)

    logger.info(f"Prepared MFA text for {speaker}: "
                f"{stats['n_segments_original']} → "
                f"{stats['n_segments_filtered']} → "
                f"{stats['n_segments_written']} segments, "
                f"{len(combined_text.split())} words")

    if stats['n_oov_words'] > 0:
        logger.info(f"Removed {stats['n_oov_words']} OOV words")

    return output_path, stats
