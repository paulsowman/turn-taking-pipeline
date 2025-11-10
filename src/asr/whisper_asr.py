"""
Automatic Speech Recognition using OpenAI Whisper.

Provides word-level timestamps for turn-taking analysis.
"""

import numpy as np
import pandas as pd
import whisper
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

from utils.logging_setup import get_logger

logger = get_logger(__name__)


def transcribe_audio(
    audio_path: Path,
    model_name: str = "base",
    language: str = "en",
    device: Optional[str] = None,
) -> pd.DataFrame:
    """
    Transcribe audio file using Whisper with word-level timestamps.

    Parameters
    ----------
    audio_path : Path
        Path to audio file (WAV format).
    model_name : str
        Whisper model size: 'tiny', 'base', 'small', 'medium', 'large'
        Default 'base' provides good balance of speed/accuracy.
    language : str
        Language code (e.g., 'en' for English).
    device : str, optional
        Device to run model on ('cpu' or 'cuda'). Auto-detected if None.

    Returns
    -------
    transcript : pd.DataFrame
        DataFrame with columns:
        - segment_id: Segment number
        - start: Start time in audio (seconds)
        - end: End time in audio (seconds)
        - text: Transcribed text
        - words: List of word dicts with {word, start, end, probability}
        - avg_logprob: Average log probability (confidence measure)
        - no_speech_prob: Probability of no speech

    Notes
    -----
    - Uses Whisper's word-level timestamps (requires Whisper >= 20230314)
    - Segments are determined by Whisper's VAD and natural pauses
    - Word timestamps enable fine-grained prosody analysis
    """
    audio_path = Path(audio_path)
    logger.info(f"Loading Whisper model: {model_name}")

    # Load model
    model = whisper.load_model(model_name, device=device)

    logger.info(f"Transcribing: {audio_path.name}")

    # Transcribe with word-level timestamps
    try:
        result = model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,  # Enable word-level timing
            verbose=False,
        )
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise

    # Extract segments and words
    segments = []
    for i, segment in enumerate(result["segments"]):
        seg_dict = {
            "segment_id": i,
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"].strip(),
            "words": segment.get("words", []),
            "avg_logprob": segment.get("avg_logprob", None),
            "no_speech_prob": segment.get("no_speech_prob", None),
        }
        segments.append(seg_dict)

    transcript_df = pd.DataFrame(segments)

    logger.info(f"Transcription complete: {len(transcript_df)} segments, "
                f"{sum(len(s['words']) for s in segments)} words")

    return transcript_df


def align_to_meg_time(
    transcript: pd.DataFrame,
    sync_params: Dict,
) -> pd.DataFrame:
    """
    Convert transcript timestamps from external audio time to MEG time.

    Parameters
    ----------
    transcript : pd.DataFrame
        Transcript with 'start' and 'end' columns in external audio time.
    sync_params : dict
        Synchronization parameters from audio_meg_sync module.

    Returns
    -------
    transcript_meg : pd.DataFrame
        Transcript with times aligned to MEG timebase.
        Adds columns: start_meg, end_meg

    Notes
    -----
    Uses ext_to_meg_time conversion: meg_time = ext_time - offset
    """
    offset_s = sync_params["initial_offset_s"]

    transcript_meg = transcript.copy()

    # Convert segment times
    # Positive offset means external audio starts AFTER MEG
    transcript_meg["start_meg"] = transcript["start"] + offset_s
    transcript_meg["end_meg"] = transcript["end"] + offset_s

    # Convert word times
    def convert_words(words):
        if not words:
            return []
        return [
            {
                **word,
                "start_meg": word["start"] + offset_s,
                "end_meg": word["end"] + offset_s,
            }
            for word in words
        ]

    transcript_meg["words_meg"] = transcript["words"].apply(convert_words)

    logger.info(f"Aligned {len(transcript_meg)} segments to MEG time "
                f"(offset: {offset_s:.3f}s)")

    return transcript_meg


def extract_word_features(transcript: pd.DataFrame) -> pd.DataFrame:
    """
    Extract word-level features for turn-taking analysis.

    Parameters
    ----------
    transcript : pd.DataFrame
        Transcript with word-level timestamps (from transcribe_audio).

    Returns
    -------
    word_features : pd.DataFrame
        DataFrame with one row per word:
        - word: Word text
        - start: Start time (seconds)
        - end: End time (seconds)
        - duration: Word duration
        - probability: Word confidence
        - segment_id: Parent segment ID
        - position_in_segment: Word position within segment
        - is_segment_final: Boolean, True if last word in segment

    Notes
    -----
    Useful for:
    - Prosody analysis (duration patterns)
    - TRP detection (segment-final words)
    - Surprisal calculation (word probabilities)
    """
    word_rows = []

    for _, seg in transcript.iterrows():
        words = seg.get("words", [])
        if not words:
            continue

        for pos, word in enumerate(words):
            word_rows.append({
                "word": word.get("word", "").strip(),
                "start": word.get("start"),
                "end": word.get("end"),
                "duration": word.get("end", 0) - word.get("start", 0),
                "probability": word.get("probability", None),
                "segment_id": seg["segment_id"],
                "position_in_segment": pos,
                "is_segment_final": (pos == len(words) - 1),
            })

    word_df = pd.DataFrame(word_rows)

    logger.info(f"Extracted features for {len(word_df)} words")

    return word_df


def detect_turns(
    transcript: pd.DataFrame,
    min_gap_s: float = 0.2,
    min_duration_s: float = 0.5,
) -> pd.DataFrame:
    """
    Detect conversational turns from transcript segments.

    Parameters
    ----------
    transcript : pd.DataFrame
        Transcript with segment-level timestamps.
    min_gap_s : float
        Minimum gap between segments to consider a turn boundary (seconds).
    min_duration_s : float
        Minimum turn duration to include (filters out backchannels).

    Returns
    -------
    turns : pd.DataFrame
        DataFrame with columns:
        - turn_id: Turn number
        - start: Turn start time
        - end: Turn end time
        - duration: Turn duration
        - text: Concatenated text of all segments in turn
        - n_segments: Number of segments in turn
        - n_words: Total words in turn
        - segment_ids: List of segment IDs in this turn

    Notes
    -----
    Turn boundaries detected when:
    1. Gap between segments > min_gap_s
    2. Natural conversation breaks (Whisper segments already respect pauses)

    This is a simplified TRP detector - will be enhanced in prosody module.
    """
    if len(transcript) == 0:
        return pd.DataFrame(columns=["turn_id", "start", "end", "duration",
                                      "text", "n_segments", "n_words", "segment_ids"])

    # Sort by start time
    transcript = transcript.sort_values("start").reset_index(drop=True)

    turns = []
    current_turn = {
        "segment_ids": [transcript.iloc[0]["segment_id"]],
        "start": transcript.iloc[0]["start"],
        "end": transcript.iloc[0]["end"],
        "texts": [transcript.iloc[0]["text"]],
        "words": transcript.iloc[0].get("words", []),
    }

    for i in range(1, len(transcript)):
        prev_end = transcript.iloc[i - 1]["end"]
        curr_start = transcript.iloc[i]["start"]
        gap = curr_start - prev_end

        # Check if this starts a new turn
        if gap > min_gap_s:
            # Save current turn
            turn_duration = current_turn["end"] - current_turn["start"]
            if turn_duration >= min_duration_s:
                turns.append({
                    "turn_id": len(turns),
                    "start": current_turn["start"],
                    "end": current_turn["end"],
                    "duration": turn_duration,
                    "text": " ".join(current_turn["texts"]),
                    "n_segments": len(current_turn["segment_ids"]),
                    "n_words": len(current_turn["words"]),
                    "segment_ids": current_turn["segment_ids"],
                })

            # Start new turn
            current_turn = {
                "segment_ids": [transcript.iloc[i]["segment_id"]],
                "start": transcript.iloc[i]["start"],
                "end": transcript.iloc[i]["end"],
                "texts": [transcript.iloc[i]["text"]],
                "words": transcript.iloc[i].get("words", []),
            }
        else:
            # Continue current turn
            current_turn["segment_ids"].append(transcript.iloc[i]["segment_id"])
            current_turn["end"] = transcript.iloc[i]["end"]
            current_turn["texts"].append(transcript.iloc[i]["text"])
            current_turn["words"].extend(transcript.iloc[i].get("words", []))

    # Add final turn
    if current_turn["end"] - current_turn["start"] >= min_duration_s:
        turns.append({
            "turn_id": len(turns),
            "start": current_turn["start"],
            "end": current_turn["end"],
            "duration": current_turn["end"] - current_turn["start"],
            "text": " ".join(current_turn["texts"]),
            "n_segments": len(current_turn["segment_ids"]),
            "n_words": len(current_turn["words"]),
            "segment_ids": current_turn["segment_ids"],
        })

    turns_df = pd.DataFrame(turns)

    logger.info(f"Detected {len(turns_df)} turns "
                f"(min_gap={min_gap_s}s, min_duration={min_duration_s}s)")

    return turns_df
