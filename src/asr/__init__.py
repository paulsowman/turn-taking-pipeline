"""
Automatic Speech Recognition (ASR) module.

Uses Whisper for word-level timestamps from audio.
"""

from .whisper_asr import transcribe_audio, align_to_meg_time

__all__ = ["transcribe_audio", "align_to_meg_time"]
