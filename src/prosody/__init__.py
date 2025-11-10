"""
Prosody extraction module.

Extracts prosodic features (F0, energy, pauses, speech rate) for turn-taking analysis.
"""

from .features import extract_prosody_features, detect_pauses, calculate_speech_rate

__all__ = ["extract_prosody_features", "detect_pauses", "calculate_speech_rate"]
