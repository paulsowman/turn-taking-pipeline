"""
TRF Predictor Generation

Functions for creating predictor time series for Temporal Response Function (TRF) analysis.
All predictors are generated at MEG sampling rate (1000 Hz) in MEG timebase.
"""

from .trf_predictors import (
    create_envelope_predictor,
    create_f0_predictor,
    create_word_onset_predictor,
    create_surprisal_predictor,
    create_duration_predictor,
    create_speaker_predictor,
    compute_word_surprisal,
)

__all__ = [
    "create_envelope_predictor",
    "create_f0_predictor",
    "create_word_onset_predictor",
    "create_surprisal_predictor",
    "create_duration_predictor",
    "create_speaker_predictor",
    "compute_word_surprisal",
]
