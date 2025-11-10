"""Audio-MEG synchronization module."""

from .audio_meg_sync import synchronize_audio_meg, ext_to_meg_time
from .drift_correction import estimate_drift, apply_drift_correction

__all__ = [
    "synchronize_audio_meg",
    "ext_to_meg_time",
    "estimate_drift",
    "apply_drift_correction",
]
