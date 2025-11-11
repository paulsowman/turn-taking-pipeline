"""I/O utilities for loading and saving data."""

import json
import pandas as pd
import mne
import librosa
from pathlib import Path
from typing import Dict, Union, List, Optional, Tuple
import numpy as np


def load_meg_raw(
    file_path: Union[str, Path],
    preload: bool = False,
    verbose: Optional[Union[bool, str]] = None,
) -> mne.io.Raw:
    """
    Load MEG raw data.

    Parameters
    ----------
    file_path : str or Path
        Path to FIF file.
    preload : bool
        Whether to preload data into memory.
    verbose : bool, str, or None
        MNE verbosity level.

    Returns
    -------
    raw : mne.io.Raw
        Raw MEG data.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"MEG file not found: {file_path}")

    raw = mne.io.read_raw_fif(file_path, preload=preload, verbose=verbose)
    return raw


def load_transcript(
    file_path: Union[str, Path],
    format: str = "csv",
) -> pd.DataFrame:
    """
    Load transcript file.

    Parameters
    ----------
    file_path : str or Path
        Path to transcript file.
    format : str
        File format ("csv" or "json").

    Returns
    -------
    transcript : pd.DataFrame
        Transcript data with columns:
        - index : int
        - person : str ("interviewer" or "participant")
        - start : float (seconds)
        - end : float (seconds)
        - text : str
        - is_full_turn : bool
        - encloses : list (indices of enclosed utterances)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {file_path}")

    if format == "csv":
        df = pd.read_csv(file_path)
        # Parse encloses column (string representation of list)
        if "encloses" in df.columns:
            df["encloses"] = df["encloses"].apply(eval)
        return df
    elif format == "json":
        with open(file_path) as f:
            data = json.load(f)
        return pd.DataFrame(data)
    else:
        raise ValueError(f"Unknown format: {format}")


def load_audio(
    file_path: Union[str, Path],
    sr: Optional[int] = None,
    channel: Optional[int] = 0,
    **kwargs
) -> Tuple[np.ndarray, float]:
    """
    Load audio file with proper channel selection for stereo files.

    Parameters
    ----------
    file_path : str or Path
        Path to audio file.
    sr : int, optional
        Target sampling rate. If None, uses file's native rate.
    channel : int, optional
        Channel to load for stereo files:
        - 0: Left channel (typically interviewer/console mic)
        - 1: Right channel (typically participant/subject mic)
        - None: Mix to mono (average both channels - NOT RECOMMENDED for dual-mic recordings)
        Default: 0 (left channel)
    **kwargs
        Additional arguments passed to librosa.load()

    Returns
    -------
    audio : np.ndarray
        Audio signal (1D array).
    sr : float
        Sampling rate.

    Notes
    -----
    For dual-microphone recordings where interviewer and participant are on
    separate channels, ALWAYS specify channel=0 or channel=1 to avoid leakage.
    Using mono=True averages both channels and causes cross-talk.

    Examples
    --------
    >>> # Load interviewer channel (left) from console_mic file
    >>> audio, sr = load_audio('console_mic_B1.wav', channel=0)
    >>>
    >>> # Load participant channel (right)
    >>> audio, sr = load_audio('console_mic_B1.wav', channel=1)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    if channel is None:
        # Mix to mono (average channels)
        audio, sr_out = librosa.load(str(file_path), sr=sr, mono=True, **kwargs)
    else:
        # Load as stereo, then select channel
        audio, sr_out = librosa.load(str(file_path), sr=sr, mono=False, **kwargs)

        # Handle mono files (returned as 1D array)
        if audio.ndim == 1:
            if channel != 0:
                raise ValueError(
                    f"Requested channel {channel} but file is mono (single channel). "
                    f"Use channel=0 or channel=None."
                )
            # Already mono, return as-is
            pass
        else:
            # Stereo (or multi-channel): select requested channel
            if channel >= audio.shape[0]:
                raise ValueError(
                    f"Requested channel {channel} but file has only {audio.shape[0]} channels. "
                    f"Valid channels: 0-{audio.shape[0]-1}"
                )
            audio = audio[channel]

    return audio, sr_out


def save_features(
    features: pd.DataFrame,
    output_path: Union[str, Path],
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save feature table to CSV with optional metadata.

    Parameters
    ----------
    features : pd.DataFrame
        Feature table.
    output_path : str or Path
        Output CSV path.
    metadata : dict, optional
        Metadata to save as JSON sidecar.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save features
    features.to_csv(output_path, index=False)

    # Save metadata
    if metadata is not None:
        metadata_path = output_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)


def load_features(
    input_path: Union[str, Path],
    load_metadata: bool = False,
) -> Union[pd.DataFrame, tuple]:
    """
    Load feature table from CSV.

    Parameters
    ----------
    input_path : str or Path
        Path to feature CSV.
    load_metadata : bool
        Whether to also load metadata JSON.

    Returns
    -------
    features : pd.DataFrame
        Feature table.
    metadata : dict (if load_metadata=True)
        Metadata dictionary.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Feature file not found: {input_path}")

    features = pd.read_csv(input_path)

    if load_metadata:
        metadata_path = input_path.with_suffix(".json")
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        else:
            metadata = {}
        return features, metadata
    else:
        return features


def save_sync_params(
    params: Dict,
    output_path: Union[str, Path],
) -> None:
    """
    Save synchronization parameters to JSON.

    Parameters
    ----------
    params : dict
        Synchronization parameters including:
        - initial_offset_s : float
        - drift_coefficients : dict
        - window_stats : list
        - qc_metrics : dict
    output_path : str or Path
        Output JSON path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    params_serializable = _make_json_serializable(params)

    with open(output_path, "w") as f:
        json.dump(params_serializable, f, indent=2)


def load_sync_params(input_path: Union[str, Path]) -> Dict:
    """Load synchronization parameters from JSON."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Sync params not found: {input_path}")

    with open(input_path) as f:
        params = json.load(f)

    return params


def _make_json_serializable(obj):
    """Recursively convert numpy arrays and other non-serializable objects."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj
