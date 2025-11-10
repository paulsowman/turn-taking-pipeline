"""Configuration loading and path management."""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Union, Any


def load_config(config_path: Union[str, Path] = None) -> Dict[str, Any]:
    """
    Load pipeline configuration from YAML file.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to config file. If None, looks for config/config.yaml
        relative to project root.

    Returns
    -------
    config : dict
        Configuration dictionary.
    """
    if config_path is None:
        # Look for config in standard location
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Expand paths
    config = _expand_paths(config)

    return config


def _expand_paths(config: Dict) -> Dict:
    """Expand relative paths in config to absolute paths."""
    if "data" in config:
        data_cfg = config["data"]
        # Convert output_dir to absolute if relative
        if "output_dir" in data_cfg and not Path(data_cfg["output_dir"]).is_absolute():
            project_root = Path(__file__).parent.parent.parent
            data_cfg["output_dir"] = str(project_root / data_cfg["output_dir"])

    return config


def get_subject_list(config: Dict) -> List[str]:
    """
    Get list of subjects to process based on config.

    Parameters
    ----------
    config : dict
        Pipeline configuration.

    Returns
    -------
    subjects : list of str
        List of subject IDs (e.g., ["sub-01", "sub-02", ...])
    """
    subj_cfg = config.get("subjects", {})
    include = subj_cfg.get("include", "all")
    exclude = subj_cfg.get("exclude", [])

    if isinstance(include, list):
        subjects = include
    elif include == "all":
        # Generate from range
        start, end = subj_cfg.get("subject_range", [1, 32])
        subjects = [f"sub-{i:02d}" for i in range(start, end + 1)]
    else:
        raise ValueError(f"Invalid subjects.include value: {include}")

    # Apply exclusions
    subjects = [s for s in subjects if s not in exclude]

    return subjects


def get_subject_paths(
    subject: str, run: int, config: Dict
) -> Dict[str, Path]:
    """
    Get all file paths for a subject and run.

    Parameters
    ----------
    subject : str
        Subject ID (e.g., "sub-01").
    run : int
        Run number (1-6).
    config : dict
        Pipeline configuration.

    Returns
    -------
    paths : dict
        Dictionary with keys:
        - meg_raw : Path to MEG FIF file
        - fwd : Path to forward solution
        - rest : Path to rest recording (for noise cov)
        - transcript : Path to transcript CSV
        - subject_dir : Path to subject's FreeSurfer directory
    """
    data_cfg = config["data"]
    meg_base = Path(data_cfg["meg_base_dir"])
    transcript_dir = Path(data_cfg["transcript_dir"])
    freesurfer_dir = Path(data_cfg["freesurfer_dir"])

    # Extract subject number (e.g., "01" from "sub-01")
    subject_num = subject.replace("sub-", "")

    # MEG file
    meg_pattern = data_cfg["meg_pattern"]
    meg_path = meg_base / meg_pattern.format(subject=subject, run=run)

    # Forward solution
    fwd_pattern = data_cfg["fwd_pattern"]
    fwd_path = meg_base / fwd_pattern.format(subject=subject)

    # Rest recording
    rest_pattern = data_cfg["rest_pattern"]
    rest_path = meg_base / rest_pattern.format(subject=subject)

    # Transcript
    # Convert run to block (1->B1, 2->B2, etc.)
    transcript_pattern = data_cfg["transcript_pattern"]
    transcript_path = transcript_dir / transcript_pattern.format(
        subject_num=subject_num, block=run
    )

    # FreeSurfer subject directory
    fs_subject_dir = freesurfer_dir / subject

    paths = {
        "meg_raw": meg_path,
        "fwd": fwd_path,
        "rest": rest_path,
        "transcript": transcript_path,
        "subject_dir": fs_subject_dir,
    }

    # External audio (if configured)
    if "external_audio_base_dir" in data_cfg:
        import glob

        audio_base = Path(data_cfg["external_audio_base_dir"])
        # Subject audio directory: G{XX}/
        audio_subject_dir = audio_base / f"G{subject_num}"

        if audio_subject_dir.exists():
            # Interviewer audio: console_mic_B{run}.wav
            interviewer_pattern = data_cfg.get(
                "external_audio_interviewer_pattern",
                "console_mic_B{block}.wav"
            )
            interviewer_file = audio_subject_dir / interviewer_pattern.format(block=run)

            # Participant audio: subject_mic_B{run}.wav
            participant_pattern = data_cfg.get(
                "external_audio_participant_pattern",
                "subject_mic_B{block}.wav"
            )
            participant_file = audio_subject_dir / participant_pattern.format(block=run)

            if interviewer_file.exists():
                paths["external_audio_interviewer"] = interviewer_file
            if participant_file.exists():
                paths["external_audio_participant"] = participant_file

    return paths


def get_output_paths(
    subject: str,
    run: int,
    config: Dict,
    stage: str = "sync",
) -> Dict[str, Path]:
    """
    Get output file paths for a specific processing stage.

    Parameters
    ----------
    subject : str
        Subject ID.
    run : int
        Run number.
    config : dict
        Pipeline configuration.
    stage : str
        Processing stage ("sync", "features", "events", "qc").

    Returns
    -------
    paths : dict
        Output paths for this stage.
    """
    output_base = Path(config["data"]["output_dir"])
    stage_dir = output_base / stage / subject / f"run-{run:02d}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    if stage == "sync":
        return {
            "sync_params": stage_dir / "sync_params.json",
            "envelope_plot": stage_dir / "envelope_alignment.png",
            "drift_plot": stage_dir / "drift_correction.png",
            "onset_plot": stage_dir / "onset_alignment.png",
            "qc_report": stage_dir / "sync_qc_report.txt",
        }
    elif stage == "features":
        return {
            "prosody": stage_dir / "prosody_features.csv",
            "asr_output": stage_dir / "asr_transcript.json",
            "word_timestamps": stage_dir / "word_timestamps.csv",
        }
    elif stage == "events":
        return {
            "trp_events": stage_dir / "trp_events.csv",
            "turn_outcomes": stage_dir / "turn_outcomes.csv",
        }
    elif stage == "qc":
        return {
            "feature_dist": stage_dir / "feature_distributions.png",
            "vif_report": stage_dir / "vif_report.csv",
            "qc_summary": stage_dir / "qc_summary.txt",
        }
    else:
        raise ValueError(f"Unknown stage: {stage}")


def subject_id_to_number(subject: str) -> str:
    """Convert sub-01 to 01."""
    return subject.replace("sub-", "")


def number_to_subject_id(num: Union[int, str]) -> str:
    """Convert 1 or '01' to 'sub-01'."""
    if isinstance(num, int):
        return f"sub-{num:02d}"
    else:
        return f"sub-{int(num):02d}"
