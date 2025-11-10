"""
Data setup utilities for creating reproducible project structure.

This module handles:
1. Copying small files (transcripts, configs) into project
2. Creating symbolic links to large files (MEG, FreeSurfer)
3. Generating data manifest for reproducibility
"""

import shutil
import json
from pathlib import Path
from typing import Dict, List
import hashlib


def setup_project_data(config: Dict, dry_run: bool = False) -> Dict:
    """
    Set up project data directory with copies and symlinks.

    Parameters
    ----------
    config : dict
        Pipeline configuration.
    dry_run : bool
        If True, only print what would be done.

    Returns
    -------
    manifest : dict
        Data manifest with file locations and checksums.
    """
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"

    manifest = {
        "project_root": str(project_root),
        "created_by": "setup_project_data",
        "transcripts": {},
        "meg_files": {},
        "freesurfer": {},
    }

    # 1. Copy transcripts (small files)
    print("\n=== Copying transcript files ===")
    transcript_src = Path(config["data"]["transcript_dir"])
    transcript_dst = data_dir / "raw" / "transcripts"
    transcript_dst.mkdir(parents=True, exist_ok=True)

    transcript_files = list(transcript_src.glob("G*_B*.csv"))
    for src_file in transcript_files:
        dst_file = transcript_dst / src_file.name
        if dry_run:
            print(f"Would copy: {src_file.name}")
        else:
            if not dst_file.exists():
                shutil.copy2(src_file, dst_file)
                print(f"Copied: {src_file.name}")
            manifest["transcripts"][src_file.name] = {
                "original": str(src_file),
                "local": str(dst_file),
                "md5": _compute_md5(dst_file) if dst_file.exists() else None,
            }

    # 2. Create symlinks to MEG files (large files)
    print("\n=== Creating MEG file symlinks ===")
    meg_base = Path(config["data"]["meg_base_dir"])
    meg_link_dir = data_dir / "raw" / "meg_links"
    meg_link_dir.mkdir(parents=True, exist_ok=True)

    # For each subject
    from .config import get_subject_list

    subjects = get_subject_list(config)
    for subject in subjects[:3]:  # Start with first 3 subjects
        subject_meg_dir = meg_base / subject / "meg"
        if not subject_meg_dir.exists():
            continue

        # Link subject directory
        link_path = meg_link_dir / subject
        if dry_run:
            print(f"Would link: {subject} -> {subject_meg_dir}")
        else:
            if not link_path.exists():
                link_path.symlink_to(subject_meg_dir)
                print(f"Linked: {subject}")

            manifest["meg_files"][subject] = {
                "original": str(subject_meg_dir),
                "link": str(link_path),
                "files": [f.name for f in subject_meg_dir.glob("*.fif")],
            }

    # 3. Link FreeSurfer subjects directory
    print("\n=== Linking FreeSurfer directory ===")
    fs_src = Path(config["data"]["freesurfer_dir"])
    fs_link = data_dir / "raw" / "freesurfer"
    if dry_run:
        print(f"Would link: freesurfer -> {fs_src}")
    else:
        if not fs_link.exists() and fs_src.exists():
            fs_link.symlink_to(fs_src)
            print(f"Linked: freesurfer")
        manifest["freesurfer"] = {
            "original": str(fs_src),
            "link": str(fs_link),
        }

    # 4. Save manifest
    manifest_file = data_dir / "DATA_MANIFEST.json"
    if not dry_run:
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nData manifest saved to: {manifest_file}")

    return manifest


def _compute_md5(file_path: Path, chunk_size: int = 8192) -> str:
    """Compute MD5 checksum of a file."""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def update_config_for_local_data(config: Dict, use_local: bool = True) -> Dict:
    """
    Update config to use local data directory instead of original paths.

    Parameters
    ----------
    config : dict
        Original configuration.
    use_local : bool
        If True, use local data directory.

    Returns
    -------
    config : dict
        Updated configuration.
    """
    if not use_local:
        return config

    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "raw"

    # Update paths to use local data
    config["data"]["transcript_dir"] = str(data_dir / "transcripts")
    config["data"]["meg_base_dir"] = str(data_dir / "meg_links")
    config["data"]["freesurfer_dir"] = str(data_dir / "freesurfer")

    return config


def verify_data_integrity(manifest_path: Path = None) -> bool:
    """
    Verify data integrity using manifest checksums.

    Parameters
    ----------
    manifest_path : Path, optional
        Path to DATA_MANIFEST.json

    Returns
    -------
    valid : bool
        True if all files match checksums.
    """
    if manifest_path is None:
        project_root = Path(__file__).parent.parent.parent
        manifest_path = project_root / "data" / "DATA_MANIFEST.json"

    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return False

    with open(manifest_path) as f:
        manifest = json.load(f)

    print("Verifying transcript checksums...")
    all_valid = True
    for filename, info in manifest["transcripts"].items():
        local_path = Path(info["local"])
        if not local_path.exists():
            print(f"MISSING: {filename}")
            all_valid = False
            continue

        expected_md5 = info.get("md5")
        if expected_md5:
            actual_md5 = _compute_md5(local_path)
            if actual_md5 != expected_md5:
                print(f"CHECKSUM MISMATCH: {filename}")
                all_valid = False
            else:
                print(f"OK: {filename}")

    return all_valid
