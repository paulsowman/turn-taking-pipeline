#!/usr/bin/env python
"""
Test audio-MEG synchronization on a single subject/run.

This script verifies that the synchronization module works correctly
by processing one session and generating diagnostic outputs.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import load_config, get_subject_paths, get_output_paths
from utils.io import load_meg_raw, save_sync_params
from utils.logging_setup import setup_logging
from sync.audio_meg_sync import synchronize_audio_meg, ext_to_meg_time
from qc.sync_qc import plot_sync_diagnostics, generate_sync_report


def test_synchronization(subject: str, run: int, config_path: Path = None):
    """
    Test synchronization for one subject and run.

    Parameters
    ----------
    subject : str
        Subject ID (e.g., "sub-01").
    run : int
        Run number (1-6).
    config_path : Path, optional
        Path to config file.
    """
    # Load config
    config = load_config(config_path)

    # Setup logging
    logger = setup_logging(
        log_file=config["processing"].get("log_file"),
        level=config["processing"].get("log_level", "INFO"),
    )

    logger.info("=" * 70)
    logger.info("TESTING AUDIO-MEG SYNCHRONIZATION")
    logger.info("=" * 70)
    logger.info(f"Subject: {subject}")
    logger.info(f"Run: {run}")
    logger.info("")

    # Get file paths
    try:
        paths = get_subject_paths(subject, run, config)
        logger.info("Input files:")
        logger.info(f"  MEG: {paths['meg_raw']}")
        logger.info(f"  Transcript: {paths['transcript']}")
    except Exception as e:
        logger.error(f"Error getting file paths: {e}")
        logger.error("Please check your config.yaml paths are correct")
        return False

    # Check if MEG file exists
    if not paths['meg_raw'].exists():
        logger.error(f"MEG file not found: {paths['meg_raw']}")
        logger.error("Please verify the meg_base_dir in config.yaml")
        return False

    logger.info("")
    logger.info("Loading MEG data...")

    # Load MEG raw
    try:
        meg_raw = load_meg_raw(paths['meg_raw'], preload=True, verbose='WARNING')
        logger.info(f"  Loaded: {len(meg_raw.times)} samples, {meg_raw.info['sfreq']} Hz")
        logger.info(f"  Duration: {meg_raw.times[-1]:.1f} seconds")
        logger.info(f"  Channels: {len(meg_raw.ch_names)}")
    except Exception as e:
        logger.error(f"Error loading MEG file: {e}")
        return False

    # Check for auxiliary channel
    aux_channel = config["sync"].get("aux_channel_name", "MISC007")
    if aux_channel not in meg_raw.ch_names:
        logger.error(f"Auxiliary channel {aux_channel} not found in MEG data")
        logger.error(f"Available channels: {[ch for ch in meg_raw.ch_names if 'MISC' in ch or 'STI' in ch]}")
        logger.error("Please update aux_channel_name in config.yaml")
        return False

    logger.info(f"  Found aux channel: {aux_channel}")

    # TODO: Get external audio path
    # For now, we need to figure out where your external audio files are
    # They're not in the MEG BIDS directory based on your previous code

    logger.info("")
    logger.warning("EXTERNAL AUDIO PATH NEEDED")
    logger.warning("=" * 70)
    logger.warning("The synchronization module needs the external WAV audio file.")
    logger.warning("")
    logger.warning("Based on your previous code (modified_audio_processing_2.py),")
    logger.warning("external audio is stored in a separate location.")
    logger.warning("")
    logger.warning("Please provide the path to:")
    logger.warning(f"  - Interviewer audio: console_mic*run-{run:02d}.wav")
    logger.warning(f"  - Participant audio: subject_mic*run-{run:02d}.wav")
    logger.warning("")
    logger.warning("Once you add this to config.yaml, the test will complete.")
    logger.warning("=" * 70)

    # For demonstration, let's show what WOULD happen:
    logger.info("")
    logger.info("NEXT STEPS (once external audio path is configured):")
    logger.info("1. Load external WAV file")
    logger.info("2. Run synchronization (cross-correlation + drift estimation)")
    logger.info("3. Generate QC plots:")
    logger.info("   - Envelope alignment")
    logger.info("   - Drift correction over time")
    logger.info("   - Waveform alignment at multiple time points")
    logger.info("4. Save synchronization parameters to JSON")
    logger.info("5. Generate QC report")

    # Get output paths
    output_paths = get_output_paths(subject, run, config, stage="sync")
    logger.info("")
    logger.info("Outputs would be saved to:")
    for key, path in output_paths.items():
        logger.info(f"  {key}: {path}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    logger.info("✓ Configuration loaded successfully")
    logger.info("✓ MEG file found and loaded")
    logger.info("✓ Auxiliary audio channel verified")
    logger.info("✗ External audio path needs to be configured")
    logger.info("")
    logger.info("TO COMPLETE SETUP:")
    logger.info("1. Add external_audio_dir to config.yaml under data:")
    logger.info('   external_audio_dir: "/path/to/your/audio/files"')
    logger.info("2. Add external_audio_pattern:")
    logger.info('   external_audio_pattern: "console_mic_{subject}*run-{run:02d}.wav"')
    logger.info("3. Re-run this test script")
    logger.info("=" * 70)

    return False  # Not fully complete yet


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test audio-MEG synchronization on a single subject/run"
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="sub-01",
        help="Subject ID (e.g., sub-01)",
    )
    parser.add_argument(
        "--run",
        type=int,
        default=1,
        help="Run number (1-6)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config file (default: config/config.yaml)",
    )

    args = parser.parse_args()

    success = test_synchronization(args.subject, args.run, args.config)

    if success:
        print("\n✓ Synchronization test PASSED")
        sys.exit(0)
    else:
        print("\n✗ Synchronization test INCOMPLETE (see above)")
        sys.exit(1)


if __name__ == "__main__":
    main()
