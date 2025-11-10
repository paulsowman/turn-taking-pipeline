#!/usr/bin/env python
"""
Test audio-MEG synchronization on a single subject/run - COMPLETE VERSION.

This script runs the full synchronization pipeline and generates all QC outputs.
"""

import sys
import argparse
from pathlib import Path

# Import from installed package (no need to modify sys.path after pip install -e .)
try:
    from utils.config import load_config, get_subject_paths, get_output_paths
    from utils.io import load_meg_raw, save_sync_params
    from utils.logging_setup import setup_logging
    from sync.audio_meg_sync import synchronize_audio_meg
    from qc.sync_qc import plot_sync_diagnostics, generate_sync_report
except ImportError:
    # Fallback: add src to path if not installed
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from utils.config import load_config, get_subject_paths, get_output_paths
    from utils.io import load_meg_raw, save_sync_params
    from utils.logging_setup import setup_logging
    from sync.audio_meg_sync import synchronize_audio_meg
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
        if "external_audio_interviewer" in paths:
            logger.info(f"  External audio: {paths['external_audio_interviewer']}")
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

    # Check for external audio
    if "external_audio_interviewer" not in paths:
        logger.error("")
        logger.error("External audio file not found!")
        subject_num = subject.replace("sub-", "")
        logger.error(f"Expected: {config['data']['external_audio_base_dir']}/G{subject_num}/console_mic_B{run}.wav")
        logger.error("")
        logger.error("Please check:")
        logger.error("1. external_audio_base_dir is correct in config.yaml")
        logger.error("2. Audio files exist for this subject")
        return False

    external_audio_path = paths["external_audio_interviewer"]
    logger.info(f"  Found external audio: {external_audio_path}")

    # Get output paths
    output_paths = get_output_paths(subject, run, config, stage="sync")
    logger.info("")
    logger.info("Output directory:")
    logger.info(f"  {output_paths['sync_params'].parent}")

    # Run synchronization
    logger.info("")
    logger.info("=" * 70)
    logger.info("RUNNING SYNCHRONIZATION")
    logger.info("=" * 70)

    try:
        sync_params = synchronize_audio_meg(
            meg_raw,
            external_audio_path,
            config,
            aux_channel=aux_channel,
        )

        logger.info("")
        logger.info("Synchronization successful!")
        logger.info(f"  Initial offset: {sync_params['initial_offset_s']:.4f} s")
        logger.info(f"  Drift rate: {sync_params['qc_metrics']['drift_rate_ppm']:.2f} ppm")
        logger.info(f"  Median error: {sync_params['qc_metrics']['median_error_ms']:.2f} ms")
        logger.info(f"  Quality: {sync_params['qc_metrics']['sync_quality'].upper()}")

    except Exception as e:
        logger.error(f"Synchronization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

    # Save sync parameters
    logger.info("")
    logger.info("Saving synchronization parameters...")
    try:
        save_sync_params(sync_params, output_paths["sync_params"])
        logger.info(f"  Saved: {output_paths['sync_params']}")
    except Exception as e:
        logger.error(f"Error saving sync params: {e}")
        return False

    # Generate QC report
    logger.info("")
    logger.info("Generating QC report...")
    try:
        generate_sync_report(sync_params, output_paths["qc_report"])
        logger.info(f"  Saved: {output_paths['qc_report']}")
    except Exception as e:
        logger.error(f"Error generating QC report: {e}")
        return False

    # Generate QC plots
    logger.info("")
    logger.info("Generating QC plots...")
    logger.info("  (This may take a minute...)")

    try:
        # Need to re-extract audio for plotting
        from sync.audio_meg_sync import _extract_meg_audio, _load_external_audio, _preprocess_audio, _compute_envelope

        sync_cfg = config["sync"]

        # Extract and preprocess again
        meg_audio, meg_sfreq = _extract_meg_audio(meg_raw, aux_channel)
        ext_audio, ext_sfreq = _load_external_audio(external_audio_path, sync_cfg)

        meg_audio_prep = _preprocess_audio(meg_audio, meg_sfreq, sync_cfg)
        ext_audio_prep = _preprocess_audio(ext_audio, ext_sfreq, sync_cfg, target_sr=meg_sfreq)

        meg_env = _compute_envelope(meg_audio_prep, meg_sfreq, sync_cfg)
        ext_env = _compute_envelope(ext_audio_prep, meg_sfreq, sync_cfg)

        plot_sync_diagnostics(
            meg_audio,
            ext_audio,
            meg_env,
            ext_env,
            sync_params,
            output_paths["sync_params"].parent,
            meg_sfreq,
            ext_sfreq,
        )

        logger.info(f"  Saved plots to: {output_paths['sync_params'].parent}")

    except Exception as e:
        logger.warning(f"Error generating plots (non-fatal): {e}")
        import traceback
        logger.debug(traceback.format_exc())

    # Final summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("TEST COMPLETE - RESULTS SUMMARY")
    logger.info("=" * 70)

    qc = sync_params["qc_metrics"]

    if qc["sync_quality"] == "excellent":
        logger.info("✓✓✓ EXCELLENT: Median error < 10 ms")
    elif qc["sync_quality"] == "pass":
        logger.info("✓✓  PASS: Median error < 50 ms")
    else:
        logger.warning("✗   FAIL: Median error > 50 ms")
        logger.warning("    Check QC plots and audio files")

    logger.info("")
    logger.info(f"Median alignment error: {qc['median_error_ms']:.2f} ms")
    logger.info(f"Max alignment error:    {qc['max_error_ms']:.2f} ms")
    logger.info(f"Drift rate:             {qc['drift_rate_ppm']:.2f} ppm")
    logger.info("")
    logger.info("Outputs:")
    logger.info(f"  Sync params:  {output_paths['sync_params']}")
    logger.info(f"  QC report:    {output_paths['qc_report']}")
    logger.info(f"  QC plots:     {output_paths['sync_params'].parent}/*.png")
    logger.info("=" * 70)

    return qc["sync_quality"] in ["excellent", "pass"]


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
        print("\n✗ Synchronization test FAILED (see log above)")
        sys.exit(1)


if __name__ == "__main__":
    main()
