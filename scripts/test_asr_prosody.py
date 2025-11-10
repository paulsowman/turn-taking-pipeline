#!/usr/bin/env python3
"""
Test ASR and Prosody Extraction Pipeline

Demonstrates end-to-end processing:
1. Audio-MEG synchronization
2. ASR (Whisper) transcription
3. Prosody feature extraction
4. TRP detection
"""

import sys
from pathlib import Path
import argparse

# Add src to path if not installed
try:
    from utils.config import load_config, get_subject_paths
    from utils.io import load_meg_raw, save_sync_params
    from utils.logging_setup import setup_logging
    from sync.audio_meg_sync import synchronize_audio_meg
    from asr.whisper_asr import transcribe_audio, align_to_meg_time, detect_turns
    from prosody.features import (extract_prosody_features, detect_pauses,
                                   calculate_speech_rate, extract_trp_features)
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from utils.config import load_config, get_subject_paths
    from utils.io import load_meg_raw, save_sync_params
    from utils.logging_setup import setup_logging
    from sync.audio_meg_sync import synchronize_audio_meg
    from asr.whisper_asr import transcribe_audio, align_to_meg_time, detect_turns
    from prosody.features import (extract_prosody_features, detect_pauses,
                                   calculate_speech_rate, extract_trp_features)


def test_asr_prosody_pipeline(subject: str, run: int, config_path: Path = None):
    """
    Test complete ASR + Prosody pipeline on one subject/run.

    Parameters
    ----------
    subject : str
        Subject ID (e.g., "sub-01")
    run : int
        Run number (1-5)
    config_path : Path, optional
        Path to config file. Uses default if None.
    """
    # Setup logging
    logger = setup_logging("turn_taking_pipeline")

    logger.info("=" * 70)
    logger.info("TESTING ASR + PROSODY PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Subject: {subject}")
    logger.info(f"Run: {run}")
    logger.info("")

    # Load config
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"

    config = load_config(config_path)
    paths = get_subject_paths(subject, run, config)

    # Get auxiliary channel from config
    aux_channel = config["sync"].get("aux_channel_name", "MISC 007")

    # Check files exist
    logger.info("Input files:")
    logger.info(f"  MEG: {paths['meg_raw']}")
    logger.info(f"  External audio: {paths['external_audio_interviewer']}")
    logger.info("")

    if not paths["meg_raw"].exists():
        logger.error(f"MEG file not found: {paths['meg_raw']}")
        return False

    if "external_audio_interviewer" not in paths:
        logger.error("External audio not found in paths")
        return False

    external_audio_path = paths["external_audio_interviewer"]

    if not external_audio_path.exists():
        logger.error(f"External audio not found: {external_audio_path}")
        return False

    # Output directory
    output_base = Path(config["data"]["output_dir"]) / "features" / subject / f"run-{run:02d}"
    output_base.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_base}")
    logger.info("")

    # ========================================================================
    # Step 1: Audio-MEG Synchronization
    # ========================================================================
    logger.info("=" * 70)
    logger.info("STEP 1: AUDIO-MEG SYNCHRONIZATION")
    logger.info("=" * 70)

    meg_raw = load_meg_raw(paths["meg_raw"])

    sync_params = synchronize_audio_meg(
        meg_raw,
        external_audio_path,
        config,
        aux_channel=aux_channel,
    )

    logger.info(f"  Offset: {sync_params['initial_offset_s']:.4f} s")
    logger.info(f"  Correlation: {sync_params['qc_metrics']['peak_correlation']:.4f}")
    logger.info(f"  Quality: {sync_params['qc_metrics']['sync_quality'].upper()}")
    logger.info("")

    # Save sync params
    sync_path = output_base / "sync_params.json"
    save_sync_params(sync_params, sync_path)
    logger.info(f"  Saved: {sync_path}")
    logger.info("")

    # ========================================================================
    # Step 2: ASR Transcription
    # ========================================================================
    logger.info("=" * 70)
    logger.info("STEP 2: ASR TRANSCRIPTION (WHISPER)")
    logger.info("=" * 70)

    # Get Whisper model from config
    whisper_model = config.get("asr", {}).get("whisper_model", "base")

    logger.info(f"  Model: {whisper_model}")

    transcript = transcribe_audio(
        external_audio_path,
        model_name=whisper_model,
        language="en",
    )

    logger.info(f"  Segments: {len(transcript)}")
    logger.info(f"  Total words: {sum(len(seg.get('words', [])) for _, seg in transcript.iterrows())}")
    logger.info("")

    # Align to MEG time
    transcript_meg = align_to_meg_time(transcript, sync_params)

    # Detect turns
    turns = detect_turns(transcript_meg, min_gap_s=0.2, min_duration_s=0.5)

    logger.info(f"  Detected turns: {len(turns)}")
    logger.info("")

    # Save transcript
    transcript_path = output_base / "transcript.csv"
    transcript_meg.to_csv(transcript_path, index=False)
    logger.info(f"  Saved: {transcript_path}")

    turns_path = output_base / "turns.csv"
    turns.to_csv(turns_path, index=False)
    logger.info(f"  Saved: {turns_path}")
    logger.info("")

    # ========================================================================
    # Step 3: Prosody Extraction
    # ========================================================================
    logger.info("=" * 70)
    logger.info("STEP 3: PROSODY EXTRACTION")
    logger.info("=" * 70)

    prosody = extract_prosody_features(
        external_audio_path,
        sr=None,  # Use native sample rate
        frame_shift=0.01,  # 10ms frames
        f0_min=75.0,
        f0_max=500.0,
    )

    logger.info(f"  Frames: {len(prosody)}")
    logger.info(f"  Duration: {prosody['time'].max():.1f}s")
    logger.info(f"  Mean F0: {prosody['f0'].mean():.1f} Hz")
    logger.info(f"  Voiced: {prosody['is_voiced'].mean()*100:.1f}%")
    logger.info("")

    # Detect pauses
    pauses = detect_pauses(prosody, min_pause_duration=0.2)

    logger.info(f"  Pauses detected: {len(pauses)}")
    logger.info("")

    # Calculate speech rate
    speech_rate = calculate_speech_rate(transcript_meg, window_size=5.0)

    logger.info(f"  Speech rate timepoints: {len(speech_rate)}")
    if len(speech_rate) > 0:
        logger.info(f"  Mean rate: {speech_rate['words_per_second'].mean():.2f} words/s")
    logger.info("")

    # Save prosody features
    prosody_path = output_base / "prosody.csv"
    prosody.to_csv(prosody_path, index=False)
    logger.info(f"  Saved: {prosody_path}")

    pauses_path = output_base / "pauses.csv"
    pauses.to_csv(pauses_path, index=False)
    logger.info(f"  Saved: {pauses_path}")

    speech_rate_path = output_base / "speech_rate.csv"
    speech_rate.to_csv(speech_rate_path, index=False)
    logger.info(f"  Saved: {speech_rate_path}")
    logger.info("")

    # ========================================================================
    # Step 4: TRP Feature Extraction
    # ========================================================================
    logger.info("=" * 70)
    logger.info("STEP 4: TRP FEATURE EXTRACTION")
    logger.info("=" * 70)

    trp_features = extract_trp_features(
        prosody,
        pauses,
        speech_rate,
        transcript_meg,
    )

    logger.info(f"  TRP candidates: {len(trp_features)}")
    if len(trp_features) > 0:
        logger.info(f"  Mean F0 slope: {trp_features['f0_slope'].mean():.2f} Hz/s")
        logger.info(f"  Segment-final TRPs: {trp_features['is_segment_final'].sum()}")
    logger.info("")

    # Save TRP features
    trp_path = output_base / "trp_features.csv"
    trp_features.to_csv(trp_path, index=False)
    logger.info(f"  Saved: {trp_path}")
    logger.info("")

    # ========================================================================
    # Summary
    # ========================================================================
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE - SUMMARY")
    logger.info("=" * 70)
    logger.info("")
    logger.info(f"✓ Synchronization: {sync_params['qc_metrics']['sync_quality'].upper()}")
    logger.info(f"✓ Transcription: {len(transcript)} segments, {len(turns)} turns")
    logger.info(f"✓ Prosody: {len(prosody)} frames, {len(pauses)} pauses")
    logger.info(f"✓ TRP candidates: {len(trp_features)}")
    logger.info("")
    logger.info("Output files:")
    logger.info(f"  {output_base}/sync_params.json")
    logger.info(f"  {output_base}/transcript.csv")
    logger.info(f"  {output_base}/turns.csv")
    logger.info(f"  {output_base}/prosody.csv")
    logger.info(f"  {output_base}/pauses.csv")
    logger.info(f"  {output_base}/speech_rate.csv")
    logger.info(f"  {output_base}/trp_features.csv")
    logger.info("")
    logger.info("=" * 70)
    logger.info("")
    logger.info("✓ TEST PASSED - Ready for TRF modeling!")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test ASR + Prosody extraction pipeline"
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
        help="Run number (1-5)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config file (default: config/config.yaml)",
    )

    args = parser.parse_args()

    success = test_asr_prosody_pipeline(args.subject, args.run, args.config)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
