#!/usr/bin/env python3
"""
Create audio playback with MFA word onset markers for verification.

Generates an audio file with click/beep sounds at MFA word onset times.
MFA provides ±10-20ms accuracy vs Whisper's ±50-200ms.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
import librosa
import soundfile as sf

try:
    from utils.config import load_config, get_subject_paths
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from utils.config import load_config, get_subject_paths


def load_mfa_word_onsets(subject, run, base_dir):
    """Load MFA word onset times."""
    feature_dir = base_dir / "outputs" / "features" / subject / f"run-{run:02d}"
    mfa_file = feature_dir / "transcript_mfa.csv"

    if not mfa_file.exists():
        raise FileNotFoundError(
            f"MFA transcript not found: {mfa_file}\n"
            f"Run: python scripts/run_mfa_alignment.py --subject {subject} --run {run}"
        )

    mfa_df = pd.read_csv(mfa_file)

    # MFA CSV has columns: start, end, duration, word
    word_onsets = []
    for _, row in mfa_df.iterrows():
        word_onsets.append({
            'time_audio': row['start'],
            'word': row['word']
        })

    return pd.DataFrame(word_onsets)


def generate_click(sr, duration_ms=10, frequency=1000):
    """Generate a short click/beep sound."""
    duration_s = duration_ms / 1000
    n_samples = int(duration_s * sr)
    t = np.linspace(0, duration_s, n_samples)

    # Short sine wave with envelope
    click = np.sin(2 * np.pi * frequency * t)
    # Apply envelope to avoid clicks
    envelope = np.hanning(n_samples)
    click = click * envelope

    return click


def create_onset_playback_mfa(subject, run, base_dir, start_time=0, duration=30,
                               click_duration_ms=10, click_freq=1000, click_volume=0.3):
    """
    Create audio file with MFA onset markers.

    Parameters
    ----------
    subject : str
        Subject ID
    run : int
        Run number
    base_dir : Path
        Base directory
    start_time : float
        Start time in seconds (audio timebase)
    duration : float
        Duration of output audio (seconds)
    click_duration_ms : float
        Duration of each click (milliseconds)
    click_freq : float
        Frequency of click tone (Hz)
    click_volume : float
        Volume of clicks relative to audio (0-1)
    """
    print(f"\n{'='*70}")
    print(f"CREATING MFA ONSET PLAYBACK: {subject} run-{run:02d}")
    print(f"{'='*70}\n")

    # Load config and paths
    config = load_config()
    paths = get_subject_paths(subject, run, config)

    # Load audio
    print(f"Loading audio from {start_time:.1f}s to {start_time + duration:.1f}s...")
    audio_file = paths['external_audio_interviewer']

    # IMPORTANT: Load stereo file and select only left channel (interviewer)
    sys.path.insert(0, str(base_dir / "src"))
    from utils.io import load_audio
    audio, sr = load_audio(audio_file, sr=None, channel=0, offset=start_time, duration=duration)

    print(f"  Audio: {len(audio)/sr:.1f}s @ {sr} Hz")
    print(f"  Channel: 0 (left/interviewer only)")

    # Verify audio is mono (1D array)
    if audio.ndim != 1:
        raise ValueError(f"ERROR: Audio should be 1D (mono) but got shape {audio.shape}")

    # Load MFA word onsets
    print("Loading MFA word onsets...")
    words_df = load_mfa_word_onsets(subject, run, base_dir)

    # Filter to time window
    words_in_window = words_df[
        (words_df['time_audio'] >= start_time) &
        (words_df['time_audio'] < start_time + duration)
    ].copy()

    # Adjust times relative to window start
    words_in_window['time_rel'] = words_in_window['time_audio'] - start_time

    print(f"  Found {len(words_in_window)} MFA words in window")
    print(f"  MFA timing accuracy: ±10-20ms (vs Whisper ±50-200ms)\n")

    # Generate click
    click = generate_click(sr, duration_ms=click_duration_ms, frequency=click_freq)

    # Normalize audio to prevent clipping
    audio_normalized = audio / np.max(np.abs(audio)) * 0.7

    # Create output audio with clicks
    audio_with_clicks = audio_normalized.copy()

    for _, word in words_in_window.iterrows():
        time_rel = word['time_rel']
        sample_idx = int(time_rel * sr)

        if 0 <= sample_idx < len(audio_with_clicks) - len(click):
            # Add click (mixed with existing audio)
            audio_with_clicks[sample_idx:sample_idx + len(click)] += click * click_volume

    # Prevent clipping
    audio_with_clicks = np.clip(audio_with_clicks, -1.0, 1.0)

    # Save output
    output_dir = base_dir / "outputs" / "sync" / subject / f"run-{run:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"onset_playback_mfa_{start_time:.0f}s-{start_time+duration:.0f}s.wav"
    sf.write(output_file, audio_with_clicks, sr)

    print(f"✓ Saved: {output_file}")
    print(f"\nListening instructions:")
    print(f"  1. Play the audio file")
    print(f"  2. You should hear a {click_freq}Hz beep at each word onset")
    print(f"  3. Verify that beeps align with word beginnings")
    print(f"  4. MFA timing should be more precise than Whisper")
    print(f"  5. Compare with Whisper version to hear the improvement")
    print()

    # Also save transcript for reference
    transcript_file = output_dir / f"onset_playback_mfa_{start_time:.0f}s-{start_time+duration:.0f}s_transcript.txt"
    with open(transcript_file, 'w') as f:
        f.write(f"MFA word onset times for {subject} run-{run:02d}\n")
        f.write(f"Time window: {start_time:.1f}s - {start_time + duration:.1f}s\n")
        f.write(f"Timing accuracy: ±10-20ms (MFA forced alignment)\n")
        f.write("="*70 + "\n\n")

        for _, word in words_in_window.iterrows():
            f.write(f"{word['time_audio']:7.2f}s  ({word['time_rel']:6.2f}s rel)  '{word['word']}'\n")

    print(f"✓ Saved transcript: {transcript_file}")
    print(f"\n{'='*70}\n")

    return output_file, words_in_window


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create audio playback with MFA onset markers')
    parser.add_argument('--subject', type=str, required=True, help='Subject ID (e.g., sub-01)')
    parser.add_argument('--run', type=int, required=True, help='Run number (1-5)')
    parser.add_argument('--start', type=float, default=10,
                       help='Start time in seconds (default: 10)')
    parser.add_argument('--duration', type=float, default=30,
                       help='Duration in seconds (default: 30)')
    parser.add_argument('--click-freq', type=float, default=1000,
                       help='Click frequency in Hz (default: 1000)')
    parser.add_argument('--click-duration', type=float, default=10,
                       help='Click duration in ms (default: 10)')
    parser.add_argument('--click-volume', type=float, default=0.3,
                       help='Click volume 0-1 (default: 0.3)')

    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent

    output_file, words = create_onset_playback_mfa(
        args.subject,
        args.run,
        base_dir,
        start_time=args.start,
        duration=args.duration,
        click_duration_ms=args.click_duration,
        click_freq=args.click_freq,
        click_volume=args.click_volume
    )

    print(f"\nPlay this file to verify MFA alignment:")
    print(f"  {output_file}")
