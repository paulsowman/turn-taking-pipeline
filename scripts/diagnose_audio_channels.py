#!/usr/bin/env python3
"""
Diagnose audio channel structure to verify stereo separation.

This script loads an audio file and checks:
1. Number of channels
2. Whether channels are different (stereo) or same (mono duplicated)
3. Power/RMS of each channel separately
"""

import sys
from pathlib import Path
import numpy as np

def diagnose_audio(audio_path):
    """Diagnose audio file channel structure."""
    print(f"\n{'='*70}")
    print(f"AUDIO FILE DIAGNOSIS: {Path(audio_path).name}")
    print(f"{'='*70}\n")

    # Test 1: Load with mono=False using librosa
    try:
        import librosa
        print("Test 1: Loading with librosa (mono=False)")
        audio, sr = librosa.load(audio_path, sr=None, mono=False)
        print(f"  Sample rate: {sr} Hz")
        print(f"  Shape: {audio.shape}")
        print(f"  Dtype: {audio.dtype}")

        if audio.ndim == 1:
            print(f"  → File is MONO (single channel)")
            print(f"  → RMS: {np.sqrt(np.mean(audio**2)):.6f}")
        else:
            n_channels = audio.shape[0]
            print(f"  → File has {n_channels} channels")

            for i in range(n_channels):
                ch = audio[i]
                rms = np.sqrt(np.mean(ch**2))
                print(f"  → Channel {i}: RMS = {rms:.6f}, max = {np.max(np.abs(ch)):.6f}")

            # Check if channels are identical
            if n_channels == 2:
                ch0 = audio[0]
                ch1 = audio[1]
                diff = np.abs(ch0 - ch1)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)

                print(f"\n  Channel comparison (L vs R):")
                print(f"  → Max difference: {max_diff:.6f}")
                print(f"  → Mean difference: {mean_diff:.6f}")

                if max_diff < 1e-6:
                    print(f"  → ⚠️  Channels are IDENTICAL (duplicated mono)")
                else:
                    print(f"  → ✓ Channels are DIFFERENT (true stereo)")

                # Check correlation
                corr = np.corrcoef(ch0, ch1)[0, 1]
                print(f"  → Correlation: {corr:.6f}")

    except ImportError:
        print("  librosa not available")

    # Test 2: Load with mono=True
    try:
        print("\n\nTest 2: Loading with librosa (mono=True)")
        audio_mono, sr = librosa.load(audio_path, sr=None, mono=True)
        print(f"  Shape: {audio_mono.shape}")
        print(f"  RMS: {np.sqrt(np.mean(audio_mono**2)):.6f}")
        print(f"  → This is what you get with mono=True (averages all channels)")

    except (ImportError, NameError):
        print("  librosa not available")

    # Test 3: Load with soundfile
    try:
        import soundfile as sf
        print("\n\nTest 3: Loading with soundfile")
        audio_sf, sr_sf = sf.read(audio_path, dtype='float32')
        print(f"  Sample rate: {sr_sf} Hz")
        print(f"  Shape: {audio_sf.shape}")

        if audio_sf.ndim == 1:
            print(f"  → File is MONO")
        else:
            n_channels = audio_sf.shape[1]  # soundfile uses (samples, channels)
            print(f"  → File has {n_channels} channels")

            for i in range(n_channels):
                ch = audio_sf[:, i]
                rms = np.sqrt(np.mean(ch**2))
                print(f"  → Channel {i}: RMS = {rms:.6f}")

    except ImportError:
        print("  soundfile not available")

    # Test 4: Load with offset/duration
    try:
        print("\n\nTest 4: Loading with offset and duration (mono=False)")
        audio_offset, sr = librosa.load(audio_path, sr=None, mono=False, offset=10.0, duration=5.0)
        print(f"  Shape: {audio_offset.shape}")
        print(f"  Duration: {len(audio_offset)/sr if audio_offset.ndim == 1 else len(audio_offset[0])/sr:.2f}s")

        if audio_offset.ndim == 1:
            print(f"  → ⚠️  WARNING: Result is 1D even with mono=False!")
            print(f"  → This suggests librosa may have mixed channels despite mono=False")
            print(f"  → RMS: {np.sqrt(np.mean(audio_offset**2)):.6f}")
        else:
            print(f"  → ✓ Result is 2D (stereo preserved)")
            for i in range(audio_offset.shape[0]):
                ch = audio_offset[i]
                rms = np.sqrt(np.mean(ch**2))
                print(f"  → Channel {i}: RMS = {rms:.6f}")

    except (ImportError, NameError):
        print("  librosa not available")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Diagnose audio channel structure')
    parser.add_argument('--subject', type=str, default='sub-01', help='Subject ID')
    parser.add_argument('--run', type=int, default=1, help='Run number')

    args = parser.parse_args()

    # Get audio file path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from utils.config import load_config, get_subject_paths

    config = load_config()
    paths = get_subject_paths(args.subject, args.run, config)
    audio_file = paths['external_audio_interviewer']

    print(f"Audio file: {audio_file}")

    if not audio_file.exists():
        print(f"ERROR: File not found: {audio_file}")
        sys.exit(1)

    diagnose_audio(audio_file)
