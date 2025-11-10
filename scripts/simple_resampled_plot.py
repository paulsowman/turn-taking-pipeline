#!/usr/bin/env python3
"""
Simple plot of MEG aux audio vs external audio after resampling to a common rate.

Usage:
  python simple_resampled_plot.py --meg path/to/meg_aux.wav --ext path/to/external.wav --seconds 60

Notes:
- Both signals are resampled to target_sr = min(meg_sr, ext_sr)
- Time axes are built from target_sr to ensure correct alignment in the plots
"""

import argparse
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot resampled MEG vs external audio")
    parser.add_argument("--meg", required=True, help="Path to MEG aux WAV file")
    parser.add_argument("--ext", required=True, help="Path to external WAV file")
    parser.add_argument(
        "--seconds", type=float, default=60.0, help="Duration (s) to display"
    )
    args = parser.parse_args()

    # Load audio (mono or stereo OK; if stereo, take mean to mono)
    meg_data, meg_sr = sf.read(args.meg)
    ext_data, ext_sr = sf.read(args.ext)

    if meg_data.ndim > 1:
        meg_data = meg_data.mean(axis=1)
    if ext_data.ndim > 1:
        ext_data = ext_data.mean(axis=1)

    print(f"Loaded MEG: {len(meg_data)/meg_sr:.2f}s at {meg_sr} Hz")
    print(f"Loaded EXT: {len(ext_data)/ext_sr:.2f}s at {ext_sr} Hz")

    # Resample both to a common rate (min of the two), as in your alignment code [1]
    target_sr = int(min(meg_sr, ext_sr))
    print(f"Resampling both to common rate: {target_sr} Hz")

    if meg_sr != target_sr:
        meg_rs = librosa.resample(
            y=meg_data.astype(float), orig_sr=meg_sr, target_sr=target_sr
        )
    else:
        meg_rs = meg_data.astype(float).copy()

    if ext_sr != target_sr:
        ext_rs = librosa.resample(
            y=ext_data.astype(float), orig_sr=ext_sr, target_sr=target_sr
        )
    else:
        ext_rs = ext_data.astype(float).copy()

    # Build time axes from the same clock [1]
    meg_time = np.arange(len(meg_rs)) / target_sr
    ext_time = np.arange(len(ext_rs)) / target_sr

    # Determine plotting window
    plot_duration = min(args.seconds, meg_time[-1], ext_time[-1])
    plot_samples = int(plot_duration * target_sr)

    # Normalise for visual comparison (optional; does not affect timing)
    def safe_norm(x):
        rms = np.sqrt(np.mean(x**2)) + 1e-12
        return x / rms

    meg_plot = safe_norm(meg_rs[:plot_samples])
    ext_plot = safe_norm(ext_rs[:plot_samples])

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(meg_time[:plot_samples], meg_plot, label="MEG (resampled)", alpha=0.8)
    plt.plot(ext_time[:plot_samples], ext_plot, label="External (resampled)", alpha=0.7)
    plt.title(
        f"MEG vs External after resampling (first {plot_duration:.0f}s at {target_sr} Hz)"
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (normalised)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
