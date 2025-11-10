#!/usr/bin/env python3
"""
Show FULL time series of both MEG and external audio to understand the alignment.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import mne
import librosa
from scipy import signal
from scipy.signal import hilbert, correlate
from pathlib import Path

# Hardcoded paths for sub-01 run-01
SUBJECT = "sub-01"
RUN = 1

MEG_FILE = Path("/Users/em18033/Library/CloudStorage/OneDrive-SharedLibraries-MacquarieUniversity/Natural_Conversations_study - Documents/analysis/natural-conversations-bids/derivatives/mne-bids-pipeline-20240628/sub-01/meg/sub-01_task-conversation_run-01_proc-clean_raw.fif")
AUDIO_FILE = Path("/Users/em18033/Library/CloudStorage/OneDrive-AUTUniversity/Projects/Conversational_AI/Archive/data/audios/G01/console_mic_B1.wav")
OUTPUT_DIR = Path("outputs/sync_debug")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("FULL TIME SERIES VIEW")
print("=" * 70)
print(f"Subject: {SUBJECT}, Run: {RUN}")
print()

# ==============================================================================
# Load and preprocess
# ==============================================================================
print("Loading...")

# Load MEG
meg_raw = mne.io.read_raw_fif(MEG_FILE, preload=True, verbose=False)
meg_sfreq = meg_raw.info['sfreq']
aux_channel = "MISC 007"
meg_audio_ch = meg_raw.copy().pick_channels([aux_channel])
meg_audio = meg_audio_ch.get_data()[0]

# Load external
ext_audio, ext_sfreq = librosa.load(str(AUDIO_FILE), sr=None, mono=True)

meg_duration = len(meg_audio) / meg_sfreq
ext_duration = len(ext_audio) / ext_sfreq

print(f"  MEG: {meg_duration:.2f}s @ {meg_sfreq} Hz")
print(f"  External: {ext_duration:.2f}s @ {ext_sfreq} Hz")

# Preprocess
target_sr = 1000.0
lowpass_freq = 400

# Filter MEG
meg_nyq = meg_sfreq / 2
sos = signal.butter(4, lowpass_freq / meg_nyq, btype="low", output="sos")
meg_filtered = signal.sosfilt(sos, meg_audio)

# Filter and resample external
ext_nyq = ext_sfreq / 2
sos = signal.butter(4, lowpass_freq / ext_nyq, btype="low", output="sos")
ext_filtered = signal.sosfilt(sos, ext_audio)
ext_resampled = librosa.resample(ext_filtered, orig_sr=ext_sfreq, target_sr=target_sr)

# Compute envelopes
meg_env = np.abs(hilbert(meg_filtered))
ext_env = np.abs(hilbert(ext_resampled))

# Lowpass filter envelopes at 15 Hz
env_lowpass = 15
nyq = target_sr / 2
sos_env = signal.butter(3, env_lowpass / nyq, btype="low", output="sos")
meg_env_filt = signal.sosfilt(sos_env, meg_env)
ext_env_filt = signal.sosfilt(sos_env, ext_env)

# Normalize
meg_norm = (meg_env_filt - np.mean(meg_env_filt)) / np.std(meg_env_filt)
ext_norm = (ext_env_filt - np.mean(ext_env_filt)) / np.std(ext_env_filt)

print(f"  ✓ Preprocessed")
print()

# ==============================================================================
# Cross-correlation
# ==============================================================================
print("Computing cross-correlation...")

# Full cross-correlation
xcorr = correlate(meg_norm, ext_norm, mode="full", method="fft")
lags = np.arange(len(xcorr)) - (len(ext_norm) - 1)
lag_times = lags / target_sr

# Realistic search: -30s to +30s
max_lag_s = 30.0
max_lag_samples = int(max_lag_s * target_sr)
zero_lag_idx = len(ext_norm) - 1

search_start = max(0, zero_lag_idx - max_lag_samples)
search_end = min(len(xcorr), zero_lag_idx + max_lag_samples + 1)

search_region = xcorr[search_start:search_end]
peak_idx_local = np.argmax(search_region)
peak_idx_global = search_start + peak_idx_local

offset_s = lag_times[peak_idx_global]
correlation = xcorr[peak_idx_global] / len(ext_norm)

print(f"  Offset: {offset_s:.4f}s")
print(f"  Correlation: {correlation:.4f}")
print()

# ==============================================================================
# FULL TIME SERIES VISUALIZATION
# ==============================================================================
print("Creating full time series plots...")

meg_t = np.arange(len(meg_env_filt)) / target_sr
ext_t = np.arange(len(ext_env_filt)) / target_sr

# Create time vector for aligned external audio
# Positive offset means external starts AFTER MEG, so ADD offset to external time
ext_t_aligned = ext_t + offset_s

# Normalize envelopes for plotting
meg_env_norm = meg_env_filt / np.max(meg_env_filt)
ext_env_norm = ext_env_filt / np.max(ext_env_filt)

# Plot 1: Full overview
fig, axes = plt.subplots(3, 1, figsize=(20, 12))

# Raw waveforms (downsampled for visibility)
downsample = 100
axes[0].plot(meg_t[::downsample], meg_audio[::downsample],
             alpha=0.5, linewidth=0.3, label='MEG raw audio', color='blue')
axes[0].set_ylabel('Amplitude')
axes[0].set_title(f'MEG Auxiliary Audio (MISC 007) - Full {meg_duration:.1f}s')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, meg_duration)

# External raw (downsampled)
ext_t_full = np.arange(len(ext_audio)) / ext_sfreq
downsample_ext = int(ext_sfreq / 10)  # Downsample to ~10 Hz for plotting
axes[1].plot(ext_t_full[::downsample_ext], ext_audio[::downsample_ext],
             alpha=0.5, linewidth=0.3, label='External raw audio', color='orange')
axes[1].set_ylabel('Amplitude')
axes[1].set_title(f'External Audio - Full {ext_duration:.1f}s (NOT aligned)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, ext_duration)

# Envelopes - BEFORE alignment
downsample_env = 10
axes[2].plot(meg_t[::downsample_env], meg_env_norm[::downsample_env],
             alpha=0.7, linewidth=0.5, label='MEG envelope', color='blue')
axes[2].plot(ext_t[::downsample_env], ext_env_norm[::downsample_env],
             alpha=0.7, linewidth=0.5, label='External envelope (NOT aligned)', color='orange')
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Normalized Envelope')
axes[2].set_title('Amplitude Envelopes - BEFORE Alignment')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fullview_before_alignment.png", dpi=100)
plt.close()
print(f"  ✓ Saved: {OUTPUT_DIR}/fullview_before_alignment.png")

# Plot 2: Aligned view
fig, axes = plt.subplots(4, 1, figsize=(20, 16))

# Envelope - AFTER alignment (full view)
axes[0].plot(meg_t[::downsample_env], meg_env_norm[::downsample_env],
             alpha=0.7, linewidth=0.5, label='MEG envelope', color='blue')
axes[0].plot(ext_t_aligned[::downsample_env], ext_env_norm[::downsample_env],
             alpha=0.7, linewidth=0.5, label=f'External envelope (aligned, offset={offset_s:.2f}s)', color='orange')
axes[0].set_ylabel('Normalized Envelope')
axes[0].set_title(f'AFTER Alignment - Full View (offset = {offset_s:.2f}s, correlation = {correlation:.3f})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].axvline(offset_s, color='red', linestyle='--', alpha=0.3, linewidth=1)
axes[0].text(offset_s, 0.9, f' External starts here ({offset_s:.1f}s)',
             color='red', fontsize=10)

# First 60 seconds after alignment starts
start1, end1 = max(0, offset_s), min(meg_duration, offset_s + 60)
mask_meg_1 = (meg_t >= start1) & (meg_t <= end1)
mask_ext_1 = (ext_t_aligned >= start1) & (ext_t_aligned <= end1)
axes[1].plot(meg_t[mask_meg_1], meg_env_norm[mask_meg_1],
             alpha=0.8, linewidth=1, label='MEG', color='blue')
axes[1].plot(ext_t_aligned[mask_ext_1], ext_env_norm[mask_ext_1],
             alpha=0.8, linewidth=1, label='External (aligned)', color='orange')
axes[1].set_ylabel('Normalized Envelope')
axes[1].set_title(f'First 60s after alignment ({start1:.1f}-{end1:.1f}s)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(start1, end1)

# Middle 60 seconds
mid_point = (meg_duration + offset_s) / 2
start2, end2 = mid_point - 30, mid_point + 30
mask_meg_2 = (meg_t >= start2) & (meg_t <= end2)
mask_ext_2 = (ext_t_aligned >= start2) & (ext_t_aligned <= end2)
axes[2].plot(meg_t[mask_meg_2], meg_env_norm[mask_meg_2],
             alpha=0.8, linewidth=1, label='MEG', color='blue')
axes[2].plot(ext_t_aligned[mask_ext_2], ext_env_norm[mask_ext_2],
             alpha=0.8, linewidth=1, label='External (aligned)', color='orange')
axes[2].set_ylabel('Normalized Envelope')
axes[2].set_title(f'Middle 60s ({start2:.1f}-{end2:.1f}s)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim(start2, end2)

# Last 60 seconds before external ends
ext_end_time = ext_duration + offset_s
start3 = max(0, ext_end_time - 60)
end3 = min(meg_duration, ext_end_time)
mask_meg_3 = (meg_t >= start3) & (meg_t <= end3)
mask_ext_3 = (ext_t_aligned >= start3) & (ext_t_aligned <= end3)
axes[3].plot(meg_t[mask_meg_3], meg_env_norm[mask_meg_3],
             alpha=0.8, linewidth=1, label='MEG', color='blue')
axes[3].plot(ext_t_aligned[mask_ext_3], ext_env_norm[mask_ext_3],
             alpha=0.8, linewidth=1, label='External (aligned)', color='orange')
axes[3].set_xlabel('Time (s)')
axes[3].set_ylabel('Normalized Envelope')
axes[3].set_title(f'Last 60s before external ends ({start3:.1f}-{end3:.1f}s)')
axes[3].legend()
axes[3].grid(True, alpha=0.3)
axes[3].set_xlim(start3, end3)
axes[3].axvline(ext_end_time, color='red', linestyle='--', alpha=0.3, linewidth=1)
axes[3].text(ext_end_time, 0.9, f' External ends ({ext_end_time:.1f}s)',
             color='red', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fullview_after_alignment.png", dpi=100)
plt.close()
print(f"  ✓ Saved: {OUTPUT_DIR}/fullview_after_alignment.png")

# Plot 3: Check if we're using the right channel - show all MISC channels
print()
print("Checking all MISC channels in MEG file...")
misc_channels = [ch for ch in meg_raw.ch_names if 'MISC' in ch]
print(f"  Found {len(misc_channels)} MISC channels: {misc_channels}")

if len(misc_channels) > 1:
    fig, axes = plt.subplots(len(misc_channels), 1, figsize=(20, 3*len(misc_channels)))
    if len(misc_channels) == 1:
        axes = [axes]

    for idx, ch in enumerate(misc_channels):
        ch_data = meg_raw.copy().pick_channels([ch]).get_data()[0]
        ch_t = np.arange(len(ch_data)) / meg_sfreq

        # Downsample for plotting
        downsample = 100
        axes[idx].plot(ch_t[::downsample], ch_data[::downsample],
                      alpha=0.5, linewidth=0.3, color='blue')
        axes[idx].set_ylabel('Amplitude')
        axes[idx].set_title(f'{ch} - Full recording')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim(0, meg_duration)

        # Calculate RMS to identify active channels
        rms = np.sqrt(np.mean(ch_data**2))
        axes[idx].text(0.02, 0.95, f'RMS: {rms:.6f}',
                      transform=axes[idx].transAxes,
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "all_misc_channels.png", dpi=100)
    plt.close()
    print(f"  ✓ Saved: {OUTPUT_DIR}/all_misc_channels.png")

print()
print("=" * 70)
print("FULL VIEW COMPLETE")
print("=" * 70)
print()
print(f"Offset: {offset_s:.4f}s")
print(f"Correlation: {correlation:.4f}")
print()
print("Timeline:")
print(f"  MEG recording: 0s to {meg_duration:.1f}s")
print(f"  External audio: {offset_s:.1f}s to {ext_duration + offset_s:.1f}s (after alignment)")
print(f"  Overlap: {max(0, offset_s):.1f}s to {min(meg_duration, ext_duration + offset_s):.1f}s")
print()
print("Files saved:")
print(f"  - {OUTPUT_DIR}/fullview_before_alignment.png")
print(f"  - {OUTPUT_DIR}/fullview_after_alignment.png")
print(f"  - {OUTPUT_DIR}/all_misc_channels.png")
print("=" * 70)
