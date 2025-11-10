#!/usr/bin/env python3
"""
Test synchronization for sub-05 run-2 to check if issue is run-specific.
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

# Paths for sub-05 run-02
SUBJECT = "sub-05"
RUN = 2

MEG_FILE = Path("/Users/em18033/Library/CloudStorage/OneDrive-SharedLibraries-MacquarieUniversity/Natural_Conversations_study - Documents/analysis/natural-conversations-bids/derivatives/mne-bids-pipeline-20240628/sub-05/meg/sub-05_task-conversation_run-02_proc-clean_raw.fif")
AUDIO_FILE = Path("/Users/em18033/Library/CloudStorage/OneDrive-AUTUniversity/Projects/Conversational_AI/Archive/data/audios/G05/console_mic_B2.wav")
OUTPUT_DIR = Path("outputs/sync_debug")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print(f"TESTING SYNC FOR {SUBJECT} RUN-{RUN:02d}")
print("=" * 70)
print()

# Check files exist
if not MEG_FILE.exists():
    print(f"ERROR: MEG file not found: {MEG_FILE}")
    exit(1)
if not AUDIO_FILE.exists():
    print(f"ERROR: Audio file not found: {AUDIO_FILE}")
    exit(1)

# ==============================================================================
# Load and preprocess
# ==============================================================================
print("Loading...")

meg_raw = mne.io.read_raw_fif(MEG_FILE, preload=True, verbose=False)
meg_sfreq = meg_raw.info['sfreq']
aux_channel = "MISC 007"
meg_audio_ch = meg_raw.copy().pick_channels([aux_channel])
meg_audio = meg_audio_ch.get_data()[0]

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
# Cross-correlation with multiple search windows
# ==============================================================================
print("Computing cross-correlation...")

# Full cross-correlation
xcorr = correlate(meg_norm, ext_norm, mode="full", method="fft")
lags = np.arange(len(xcorr)) - (len(ext_norm) - 1)
lag_times = lags / target_sr

# Try different search windows
search_windows = [30, 60, 120, 300]
results = []

for max_lag_s in search_windows:
    max_lag_samples = int(max_lag_s * target_sr)
    zero_lag_idx = len(ext_norm) - 1

    search_start = max(0, zero_lag_idx - max_lag_samples)
    search_end = min(len(xcorr), zero_lag_idx + max_lag_samples + 1)

    search_region = xcorr[search_start:search_end]
    peak_idx_local = np.argmax(search_region)
    peak_idx_global = search_start + peak_idx_local

    offset_s = lag_times[peak_idx_global]
    correlation = xcorr[peak_idx_global] / len(ext_norm)

    results.append({
        'window': max_lag_s,
        'offset': offset_s,
        'correlation': correlation
    })

    print(f"  Window ±{max_lag_s}s: offset={offset_s:+7.2f}s, corr={correlation:.4f}")

print()

# Use best result
best = results[-1]
offset_s = best['offset']
correlation = best['correlation']

print(f"Best offset: {offset_s:.2f}s (correlation: {correlation:.4f})")
print(f"Quality: {'EXCELLENT' if abs(correlation) > 0.7 else 'GOOD' if abs(correlation) > 0.5 else 'MODERATE' if abs(correlation) > 0.3 else 'POOR'}")
print()

# ==============================================================================
# Visualization
# ==============================================================================
print("Generating plots...")

meg_t = np.arange(len(meg_env_filt)) / target_sr
ext_t = np.arange(len(ext_env_filt)) / target_sr

# Add offset to external time
ext_t_aligned = ext_t + offset_s

# Normalize envelopes for plotting
meg_env_norm = meg_env_filt / np.max(meg_env_filt)
ext_env_norm = ext_env_filt / np.max(ext_env_filt)

# Create plot
fig, axes = plt.subplots(4, 1, figsize=(20, 16))

# Plot 1: Full aligned view
downsample = 10
axes[0].plot(meg_t[::downsample], meg_env_norm[::downsample],
             alpha=0.7, linewidth=0.5, label='MEG envelope', color='blue')
axes[0].plot(ext_t_aligned[::downsample], ext_env_norm[::downsample],
             alpha=0.7, linewidth=0.5, label=f'External envelope (offset={offset_s:.2f}s)', color='orange')
axes[0].set_ylabel('Normalized Envelope')
axes[0].set_title(f'{SUBJECT} Run-{RUN:02d} - Full View (offset = {offset_s:.2f}s, correlation = {correlation:.3f})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
if offset_s > 0:
    axes[0].axvline(offset_s, color='red', linestyle='--', alpha=0.3, linewidth=1)

# Calculate overlap region
overlap_start = max(0, offset_s)
overlap_end = min(meg_duration, ext_duration + offset_s)

# Plot 2: First 60s after external starts
start1 = overlap_start
end1 = min(meg_duration, start1 + 60)
mask_meg_1 = (meg_t >= start1) & (meg_t <= end1)
mask_ext_1 = (ext_t_aligned >= start1) & (ext_t_aligned <= end1)
axes[1].plot(meg_t[mask_meg_1], meg_env_norm[mask_meg_1],
             alpha=0.8, linewidth=1, label='MEG', color='blue')
axes[1].plot(ext_t_aligned[mask_ext_1], ext_env_norm[mask_ext_1],
             alpha=0.8, linewidth=1, label='External (aligned)', color='orange')
axes[1].set_ylabel('Normalized Envelope')
axes[1].set_title(f'First 60s of overlap ({start1:.1f}-{end1:.1f}s)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(start1, end1)

# Plot 3: Middle 60s
mid_point = (overlap_start + overlap_end) / 2
start2 = mid_point - 30
end2 = mid_point + 30
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

# Plot 4: 10s zoom in middle
zoom_start = mid_point - 5
zoom_end = mid_point + 5
mask_meg_3 = (meg_t >= zoom_start) & (meg_t <= zoom_end)
mask_ext_3 = (ext_t_aligned >= zoom_start) & (ext_t_aligned <= zoom_end)
axes[3].plot(meg_t[mask_meg_3], meg_env_norm[mask_meg_3],
             alpha=0.8, linewidth=2, label='MEG', color='blue')
axes[3].plot(ext_t_aligned[mask_ext_3], ext_env_norm[mask_ext_3],
             alpha=0.8, linewidth=2, label='External (aligned)', color='orange')
axes[3].set_xlabel('Time (s)')
axes[3].set_ylabel('Normalized Envelope')
axes[3].set_title(f'10s Zoom ({zoom_start:.1f}-{zoom_end:.1f}s)')
axes[3].legend()
axes[3].grid(True, alpha=0.3)
axes[3].set_xlim(zoom_start, zoom_end)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"sync_test_{SUBJECT}_run{RUN:02d}.png", dpi=100)
plt.close()
print(f"  ✓ Saved: {OUTPUT_DIR}/sync_test_{SUBJECT}_run{RUN:02d}.png")

print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)
print(f"Offset: {offset_s:.4f}s")
print(f"Correlation: {correlation:.4f}")
print(f"Quality: {'EXCELLENT' if abs(correlation) > 0.7 else 'GOOD' if abs(correlation) > 0.5 else 'MODERATE' if abs(correlation) > 0.3 else 'POOR'}")
print()
print(f"Timeline:")
print(f"  MEG recording: 0s to {meg_duration:.1f}s")
print(f"  External audio: {offset_s:.1f}s to {ext_duration + offset_s:.1f}s")
print(f"  Overlap: {overlap_start:.1f}s to {overlap_end:.1f}s ({overlap_end - overlap_start:.1f}s)")
print()
print(f"Plot: {OUTPUT_DIR}/sync_test_{SUBJECT}_run{RUN:02d}.png")
print("=" * 70)
