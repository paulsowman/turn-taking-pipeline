#!/usr/bin/env python3
"""
Verify sync alignment with better visualization showing middle segments of recording.
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
print("SYNC VERIFICATION WITH IMPROVED VISUALIZATION")
print("=" * 70)
print(f"Subject: {SUBJECT}, Run: {RUN}")
print()

# ==============================================================================
# Load and preprocess
# ==============================================================================
print("Loading and preprocessing...")

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
# Cross-correlation with realistic search window
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
# Better visualizations - show multiple segments
# ==============================================================================
print("Generating improved visualizations...")

meg_t = np.arange(len(meg_env_filt)) / target_sr
ext_t = np.arange(len(ext_env_filt)) / target_sr
ext_t_aligned = ext_t - offset_s

# Normalize envelopes for plotting
meg_env_norm = meg_env_filt / np.max(meg_env_filt)
ext_env_norm = ext_env_filt / np.max(ext_env_filt)

# Define viewing windows - start, middle, end
# Make sure we're looking at regions where both signals exist after alignment
overlap_start = max(0, offset_s)
overlap_end = min(meg_duration, ext_duration + offset_s)
overlap_duration = overlap_end - overlap_start

# Select 3 windows of 30 seconds each
window_dur = 30.0

# Early segment (20% through overlap)
seg1_start = overlap_start + 0.2 * overlap_duration
seg1_end = seg1_start + window_dur

# Middle segment (50% through overlap)
seg2_start = overlap_start + 0.5 * overlap_duration
seg2_end = seg2_start + window_dur

# Late segment (80% through overlap)
seg3_start = overlap_start + 0.8 * overlap_duration
seg3_end = seg3_start + window_dur

# Create figure with 4 subplots
fig, axes = plt.subplots(4, 1, figsize=(18, 16))

# Plot 1: Cross-correlation
search_lags = lag_times[search_start:search_end]
axes[0].plot(search_lags, search_region / len(ext_norm), linewidth=1, alpha=0.7)
axes[0].axvline(offset_s, color='red', linestyle='--', linewidth=2,
                label=f'Peak: {offset_s:.2f}s, corr={correlation:.3f}')
axes[0].axhline(0, color='black', linestyle=':', linewidth=0.5)
axes[0].set_xlabel('Lag (seconds)')
axes[0].set_ylabel('Normalized Cross-Correlation')
axes[0].set_title(f'Cross-Correlation (search window: ±{max_lag_s}s)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Early segment
mask_meg = (meg_t >= seg1_start) & (meg_t <= seg1_end)
mask_ext = (ext_t_aligned >= seg1_start) & (ext_t_aligned <= seg1_end)
axes[1].plot(meg_t[mask_meg], meg_env_norm[mask_meg],
             alpha=0.8, label='MEG', linewidth=1.5, color='blue')
axes[1].plot(ext_t_aligned[mask_ext], ext_env_norm[mask_ext],
             alpha=0.8, label='External (aligned)', linewidth=1.5, color='orange')
axes[1].set_ylabel('Normalized Envelope')
axes[1].set_title(f'Early Segment ({seg1_start:.0f}-{seg1_end:.0f}s)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(seg1_start, seg1_end)

# Plot 3: Middle segment
mask_meg = (meg_t >= seg2_start) & (meg_t <= seg2_end)
mask_ext = (ext_t_aligned >= seg2_start) & (ext_t_aligned <= seg2_end)
axes[2].plot(meg_t[mask_meg], meg_env_norm[mask_meg],
             alpha=0.8, label='MEG', linewidth=1.5, color='blue')
axes[2].plot(ext_t_aligned[mask_ext], ext_env_norm[mask_ext],
             alpha=0.8, label='External (aligned)', linewidth=1.5, color='orange')
axes[2].set_ylabel('Normalized Envelope')
axes[2].set_title(f'Middle Segment ({seg2_start:.0f}-{seg2_end:.0f}s)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim(seg2_start, seg2_end)

# Plot 4: Late segment
mask_meg = (meg_t >= seg3_start) & (meg_t <= seg3_end)
mask_ext = (ext_t_aligned >= seg3_start) & (ext_t_aligned <= seg3_end)
axes[3].plot(meg_t[mask_meg], meg_env_norm[mask_meg],
             alpha=0.8, label='MEG', linewidth=1.5, color='blue')
axes[3].plot(ext_t_aligned[mask_ext], ext_env_norm[mask_ext],
             alpha=0.8, label='External (aligned)', linewidth=1.5, color='orange')
axes[3].set_xlabel('Time (s)')
axes[3].set_ylabel('Normalized Envelope')
axes[3].set_title(f'Late Segment ({seg3_start:.0f}-{seg3_end:.0f}s)')
axes[3].legend()
axes[3].grid(True, alpha=0.3)
axes[3].set_xlim(seg3_start, seg3_end)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sync_verification_segments.png", dpi=150)
plt.close()
print(f"  ✓ Saved: {OUTPUT_DIR}/sync_verification_segments.png")

# Also create a zoomed view of just one 10-second window in the middle
fig, ax = plt.subplots(1, 1, figsize=(18, 6))

zoom_start = seg2_start + 10  # 10s into the middle segment
zoom_end = zoom_start + 10

mask_meg = (meg_t >= zoom_start) & (meg_t <= zoom_end)
mask_ext = (ext_t_aligned >= zoom_start) & (ext_t_aligned <= zoom_end)

ax.plot(meg_t[mask_meg], meg_env_norm[mask_meg],
        alpha=0.8, label='MEG envelope', linewidth=2, color='blue')
ax.plot(ext_t_aligned[mask_ext], ext_env_norm[mask_ext],
        alpha=0.8, label='External envelope (aligned)', linewidth=2, color='orange')
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Normalized Envelope', fontsize=12)
ax.set_title(f'10-Second Zoom ({zoom_start:.0f}-{zoom_end:.0f}s) - Offset = {offset_s:.2f}s, Corr = {correlation:.3f}',
             fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim(zoom_start, zoom_end)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sync_verification_zoom.png", dpi=150)
plt.close()
print(f"  ✓ Saved: {OUTPUT_DIR}/sync_verification_zoom.png")

print()
print("=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
print(f"Offset: {offset_s:.4f} seconds")
print(f"Correlation: {correlation:.4f}")
print()
print(f"Overlap region: {overlap_start:.1f}s to {overlap_end:.1f}s ({overlap_duration:.1f}s)")
print()
print("Check the plots to verify alignment quality:")
print(f"  - {OUTPUT_DIR}/sync_verification_segments.png")
print(f"  - {OUTPUT_DIR}/sync_verification_zoom.png")
print("=" * 70)
