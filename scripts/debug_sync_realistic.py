#!/usr/bin/env python3
"""
Debug script with REALISTIC search window for audio-MEG synchronization.
User reports visual inspection shows external audio starts ~10s after MEG.
This script restricts cross-correlation search to realistic range: -30s to +30s.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import mne
import librosa
from scipy import signal
from scipy.signal import hilbert, correlate
from pathlib import Path
import sys

# Hardcoded paths for sub-01 run-01
SUBJECT = "sub-01"
RUN = 1

MEG_FILE = Path("/Users/em18033/Library/CloudStorage/OneDrive-SharedLibraries-MacquarieUniversity/Natural_Conversations_study - Documents/analysis/natural-conversations-bids/derivatives/mne-bids-pipeline-20240628/sub-01/meg/sub-01_task-conversation_run-01_proc-clean_raw.fif")
AUDIO_FILE = Path("/Users/em18033/Library/CloudStorage/OneDrive-AUTUniversity/Projects/Conversational_AI/Archive/data/audios/G01/console_mic_B1.wav")
OUTPUT_DIR = Path("outputs/sync_debug")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("REALISTIC SYNC WITH RESTRICTED SEARCH WINDOW")
print("=" * 70)
print(f"Subject: {SUBJECT}, Run: {RUN}")
print()

# ==============================================================================
# Load and preprocess
# ==============================================================================
print("Loading and preprocessing...")
print("-" * 70)

# Load MEG
meg_raw = mne.io.read_raw_fif(MEG_FILE, preload=True, verbose=False)
meg_sfreq = meg_raw.info['sfreq']
aux_channel = "MISC 007"
meg_audio_ch = meg_raw.copy().pick_channels([aux_channel])
meg_audio = meg_audio_ch.get_data()[0]

# Load external
ext_audio, ext_sfreq = librosa.load(str(AUDIO_FILE), sr=None, mono=True)

print(f"  MEG: {len(meg_audio)/meg_sfreq:.2f}s @ {meg_sfreq} Hz")
print(f"  External: {len(ext_audio)/ext_sfreq:.2f}s @ {ext_sfreq} Hz")

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

print(f"  ✓ Preprocessed and normalized")
print()

# ==============================================================================
# Cross-correlation with REALISTIC search window
# ==============================================================================
print("Computing cross-correlation...")
print("-" * 70)

# Compute full cross-correlation
xcorr = correlate(meg_norm, ext_norm, mode="full", method="fft")
lags = np.arange(len(xcorr)) - (len(ext_norm) - 1)
lag_times = lags / target_sr

# REALISTIC search: -30s to +30s (expecting positive offset ~10s)
max_lag_s = 30.0
max_lag_samples = int(max_lag_s * target_sr)

# Find center (zero lag)
zero_lag_idx = len(ext_norm) - 1

# Define search region
search_start = zero_lag_idx - max_lag_samples
search_end = zero_lag_idx + max_lag_samples + 1

# Ensure valid indices
search_start = max(0, search_start)
search_end = min(len(xcorr), search_end)

# Extract search region
search_region = xcorr[search_start:search_end]
search_lags = lag_times[search_start:search_end]

# Find peak in search region
peak_idx_local = np.argmax(search_region)
peak_idx_global = search_start + peak_idx_local

offset_s = lag_times[peak_idx_global]
correlation = xcorr[peak_idx_global] / len(ext_norm)

print(f"  Search window: {-max_lag_s:.1f}s to +{max_lag_s:.1f}s")
print(f"  ✓ Peak found at: {offset_s:.4f}s")
print(f"  Correlation: {correlation:.4f}")
print(f"  Quality: {'EXCELLENT' if abs(correlation) > 0.7 else 'GOOD' if abs(correlation) > 0.5 else 'MODERATE' if abs(correlation) > 0.3 else 'POOR'}")
print()

# Also check what OLD method found (unrestricted)
peak_idx_old = np.argmax(xcorr)
offset_s_old = lag_times[peak_idx_old]
correlation_old = xcorr[peak_idx_old] / len(ext_norm)

print(f"  OLD unrestricted peak: {offset_s_old:.4f}s, corr={correlation_old:.4f}")
print()

# Show top 5 peaks within search region
search_peaks = np.argsort(search_region)[-5:][::-1]
print("  TOP 5 PEAKS in ±30s window:")
for i, idx in enumerate(search_peaks, 1):
    global_idx = search_start + idx
    lag_s = lag_times[global_idx]
    corr = xcorr[global_idx] / len(ext_norm)
    print(f"    {i}. Lag: {lag_s:+7.2f}s, Correlation: {corr:+.4f}")
print()

# ==============================================================================
# Visualizations
# ==============================================================================
print("Generating plots...")
print("-" * 70)

# Plot 1: Cross-correlation with restricted search
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Full xcorr (zoomed to ±60s)
zoom_mask = (lag_times >= -60) & (lag_times <= 60)
axes[0].plot(lag_times[zoom_mask], xcorr[zoom_mask] / len(ext_norm), linewidth=0.8, alpha=0.7)
axes[0].axvline(offset_s, color='green', linestyle='--', linewidth=2,
                label=f'NEW (±30s): {offset_s:.2f}s, corr={correlation:.3f}')
axes[0].axvline(offset_s_old, color='red', linestyle=':', linewidth=2,
                label=f'OLD (unrestricted): {offset_s_old:.2f}s, corr={correlation_old:.3f}')
axes[0].axhline(0, color='black', linestyle=':', linewidth=0.5)
axes[0].axvspan(-max_lag_s, max_lag_s, alpha=0.1, color='green', label='Search window')
axes[0].set_ylabel('Normalized Cross-Correlation')
axes[0].set_title('Cross-Correlation Function (±60s view)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(-60, 60)

# Zoomed to search region
axes[1].plot(search_lags, search_region / len(ext_norm), linewidth=1, alpha=0.7)
axes[1].axvline(offset_s, color='green', linestyle='--', linewidth=2,
                label=f'Peak: {offset_s:.2f}s')
axes[1].axhline(0, color='black', linestyle=':', linewidth=0.5)
# Mark top 5 peaks
for idx in search_peaks[:3]:
    global_idx = search_start + idx
    axes[1].axvline(lag_times[global_idx], color='purple', linestyle=':', alpha=0.5)
axes[1].set_xlabel('Lag (seconds)')
axes[1].set_ylabel('Normalized Cross-Correlation')
axes[1].set_title(f'Cross-Correlation in Search Window (±{max_lag_s}s)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "realistic_sync_xcorr.png", dpi=150)
plt.close()
print(f"  ✓ Saved: {OUTPUT_DIR}/realistic_sync_xcorr.png")

# Plot 2: Aligned envelopes (first 60 seconds)
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

meg_t = np.arange(len(meg_env_filt)) / target_sr
ext_t = np.arange(len(ext_env_filt)) / target_sr
ext_t_aligned = ext_t - offset_s

duration = 60

# Raw envelopes (not aligned)
meg_mask = meg_t <= duration
ext_mask = ext_t <= duration
axes[0].plot(meg_t[meg_mask], meg_env_filt[meg_mask] / np.max(meg_env_filt),
             alpha=0.7, label='MEG', linewidth=1)
axes[0].plot(ext_t[ext_mask], ext_env_filt[ext_mask] / np.max(ext_env_filt),
             alpha=0.7, label='External (NOT aligned)', linewidth=1)
axes[0].set_ylabel('Normalized Envelope')
axes[0].set_title('BEFORE Alignment')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, duration)

# Aligned envelopes - full view
ext_mask_aligned = (ext_t_aligned >= 0) & (ext_t_aligned <= duration)
axes[1].plot(meg_t[meg_mask], meg_env_filt[meg_mask] / np.max(meg_env_filt),
             alpha=0.7, label='MEG', linewidth=1)
axes[1].plot(ext_t_aligned[ext_mask_aligned], ext_env_filt[ext_mask_aligned] / np.max(ext_env_filt),
             alpha=0.7, label=f'External (aligned, offset={offset_s:.2f}s)', linewidth=1)
axes[1].set_ylabel('Normalized Envelope')
axes[1].set_title(f'AFTER Alignment (offset = {offset_s:.2f}s, correlation = {correlation:.3f})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, duration)

# Zoomed view (20-40s)
zoom_start, zoom_end = 20, 40
zoom_mask_meg = (meg_t >= zoom_start) & (meg_t <= zoom_end)
zoom_mask_ext = (ext_t_aligned >= zoom_start) & (ext_t_aligned <= zoom_end)
axes[2].plot(meg_t[zoom_mask_meg], meg_env_filt[zoom_mask_meg] / np.max(meg_env_filt),
             alpha=0.7, label='MEG', linewidth=1.5)
axes[2].plot(ext_t_aligned[zoom_mask_ext], ext_env_filt[zoom_mask_ext] / np.max(ext_env_filt),
             alpha=0.7, label='External (aligned)', linewidth=1.5)
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Normalized Envelope')
axes[2].set_title(f'Zoomed View ({zoom_start}-{zoom_end}s)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "realistic_sync_alignment.png", dpi=150)
plt.close()
print(f"  ✓ Saved: {OUTPUT_DIR}/realistic_sync_alignment.png")

print()
print("=" * 70)
print("REALISTIC SYNC COMPLETE")
print("=" * 70)
print(f"Offset: {offset_s:.4f} seconds")
print(f"Correlation: {correlation:.4f}")
print(f"")
print(f"This offset means the external audio starts {abs(offset_s):.2f}s")
print(f"{'AFTER' if offset_s > 0 else 'BEFORE'} the MEG recording.")
print()
print(f"Compare to OLD unrestricted method: {offset_s_old:.2f}s (correlation={correlation_old:.3f})")
print(f"Difference: {abs(offset_s - offset_s_old):.2f}s")
print()
print("Check the alignment plots to verify this looks correct!")
print("=" * 70)
