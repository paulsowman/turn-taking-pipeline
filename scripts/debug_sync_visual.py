#!/usr/bin/env python3
"""
Diagnostic script to debug audio-MEG synchronization step-by-step with visualizations.
Breaks down the sync process to identify exactly where it's failing.
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
print("AUDIO-MEG SYNCHRONIZATION DEBUG")
print("=" * 70)
print(f"Subject: {SUBJECT}")
print(f"Run: {RUN}")
print(f"MEG file: {MEG_FILE}")
print(f"Audio file: {AUDIO_FILE}")
print(f"Output: {OUTPUT_DIR}")
print("=" * 70)
print()

# ==============================================================================
# STEP 1: Load MEG auxiliary audio
# ==============================================================================
print("STEP 1: Loading MEG auxiliary audio...")
print("-" * 70)

meg_raw = mne.io.read_raw_fif(MEG_FILE, preload=True, verbose=False)
meg_sfreq = meg_raw.info['sfreq']
print(f"  MEG sampling rate: {meg_sfreq} Hz")
print(f"  MEG duration: {meg_raw.times[-1]:.2f} s")

# Extract auxiliary channel
aux_channel = "MISC 007"
if aux_channel not in meg_raw.ch_names:
    print(f"  ERROR: Channel {aux_channel} not found!")
    print(f"  Available MISC channels: {[ch for ch in meg_raw.ch_names if 'MISC' in ch]}")
    sys.exit(1)

meg_audio_ch = meg_raw.copy().pick_channels([aux_channel])
meg_audio = meg_audio_ch.get_data()[0]
print(f"  ✓ Extracted {aux_channel}: {len(meg_audio)} samples")
print(f"  MEG audio stats: mean={np.mean(meg_audio):.6f}, std={np.std(meg_audio):.6f}, max={np.max(np.abs(meg_audio)):.6f}")
print()

# ==============================================================================
# STEP 2: Load external audio
# ==============================================================================
print("STEP 2: Loading external audio...")
print("-" * 70)

ext_audio, ext_sfreq = librosa.load(str(AUDIO_FILE), sr=None, mono=True)
print(f"  External sampling rate: {ext_sfreq} Hz")
print(f"  External duration: {len(ext_audio)/ext_sfreq:.2f} s")
print(f"  External audio stats: mean={np.mean(ext_audio):.6f}, std={np.std(ext_audio):.6f}, max={np.max(np.abs(ext_audio)):.6f}")
print()

# ==============================================================================
# STEP 3: Preprocessing - Lowpass filter and resample
# ==============================================================================
print("STEP 3: Preprocessing (lowpass filter + resample)...")
print("-" * 70)

# Lowpass filter to 400 Hz
lowpass_freq = 400
target_sr = 1000.0  # Resample to MEG rate

# Filter MEG audio
meg_nyq = meg_sfreq / 2
if lowpass_freq < meg_nyq:
    sos = signal.butter(4, lowpass_freq / meg_nyq, btype="low", output="sos")
    meg_filtered = signal.sosfilt(sos, meg_audio)
    print(f"  ✓ MEG audio lowpass filtered at {lowpass_freq} Hz")
else:
    meg_filtered = meg_audio.copy()
    print(f"  MEG audio not filtered (lowpass {lowpass_freq} >= Nyquist {meg_nyq})")

# Filter and resample external audio
ext_nyq = ext_sfreq / 2
if lowpass_freq < ext_nyq:
    sos = signal.butter(4, lowpass_freq / ext_nyq, btype="low", output="sos")
    ext_filtered = signal.sosfilt(sos, ext_audio)
    print(f"  ✓ External audio lowpass filtered at {lowpass_freq} Hz")
else:
    ext_filtered = ext_audio.copy()

# Resample external audio to MEG rate
if ext_sfreq != target_sr:
    ext_resampled = librosa.resample(ext_filtered, orig_sr=ext_sfreq, target_sr=target_sr)
    print(f"  ✓ External audio resampled: {ext_sfreq} Hz → {target_sr} Hz")
    print(f"    Resampled length: {len(ext_resampled)} samples ({len(ext_resampled)/target_sr:.2f} s)")
else:
    ext_resampled = ext_filtered.copy()

print(f"  Preprocessed MEG stats: mean={np.mean(meg_filtered):.6f}, std={np.std(meg_filtered):.6f}")
print(f"  Preprocessed Ext stats: mean={np.mean(ext_resampled):.6f}, std={np.std(ext_resampled):.6f}")
print()

# ==============================================================================
# STEP 4: Compute amplitude envelopes via Hilbert transform
# ==============================================================================
print("STEP 4: Computing amplitude envelopes...")
print("-" * 70)

# Hilbert transform
meg_analytic = hilbert(meg_filtered)
meg_env = np.abs(meg_analytic)
print(f"  ✓ MEG envelope computed: {len(meg_env)} samples")

ext_analytic = hilbert(ext_resampled)
ext_env = np.abs(ext_analytic)
print(f"  ✓ External envelope computed: {len(ext_env)} samples")

# Lowpass filter envelopes at 15 Hz
env_lowpass = 15
nyq_env = target_sr / 2

sos_env = signal.butter(3, env_lowpass / nyq_env, btype="low", output="sos")
meg_env_filt = signal.sosfilt(sos_env, meg_env)
ext_env_filt = signal.sosfilt(sos_env, ext_env)
print(f"  ✓ Envelopes lowpass filtered at {env_lowpass} Hz")

print(f"  MEG envelope stats: mean={np.mean(meg_env_filt):.6f}, std={np.std(meg_env_filt):.6f}")
print(f"  Ext envelope stats: mean={np.mean(ext_env_filt):.6f}, std={np.std(ext_env_filt):.6f}")
print()

# ==============================================================================
# STEP 5: Normalize signals for cross-correlation
# ==============================================================================
print("STEP 5: Normalizing signals for cross-correlation...")
print("-" * 70)

meg_norm = (meg_env_filt - np.mean(meg_env_filt)) / np.std(meg_env_filt)
ext_norm = (ext_env_filt - np.mean(ext_env_filt)) / np.std(ext_env_filt)

print(f"  MEG normalized: mean={np.mean(meg_norm):.6f}, std={np.std(meg_norm):.6f}")
print(f"  Ext normalized: mean={np.mean(ext_norm):.6f}, std={np.std(ext_norm):.6f}")
print()

# ==============================================================================
# STEP 6: Cross-correlation to find time offset
# ==============================================================================
print("STEP 6: Computing cross-correlation...")
print("-" * 70)

xcorr = correlate(meg_norm, ext_norm, mode="valid", method="fft")
print(f"  ✓ Cross-correlation computed: {len(xcorr)} lags")

# Find peak within ±10 second search window
max_lag_s = 10.0
max_lag_samples = int(max_lag_s * target_sr)

n_xcorr = len(xcorr)
center = n_xcorr // 2
search_start = max(0, center - max_lag_samples)
search_end = min(n_xcorr, center + max_lag_samples)

search_region = xcorr[search_start:search_end]
peak_idx = np.argmax(np.abs(search_region))
peak_idx_global = search_start + peak_idx

lag_samples = peak_idx_global - (len(ext_norm) - 1)
offset_s = lag_samples / target_sr
peak_value = xcorr[peak_idx_global] / len(ext_norm)

print(f"  Peak lag: {lag_samples} samples ({offset_s:.4f} s)")
print(f"  Peak correlation: {peak_value:.4f}")
print()

# ==============================================================================
# STEP 7: Visualizations
# ==============================================================================
print("STEP 7: Generating diagnostic plots...")
print("-" * 70)

# Plot 1: Raw signals (first 10 seconds)
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
duration = min(10, meg_raw.times[-1])
meg_mask = meg_raw.times <= duration
meg_t = meg_raw.times[meg_mask]
ext_t = np.arange(len(ext_audio)) / ext_sfreq
ext_mask = ext_t <= duration

axes[0].plot(meg_t, meg_audio[meg_mask], alpha=0.7, linewidth=0.5, label='MEG aux audio')
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Step 1: Raw MEG Auxiliary Audio (first 10s)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(ext_t[ext_mask], ext_audio[ext_mask], alpha=0.7, linewidth=0.5, label='External audio', color='orange')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Amplitude')
axes[1].set_title('Step 2: Raw External Audio (first 10s)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "step_1-2_raw_audio.png", dpi=150)
plt.close()
print(f"  ✓ Saved: {OUTPUT_DIR}/step_1-2_raw_audio.png")

# Plot 2: Filtered signals
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
duration = min(10, len(meg_filtered)/target_sr)
meg_mask = np.arange(len(meg_filtered)) / target_sr <= duration
ext_mask = np.arange(len(ext_resampled)) / target_sr <= duration
meg_t = np.arange(len(meg_filtered)) / target_sr
ext_t = np.arange(len(ext_resampled)) / target_sr

axes[0].plot(meg_t[meg_mask], meg_filtered[meg_mask], alpha=0.7, linewidth=0.5)
axes[0].set_ylabel('Amplitude')
axes[0].set_title(f'Step 3: MEG Audio (lowpass {lowpass_freq} Hz, first 10s)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(ext_t[ext_mask], ext_resampled[ext_mask], alpha=0.7, linewidth=0.5, color='orange')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Amplitude')
axes[1].set_title(f'Step 3: External Audio (lowpass {lowpass_freq} Hz, resampled to {target_sr} Hz, first 10s)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "step_3_filtered.png", dpi=150)
plt.close()
print(f"  ✓ Saved: {OUTPUT_DIR}/step_3_filtered.png")

# Plot 3: Envelopes
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
duration = min(30, len(meg_env_filt)/target_sr)
meg_mask = np.arange(len(meg_env_filt)) / target_sr <= duration
ext_mask = np.arange(len(ext_env_filt)) / target_sr <= duration
meg_t = np.arange(len(meg_env_filt)) / target_sr
ext_t = np.arange(len(ext_env_filt)) / target_sr

axes[0].plot(meg_t[meg_mask], meg_env_filt[meg_mask], alpha=0.7, label='MEG envelope')
axes[0].set_ylabel('Amplitude')
axes[0].set_title(f'Step 4: MEG Envelope (Hilbert + lowpass {env_lowpass} Hz, first 30s)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(ext_t[ext_mask], ext_env_filt[ext_mask], alpha=0.7, label='External envelope', color='orange')
axes[1].set_ylabel('Amplitude')
axes[1].set_title(f'Step 4: External Envelope (Hilbert + lowpass {env_lowpass} Hz, first 30s)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Overlay (before alignment)
axes[2].plot(meg_t[meg_mask], meg_env_filt[meg_mask] / np.max(meg_env_filt), alpha=0.7, label='MEG (norm)')
axes[2].plot(ext_t[ext_mask], ext_env_filt[ext_mask] / np.max(ext_env_filt), alpha=0.7, label='External (norm)')
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Normalized Amplitude')
axes[2].set_title('Step 4: Envelope Overlay (BEFORE alignment)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "step_4_envelopes.png", dpi=150)
plt.close()
print(f"  ✓ Saved: {OUTPUT_DIR}/step_4_envelopes.png")

# Plot 4: Cross-correlation function
fig, ax = plt.subplots(figsize=(14, 6))
lags = np.arange(len(xcorr)) - (len(ext_norm) - 1)
lag_times = lags / target_sr

ax.plot(lag_times, xcorr / len(ext_norm), linewidth=0.5, alpha=0.7)
ax.axvline(offset_s, color='red', linestyle='--', label=f'Peak at {offset_s:.3f}s (corr={peak_value:.3f})')
ax.axhline(0, color='black', linestyle=':', linewidth=0.5)
ax.set_xlabel('Lag (s)')
ax.set_ylabel('Normalized Cross-Correlation')
ax.set_title('Step 6: Cross-Correlation Function')
ax.set_xlim(-max_lag_s, max_lag_s)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "step_6_xcorr.png", dpi=150)
plt.close()
print(f"  ✓ Saved: {OUTPUT_DIR}/step_6_xcorr.png")

# Plot 5: Aligned envelopes
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
duration = min(30, len(meg_env_filt)/target_sr, (len(ext_env_filt)/target_sr) + offset_s)
meg_mask = (meg_t >= 0) & (meg_t <= duration)
ext_t_aligned = ext_t - offset_s
ext_mask = (ext_t_aligned >= 0) & (ext_t_aligned <= duration)

# Top: Overlaid
axes[0].plot(meg_t[meg_mask], meg_env_filt[meg_mask] / np.max(meg_env_filt), alpha=0.7, label='MEG')
axes[0].plot(ext_t_aligned[ext_mask], ext_env_filt[ext_mask] / np.max(ext_env_filt), alpha=0.7, label='External (aligned)')
axes[0].set_ylabel('Normalized Amplitude')
axes[0].set_title(f'Step 7: Aligned Envelopes (offset = {offset_s:.3f}s, corr = {peak_value:.3f})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Bottom: Zoomed 5-second window
zoom_start = 10
zoom_end = 15
zoom_mask_meg = (meg_t >= zoom_start) & (meg_t <= zoom_end)
zoom_mask_ext = (ext_t_aligned >= zoom_start) & (ext_t_aligned <= zoom_end)

axes[1].plot(meg_t[zoom_mask_meg], meg_env_filt[zoom_mask_meg] / np.max(meg_env_filt), alpha=0.7, label='MEG', linewidth=1.5)
axes[1].plot(ext_t_aligned[zoom_mask_ext], ext_env_filt[zoom_mask_ext] / np.max(ext_env_filt), alpha=0.7, label='External (aligned)', linewidth=1.5)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Normalized Amplitude')
axes[1].set_title(f'Step 7: Zoomed View ({zoom_start}-{zoom_end}s)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "step_7_aligned.png", dpi=150)
plt.close()
print(f"  ✓ Saved: {OUTPUT_DIR}/step_7_aligned.png")

print()
print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
print(f"Result: Offset = {offset_s:.4f} s, Correlation = {peak_value:.4f}")
print(f"Quality: {'EXCELLENT' if peak_value > 0.7 else 'GOOD' if peak_value > 0.5 else 'MODERATE' if peak_value > 0.3 else 'POOR'}")
print(f"All plots saved to: {OUTPUT_DIR}")
print("=" * 70)
