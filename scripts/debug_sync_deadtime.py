#!/usr/bin/env python3
"""
Debug script to investigate why sync offset is -348 seconds (likely correlating with dead time).
This script analyzes where actual speech occurs vs. silence in both MEG and external audio.
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
print("INVESTIGATING DEAD TIME IN AUDIO-MEG SYNCHRONIZATION")
print("=" * 70)
print(f"Subject: {SUBJECT}")
print(f"Run: {RUN}")
print()

# ==============================================================================
# STEP 1: Load both audio signals
# ==============================================================================
print("STEP 1: Loading audio signals...")
print("-" * 70)

# Load MEG
meg_raw = mne.io.read_raw_fif(MEG_FILE, preload=True, verbose=False)
meg_sfreq = meg_raw.info['sfreq']
aux_channel = "MISC 007"
meg_audio_ch = meg_raw.copy().pick_channels([aux_channel])
meg_audio = meg_audio_ch.get_data()[0]
meg_duration = len(meg_audio) / meg_sfreq

# Load external
ext_audio, ext_sfreq = librosa.load(str(AUDIO_FILE), sr=None, mono=True)
ext_duration = len(ext_audio) / ext_sfreq

print(f"  MEG audio: {meg_duration:.2f} seconds @ {meg_sfreq} Hz")
print(f"  External audio: {ext_duration:.2f} seconds @ {ext_sfreq} Hz")
print(f"  Duration difference: {abs(meg_duration - ext_duration):.2f} seconds")
print()

# ==============================================================================
# STEP 2: Compute RMS energy in 1-second windows to identify speech regions
# ==============================================================================
print("STEP 2: Computing RMS energy to identify speech regions...")
print("-" * 70)

def compute_rms_windowed(audio, sfreq, window_s=1.0):
    """Compute RMS energy in non-overlapping windows."""
    window_samples = int(window_s * sfreq)
    n_windows = len(audio) // window_samples

    rms = np.zeros(n_windows)
    for i in range(n_windows):
        start = i * window_samples
        end = start + window_samples
        rms[i] = np.sqrt(np.mean(audio[start:end]**2))

    return rms

meg_rms = compute_rms_windowed(meg_audio, meg_sfreq, window_s=1.0)
ext_rms = compute_rms_windowed(ext_audio, ext_sfreq, window_s=1.0)

meg_time_windows = np.arange(len(meg_rms))
ext_time_windows = np.arange(len(ext_rms))

print(f"  MEG: {len(meg_rms)} 1-second windows")
print(f"  External: {len(ext_rms)} 1-second windows")

# Compute percentile thresholds to identify "active" regions
meg_threshold = np.percentile(meg_rms, 25)  # 25th percentile
ext_threshold = np.percentile(ext_rms, 25)

meg_active = meg_rms > meg_threshold
ext_active = ext_rms > ext_threshold

meg_active_pct = 100 * np.sum(meg_active) / len(meg_active)
ext_active_pct = 100 * np.sum(ext_active) / len(ext_active)

print(f"  MEG: {meg_active_pct:.1f}% windows above threshold (likely speech)")
print(f"  External: {ext_active_pct:.1f}% windows above threshold (likely speech)")
print()

# Find first and last active regions
meg_first_active = np.where(meg_active)[0][0] if np.any(meg_active) else 0
meg_last_active = np.where(meg_active)[0][-1] if np.any(meg_active) else len(meg_active)-1

ext_first_active = np.where(ext_active)[0][0] if np.any(ext_active) else 0
ext_last_active = np.where(ext_active)[0][-1] if np.any(ext_active) else len(ext_active)-1

print(f"  MEG active region: {meg_first_active}s to {meg_last_active}s")
print(f"  External active region: {ext_first_active}s to {ext_last_active}s")
print(f"  MEG leading silence: {meg_first_active}s")
print(f"  MEG trailing silence: {len(meg_rms) - meg_last_active}s")
print(f"  External leading silence: {ext_first_active}s")
print(f"  External trailing silence: {len(ext_rms) - ext_last_active}s")
print()

# ==============================================================================
# STEP 3: Run cross-correlation with RESTRICTED search window
# ==============================================================================
print("STEP 3: Computing cross-correlation with restricted search window...")
print("-" * 70)

# Preprocess as before
target_sr = 1000.0
lowpass_freq = 400

# Filter and resample
meg_nyq = meg_sfreq / 2
sos = signal.butter(4, lowpass_freq / meg_nyq, btype="low", output="sos")
meg_filtered = signal.sosfilt(sos, meg_audio)

ext_nyq = ext_sfreq / 2
sos = signal.butter(4, lowpass_freq / ext_nyq, btype="low", output="sos")
ext_filtered = signal.sosfilt(sos, ext_audio)
ext_resampled = librosa.resample(ext_filtered, orig_sr=ext_sfreq, target_sr=target_sr)

# Compute envelopes
meg_env = np.abs(hilbert(meg_filtered))
ext_env = np.abs(hilbert(ext_resampled))

# Lowpass filter envelopes
env_lowpass = 15
nyq = target_sr / 2
sos_env = signal.butter(3, env_lowpass / nyq, btype="low", output="sos")
meg_env_filt = signal.sosfilt(sos_env, meg_env)
ext_env_filt = signal.sosfilt(sos_env, ext_env)

# Normalize
meg_norm = (meg_env_filt - np.mean(meg_env_filt)) / np.std(meg_env_filt)
ext_norm = (ext_env_filt - np.mean(ext_env_filt)) / np.std(ext_env_filt)

# Cross-correlation
xcorr = correlate(meg_norm, ext_norm, mode="valid", method="fft")

# OLD METHOD: ±10 second search window
max_lag_s_old = 10.0
max_lag_samples_old = int(max_lag_s_old * target_sr)
n_xcorr = len(xcorr)
center = n_xcorr // 2
search_start_old = max(0, center - max_lag_samples_old)
search_end_old = min(n_xcorr, center + max_lag_samples_old)
search_region_old = xcorr[search_start_old:search_end_old]
peak_idx_old = np.argmax(np.abs(search_region_old))
peak_idx_global_old = search_start_old + peak_idx_old
lag_samples_old = peak_idx_global_old - (len(ext_norm) - 1)
offset_s_old = lag_samples_old / target_sr
peak_value_old = xcorr[peak_idx_global_old] / len(ext_norm)

print(f"  OLD METHOD (±10s window):")
print(f"    Offset: {offset_s_old:.4f} s")
print(f"    Correlation: {peak_value_old:.4f}")
print()

# NEW METHOD: Restrict search to reasonable range based on expected delay
# Assumption: external audio should align somewhere near the start (within ±30s)
max_lag_s_new = 30.0
max_lag_samples_new = int(max_lag_s_new * target_sr)
search_start_new = max(0, center - max_lag_samples_new)
search_end_new = min(n_xcorr, center + max_lag_samples_new)
search_region_new = xcorr[search_start_new:search_end_new]
peak_idx_new = np.argmax(np.abs(search_region_new))
peak_idx_global_new = search_start_new + peak_idx_new
lag_samples_new = peak_idx_global_new - (len(ext_norm) - 1)
offset_s_new = lag_samples_new / target_sr
peak_value_new = xcorr[peak_idx_global_new] / len(ext_norm)

print(f"  NEW METHOD (±30s window):")
print(f"    Offset: {offset_s_new:.4f} s")
print(f"    Correlation: {peak_value_new:.4f}")
print()

# Find top 5 peaks in full xcorr to see alternatives
xcorr_normalized = xcorr / len(ext_norm)
lags = np.arange(len(xcorr)) - (len(ext_norm) - 1)
lag_times = lags / target_sr

# Get top 5 peaks
peak_indices = np.argsort(np.abs(xcorr_normalized))[-5:][::-1]
print("  TOP 5 CORRELATION PEAKS:")
for i, idx in enumerate(peak_indices, 1):
    lag_s = lag_times[idx]
    corr = xcorr_normalized[idx]
    print(f"    {i}. Lag: {lag_s:+8.2f}s, Correlation: {corr:+.4f}")
print()

# ==============================================================================
# STEP 4: Visualizations
# ==============================================================================
print("STEP 4: Generating diagnostic plots...")
print("-" * 70)

# Plot 1: RMS energy over time
fig, axes = plt.subplots(3, 1, figsize=(16, 10))

# MEG RMS
axes[0].plot(meg_time_windows, meg_rms, alpha=0.7, linewidth=1)
axes[0].axhline(meg_threshold, color='red', linestyle='--', linewidth=1, label='Threshold (25th percentile)')
axes[0].fill_between(meg_time_windows, 0, np.max(meg_rms), where=meg_active, alpha=0.2, color='green', label='Active regions')
axes[0].set_ylabel('RMS Energy')
axes[0].set_title(f'MEG Auxiliary Audio - RMS Energy Over Time ({meg_active_pct:.1f}% active)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, len(meg_rms))

# External RMS
axes[1].plot(ext_time_windows, ext_rms, alpha=0.7, linewidth=1, color='orange')
axes[1].axhline(ext_threshold, color='red', linestyle='--', linewidth=1, label='Threshold (25th percentile)')
axes[1].fill_between(ext_time_windows, 0, np.max(ext_rms), where=ext_active, alpha=0.2, color='green', label='Active regions')
axes[1].set_ylabel('RMS Energy')
axes[1].set_title(f'External Audio - RMS Energy Over Time ({ext_active_pct:.1f}% active)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, len(ext_rms))

# Overlay comparison (normalized)
meg_rms_norm = meg_rms / np.max(meg_rms)
ext_rms_norm = ext_rms / np.max(ext_rms)
axes[2].plot(meg_time_windows, meg_rms_norm, alpha=0.7, linewidth=1, label='MEG (normalized)')
axes[2].plot(ext_time_windows, ext_rms_norm, alpha=0.7, linewidth=1, label='External (normalized)')
axes[2].set_xlabel('Time (seconds)')
axes[2].set_ylabel('Normalized RMS')
axes[2].set_title('RMS Energy Comparison (both normalized)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "deadtime_analysis_rms.png", dpi=150)
plt.close()
print(f"  ✓ Saved: {OUTPUT_DIR}/deadtime_analysis_rms.png")

# Plot 2: Cross-correlation with multiple peaks marked
fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

# Full cross-correlation
axes[0].plot(lag_times, xcorr_normalized, linewidth=0.5, alpha=0.7)
axes[0].axvline(offset_s_old, color='red', linestyle='--', linewidth=2,
                label=f'OLD (±10s): {offset_s_old:.2f}s, corr={peak_value_old:.3f}')
axes[0].axvline(offset_s_new, color='green', linestyle='--', linewidth=2,
                label=f'NEW (±30s): {offset_s_new:.2f}s, corr={peak_value_new:.3f}')
for i, idx in enumerate(peak_indices[:3], 1):
    axes[0].axvline(lag_times[idx], color='purple', linestyle=':', alpha=0.5, linewidth=1)
axes[0].axhline(0, color='black', linestyle=':', linewidth=0.5)
axes[0].set_ylabel('Normalized Cross-Correlation')
axes[0].set_title('Full Cross-Correlation Function')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Zoomed to ±60 seconds
axes[1].plot(lag_times, xcorr_normalized, linewidth=0.8, alpha=0.7)
axes[1].axvline(offset_s_old, color='red', linestyle='--', linewidth=2,
                label=f'OLD: {offset_s_old:.2f}s')
axes[1].axvline(offset_s_new, color='green', linestyle='--', linewidth=2,
                label=f'NEW: {offset_s_new:.2f}s')
axes[1].axhline(0, color='black', linestyle=':', linewidth=0.5)
axes[1].set_xlim(-60, 60)
axes[1].set_xlabel('Lag (seconds)')
axes[1].set_ylabel('Normalized Cross-Correlation')
axes[1].set_title('Cross-Correlation (zoomed to ±60s)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "deadtime_analysis_xcorr.png", dpi=150)
plt.close()
print(f"  ✓ Saved: {OUTPUT_DIR}/deadtime_analysis_xcorr.png")

# Plot 3: Aligned envelopes comparison (OLD vs NEW)
fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

meg_t = np.arange(len(meg_env_filt)) / target_sr
ext_t = np.arange(len(ext_env_filt)) / target_sr

# Show 60-second window
duration = 60
meg_mask = meg_t <= duration

# OLD alignment
ext_t_aligned_old = ext_t - offset_s_old
ext_mask_old = (ext_t_aligned_old >= 0) & (ext_t_aligned_old <= duration)
axes[0].plot(meg_t[meg_mask], meg_env_filt[meg_mask] / np.max(meg_env_filt), alpha=0.7, label='MEG', linewidth=1)
axes[0].plot(ext_t_aligned_old[ext_mask_old], ext_env_filt[ext_mask_old] / np.max(ext_env_filt),
             alpha=0.7, label=f'External (OLD: {offset_s_old:.2f}s)', linewidth=1)
axes[0].set_ylabel('Normalized Envelope')
axes[0].set_title(f'OLD Method Alignment (offset={offset_s_old:.2f}s, corr={peak_value_old:.3f})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# NEW alignment
ext_t_aligned_new = ext_t - offset_s_new
ext_mask_new = (ext_t_aligned_new >= 0) & (ext_t_aligned_new <= duration)
axes[1].plot(meg_t[meg_mask], meg_env_filt[meg_mask] / np.max(meg_env_filt), alpha=0.7, label='MEG', linewidth=1)
axes[1].plot(ext_t_aligned_new[ext_mask_new], ext_env_filt[ext_mask_new] / np.max(ext_env_filt),
             alpha=0.7, label=f'External (NEW: {offset_s_new:.2f}s)', linewidth=1, color='orange')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Normalized Envelope')
axes[1].set_title(f'NEW Method Alignment (offset={offset_s_new:.2f}s, corr={peak_value_new:.3f})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "deadtime_analysis_alignment.png", dpi=150)
plt.close()
print(f"  ✓ Saved: {OUTPUT_DIR}/deadtime_analysis_alignment.png")

print()
print("=" * 70)
print("DEAD TIME ANALYSIS COMPLETE")
print("=" * 70)
print()
print("KEY FINDINGS:")
print(f"  • MEG duration: {meg_duration:.2f}s, External duration: {ext_duration:.2f}s")
print(f"  • MEG has {meg_first_active}s leading silence, {len(meg_rms)-meg_last_active}s trailing silence")
print(f"  • External has {ext_first_active}s leading silence, {len(ext_rms)-ext_last_active}s trailing silence")
print()
print(f"  • OLD method (±10s window): offset = {offset_s_old:.2f}s, corr = {peak_value_old:.3f}")
print(f"  • NEW method (±30s window): offset = {offset_s_new:.2f}s, corr = {peak_value_new:.3f}")
print()
if abs(offset_s_old) > 100:
    print("  ⚠️  WARNING: OLD offset is suspiciously large!")
    print("      This suggests correlation with dead time rather than speech.")
print()
print("RECOMMENDATION:")
print("  Use the NEW method with restricted search window (±30s)")
print("  OR trim dead time from beginning/end before cross-correlation")
print("=" * 70)
