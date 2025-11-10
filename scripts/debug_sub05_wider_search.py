#!/usr/bin/env python3
"""
Debug sub-05 with wider search window and investigate the phase/lag issue.
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

SUBJECT = "sub-05"
RUN = 1

MEG_FILE = Path("/Users/em18033/Library/CloudStorage/OneDrive-SharedLibraries-MacquarieUniversity/Natural_Conversations_study - Documents/analysis/natural-conversations-bids/derivatives/mne-bids-pipeline-20240628/sub-05/meg/sub-05_task-conversation_run-01_proc-clean_raw.fif")
AUDIO_FILE = Path("/Users/em18033/Library/CloudStorage/OneDrive-AUTUniversity/Projects/Conversational_AI/Archive/data/audios/G05/console_mic_B1.wav")
OUTPUT_DIR = Path("outputs/sync_debug")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print(f"DEBUG {SUBJECT} RUN-{RUN:02d} - WIDER SEARCH")
print("=" * 70)
print()

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
print(f"  Duration difference: {abs(meg_duration - ext_duration):.2f}s")
print()

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
# Cross-correlation with MULTIPLE search windows
# ==============================================================================
print("Computing cross-correlation with multiple search windows...")
print()

# Full cross-correlation
xcorr = correlate(meg_norm, ext_norm, mode="full", method="fft")
lags = np.arange(len(xcorr)) - (len(ext_norm) - 1)
lag_times = lags / target_sr

# Try different search windows
search_windows = [30, 60, 120, 300]  # seconds

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
print("Top 10 peaks in ±300s window:")
max_lag_samples = int(300 * target_sr)
zero_lag_idx = len(ext_norm) - 1
search_start = max(0, zero_lag_idx - max_lag_samples)
search_end = min(len(xcorr), zero_lag_idx + max_lag_samples + 1)
search_region = xcorr[search_start:search_end]

top_peaks = np.argsort(search_region)[-10:][::-1]
for i, idx in enumerate(top_peaks, 1):
    global_idx = search_start + idx
    lag_s = lag_times[global_idx]
    corr = xcorr[global_idx] / len(ext_norm)
    print(f"  {i:2d}. Lag: {lag_s:+7.2f}s, Correlation: {corr:+.4f}")

print()

# Use the best result (from largest window)
best = results[-1]
offset_s = best['offset']
correlation = best['correlation']

print(f"Using offset: {offset_s:.2f}s (from ±{best['window']}s window)")
print()

# ==============================================================================
# Visualization
# ==============================================================================
print("Generating plots...")

meg_t = np.arange(len(meg_env_filt)) / target_sr
ext_t = np.arange(len(ext_env_filt)) / target_sr

# Apply offset: ADD to external time (shifts it forward if positive)
ext_t_aligned = ext_t + offset_s

# Normalize envelopes for plotting
meg_env_norm = meg_env_filt / np.max(meg_env_filt)
ext_env_norm = ext_env_filt / np.max(ext_env_filt)

# Create comprehensive plot
fig, axes = plt.subplots(5, 1, figsize=(20, 20))

# Plot 1: Raw audio waveforms (first 30s)
duration = 30
downsample = 100
mask_meg = meg_t <= duration
mask_ext = ext_t <= duration
axes[0].plot(meg_t[mask_meg][::downsample], meg_audio[mask_meg][::downsample],
             alpha=0.5, linewidth=0.3, label='MEG raw', color='blue')
ext_t_full = np.arange(len(ext_audio)) / ext_sfreq
mask_ext_full = ext_t_full <= duration
downsample_ext = int(ext_sfreq / 10)
axes[0].plot(ext_t_full[mask_ext_full][::downsample_ext], ext_audio[mask_ext_full][::downsample_ext],
             alpha=0.5, linewidth=0.3, label='External raw', color='orange')
axes[0].set_ylabel('Amplitude')
axes[0].set_title(f'{SUBJECT} - Raw Audio (first {duration}s, NOT aligned)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, duration)

# Plot 2: Cross-correlation function
max_lag_plot = 300
mask_xcorr = (lag_times >= -max_lag_plot) & (lag_times <= max_lag_plot)
axes[1].plot(lag_times[mask_xcorr], xcorr[mask_xcorr] / len(ext_norm),
             linewidth=0.5, alpha=0.7)
axes[1].axvline(offset_s, color='red', linestyle='--', linewidth=2,
                label=f'Peak: {offset_s:.2f}s, corr={correlation:.3f}')
axes[1].axhline(0, color='black', linestyle=':', linewidth=0.5)
axes[1].set_xlabel('Lag (s)')
axes[1].set_ylabel('Normalized Cross-Correlation')
axes[1].set_title(f'Cross-Correlation (±{max_lag_plot}s view)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Full aligned envelopes
downsample_env = 10
axes[2].plot(meg_t[::downsample_env], meg_env_norm[::downsample_env],
             alpha=0.7, linewidth=0.5, label='MEG envelope', color='blue')
axes[2].plot(ext_t_aligned[::downsample_env], ext_env_norm[::downsample_env],
             alpha=0.7, linewidth=0.5, label=f'External envelope (offset={offset_s:.2f}s)', color='orange')
axes[2].set_ylabel('Normalized Envelope')
axes[2].set_title(f'Full View - Aligned Envelopes (correlation = {correlation:.3f})')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
if offset_s > 0:
    axes[2].axvline(offset_s, color='red', linestyle='--', alpha=0.3, linewidth=1)

# Plot 4: First 60s after overlap starts
overlap_start = max(0, offset_s)
start = overlap_start
end = min(meg_duration, start + 60)
mask_meg = (meg_t >= start) & (meg_t <= end)
mask_ext = (ext_t_aligned >= start) & (ext_t_aligned <= end)
axes[3].plot(meg_t[mask_meg], meg_env_norm[mask_meg],
             alpha=0.8, linewidth=1, label='MEG', color='blue')
axes[3].plot(ext_t_aligned[mask_ext], ext_env_norm[mask_ext],
             alpha=0.8, linewidth=1, label='External (aligned)', color='orange')
axes[3].set_ylabel('Normalized Envelope')
axes[3].set_title(f'First 60s of overlap ({start:.1f}-{end:.1f}s)')
axes[3].legend()
axes[3].grid(True, alpha=0.3)
axes[3].set_xlim(start, end)

# Plot 5: 10s zoom
zoom_mid = (start + end) / 2
zoom_start = zoom_mid - 5
zoom_end = zoom_mid + 5
mask_meg = (meg_t >= zoom_start) & (meg_t <= zoom_end)
mask_ext = (ext_t_aligned >= zoom_start) & (ext_t_aligned <= zoom_end)
axes[4].plot(meg_t[mask_meg], meg_env_norm[mask_meg],
             alpha=0.8, linewidth=2, label='MEG', color='blue')
axes[4].plot(ext_t_aligned[mask_ext], ext_env_norm[mask_ext],
             alpha=0.8, linewidth=2, label='External (aligned)', color='orange')
axes[4].set_xlabel('Time (s)')
axes[4].set_ylabel('Normalized Envelope')
axes[4].set_title(f'10s Zoom ({zoom_start:.1f}-{zoom_end:.1f}s)')
axes[4].legend()
axes[4].grid(True, alpha=0.3)
axes[4].set_xlim(zoom_start, zoom_end)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"debug_{SUBJECT}_wider_search.png", dpi=100)
plt.close()
print(f"  ✓ Saved: {OUTPUT_DIR}/debug_{SUBJECT}_wider_search.png")

print()
print("=" * 70)
print("WIDER SEARCH COMPLETE")
print("=" * 70)
print(f"Best offset: {offset_s:.4f}s")
print(f"Correlation: {correlation:.4f}")
print()
print("Interpretation:")
if offset_s > 0:
    print(f"  External audio starts {offset_s:.2f}s AFTER MEG recording")
elif offset_s < 0:
    print(f"  External audio starts {-offset_s:.2f}s BEFORE MEG recording")
    print(f"  (or equivalently: MEG recording starts {-offset_s:.2f}s AFTER external)")
else:
    print(f"  Signals are synchronized at t=0")
print()
print(f"Plot: {OUTPUT_DIR}/debug_{SUBJECT}_wider_search.png")
print("=" * 70)
