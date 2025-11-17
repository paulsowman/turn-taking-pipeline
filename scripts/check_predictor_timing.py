#!/usr/bin/env python3
"""
Quick diagnostic to check if predictor timing is correct.

Loads the combined TRF FIF and plots:
1. MEG data (first good channel)
2. Envelope predictor
3. Word onset predictor

This helps verify predictors are aligned correctly with MEG timebase.
"""

import mne
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

subject = 'sub-01'
condition = 'conversation'

# Load combined file
fif_file = f'outputs/trf_combined/{subject}/{subject}_{condition}_trf_raw.fif'
print(f"Loading: {fif_file}")
raw = mne.io.read_raw_fif(fif_file, preload=True, verbose=False)

print(f"Sampling rate: {raw.info['sfreq']} Hz")
print(f"Duration: {raw.times[-1]:.1f}s")
print(f"Channels: {len(raw.ch_names)}")

# Get first MEG channel
meg_picks = mne.pick_types(raw.info, meg=True)
meg_ch_idx = meg_picks[0]
meg_data = raw[meg_ch_idx, :][0][0]

# Get predictors
try:
    env_idx = raw.ch_names.index('MISC_envelope_participant')
    env_data = raw[env_idx, :][0][0]
except ValueError:
    print("Warning: MISC_envelope_participant not found")
    env_data = None

try:
    onset_idx = raw.ch_names.index('MISC_word_onsets_participant')
    onset_data = raw[onset_idx, :][0][0]
except ValueError:
    print("Warning: MISC_word_onsets_participant not found")
    onset_data = None

times = raw.times

# Plot segment from 60-70 seconds (likely to have speech activity)
t_start = 60.0
t_end = 70.0
mask = (times >= t_start) & (times < t_end)

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Panel 1: MEG data
ax = axes[0]
ax.plot(times[mask], meg_data[mask], linewidth=0.5)
ax.set_ylabel('MEG amplitude')
ax.set_title(f'MEG Channel: {raw.ch_names[meg_ch_idx]} ({t_start}-{t_end}s)')
ax.set_xlim(t_start, t_end)
ax.grid(True, alpha=0.3)

# Panel 2: Envelope
if env_data is not None:
    ax = axes[1]
    ax.plot(times[mask], env_data[mask], linewidth=0.5, color='orange')
    ax.set_ylabel('Envelope')
    ax.set_title(f'Envelope Predictor (MISC_envelope_participant) ({t_start}-{t_end}s)')
    ax.set_xlim(t_start, t_end)
    ax.grid(True, alpha=0.3)
    print(f"Envelope: {np.sum(env_data != 0)} non-zero samples")

# Panel 3: Word onsets
if onset_data is not None:
    ax = axes[2]
    total_onsets = np.sum(onset_data != 0)
    print(f"Word onsets: {total_onsets} total events in file")

    # Find where onsets actually are (in full file)
    onset_indices = np.where(onset_data != 0)[0]
    if len(onset_indices) > 0:
        first_onset_time = times[onset_indices[0]]
        print(f"First onset at: {first_onset_time:.3f}s")
        print(f"Last onset at: {times[onset_indices[-1]]:.3f}s")

    # Plot onsets in the SAME time window as other panels
    onset_times = times[mask][onset_data[mask] != 0]
    onset_values = onset_data[mask][onset_data[mask] != 0]

    if len(onset_times) > 0:
        ax.stem(onset_times, onset_values, linefmt='C2-', markerfmt='C2o', basefmt=' ')
        ax.set_title(f'Word Onsets (MISC_word_onsets_participant) - {len(onset_times)} in window')
    else:
        ax.text(0.5, 0.5, f'No word onsets in {t_start}-{t_end}s window',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(f'Word Onsets - None in {t_start}-{t_end}s')
        print(f"Note: No onsets in {t_start}-{t_end}s window")

    ax.set_ylabel('Word onset amplitude')
    ax.set_xlabel('Time (s)')
    ax.set_xlim(t_start, t_end)  # Ensure same x-axis as other panels
    ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save to outputs/diagnostics
output_dir = Path('outputs/diagnostics')
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / f'{subject}_{condition}_predictor_timing_check.png'
fig.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\nSaved: {output_file}")
print("\nCheck this plot to verify:")
print("1. Envelope follows MEG amplitude envelope")
print("2. Word onsets align with speech onsets in MEG")
print("3. No obvious time shifts or delays")
