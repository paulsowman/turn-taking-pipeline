#!/usr/bin/env python3
"""
Create TRF Analysis FIF Files

Adds TRF predictor time series as MISC channels to MEG .fif files.
All predictors are at MEG sampling rate (1000 Hz) in MEG timebase.

TODO: Add downsampling to 100 Hz for computational efficiency
--------------------------------------------------------------
For future pipeline regeneration, downsample MEG data BEFORE creating predictors:
- Load MEG at native 1000 Hz
- Downsample to 100 Hz (10ms resolution, 10x speedup for TRF fitting)
- Create all predictors at 100 Hz timebase
- Word onsets placed at 10ms grid (vs current 1ms impulses)
- More principled than downsampling post-hoc at analysis time
- Avoids anti-aliasing filter spreading sharp impulses

Benefits:
- TRF fitting: ~90 min → ~9 min per condition (conversation)
- Clean predictor encoding at target sampling rate
- 10ms resolution adequate for word-level analysis (~200ms between words)

Predictors added (19 total, grouped by speaker):

INTERVIEWER (9 channels):
- MISC_envelope_interviewer: Interviewer audio envelope (external)
- MISC_envelope_meg_mic7: MEG MISC 007 envelope (interviewer, for sync verification)
- MISC_f0_interviewer: Interviewer F0 contour (speaker-masked)
- MISC_word_onsets_interviewer: Delta functions at interviewer word onsets
- MISC_surprisal_interviewer: Conversation-aware surprisal (GPT-2, chronological)
- MISC_duration_interviewer: Delta functions weighted by interviewer word duration
- MISC_f0_deviation_interviewer: Z-scored F0 deviations (prosodic unexpectedness)
- MISC_duration_deviation_interviewer: Z-scored duration deviations
- MISC_pause_interviewer: Time since last interviewer word (normalized to 0-1)

PARTICIPANT (9 channels):
- MISC_envelope_participant: Participant audio envelope (external)
- MISC_envelope_meg_mic8: MEG MISC 008 envelope (participant, for sync verification)
- MISC_f0_participant: Participant F0 contour (speaker-masked)
- MISC_word_onsets_participant: Delta functions at participant word onsets
- MISC_surprisal_participant: Conversation-aware surprisal (GPT-2, chronological)
- MISC_duration_participant: Delta functions weighted by participant word duration
- MISC_f0_deviation_participant: Z-scored F0 deviations (prosodic unexpectedness)
- MISC_duration_deviation_participant: Z-scored duration deviations
- MISC_pause_participant: Time since last participant word (normalized to 0-1)

Note: Surprisal is conversation-aware - each word's surprisal is calculated
based on the full chronological conversation history (both speakers), capturing
true conversational predictability.

SHARED (1 channel):
- MISC_speaker: Categorical (0=silence, 1=interviewer, 2=participant, 3=overlap)

Usage:
    # Single subject/run
    python scripts/create_trf_fif.py --subject sub-01 --run 1

    # Multiple runs
    python scripts/create_trf_fif.py --subject sub-01 --runs 1 2 3

    # All subjects
    python scripts/create_trf_fif.py --all

    # Skip surprisal computation (faster, for testing)
    python scripts/create_trf_fif.py --subject sub-01 --run 1 --no-surprisal
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import mne
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import load_config, get_subject_paths
from utils.logging_setup import setup_logging
from utils.io import load_audio
from predictors.trf_predictors import (
    create_envelope_predictor,
    create_meg_audio_envelope,
    create_f0_predictor,
    create_word_onset_predictor,
    create_surprisal_predictor,
    create_duration_predictor,
    create_speaker_predictor,
    compute_word_surprisal,
    create_f0_deviation_predictor,
    create_duration_deviation_predictor,
    create_pause_predictor,
)

# All available subjects
ALL_SUBJECTS = [
    'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05',
    'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10',
    'sub-11', 'sub-13', 'sub-14', 'sub-15', 'sub-16',
    'sub-17', 'sub-18', 'sub-19', 'sub-21', 'sub-22',
    'sub-23', 'sub-24', 'sub-25', 'sub-26', 'sub-27',
    'sub-29', 'sub-31', 'sub-32'
]


def process_subject_run(
    subject: str,
    run: int,
    base_dir: Path,
    config: dict,
    compute_surprisal_flag: bool = True,
    overwrite: bool = False,
) -> bool:
    """
    Create TRF predictors and add to MEG .fif file.

    Parameters
    ----------
    subject : str
        Subject ID
    run : int
        Run number
    base_dir : Path
        Base directory
    config : dict
        Configuration
    compute_surprisal_flag : bool
        Whether to compute GPT-2 surprisal (slow)
    overwrite : bool
        Whether to overwrite existing TRF .fif file

    Returns
    -------
    success : bool
    """
    print(f"\n{'='*70}")
    print(f"CREATING TRF FIF: {subject} run-{run:02d}")
    print(f"{'='*70}\n")

    # Check if output already exists
    output_dir = base_dir / "outputs" / "trf" / subject / f"run-{run:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{subject}_run-{run:02d}_trf_raw.fif"

    if output_file.exists() and not overwrite:
        print(f"✓ TRF FIF already exists: {output_file}")
        print(f"  Use --overwrite to regenerate")
        return True

    # Get paths
    paths = get_subject_paths(subject, run, config)

    # Load MEG data
    print("Loading MEG data...")
    meg_raw = mne.io.read_raw_fif(paths['meg_raw'], preload=False, verbose=False)
    meg_times = meg_raw.times
    meg_sfreq = meg_raw.info['sfreq']
    print(f"  MEG: {len(meg_times)} samples @ {meg_sfreq} Hz")
    print(f"  Duration: {meg_times[-1]:.1f}s")

    # Load sync parameters
    sync_dir = base_dir / "outputs" / "sync" / subject / f"run-{run:02d}"
    sync_params_file = sync_dir / "sync_params.json"

    if not sync_params_file.exists():
        print(f"✗ Sync params not found: {sync_params_file}")
        print(f"  Run: python scripts/run_audio_meg_sync.py --subject {subject} --run {run}")
        return False

    with open(sync_params_file, 'r') as f:
        sync_params = json.load(f)

    sync_offset = sync_params['initial_offset_s']
    print(f"  Sync offset: {sync_offset:.4f}s")

    # Load audio (both channels)
    print("\nLoading audio...")
    audio_file_interviewer = paths['external_audio_interviewer']
    audio_file_participant = paths.get('external_audio_participant', audio_file_interviewer)

    # Interviewer audio (channel 0 / left)
    audio_interviewer, sr = load_audio(audio_file_interviewer, sr=None, channel=0)
    print(f"  Interviewer: {len(audio_interviewer)/sr:.1f}s @ {sr} Hz")
    print(f"    File: {audio_file_interviewer.name}")
    print(f"    Channel: 0 (left)")

    # Participant audio
    # If separate file (subject_mic), use channel 0
    # If same file (console_mic), use channel 1
    if audio_file_participant != audio_file_interviewer:
        # Separate participant file - assume mono or participant on channel 0
        participant_channel = 0
        print(f"  Participant: Using separate file")
    else:
        # Same file as interviewer - participant on channel 1 (right)
        participant_channel = 1
        print(f"  Participant: Using same file as interviewer")

    audio_participant, sr = load_audio(audio_file_participant, sr=None, channel=participant_channel)
    print(f"    {len(audio_participant)/sr:.1f}s @ {sr} Hz")
    print(f"    File: {audio_file_participant.name}")
    print(f"    Channel: {participant_channel}")

    # Check if participant audio is actually silent (common issue)
    participant_rms = np.sqrt(np.mean(audio_participant**2))
    if participant_rms < 1e-6:
        print(f"  ⚠ WARNING: Participant audio appears to be silent (RMS={participant_rms:.2e})")
        print(f"  This may indicate a channel selection issue or zeroed-out audio.")

    # Load MFA word timing (both speakers)
    print("\nLoading MFA word timing (dual-speaker)...")
    feature_dir = base_dir / "outputs" / "features" / subject / f"run-{run:02d}"
    mfa_interviewer_file = feature_dir / "transcript_mfa_interviewer.csv"
    mfa_participant_file = feature_dir / "transcript_mfa_participant.csv"

    if not mfa_interviewer_file.exists():
        print(f"✗ Interviewer MFA transcript not found: {mfa_interviewer_file}")
        print(f"  Run: python scripts/run_mfa_alignment.py --subject {subject} --run {run}")
        return False

    if not mfa_participant_file.exists():
        print(f"✗ Participant MFA transcript not found: {mfa_participant_file}")
        print(f"  Run: python scripts/run_mfa_alignment.py --subject {subject} --run {run}")
        return False

    mfa_interviewer_df = pd.read_csv(mfa_interviewer_file)
    mfa_participant_df = pd.read_csv(mfa_participant_file)
    print(f"  Interviewer words: {len(mfa_interviewer_df)}")
    print(f"  Participant words: {len(mfa_participant_df)}")

    # Extract word data for interviewer
    # Filter out rows with missing words
    valid_mask_int = mfa_interviewer_df['word'].notna()
    mfa_interviewer_df = mfa_interviewer_df[valid_mask_int].copy()

    word_times_interviewer_meg = (
        mfa_interviewer_df['start_meg'].values if 'start_meg' in mfa_interviewer_df
        else (mfa_interviewer_df['start'].values - sync_offset)
    )
    word_durations_interviewer = mfa_interviewer_df['duration'].values
    # Convert words to strings (handles numeric values like "1", "2")
    words_interviewer = [str(w) for w in mfa_interviewer_df['word'].tolist()]

    # Extract word data for participant
    # Filter out rows with missing words
    valid_mask_part = mfa_participant_df['word'].notna()
    mfa_participant_df = mfa_participant_df[valid_mask_part].copy()

    word_times_participant_meg = (
        mfa_participant_df['start_meg'].values if 'start_meg' in mfa_participant_df
        else (mfa_participant_df['start'].values - sync_offset)
    )
    word_durations_participant = mfa_participant_df['duration'].values
    # Convert words to strings (handles numeric values like "1", "2")
    words_participant = [str(w) for w in mfa_participant_df['word'].tolist()]

    # Create predictors
    print("\nCreating predictors...")

    # 1. External audio envelopes
    print("  1/13: Envelope (interviewer, external)...")
    env_interviewer = create_envelope_predictor(
        audio_interviewer, sr, meg_times, sync_offset
    )

    print("  2/13: Envelope (participant, external)...")
    env_participant = create_envelope_predictor(
        audio_participant, sr, meg_times, sync_offset
    )

    # 2. MEG-recorded audio envelopes (for sync verification)
    print("  3/13: Envelope (MEG MIC 7)...")
    env_meg_mic7 = create_meg_audio_envelope(meg_raw, 'MISC 007')

    print("  4/13: Envelope (MEG MIC 8)...")
    env_meg_mic8 = create_meg_audio_envelope(meg_raw, 'MISC 008')

    # 3. F0 (normalized to 0-1)
    print("  5/13: F0 (interviewer)...")
    f0_interviewer = create_f0_predictor(
        audio_interviewer, sr, meg_times, sync_offset, normalize=True
    )

    print("  6/13: F0 (participant)...")
    f0_participant = create_f0_predictor(
        audio_participant, sr, meg_times, sync_offset, normalize=True
    )

    # 4. Word onsets (speaker-specific)
    print("  7/13: Word onsets (interviewer)...")
    word_onsets_interviewer = create_word_onset_predictor(word_times_interviewer_meg, meg_times)

    print("  8/13: Word onsets (participant)...")
    word_onsets_participant = create_word_onset_predictor(word_times_participant_meg, meg_times)

    # 5. Surprisal (normalized to 0-1, conversation-aware)
    # Calculate on chronological conversation sequence, then split by speaker
    surprisal_interviewer_raw_stats = None
    surprisal_participant_raw_stats = None
    if compute_surprisal_flag:
        print("  9-10/13: Surprisal (chronological conversation, GPT-2)...")
        try:
            # Merge words from both speakers chronologically
            all_words_with_info = []
            for i, (word, time) in enumerate(zip(words_interviewer, word_times_interviewer_meg)):
                all_words_with_info.append({
                    'time': time,
                    'word': word,
                    'speaker': 'interviewer',
                    'idx_in_speaker': i,
                })
            for i, (word, time) in enumerate(zip(words_participant, word_times_participant_meg)):
                all_words_with_info.append({
                    'time': time,
                    'word': word,
                    'speaker': 'participant',
                    'idx_in_speaker': i,
                })

            # Sort by time to get chronological conversation sequence
            all_words_with_info.sort(key=lambda x: x['time'])
            words_chronological = [w['word'] for w in all_words_with_info]

            print(f"    Total words in conversation: {len(words_chronological)}")
            print(f"    Computing surprisal on full conversation sequence...")

            # Calculate surprisal on chronological sequence
            # Each word conditioned on ALL previous words (both speakers)
            surprisal_chronological = compute_word_surprisal(words_chronological)

            # Split back to speaker-specific arrays
            surprisal_values_interviewer = np.zeros(len(words_interviewer))
            surprisal_values_participant = np.zeros(len(words_participant))

            for word_info, surprisal_val in zip(all_words_with_info, surprisal_chronological):
                if word_info['speaker'] == 'interviewer':
                    surprisal_values_interviewer[word_info['idx_in_speaker']] = surprisal_val
                else:
                    surprisal_values_participant[word_info['idx_in_speaker']] = surprisal_val

            # Save raw statistics before normalization
            surprisal_interviewer_raw_stats = {
                'mean': float(np.mean(surprisal_values_interviewer)),
                'std': float(np.std(surprisal_values_interviewer)),
                'min': float(surprisal_values_interviewer.min()),
                'max': float(surprisal_values_interviewer.max()),
            }
            surprisal_participant_raw_stats = {
                'mean': float(np.mean(surprisal_values_participant)),
                'std': float(np.std(surprisal_values_participant)),
                'min': float(surprisal_values_participant.min()),
                'max': float(surprisal_values_participant.max()),
            }

            print(f"    Interviewer surprisal: mean={surprisal_interviewer_raw_stats['mean']:.2f} nats")
            print(f"    Participant surprisal: mean={surprisal_participant_raw_stats['mean']:.2f} nats")

            # Create speaker-specific predictors
            surprisal_interviewer = create_surprisal_predictor(
                word_times_interviewer_meg, surprisal_values_interviewer, meg_times, normalize=True
            )
            surprisal_participant = create_surprisal_predictor(
                word_times_participant_meg, surprisal_values_participant, meg_times, normalize=True
            )

        except Exception as e:
            print(f"  ⚠ Warning: Surprisal computation failed: {e}")
            import traceback
            traceback.print_exc()
            print(f"  Creating zero surprisal predictors")
            surprisal_interviewer = np.zeros(len(meg_times))
            surprisal_participant = np.zeros(len(meg_times))
    else:
        print("  9-10/13: Surprisal (skipped)...")
        surprisal_interviewer = np.zeros(len(meg_times))
        surprisal_participant = np.zeros(len(meg_times))

    # 6. Duration (normalized to 0-1, speaker-specific)
    print("  11/13: Duration (interviewer)...")
    # Save raw statistics before normalization
    duration_interviewer_raw_stats = {
        'mean_ms': float(np.mean(word_durations_interviewer) * 1000),
        'std_ms': float(np.std(word_durations_interviewer) * 1000),
        'min_ms': float(word_durations_interviewer.min() * 1000),
        'max_ms': float(word_durations_interviewer.max() * 1000),
    }
    duration_interviewer = create_duration_predictor(
        word_times_interviewer_meg, word_durations_interviewer, meg_times, normalize=True
    )

    print("  12/13: Duration (participant)...")
    # Save raw statistics before normalization
    duration_participant_raw_stats = {
        'mean_ms': float(np.mean(word_durations_participant) * 1000),
        'std_ms': float(np.std(word_durations_participant) * 1000),
        'min_ms': float(word_durations_participant.min() * 1000),
        'max_ms': float(word_durations_participant.max() * 1000),
    }
    duration_participant = create_duration_predictor(
        word_times_participant_meg, word_durations_participant, meg_times, normalize=True
    )

    # 7. Speaker
    print("  13/13: Speaker...")
    speaker = create_speaker_predictor(env_interviewer, env_participant)

    # Apply speaker-based masking to F0 to remove noise/bleed-through
    print("\nApplying speaker-based F0 masking...")
    # Zero interviewer F0 when not speaking (speaker != 1 and speaker != 3)
    interviewer_active = (speaker == 1) | (speaker == 3)  # Speaking or overlap
    f0_interviewer_masked = f0_interviewer * interviewer_active
    n_zeroed_int = np.sum((f0_interviewer > 0) & ~interviewer_active)
    print(f"  Interviewer F0: Zeroed {n_zeroed_int} samples during non-speech")

    # Zero participant F0 when not speaking (speaker != 2 and speaker != 3)
    participant_active = (speaker == 2) | (speaker == 3)  # Speaking or overlap
    f0_participant_masked = f0_participant * participant_active
    n_zeroed_part = np.sum((f0_participant > 0) & ~participant_active)
    print(f"  Participant F0: Zeroed {n_zeroed_part} samples during non-speech")

    # Replace original F0 with masked versions
    f0_interviewer = f0_interviewer_masked
    f0_participant = f0_participant_masked

    # 8. Prosodic deviation predictors
    print("\nCreating prosodic deviation predictors...")

    print("  14/19: F0 deviation (interviewer)...")
    f0_deviation_interviewer = create_f0_deviation_predictor(
        word_times_interviewer_meg,
        word_durations_interviewer,
        audio_interviewer,
        sr,
        meg_times,
        sync_offset
    )

    print("  15/19: F0 deviation (participant)...")
    f0_deviation_participant = create_f0_deviation_predictor(
        word_times_participant_meg,
        word_durations_participant,
        audio_participant,
        sr,
        meg_times,
        sync_offset
    )

    print("  16/19: Duration deviation (interviewer)...")
    duration_deviation_interviewer = create_duration_deviation_predictor(
        word_times_interviewer_meg,
        word_durations_interviewer,
        meg_times
    )

    print("  17/19: Duration deviation (participant)...")
    duration_deviation_participant = create_duration_deviation_predictor(
        word_times_participant_meg,
        word_durations_participant,
        meg_times
    )

    print("  18/19: Pause (interviewer)...")
    pause_interviewer = create_pause_predictor(
        word_times_interviewer_meg,
        meg_times,
        normalize=True
    )

    print("  19/19: Pause (participant)...")
    pause_participant = create_pause_predictor(
        word_times_participant_meg,
        meg_times,
        normalize=True
    )

    # Create info for new channels
    print("\nAdding predictors as MISC channels...")
    # Channels grouped by speaker for easy visualization
    ch_names = [
        # Interviewer group (9 channels)
        'MISC_envelope_interviewer',
        'MISC_envelope_meg_mic7',
        'MISC_f0_interviewer',
        'MISC_word_onsets_interviewer',
        'MISC_surprisal_interviewer',
        'MISC_duration_interviewer',
        'MISC_f0_deviation_interviewer',
        'MISC_duration_deviation_interviewer',
        'MISC_pause_interviewer',
        # Participant group (9 channels)
        'MISC_envelope_participant',
        'MISC_envelope_meg_mic8',
        'MISC_f0_participant',
        'MISC_word_onsets_participant',
        'MISC_surprisal_participant',
        'MISC_duration_participant',
        'MISC_f0_deviation_participant',
        'MISC_duration_deviation_participant',
        'MISC_pause_participant',
        # Shared (1 channel)
        'MISC_speaker',
    ]

    info = mne.create_info(ch_names, meg_sfreq, ch_types='misc')

    # Stack predictors (same order as ch_names)
    predictor_data = np.vstack([
        # Interviewer group
        env_interviewer,
        env_meg_mic7,
        f0_interviewer,
        word_onsets_interviewer,
        surprisal_interviewer,
        duration_interviewer,
        f0_deviation_interviewer,
        duration_deviation_interviewer,
        pause_interviewer,
        # Participant group
        env_participant,
        env_meg_mic8,
        f0_participant,
        word_onsets_participant,
        surprisal_participant,
        duration_participant,
        f0_deviation_participant,
        duration_deviation_participant,
        pause_participant,
        # Shared
        speaker.astype(float),
    ])

    # Create RawArray
    predictor_raw = mne.io.RawArray(predictor_data, info)

    # Add to MEG data
    print("  Loading full MEG data...")
    meg_raw.load_data()  # Load into memory

    print("  Adding predictor channels...")
    meg_raw.add_channels([predictor_raw], force_update_info=True)

    # Save
    print(f"\nSaving TRF FIF file...")
    meg_raw.save(output_file, overwrite=True)

    print(f"✓ Saved: {output_file}")
    print(f"  Total channels: {len(meg_raw.ch_names)}")
    print(f"  MEG channels: {len(mne.pick_types(meg_raw.info, meg=True))}")
    print(f"  MISC channels: {len(mne.pick_types(meg_raw.info, misc=True))}")

    # Save predictor metadata
    metadata_file = output_dir / f"{subject}_run-{run:02d}_predictor_metadata.json"
    metadata = {
        'subject': subject,
        'run': run,
        'n_words_interviewer': len(words_interviewer),
        'n_words_participant': len(words_participant),
        'n_words_total': len(words_interviewer) + len(words_participant),
        'sync_offset_s': sync_offset,
        'meg_sfreq': meg_sfreq,
        'meg_duration_s': float(meg_times[-1]),
        'predictors': {
            'envelope_interviewer': {
                'source': 'external_audio',
                'min': float(env_interviewer.min()),
                'max': float(env_interviewer.max()),
                'mean': float(env_interviewer.mean()),
            },
            'envelope_participant': {
                'source': 'external_audio',
                'min': float(env_participant.min()),
                'max': float(env_participant.max()),
                'mean': float(env_participant.mean()),
            },
            'envelope_meg_mic7': {
                'source': 'meg_internal',
                'channel': 'MISC 007',
                'min': float(env_meg_mic7.min()),
                'max': float(env_meg_mic7.max()),
                'mean': float(env_meg_mic7.mean()),
            },
            'envelope_meg_mic8': {
                'source': 'meg_internal',
                'channel': 'MISC 008',
                'min': float(env_meg_mic8.min()),
                'max': float(env_meg_mic8.max()),
                'mean': float(env_meg_mic8.mean()),
            },
            'f0_interviewer': {
                'pct_voiced': float(100 * np.sum(f0_interviewer > 0) / len(f0_interviewer)),
                'normalized': True,
                'speaker_masked': True,
                'note': 'Voiced F0 normalized to 0-1, speaker-masked (zeroed when not speaking)',
            },
            'f0_participant': {
                'pct_voiced': float(100 * np.sum(f0_participant > 0) / len(f0_participant)),
                'normalized': True,
                'speaker_masked': True,
                'note': 'Voiced F0 normalized to 0-1, speaker-masked (zeroed when not speaking)',
            },
            'word_onsets_interviewer': {
                'n_onsets': int(np.sum(word_onsets_interviewer > 0)),
                'normalized': False,
                'note': 'Binary delta functions (0 or 1) for interviewer words',
            },
            'word_onsets_participant': {
                'n_onsets': int(np.sum(word_onsets_participant > 0)),
                'normalized': False,
                'note': 'Binary delta functions (0 or 1) for participant words',
            },
            'surprisal_interviewer': {
                'computed': compute_surprisal_flag,
                'normalized': True,
                'conversation_aware': True,
                'raw_stats': surprisal_interviewer_raw_stats if surprisal_interviewer_raw_stats else None,
                'note': 'GPT-2 surprisal in nats, normalized to 0-1. Conversation-aware: each word conditioned on full chronological conversation history (both speakers)' if compute_surprisal_flag else 'Not computed (zeros)',
            },
            'surprisal_participant': {
                'computed': compute_surprisal_flag,
                'normalized': True,
                'conversation_aware': True,
                'raw_stats': surprisal_participant_raw_stats if surprisal_participant_raw_stats else None,
                'note': 'GPT-2 surprisal in nats, normalized to 0-1. Conversation-aware: each word conditioned on full chronological conversation history (both speakers)' if compute_surprisal_flag else 'Not computed (zeros)',
            },
            'duration_interviewer': {
                'normalized': True,
                'raw_stats': duration_interviewer_raw_stats,
                'note': 'Interviewer word durations normalized to 0-1',
            },
            'duration_participant': {
                'normalized': True,
                'raw_stats': duration_participant_raw_stats,
                'note': 'Participant word durations normalized to 0-1',
            },
            'speaker': {
                'pct_silence': float(100 * np.sum(speaker == 0) / len(speaker)),
                'pct_interviewer': float(100 * np.sum(speaker == 1) / len(speaker)),
                'pct_participant': float(100 * np.sum(speaker == 2) / len(speaker)),
                'pct_overlap': float(100 * np.sum(speaker == 3) / len(speaker)),
                'normalized': False,
                'note': 'Categorical: 0=silence, 1=interviewer, 2=participant, 3=overlap',
            },
            'f0_deviation_interviewer': {
                'n_values': int(np.sum(f0_deviation_interviewer != 0)),
                'mean_z': float(np.mean(f0_deviation_interviewer[f0_deviation_interviewer != 0])) if np.any(f0_deviation_interviewer != 0) else 0,
                'std_z': float(np.std(f0_deviation_interviewer[f0_deviation_interviewer != 0])) if np.any(f0_deviation_interviewer != 0) else 0,
                'note': 'Z-scored F0 deviations from speaker mean (prosodic unexpectedness)',
            },
            'f0_deviation_participant': {
                'n_values': int(np.sum(f0_deviation_participant != 0)),
                'mean_z': float(np.mean(f0_deviation_participant[f0_deviation_participant != 0])) if np.any(f0_deviation_participant != 0) else 0,
                'std_z': float(np.std(f0_deviation_participant[f0_deviation_participant != 0])) if np.any(f0_deviation_participant != 0) else 0,
                'note': 'Z-scored F0 deviations from speaker mean (prosodic unexpectedness)',
            },
            'duration_deviation_interviewer': {
                'n_values': int(np.sum(duration_deviation_interviewer != 0)),
                'mean_z': float(np.mean(duration_deviation_interviewer[duration_deviation_interviewer != 0])) if np.any(duration_deviation_interviewer != 0) else 0,
                'std_z': float(np.std(duration_deviation_interviewer[duration_deviation_interviewer != 0])) if np.any(duration_deviation_interviewer != 0) else 0,
                'note': 'Z-scored duration deviations from speaker mean',
            },
            'duration_deviation_participant': {
                'n_values': int(np.sum(duration_deviation_participant != 0)),
                'mean_z': float(np.mean(duration_deviation_participant[duration_deviation_participant != 0])) if np.any(duration_deviation_participant != 0) else 0,
                'std_z': float(np.std(duration_deviation_participant[duration_deviation_participant != 0])) if np.any(duration_deviation_participant != 0) else 0,
                'note': 'Z-scored duration deviations from speaker mean',
            },
            'pause_interviewer': {
                'n_values': int(np.sum(pause_interviewer != 0)),
                'normalized': True,
                'note': 'Time since last interviewer word (inter-word interval), normalized to 0-1',
            },
            'pause_participant': {
                'n_values': int(np.sum(pause_participant != 0)),
                'normalized': True,
                'note': 'Time since last participant word (inter-word interval), normalized to 0-1',
            },
        },
    }

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved metadata: {metadata_file}")
    print()

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Create TRF analysis FIF files with predictor channels'
    )
    parser.add_argument(
        '--subject',
        nargs='+',
        help='Subject IDs (e.g., sub-01 sub-02)'
    )
    parser.add_argument(
        '--runs',
        nargs='+',
        type=int,
        help='Run numbers (e.g., 1 2 3)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all subjects and runs'
    )
    parser.add_argument(
        '--no-surprisal',
        action='store_true',
        help='Skip GPT-2 surprisal computation (faster)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing TRF FIF files'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging("create_trf_fif")

    # Base directory
    base_dir = Path(__file__).parent.parent

    # Load config
    config = load_config()

    # Determine subjects and runs
    if args.all:
        subjects = ALL_SUBJECTS
        runs = [1, 2, 3, 4, 5]
    else:
        if not args.subject:
            print("Error: Must specify --subject or --all")
            sys.exit(1)
        subjects = args.subject
        runs = args.runs if args.runs else [1]

    print("="*70)
    print("CREATE TRF FIF FILES")
    print("="*70)
    print(f"\nSubjects: {', '.join(subjects)}")
    print(f"Runs: {', '.join(map(str, runs))}")
    print(f"Compute surprisal: {not args.no_surprisal}")
    print(f"Overwrite: {args.overwrite}")
    print(f"\nTotal files: {len(subjects)} × {len(runs)} = {len(subjects) * len(runs)}")

    # Process each subject/run
    results = []
    for subject in subjects:
        for run in runs:
            try:
                success = process_subject_run(
                    subject=subject,
                    run=run,
                    base_dir=base_dir,
                    config=config,
                    compute_surprisal_flag=not args.no_surprisal,
                    overwrite=args.overwrite,
                )
                results.append({
                    'subject': subject,
                    'run': run,
                    'success': success
                })
            except Exception as e:
                print(f"\n✗ Error processing {subject} run-{run}: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'subject': subject,
                    'run': run,
                    'success': False
                })

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    n_success = sum(1 for r in results if r['success'])
    n_fail = sum(1 for r in results if not r['success'])

    print(f"\nTotal: {len(results)}")
    print(f"  ✓ Success: {n_success}")
    print(f"  ✗ Failed: {n_fail}")

    if n_fail > 0:
        print(f"\nFailed:")
        for r in results:
            if not r['success']:
                print(f"  - {r['subject']} run-{r['run']}")

    print()


if __name__ == "__main__":
    main()
