#!/usr/bin/env python3
"""
Create TRF Analysis FIF Files

Adds TRF predictor time series as MISC channels to MEG .fif files.
All predictors are at MEG sampling rate (1000 Hz) in MEG timebase.

Predictors added:
- MISC_envelope_interviewer: Interviewer audio envelope
- MISC_envelope_participant: Participant audio envelope
- MISC_f0_interviewer: Interviewer F0 contour
- MISC_f0_participant: Participant F0 contour
- MISC_word_onsets: Delta functions at MFA word onsets
- MISC_surprisal: Delta functions weighted by GPT-2 surprisal
- MISC_duration: Delta functions weighted by word duration
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
    create_f0_predictor,
    create_word_onset_predictor,
    create_surprisal_predictor,
    create_duration_predictor,
    create_speaker_predictor,
    compute_word_surprisal,
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
    print(f"  Interviewer: {len(audio_interviewer)/sr:.1f}s @ {sr} Hz (channel 0)")

    # Participant audio (channel 1 / right)
    audio_participant, sr = load_audio(audio_file_participant, sr=None, channel=1)
    print(f"  Participant: {len(audio_participant)/sr:.1f}s @ {sr} Hz (channel 1)")

    # Load MFA word timing
    print("\nLoading MFA word timing...")
    feature_dir = base_dir / "outputs" / "features" / subject / f"run-{run:02d}"
    mfa_file = feature_dir / "transcript_mfa.csv"

    if not mfa_file.exists():
        print(f"✗ MFA transcript not found: {mfa_file}")
        print(f"  Run: python scripts/run_mfa_alignment.py --subject {subject} --run {run}")
        return False

    mfa_df = pd.read_csv(mfa_file)
    print(f"  Words: {len(mfa_df)}")

    # Extract word data
    word_times_meg = mfa_df['start_meg'].values if 'start_meg' in mfa_df else (mfa_df['start'].values - sync_offset)
    word_durations = mfa_df['duration'].values
    words = mfa_df['word'].tolist()

    # Create predictors
    print("\nCreating predictors...")

    # 1. Envelopes
    print("  1/8: Envelope (interviewer)...")
    env_interviewer = create_envelope_predictor(
        audio_interviewer, sr, meg_times, sync_offset
    )

    print("  2/8: Envelope (participant)...")
    env_participant = create_envelope_predictor(
        audio_participant, sr, meg_times, sync_offset
    )

    # 2. F0
    print("  3/8: F0 (interviewer)...")
    f0_interviewer = create_f0_predictor(
        audio_interviewer, sr, meg_times, sync_offset
    )

    print("  4/8: F0 (participant)...")
    f0_participant = create_f0_predictor(
        audio_participant, sr, meg_times, sync_offset
    )

    # 3. Word onsets
    print("  5/8: Word onsets...")
    word_onsets = create_word_onset_predictor(word_times_meg, meg_times)

    # 4. Surprisal
    if compute_surprisal_flag:
        print("  6/8: Surprisal (GPT-2, this may take a few minutes)...")
        try:
            surprisal_values = compute_word_surprisal(words)
            surprisal = create_surprisal_predictor(
                word_times_meg, surprisal_values, meg_times
            )
        except Exception as e:
            print(f"  ⚠ Warning: Surprisal computation failed: {e}")
            print(f"  Creating zero surprisal predictor")
            surprisal = np.zeros(len(meg_times))
    else:
        print("  6/8: Surprisal (skipped, using zeros)...")
        surprisal = np.zeros(len(meg_times))

    # 5. Duration
    print("  7/8: Duration...")
    duration = create_duration_predictor(
        word_times_meg, word_durations, meg_times
    )

    # 6. Speaker
    print("  8/8: Speaker...")
    speaker = create_speaker_predictor(env_interviewer, env_participant)

    # Create info for new channels
    print("\nAdding predictors as MISC channels...")
    ch_names = [
        'MISC_envelope_interviewer',
        'MISC_envelope_participant',
        'MISC_f0_interviewer',
        'MISC_f0_participant',
        'MISC_word_onsets',
        'MISC_surprisal',
        'MISC_duration',
        'MISC_speaker',
    ]

    info = mne.create_info(ch_names, meg_sfreq, ch_types='misc')

    # Stack predictors
    predictor_data = np.vstack([
        env_interviewer,
        env_participant,
        f0_interviewer,
        f0_participant,
        word_onsets,
        surprisal,
        duration,
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
        'n_words': len(words),
        'sync_offset_s': sync_offset,
        'meg_sfreq': meg_sfreq,
        'meg_duration_s': float(meg_times[-1]),
        'predictors': {
            'envelope_interviewer': {
                'min': float(env_interviewer.min()),
                'max': float(env_interviewer.max()),
                'mean': float(env_interviewer.mean()),
            },
            'envelope_participant': {
                'min': float(env_participant.min()),
                'max': float(env_participant.max()),
                'mean': float(env_participant.mean()),
            },
            'f0_interviewer': {
                'pct_voiced': float(100 * np.sum(f0_interviewer > 0) / len(f0_interviewer)),
            },
            'f0_participant': {
                'pct_voiced': float(100 * np.sum(f0_participant > 0) / len(f0_participant)),
            },
            'word_onsets': {
                'n_onsets': int(np.sum(word_onsets > 0)),
            },
            'surprisal': {
                'computed': compute_surprisal_flag,
                'mean': float(np.mean(surprisal[surprisal > 0])) if np.any(surprisal > 0) else 0.0,
            },
            'speaker': {
                'pct_silence': float(100 * np.sum(speaker == 0) / len(speaker)),
                'pct_interviewer': float(100 * np.sum(speaker == 1) / len(speaker)),
                'pct_participant': float(100 * np.sum(speaker == 2) / len(speaker)),
                'pct_overlap': float(100 * np.sum(speaker == 3) / len(speaker)),
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
