#!/usr/bin/env python3
"""
Check F0 and word onset alignment statistics.

Analyzes how many word onsets have F0 present and provides
phonetic analysis to verify alignment is linguistically reasonable.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
import librosa

try:
    from utils.config import load_config, get_subject_paths
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from utils.config import load_config, get_subject_paths


def load_word_onsets(subject, run, base_dir):
    """Load word onset times from transcript and recalculate MEG times."""
    feature_dir = base_dir / "outputs" / "features" / subject / f"run-{run:02d}"
    transcript = pd.read_csv(feature_dir / "transcript.csv")

    # Load sync offset to recalculate MEG times
    # (Don't trust the start_meg/end_meg in the file - may be calculated with old formula)
    sync_dir = base_dir / "outputs" / "sync" / subject / f"run-{run:02d}"
    sync_params_file = sync_dir / "sync_params.json"

    with open(sync_params_file, 'r') as f:
        import json
        sync_params = json.load(f)

    sync_offset = sync_params['initial_offset_s']

    word_onsets = []
    for _, seg in transcript.iterrows():
        if 'words' not in seg or pd.isna(seg['words']):
            continue

        words_data = eval(seg['words']) if isinstance(seg['words'], str) else seg['words']

        for w in words_data:
            word_time_audio = w['start']
            # CORRECT formula: meg_time = audio_time - offset
            # This accounts for the REVERSED sign convention in sync function
            word_time_meg = word_time_audio - sync_offset
            word_onsets.append({
                'time_meg': word_time_meg,
                'time_audio': word_time_audio,
                'word': w['word'].strip()
            })

    return pd.DataFrame(word_onsets)


def compute_f0(audio, sr, hop_length_ms=10):
    """Compute F0."""
    hop_length = int(hop_length_ms * sr / 1000)

    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        sr=sr,
        fmin=80,
        fmax=400,
        hop_length=hop_length,
        frame_length=int(0.025 * sr),
        fill_na=0.0
    )
    f0_times = np.arange(len(f0)) * hop_length / sr

    return f0, f0_times


def classify_consonant(word):
    """Classify if word starts with voiced or unvoiced consonant."""
    word = word.lower().strip()

    # Unvoiced consonants at start
    unvoiced_starts = ['p', 't', 'k', 'f', 's', 'sh', 'th', 'ch', 'h']
    # Voiced consonants
    voiced_starts = ['b', 'd', 'g', 'v', 'z', 'j', 'm', 'n', 'l', 'r', 'w', 'y']
    # Vowels
    vowels = ['a', 'e', 'i', 'o', 'u']

    if not word:
        return "unknown"

    # Check first character
    if word[0] in vowels:
        return "vowel"
    elif word[0] in unvoiced_starts:
        return "unvoiced"
    elif word[0] in voiced_starts:
        return "voiced"
    # Check two-character starts
    elif len(word) >= 2:
        if word[:2] in ['sh', 'ch', 'th']:
            return "unvoiced"

    return "unknown"


def analyze_f0_word_alignment(subject, run, base_dir, window_ms=50):
    """
    Analyze F0 presence at word onsets.

    Parameters
    ----------
    window_ms : float
        Time window (ms) around word onset to check for F0
    """
    print(f"\n{'='*70}")
    print(f"F0-WORD ONSET ALIGNMENT ANALYSIS: {subject} run-{run:02d}")
    print(f"{'='*70}\n")

    # Load config and paths
    config = load_config()
    paths = get_subject_paths(subject, run, config)

    # Load sync parameters
    sync_dir = base_dir / "outputs" / "sync" / subject / f"run-{run:02d}"
    sync_params_file = sync_dir / "sync_params.json"

    with open(sync_params_file, 'r') as f:
        sync_params = json.load(f)
    sync_offset = sync_params['initial_offset_s']

    # Load audio
    print("Loading audio...")
    audio_file = paths['external_audio_interviewer']
    # Use channel 0 (left) for console_mic files - interviewer only
    sys.path.insert(0, str(base_dir / "src"))
    from utils.io import load_audio
    audio, sr = load_audio(audio_file, sr=None, channel=0)

    # Compute F0
    print("Computing F0...")
    f0, f0_times_audio = compute_f0(audio, sr)

    # Convert F0 times to MEG timebase
    f0_times_meg = f0_times_audio - sync_offset

    # F0 statistics
    voiced_frames = f0 > 0
    n_voiced = np.sum(voiced_frames)
    n_total = len(f0)
    voicing_rate = n_voiced / n_total

    print(f"  Total F0 frames: {n_total}")
    print(f"  Voiced frames: {n_voiced} ({voicing_rate*100:.1f}%)")
    print(f"  F0 range: {np.min(f0[voiced_frames]):.1f} - {np.max(f0[voiced_frames]):.1f} Hz")
    print(f"  F0 mean: {np.mean(f0[voiced_frames]):.1f} Hz")
    print()

    # Load word onsets
    print("Loading word onsets...")
    words_df = load_word_onsets(subject, run, base_dir)
    print(f"  Found {len(words_df)} words\n")

    # Analyze each word
    window_s = window_ms / 1000
    results = []

    for _, word_row in words_df.iterrows():
        word_time = word_row['time_meg']
        word_text = word_row['word']

        # Find F0 frames within window of word onset
        # Check slightly AFTER onset (0 to +window_ms) since consonant comes first
        in_window = (f0_times_meg >= word_time) & (f0_times_meg <= word_time + window_s)

        if np.any(in_window):
            f0_values = f0[in_window]
            has_f0 = np.any(f0_values > 0)

            if has_f0:
                mean_f0 = np.mean(f0_values[f0_values > 0])
                max_f0 = np.max(f0_values)
            else:
                mean_f0 = 0
                max_f0 = 0
        else:
            has_f0 = False
            mean_f0 = 0
            max_f0 = 0

        # Classify word onset
        onset_type = classify_consonant(word_text)

        results.append({
            'word': word_text,
            'time_meg': word_time,
            'has_f0': has_f0,
            'mean_f0': mean_f0,
            'max_f0': max_f0,
            'onset_type': onset_type
        })

    results_df = pd.DataFrame(results)

    # Summary statistics
    print("="*70)
    print("ALIGNMENT STATISTICS")
    print("="*70)

    n_words = len(results_df)
    n_with_f0 = results_df['has_f0'].sum()
    pct_with_f0 = 100 * n_with_f0 / n_words

    print(f"\nOverall:")
    print(f"  Total words: {n_words}")
    print(f"  Words with F0 within {window_ms}ms: {n_with_f0} ({pct_with_f0:.1f}%)")
    print(f"  Words without F0: {n_words - n_with_f0} ({100-pct_with_f0:.1f}%)")

    # By onset type
    print(f"\nBy onset type (within {window_ms}ms window):")
    for onset_type in ['vowel', 'voiced', 'unvoiced', 'unknown']:
        subset = results_df[results_df['onset_type'] == onset_type]
        if len(subset) > 0:
            n_type = len(subset)
            n_type_f0 = subset['has_f0'].sum()
            pct_type_f0 = 100 * n_type_f0 / n_type
            print(f"  {onset_type:10s}: {n_type:3d} words, {n_type_f0:3d} with F0 ({pct_type_f0:5.1f}%)")

    # Expected vs observed
    print(f"\nExpected behavior:")
    print(f"  - Vowel-initial words: Should have F0 immediately (~100%)")
    print(f"  - Voiced consonant words: Should have F0 within ~50ms (~80-100%)")
    print(f"  - Unvoiced consonant words: Should NOT have F0 initially (~0-20%)")
    print(f"    (F0 appears later when vowel starts)")

    # Examples of misalignment concerns
    vowel_initial_no_f0 = results_df[
        (results_df['onset_type'] == 'vowel') & (~results_df['has_f0'])
    ]

    if len(vowel_initial_no_f0) > 0:
        print(f"\n⚠ POTENTIAL ISSUES:")
        print(f"  Found {len(vowel_initial_no_f0)} vowel-initial words WITHOUT F0")
        print(f"  (Should be rare - may indicate misalignment or low energy)")
        print(f"\n  Examples:")
        for _, row in vowel_initial_no_f0.head(5).iterrows():
            print(f"    '{row['word']}' @ {row['time_meg']:.2f}s")
    else:
        print(f"\n✓ All vowel-initial words have F0 - alignment looks good!")

    # Sample words for manual verification
    print(f"\n{'='*70}")
    print("SAMPLE WORDS FOR MANUAL VERIFICATION:")
    print(f"{'='*70}\n")

    # Show a few examples of each type
    print("Vowel-initial (should have immediate F0):")
    vowel_words = results_df[results_df['onset_type'] == 'vowel'].head(3)
    for _, row in vowel_words.iterrows():
        f0_status = f"F0={row['mean_f0']:.0f}Hz" if row['has_f0'] else "NO F0"
        print(f"  '{row['word']:15s}' @ {row['time_meg']:6.2f}s  [{f0_status}]")

    print("\nUnvoiced-initial (should NOT have immediate F0):")
    unvoiced_words = results_df[results_df['onset_type'] == 'unvoiced'].head(3)
    for _, row in unvoiced_words.iterrows():
        f0_status = f"F0={row['mean_f0']:.0f}Hz" if row['has_f0'] else "NO F0"
        print(f"  '{row['word']:15s}' @ {row['time_meg']:6.2f}s  [{f0_status}]")

    print(f"\n{'='*70}\n")

    return results_df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Analyze F0-word onset alignment')
    parser.add_argument('--subject', type=str, required=True, help='Subject ID (e.g., sub-01)')
    parser.add_argument('--run', type=int, required=True, help='Run number (1-5)')
    parser.add_argument('--window-ms', type=float, default=50,
                       help='Time window after onset to check for F0 (ms, default: 50)')

    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent

    results_df = analyze_f0_word_alignment(
        args.subject,
        args.run,
        base_dir,
        window_ms=args.window_ms
    )

    # Optionally save results
    output_dir = base_dir / "outputs" / "sync" / args.subject / f"run-{args.run:02d}"
    output_file = output_dir / "f0_word_alignment_check.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Saved detailed results to: {output_file}")
