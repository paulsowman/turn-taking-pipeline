#!/usr/bin/env python3
"""
Visualize turn-taking structure from pipeline outputs.

Creates an interleaved transcript showing detected turns with timestamps,
useful for sense-checking that turn detection is working correctly.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def load_turn_data(subject: str, run: int, base_dir: Path):
    """Load transcript and turn data."""
    feature_dir = base_dir / "outputs" / "features" / subject / f"run-{run:02d}"

    transcript = pd.read_csv(feature_dir / "transcript.csv")
    turns = pd.read_csv(feature_dir / "turns.csv")

    return transcript, turns


def create_interleaved_transcript(transcript, turns, time_col="start_meg"):
    """
    Create interleaved transcript with turn structure.

    Parameters
    ----------
    transcript : pd.DataFrame
        Word-level transcript
    turns : pd.DataFrame
        Turn segmentation
    time_col : str
        Time column to use ('start_meg' or 'start')

    Returns
    -------
    lines : list of str
        Formatted transcript lines
    """
    lines = []
    lines.append("="*80)
    lines.append("INTERLEAVED TURN TRANSCRIPT")
    lines.append("="*80)
    lines.append("")

    # Add metadata
    lines.append(f"Total turns: {len(turns)}")
    lines.append(f"Total segments: {len(transcript)}")

    if 'words' in transcript.columns:
        total_words = transcript['words'].apply(lambda x: len(eval(x)) if isinstance(x, str) else 0).sum()
        lines.append(f"Total words: {total_words}")

    lines.append("")
    lines.append("="*80)
    lines.append("")

    # Process each turn
    for turn_idx, turn in turns.iterrows():
        # Turn header
        speaker = turn.get('speaker', 'Unknown')
        start_time = turn[time_col] if time_col in turn else turn['start']
        end_time = turn.get(f'end_meg' if 'meg' in time_col else 'end', start_time + turn['duration'])
        duration = turn['duration']
        n_words = turn.get('n_words', 0)

        lines.append(f"TURN {turn_idx + 1} | Speaker: {speaker} | "
                    f"Time: {start_time:.2f}s - {end_time:.2f}s ({duration:.2f}s) | "
                    f"Words: {n_words}")
        lines.append("-"*80)

        # Get segments for this turn
        segment_ids = eval(turn['segment_ids']) if isinstance(turn['segment_ids'], str) else turn['segment_ids']

        for seg_id in segment_ids:
            seg = transcript[transcript['segment_id'] == seg_id].iloc[0]

            seg_start = seg[time_col] if time_col in seg else seg['start']
            seg_text = seg['text']

            lines.append(f"  [{seg_start:7.2f}s] {seg_text}")

            # Show words if available
            if 'words' in seg and pd.notna(seg['words']):
                words = eval(seg['words']) if isinstance(seg['words'], str) else seg['words']
                if words:
                    word_strs = []
                    for w in words:
                        w_start = w['start'] - (seg['start'] - seg_start)  # Adjust to MEG time
                        word_strs.append(f"{w['word']}({w_start:.2f})")
                    lines.append(f"    Words: {' '.join(word_strs[:10])}")  # Show first 10
                    if len(words) > 10:
                        lines.append(f"           ... and {len(words) - 10} more")

        lines.append("")

    return lines


def analyze_turn_statistics(turns, transcript):
    """Analyze turn-taking statistics."""
    stats = {}

    # Basic counts
    stats['n_turns'] = len(turns)
    stats['n_segments'] = len(transcript)

    # Duration stats
    stats['total_duration'] = turns['duration'].sum()
    stats['mean_turn_duration'] = turns['duration'].mean()
    stats['median_turn_duration'] = turns['duration'].median()
    stats['min_turn_duration'] = turns['duration'].min()
    stats['max_turn_duration'] = turns['duration'].max()

    # Word stats
    if 'n_words' in turns.columns:
        stats['total_words'] = turns['n_words'].sum()
        stats['mean_words_per_turn'] = turns['n_words'].mean()
        stats['speech_rate_words_per_sec'] = stats['total_words'] / stats['total_duration']

    # Gap analysis (time between turns)
    if 'start_meg' in turns.columns:
        time_col = 'start_meg'
        end_col = 'end_meg'
    else:
        time_col = 'start'
        end_col = 'end'

    gaps = []
    for i in range(len(turns) - 1):
        gap = turns.iloc[i + 1][time_col] - turns.iloc[i][end_col]
        gaps.append(gap)

    if gaps:
        stats['mean_gap'] = np.mean(gaps)
        stats['median_gap'] = np.median(gaps)
        stats['overlaps'] = sum(1 for g in gaps if g < 0)
        stats['overlap_rate'] = stats['overlaps'] / len(gaps)

    return stats


def print_statistics(stats):
    """Print turn-taking statistics."""
    print("\n" + "="*80)
    print("TURN-TAKING STATISTICS")
    print("="*80)
    print(f"\nCounts:")
    print(f"  Turns: {stats['n_turns']}")
    print(f"  Segments: {stats['n_segments']}")

    if 'total_words' in stats:
        print(f"  Total words: {stats['total_words']:.0f}")

    print(f"\nDuration:")
    print(f"  Total speech time: {stats['total_duration']:.1f}s")
    print(f"  Mean turn duration: {stats['mean_turn_duration']:.2f}s")
    print(f"  Median turn duration: {stats['median_turn_duration']:.2f}s")
    print(f"  Range: {stats['min_turn_duration']:.2f}s - {stats['max_turn_duration']:.2f}s")

    if 'speech_rate_words_per_sec' in stats:
        print(f"\nSpeech rate:")
        print(f"  Words per second: {stats['speech_rate_words_per_sec']:.2f}")
        print(f"  Words per turn: {stats['mean_words_per_turn']:.1f}")

    if 'mean_gap' in stats:
        print(f"\nTurn transitions:")
        print(f"  Mean gap: {stats['mean_gap']:.3f}s")
        print(f"  Median gap: {stats['median_gap']:.3f}s")
        print(f"  Overlaps: {stats['overlaps']} ({stats['overlap_rate']*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Visualize turn-taking structure")
    parser.add_argument("--subject", required=True, help="Subject ID (e.g., sub-01)")
    parser.add_argument("--run", type=int, required=True, help="Run number")
    parser.add_argument("--base-dir", type=Path,
                       default=Path("/Users/em18033/Library/CloudStorage/OneDrive-AUTUniversity/Projects/turn-taking-pipeline"),
                       help="Pipeline base directory")
    parser.add_argument("--output", type=Path, help="Output file (default: print to screen)")
    parser.add_argument("--use-original-time", action="store_true",
                       help="Use original audio time instead of MEG time")

    args = parser.parse_args()

    # Load data
    print(f"Loading turn data for {args.subject} run {args.run}...")
    transcript, turns = load_turn_data(args.subject, args.run, args.base_dir)

    # Analyze statistics
    stats = analyze_turn_statistics(turns, transcript)
    print_statistics(stats)

    # Create transcript
    print("\nGenerating interleaved transcript...")
    time_col = "start" if args.use_original_time else "start_meg"
    lines = create_interleaved_transcript(transcript, turns, time_col)

    # Output
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            f.write('\n'.join(lines))
        print(f"\n✓ Transcript saved to: {args.output}")
    else:
        print("\n")
        for line in lines[:100]:  # Print first 100 lines to screen
            print(line)

        if len(lines) > 100:
            print(f"\n... (showing first 100 of {len(lines)} lines)")
            print(f"\nUse --output to save full transcript to file")


if __name__ == "__main__":
    main()
