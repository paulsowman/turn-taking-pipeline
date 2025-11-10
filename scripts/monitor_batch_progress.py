#!/usr/bin/env python
"""
Monitor batch processing progress in real-time.
Continuously checks for completed analyses and displays status.
"""

import time
from pathlib import Path
import json
from datetime import datetime
import sys

# Project paths
PIPELINE_DIR = Path("/Users/em18033/Library/CloudStorage/OneDrive-AUTUniversity/Projects/turn-taking-pipeline")

# All subjects to monitor
ALL_SUBJECTS = [
    'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05',
    'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10',
    'sub-11', 'sub-13', 'sub-14', 'sub-15', 'sub-16',
    'sub-17', 'sub-18', 'sub-19', 'sub-21', 'sub-22',
    'sub-23', 'sub-24', 'sub-25', 'sub-26', 'sub-27',
    'sub-29', 'sub-31', 'sub-32'
]


def check_file_exists(file_path):
    """Check if a file exists."""
    return Path(file_path).exists()


def get_file_mtime(file_path):
    """Get file modification time."""
    p = Path(file_path)
    if p.exists():
        return datetime.fromtimestamp(p.stat().st_mtime)
    return None


def check_subject_status(subject):
    """Check completion status for a subject."""
    status = {
        'subject': subject,
        'localizer_mag': False,
        'localizer_eeg': False,
        'trf_135_mag': False,
        'trf_135_eeg': False,
        'trf_24_mag': False,
        'trf_24_eeg': False,
        'localizer_mag_time': None,
        'localizer_eeg_time': None,
        'trf_135_mag_time': None,
        'trf_135_eeg_time': None,
        'trf_24_mag_time': None,
        'trf_24_eeg_time': None,
    }

    # Check localizer
    roi_mag = PIPELINE_DIR / f"outputs/bada_localizer/{subject}/roi_sensors_mag.json"
    roi_eeg = PIPELINE_DIR / f"outputs/bada_localizer/{subject}/roi_sensors_eeg.json"

    if roi_mag.exists():
        status['localizer_mag'] = True
        status['localizer_mag_time'] = get_file_mtime(roi_mag)

    if roi_eeg.exists():
        status['localizer_eeg'] = True
        status['localizer_eeg_time'] = get_file_mtime(roi_eeg)

    # Check TRF analyses
    summary_135_mag = PIPELINE_DIR / f"outputs/trf_multipredictor/{subject}/runs-1_3_5/mag/summary.json"
    summary_135_eeg = PIPELINE_DIR / f"outputs/trf_multipredictor/{subject}/runs-1_3_5/eeg/summary.json"
    summary_24_mag = PIPELINE_DIR / f"outputs/trf_multipredictor/{subject}/runs-2_4/mag/summary.json"
    summary_24_eeg = PIPELINE_DIR / f"outputs/trf_multipredictor/{subject}/runs-2_4/eeg/summary.json"

    if summary_135_mag.exists():
        status['trf_135_mag'] = True
        status['trf_135_mag_time'] = get_file_mtime(summary_135_mag)

    if summary_135_eeg.exists():
        status['trf_135_eeg'] = True
        status['trf_135_eeg_time'] = get_file_mtime(summary_135_eeg)

    if summary_24_mag.exists():
        status['trf_24_mag'] = True
        status['trf_24_mag_time'] = get_file_mtime(summary_24_mag)

    if summary_24_eeg.exists():
        status['trf_24_eeg'] = True
        status['trf_24_eeg_time'] = get_file_mtime(summary_24_eeg)

    return status


def print_status_table(all_status, show_times=False):
    """Print a formatted status table."""
    print("\n" + "="*120)
    print(f"BATCH PROCESSING PROGRESS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*120)

    # Header
    print(f"{'Subject':<10} {'Localizer':<15} {'TRF 1+3+5':<15} {'TRF 2+4':<15} {'Complete':<10} {'Recent Activity':<30}")
    print("-"*120)

    total_tasks = 0
    completed_tasks = 0
    recent_activities = []

    for status in all_status:
        subject = status['subject']

        # Count tasks
        tasks = [
            status['localizer_mag'],
            status['localizer_eeg'],
            status['trf_135_mag'],
            status['trf_135_eeg'],
            status['trf_24_mag'],
            status['trf_24_eeg']
        ]

        total_tasks += 6
        completed_tasks += sum(tasks)

        # Localizer status
        loc_mag = "✓" if status['localizer_mag'] else "✗"
        loc_eeg = "✓" if status['localizer_eeg'] else "✗"
        loc_status = f"MAG:{loc_mag} EEG:{loc_eeg}"

        # TRF 1+3+5 status
        trf_135_mag = "✓" if status['trf_135_mag'] else "✗"
        trf_135_eeg = "✓" if status['trf_135_eeg'] else "✗"
        trf_135_status = f"MAG:{trf_135_mag} EEG:{trf_135_eeg}"

        # TRF 2+4 status
        trf_24_mag = "✓" if status['trf_24_mag'] else "✗"
        trf_24_eeg = "✓" if status['trf_24_eeg'] else "✗"
        trf_24_status = f"MAG:{trf_24_mag} EEG:{trf_24_eeg}"

        # Overall completion
        complete_count = sum(tasks)
        complete_pct = f"{complete_count}/6"

        # Find most recent activity
        times = [
            ('LOC-MAG', status['localizer_mag_time']),
            ('LOC-EEG', status['localizer_eeg_time']),
            ('135-MAG', status['trf_135_mag_time']),
            ('135-EEG', status['trf_135_eeg_time']),
            ('24-MAG', status['trf_24_mag_time']),
            ('24-EEG', status['trf_24_eeg_time']),
        ]
        times = [(name, t) for name, t in times if t is not None]

        if times:
            recent = max(times, key=lambda x: x[1])
            recent_str = f"{recent[0]} {recent[1].strftime('%H:%M')}"
            recent_activities.append((subject, recent[0], recent[1]))
        else:
            recent_str = "-"

        print(f"{subject:<10} {loc_status:<15} {trf_135_status:<15} {trf_24_status:<15} {complete_pct:<10} {recent_str:<30}")

    print("-"*120)

    # Overall progress
    overall_pct = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
    print(f"\nOVERALL PROGRESS: {completed_tasks}/{total_tasks} tasks ({overall_pct:.1f}%)")

    # Progress bar
    bar_length = 50
    filled = int(bar_length * completed_tasks / total_tasks)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"[{bar}] {overall_pct:.1f}%")

    # Recent activities (last 5)
    if recent_activities:
        print("\nRECENT COMPLETIONS:")
        recent_activities.sort(key=lambda x: x[2], reverse=True)
        for subj, task, time in recent_activities[:5]:
            print(f"  {subj} {task}: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    print()


def monitor_progress(subjects=None, interval=30, show_times=False):
    """Monitor progress continuously."""
    if subjects is None:
        subjects = ALL_SUBJECTS

    try:
        while True:
            all_status = [check_subject_status(s) for s in subjects]

            # Clear screen (works on Unix-like systems)
            print("\033[2J\033[H", end="")

            print_status_table(all_status, show_times)

            print(f"Refreshing every {interval} seconds. Press Ctrl+C to stop.")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        sys.exit(0)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Monitor batch processing progress')
    parser.add_argument('--subjects', nargs='+', default=None, help='Specific subjects to monitor')
    parser.add_argument('--interval', type=int, default=30, help='Refresh interval in seconds (default: 30)')
    parser.add_argument('--once', action='store_true', help='Show status once and exit')
    parser.add_argument('--show-times', action='store_true', help='Show completion times')

    args = parser.parse_args()

    subjects = args.subjects if args.subjects else ALL_SUBJECTS

    if args.once:
        all_status = [check_subject_status(s) for s in subjects]
        print_status_table(all_status, args.show_times)
    else:
        monitor_progress(subjects, args.interval, args.show_times)


if __name__ == "__main__":
    main()
