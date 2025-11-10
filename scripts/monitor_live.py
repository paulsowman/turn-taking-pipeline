#!/usr/bin/env python
"""
Live batch processing monitor with current activity and error tracking.
"""

import time
from pathlib import Path
import json
from datetime import datetime
import sys
import subprocess

# Project paths
PIPELINE_DIR = Path("/Users/em18033/Library/CloudStorage/OneDrive-AUTUniversity/Projects/turn-taking-pipeline")


def get_recent_log_files(log_dir, n=10):
    """Get the most recently modified log files."""
    if not log_dir.exists():
        return []

    log_files = list(log_dir.glob("*.log"))
    log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return log_files[:n]


def check_process_running(pid):
    """Check if a process is running."""
    try:
        result = subprocess.run(['ps', '-p', str(pid)],
                              capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False


def tail_file(file_path, n=5):
    """Get last N lines of a file."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            return lines[-n:] if len(lines) > n else lines
    except:
        return []


def detect_errors_in_log(log_file):
    """Scan log file for errors."""
    errors = []
    try:
        with open(log_file, 'r') as f:
            for i, line in enumerate(f):
                lower_line = line.lower()
                if 'error' in lower_line or 'exception' in lower_line or 'traceback' in lower_line:
                    errors.append((i+1, line.strip()))
    except:
        pass
    return errors


def get_log_summary(log_file):
    """Get summary info from log file."""
    info = {
        'file': log_file.name,
        'size': log_file.stat().st_size,
        'modified': datetime.fromtimestamp(log_file.stat().st_mtime),
        'age_seconds': time.time() - log_file.stat().st_mtime,
    }

    # Parse subject and analysis type from filename
    name_parts = log_file.stem.split('_')
    if len(name_parts) >= 2:
        info['subject'] = name_parts[0]
        info['analysis'] = '_'.join(name_parts[1:])

    return info


def show_live_status():
    """Show live status of batch processing."""
    print("\033[2J\033[H", end="")  # Clear screen

    print("="*100)
    print(f"LIVE BATCH PROCESSING MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100)

    # Check if batch process is running
    pid_file = PIPELINE_DIR / "batch_pid.txt"
    batch_running = False
    batch_pid = None

    if pid_file.exists():
        try:
            with open(pid_file, 'r') as f:
                batch_pid = int(f.read().strip())
                batch_running = check_process_running(batch_pid)
        except:
            pass

    print(f"\nBatch Process Status: {'🟢 RUNNING (PID: {})'.format(batch_pid) if batch_running else '🔴 NOT RUNNING'}")

    # Find most recent log directory
    logs_dir = PIPELINE_DIR / "logs"
    if not logs_dir.exists():
        print("\n❌ No logs directory found!")
        return

    batch_dirs = sorted(logs_dir.glob("batch_*"), key=lambda x: x.stat().st_mtime, reverse=True)

    if not batch_dirs:
        print("\n❌ No batch log directories found!")
        return

    current_batch_dir = batch_dirs[0]
    print(f"Current Batch: {current_batch_dir.name}")

    # Get recent log files
    recent_logs = get_recent_log_files(current_batch_dir, n=20)

    if not recent_logs:
        print("\n⚠️  No log files found in current batch directory")
        return

    # Find actively updating files (modified in last 60 seconds)
    active_logs = [log for log in recent_logs if (time.time() - log.stat().st_mtime) < 60]

    print(f"\n{'='*100}")
    print("ACTIVE PROCESSES (updated in last 60 seconds)")
    print("="*100)

    if active_logs:
        for log_file in active_logs[:5]:  # Show top 5 active
            info = get_log_summary(log_file)
            age = info['age_seconds']

            print(f"\n📝 {info['subject']} - {info['analysis']}")
            print(f"   Last update: {age:.0f}s ago ({info['modified'].strftime('%H:%M:%S')})")
            print(f"   Log size: {info['size']:,} bytes")

            # Show last few lines
            last_lines = tail_file(log_file, n=3)
            if last_lines:
                print(f"   Recent output:")
                for line in last_lines:
                    line_clean = line.strip()[:120]  # Limit line length
                    if line_clean:
                        print(f"     {line_clean}")

            # Check for errors
            errors = detect_errors_in_log(log_file)
            if errors:
                print(f"   ⚠️  ERRORS FOUND: {len(errors)} error(s)")
                for line_num, error_line in errors[-2:]:  # Show last 2 errors
                    print(f"     Line {line_num}: {error_line[:100]}")
    else:
        print("\n⏸️  No actively updating log files (nothing has been modified in last 60 seconds)")
        print("   This could mean:")
        print("   - Batch process has not started yet")
        print("   - Batch process completed")
        print("   - Long-running analysis in progress")

    # Show recently completed
    print(f"\n{'='*100}")
    print("RECENTLY COMPLETED (in last 5 minutes)")
    print("="*100)

    recent_completed = [log for log in recent_logs
                       if 60 < (time.time() - log.stat().st_mtime) < 300]

    if recent_completed:
        for log_file in recent_completed[:5]:
            info = get_log_summary(log_file)
            age_min = info['age_seconds'] / 60

            errors = detect_errors_in_log(log_file)
            status = "❌ FAILED" if errors else "✅ SUCCESS"

            print(f"  {status} - {info['subject']} {info['analysis']} ({age_min:.1f}m ago)")
    else:
        print("  None")

    # Overall progress
    print(f"\n{'='*100}")
    print("OVERALL PROGRESS")
    print("="*100)

    # Quick count of completed analyses
    all_logs = list(current_batch_dir.glob("*.log"))
    localizer_mag = len(list(current_batch_dir.glob("*_localizer_mag.log")))
    localizer_eeg = len(list(current_batch_dir.glob("*_localizer_eeg.log")))
    trf_logs = len(list(current_batch_dir.glob("*_trf_*.log")))

    print(f"  Total log files: {len(all_logs)}")
    print(f"  Localizer analyses: {localizer_mag} MAG, {localizer_eeg} EEG")
    print(f"  TRF analyses: {trf_logs}")

    # Check for errors across all logs
    print(f"\n{'='*100}")
    print("ERROR SUMMARY")
    print("="*100)

    error_count = 0
    error_subjects = []

    for log_file in recent_logs[:10]:
        errors = detect_errors_in_log(log_file)
        if errors:
            error_count += len(errors)
            info = get_log_summary(log_file)
            error_subjects.append(f"{info['subject']} ({info['analysis']})")

    if error_count > 0:
        print(f"  ⚠️  {error_count} error(s) found in recent logs")
        print(f"  Affected: {', '.join(error_subjects[:5])}")
    else:
        print(f"  ✅ No errors detected in recent logs")

    print(f"\n{'='*100}")
    print("Press Ctrl+C to stop monitoring")
    print("Refreshing every 10 seconds...")
    print("="*100)


def main():
    try:
        while True:
            show_live_status()
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
