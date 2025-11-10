#!/usr/bin/env python3
"""
Cleanup redundant development and debugging files.

Removes:
- Debugging/development scripts
- Old backup sync parameter files
- Redundant visualization files

Preserves:
- sync_params.json (CRITICAL for TRF!)
- Current QC plots (envelope_alignment.png, waveform_alignment.png, drift_correction.png, sync_qc_report.txt)
- Core pipeline files
"""

import argparse
from pathlib import Path
import shutil
from typing import List, Tuple


def find_files_to_remove(base_dir: Path, dry_run: bool = True) -> Tuple[List[Path], int]:
    """
    Find all files to be removed.

    Returns:
        Tuple of (list of files, total size in bytes)
    """
    files_to_remove = []
    total_size = 0

    # 1. Debugging scripts in scripts/
    script_dir = base_dir / "scripts"
    debug_scripts = [
        "check_basic_alignment.py",
        "batch_alignment_check.py",
        "verify_sync_alignment.py",
        "find_poor_sync.py",
        "compare_sync_improvements.py",
        "batch_regenerate_qc.py",
    ]

    for script_name in debug_scripts:
        script_path = script_dir / script_name
        if script_path.exists():
            size = script_path.stat().st_size
            files_to_remove.append(script_path)
            total_size += size
            print(f"  📄 {script_path.relative_to(base_dir)} ({size/1024:.1f} KB)")

    # 2. Old backup sync params in outputs/sync/
    sync_dir = base_dir / "outputs" / "sync"
    if sync_dir.exists():
        backup_patterns = [
            "**/sync_params_old.json",
            "**/sync_params_chunked.json",
        ]

        for pattern in backup_patterns:
            for backup_file in sync_dir.glob(pattern):
                size = backup_file.stat().st_size
                files_to_remove.append(backup_file)
                total_size += size
                print(f"  📄 {backup_file.relative_to(base_dir)} ({size/1024:.1f} KB)")

    # 3. Old/redundant visualization files in outputs/sync/
    if sync_dir.exists():
        redundant_viz_patterns = [
            "**/basic_alignment_check.png",
        ]

        for pattern in redundant_viz_patterns:
            for viz_file in sync_dir.glob(pattern):
                size = viz_file.stat().st_size
                files_to_remove.append(viz_file)
                total_size += size
                print(f"  🖼️  {viz_file.relative_to(base_dir)} ({size/1024:.1f} KB)")

    # 4. Old log files (keep only recent ones)
    logs_dir = base_dir / "logs"
    if logs_dir.exists():
        # Keep the most recent log file for each type
        log_patterns = {
            "batch_sync_hybrid_*.log": 1,  # Keep 1 most recent
            "batch_alignment_check_*.log": 0,  # Remove all (debugging)
            "verify_sync_*.log": 0,  # Remove all (debugging)
        }

        for pattern, keep_count in log_patterns.items():
            log_files = sorted(logs_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

            # Remove older logs beyond keep_count
            for log_file in log_files[keep_count:]:
                size = log_file.stat().st_size
                files_to_remove.append(log_file)
                total_size += size
                print(f"  📝 {log_file.relative_to(base_dir)} ({size/1024:.1f} KB)")

    return files_to_remove, total_size


def main():
    parser = argparse.ArgumentParser(
        description="Cleanup redundant development and debugging files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
SAFETY FEATURES:
  - Dry-run mode by default (use --execute to actually delete)
  - Shows all files before deletion
  - Preserves critical files (sync_params.json, current QC plots)
  - Provides summary of space to be saved

PRESERVED FILES:
  - sync_params.json (all recordings)
  - envelope_alignment.png, waveform_alignment.png, drift_correction.png
  - sync_qc_report.txt
  - Core pipeline scripts (batch_audio_meg_sync.py, etc.)

REMOVED FILES:
  - Debugging scripts (check_basic_alignment.py, etc.)
  - Old backup sync params (*_old.json, *_chunked.json)
  - Redundant visualizations (basic_alignment_check.png)
  - Old log files (keeping only most recent)
        """
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete files (default is dry-run)"
    )

    args = parser.parse_args()

    base_dir = Path("/Users/em18033/Library/CloudStorage/OneDrive-AUTUniversity/Projects/turn-taking-pipeline")

    if not base_dir.exists():
        print(f"Error: Base directory not found: {base_dir}")
        return 1

    print("=" * 70)
    print("CLEANUP REDUNDANT FILES")
    print("=" * 70)

    if args.execute:
        print("⚠️  EXECUTE MODE: Files will be PERMANENTLY DELETED")
    else:
        print("🔍 DRY-RUN MODE: No files will be deleted (use --execute to delete)")

    print("=" * 70)
    print("\nScanning for redundant files...\n")

    # Find files
    files_to_remove, total_size = find_files_to_remove(base_dir, dry_run=not args.execute)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Files to remove: {len(files_to_remove)}")
    print(f"Total size:      {total_size / (1024*1024):.2f} MB")
    print("=" * 70)

    if not files_to_remove:
        print("\n✓ No redundant files found!")
        return 0

    # Execute deletion if requested
    if args.execute:
        print("\n⚠️  Deleting files...")

        deleted_count = 0
        failed_count = 0

        for file_path in files_to_remove:
            try:
                file_path.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"\n✗ Failed to delete {file_path.relative_to(base_dir)}: {e}")
                failed_count += 1

        print("\n" + "=" * 70)
        print("DELETION COMPLETE")
        print("=" * 70)
        print(f"Successfully deleted: {deleted_count} files")
        print(f"Failed:              {failed_count} files")
        print(f"Space freed:         {total_size / (1024*1024):.2f} MB")
        print("=" * 70)
    else:
        print("\n💡 To actually delete these files, run:")
        print(f"   python {Path(__file__).name} --execute")

    return 0


if __name__ == "__main__":
    exit(main())
