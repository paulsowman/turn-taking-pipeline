#!/usr/bin/env python
"""
Run sync analysis for subjects missing sync params (sub-31, sub-32).
"""

import subprocess
from pathlib import Path

PIPELINE_DIR = Path("/Users/em18033/Library/CloudStorage/OneDrive-AUTUniversity/Projects/turn-taking-pipeline")
VENV_PYTHON = PIPELINE_DIR / "venv" / "bin" / "python"
BIDS_DIR = Path("/Users/em18033/Library/CloudStorage/OneDrive-SharedLibraries-MacquarieUniversity/Natural_Conversations_study - Documents/analysis/natural-conversations-bids/derivatives/mne-bids-pipeline-20240628")

# Subjects and their audio group mappings
SUBJECTS = {
    'sub-31': 'G31',
    'sub-32': 'G32'
}

def run_sync(subject, run):
    """Run synchronization analysis for one subject/run."""
    meg_file = BIDS_DIR / subject / "meg" / f"{subject}_task-conversation_run-{run:02d}_proc-clean_raw.fif"

    if not meg_file.exists():
        print(f"  ⚠️  MEG file not found: {meg_file}")
        return False

    print(f"  Running sync for {subject} run {run}...")

    cmd = [
        str(VENV_PYTHON),
        str(PIPELINE_DIR / "scripts" / "synchronize_audio_meg.py"),
        "--subject", subject,
        "--run", str(run),
        "--meg-file", str(meg_file)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PIPELINE_DIR)

    if result.returncode == 0:
        print(f"  ✓ Success")
        return True
    else:
        print(f"  ❌ Failed:")
        print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
        print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
        return False


def main():
    print("="*70)
    print("SYNC ANALYSIS FOR MISSING SUBJECTS")
    print("="*70)
    print()

    for subject in SUBJECTS:
        print(f"\n{subject}:")
        print("-" * 70)

        # Run sync for runs 1-5
        for run in [1, 2, 3, 4, 5]:
            sync_params = PIPELINE_DIR / f"outputs/sync/{subject}/run-{run:02d}/sync_params.json"

            if sync_params.exists():
                print(f"  Run {run}: Already has sync params, skipping")
            else:
                run_sync(subject, run)

    print("\n" + "="*70)
    print("SYNC ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
