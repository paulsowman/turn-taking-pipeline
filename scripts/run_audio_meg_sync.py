#!/usr/bin/env python3
"""
CLI wrapper for audio-MEG synchronization.
Creates sync params for TRF analyses.
"""

import argparse
import mne
import json
from pathlib import Path
import sys
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sync.audio_meg_sync import synchronize_audio_meg
from src.qc.sync_qc import plot_sync_diagnostics, generate_sync_report
import librosa


def main():
    parser = argparse.ArgumentParser(description='Synchronize external audio with MEG')
    parser.add_argument('--subject', required=True, help='Subject ID (e.g., sub-01)')
    parser.add_argument('--run', required=True, type=int, help='Run number (1-5)')
    parser.add_argument('--meg-file', required=True, help='Path to MEG file')

    args = parser.parse_args()

    # Map subject to audio group
    subject_to_group = {
        'sub-01': 'G01', 'sub-02': 'G02', 'sub-03': 'G03', 'sub-04': 'G04', 'sub-05': 'G05',
        'sub-06': 'G06', 'sub-07': 'G07', 'sub-08': 'G08', 'sub-09': 'G09', 'sub-10': 'G10',
        'sub-11': 'G11', 'sub-12': 'G12', 'sub-13': 'G13', 'sub-14': 'G14', 'sub-15': 'G15',
        'sub-16': 'G16', 'sub-17': 'G17', 'sub-18': 'G18', 'sub-19': 'G19', 'sub-20': 'G20',
        'sub-21': 'G21', 'sub-22': 'G22', 'sub-23': 'G23', 'sub-24': 'G24', 'sub-25': 'G25',
        'sub-26': 'G26', 'sub-27': 'G27', 'sub-28': 'G28', 'sub-29': 'G29', 'sub-30': 'G30',
        'sub-31': 'G31', 'sub-32': 'G32'
    }

    audio_group = subject_to_group.get(args.subject)
    if not audio_group:
        print(f"ERROR: Unknown subject {args.subject}")
        sys.exit(1)

    # Build audio path
    audio_base = Path.home() / "Library/CloudStorage/OneDrive-AUTUniversity/Projects/Conversational_AI/Archive/data/audios"
    external_audio_path = audio_base / audio_group / f"console_mic_B{args.run}.wav"

    # Load MEG
    print(f"Loading MEG data: {args.meg_file}")
    meg_raw = mne.io.read_raw_fif(args.meg_file, preload=True, verbose=False)

    if not external_audio_path.exists():
        print(f"ERROR: Audio file not found: {external_audio_path}")
        sys.exit(1)

    # Default sync config
    config = {
        'sync': {
            'max_shift_samples': 50000,
            'correlation_threshold': 0.3
        }
    }

    # Run sync
    print(f"Synchronizing audio for {args.subject} run-{args.run:02d}...")
    sync_params = synchronize_audio_meg(
        meg_raw=meg_raw,
        external_audio_path=Path(external_audio_path),
        config=config,
        aux_channel='MISC 007'
    )

    # Save sync params
    output_dir = Path('outputs/sync') / args.subject / f'run-{args.run:02d}'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'sync_params.json'

    with open(output_file, 'w') as f:
        json.dump(sync_params, f, indent=2)

    print(f"\nSync complete!")
    print(f"  Offset: {sync_params['initial_offset_s']:.3f}s")
    print(f"  Correlation: {sync_params['qc_metrics'].get('peak_correlation', 0):.3f}")
    print(f"  Saved to: {output_file}")

    # Generate QC visualizations and report
    print(f"\nGenerating QC outputs...")
    try:
        # Load external audio for QC
        ext_audio, ext_sfreq = librosa.load(str(external_audio_path), sr=None, mono=True)

        # Extract MEG audio
        meg_audio_ch = meg_raw.copy().pick_channels(['MISC 007'])
        meg_audio = meg_audio_ch.get_data()[0]
        meg_sfreq = meg_raw.info['sfreq']

        # Extract envelopes (simplified - will be computed in qc module)
        from scipy import signal as scipy_signal
        meg_env = np.abs(scipy_signal.hilbert(meg_audio))
        ext_env = np.abs(scipy_signal.hilbert(ext_audio))

        # Generate plots
        plot_sync_diagnostics(
            meg_audio=meg_audio,
            ext_audio=ext_audio,
            meg_env=meg_env,
            ext_env=ext_env,
            sync_params=sync_params,
            output_dir=output_dir,
            meg_sfreq=meg_sfreq,
            ext_sfreq=ext_sfreq
        )

        # Generate QC report
        generate_sync_report(
            sync_params=sync_params,
            output_path=output_dir / 'sync_qc_report.txt'
        )

        print(f"  Visualizations: {output_dir}/waveform_alignment.png, envelope_alignment.png")
        print(f"  QC Report: {output_dir}/sync_qc_report.txt")

    except Exception as e:
        print(f"  Warning: QC generation failed: {e}")
        print(f"  Sync params saved successfully, but skipping QC outputs")


if __name__ == '__main__':
    main()
