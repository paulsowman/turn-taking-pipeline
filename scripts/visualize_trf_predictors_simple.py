#!/usr/bin/env python3
"""
Simple Interactive TRF Predictor Visualization

Creates a single-panel interactive visualization with all channels stacked.
Simpler and faster than the subplot version.

Usage:
    python scripts/visualize_trf_predictors_simple.py --subject sub-01 --run 1 --open
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import mne
import plotly.graph_objects as go

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils.config import load_config


def create_stacked_plot(
    raw: mne.io.Raw,
    tmin: float = None,
    tmax: float = None,
    spacing: float = 4.0,
):
    """
    Create stacked interactive visualization of all predictors.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data with MISC predictor channels
    tmin : float, optional
        Start time (seconds)
    tmax : float, optional
        End time (seconds)
    spacing : float
        Vertical spacing between channels (in std units)

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    # Get MISC channels
    misc_picks = mne.pick_types(raw.info, misc=True)
    all_ch_names = [raw.ch_names[i] for i in misc_picks]

    # Filter to only TRF predictor channels
    trf_predictor_keywords = [
        'envelope_interviewer', 'envelope_participant',
        'envelope_meg_mic7', 'envelope_meg_mic8',
        'f0_interviewer', 'f0_participant',
        'word_onsets_interviewer', 'word_onsets_participant',
        'surprisal_interviewer', 'surprisal_participant',
        'duration_interviewer', 'duration_participant',
        'f0_deviation_interviewer', 'f0_deviation_participant',
        'duration_deviation_interviewer', 'duration_deviation_participant',
        'pause_interviewer', 'pause_participant',
        'speaker',
    ]

    # Find channels matching our TRF predictors
    trf_picks = []
    ch_names = []
    for i, ch_name in enumerate(all_ch_names):
        ch_lower = ch_name.lower()
        for keyword in trf_predictor_keywords:
            if keyword in ch_lower:
                trf_picks.append(misc_picks[i])
                ch_names.append(ch_name)
                break

    n_channels = len(ch_names)

    # Get time range
    times = raw.times
    if tmin is None:
        tmin = times[0]
    if tmax is None:
        tmax = times[-1]

    idx_min = np.argmin(np.abs(times - tmin))
    idx_max = np.argmin(np.abs(times - tmax))
    times_cropped = times[idx_min:idx_max]

    # Extract data (only TRF predictor channels)
    data, _ = raw[trf_picks, idx_min:idx_max]

    # Create figure
    fig = go.Figure()

    # Color scheme
    interviewer_color = '#1f77b4'  # Blue
    participant_color = '#ff7f0e'  # Orange
    shared_color = '#2ca02c'      # Green

    # Add traces with vertical offset
    y_positions = []
    for i, (ch_name, ch_data) in enumerate(zip(ch_names, data)):
        # Normalize
        if np.std(ch_data) > 0:
            ch_data_norm = (ch_data - np.mean(ch_data)) / np.std(ch_data)
        else:
            ch_data_norm = ch_data

        # Offset
        offset = i * spacing
        ch_data_offset = ch_data_norm + offset
        y_positions.append(offset)

        # Color by speaker
        short_name = ch_name.replace('MISC_', '')
        if 'interviewer' in ch_name.lower() or 'mic7' in ch_name.lower():
            color = interviewer_color
            group = 'Interviewer'
        elif 'participant' in ch_name.lower() or 'mic8' in ch_name.lower():
            color = participant_color
            group = 'Participant'
        else:
            color = shared_color
            group = 'Shared'

        fig.add_trace(go.Scatter(
            x=times_cropped,
            y=ch_data_offset,
            mode='lines',
            name=f"{group}: {short_name}",
            line=dict(color=color, width=1),
            hovertemplate=(
                f'<b>{short_name}</b><br>' +
                'Time: %{x:.3f}s<br>' +
                'Value: %{customdata:.4f}<br>' +
                '<extra></extra>'
            ),
            customdata=ch_data,
            legendgroup=group,
        ))

    # Update layout
    fig.update_layout(
        title=f'TRF Predictors: {raw.filenames[0].stem if raw.filenames else ""}',
        xaxis=dict(
            title='Time (s)',
            rangeslider=dict(visible=True, thickness=0.05),
        ),
        yaxis=dict(
            title='Channels (offset)',
            ticktext=[ch.replace('MISC_', '') for ch in ch_names],
            tickvals=y_positions,
            showgrid=True,
            gridcolor='lightgray',
            zeroline=False,
        ),
        height=max(600, n_channels * 30 + 200),
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
    )

    return fig


def main():
    parser = argparse.ArgumentParser(description='Simple TRF predictor visualization')
    parser.add_argument('--subject', required=True)
    parser.add_argument('--run', type=int, required=True)
    parser.add_argument('--tmin', type=float, help='Start time (s)')
    parser.add_argument('--tmax', type=float, help='End time (s)')
    parser.add_argument('--spacing', type=float, default=4.0, help='Channel spacing')
    parser.add_argument('--output', help='Output HTML file')
    parser.add_argument('--open', action='store_true', help='Open in browser')

    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    config = load_config()

    # Find file
    trf_dir = base_dir / "outputs" / "trf" / args.subject / f"run-{args.run:02d}"
    trf_file = trf_dir / f"{args.subject}_run-{args.run:02d}_trf_raw.fif"

    if not trf_file.exists():
        print(f"✗ File not found: {trf_file}")
        return 1

    print(f"Loading: {trf_file}")
    raw = mne.io.read_raw_fif(trf_file, preload=True, verbose=False)

    print(f"Creating visualization...")
    fig = create_stacked_plot(raw, tmin=args.tmin, tmax=args.tmax, spacing=args.spacing)

    # Save
    if args.output:
        output_file = Path(args.output)
    else:
        output_dir = trf_dir / "visualizations"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"{args.subject}_run-{args.run:02d}_predictors_simple.html"

    print(f"Saving to: {output_file}")
    fig.write_html(
        str(output_file),
        include_plotlyjs='cdn',
        config={'scrollZoom': True, 'displaylogo': False}
    )

    print(f"✓ Done!")

    if args.open:
        import webbrowser
        webbrowser.open(f'file://{output_file.absolute()}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
