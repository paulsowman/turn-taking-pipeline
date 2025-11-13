#!/usr/bin/env python3
"""
Interactive TRF Predictor Visualization

Creates an interactive HTML visualization of all TRF predictor channels
with zoom, pan, and channel selection capabilities.

Usage:
    # Visualize specific subject/run
    python scripts/visualize_trf_predictors.py --subject sub-01 --run 1

    # Auto-open in browser
    python scripts/visualize_trf_predictors.py --subject sub-01 --run 1 --open

    # Save to custom location
    python scripts/visualize_trf_predictors.py --subject sub-01 --run 1 --output my_viz.html

    # Show specific time range (in seconds)
    python scripts/visualize_trf_predictors.py --subject sub-01 --run 1 --tmin 10 --tmax 60
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import mne
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils.config import load_config

# Channel colors by type
CHANNEL_COLORS = {
    'envelope': '#1f77b4',      # Blue
    'f0': '#ff7f0e',            # Orange
    'word_onsets': '#2ca02c',   # Green
    'surprisal': '#d62728',     # Red
    'duration': '#9467bd',      # Purple
    'f0_deviation': '#8c564b',  # Brown
    'duration_deviation': '#e377c2',  # Pink
    'pause': '#7f7f7f',         # Gray
    'speaker': '#17becf',       # Cyan
}


def get_channel_color(ch_name):
    """Get color for channel based on predictor type."""
    ch_lower = ch_name.lower()
    for key, color in CHANNEL_COLORS.items():
        if key in ch_lower:
            return color
    return '#000000'  # Black default


def get_channel_group(ch_name):
    """Get group label (Interviewer/Participant/Shared) for channel."""
    ch_lower = ch_name.lower()
    if 'interviewer' in ch_lower or 'mic7' in ch_lower or 'mic_7' in ch_lower:
        return 'Interviewer'
    elif 'participant' in ch_lower or 'mic8' in ch_lower or 'mic_8' in ch_lower:
        return 'Participant'
    else:
        return 'Shared'


def create_interactive_plot(
    raw: mne.io.Raw,
    tmin: float = None,
    tmax: float = None,
    height_per_channel: int = 80,
    show_legend: bool = True,
):
    """
    Create interactive Plotly visualization of TRF predictors.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data with MISC predictor channels
    tmin : float, optional
        Start time to display (seconds)
    tmax : float, optional
        End time to display (seconds)
    height_per_channel : int
        Height in pixels per channel
    show_legend : bool
        Whether to show legend

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive figure
    """
    # Get MISC channels (predictors)
    misc_picks = mne.pick_types(raw.info, misc=True)
    all_ch_names = [raw.ch_names[i] for i in misc_picks]

    # Filter to only TRF predictor channels (the ones we added)
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
    print(f"Found {len(all_ch_names)} total MISC channels")
    print(f"Displaying {n_channels} TRF predictor channels")

    # Get time range
    times = raw.times
    if tmin is None:
        tmin = times[0]
    if tmax is None:
        tmax = times[-1]

    # Get time indices
    idx_min = np.argmin(np.abs(times - tmin))
    idx_max = np.argmin(np.abs(times - tmax))
    times_cropped = times[idx_min:idx_max]

    print(f"Displaying time range: {tmin:.1f}s to {tmax:.1f}s ({len(times_cropped)} samples)")

    # Extract data (only TRF predictor channels)
    data, _ = raw[trf_picks, idx_min:idx_max]

    # Create figure with subplots
    fig = make_subplots(
        rows=n_channels,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.01,
        subplot_titles=[f"{get_channel_group(ch)}: {ch.replace('MISC_', '')}" for ch in ch_names],
    )

    # Add traces for each channel
    for i, (ch_name, ch_data) in enumerate(zip(ch_names, data)):
        color = get_channel_color(ch_name)
        group = get_channel_group(ch_name)
        short_name = ch_name.replace('MISC_', '')

        # Normalize data for display (z-score)
        if np.std(ch_data) > 0:
            ch_data_display = (ch_data - np.mean(ch_data)) / np.std(ch_data)
        else:
            ch_data_display = ch_data

        fig.add_trace(
            go.Scatter(
                x=times_cropped,
                y=ch_data_display,
                mode='lines',
                name=f"{group}: {short_name}",
                line=dict(color=color, width=1),
                hovertemplate=(
                    f'<b>{short_name}</b><br>' +
                    'Time: %{x:.3f}s<br>' +
                    'Value: %{customdata:.4f}<br>' +
                    '<extra></extra>'
                ),
                customdata=ch_data,  # Show original values in hover
                legendgroup=group,
            ),
            row=i+1,
            col=1
        )

        # Update y-axis for this subplot
        fig.update_yaxes(
            title_text=short_name,
            row=i+1,
            col=1,
            showticklabels=False,
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='lightgray',
        )

    # Update x-axis (only bottom one)
    fig.update_xaxes(
        title_text='Time (s)',
        row=n_channels,
        col=1,
        rangeslider=dict(visible=True, thickness=0.05),
    )

    # Update layout
    total_height = max(600, n_channels * height_per_channel)
    fig.update_layout(
        height=total_height,
        showlegend=show_legend,
        title=dict(
            text='TRF Predictor Channels (normalized for display)',
            x=0.5,
            xanchor='center',
        ),
        hovermode='x unified',
        template='plotly_white',
    )

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Create interactive visualization of TRF predictors'
    )
    parser.add_argument(
        '--subject',
        required=True,
        help='Subject ID (e.g., sub-01)'
    )
    parser.add_argument(
        '--run',
        type=int,
        required=True,
        help='Run number (e.g., 1)'
    )
    parser.add_argument(
        '--tmin',
        type=float,
        help='Start time in seconds (default: 0)'
    )
    parser.add_argument(
        '--tmax',
        type=float,
        help='End time in seconds (default: end of recording)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output HTML file path (default: auto-generate)'
    )
    parser.add_argument(
        '--open',
        action='store_true',
        help='Open visualization in browser after creating'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=80,
        help='Height per channel in pixels (default: 80)'
    )

    args = parser.parse_args()

    # Base directory
    base_dir = Path(__file__).parent.parent

    # Load config
    config = load_config()

    # Find TRF FIF file
    trf_dir = base_dir / "outputs" / "trf" / args.subject / f"run-{args.run:02d}"
    trf_file = trf_dir / f"{args.subject}_run-{args.run:02d}_trf_raw.fif"

    if not trf_file.exists():
        print(f"✗ TRF FIF file not found: {trf_file}")
        print(f"  Run: python scripts/create_trf_fif.py --subject {args.subject} --run {args.run}")
        return 1

    print(f"\n{'='*70}")
    print(f"VISUALIZING TRF PREDICTORS: {args.subject} run-{args.run:02d}")
    print(f"{'='*70}\n")

    # Load TRF FIF file
    print(f"Loading TRF FIF file...")
    raw = mne.io.read_raw_fif(trf_file, preload=True, verbose=False)
    print(f"  Total channels: {len(raw.ch_names)}")
    print(f"  Duration: {raw.times[-1]:.1f}s")

    # Create visualization
    print(f"\nCreating interactive visualization...")
    fig = create_interactive_plot(
        raw,
        tmin=args.tmin,
        tmax=args.tmax,
        height_per_channel=args.height,
    )

    # Determine output path
    if args.output:
        output_file = Path(args.output)
    else:
        output_dir = trf_dir / "visualizations"
        output_dir.mkdir(exist_ok=True)
        if args.tmin is not None or args.tmax is not None:
            tmin_str = f"{args.tmin:.0f}" if args.tmin is not None else "0"
            tmax_str = f"{args.tmax:.0f}" if args.tmax is not None else "end"
            output_file = output_dir / f"{args.subject}_run-{args.run:02d}_predictors_{tmin_str}-{tmax_str}s.html"
        else:
            output_file = output_dir / f"{args.subject}_run-{args.run:02d}_predictors.html"

    # Save
    print(f"\nSaving to: {output_file}")
    fig.write_html(
        str(output_file),
        include_plotlyjs='cdn',
        config={
            'scrollZoom': True,
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        }
    )

    print(f"✓ Visualization saved!")
    print(f"\nVisualization features:")
    print(f"  • Zoom: Click and drag on plot")
    print(f"  • Pan: Shift + click and drag")
    print(f"  • Scroll zoom: Scroll wheel on time axis")
    print(f"  • Reset: Double-click on plot")
    print(f"  • Toggle channels: Click legend items")
    print(f"  • Hover: See exact values at time points")
    print(f"  • Time slider: Use range slider at bottom")

    # Open in browser if requested
    if args.open:
        import webbrowser
        print(f"\nOpening in browser...")
        webbrowser.open(f'file://{output_file.absolute()}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
