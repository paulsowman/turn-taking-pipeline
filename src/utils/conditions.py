"""
Condition and run management for Natural Conversations study.

Study design:
- Run 1, 3, 5: Conversation conditions (combine for analysis)
- Run 2, 4: Repetition conditions (combine for analysis)
- Run 6: Ba-da localizer
"""

from typing import List, Dict
from pathlib import Path


# Condition definitions
CONVERSATION_RUNS = [1, 3, 5]
REPETITION_RUNS = [2, 4]
LOCALIZER_RUN = 6

CONDITION_MAP = {
    1: 'conversation',
    2: 'repetition',
    3: 'conversation',
    4: 'repetition',
    5: 'conversation',
    6: 'localizer'
}


def get_condition(run: int) -> str:
    """Get condition name for a run number."""
    return CONDITION_MAP.get(run, 'unknown')


def get_runs_for_condition(condition: str) -> List[int]:
    """Get run numbers for a condition."""
    if condition == 'conversation':
        return CONVERSATION_RUNS
    elif condition == 'repetition':
        return REPETITION_RUNS
    elif condition == 'localizer':
        return [LOCALIZER_RUN]
    else:
        raise ValueError(f"Unknown condition: {condition}")


def should_combine_runs(run1: int, run2: int) -> bool:
    """Check if two runs should be combined for analysis."""
    return get_condition(run1) == get_condition(run2)


def get_combinable_runs(subject: str, base_dir: Path, condition: str = None) -> Dict[str, List[int]]:
    """
    Find which runs are available for a subject and group by condition.

    Parameters
    ----------
    subject : str
        Subject ID (e.g., 'sub-01')
    base_dir : Path
        Pipeline base directory
    condition : str, optional
        Filter for specific condition ('conversation', 'repetition', 'localizer')

    Returns
    -------
    available_runs : dict
        Dict mapping condition -> list of available run numbers
    """
    feature_dir = base_dir / "outputs" / "features" / subject

    if not feature_dir.exists():
        return {}

    # Find available runs
    available = {}

    for run_dir in sorted(feature_dir.glob("run-*")):
        run_num = int(run_dir.name.split('-')[1])
        cond = get_condition(run_num)

        # Check if transcript exists (proxy for completed processing)
        if (run_dir / "transcript.csv").exists():
            if cond not in available:
                available[cond] = []
            available[cond].append(run_num)

    # Filter by condition if specified
    if condition and condition in available:
        return {condition: available[condition]}
    elif condition:
        return {}

    return available


def format_run_summary(available_runs: Dict[str, List[int]]) -> str:
    """Format available runs as readable summary."""
    lines = []
    for cond, runs in sorted(available_runs.items()):
        lines.append(f"  {cond.capitalize()}: runs {', '.join(map(str, runs))}")
    return '\n'.join(lines)


def validate_run_combination(runs: List[int]) -> bool:
    """Check if a list of runs can be validly combined."""
    if not runs:
        return False

    conditions = [get_condition(r) for r in runs]
    return len(set(conditions)) == 1  # All same condition
