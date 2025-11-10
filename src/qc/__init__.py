"""Quality control module for pipeline outputs."""

from .sync_qc import plot_sync_diagnostics, generate_sync_report

__all__ = [
    "plot_sync_diagnostics",
    "generate_sync_report",
]
