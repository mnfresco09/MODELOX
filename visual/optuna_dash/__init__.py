"""
MODELOX · AI Parameter Intelligence Dashboard
Package init — exports start_dashboard entry point.
"""
from __future__ import annotations


def start_dashboard(
    db_path: str,
    port: int = 8050,
    debug: bool = False,
    perf_debug: bool = False,
) -> None:
    """Launch the MODELOX analytics dashboard."""
    from visual.optuna_dash.app import run
    run(db_path, port=port, debug=debug, perf_debug=perf_debug)


__all__ = ["start_dashboard"]
