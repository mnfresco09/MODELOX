"""
modelox/core/__init__.py

Core package - Scoring Unificado v7.0

Nota importante:
- Este __init__ debe ser liviano y no importar dependencias pesadas (optuna/numba)
    ni módulos que puedan cambiar con refactors (runner/engine).
- v7.0: Todo el sistema de scoring y vecindario está unificado en scoring.py
"""

from .types import BacktestConfig, Reporter, Strategy, TrialArtifacts
from .scoring import (
    NeighborhoodConfig, 
    NeighborhoodResult, 
    DEFAULT_NEIGHBORHOOD_CONFIG,
    score_unified,
    score_optuna,
    run_neighborhood_analysis,
)
from .data import (
    load_data,
    prepare_multitimeframe_data,
    resample_ohlcv,
    get_available_timeframes,
)

__all__ = [
    "BacktestConfig",
    "Reporter",
    "Strategy",
    "TrialArtifacts",
    "NeighborhoodConfig",
    "NeighborhoodResult",
    "DEFAULT_NEIGHBORHOOD_CONFIG",
    "score_unified",
    "score_optuna",
    "run_neighborhood_analysis",
    "load_data",
    "prepare_multitimeframe_data",
    "resample_ohlcv",
    "get_available_timeframes",
]
