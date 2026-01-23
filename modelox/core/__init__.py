"""
modelox/core/__init__.py

Core package.

Nota importante:
- Este __init__ debe ser liviano y no importar dependencias pesadas (optuna/numba)
    ni módulos que puedan cambiar con refactors (runner/engine).
"""

from .types import BacktestConfig, Reporter, Strategy, TrialArtifacts
from .neighborhood_fitness import NeighborhoodConfig, NeighborhoodResult
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
    "load_data",
    "prepare_multitimeframe_data",
    "resample_ohlcv",
    "get_available_timeframes",
]
