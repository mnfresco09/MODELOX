"""
================================================================================
MODELOX/CORE/__INIT__.PY — CORE PACKAGE PÚBLICO
================================================================================

PROPÓSITO:
    Exporta la API pública del paquete `modelox.core`.
    Centraliza imports de types, data, metrics y exits.

ARQUITECTURA:
    - types.py:   Tipos y configuraciones base
    - data.py:    Carga y preparación de datos
    - metrics.py: FUENTE ÚNICA para cálculo de métricas
    - exits.py:   FUENTE ÚNICA para parámetros de salida
    - engine.py:  Motor de backtest Numba
    - runner.py:  Orquestador de optimización

SCORING (módulo independiente):
    from modelox.optimizers import CMAScorer, TPEScorer, score_cma, score_tpe

================================================================================
"""

from .types import BacktestConfig, Reporter, Strategy, TrialArtifacts
from .data import (
    load_data,
    prepare_multitimeframe_data,
    resample_ohlcv,
    resample_to_base_timeframe,
    candles_per_month_for_tf,
    get_available_timeframes,
)
# Métricas centralizadas
from .metrics import (
    resumen_metricas,
    sharpe,
    sortino,
    sqn,
    max_drawdown,
    profit_factor,
    payoff_ratio,
    expectativa,
    winrate_pct,
    roi_pct,
    calmar,
    trades_por_dia,
)
# Salidas centralizadas
from .exits import (
    ExitSettings,
    ExitResult,
    resolve_exit_settings_for_trial,
    exit_settings_from_params,
    DEFAULT_EXIT_TYPE,
    DEFAULT_EXIT_SL_PCT,
    DEFAULT_EXIT_TP_PCT,
)

__all__ = [
    # =========================================================================
    # TYPES
    # =========================================================================
    "BacktestConfig",
    "Reporter",
    "Strategy",
    "TrialArtifacts",
    
    # =========================================================================
    # DATA
    # =========================================================================
    "load_data",
    "prepare_multitimeframe_data",
    "resample_ohlcv",
    "get_available_timeframes",
    
    # =========================================================================
    # METRICS (CENTRALIZADAS)
    # =========================================================================
    "resumen_metricas",
    "sharpe",
    "sortino",
    "sqn",
    "max_drawdown",
    "profit_factor",
    "payoff_ratio",
    "expectativa",
    "winrate_pct",
    "roi_pct",
    "calmar",
    "trades_por_dia",
    
    # =========================================================================
    # EXITS (CENTRALIZADOS)
    # =========================================================================
    "ExitSettings",
    "ExitResult",
    "resolve_exit_settings_for_trial",
    "exit_settings_from_params",
    "DEFAULT_EXIT_TYPE",
    "DEFAULT_EXIT_SL_PCT",
    "DEFAULT_EXIT_TP_PCT",
]
