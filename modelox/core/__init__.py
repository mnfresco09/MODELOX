"""
modelox/core/__init__.py

═══════════════════════════════════════════════════════════════════════════════
CORE PACKAGE - MODELOX v3.0 + TOPÓGRAFO DE MESETAS
═══════════════════════════════════════════════════════════════════════════════

Exports principales del sistema de backtesting y optimización.

NOTA: Este __init__ debe ser liviano - NO importar dependencias pesadas 
      (optuna/numba) ni módulos que puedan cambiar con refactors.
"""

# Tipos base
from .types import BacktestConfig, Reporter, Strategy, TrialArtifacts

# Sistema de scoring institucional
from .scoring import (
    ScoringConfig,
    score_unified,
    score_optuna,
    score_quality_only,
    set_study_for_scorer,
    cleanup_scoring_resources,
)

# Carga de datos
from .data import (
    load_data,
    prepare_multitimeframe_data,
    resample_ohlcv,
    get_available_timeframes,
)

# Sistema de mesetas (carga diferida para evitar dependencias circulares)
# from .topology import PlateauConfig, PlateauResult, TopologyAnalysis, analyze_topology
# from .plateau_optimizer import PlateauOptimizer, PlateauOptimizerConfig, run_plateau_optimization

__all__ = [
    # Tipos
    "BacktestConfig",
    "Reporter",
    "Strategy",
    "TrialArtifacts",
    
    # Scoring
    "ScoringConfig",
    "score_unified",
    "score_optuna",
    "score_quality_only",
    "set_study_for_scorer",
    "cleanup_scoring_resources",
    
    # Data
    "load_data",
    "prepare_multitimeframe_data",
    "resample_ohlcv",
    "get_available_timeframes",
    
    # Mesetas (importar explícitamente cuando se necesiten)
    # "PlateauConfig", "PlateauResult", "TopologyAnalysis", "analyze_topology",
    # "PlateauOptimizer", "PlateauOptimizerConfig", "run_plateau_optimization",
]
