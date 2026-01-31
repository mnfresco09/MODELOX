"""
# =============================================================================
#
#      ██████╗ ██████╗ ██████╗ ███████╗
#     ██╔════╝██╔═══██╗██╔══██╗██╔════╝
#     ██║     ██║   ██║██████╔╝█████╗
#     ██║     ██║   ██║██╔══██╗██╔══╝
#     ╚██████╗╚██████╔╝██║  ██║███████╗
#      ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝
#
#     CORE PACKAGE - MOTOR DE BACKTESTING
#
# =============================================================================
"""

from .types import BacktestConfig, Reporter, Strategy, TrialArtifacts

from .scoring import (
    ScoringConfig,
    score_unified,
    score_optuna,
    score_quality_only,
    set_study_for_scorer,
    cleanup_scoring_resources,
)

from .data import (
    load_data,
    prepare_multitimeframe_data,
    resample_ohlcv,
    get_available_timeframes,
)


__all__ = [
    # TIPOS
    "BacktestConfig",
    "Reporter",
    "Strategy",
    "TrialArtifacts",
    
    # SCORING
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
