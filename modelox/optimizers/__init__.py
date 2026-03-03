"""modelox.optimizers — Optimizadores Optuna (TPE, QMC, Hybrid QMC→TPE)."""

from .tpe import (
    TPEOptimizer, TPEOptimizerConfig, TPEScorer, TPEScoringConfig,
    TPE_SCORING_CONFIG, TPE_OPTIMIZER_CONFIG, create_tpe_study, score_tpe,
)
from .qmc import (
    QMCOptimizer, QMCOptimizerConfig, QMCScorer, QMCScoringConfig,
    QMC_SCORING_CONFIG, QMC_OPTIMIZER_CONFIG, create_qmc_study, score_qmc,
)
from .hybrid import (
    HybridOptimizer, HybridOptimizerConfig, HybridScorer, HybridScoringConfig,
    HYBRID_SCORING_CONFIG, HYBRID_OPTIMIZER_CONFIG, create_hybrid_study, score_hybrid,
)

from typing import Optional
import optuna
from .storage import resolve_storage_for_strategy


def create_study(
    sampler: str,
    strategy_name: str,
    strategy_id: Optional[int] = None,
    activo: Optional[str] = None,
    seed: Optional[int] = None,
    study_name_prefix: str = "MODELOX",
    create_database: bool = True,
) -> optuna.Study:
    """Factory: crea un estudio Optuna con el sampler indicado (TPE | QMC | HYBRID)."""
    sampler_type = sampler.upper() if sampler else "HYBRID"
    storage = resolve_storage_for_strategy(
        create_database=create_database,
        strategy_id=strategy_id,
    )
    kwargs = dict(
        strategy_name=strategy_name,
        strategy_id=strategy_id,
        activo=activo,
        seed=seed,
        study_name_prefix=study_name_prefix,
        create_database=create_database,
        storage=storage,
    )
    if sampler_type == "TPE":
        return create_tpe_study(**kwargs)
    elif sampler_type == "QMC":
        return create_qmc_study(**kwargs)
    else:
        return create_hybrid_study(**kwargs)


__all__ = [
    "create_study",
    "TPEOptimizer", "TPEOptimizerConfig", "TPEScorer", "TPEScoringConfig",
    "TPE_SCORING_CONFIG", "TPE_OPTIMIZER_CONFIG", "create_tpe_study", "score_tpe",
    "QMCOptimizer", "QMCOptimizerConfig", "QMCScorer", "QMCScoringConfig",
    "QMC_SCORING_CONFIG", "QMC_OPTIMIZER_CONFIG", "create_qmc_study", "score_qmc",
    "HybridOptimizer", "HybridOptimizerConfig", "HybridScorer", "HybridScoringConfig",
    "HYBRID_SCORING_CONFIG", "HYBRID_OPTIMIZER_CONFIG", "create_hybrid_study", "score_hybrid",
]
