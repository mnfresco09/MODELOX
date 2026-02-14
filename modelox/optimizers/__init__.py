"""modelox/optimizers/__init__.py

═══════════════════════════════════════════════════════════════════════════════
    ██████╗ ██████╗ ████████╗██╗███╗   ███╗██╗███████╗███████╗██████╗ ███████╗
   ██╔═══██╗██╔══██╗╚══██╔══╝██║████╗ ████║██║╚══███╔╝██╔════╝██╔══██╗██╔════╝
   ██║   ██║██████╔╝   ██║   ██║██╔████╔██║██║  ███╔╝ █████╗  ██████╔╝███████╗
   ██║   ██║██╔═══╝    ██║   ██║██║╚██╔╝██║██║ ███╔╝  ██╔══╝  ██╔══██╗╚════██║
   ╚██████╔╝██║        ██║   ██║██║ ╚═╝ ██║██║███████╗███████╗██║  ██║███████║
    ╚═════╝ ╚═╝        ╚═╝   ╚═╝╚═╝     ╚═╝╚═╝╚══════╝╚══════╝╚═╝  ╚═╝╚══════╝
    
    MÓDULO DE OPTIMIZADORES - MODELOX
═══════════════════════════════════════════════════════════════════════════════

DESCRIPCIÓN:
============
Este módulo contiene los algoritmos de optimización disponibles en MODELOX.
Cada optimizador tiene su propio sistema de scoring independiente.

ARQUITECTURA:
=============
    modelox/
    ├── core/           ← NÚCLEO DEL SISTEMA (engine, metrics, types, exits)
    │   ├── engine.py   ← MOTOR DE BACKTEST (NUMBA)
    │   ├── metrics.py  ← MÉTRICAS CENTRALIZADAS
    │   ├── types.py    ← TIPOS Y DATACLASSES
    │   ├── exits.py    ← CONFIGURACIÓN DE SALIDAS
    │   ├── data.py     ← CARGA DE DATOS
    │   └── runner.py   ← GENERADOR DE SEÑALES Y UTILIDADES
    │
    └── optimizers/     ← MÓDULO DE OPTIMIZADORES (AQUÍ)
        ├── __init__.py ← ESTE ARCHIVO
        ├── cma.py      ← CMA-ES + SCORING INSTITUCIONAL
        ├── tpe.py      ← TPE + SCORING EXPLORATORIA
        ├── gt.py       ← GT-SCORE + INTERCEPTOR IA TOPOLÓGICO
        ├── ml.py       ← ML-FOREST + MACHINE LEARNING
        └── qmc.py      ← QMC + SECUENCIAS SOBOL (COBERTURA EQUIDISTANTE)

OPTIMIZADORES DISPONIBLES:
==========================

┌─────────────────────────────────────────────────────────────────────────────┐
│  1. CMA-ES (cma.py) - RECOMENDADO PARA TRADING CUANTITATIVO                │
├─────────────────────────────────────────────────────────────────────────────┤
│  VENTAJAS:                                                                  │
│    ✓ APRENDE de los scores para adaptar la búsqueda                        │
│    ✓ CONVERGE hacia regiones de alta calidad                               │
│    ✓ Ideal para encontrar "mesetas de parámetros" (robustez)               │
│    ✓ PENALIZA implícitamente los picos aislados (overfitting)              │
│                                                                             │
│  SCORING INSTITUCIONAL:                                                     │
│    • Score Base: Sigmoide de Sharpe → [100, 900]                           │
│    • PSR/DSR: Inferencia probabilística con deflación                      │
│    • SAM: Estabilidad de vecindario                                         │
│    • RÉGIMEN: Consistencia en diferentes volatilidades                      │
│    • K-RATIO: Calidad de curva de equity                                   │
│    • DECAY: Factor de exploración                                          │
│                                                                             │
│  USO:                                                                       │
│    from modelox.optimizers import CMAOptimizer                             │
│    optimizer = CMAOptimizer(config, n_trials=1000)                         │
│    study = optimizer.optimize(df=df, strategy=strategy)                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  2. TPE (tpe.py) - PARA EXPLORACIÓN AMPLIA                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  VENTAJAS:                                                                  │
│    ✓ EXPLORACIÓN amplia del espacio de parámetros                          │
│    ✓ DIVERSIDAD de soluciones encontradas                                  │
│    ✓ Ideal para ESPACIOS GRANDES y DISCONTINUOS                            │
│    ✓ Maneja PARÁMETROS CATEGÓRICOS nativamente                             │
│    ✓ RÁPIDO en convergencia inicial                                        │
│                                                                             │
│  SCORING SIMPLE:                                                            │
│    • Promedio ponderado de métricas normalizadas                           │
│    • Sharpe (35%), SQN (20%), ROI (20%), DD (15%), Trades (10%)           │
│    • PSR simple opcional (sin deflación)                                   │
│    • Sin SAM ni régimen (más velocidad)                                    │
│                                                                             │
│  USO:                                                                       │
│    from modelox.optimizers import TPEOptimizer                             │
│    optimizer = TPEOptimizer(config, n_trials=1000)                         │
│    study = optimizer.optimize(df=df, strategy=strategy)                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  3. GT (gt.py) - ANTI-OVERFITTING CON INTELIGENCIA TOPOLÓGICA              │
├─────────────────────────────────────────────────────────────────────────────┤
│  VENTAJAS:                                                                  │
│    ✓ GT-SCORE como métrica base anti-sobreajuste                           │
│    ✓ INTERCEPTOR TOPOLÓGICO k-NN con distancia Gower ponderada            │
│    ✓ fANOVA para pesos de importancia por hiperparámetro                   │
│    ✓ PSR como filtro de significancia estadística                          │
│    ✓ MOTOR MODULAR (CMA-ES por defecto, intercambiable)                   │
│                                                                             │
│  SCORING GT-SCORE:                                                          │
│    • GT_base = μ × ln(z) × r² × (1/σ_d) × Sharpe × SQN                   │
│    • Interceptor IA: k-NN detecta picos vs mesetas                         │
│    • Score_final = GT_base × (1 - Penalización_topológica) × PSR           │
│                                                                             │
│  USO:                                                                       │
│    from modelox.optimizers import GTOptimizer                              │
│    optimizer = GTOptimizer(config, n_trials=1000)                          │
│    study = optimizer.optimize(df=df, strategy=strategy)                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  4. QMC (qmc.py) - COBERTURA EQUIDISTANTE CON SECUENCIAS SOBOL             │
├─────────────────────────────────────────────────────────────────────────────┤
│  VENTAJAS:                                                                  │
│    ✓ COBERTURA MÁXIMA del espacio de parámetros (Sobol)                    │
│    ✓ EQUIDISTANTE: Puntos perfectamente distribuidos                       │
│    ✓ DETERMINISTA y REPRODUCIBLE (misma seed → mismos puntos)              │
│    ✓ NO NECESITA SCORING — pura matemática                                │
│    ✓ ANTI-OVERFITTING extremo (cero sesgo hacia resultados)               │
│    ✓ IDEAL para mapear el terreno antes de refinar con CMA/GT             │
│                                                                             │
│  SCORING PASIVO:                                                            │
│    • Score solo INFORMATIVO (no guía la búsqueda)                          │
│    • Promedio ponderado simple para registro post-hoc                      │
│    • Sirve para identificar mejores combinaciones DESPUÉS                  │
│                                                                             │
│  USO:                                                                       │
│    from modelox.optimizers import QMCOptimizer                             │
│    optimizer = QMCOptimizer(config, n_trials=6000)                         │
│    study = optimizer.optimize(df=df, strategy=strategy)                    │
└─────────────────────────────────────────────────────────────────────────────┘

COMPARACIÓN:
============
    ┌────────────────┬──────────────────────┬──────────────────────┬──────────────────────┬──────────────────────┐
    │  CARACTERÍSTICA │       CMA-ES         │        TPE           │        GT            │        QMC           │
    ├────────────────┼──────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
    │  ROBUSTEZ      │  ★★★★★ (PSR/DSR/SAM) │  ★★★ (PSR simple)    │  ★★★★★ (GT+k-NN)    │  ★★★★★ (sin sesgo)  │
    │  EXPLORACIÓN   │  ★★★                 │  ★★★★★               │  ★★★★ (CMA+topo)    │  ★★★★★+ (Sobol)     │
    │  VELOCIDAD     │  ★★★                 │  ★★★★                │  ★★★ (fANOVA+kNN)   │  ★★★★★ (sin score)  │
    │  CATEGÓRICOS   │  ★★                  │  ★★★★★               │  ★★★★ (Gower)       │  ★★★ (fallback)     │
    │  OVERFITTING   │  ★★★★★ (resiste)     │  ★★★ (vulnerable)    │  ★★★★★+ (topología) │  ★★★★★+ (cero sesgo)│
    │  COBERTURA     │  ★★★ (converge)      │  ★★★★ (explora)      │  ★★★★ (topología)   │  ★★★★★+ (Sobol)     │
    └────────────────┴──────────────────────┴──────────────────────┴──────────────────────┴──────────────────────┘

RECOMENDACIÓN:
==============
    • USAR QMC para exploración INICIAL exhaustiva (mapear el terreno)
    • USAR GT para optimización ANTI-OVERFITTING (producción robusta)
    • USAR CMA-ES para optimización FINAL (producción)
    • USAR TPE para exploración rápida con CATEGÓRICOS
    • COMBINAR: QMC → 500 trials → CMA/GT → 500 trials (máxima cobertura + refinamiento)

═══════════════════════════════════════════════════════════════════════════════
"""

# =============================================================================
# IMPORTS CMA-ES
# =============================================================================
from .cma import (
    CMAOptimizer,
    CMAOptimizerConfig,
    CMAScorer,
    CMAScoringConfig,
    CMA_SCORING_CONFIG,
    CMA_OPTIMIZER_CONFIG,
    create_cma_study,
    score_cma,
)

# =============================================================================
# IMPORTS TPE
# =============================================================================
from .tpe import (
    TPEOptimizer,
    TPEOptimizerConfig,
    TPEScorer,
    TPEScoringConfig,
    TPE_SCORING_CONFIG,
    TPE_OPTIMIZER_CONFIG,
    create_tpe_study,
    score_tpe,
)

# =============================================================================
# IMPORTS GT (GT-SCORE + INTERCEPTOR IA TOPOLÓGICO)
# =============================================================================
from .gt import (
    GTOptimizer,
    GTOptimizerConfig,
    GTScorer,
    GTScoringConfig,
    GT_SCORING_CONFIG,
    GT_OPTIMIZER_CONFIG,
    create_gt_study,
    score_gt,
)

# =============================================================================
# IMPORTS ML-FOREST (MACHINE LEARNING CON RANDOM FOREST)
# =============================================================================
from .ml import (
    MLForestOptimizer,
    MLForestOptimizerConfig,
    MLForestScorer,
    MLForestScoringConfig,
    ML_SCORING_CONFIG,
    ML_OPTIMIZER_CONFIG,
    create_ml_study,
    score_ml,
)

# =============================================================================
# IMPORTS QMC (QUASI-MONTE CARLO — SECUENCIAS SOBOL)
# =============================================================================
from .qmc import (
    QMCOptimizer,
    QMCOptimizerConfig,
    QMCScorer,
    QMCScoringConfig,
    QMC_SCORING_CONFIG,
    QMC_OPTIMIZER_CONFIG,
    create_qmc_study,
    score_qmc,
)

# =============================================================================
# IMPORTS TYPING
# =============================================================================
from typing import Optional
import optuna


# =============================================================================
# FACTORY: CREAR ESTUDIO SEGÚN SAMPLER
# =============================================================================

def create_study(
    sampler: str,
    strategy_name: str,
    activo: Optional[str] = None,
    seed: Optional[int] = None,
    study_name_prefix: str = "MODELOX",
    storage: Optional[str] = None,
) -> optuna.Study:
    """
    Factory que crea un estudio Optuna con el sampler indicado.
    
    Args:
        sampler: "CMA" o "TPE"
        strategy_name: Nombre de la estrategia
        activo: Nombre del activo (opcional)
        seed: Semilla aleatoria
        study_name_prefix: Prefijo para el nombre del estudio
        storage: URI de almacenamiento (None = RAM)
    
    Returns:
        optuna.Study configurado con el sampler elegido
    """
    sampler_type = sampler.upper() if sampler else "CMA"
    
    if sampler_type == "TPE":
        return create_tpe_study(
            strategy_name=strategy_name,
            activo=activo,
            seed=seed,
            study_name_prefix=study_name_prefix,
            storage=storage,
        )
    elif sampler_type == "GT":
        return create_gt_study(
            strategy_name=strategy_name,
            activo=activo,
            seed=seed,
            study_name_prefix=study_name_prefix,
            storage=storage,
        )
    elif sampler_type in ("ML", "MLFOREST", "ML_FOREST"):
        return create_ml_study(
            strategy_name=strategy_name,
            activo=activo,
            seed=seed,
            study_name_prefix=study_name_prefix,
            storage=storage,
        )
    elif sampler_type == "QMC":
        return create_qmc_study(
            strategy_name=strategy_name,
            activo=activo,
            seed=seed,
            study_name_prefix=study_name_prefix,
            storage=storage,
        )
    else:
        # Default: CMA-ES
        return create_cma_study(
            strategy_name=strategy_name,
            activo=activo,
            seed=seed,
            study_name_prefix=study_name_prefix,
            storage=storage,
        )


# =============================================================================
# EXPORTACIONES
# =============================================================================
__all__ = [
    # =========================================================================
    # FACTORY
    # =========================================================================
    "create_study",           # FACTORY PARA CREAR ESTUDIO SEGÚN SAMPLER
    
    # =========================================================================
    # CMA-ES (COVARIANCE MATRIX ADAPTATION EVOLUTION STRATEGY)
    # =========================================================================
    "CMAOptimizer",           # CLASE OPTIMIZADOR CMA-ES
    "CMAOptimizerConfig",     # CONFIGURACIÓN DEL OPTIMIZADOR
    "CMAScorer",              # CLASE SCORER INSTITUCIONAL
    "CMAScoringConfig",       # CONFIGURACIÓN DEL SCORING
    "CMA_SCORING_CONFIG",     # INSTANCIA DEFAULT DE SCORING
    "CMA_OPTIMIZER_CONFIG",   # INSTANCIA DEFAULT DE OPTIMIZADOR
    "create_cma_study",       # CREAR ESTUDIO CMA-ES
    "score_cma",              # FUNCIÓN STANDALONE DE SCORING
    
    # =========================================================================
    # TPE (TREE-STRUCTURED PARZEN ESTIMATOR)
    # =========================================================================
    "TPEOptimizer",           # CLASE OPTIMIZADOR TPE
    "TPEOptimizerConfig",     # CONFIGURACIÓN DEL OPTIMIZADOR
    "TPEScorer",              # CLASE SCORER SIMPLE
    "TPEScoringConfig",       # CONFIGURACIÓN DEL SCORING
    "TPE_SCORING_CONFIG",     # INSTANCIA DEFAULT DE SCORING
    "TPE_OPTIMIZER_CONFIG",   # INSTANCIA DEFAULT DE OPTIMIZADOR
    "create_tpe_study",       # CREAR ESTUDIO TPE
    "score_tpe",              # FUNCIÓN STANDALONE DE SCORING
    
    # =========================================================================
    # GT (GT-SCORE + INTERCEPTOR IA TOPOLÓGICO)
    # =========================================================================
    "GTOptimizer",            # CLASE OPTIMIZADOR GT
    "GTOptimizerConfig",      # CONFIGURACIÓN DEL OPTIMIZADOR
    "GTScorer",               # CLASE SCORER GT-SCORE
    "GTScoringConfig",        # CONFIGURACIÓN DEL SCORING
    "GT_SCORING_CONFIG",      # INSTANCIA DEFAULT DE SCORING
    "GT_OPTIMIZER_CONFIG",    # INSTANCIA DEFAULT DE OPTIMIZADOR
    "create_gt_study",        # CREAR ESTUDIO GT
    "score_gt",               # FUNCIÓN STANDALONE DE SCORING
    
    # =========================================================================
    # ML-FOREST (MACHINE LEARNING CON RANDOM FOREST)
    # =========================================================================
    "MLForestOptimizer",      # CLASE OPTIMIZADOR ML-FOREST
    "MLForestOptimizerConfig",# CONFIGURACIÓN DEL OPTIMIZADOR
    "MLForestScorer",         # CLASE SCORER ML-FOREST
    "MLForestScoringConfig",  # CONFIGURACIÓN DEL SCORING
    "ML_SCORING_CONFIG",      # INSTANCIA DEFAULT DE SCORING
    "ML_OPTIMIZER_CONFIG",    # INSTANCIA DEFAULT DE OPTIMIZADOR
    "create_ml_study",        # CREAR ESTUDIO ML-FOREST
    "score_ml",               # FUNCIÓN STANDALONE DE SCORING
    
    # =========================================================================
    # QMC (QUASI-MONTE CARLO — SECUENCIAS SOBOL)
    # =========================================================================
    "QMCOptimizer",           # CLASE OPTIMIZADOR QMC
    "QMCOptimizerConfig",     # CONFIGURACIÓN DEL OPTIMIZADOR
    "QMCScorer",              # CLASE SCORER PASIVO (INFORMATIVO)
    "QMCScoringConfig",       # CONFIGURACIÓN DEL SCORING
    "QMC_SCORING_CONFIG",     # INSTANCIA DEFAULT DE SCORING
    "QMC_OPTIMIZER_CONFIG",   # INSTANCIA DEFAULT DE OPTIMIZADOR
    "create_qmc_study",       # CREAR ESTUDIO QMC
    "score_qmc",              # FUNCIÓN STANDALONE DE SCORING
]
