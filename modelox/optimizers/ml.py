"""modelox/optimizers/ml.py

═══════════════════════════════════════════════════════════════════════════════
   ███╗   ███╗██╗         ███████╗ ██████╗ ██████╗ ███████╗███████╗████████╗
   ████╗ ████║██║         ██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔════╝╚══██╔══╝
   ██╔████╔██║██║         █████╗  ██║   ██║██████╔╝█████╗  ███████╗   ██║
   ██║╚██╔╝██║██║         ██╔══╝  ██║   ██║██╔══██╗██╔══╝  ╚════██║   ██║
   ██║ ╚═╝ ██║███████╗    ██║     ╚██████╔╝██║  ██║███████╗███████║   ██║
   ╚═╝     ╚═╝╚══════╝    ╚═╝      ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝   ╚═╝

    ML-FOREST OPTIMIZER — ANALISTA SENIOR CON MACHINE LEARNING
═══════════════════════════════════════════════════════════════════════════════

DESCRIPCIÓN:
============
MLForestOptimizer actúa como un Analista Senior que:
  1. OBSERVA lo que pasó (guarda TODO: lo bueno, lo malo y lo desastroso)
  2. ENTIENDE las causas (separa ruido de lo importante con Random Forest)
  3. SIMULA escenarios antes de arriesgar (genera candidatos inteligentes)

ARQUITECTURA DE DOS FASES:
==========================

  FASE 1 — EXPLORACIÓN (Primeros N_EXPLORATION trials):
  ╔═══════════════════════════════════════════════════════════════════╗
  ║  Como un niño explorando: prueba cosas al azar para llenar       ║
  ║  el EXPEDIENTE (memoria) con datos que analizar.                 ║
  ║  Guarda TODO: parámetros + métricas + score de cada trial.       ║
  ║  Sin ML activo. Score = 0-100 basado en métricas directas.       ║
  ╚═══════════════════════════════════════════════════════════════════╝

  FASE 2 — ML-GUIDED (Tras N_EXPLORATION trials):
  ╔═══════════════════════════════════════════════════════════════════╗
  ║  Entrena DOS "cerebros" (Random Forests):                        ║
  ║                                                                   ║
  ║  🟢 EL CODICIOSO (Buscador de Beneficios):                      ║
  ║     - Mira todos los datos y busca patrones de ganancia          ║
  ║     - Predice qué combinaciones de parámetros dan buen score     ║
  ║     - Separa mentalmente lo importante de lo irrelevante         ║
  ║                                                                   ║
  ║  🔴 EL PARANOICO (Detector de Peligro):                         ║
  ║     - Solo mira quiebras y desastres (score bajo)                ║
  ║     - Crea REGLAS DE SEGURIDAD: "prohibido SL < 1%"             ║
  ║     - Veta combinaciones peligrosas antes de probarlas           ║
  ║                                                                   ║
  ║  Cada trial nuevo:                                               ║
  ║    1. El Codicioso sugiere parámetros prometedores               ║
  ║    2. El Paranoico veta los peligrosos                           ║
  ║    3. Se ejecuta el backtest                                     ║
  ║    4. Se calcula score 0-100 (estabilidad + anti-overfitting)    ║
  ║    5. Se actualiza el Expediente                                 ║
  ╚═══════════════════════════════════════════════════════════════════╝

SCORING (0-100):
================
  - Score compuesto de estabilidad y anti-overfitting
  - HARD-KILL: Si trades_por_dia < 0.20 → Score = 0
  - Penaliza overfitting, premia consistencia
  - El ML intenta maximizar este score

VENTAJAS:
=========
  ✓ MEMORIA COMPLETA: guarda TODO (éxitos y fracasos)
  ✓ DOS CEREBROS: uno busca beneficio, otro detecta peligro
  ✓ ANTI-OVERFITTING: score diseñado contra el sobreajuste
  ✓ APRENDIZAJE CONTINUO: los modelos se re-entrenan periódicamente
  ✓ IMPORTANCIA DE FEATURES: sabe qué parámetros importan MÁS

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import gc
import math
import os
import re
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
)

import numpy as np
import optuna
import polars as pl
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import TPESampler, RandomSampler

if TYPE_CHECKING:
    pass

# =============================================================================
# IMPORTS INTERNOS
# =============================================================================
from modelox.core.engine import BacktestParams, calculate_performance_vectorized_numba
from modelox.core.metrics import resumen_metricas
from modelox.core.types import (
    BacktestConfig,
    Reporter,
    Strategy,
    TrialArtifacts,
    normalize_timeframe_to_suffix,
)
from modelox.core.exits import resolve_exit_settings_for_trial

# =============================================================================
# SILENCIAR WARNINGS
# =============================================================================
warnings.filterwarnings("ignore", category=ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# DEPENDENCIA OPCIONAL: SCIKIT-LEARN
# =============================================================================
try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# █████████████████████████████████████████████████████████████████████████████
# ██                                                                         ██
# ██   SECCIÓN 1: CONFIGURACIÓN DEL SCORING ML-FOREST                       ██
# ██                                                                         ██
# █████████████████████████████████████████████████████████████████████████████

@dataclass
class MLForestScoringConfig:
    """
    ┌────────────────────────────────────────────────────────────────────────┐
    │           CONFIGURACIÓN DEL SCORING ML-FOREST v1.0                      │
    │                                                                         │
    │  SCORE: [0, 100] — ESTABILIDAD + ANTI-OVERFITTING                      │
    │  HARD-KILL: trades_por_dia < 0.20 → Score = 0                          │
    │  FILOSOFÍA: El ML maximiza este score compuesto                        │
    └────────────────────────────────────────────────────────────────────────┘
    """

    # =========================================================================
    # 1.1 RANGO DE SALIDA DEL SCORE
    # =========================================================================
    SCORE_MIN: float = 0.0               # MÍNIMO ABSOLUTO
    SCORE_MAX: float = 100.0             # MÁXIMO ABSOLUTO

    # =========================================================================
    # 1.2 HARD-KILL: TRADES POR DÍA MÍNIMO ABSOLUTO
    # =========================================================================
    # SI TRADES_POR_DIA < ESTE VALOR → SCORE = 0 (SIN EXCEPCIONES)
    HARD_MIN_TRADES_POR_DIA: float = 0.20

    # =========================================================================
    # 1.3 COMPONENTES DEL SCORE (SUMAN 1.0)
    # =========================================================================
    # ESTABILIDAD Y ANTI-OVERFITTING
    PESO_SHARPE: float = 0.20            # SHARPE RATIO NORMALIZADO
    PESO_SQN: float = 0.15              # SYSTEM QUALITY NUMBER
    PESO_CONSISTENCIA_R2: float = 0.20   # LINEALIDAD DE LA EQUITY CURVE
    PESO_SORTINO: float = 0.10          # RATIO SORTINO (RIESGO ASIMÉTRICO)
    PESO_PROFIT_FACTOR: float = 0.10    # PROFIT FACTOR
    PESO_DRAWDOWN: float = 0.15         # PENALIZACIÓN POR DRAWDOWN
    PESO_ACTIVIDAD: float = 0.10        # TRADES/DÍA NORMALIZADOS

    # =========================================================================
    # 1.4 NORMALIZACIÓN DE MÉTRICAS
    # =========================================================================
    SHARPE_SIGMOID_CENTER: float = 1.0
    SHARPE_SIGMOID_SCALE: float = 1.5
    SQN_TARGET: float = 4.0
    SORTINO_SIGMOID_CENTER: float = 1.5
    SORTINO_SIGMOID_SCALE: float = 1.0
    PROFIT_FACTOR_TARGET: float = 2.0
    DRAWDOWN_MAX_ACCEPTABLE: float = 25.0
    DRAWDOWN_LIMIT: float = 60.0
    TRADES_DIA_TARGET: float = 1.0

    # =========================================================================
    # 1.5 UMBRALES MÍNIMOS (SOFT-VETO)
    # =========================================================================
    MIN_TRADES_TOTAL: int = 15
    MIN_TRADES_POR_DIA_SOFT: float = 0.10
    MAX_DRAWDOWN_PERMITIDO: float = 60.0
    MIN_ROI_PERMITIDO: float = -150.0
    UMBRAL_FACTOR_PENALIZACION: float = 0.20

    # =========================================================================
    # 1.6 CONSISTENCIA DE EQUITY (R²)
    # =========================================================================
    CONSISTENCIA_R2_MIN: float = 0.50

    # =========================================================================
    # 1.7 PSR (PROBABILISTIC SHARPE RATIO)
    # =========================================================================
    PSR_ENABLED: bool = True
    PSR_BENCHMARK_SR: float = 0.0
    PSR_MIN_TRADES: int = 30
    PSR_FLOOR: float = 0.30
    PSR_PESO_EN_FINAL: float = 0.10


# =============================================================================
# INSTANCIA POR DEFECTO
# =============================================================================
ML_SCORING_CONFIG = MLForestScoringConfig()


# █████████████████████████████████████████████████████████████████████████████
# ██                                                                         ██
# ██   SECCIÓN 2: CLASE MLForestScorer — SCORING 0-100                       ██
# ██                                                                         ██
# █████████████████████████████████████████████████████████████████████████████

class MLForestScorer:
    """
    ┌────────────────────────────────────────────────────────────────────────┐
    │                   SCORER ML-FOREST v1.0                                 │
    │                                                                         │
    │  SCORE: [0, 100]                                                        │
    │  HARD-KILL: trades_por_dia < 0.20 → 0                                  │
    │  FILOSOFÍA: Estabilidad + Anti-overfitting                             │
    │                                                                         │
    │  COMPONENTES:                                                           │
    │    • Sharpe normalizado (sigmoide)                                      │
    │    • SQN normalizado (lineal)                                          │
    │    • Consistencia R² de equity curve                                   │
    │    • Sortino normalizado                                               │
    │    • Profit Factor normalizado                                         │
    │    • Penalización drawdown                                             │
    │    • Actividad (trades/día)                                            │
    │    × PSR factor (probabilistic sharpe)                                 │
    │    × Soft-veto factors                                                 │
    └────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        study: Optional[optuna.Study] = None,
        config: Optional[MLForestScoringConfig] = None,
    ):
        self.study = study
        self.config = config or ML_SCORING_CONFIG

    # =========================================================================
    # FUNCIONES AUXILIARES
    # =========================================================================

    @staticmethod
    def _SAFE_GET(metrics: Mapping[str, Any], key: str, default: float = 0.0) -> float:
        """EXTRAE VALOR NUMÉRICO DE FORMA SEGURA."""
        try:
            VAL = metrics.get(key, default)
            if VAL is None:
                return default
            F_VAL = float(VAL)
            if math.isnan(F_VAL) or math.isinf(F_VAL):
                return default
            return F_VAL
        except Exception:
            return default

    @staticmethod
    def _SIGMOID(x: float, center: float = 0.0, scale: float = 1.0) -> float:
        """SIGMOIDE GENÉRICA → [0, 1]."""
        try:
            EXPONENT = -scale * (x - center)
            if EXPONENT > 500:
                return 0.0
            elif EXPONENT < -500:
                return 1.0
            return 1.0 / (1.0 + math.exp(EXPONENT))
        except (OverflowError, ValueError):
            return 0.5

    # =========================================================================
    # NORMALIZACIÓN DE COMPONENTES
    # =========================================================================

    def _NORM_SHARPE(self, sharpe: float) -> float:
        """Normaliza Sharpe → [0, 1] con sigmoide."""
        CFG = self.config
        return float(np.clip(
            self._SIGMOID(sharpe, CFG.SHARPE_SIGMOID_CENTER, CFG.SHARPE_SIGMOID_SCALE),
            0.01, 0.99
        ))

    def _NORM_SQN(self, sqn: float) -> float:
        """Normaliza SQN → [0, 1] lineal."""
        if sqn <= 0:
            return 0.0
        return float(np.clip(sqn / self.config.SQN_TARGET, 0.0, 1.0))

    def _NORM_SORTINO(self, sortino: float) -> float:
        """Normaliza Sortino → [0, 1] con sigmoide."""
        CFG = self.config
        return float(np.clip(
            self._SIGMOID(sortino, CFG.SORTINO_SIGMOID_CENTER, CFG.SORTINO_SIGMOID_SCALE),
            0.01, 0.99
        ))

    def _NORM_PROFIT_FACTOR(self, pf: float) -> float:
        """Normaliza Profit Factor → [0, 1]."""
        if pf <= 0:
            return 0.0
        return float(np.clip(pf / self.config.PROFIT_FACTOR_TARGET, 0.0, 1.0))

    def _NORM_DRAWDOWN(self, drawdown: float) -> float:
        """Penalización por drawdown → [0, 1] (1 = sin DD)."""
        CFG = self.config
        if drawdown <= CFG.DRAWDOWN_MAX_ACCEPTABLE:
            return 1.0 - 0.3 * (drawdown / CFG.DRAWDOWN_MAX_ACCEPTABLE)
        else:
            EXCESS = drawdown - CFG.DRAWDOWN_MAX_ACCEPTABLE
            MAX_EXCESS = CFG.DRAWDOWN_LIMIT - CFG.DRAWDOWN_MAX_ACCEPTABLE
            return max(0.05, 0.7 * (1.0 - min(1.0, EXCESS / MAX_EXCESS)))

    def _NORM_ACTIVIDAD(self, trades_dia: float) -> float:
        """Normaliza actividad → [0, 1] con log."""
        if trades_dia <= 0:
            return 0.0
        return float(np.clip(
            math.log1p(trades_dia) / math.log1p(self.config.TRADES_DIA_TARGET),
            0.0, 1.0
        ))

    def _CALC_R2_EQUITY(self, equity_curve: Optional[np.ndarray]) -> float:
        """Calcula R² de la curva de equity → [0, 1]."""
        CFG = self.config

        if equity_curve is None:
            return 0.5

        EQUITY = np.asarray(equity_curve, dtype=np.float64)
        EQUITY = EQUITY[np.isfinite(EQUITY)]
        N = len(EQUITY)

        if N < 10:
            return 0.1

        X = np.arange(N, dtype=np.float64)
        X_MEAN = np.mean(X)
        Y_MEAN = np.mean(EQUITY)
        SS_TOT = np.sum((EQUITY - Y_MEAN) ** 2)

        if SS_TOT < 1e-10:
            return 1.0

        NUMERATOR = np.sum((X - X_MEAN) * (EQUITY - Y_MEAN))
        DENOMINATOR = np.sum((X - X_MEAN) ** 2)
        if DENOMINATOR < 1e-10:
            return 0.1

        SLOPE = NUMERATOR / DENOMINATOR
        INTERCEPT = Y_MEAN - SLOPE * X_MEAN
        Y_PRED = SLOPE * X + INTERCEPT
        SS_RES = np.sum((EQUITY - Y_PRED) ** 2)
        R2 = 1.0 - (SS_RES / SS_TOT)
        R2 = float(np.clip(R2, 0.0, 1.0))

        if R2 < CFG.CONSISTENCIA_R2_MIN:
            R2 *= (R2 / CFG.CONSISTENCIA_R2_MIN)

        return float(np.clip(R2, 0.0, 1.0))

    def _CALC_PSR(self, returns: Optional[np.ndarray]) -> float:
        """Calcula PSR simplificado → [0, 1]."""
        CFG = self.config
        if not CFG.PSR_ENABLED or returns is None:
            return 1.0

        RETURNS_CLEAN = np.asarray(returns, dtype=np.float64)
        RETURNS_CLEAN = RETURNS_CLEAN[np.isfinite(RETURNS_CLEAN)]
        N = len(RETURNS_CLEAN)

        if N < CFG.PSR_MIN_TRADES:
            return 0.5

        MEAN_VAL = float(np.mean(RETURNS_CLEAN))
        STD_VAL = float(np.std(RETURNS_CLEAN, ddof=1))
        if STD_VAL < 1e-10:
            return 0.9 if MEAN_VAL > 0 else 0.1

        SR = MEAN_VAL / STD_VAL

        # SKEWNESS Y KURTOSIS
        M3 = float(np.mean((RETURNS_CLEAN - MEAN_VAL) ** 3))
        M4 = float(np.mean((RETURNS_CLEAN - MEAN_VAL) ** 4))
        SKEW = M3 / (STD_VAL ** 3)
        KURT = M4 / (STD_VAL ** 4) if STD_VAL > 0 else 3.0

        SR_SQ = SR ** 2
        VAR_FACTOR = max(0.01, 1.0 - SKEW * SR + ((KURT - 1.0) / 4.0) * SR_SQ)
        SIGMA_SR = math.sqrt(VAR_FACTOR / max(1, N - 1))

        if SIGMA_SR < 1e-10:
            return 0.9 if SR > CFG.PSR_BENCHMARK_SR else 0.1

        Z_SCORE = (SR - CFG.PSR_BENCHMARK_SR) / SIGMA_SR
        PSR_VAL = 0.5 * (1.0 + math.erf(Z_SCORE / math.sqrt(2)))

        return float(np.clip(PSR_VAL, 0.01, 0.99))

    # =========================================================================
    # FUNCIÓN PRINCIPAL: COMPUTE SCORE → [0, 100]
    # =========================================================================

    def COMPUTE_SCORE(
        self,
        TRIAL: Optional[optuna.Trial],
        METRICS: Mapping[str, Any],
        RETURNS: Optional[np.ndarray] = None,
        EQUITY_CURVE: Optional[np.ndarray] = None,
    ) -> float:
        """
        ┌────────────────────────────────────────────────────────────────────┐
        │           FUNCIÓN PRINCIPAL DE SCORING ML-FOREST                    │
        │                                                                     │
        │  SCORE: [0, 100]                                                    │
        │  HARD-KILL: trades_por_dia < 0.20 → 0                              │
        │                                                                     │
        │  PIPELINE:                                                          │
        │   1. Extraer métricas                                              │
        │   2. Hard-kill por trades/día                                      │
        │   3. Normalizar componentes                                        │
        │   4. Score ponderado × PSR × Soft-vetos                            │
        │   5. Escalar a [0, 100]                                            │
        └────────────────────────────────────────────────────────────────────┘
        """
        CFG = self.config

        # =====================================================================
        # PASO 1: EXTRAER MÉTRICAS
        # =====================================================================
        TRADES_DIA = self._SAFE_GET(METRICS, "trades_por_dia", 0.0)
        if TRADES_DIA == 0:
            TRADES_DIA = self._SAFE_GET(METRICS, "trades_dia", 0.0)
        if TRADES_DIA == 0:
            TRADES_DIA = self._SAFE_GET(METRICS, "trades_per_day", 0.0)

        # =====================================================================
        # HARD-KILL: trades/día < 0.20 → SCORE = 0
        # =====================================================================
        if TRADES_DIA < CFG.HARD_MIN_TRADES_POR_DIA:
            if TRIAL is not None:
                try:
                    TRIAL.set_user_attr("HARD_KILL", True)
                    TRIAL.set_user_attr("HARD_KILL_RAZON",
                        f"trades_dia={TRADES_DIA:.3f} < {CFG.HARD_MIN_TRADES_POR_DIA}")
                    TRIAL.set_user_attr("SCORE_FINAL", 0.0)
                except Exception:
                    pass
            return 0.0

        N_TRADES = int(self._SAFE_GET(METRICS, "n_trades", 0))
        if N_TRADES == 0:
            N_TRADES = int(self._SAFE_GET(METRICS, "total_trades", 0))

        DRAWDOWN = self._SAFE_GET(METRICS, "drawdown", 50.0)
        if DRAWDOWN == 0:
            DRAWDOWN = self._SAFE_GET(METRICS, "max_drawdown", 50.0)

        ROI = self._SAFE_GET(METRICS, "roi", 0.0)

        SHARPE = self._SAFE_GET(METRICS, "sharpe", 0.0)
        if SHARPE == 0:
            SHARPE = self._SAFE_GET(METRICS, "sharpe_ratio", 0.0)

        SQN = self._SAFE_GET(METRICS, "sqn", 0.0)

        SORTINO = self._SAFE_GET(METRICS, "sortino", 0.0)
        if SORTINO == 0:
            SORTINO = self._SAFE_GET(METRICS, "sortino_ratio", 0.0)

        PF = self._SAFE_GET(METRICS, "profit_factor", 0.0)

        # =====================================================================
        # PASO 1B: RECUPERAR RETURNS Y EQUITY SI NO SE PASAN
        # =====================================================================
        if RETURNS is None:
            RAW = METRICS.get("returns", None) or METRICS.get("trade_returns", None)
            if RAW is not None:
                try:
                    RETURNS = np.asarray(RAW, dtype=np.float64)
                except Exception:
                    RETURNS = None

        if EQUITY_CURVE is None:
            RAW_EQ = METRICS.get("equity_curve", None) or METRICS.get("equity", None)
            if RAW_EQ is not None:
                try:
                    EQUITY_CURVE = np.asarray(RAW_EQ, dtype=np.float64)
                except Exception:
                    EQUITY_CURVE = None

        # =====================================================================
        # PASO 2: SOFT-VETO
        # =====================================================================
        FACTOR_VETO = 1.0

        if TRADES_DIA < CFG.MIN_TRADES_POR_DIA_SOFT:
            FACTOR_VETO *= CFG.UMBRAL_FACTOR_PENALIZACION

        if N_TRADES < CFG.MIN_TRADES_TOTAL:
            FACTOR_VETO *= CFG.UMBRAL_FACTOR_PENALIZACION

        if DRAWDOWN > CFG.MAX_DRAWDOWN_PERMITIDO:
            FACTOR_VETO *= CFG.UMBRAL_FACTOR_PENALIZACION

        if ROI < CFG.MIN_ROI_PERMITIDO:
            FACTOR_VETO *= CFG.UMBRAL_FACTOR_PENALIZACION

        FACTOR_VETO = max(0.05, FACTOR_VETO)

        # =====================================================================
        # PASO 3: NORMALIZAR COMPONENTES
        # =====================================================================
        COMP_SHARPE = self._NORM_SHARPE(SHARPE)
        COMP_SQN = self._NORM_SQN(SQN)
        COMP_R2 = self._CALC_R2_EQUITY(EQUITY_CURVE)
        COMP_SORTINO = self._NORM_SORTINO(SORTINO)
        COMP_PF = self._NORM_PROFIT_FACTOR(PF)
        COMP_DD = self._NORM_DRAWDOWN(DRAWDOWN)
        COMP_ACT = self._NORM_ACTIVIDAD(TRADES_DIA)

        # =====================================================================
        # PASO 4: SCORE PONDERADO
        # =====================================================================
        WEIGHTED = (
            CFG.PESO_SHARPE * COMP_SHARPE
            + CFG.PESO_SQN * COMP_SQN
            + CFG.PESO_CONSISTENCIA_R2 * COMP_R2
            + CFG.PESO_SORTINO * COMP_SORTINO
            + CFG.PESO_PROFIT_FACTOR * COMP_PF
            + CFG.PESO_DRAWDOWN * COMP_DD
            + CFG.PESO_ACTIVIDAD * COMP_ACT
        )

        # =====================================================================
        # PASO 5: PSR FACTOR
        # =====================================================================
        PSR_VAL = self._CALC_PSR(RETURNS)
        PSR_FACTOR = 1.0 - CFG.PSR_PESO_EN_FINAL * (1.0 - max(CFG.PSR_FLOOR, PSR_VAL))

        # =====================================================================
        # PASO 6: COMBINAR Y ESCALAR A [0, 100]
        # =====================================================================
        SCORE_CRUDO = WEIGHTED * PSR_FACTOR * FACTOR_VETO
        SCORE_FINAL = CFG.SCORE_MAX * SCORE_CRUDO
        SCORE_FINAL = max(CFG.SCORE_MIN, min(CFG.SCORE_MAX, SCORE_FINAL))

        # =====================================================================
        # AUDITORÍA
        # =====================================================================
        if TRIAL is not None:
            try:
                TRIAL.set_user_attr("COMP_SHARPE", float(COMP_SHARPE))
                TRIAL.set_user_attr("COMP_SQN", float(COMP_SQN))
                TRIAL.set_user_attr("COMP_R2", float(COMP_R2))
                TRIAL.set_user_attr("COMP_SORTINO", float(COMP_SORTINO))
                TRIAL.set_user_attr("COMP_PF", float(COMP_PF))
                TRIAL.set_user_attr("COMP_DD", float(COMP_DD))
                TRIAL.set_user_attr("COMP_ACT", float(COMP_ACT))
                TRIAL.set_user_attr("PSR", float(PSR_VAL))
                TRIAL.set_user_attr("PSR_FACTOR", float(PSR_FACTOR))
                TRIAL.set_user_attr("FACTOR_VETO", float(FACTOR_VETO))
                TRIAL.set_user_attr("SCORE_FINAL", float(SCORE_FINAL))
                TRIAL.set_user_attr("SR_NOMINAL", float(SHARPE))
            except Exception:
                pass

        return float(SCORE_FINAL)


# █████████████████████████████████████████████████████████████████████████████
# ██                                                                         ██
# ██   SECCIÓN 3: EL EXPEDIENTE — MEMORIA COMPLETA DE TRIALS                ██
# ██                                                                         ██
# █████████████████████████████████████████████████████████████████████████████

class _Expediente:
    """
    ┌────────────────────────────────────────────────────────────────────────┐
    │                      EL EXPEDIENTE (MEMORIA)                            │
    │                                                                         │
    │  Guarda TODO: lo bueno, lo malo y lo desastroso.                       │
    │  Cada entrada = (parámetros_numéricos, parámetros_categóricos,         │
    │                   métricas_clave, score)                                │
    │                                                                         │
    │  Alimenta los dos cerebros (Random Forests).                           │
    └────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self):
        self.PARAMS_NUM: List[Dict[str, float]] = []    # Parámetros numéricos
        self.PARAMS_CAT: List[Dict[str, str]] = []      # Parámetros categóricos
        self.METRICS: List[Dict[str, float]] = []       # Métricas clave
        self.SCORES: List[float] = []                   # Score 0-100
        self.IS_DISASTER: List[bool] = []               # ¿Fue un desastre?

        # Claves descubiertas
        self._NUM_KEYS: set = set()
        self._CAT_KEYS: set = set()

    def REGISTRAR(
        self,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        score: float,
        disaster_threshold: float = 15.0,
    ) -> None:
        """Registra un trial completo en el expediente."""
        NUM = {}
        CAT = {}
        for K, V in params.items():
            if K.startswith("__"):
                continue
            if isinstance(V, (int, float)):
                if math.isfinite(V):
                    NUM[K] = float(V)
                    self._NUM_KEYS.add(K)
            elif isinstance(V, str):
                CAT[K] = V
                self._CAT_KEYS.add(K)
            elif isinstance(V, bool):
                NUM[K] = 1.0 if V else 0.0
                self._NUM_KEYS.add(K)

        self.PARAMS_NUM.append(NUM)
        self.PARAMS_CAT.append(CAT)
        self.METRICS.append(dict(metrics))
        self.SCORES.append(score)
        self.IS_DISASTER.append(score < disaster_threshold)

    def __len__(self) -> int:
        return len(self.SCORES)

    def TO_FEATURE_MATRIX(self) -> Tuple[Optional[np.ndarray], List[str]]:
        """
        Convierte el expediente en una matriz de features para ML.

        Returns:
            (X, feature_names) o (None, []) si no hay datos
        """
        if not self.PARAMS_NUM:
            return None, []

        ALL_NUM_KEYS = sorted(self._NUM_KEYS)
        ALL_CAT_KEYS = sorted(self._CAT_KEYS)

        # Codificar categóricos con LabelEncoder
        CAT_ENCODERS: Dict[str, LabelEncoder] = {}
        CAT_ENCODED: Dict[str, List[float]] = {}

        if SKLEARN_AVAILABLE and ALL_CAT_KEYS:
            for KEY in ALL_CAT_KEYS:
                VALUES = [P.get(KEY, "__MISSING__") for P in self.PARAMS_CAT]
                LE = LabelEncoder()
                ENCODED = LE.fit_transform(VALUES).astype(float)
                CAT_ENCODERS[KEY] = LE
                CAT_ENCODED[KEY] = ENCODED.tolist()

        FEATURE_NAMES = ALL_NUM_KEYS + [f"CAT_{K}" for K in ALL_CAT_KEYS]
        N = len(self.SCORES)
        N_FEATURES = len(FEATURE_NAMES)

        if N_FEATURES == 0:
            return None, []

        X = np.zeros((N, N_FEATURES), dtype=np.float64)

        for I in range(N):
            for J, KEY in enumerate(ALL_NUM_KEYS):
                X[I, J] = self.PARAMS_NUM[I].get(KEY, 0.0)

            for J, KEY in enumerate(ALL_CAT_KEYS):
                COL_IDX = len(ALL_NUM_KEYS) + J
                if KEY in CAT_ENCODED:
                    X[I, COL_IDX] = CAT_ENCODED[KEY][I]

        return X, FEATURE_NAMES

    def GET_SCORES_ARRAY(self) -> np.ndarray:
        return np.array(self.SCORES, dtype=np.float64)

    def GET_DISASTERS_ARRAY(self) -> np.ndarray:
        return np.array(self.IS_DISASTER, dtype=np.int32)


# █████████████████████████████████████████████████████████████████████████████
# ██                                                                         ██
# ██   SECCIÓN 4: LOS DOS CEREBROS — RANDOM FOREST ML                       ██
# ██                                                                         ██
# █████████████████████████████████████████████████████████████████████████████

class _CerebroML:
    """
    ┌────────────────────────────────────────────────────────────────────────┐
    │                   LOS DOS CEREBROS DEL ANALISTA                         │
    │                                                                         │
    │  🟢 EL CODICIOSO (RandomForestRegressor):                              │
    │     Predice el SCORE esperado para una combinación de parámetros.       │
    │     Busca patrones de GANANCIA. Separa lo importante del ruido.        │
    │                                                                         │
    │  🔴 EL PARANOICO (RandomForestClassifier):                             │
    │     Predice si una combinación será un DESASTRE (score < umbral).      │
    │     Crea REGLAS DE SEGURIDAD. Veta combinaciones peligrosas.           │
    └────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, n_estimators: int = 100, random_state: Optional[int] = None):
        self._N_ESTIMATORS = n_estimators
        self._RANDOM_STATE = random_state

        # Los modelos (se crean al entrenar)
        self._CODICIOSO: Optional[RandomForestRegressor] = None
        self._PARANOICO: Optional[RandomForestClassifier] = None
        self._FEATURE_NAMES: List[str] = []
        self._FEATURE_IMPORTANCES: Optional[np.ndarray] = None
        self._IS_TRAINED: bool = False

    def ENTRENAR(self, expediente: _Expediente) -> bool:
        """
        Entrena ambos cerebros con el expediente completo.

        Returns:
            True si se entrenó exitosamente, False si no hay datos suficientes
        """
        if not SKLEARN_AVAILABLE:
            return False

        X, FEATURE_NAMES = expediente.TO_FEATURE_MATRIX()
        if X is None or len(FEATURE_NAMES) == 0:
            return False

        Y_SCORE = expediente.GET_SCORES_ARRAY()
        Y_DISASTER = expediente.GET_DISASTERS_ARRAY()

        N = len(Y_SCORE)
        if N < 20:
            return False

        self._FEATURE_NAMES = FEATURE_NAMES

        try:
            # ─────────────────────────────────────────────────────────────
            # 🟢 ENTRENAR EL CODICIOSO (Regressor → predice score)
            # ─────────────────────────────────────────────────────────────
            self._CODICIOSO = RandomForestRegressor(
                n_estimators=self._N_ESTIMATORS,
                max_depth=min(10, max(3, N // 20)),
                min_samples_leaf=max(2, N // 50),
                random_state=self._RANDOM_STATE,
                n_jobs=1,
            )
            self._CODICIOSO.fit(X, Y_SCORE)

            # Guardar importancias de features
            self._FEATURE_IMPORTANCES = self._CODICIOSO.feature_importances_

            # ─────────────────────────────────────────────────────────────
            # 🔴 ENTRENAR EL PARANOICO (Classifier → predice desastre)
            # ─────────────────────────────────────────────────────────────
            # Solo entrenar si hay suficientes desastres para aprender
            N_DISASTERS = int(np.sum(Y_DISASTER))
            if N_DISASTERS >= 5 and N_DISASTERS < N - 5:
                self._PARANOICO = RandomForestClassifier(
                    n_estimators=self._N_ESTIMATORS,
                    max_depth=min(8, max(3, N // 25)),
                    min_samples_leaf=max(2, N // 50),
                    class_weight="balanced",  # Compensa desbalanceo
                    random_state=self._RANDOM_STATE,
                    n_jobs=1,
                )
                self._PARANOICO.fit(X, Y_DISASTER)
            else:
                self._PARANOICO = None

            self._IS_TRAINED = True
            return True

        except Exception:
            self._IS_TRAINED = False
            return False

    def PREDECIR_SCORE(self, X: np.ndarray) -> np.ndarray:
        """
        🟢 EL CODICIOSO predice el score esperado.

        Args:
            X: Matriz de features (N_samples, N_features)

        Returns:
            Array de scores predichos
        """
        if not self._IS_TRAINED or self._CODICIOSO is None:
            return np.full(len(X), 50.0)

        try:
            return self._CODICIOSO.predict(X)
        except Exception:
            return np.full(len(X), 50.0)

    def PREDECIR_PELIGRO(self, X: np.ndarray) -> np.ndarray:
        """
        🔴 EL PARANOICO predice la probabilidad de desastre.

        Args:
            X: Matriz de features (N_samples, N_features)

        Returns:
            Array de probabilidades de desastre [0, 1]
        """
        if not self._IS_TRAINED or self._PARANOICO is None:
            return np.zeros(len(X))

        try:
            # predict_proba retorna [[p_no_disaster, p_disaster], ...]
            PROBA = self._PARANOICO.predict_proba(X)
            if PROBA.shape[1] >= 2:
                return PROBA[:, 1]
            return np.zeros(len(X))
        except Exception:
            return np.zeros(len(X))

    def GET_IMPORTANCIAS(self) -> Dict[str, float]:
        """Retorna la importancia de cada feature según el Codicioso."""
        if self._FEATURE_IMPORTANCES is None or not self._FEATURE_NAMES:
            return {}
        return dict(zip(self._FEATURE_NAMES, self._FEATURE_IMPORTANCES.tolist()))

    @property
    def IS_TRAINED(self) -> bool:
        return self._IS_TRAINED


# █████████████████████████████████████████████████████████████████████████████
# ██                                                                         ██
# ██   SECCIÓN 5: SAMPLER ML-FOREST — GENERA CANDIDATOS INTELIGENTES        ██
# ██                                                                         ██
# █████████████████████████████████████████████████████████████████████████████

class _MLForestSampler(optuna.samplers.BaseSampler):
    """
    ┌────────────────────────────────────────────────────────────────────────┐
    │                SAMPLER ML-FOREST HÍBRIDO                                │
    │                                                                         │
    │  FASE 1 (Exploración): Usa TPESampler para explorar                   │
    │  FASE 2 (ML-Guided):   Genera N candidatos con TPE, luego:            │
    │    1. El Codicioso puntúa cada candidato                              │
    │    2. El Paranoico veta los peligrosos                                │
    │    3. Se elige el mejor candidato no-vetado                           │
    └────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        expediente: _Expediente,
        cerebro: _CerebroML,
        n_exploration: int = 500,
        n_candidates: int = 50,
        danger_threshold: float = 0.60,
        retrain_every: int = 50,
        seed: Optional[int] = None,
    ):
        self._EXPEDIENTE = expediente
        self._CEREBRO = cerebro
        self._N_EXPLORATION = n_exploration
        self._N_CANDIDATES = n_candidates
        self._DANGER_THRESHOLD = danger_threshold
        self._RETRAIN_EVERY = retrain_every
        self._SEED = seed

        # Sampler base para exploración y generación de candidatos
        self._TPE = TPESampler(
            seed=seed,
            n_startup_trials=10,
            multivariate=True,
            group=True,
            constant_liar=True,
        )

        # Estado
        self._LAST_TRAIN_SIZE: int = 0
        self._TRIAL_COUNT: int = 0

    def infer_relative_search_space(
        self, study: optuna.Study, trial: optuna.trial.FrozenTrial
    ) -> Dict[str, optuna.distributions.BaseDistribution]:
        """Delega al TPE para inferir el espacio de búsqueda."""
        return self._TPE.infer_relative_search_space(study, trial)

    def sample_relative(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        search_space: Dict[str, optuna.distributions.BaseDistribution],
    ) -> Dict[str, Any]:
        """
        LÓGICA PRINCIPAL DE MUESTREO:

        FASE 1 (< N_EXPLORATION): Delega al TPE
        FASE 2 (>= N_EXPLORATION): ML-guided selection
        """
        self._TRIAL_COUNT += 1

        # ─────────────────────────────────────────────────────────────────
        # FASE 1: EXPLORACIÓN PURA
        # ─────────────────────────────────────────────────────────────────
        if self._TRIAL_COUNT <= self._N_EXPLORATION:
            return self._TPE.sample_relative(study, trial, search_space)

        # ─────────────────────────────────────────────────────────────────
        # RE-ENTRENAR CEREBROS PERIÓDICAMENTE
        # ─────────────────────────────────────────────────────────────────
        CURRENT_SIZE = len(self._EXPEDIENTE)
        NEEDS_RETRAIN = (
            not self._CEREBRO.IS_TRAINED
            or CURRENT_SIZE - self._LAST_TRAIN_SIZE >= self._RETRAIN_EVERY
        )

        if NEEDS_RETRAIN and CURRENT_SIZE >= 20:
            SUCCESS = self._CEREBRO.ENTRENAR(self._EXPEDIENTE)
            if SUCCESS:
                self._LAST_TRAIN_SIZE = CURRENT_SIZE

        # ─────────────────────────────────────────────────────────────────
        # FASE 2: ML-GUIDED SELECTION
        # ─────────────────────────────────────────────────────────────────
        if not self._CEREBRO.IS_TRAINED or not search_space:
            # Fallback al TPE si no hay modelo entrenado
            return self._TPE.sample_relative(study, trial, search_space)

        # Generar N candidatos con TPE
        CANDIDATES: List[Dict[str, Any]] = []
        for _ in range(self._N_CANDIDATES):
            try:
                CAND = self._TPE.sample_relative(study, trial, search_space)
                if CAND:
                    CANDIDATES.append(CAND)
            except Exception:
                continue

        if not CANDIDATES:
            return self._TPE.sample_relative(study, trial, search_space)

        # Convertir candidatos a matriz de features
        FEATURE_NAMES = self._CEREBRO._FEATURE_NAMES
        if not FEATURE_NAMES:
            return self._TPE.sample_relative(study, trial, search_space)

        X_CAND = np.zeros((len(CANDIDATES), len(FEATURE_NAMES)), dtype=np.float64)
        for I, CAND in enumerate(CANDIDATES):
            for J, FNAME in enumerate(FEATURE_NAMES):
                # Quitar prefijo "CAT_" para categóricos
                CLEAN_NAME = FNAME[4:] if FNAME.startswith("CAT_") else FNAME
                VAL = CAND.get(CLEAN_NAME, 0.0)
                if isinstance(VAL, (int, float)):
                    X_CAND[I, J] = float(VAL)
                # Los categóricos se dejan en 0 (simplificación)

        # 🟢 El Codicioso puntúa los candidatos
        SCORES_PRED = self._CEREBRO.PREDECIR_SCORE(X_CAND)

        # 🔴 El Paranoico detecta peligros
        DANGER_PROBS = self._CEREBRO.PREDECIR_PELIGRO(X_CAND)

        # Combinar: score ajustado = score_predicho × (1 - peligro)
        # Vetar candidatos con probabilidad de peligro > umbral
        BEST_IDX = -1
        BEST_ADJUSTED = -float("inf")

        for I in range(len(CANDIDATES)):
            if DANGER_PROBS[I] > self._DANGER_THRESHOLD:
                continue  # 🔴 VETADO POR EL PARANOICO

            ADJUSTED = SCORES_PRED[I] * (1.0 - DANGER_PROBS[I])
            if ADJUSTED > BEST_ADJUSTED:
                BEST_ADJUSTED = ADJUSTED
                BEST_IDX = I

        if BEST_IDX >= 0:
            return CANDIDATES[BEST_IDX]
        else:
            # Todos vetados → usar el de menor peligro
            SAFEST_IDX = int(np.argmin(DANGER_PROBS))
            return CANDIDATES[SAFEST_IDX]

    def sample_independent(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        name: str,
        distribution: optuna.distributions.BaseDistribution,
    ) -> Any:
        """Delega muestreo independiente al TPE."""
        return self._TPE.sample_independent(study, trial, name, distribution)


# █████████████████████████████████████████████████████████████████████████████
# ██                                                                         ██
# ██   SECCIÓN 6: CONFIGURACIÓN DEL OPTIMIZADOR ML-FOREST                   ██
# ██                                                                         ██
# █████████████████████████████████████████████████████████████████████████████

@dataclass
class MLForestOptimizerConfig:
    """
    ┌────────────────────────────────────────────────────────────────────────┐
    │           CONFIGURACIÓN DEL OPTIMIZADOR ML-FOREST v1.0                  │
    │                                                                         │
    │  DOS FASES:                                                             │
    │    Fase 1 (Exploración): N_EXPLORATION trials con TPE                  │
    │    Fase 2 (ML-Guided):   Random Forest guía la búsqueda               │
    └────────────────────────────────────────────────────────────────────────┘
    """

    # =========================================================================
    # 6.1 CONFIGURACIÓN GENERAL
    # =========================================================================
    SEED: Optional[int] = None
    N_JOBS: int = 1
    STORAGE: Optional[str] = None
    STUDY_NAME_PREFIX: str = "MODELOX"

    # =========================================================================
    # 6.2 FASE 1: EXPLORACIÓN
    # =========================================================================
    # Porcentaje de trials dedicados a exploración pura (con TPE)
    # antes de activar ML. Se calcula como % de N_TRIALS.
    #   0.20 = 20% → Si N_TRIALS=1000, explora 200 y luego ML guía 800
    N_EXPLORATION_PCT: float = 0.20

    # =========================================================================
    # 6.3 FASE 2: ML-GUIDED
    # =========================================================================
    # Random Forest: número de árboles
    N_ESTIMATORS: int = 100
    # Candidatos generados por TPE para que el ML elija el mejor
    N_CANDIDATES: int = 50
    # Probabilidad de peligro para vetar un candidato
    DANGER_THRESHOLD: float = 0.60
    # Re-entrenar cada N trials nuevos
    RETRAIN_EVERY: int = 50
    # Score por debajo del cual se considera "desastre" (para el Paranoico)
    DISASTER_THRESHOLD: float = 15.0


# =============================================================================
# INSTANCIA POR DEFECTO
# =============================================================================
ML_OPTIMIZER_CONFIG = MLForestOptimizerConfig()


# █████████████████████████████████████████████████████████████████████████████
# ██                                                                         ██
# ██   SECCIÓN 7: CLASE OPTIMIZADOR ML-FOREST                               ██
# ██                                                                         ██
# █████████████████████████████████████████████████████████████████████████████

class MLForestOptimizer:
    """
    ┌────────────────────────────────────────────────────────────────────────┐
    │                    OPTIMIZADOR ML-FOREST v1.0                           │
    │                                                                         │
    │  EL ANALISTA SENIOR CON MACHINE LEARNING                               │
    │                                                                         │
    │  FASE 1: Explora al azar (llena el Expediente)                         │
    │  FASE 2: Entrena 2 cerebros (Codicioso + Paranoico)                    │
    │          y guía la búsqueda con ML                                     │
    │                                                                         │
    │  SCORING: [0, 100] — Estabilidad + Anti-overfitting                    │
    │  HARD-KILL: trades/día < 0.20 → Score = 0                             │
    └────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        config: BacktestConfig,
        n_trials: int,
        reporters: Sequence[Reporter] = (),
        optimizer_config: Optional[MLForestOptimizerConfig] = None,
        scoring_config: Optional[MLForestScoringConfig] = None,
        activo: Optional[str] = None,
    ):
        self.config = config
        self.n_trials = n_trials
        self.reporters = list(reporters)
        self.optimizer_config = optimizer_config or ML_OPTIMIZER_CONFIG
        self.scoring_config = scoring_config or ML_SCORING_CONFIG
        self.activo = activo

        # Estado interno
        self._LAST_STUDY: Optional[optuna.Study] = None
        self._SCORER: Optional[MLForestScorer] = None

        # Los componentes ML
        self._EXPEDIENTE = _Expediente()
        self._CEREBRO = _CerebroML(
            n_estimators=self.optimizer_config.N_ESTIMATORS,
            random_state=self.optimizer_config.SEED,
        )

    # =========================================================================
    # [7.1] CREAR ESTUDIO OPTUNA
    # =========================================================================

    def _CREATE_STUDY(self, STRATEGY_NAME: str) -> optuna.Study:
        """Crea un estudio con el sampler ML-Forest híbrido."""
        CFG = self.optimizer_config

        PARTS = [CFG.STUDY_NAME_PREFIX, "MLFOREST", str(STRATEGY_NAME)]
        if self.activo:
            PARTS.append(str(self.activo))
        STUDY_NAME = self._SLUG("_".join(PARTS))

        # Calcular trials de exploración como 20% de n_trials
        N_EXPLORATION = max(20, int(CFG.N_EXPLORATION_PCT * self.n_trials))

        # Crear sampler ML-Forest
        SAMPLER = _MLForestSampler(
            expediente=self._EXPEDIENTE,
            cerebro=self._CEREBRO,
            n_exploration=N_EXPLORATION,
            n_candidates=CFG.N_CANDIDATES,
            danger_threshold=CFG.DANGER_THRESHOLD,
            retrain_every=CFG.RETRAIN_EVERY,
            seed=CFG.SEED,
        )

        STUDY = optuna.create_study(
            direction="maximize",
            sampler=SAMPLER,
            study_name=STUDY_NAME,
            storage=CFG.STORAGE,
            load_if_exists=False,
        )

        self._SCORER = MLForestScorer(study=STUDY, config=self.scoring_config)

        return STUDY

    @staticmethod
    def _SLUG(s: str) -> str:
        s = s.strip().lower()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        return s.strip("_")[:60]

    # =========================================================================
    # [7.2] PREPARAR PARÁMETROS
    # =========================================================================

    def _PREPARE_PARAMS(
        self,
        TRIAL: optuna.Trial,
        STRATEGY: Strategy,
        BASE_TF: str,
    ) -> Dict[str, Any]:
        """Prepara parámetros para un trial (idéntico a GT/TPE/CMA)."""
        PARAMS_PUROS = STRATEGY.suggest_params(TRIAL)
        PARAMS_RT = dict(PARAMS_PUROS)

        PARAMS_RT["__activo"] = self.activo
        PARAMS_RT["__saldo_inicial"] = float(self.config.saldo_inicial)
        PARAMS_RT["__saldo_operativo_max"] = float(self.config.saldo_operativo_max)


        PARAMS_RT["__comision_pct"] = float(self.config.comision_pct)
        PARAMS_RT["__comision_sides"] = int(self.config.comision_sides)
        PARAMS_RT["__saldo_usado"] = float(self.config.saldo_usado)
        PARAMS_RT["__apalancamiento_max"] = float(self.config.apalancamiento_max)
        PARAMS_RT["__strategy_exit_enabled"] = bool(
            getattr(STRATEGY, "SALIDAS_PERSONALIZADAS", False)
        )

        EXIT_SETTINGS = resolve_exit_settings_for_trial(trial=TRIAL, config=self.config)
        PARAMS_RT["__exit_type"] = EXIT_SETTINGS.exit_type
        PARAMS_RT["__exit_sl_pct"] = EXIT_SETTINGS.sl_pct
        PARAMS_RT["__exit_tp_pct"] = EXIT_SETTINGS.tp_pct
        PARAMS_RT["__exit_trail_act_pct"] = EXIT_SETTINGS.trail_act_pct
        PARAMS_RT["__exit_trail_dist_pct"] = EXIT_SETTINGS.trail_dist_pct

        PARAMS_RT["exit_type"] = EXIT_SETTINGS.exit_type
        PARAMS_RT["exit_sl_pct"] = EXIT_SETTINGS.sl_pct
        PARAMS_RT["exit_tp_pct"] = EXIT_SETTINGS.tp_pct
        PARAMS_RT["exit_trail_act_pct"] = EXIT_SETTINGS.trail_act_pct
        PARAMS_RT["exit_trail_dist_pct"] = EXIT_SETTINGS.trail_dist_pct

        ENTRY_TF = normalize_timeframe_to_suffix(
            getattr(STRATEGY, "timeframe_entry", None) or BASE_TF
        )
        EXIT_TF = normalize_timeframe_to_suffix(
            getattr(STRATEGY, "timeframe_exit", None) or BASE_TF
        )
        PARAMS_RT["__timeframe_base"] = BASE_TF
        PARAMS_RT["__timeframe_entry"] = ENTRY_TF
        PARAMS_RT["__timeframe_exit"] = EXIT_TF

        return PARAMS_RT

    # =========================================================================
    # [7.3] FUNCIÓN OBJETIVO
    # =========================================================================

    def _CREATE_OBJECTIVE(
        self,
        DF_BASE: pl.DataFrame,
        DF_MAP: Dict[str, pl.DataFrame],
        STRATEGY: Strategy,
        BASE_TF: str,
    ) -> Callable[[optuna.Trial], float]:
        """Crea la función objetivo que incluye registro en el Expediente."""

        from modelox.core.runner import SignalGenerator, BacktestEngine, periodic_cleanup

        def OBJECTIVE(TRIAL: optuna.Trial) -> float:
            T0 = time.perf_counter()

            periodic_cleanup(TRIAL.number)

            PARAMS_RT = self._PREPARE_PARAMS(TRIAL, STRATEGY, BASE_TF)
            ENTRY_TF = PARAMS_RT["__timeframe_entry"]
            DF_ENTRY = DF_MAP.get(ENTRY_TF, DF_BASE)

            # ─────────────────────────────────────────────────────────────
            # GENERAR SEÑALES + BACKTEST
            # ─────────────────────────────────────────────────────────────
            SIGNALS_DF = SignalGenerator.generate_signals(
                DF_ENTRY, STRATEGY, PARAMS_RT, DF_MAP
            )

            TRADES_DF, EQUITY_CURVE, METRICS = BacktestEngine.run_backtest(
                DF_ENTRY, SIGNALS_DF, self.config, PARAMS_RT, STRATEGY,
            )

            if TRADES_DF.is_empty():
                # Registrar fracaso en el Expediente
                self._EXPEDIENTE.REGISTRAR(
                    params=PARAMS_RT,
                    metrics={"roi": 0.0, "sharpe": 0.0, "sqn": 0.0},
                    score=0.0,
                    disaster_threshold=self.optimizer_config.DISASTER_THRESHOLD,
                )
                return 0.0

            TRIAL.set_user_attr("metricas", METRICS)

            # ─────────────────────────────────────────────────────────────
            # CALCULAR SCORE [0, 100]
            # ─────────────────────────────────────────────────────────────
            SCORE = self._SCORER.COMPUTE_SCORE(
                TRIAL=TRIAL,
                METRICS=METRICS,
                EQUITY_CURVE=np.array(EQUITY_CURVE) if EQUITY_CURVE else None,
            )

            # ─────────────────────────────────────────────────────────────
            # REGISTRAR EN EL EXPEDIENTE (TODO: bueno, malo, desastroso)
            # ─────────────────────────────────────────────────────────────
            METRICS_CLAVE = {
                "roi": self._SCORER._SAFE_GET(METRICS, "roi", 0.0),
                "sharpe": self._SCORER._SAFE_GET(METRICS, "sharpe", 0.0),
                "sqn": self._SCORER._SAFE_GET(METRICS, "sqn", 0.0),
                "drawdown": self._SCORER._SAFE_GET(METRICS, "drawdown", 50.0),
                "trades_dia": self._SCORER._SAFE_GET(METRICS, "trades_por_dia", 0.0),
                "sortino": self._SCORER._SAFE_GET(METRICS, "sortino", 0.0),
                "profit_factor": self._SCORER._SAFE_GET(METRICS, "profit_factor", 0.0),
                "n_trades": self._SCORER._SAFE_GET(METRICS, "n_trades", 0.0),
            }

            self._EXPEDIENTE.REGISTRAR(
                params=PARAMS_RT,
                metrics=METRICS_CLAVE,
                score=SCORE,
                disaster_threshold=self.optimizer_config.DISASTER_THRESHOLD,
            )

            # ─────────────────────────────────────────────────────────────
            # INFO ML EN TRIAL (PARA AUDITORÍA)
            # ─────────────────────────────────────────────────────────────
            try:
                TRIAL.set_user_attr("ML_EXPEDIENTE_SIZE", len(self._EXPEDIENTE))
                TRIAL.set_user_attr("ML_CEREBRO_TRAINED", self._CEREBRO.IS_TRAINED)
                # Calcular umbral de exploración dinámico
                _N_EXPL = max(20, int(self.optimizer_config.N_EXPLORATION_PCT * self.n_trials))
                TRIAL.set_user_attr("ML_FASE",
                    "EXPLORACIÓN" if TRIAL.number < _N_EXPL
                    else "ML-GUIDED"
                )
                if self._CEREBRO.IS_TRAINED:
                    IMPORTANCIAS = self._CEREBRO.GET_IMPORTANCIAS()
                    TOP_3 = sorted(IMPORTANCIAS.items(), key=lambda x: x[1], reverse=True)[:3]
                    TRIAL.set_user_attr("ML_TOP_FEATURES",
                        ", ".join(f"{k}: {v:.3f}" for k, v in TOP_3)
                    )
            except Exception:
                pass

            # ─────────────────────────────────────────────────────────────
            # CREAR ARTIFACTS
            # ─────────────────────────────────────────────────────────────
            ARTIFACTS = TrialArtifacts(
                strategy_name=STRATEGY.name,
                trial_number=TRIAL.number,
                params=PARAMS_RT,
                params_reporting=PARAMS_RT,
                score=SCORE,
                metrics=METRICS,
                df_signals=None,
                trades=TRADES_DF.to_pandas(),
                equity_curve=EQUITY_CURVE,
                indicators_used=PARAMS_RT.get("__indicators_used", []),
            )

            for REPORTER in self.reporters:
                REPORTER.on_trial_end(ARTIFACTS)

            return SCORE

        return OBJECTIVE

    # =========================================================================
    # [7.4] OPTIMIZAR — PUNTO DE ENTRADA PRINCIPAL
    # =========================================================================

    def optimize(
        self,
        *,
        df: pl.DataFrame,
        strategy: Strategy,
        df_by_timeframe: Optional[Dict[str, pl.DataFrame]] = None,
        base_timeframe: Optional[str] = None,
    ) -> optuna.Study:
        """
        ┌────────────────────────────────────────────────────────────────────┐
        │              EJECUTAR OPTIMIZACIÓN ML-FOREST                        │
        │                                                                     │
        │  FASE 1: Exploración con TPE (N_EXPLORATION trials)                │
        │  FASE 2: ML-Guided con Random Forest                               │
        │  SCORING: [0, 100] — Estabilidad + Anti-overfitting                │
        │  HARD-KILL: trades/día < 0.20 → 0                                  │
        └────────────────────────────────────────────────────────────────────┘
        """
        BASE_TF = base_timeframe or "1m"
        DF_MAP = df_by_timeframe or {BASE_TF: df}
        DF_BASE = DF_MAP.get(BASE_TF, df)

        STUDY = self._CREATE_STUDY(strategy.name)

        OBJECTIVE = self._CREATE_OBJECTIVE(DF_BASE, DF_MAP, strategy, BASE_TF)

        STUDY.optimize(
            OBJECTIVE,
            n_trials=int(self.n_trials),
            n_jobs=int(self.optimizer_config.N_JOBS),
            gc_after_trial=True,
            catch=(Exception,),
        )

        self._LAST_STUDY = STUDY
        return STUDY

    # =========================================================================
    # [7.5] PROPIEDADES
    # =========================================================================

    @property
    def last_study(self) -> Optional[optuna.Study]:
        return self._LAST_STUDY

    @property
    def scorer(self) -> Optional[MLForestScorer]:
        return self._SCORER

    @property
    def expediente(self) -> _Expediente:
        return self._EXPEDIENTE

    @property
    def cerebro(self) -> _CerebroML:
        return self._CEREBRO


# █████████████████████████████████████████████████████████████████████████████
# ██                                                                         ██
# ██   SECCIÓN 8: FUNCIONES DE UTILIDAD — STANDALONE                         ██
# ██                                                                         ██
# █████████████████████████████████████████████████████████████████████████████


def _SLUG(s: str) -> str:
    """GENERA UN SLUG VÁLIDO PARA NOMBRES DE ESTUDIO."""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "study"


def create_ml_study(
    strategy_name: str,
    activo: Optional[str] = None,
    seed: Optional[int] = None,
    study_name_prefix: str = "MODELOX",
    storage: Optional[str] = None,
) -> optuna.Study:
    """
    CREA UN ESTUDIO OPTUNA CON SAMPLER ML-FOREST.

    NOTA: Esta función standalone crea un estudio con TPE como base.
    Para el ML completo (Expediente + Cerebros), usar MLForestOptimizer.

    Args:
        strategy_name: Nombre de la estrategia
        activo: Nombre del activo (opcional)
        seed: Semilla aleatoria
        study_name_prefix: Prefijo para el nombre
        storage: URI de almacenamiento (None = RAM)

    Returns:
        optuna.Study configurado
    """
    PARTS = [study_name_prefix, "MLFOREST", str(strategy_name)]
    if activo:
        PARTS.append(str(activo))
    STUDY_NAME = _SLUG("_".join(PARTS))

    # Para el factory standalone, usamos TPE como base
    # (el ML completo solo funciona con MLForestOptimizer)
    SAMPLER = TPESampler(
        seed=seed,
        n_startup_trials=10,
        multivariate=True,
        group=True,
    )

    STUDY = optuna.create_study(
        direction="maximize",
        sampler=SAMPLER,
        study_name=STUDY_NAME,
        storage=storage,
        load_if_exists=False,
    )

    return STUDY


def score_ml(
    metrics: Mapping[str, Any],
    trial: Optional[optuna.Trial] = None,
    equity_curve: Optional[List[float]] = None,
) -> float:
    """
    FUNCIÓN DE SCORING ML-FOREST STANDALONE.

    Calcula el score [0, 100] sin necesidad de un optimizador completo.

    USO:
        from modelox.optimizers.ml import score_ml
        SCORE = score_ml(metrics)
    """
    SCORER = MLForestScorer()
    return SCORER.COMPUTE_SCORE(
        TRIAL=trial,
        METRICS=metrics,
        EQUITY_CURVE=np.array(equity_curve) if equity_curve else None,
    )


# =============================================================================
# EXPORTACIONES
# =============================================================================

__all__ = [
    # CLASE OPTIMIZADOR
    "MLForestOptimizer",
    # CONFIGURACIONES
    "MLForestOptimizerConfig",
    "MLForestScoringConfig",
    # INSTANCIAS DEFAULT
    "ML_SCORING_CONFIG",
    "ML_OPTIMIZER_CONFIG",
    # CLASE SCORER
    "MLForestScorer",
    # FUNCIONES STANDALONE
    "create_ml_study",
    "score_ml",
]
