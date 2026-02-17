"""modelox/optimizers/tpe.py

═══════════════════════════════════════════════════════════════════════════════
   ████████╗██████╗ ███████╗
   ╚══██╔══╝██╔══██╗██╔════╝
      ██║   ██████╔╝█████╗  
      ██║   ██╔═══╝ ██╔══╝  
      ██║   ██║     ███████╗
      ╚═╝   ╚═╝     ╚══════╝
      
    TREE-STRUCTURED PARZEN ESTIMATOR
═══════════════════════════════════════════════════════════════════════════════

DESCRIPCIÓN:
============
TPE es un algoritmo bayesiano que modela la distribución de parámetros buenos
y malos por separado, usando estimadores de densidad Parzen.

VENTAJAS:
=========
  ✓ EXPLORACIÓN AMPLIA del espacio de parámetros
  ✓ DIVERSIDAD de soluciones encontradas
  ✓ Ideal para ESPACIOS GRANDES y DISCONTINUOS
  ✓ Puede manejar PARÁMETROS CATEGÓRICOS
  ✓ RÁPIDO en convergencia inicial

FILOSOFÍA DEL SCORING TPE:
==========================
El scoring de TPE usa una arquitectura MÁS SIMPLE orientada a:
  - EXPLORACIÓN: No penaliza agresivamente
  - DIVERSIDAD: Score más uniforme para permitir exploración
  - VELOCIDAD: Menos cálculos complejos

DIFERENCIAS CON CMA:
====================
  - SIN DSR (permite exploración sin penalización por múltiples pruebas)
  - SIN SAM (no evalúa vecindario para mayor velocidad)
  - PENALIZACIONES MÁS SUAVES
  - IDEAL PARA FASE INICIAL DE OPTIMIZACIÓN

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import gc
import math
import os
import re
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import optuna
import polars as pl
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import TPESampler

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
# ███████╗ ██████╗ ██████╗ ██████╗ ██╗███╗   ██╗ ██████╗ 
# ██╔════╝██╔════╝██╔═══██╗██╔══██╗██║████╗  ██║██╔════╝ 
# ███████╗██║     ██║   ██║██████╔╝██║██╔██╗ ██║██║  ███╗
# ╚════██║██║     ██║   ██║██╔══██╗██║██║╚██╗██║██║   ██║
# ███████║╚██████╗╚██████╔╝██║  ██║██║██║ ╚████║╚██████╔╝
# ╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝ 
#                                                         
# SISTEMA DE SCORING TPE - SIMPLE Y EXPLORATORIA
# =============================================================================


# =============================================================================
# [SECCIÓN 1] CONFIGURACIÓN DEL SCORING TPE
# =============================================================================

@dataclass
class TPEScoringConfig:
    """
    ┌────────────────────────────────────────────────────────────────────────┐
    │              CONFIGURACIÓN DEL SCORING TPE v2.0                         │
    │                                                                         │
    │  ARQUITECTURA: Simple y Exploratoria                                   │
    │  FILOSOFÍA: Menos penalizaciones = Más diversidad                      │
    │  RANGO SALIDA: [1, 1000]                                               │
    └────────────────────────────────────────────────────────────────────────┘
    """
    
    # =========================================================================
    # 1.1 RANGO DE SALIDA DEL SCORE
    # =========================================================================
    SCORE_MIN: float = 1.0               # MÍNIMO ABSOLUTO
    SCORE_MAX: float = 1000.0            # MÁXIMO ABSOLUTO
    
    # =========================================================================
    # 1.2 COMPONENTES DEL SCORE TPE
    # =========================================================================
    # EL SCORE TPE ES MÁS SIMPLE QUE CMA
    # USA MÉTRICAS DIRECTAS SIN TRANSFORMACIONES COMPLEJAS
    
    # PESOS PARA CADA MÉTRICA (SUMAN 1.0)
    WEIGHT_SHARPE: float = 0.35          # SHARPE RATIO
    WEIGHT_SQN: float = 0.20             # SQN (SYSTEM QUALITY NUMBER)
    WEIGHT_ROI: float = 0.20             # RETORNO SOBRE INVERSIÓN
    WEIGHT_DRAWDOWN: float = 0.15        # PENALIZACIÓN POR DRAWDOWN
    WEIGHT_TRADES: float = 0.10          # ACTIVIDAD DE TRADING
    
    # =========================================================================
    # 1.3 ESCALADORES DE MÉTRICAS
    # =========================================================================
    # SHARPE: ESCALA A [0, 1] USANDO SIGMOIDE SUAVE
    SHARPE_CENTER: float = 1.0           # CENTRO DE NORMALIZACIÓN
    SHARPE_SCALE: float = 1.5            # FACTOR DE ESCALA
    
    # SQN: ESCALA LINEAL HASTA MÁXIMO
    SQN_TARGET: float = 4.0              # SQN OBJETIVO (> 4 = EXCELENTE)
    
    # ROI: ESCALA LOGARÍTMICA
    ROI_TARGET: float = 100.0            # ROI OBJETIVO (100% = DOBLAR)
    
    # DRAWDOWN: PENALIZACIÓN LINEAL
    DRAWDOWN_MAX_ACCEPTABLE: float = 30.0  # DD MÁXIMO SIN PENALIZACIÓN
    
    # TRADES: ESCALA LOGARÍTMICA
    MIN_TRADES_TARGET: int = 50          # NÚMERO OBJETIVO DE TRADES
    
    # =========================================================================
    # 1.4 UMBRALES MÍNIMOS (SOFT)
    # =========================================================================
    MIN_TRADES_FOR_VALID: int = 10       # TRADES MÍNIMOS PARA SCORE VÁLIDO
    MIN_TRADES_PER_DAY: float = 0.05     # ACTIVIDAD MÍNIMA
    MAX_DRAWDOWN_LIMIT: float = 80.0     # LÍMITE ABSOLUTO DE DRAWDOWN
    
    # =========================================================================
    # 1.5 PSR SIMPLE (OPCIONAL)
    # =========================================================================
    PSR_ENABLED: bool = True             # ACTIVAR PSR SIMPLE
    PSR_BENCHMARK: float = 0.0           # SR DE REFERENCIA
    PSR_MIN_TRADES: int = 30             # TRADES MÍNIMOS PARA PSR
    PSR_WEIGHT: float = 0.15             # PESO DEL PSR EN SCORE FINAL


# =============================================================================
# INSTANCIA DE CONFIGURACIÓN POR DEFECTO
# =============================================================================
TPE_SCORING_CONFIG = TPEScoringConfig()


# =============================================================================
# [SECCIÓN 2] CLASE SCORER TPE
# =============================================================================

class TPEScorer:
    """
    ┌────────────────────────────────────────────────────────────────────────┐
    │                     SCORER EXPLORATORIA TPE v2.0                        │
    │                                                                         │
    │  FILOSOFÍA: Score simple orientado a exploración                       │
    │  ARQUITECTURA: Promedio ponderado de métricas normalizadas             │
    │  RANGO: [1, 1000]                                                      │
    └────────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        study: Optional[optuna.Study] = None,
        config: Optional[TPEScoringConfig] = None,
    ):
        """
        INICIALIZA EL SCORER TPE.
        
        ARGS:
            study: OBJETO OPTUNA.STUDY (OPCIONAL PARA TPE)
            config: CONFIGURACIÓN PERSONALIZADA (USA DEFAULT SI NONE)
        """
        self.study = study
        self.config = config or TPE_SCORING_CONFIG
    
    # =========================================================================
    # [2.1] FUNCIONES AUXILIARES
    # =========================================================================
    
    @staticmethod
    def _safe_get(metrics: Mapping[str, Any], key: str, default: float = 0.0) -> float:
        """EXTRAE VALOR NUMÉRICO DE FORMA SEGURA."""
        try:
            val = metrics.get(key, default)
            if val is None:
                return default
            f_val = float(val)
            if math.isnan(f_val) or math.isinf(f_val):
                return default
            return f_val
        except Exception:
            return default
    
    @staticmethod
    def _sigmoid(x: float, center: float = 1.0, scale: float = 1.5) -> float:
        """FUNCIÓN SIGMOIDE SIMPLE."""
        try:
            exponent = -scale * (x - center)
            if exponent > 500:
                return 0.0
            elif exponent < -500:
                return 1.0
            return 1.0 / (1.0 + math.exp(exponent))
        except (OverflowError, ValueError):
            return 0.5
    
    # =========================================================================
    # [2.2] NORMALIZACIÓN DE MÉTRICAS
    # =========================================================================
    
    def _normalize_sharpe(self, sharpe: float) -> float:
        """
        NORMALIZA SHARPE RATIO A [0, 1] USANDO SIGMOIDE.
        
        MAPEO:
            SHARPE -2 → ~0.1
            SHARPE  0 → ~0.3
            SHARPE  1 → ~0.5 (CENTRO)
            SHARPE  2 → ~0.7
            SHARPE  4 → ~0.9
        """
        cfg = self.config
        normalized = self._sigmoid(sharpe, cfg.SHARPE_CENTER, cfg.SHARPE_SCALE)
        return float(np.clip(normalized, 0.01, 0.99))
    
    def _normalize_sqn(self, sqn: float) -> float:
        """
        NORMALIZA SQN A [0, 1].
        
        SQN ESCALAS (SEGÚN VAN THARP):
            < 1.6  → POBRE
            1.6-2  → PROMEDIO
            2-2.5  → BUENO
            2.5-3  → EXCELENTE
            > 3    → SÚPER
        """
        cfg = self.config
        normalized = sqn / cfg.SQN_TARGET
        return float(np.clip(normalized, 0.0, 1.0))
    
    def _normalize_roi(self, roi: float) -> float:
        """
        NORMALIZA ROI A [0, 1] USANDO FUNCIÓN LOGARÍTMICA.
        
        ESCALA:
            ROI < 0   → PENALIZACIÓN PROPORCIONAL
            ROI 0-50  → 0.0 - 0.5
            ROI 50-100→ 0.5 - 0.75
            ROI > 100 → 0.75 - 1.0 (ASINTÓTICO)
        """
        cfg = self.config
        
        if roi <= 0:
            # PENALIZACIÓN PARA ROI NEGATIVO
            normalized = max(0.0, 0.5 + (roi / 200.0))  # -200% → 0, 0% → 0.5
        else:
            # LOG SCALING PARA ROI POSITIVO
            log_roi = math.log1p(roi)  # log(1 + roi)
            log_target = math.log1p(cfg.ROI_TARGET)
            normalized = 0.5 + 0.5 * min(1.0, log_roi / log_target)
        
        return float(np.clip(normalized, 0.01, 0.99))
    
    def _normalize_drawdown(self, drawdown: float) -> float:
        """
        PENALIZACIÓN POR DRAWDOWN.
        
        ESCALA:
            DD 0-10%   → 1.0 (SIN PENALIZACIÓN)
            DD 10-30%  → 0.8 - 0.5
            DD 30-50%  → 0.5 - 0.2
            DD > 50%   → 0.2 - 0.0
        """
        cfg = self.config
        
        if drawdown <= cfg.DRAWDOWN_MAX_ACCEPTABLE:
            # PENALIZACIÓN SUAVE
            penalty = 1.0 - (drawdown / cfg.DRAWDOWN_MAX_ACCEPTABLE) * 0.5
        else:
            # PENALIZACIÓN FUERTE
            excess = drawdown - cfg.DRAWDOWN_MAX_ACCEPTABLE
            max_excess = cfg.MAX_DRAWDOWN_LIMIT - cfg.DRAWDOWN_MAX_ACCEPTABLE
            penalty = 0.5 * (1.0 - min(1.0, excess / max_excess))
        
        return float(np.clip(penalty, 0.01, 1.0))
    
    def _normalize_trades(self, n_trades: int, days: float) -> float:
        """
        PENALIZACIÓN POR POCA ACTIVIDAD.
        
        ESCALA:
            < 10 TRADES → PENALIZACIÓN SEVERA
            10-50       → PENALIZACIÓN MODERADA
            50+         → SIN PENALIZACIÓN
        """
        cfg = self.config
        
        if n_trades < cfg.MIN_TRADES_FOR_VALID:
            return 0.1
        
        # ESCALA LOGARÍTMICA
        log_trades = math.log1p(n_trades)
        log_target = math.log1p(cfg.MIN_TRADES_TARGET)
        normalized = min(1.0, log_trades / log_target)
        
        return float(np.clip(normalized, 0.1, 1.0))
    
    # =========================================================================
    # [2.3] PSR SIMPLE (SIN DSR)
    # =========================================================================
    
    def _calculate_simple_psr(self, returns: np.ndarray) -> float:
        """
        CALCULA UN PSR SIMPLIFICADO.
        
        NO USA DSR (DEFLATED SHARPE) PARA PERMITIR MÁS EXPLORACIÓN.
        """
        cfg = self.config
        
        if not cfg.PSR_ENABLED:
            return 1.0
        
        returns = np.asarray(returns, dtype=np.float64)
        returns = returns[np.isfinite(returns)]
        n = len(returns)
        
        if n < cfg.PSR_MIN_TRADES:
            return 0.5  # SIN PENALIZACIÓN SEVERA
        
        mean_r = float(np.mean(returns))
        std_r = float(np.std(returns, ddof=1))
        
        if std_r < 1e-10:
            return 0.5
        
        sr = mean_r / std_r
        
        # PSR SIMPLE: Z-SCORE
        std_err = 1.0 / math.sqrt(max(1, n - 1))
        z_score = (sr - cfg.PSR_BENCHMARK) / std_err
        
        # CDF NORMAL
        psr = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2)))
        
        return float(np.clip(psr, 0.1, 0.99))
    
    # =========================================================================
    # [2.4] FUNCIÓN MAESTRA: COMPUTE SCORE
    # =========================================================================
    
    def compute_score(
        self,
        trial: Optional[optuna.Trial],
        metrics: Mapping[str, Any],
        returns: Optional[np.ndarray] = None,
        equity_curve: Optional[np.ndarray] = None,
    ) -> float:
        """
        ┌────────────────────────────────────────────────────────────────────┐
        │              FUNCIÓN PRINCIPAL DE SCORING TPE                       │
        │                                                                     │
        │  Score = Σ (peso_i × métrica_normalizada_i)                        │
        │                                                                     │
        │  RANGO: [1, 1000]                                                  │
        └────────────────────────────────────────────────────────────────────┘
        """
        cfg = self.config
        
        # =================================================================
        # EXTRAER MÉTRICAS BASE
        # =================================================================
        sharpe = self._safe_get(metrics, "sharpe", 0.0)
        if sharpe == 0:
            sharpe = self._safe_get(metrics, "sharpe_ratio", 0.0)
        
        sqn = self._safe_get(metrics, "sqn", 0.0)
        
        roi = self._safe_get(metrics, "roi", 0.0)
        
        drawdown = self._safe_get(metrics, "drawdown", 50.0)
        if drawdown == 0:
            drawdown = self._safe_get(metrics, "max_drawdown", 50.0)
        
        n_trades = int(self._safe_get(metrics, "n_trades", 0))
        if n_trades == 0:
            n_trades = int(self._safe_get(metrics, "total_trades", 0))
        
        dias_totales = self._safe_get(metrics, "dias_totales", 365.0)
        
        # =================================================================
        # VERIFICACIÓN DE TRADES MÍNIMOS
        # =================================================================
        if n_trades < cfg.MIN_TRADES_FOR_VALID:
            if trial is not None:
                try:
                    trial.set_user_attr('tpe_score_reason', 'insufficient_trades')
                except Exception:
                    pass
            return cfg.SCORE_MIN
        
        # =================================================================
        # NORMALIZAR CADA MÉTRICA
        # =================================================================
        norm_sharpe = self._normalize_sharpe(sharpe)
        norm_sqn = self._normalize_sqn(sqn)
        norm_roi = self._normalize_roi(roi)
        norm_dd = self._normalize_drawdown(drawdown)
        norm_trades = self._normalize_trades(n_trades, dias_totales)
        
        # =================================================================
        # PSR SIMPLE (OPCIONAL)
        # =================================================================
        psr_factor = 1.0
        if cfg.PSR_ENABLED:
            if returns is None:
                raw_returns = metrics.get("returns", None)
                if raw_returns is None:
                    raw_returns = metrics.get("trade_returns", None)
                if raw_returns is not None:
                    try:
                        returns = np.asarray(raw_returns, dtype=np.float64)
                    except Exception:
                        returns = None
            
            if returns is not None and len(returns) >= cfg.PSR_MIN_TRADES:
                psr_val = self._calculate_simple_psr(returns)
                # AJUSTAR PESO
                psr_weight = cfg.PSR_WEIGHT
                psr_factor = (1.0 - psr_weight) + psr_weight * psr_val
            
            if trial is not None:
                try:
                    trial.set_user_attr('psr_simple', float(psr_val if 'psr_val' in dir() else 1.0))
                except Exception:
                    pass
        
        # =================================================================
        # CALCULAR SCORE COMPUESTO
        # =================================================================
        weighted_sum = (
            cfg.WEIGHT_SHARPE * norm_sharpe +
            cfg.WEIGHT_SQN * norm_sqn +
            cfg.WEIGHT_ROI * norm_roi +
            cfg.WEIGHT_DRAWDOWN * norm_dd +
            cfg.WEIGHT_TRADES * norm_trades
        )
        
        # APLICAR PSR FACTOR
        weighted_sum *= psr_factor
        
        # ESCALAR A RANGO [SCORE_MIN, SCORE_MAX]
        score_range = cfg.SCORE_MAX - cfg.SCORE_MIN
        final_score = cfg.SCORE_MIN + score_range * weighted_sum
        
        # =================================================================
        # GUARDAR ATRIBUTOS PARA AUDITORÍA
        # =================================================================
        if trial is not None:
            try:
                trial.set_user_attr('norm_sharpe', float(norm_sharpe))
                trial.set_user_attr('norm_sqn', float(norm_sqn))
                trial.set_user_attr('norm_roi', float(norm_roi))
                trial.set_user_attr('norm_dd', float(norm_dd))
                trial.set_user_attr('norm_trades', float(norm_trades))
                trial.set_user_attr('weighted_sum', float(weighted_sum))
                trial.set_user_attr('sr_nominal', float(sharpe))
            except Exception:
                pass
        
        # GARANTIZAR RANGO
        final_score = max(cfg.SCORE_MIN, min(cfg.SCORE_MAX, final_score))
        
        return float(final_score)


# =============================================================================
# ██████╗ ██████╗ ████████╗██╗███╗   ███╗██╗███████╗███████╗██████╗ 
# ██╔═══██╗██╔══██╗╚══██╔══╝██║████╗ ████║██║╚══███╔╝██╔════╝██╔══██╗
# ██║   ██║██████╔╝   ██║   ██║██╔████╔██║██║  ███╔╝ █████╗  ██████╔╝
# ██║   ██║██╔═══╝    ██║   ██║██║╚██╔╝██║██║ ███╔╝  ██╔══╝  ██╔══██╗
# ╚██████╔╝██║        ██║   ██║██║ ╚═╝ ██║██║███████╗███████╗██║  ██║
#  ╚═════╝ ╚═╝        ╚═╝   ╚═╝╚═╝     ╚═╝╚═╝╚══════╝╚══════╝╚═╝  ╚═╝
#
# [SECCIÓN 3] CLASE OPTIMIZADOR TPE
# =============================================================================


@dataclass
class TPEOptimizerConfig:
    """
    ┌────────────────────────────────────────────────────────────────────────┐
    │                  CONFIGURACIÓN DEL OPTIMIZADOR TPE                      │
    └────────────────────────────────────────────────────────────────────────┘
    """
    
    # =========================================================================
    # 3.1 CONFIGURACIÓN DE OPTUNA
    # =========================================================================
    SEED: Optional[int] = None           # SEMILLA ALEATORIA (NONE = VARIEDAD)
    N_JOBS: int = 1                       # NÚMERO DE WORKERS PARALELOS
    STORAGE: Optional[str] = None         # NONE = EJECUCIÓN EN RAM
    STUDY_NAME_PREFIX: str = "MODELOX"    # PREFIJO PARA NOMBRES DE ESTUDIO
    
    # =========================================================================
    # 3.2 CONFIGURACIÓN ESPECÍFICA TPE
    # =========================================================================
    N_STARTUP_TRIALS: int = 10            # TRIALS ALEATORIOS INICIALES
    N_EI_CANDIDATES: int = 24             # CANDIDATOS PARA EXPECTED IMPROVEMENT
    MULTIVARIATE: bool = True             # CONSIDERAR DEPENDENCIAS ENTRE PARAMS
    GROUP: bool = True                     # AGRUPAR PARÁMETROS RELACIONADOS
    CONSTANT_LIAR: bool = True             # MEJORAR PARALELISMO


# =============================================================================
# INSTANCIA DE CONFIGURACIÓN POR DEFECTO
# =============================================================================
TPE_OPTIMIZER_CONFIG = TPEOptimizerConfig()


class TPEOptimizer:
    """
    ┌────────────────────────────────────────────────────────────────────────┐
    │                       OPTIMIZADOR TPE                                   │
    │                                                                         │
    │  TREE-STRUCTURED PARZEN ESTIMATOR                                       │
    │                                                                         │
    │  CARACTERÍSTICAS:                                                       │
    │    ✓ EXPLORACIÓN AMPLIA DEL ESPACIO                                   │
    │    ✓ DIVERSIDAD DE SOLUCIONES                                         │
    │    ✓ IDEAL PARA ESPACIOS GRANDES Y DISCONTINUOS                       │
    │    ✓ MANEJA PARÁMETROS CATEGÓRICOS                                    │
    └────────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        config: BacktestConfig,
        n_trials: int,
        reporters: Sequence[Reporter] = (),
        optimizer_config: Optional[TPEOptimizerConfig] = None,
        scoring_config: Optional[TPEScoringConfig] = None,
        activo: Optional[str] = None,
    ):
        """
        INICIALIZA EL OPTIMIZADOR TPE.
        
        ARGS:
            config: CONFIGURACIÓN DE BACKTEST
            n_trials: NÚMERO DE TRIALS A EJECUTAR
            reporters: LISTA DE REPORTERS PARA RESULTADOS
            optimizer_config: CONFIGURACIÓN DEL OPTIMIZADOR
            scoring_config: CONFIGURACIÓN DEL SCORING
            activo: NOMBRE DEL ACTIVO (OPCIONAL)
        """
        self.config = config
        self.n_trials = n_trials
        self.reporters = list(reporters)
        self.optimizer_config = optimizer_config or TPE_OPTIMIZER_CONFIG
        self.scoring_config = scoring_config or TPE_SCORING_CONFIG
        self.activo = activo
        
        # ESTADO INTERNO
        self._last_study: Optional[optuna.Study] = None
        self._scorer: Optional[TPEScorer] = None
    
    # =========================================================================
    # [3.1] CREAR ESTUDIO OPTUNA
    # =========================================================================
    
    def _create_study(self, strategy_name: str) -> optuna.Study:
        """
        CREA UN ESTUDIO OPTUNA CON SAMPLER TPE.
        
        RETURNS:
            OBJETO OPTUNA.STUDY CONFIGURADO
        """
        cfg = self.optimizer_config
        
        # CONSTRUIR NOMBRE DEL ESTUDIO
        parts = [cfg.STUDY_NAME_PREFIX, str(strategy_name), "TPE"]
        if self.activo:
            parts.append(str(self.activo))
        study_name = self._slug("_".join(parts))
        
        # CREAR SAMPLER TPE
        sampler = TPESampler(
            seed=cfg.SEED,
            n_startup_trials=cfg.N_STARTUP_TRIALS,
            n_ei_candidates=cfg.N_EI_CANDIDATES,
            multivariate=cfg.MULTIVARIATE,
            group=cfg.GROUP,
            constant_liar=cfg.CONSTANT_LIAR,
        )
        
        # CREAR ESTUDIO
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=study_name,
            storage=cfg.STORAGE,
            load_if_exists=False,
        )
        
        # INICIALIZAR SCORER CON EL ESTUDIO
        self._scorer = TPEScorer(study=study, config=self.scoring_config)
        
        return study
    
    @staticmethod
    def _slug(s: str) -> str:
        """GENERA UN SLUG VÁLIDO PARA NOMBRES DE ESTUDIO."""
        s = s.strip().lower()
        s = re.sub(r'[^a-z0-9]+', '_', s)
        return s.strip('_')[:50]
    
    # =========================================================================
    # [3.2] PREPARAR PARÁMETROS
    # =========================================================================
    
    def _prepare_params(
        self,
        trial: optuna.Trial,
        strategy: Strategy,
        base_tf: str,
    ) -> Dict[str, Any]:
        """PREPARA PARÁMETROS PARA UN TRIAL."""
        params_puros = strategy.suggest_params(trial)
        params_rt = dict(params_puros)
        
        # INYECTAR VALORES DE CONFIGURACIÓN
        params_rt["__activo"] = self.activo
        params_rt["__saldo_inicial"] = float(self.config.saldo_inicial)
        params_rt["__saldo_operativo_max"] = float(self.config.saldo_operativo_max)

        
        params_rt["__comision_pct"] = float(self.config.comision_pct)
        params_rt["__comision_sides"] = int(self.config.comision_sides)
        params_rt["__saldo_usado"] = float(self.config.saldo_usado)
        params_rt["__apalancamiento_max"] = float(self.config.apalancamiento_max)
        params_rt["__strategy_exit_enabled"] = bool(getattr(strategy, "SALIDAS_PERSONALIZADAS", False))
        
        # RESOLVER CONFIGURACIÓN DE SALIDA
        exit_settings = resolve_exit_settings_for_trial(trial=trial, config=self.config)
        params_rt["__exit_type"] = exit_settings.exit_type
        params_rt["__exit_sl_pct"] = exit_settings.sl_pct
        params_rt["__exit_tp_pct"] = exit_settings.tp_pct
        params_rt["__exit_trail_act_pct"] = exit_settings.trail_act_pct
        params_rt["__exit_trail_dist_pct"] = exit_settings.trail_dist_pct
        
        # ALIASES PARA COMPATIBILIDAD
        params_rt["exit_type"] = exit_settings.exit_type
        params_rt["exit_sl_pct"] = exit_settings.sl_pct
        params_rt["exit_tp_pct"] = exit_settings.tp_pct
        params_rt["exit_trail_act_pct"] = exit_settings.trail_act_pct
        params_rt["exit_trail_dist_pct"] = exit_settings.trail_dist_pct
        
        # TIMEFRAMES
        entry_tf = normalize_timeframe_to_suffix(getattr(strategy, "timeframe_entry", None) or base_tf)
        exit_tf = normalize_timeframe_to_suffix(getattr(strategy, "timeframe_exit", None) or base_tf)
        
        params_rt["__timeframe_base"] = base_tf
        params_rt["__timeframe_entry"] = entry_tf
        params_rt["__timeframe_exit"] = exit_tf
        
        return params_rt
    
    # =========================================================================
    # [3.3] FUNCIÓN OBJETIVO
    # =========================================================================
    
    def _create_objective(
        self,
        df_base: pl.DataFrame,
        df_map: Dict[str, pl.DataFrame],
        strategy: Strategy,
        base_tf: str,
    ) -> Callable[[optuna.Trial], float]:
        """CREA LA FUNCIÓN OBJETIVO PARA TPE."""
        
        # IMPORTAR COMPONENTES NECESARIOS
        from modelox.core.runner import SignalGenerator, BacktestEngine, periodic_cleanup
        
        def objective(trial: optuna.Trial) -> float:
            t0_total = time.perf_counter()
            
            # LIMPIEZA PERIÓDICA
            periodic_cleanup(trial.number)
            
            params_rt = self._prepare_params(trial, strategy, base_tf)
            entry_tf = params_rt["__timeframe_entry"]
            df_entry = df_map.get(entry_tf, df_base)
            
            # GENERAR SEÑALES
            signals_df = SignalGenerator.generate_signals(df_entry, strategy, params_rt, df_map)
            
            # EJECUTAR BACKTEST
            trades_df, equity_curve, metrics = BacktestEngine.run_backtest(
                df_entry, signals_df, self.config, params_rt, strategy,
            )
            
            if trades_df.is_empty():
                return 0.0
            
            trial.set_user_attr("metricas", metrics)
            
            # CALCULAR SCORE CON SCORER TPE
            score = self._scorer.compute_score(
                trial=trial,
                metrics=metrics,
                equity_curve=np.array(equity_curve) if equity_curve else None,
            )
            
            # CREAR ARTIFACTS
            artifacts = TrialArtifacts(
                strategy_name=strategy.name,
                trial_number=trial.number,
                params=params_rt,
                params_reporting=params_rt,
                score=score,
                metrics=metrics,
                df_signals=None,
                trades=trades_df.to_pandas(),
                equity_curve=equity_curve,
                indicators_used=params_rt.get("__indicators_used", []),
            )
            
            for reporter in self.reporters:
                reporter.on_trial_end(artifacts)
            
            return score
        
        return objective
    
    # =========================================================================
    # [3.4] OPTIMIZAR
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
        │                    EJECUTAR OPTIMIZACIÓN TPE                        │
        └────────────────────────────────────────────────────────────────────┘
        
        ARGS:
            df: DATAFRAME CON DATOS OHLCV
            strategy: ESTRATEGIA A OPTIMIZAR
            df_by_timeframe: DICT CON DATAFRAMES POR TIMEFRAME
            base_timeframe: TIMEFRAME BASE
        
        RETURNS:
            OBJETO OPTUNA.STUDY CON RESULTADOS
        """
        base_tf = base_timeframe or "1m"
        df_map = df_by_timeframe or {base_tf: df}
        df_base = df_map.get(base_tf, df)
        
        # CREAR ESTUDIO
        study = self._create_study(strategy.name)
        
        # CREAR OBJETIVO
        objective = self._create_objective(df_base, df_map, strategy, base_tf)
        
        # EJECUTAR OPTIMIZACIÓN
        study.optimize(
            objective,
            n_trials=int(self.n_trials),
            n_jobs=int(self.optimizer_config.N_JOBS),
            gc_after_trial=True,
            catch=(Exception,),
        )
        
        self._last_study = study
        return study
    
    # =========================================================================
    # [3.5] PROPIEDADES
    # =========================================================================
    
    @property
    def last_study(self) -> Optional[optuna.Study]:
        """RETORNA EL ÚLTIMO ESTUDIO EJECUTADO."""
        return self._last_study
    
    @property
    def scorer(self) -> Optional[TPEScorer]:
        """RETORNA EL SCORER UTILIZADO."""
        return self._scorer


# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def _slug(s: str) -> str:
    """Genera un slug válido para nombres de estudio."""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "study"


def create_tpe_study(
    strategy_name: str,
    activo: Optional[str] = None,
    seed: Optional[int] = None,
    study_name_prefix: str = "MODELOX",
    n_startup_trials: int = 10,
    multivariate: bool = True,
    group: bool = True,
    storage: Optional[str] = None,
) -> optuna.Study:
    """
    Crea un estudio Optuna con sampler TPE.
    
    TPE (Tree-structured Parzen Estimator):
    - Algoritmo clásico de Optuna
    - Exploración amplia del espacio de parámetros
    - Bueno para espacios mixtos con variables categóricas
    - Ideal para fase inicial de exploración
    
    Args:
        strategy_name: Nombre de la estrategia
        activo: Nombre del activo (opcional)
        seed: Semilla aleatoria
        study_name_prefix: Prefijo para el nombre del estudio
        n_startup_trials: Trials aleatorios iniciales
        multivariate: Considerar dependencias entre parámetros
        group: Agrupar parámetros relacionados
        storage: URI de almacenamiento (None = RAM)
    
    Returns:
        optuna.Study configurado con TPE
    """
    # Construir nombre del estudio
    parts = [study_name_prefix, str(strategy_name), "tpe"]
    if activo:
        parts.append(str(activo))
    study_name = _slug("_".join(parts))
    
    # Crear sampler TPE
    sampler = TPESampler(
        seed=seed,
        multivariate=multivariate,
        group=group,
    )
    
    # Crear estudio
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        load_if_exists=False,
    )
    
    return study


def score_tpe(
    metrics: Mapping[str, Any],
    trial: Optional[optuna.Trial] = None,
    equity_curve: Optional[List[float]] = None,
) -> float:
    """
    FUNCIÓN DE SCORING TPE STANDALONE.
    
    USO:
        score = score_tpe(metrics)
    """
    scorer = TPEScorer()
    return scorer.compute_score(
        trial=trial,
        metrics=metrics,
        equity_curve=np.array(equity_curve) if equity_curve else None,
    )


# =============================================================================
# EXPORTACIONES
# =============================================================================

__all__ = [
    "TPEOptimizer",
    "TPEOptimizerConfig",
    "TPEScorer",
    "TPEScoringConfig",
    "TPE_SCORING_CONFIG",
    "TPE_OPTIMIZER_CONFIG",
    "create_tpe_study",
    "score_tpe",
]
