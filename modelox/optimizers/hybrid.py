"""modelox/optimizers/hybrid.py

═══════════════════════════════════════════════════════════════════════════════
    OPTIMIZADOR HÍBRIDO (QMC → TPE)
═══════════════════════════════════════════════════════════════════════════════

DESCRIPCIÓN:
============
Este optimizador divide los trials al 50%. La primera mitad utiliza QMCSampler
para asegurar una cobertura equidistante del espacio de búsqueda. La segunda
mitad utiliza TPESampler que aprende de los trials anteriores de QMC para
concentrarse en las zonas más prometedoras, aprovechando lo que ya encontró.

VENTAJAS:
=========
  ✓ EXPLORACIÓN PERFECTA inicial garantizada por QMC (Sobol)
  ✓ REFINAMIENTO RÁPIDO gracias al árbol bayesiano de TPE
  ✓ SIN OVERFITTING en la fase inicial, ya que QMC es determinista y no se sesga
  ✓ LA MEJOR COMBINACIÓN para espacios grandes sin caer en mínimos locales.

FILOSOFÍA DEL SCORING:
======================
  - 40% Sharpe Ratio (Calidad)
  - 40% Expectativa (Eficiencia / Esperanza por Trade)
  - 20% ROI (Rentabilidad Bruta)

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
from optuna.samplers import TPESampler, QMCSampler

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
# [SECCIÓN 1] CONFIGURACIÓN DEL SCORING HYBRID
# =============================================================================

@dataclass
class HybridScoringConfig:
    """
    Configuración del sistema de scoring híbrido.
    Basado en los pesos solicitados: 40% Sharpe, 40% Expectativa, 20% ROI.
    """
    
    # RANGO DE SALIDA DEL SCORE
    SCORE_MIN: float = 1.0               
    SCORE_MAX: float = 1000.0            
    
    # PESOS PARA CADA MÉTRICA (DEBEN SUMAR 1.0)
    WEIGHT_SHARPE: float = 0.40          
    WEIGHT_EXPECTATIVA: float = 0.40             
    WEIGHT_ROI: float = 0.20             
    
    # ESCALADORES (para normalizar a [0, 1])
    SHARPE_CENTER: float = 1.0           
    SHARPE_SCALE: float = 1.5            
    EXPECTATIVA_TARGET: float = 10.0  # Usaremos una escala lineal (o sigmoide suave)
    ROI_TARGET: float = 100.0         # ROI objetivo (100% = Doblar)
    
    # UMBRALES MÍNIMOS (SOFT)
    MIN_TRADES_FOR_VALID: int = 10       
    MIN_TRADES_PER_DAY: float = 0.15     


# Instancia por defecto
HYBRID_SCORING_CONFIG = HybridScoringConfig()


# =============================================================================
# [SECCIÓN 2] CLASE SCORER HYBRID
# =============================================================================

class HybridScorer:
    """
    Scorer para Optimizador Híbrido QMC/TPE.
    """
    
    def __init__(
        self,
        study: Optional[optuna.Study] = None,
        config: Optional[HybridScoringConfig] = None,
    ):
        self.study = study
        self.config = config or HYBRID_SCORING_CONFIG
    
    @staticmethod
    def _safe_get(metrics: Mapping[str, Any], key: str, default: float = 0.0) -> float:
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
        try:
            exponent = -scale * (x - center)
            if exponent > 500:
                return 0.0
            elif exponent < -500:
                return 1.0
            return 1.0 / (1.0 + math.exp(exponent))
        except (OverflowError, ValueError):
            return 0.5
    
    def _normalize_sharpe(self, sharpe: float) -> float:
        cfg = self.config
        normalized = self._sigmoid(sharpe, cfg.SHARPE_CENTER, cfg.SHARPE_SCALE)
        return float(np.clip(normalized, 0.01, 0.99))
    
    def _normalize_expectativa(self, expectativa: float) -> float:
        """Normalizar la expectativa. Si es negativa, castigamos. Si es positiva, premiamos hacia 1.0"""
        if expectativa <= 0:
            return max(0.01, 0.5 + (expectativa / 50.0))  # penalización suave
        else:
            cfg = self.config
            normalized = 0.5 + 0.5 * min(1.0, expectativa / cfg.EXPECTATIVA_TARGET)
            return float(np.clip(normalized, 0.5, 0.99))
            
    def _normalize_roi(self, roi: float) -> float:
        cfg = self.config
        if roi <= 0:
            normalized = max(0.0, 0.5 + (roi / 200.0))
        else:
            log_roi = math.log1p(roi)
            log_target = math.log1p(cfg.ROI_TARGET)
            normalized = 0.5 + 0.5 * min(1.0, log_roi / log_target)
        return float(np.clip(normalized, 0.01, 0.99))
    
    def compute_score(
        self,
        trial: Optional[optuna.Trial],
        metrics: Mapping[str, Any],
        returns: Optional[np.ndarray] = None,
        equity_curve: Optional[np.ndarray] = None,
    ) -> float:
        cfg = self.config
        
        # Extracción de métricas
        sharpe = self._safe_get(metrics, "sharpe", 0.0)
        if sharpe == 0:
            sharpe = self._safe_get(metrics, "sharpe_ratio", 0.0)
            
        expectativa = self._safe_get(metrics, "expectativa", 0.0)
        roi = self._safe_get(metrics, "roi", 0.0)
        
        n_trades = int(self._safe_get(metrics, "n_trades", 0))
        if n_trades == 0:
            n_trades = int(self._safe_get(metrics, "total_trades", 0))
        
        # Filtros base
        if n_trades < cfg.MIN_TRADES_FOR_VALID:
            if trial is not None:
                try: trial.set_user_attr('hybrid_score_reason', 'insufficient_trades')
                except Exception: pass
            return cfg.SCORE_MIN
        
        # Normalizaciones
        norm_sharpe = self._normalize_sharpe(sharpe)
        norm_exp = self._normalize_expectativa(expectativa)
        norm_roi = self._normalize_roi(roi)
        
        # Score ponderado
        weighted_sum = (
            cfg.WEIGHT_SHARPE * norm_sharpe +
            cfg.WEIGHT_EXPECTATIVA * norm_exp +
            cfg.WEIGHT_ROI * norm_roi
        )
        
        score_range = cfg.SCORE_MAX - cfg.SCORE_MIN
        final_score = cfg.SCORE_MIN + score_range * weighted_sum
        
        final_score = float(max(cfg.SCORE_MIN, min(cfg.SCORE_MAX, final_score)))
        
        if trial is not None:
            try:
                trial.set_user_attr('norm_sharpe', float(norm_sharpe))
                trial.set_user_attr('norm_expectativa', float(norm_exp))
                trial.set_user_attr('norm_roi', float(norm_roi))
                trial.set_user_attr('weighted_sum', float(weighted_sum))
            except Exception:
                pass
                
        return final_score


# =============================================================================
# [SECCIÓN 3] CLASE OPTIMIZADOR HYBRID
# =============================================================================

@dataclass
class HybridOptimizerConfig:
    """Configuración del optimizador Híbrido."""
    SEED: Optional[int] = None           
    N_JOBS: int = 1                       
    STORAGE: Optional[str] = None         # Base de datos. Si es None, creará en memoria
    STUDY_NAME_PREFIX: str = "MODELOX"    
    SCRAMBLE: bool = True

HYBRID_OPTIMIZER_CONFIG = HybridOptimizerConfig()

class HybridOptimizer:
    """
    Optimizador Híbrido
    QMC primero (50% de trials) -> Puntos equidistantes puros
    TPE segundo (50% de trials) -> Aprende del score de QMC en la misma base de datos Optuna
    """
    
    def __init__(
        self,
        config: BacktestConfig,
        n_trials: int,
        reporters: Sequence[Reporter] = (),
        optimizer_config: Optional[HybridOptimizerConfig] = None,
        scoring_config: Optional[HybridScoringConfig] = None,
        activo: Optional[str] = None,
    ):
        self.config = config
        self.n_trials = n_trials
        self.reporters = list(reporters)
        self.optimizer_config = optimizer_config or HYBRID_OPTIMIZER_CONFIG
        self.scoring_config = scoring_config or HYBRID_SCORING_CONFIG
        self.activo = activo
        
        self._last_study: Optional[optuna.Study] = None
        self._scorer: Optional[HybridScorer] = None
    
    @staticmethod
    def _slug(s: str) -> str:
        s = s.strip().lower()
        s = re.sub(r'[^a-z0-9]+', '_', s)
        return s.strip('_')[:50]
    
    def _prepare_params(
        self,
        trial: optuna.Trial,
        strategy: Strategy,
        base_tf: str,
    ) -> Dict[str, Any]:
        params_puros = strategy.suggest_params(trial)
        params_rt = dict(params_puros)
        
        params_rt["__activo"] = self.activo
        params_rt["__saldo_inicial"] = float(self.config.saldo_inicial)
        params_rt["__saldo_operativo_max"] = float(self.config.saldo_operativo_max)
        params_rt["__comision_pct"] = float(self.config.comision_pct)
        params_rt["__comision_sides"] = int(self.config.comision_sides)
        params_rt["__saldo_usado"] = float(self.config.saldo_usado)
        params_rt["__apalancamiento_max"] = float(self.config.apalancamiento_max)
        params_rt["__strategy_exit_enabled"] = bool(getattr(strategy, "SALIDAS_PERSONALIZADAS", False))
        
        exit_settings = resolve_exit_settings_for_trial(trial=trial, config=self.config)
        params_rt["__exit_type"] = exit_settings.exit_type
        params_rt["__exit_sl_pct"] = exit_settings.sl_pct
        params_rt["__exit_tp_pct"] = exit_settings.tp_pct
        params_rt["__exit_trail_act_pct"] = exit_settings.trail_act_pct
        params_rt["__exit_trail_dist_pct"] = exit_settings.trail_dist_pct
        
        params_rt["exit_type"] = exit_settings.exit_type
        params_rt["exit_sl_pct"] = exit_settings.sl_pct
        params_rt["exit_tp_pct"] = exit_settings.tp_pct
        params_rt["exit_trail_act_pct"] = exit_settings.trail_act_pct
        params_rt["exit_trail_dist_pct"] = exit_settings.trail_dist_pct
        
        entry_tf = normalize_timeframe_to_suffix(getattr(strategy, "timeframe_entry", None) or base_tf)
        exit_tf = normalize_timeframe_to_suffix(getattr(strategy, "timeframe_exit", None) or base_tf)
        
        params_rt["__timeframe_base"] = base_tf
        params_rt["__timeframe_entry"] = entry_tf
        params_rt["__timeframe_exit"] = exit_tf
        
        return params_rt
    
    def _create_objective(
        self,
        df_base: pl.DataFrame,
        df_map: Dict[str, pl.DataFrame],
        strategy: Strategy,
        base_tf: str,
    ) -> Callable[[optuna.Trial], float]:
        
        from modelox.core.runner import SignalGenerator, BacktestEngine, periodic_cleanup
        
        def objective(trial: optuna.Trial) -> float:
            t0_total = time.perf_counter()
            periodic_cleanup(trial.number)
            
            params_rt = self._prepare_params(trial, strategy, base_tf)
            entry_tf = params_rt["__timeframe_entry"]
            df_entry = df_map.get(entry_tf, df_base)
            
            signals_df = SignalGenerator.generate_signals(df_entry, strategy, params_rt, df_map)
            trades_df, equity_curve, metrics = BacktestEngine.run_backtest(
                df_entry, signals_df, self.config, params_rt, strategy,
            )
            
            if trades_df.is_empty():
                return 0.0
            
            trial.set_user_attr("metricas", metrics)
            
            score = self._scorer.compute_score(
                trial=trial,
                metrics=metrics,
                equity_curve=np.array(equity_curve) if equity_curve else None,
            )
            
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
    
    def optimize(
        self,
        *,
        df: pl.DataFrame,
        strategy: Strategy,
        df_by_timeframe: Optional[Dict[str, pl.DataFrame]] = None,
        base_timeframe: Optional[str] = None,
    ) -> optuna.Study:
        
        base_tf = base_timeframe or "1m"
        df_map = df_by_timeframe or {base_tf: df}
        df_base = df_map.get(base_tf, df)
        cfg = self.optimizer_config
        
        # Asegurar que ambos compartan la misma base de datos (archivo SQLite)
        if cfg.STORAGE is None:
            db_path = os.path.join(os.getcwd(), "optuna_hybrid.db")
            storage = f"sqlite:///{db_path}"
        else:
            storage = cfg.STORAGE
            
        parts = [cfg.STUDY_NAME_PREFIX, str(strategy.name), "HYBRID"]
        if self.activo:
            parts.append(str(self.activo))
        study_name = self._slug("_".join(parts))
        
        self._scorer = HybridScorer(config=self.scoring_config)
        objective = self._create_objective(df_base, df_map, strategy, base_tf)
        
        # Fase 1: QMC (Exactamente 50%)
        n_qmc = int(math.ceil(self.n_trials / 2))
        n_tpe = self.n_trials - n_qmc
        
        if n_qmc > 0:
            sampler_qmc = QMCSampler(
                seed=cfg.SEED, 
                scramble=getattr(cfg, "SCRAMBLE", True),
                warn_independent_sampling=False
            )
            study_qmc = optuna.create_study(
                direction="maximize",
                sampler=sampler_qmc,
                study_name=study_name,
                storage=storage,
                load_if_exists=True,
            )
            self._scorer.study = study_qmc
            
            study_qmc.optimize(
                objective,
                n_trials=n_qmc,
                n_jobs=int(cfg.N_JOBS),
                gc_after_trial=True,
                catch=(Exception,),
            )
            
            self._last_study = study_qmc
            
        # Fase 2: TPE (Aprovecha los trials anteriores guardados en storage)
        if n_tpe > 0:
            sampler_tpe = TPESampler(seed=cfg.SEED, n_startup_trials=0, multivariate=True)
            study_tpe = optuna.create_study(
                direction="maximize",
                sampler=sampler_tpe,
                study_name=study_name,
                storage=storage,
                load_if_exists=True, # Aquí engancha los scores previos de QMC
            )
            self._scorer.study = study_tpe
            
            study_tpe.optimize(
                objective,
                n_trials=n_tpe,
                n_jobs=int(cfg.N_JOBS),
                gc_after_trial=True,
                catch=(Exception,),
            )
            
            self._last_study = study_tpe
            return study_tpe
            
        return self._last_study


def create_hybrid_study(
    strategy_name: str,
    activo: Optional[str] = None,
    seed: Optional[int] = None,
    study_name_prefix: str = "MODELOX",
    storage: Optional[str] = None,
    phase: str = "QMC" # Opcional si se quiere forzar un sampler
) -> optuna.Study:
    """Función de factoría básica por si otros la importan."""
    parts = [study_name_prefix, str(strategy_name), "hybrid"]
    if activo:
        parts.append(str(activo))
    study_name = HybridOptimizer._slug("_".join(parts))
    
    if phase == "QMC":
        sampler = QMCSampler(seed=seed, scramble=True, warn_independent_sampling=False)
    else:
        sampler = TPESampler(seed=seed, n_startup_trials=0, multivariate=True)
    
    # Usar archivo SQLite por defecto si no se proporciona storage
    if storage is None:
        db_path = os.path.join(os.getcwd(), "optuna_hybrid.db")
        storage = f"sqlite:///{db_path}"
        
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    return study

def score_hybrid(
    metrics: Mapping[str, Any],
    trial: Optional[optuna.Trial] = None,
    equity_curve: Optional[List[float]] = None,
) -> float:
    scorer = HybridScorer()
    return scorer.compute_score(
        trial=trial,
        metrics=metrics,
        equity_curve=np.array(equity_curve) if equity_curve else None,
    )

__all__ = [
    "HybridOptimizer",
    "HybridOptimizerConfig",
    "HybridScorer",
    "HybridScoringConfig",
    "HYBRID_SCORING_CONFIG",
    "HYBRID_OPTIMIZER_CONFIG",
    "create_hybrid_study",
    "score_hybrid",
]
