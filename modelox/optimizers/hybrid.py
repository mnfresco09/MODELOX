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
  Implementa un motor de gradiente continuo (Gradient Mirroring) vectorizado:
  - Esperanza (Retorno por trade)
  - Riesgo (Racha perdedora amortiguada)
  - Estabilidad (R-Cuadrado de la curva trade a trade)
  - Frecuencia (Penalización progresiva por operaciones diarias)
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import gc
import math
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
from .storage import resolve_storage_for_strategy

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
    Configuración del sistema de scoring híbrido continuo.
    
    Fórmula:
        Score = (Esperanza / (Max_Racha_Perdedora + 1)) × R² × P_lineal(x)
    
    Donde P_lineal(x) con x = trades/día:
        Si x >= TARGET_TRADES_PER_DAY → 1.0
        Si x <  TARGET_TRADES_PER_DAY → x / TARGET_TRADES_PER_DAY
    """
    # OBJETIVOS DEL MOTOR MATEMÁTICO
    TARGET_TRADES_PER_DAY: float = 0.3    # Rampa de penalización progresiva
    MIN_TRADES_FOR_RELIABLE: int = 10     # Debajo de esto, esperanza se amortigua linealmente

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
    
    def compute_score(
        self,
        trial: Optional[optuna.Trial],
        metrics: Mapping[str, Any],
        returns: Optional[np.ndarray] = None,
        equity_curve: Optional[np.ndarray] = None,
        trades_pnl: Optional[np.ndarray] = None,
        total_days: Optional[float] = None,
    ) -> float:
        """
        SCORING HÍBRIDO — Fórmula única:
        
            Score = (Esperanza / (Max_Racha_Perdedora + 1)) × R² × P_lineal(x)
        
        Donde:
            Esperanza = Beneficio Neto Total / Número Total de Trades
                        (amortiguado si n_trades < MIN_TRADES_FOR_RELIABLE)
            R²        = Coef. determinación de capital acumulado vs nº trade
            P_lineal  = min(1, trades_per_day / 0.5)
        """
        # ── 1. Extracción y validación de datos ────────────────────────
        if trades_pnl is None:
            raw = metrics.get("_trades_pnl_array", None)
            if raw is not None:
                trades_pnl = np.asarray(raw, dtype=np.float64)

        if trades_pnl is None or trades_pnl.size == 0:
            return 0.0

        if total_days is None or total_days <= 0.0:
            td = self._safe_get(metrics, "_total_days", 0.0)
            if td <= 0.0:
                tpd = self._safe_get(metrics, "trades_por_dia", 0.0)
                n_t = int(self._safe_get(metrics, "n_trades", 0))
                if n_t == 0:
                    n_t = int(self._safe_get(metrics, "total_trades", 0))
                td = float(n_t) / tpd if tpd > 0 else 1.0
            total_days = td

        n_trades = len(trades_pnl)
        if n_trades == 0 or total_days <= 0.0:
            return 0.0

        # ── 2. Cálculos Matemáticos Core ───────────────────────────────

        # A. Esperanza = Beneficio Neto Total / N trades
        #    Con amortiguación para pocos trades: si n_trades < MIN,
        #    escalamos linealmente para evitar que la esperanza se dispare.
        esperanza_raw = float(np.sum(trades_pnl)) / n_trades
        min_trades = getattr(self.config, 'MIN_TRADES_FOR_RELIABLE', 10)
        if n_trades < min_trades and min_trades > 0:
            # Rampa lineal: 1 trade → factor 0.1, 10 trades → factor 1.0
            trade_factor = n_trades / min_trades
            esperanza = esperanza_raw * trade_factor
        else:
            esperanza = esperanza_raw

        # B. Racha Perdedora máxima (Vectorizada)
        es_negativo = (trades_pnl < 0).astype(int)
        padded = np.pad(es_negativo, (1, 1), 'constant', constant_values=0)
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        max_racha = float(np.max(ends - starts)) if len(starts) > 0 else 0.0

        # C. R² (Estabilidad de la curva de capital)
        #    Eje X = Número de trade (1, 2, 3, ...)
        #    Eje Y = Capital acumulado (cumsum de PnL por trade)
        #    R² mide qué tan bien los puntos siguen una línea recta
        if n_trades > 1:
            x = np.arange(1, n_trades + 1)
            y = np.cumsum(trades_pnl)
            if np.std(y) == 0:
                r_squared = 0.0
            else:
                correlacion = np.corrcoef(x, y)[0, 1]
                r_squared = float(correlacion**2) if not np.isnan(correlacion) else 0.0
        else:
            r_squared = 0.0

        # D. Penalización Lineal por frecuencia — P_lineal(x)
        #    x = trades_per_day
        #    Si x >= 0.5 → 1.0 (sin penalización)
        #    Si x <  0.5 → x / 0.5 (rampa suave)
        trades_per_day = n_trades / total_days
        target_tpd = getattr(self.config, 'TARGET_TRADES_PER_DAY', 0.5)
        p_lineal = min(1.0, trades_per_day / target_tpd) if target_tpd > 0 else 1.0

        # ── 3. Fórmula Final ───────────────────────────────────────────
        #    Score = (Esperanza / (Max_Racha_Perdedora + 1)) × R² × P_lineal × 100
        final_score = (esperanza / (max_racha + 1.0)) * r_squared * p_lineal * 100.0

        # ── SEGURO: proteger contra NaN / Inf por edge cases numéricos ──
        if not math.isfinite(final_score):
            final_score = 0.0

        # ── 4. Auditoría en Optuna ─────────────────────────────────────
        if trial is not None:
            try:
                trial.set_user_attr('final_score', float(final_score))
                trial.set_user_attr('esperanza', float(esperanza))
                trial.set_user_attr('esperanza_raw', float(esperanza_raw))
                trial.set_user_attr('max_racha', float(max_racha))
                trial.set_user_attr('r_squared', float(r_squared))
                trial.set_user_attr('p_lineal', float(p_lineal))
                trial.set_user_attr('n_trades', int(n_trades))
            except Exception:
                pass

        return float(final_score)


# =============================================================================
# [SECCIÓN 3] CLASE OPTIMIZADOR HYBRID
# =============================================================================

@dataclass
class HybridOptimizerConfig:
    """Configuración del optimizador Híbrido."""
    SEED: Optional[int] = None           
    N_JOBS: int = 1                       
    CREATE_DATABASE: bool = True          # True = crear SQLite por estrategia (IDx.db)
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
        # FORZAR salidas SIEMPRE en 1m para máxima precisión (SL/TP/Trailing)
        exit_tf = "1m"
        
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
            
            # Extraer pnl_neto array y total_days
            if isinstance(trades_df, pl.DataFrame):
                _pnl_arr = trades_df["pnl_neto"].to_numpy().astype(np.float64)
            else:
                _pnl_arr = trades_df["pnl_neto"].to_numpy(dtype=np.float64)
            
            # total_days desde timestamps del DataFrame de entrada
            _total_days = 0.0
            if "timestamp" in df_entry.columns and len(df_entry) > 0:
                try:
                    _ts0 = df_entry["timestamp"][0]
                    _ts1 = df_entry["timestamp"][-1]
                    _delta = pl.DataFrame({"s": [_ts0], "e": [_ts1]}).select(
                        ((pl.col("e") - pl.col("s")).dt.total_seconds() / 86400.0).alias("d")
                    )
                    _total_days = max(1.0, float(_delta["d"][0]))
                except Exception:
                    _total_days = 1.0
            
            score = self._scorer.compute_score(
                trial=trial,
                metrics=metrics,
                equity_curve=np.array(equity_curve) if equity_curve else None,
                trades_pnl=_pnl_arr,
                total_days=_total_days,
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
        
        storage = resolve_storage_for_strategy(
            create_database=bool(cfg.CREATE_DATABASE),
            strategy_id=getattr(strategy, "combinacion_id", None),
        )
            
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
    strategy_id: Optional[int] = None,
    activo: Optional[str] = None,
    seed: Optional[int] = None,
    study_name_prefix: str = "MODELOX",
    create_database: bool = True,
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
    
    if storage is None:
        storage = resolve_storage_for_strategy(
            create_database=create_database,
            strategy_id=strategy_id,
        )
        
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
    trades_pnl: Optional[np.ndarray] = None,
    total_days: Optional[float] = None,
) -> float:
    scorer = HybridScorer()
    return scorer.compute_score(
        trial=trial,
        metrics=metrics,
        equity_curve=np.array(equity_curve) if equity_curve else None,
        trades_pnl=trades_pnl,
        total_days=total_days,
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