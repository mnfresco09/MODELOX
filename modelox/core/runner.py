"""
================================================================================
MODELOX/CORE/RUNNER.PY — RUNNER PRINCIPAL DE OPTIMIZACIÓN
================================================================================

PROPÓSITO:
    Orquesta la optimización bayesiana con Optuna, soportando múltiples
    samplers (CMA-ES, TPE, GT, ML, QMC).

CONTENIDO:
     1. CACHÉ DE SEÑALES          — Reutilización entre trials vecinos
     2. CONFIGURACIÓN             — OptunaConfig
     4. HELPERS                   — create_study_for_strategy
     5. PIPELINE                  — DataLoader, SignalGenerator, BacktestEngine
     6. RUNNER PRINCIPAL          — OptimizationRunner (optimize_strategies)

MODO DE OPERACIÓN:
    Cada trial usa datos ORIGINALES.

SAMPLERS DISPONIBLES:
    - CMA: CMA-ES (recomendado para scoring institucional)
    - TPE: Tree-structured Parzen Estimator (clásico Optuna)
    - GT:  GT-Score anti-overfitting
    - ML:  ML Forest scorer
    - QMC: Quasi-Monte Carlo

DEPENDENCIAS:
    → engine.py, metrics.py, types.py, data.py, exits.py
    → modelox.optimizers (scoring functions)
    ← ejecutar.py

================================================================================
"""

from __future__ import annotations

import gc
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import optuna
import polars as pl
from optuna.exceptions import ExperimentalWarning

from .engine import BacktestParams, calculate_performance_vectorized_numba
from .metrics import resumen_metricas
from .types import (
    BacktestConfig,
    Reporter,
    Strategy,
    TrialArtifacts,
    normalize_timeframe_to_suffix,
)
from .data import load_data, prepare_multitimeframe_data
from .exits import resolve_exit_settings_for_trial

# Funciones desde módulo optimizers
from modelox.optimizers import (
    CMAScorer, TPEScorer, GTScorer, MLForestScorer, QMCScorer, HybridScorer,
    score_cma, score_tpe, score_gt, score_ml, score_qmc, score_hybrid,
    create_study,  # Factory para crear estudios
)

# Silenciar warnings experimentales de Optuna
warnings.filterwarnings("ignore", category=ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Debug timings
_TIMINGS_VERBOSE = os.environ.get("MODELOX_TIMINGS_VERBOSE", "0") in {"1", "true", "True", "YES", "yes"}
_TIMINGS_PRINT_EVERY = int(os.environ.get("MODELOX_TIMINGS_PRINT_EVERY", "1"))

# =============================================================================
# 1. CACHÉ DE SEÑALES (REUTILIZACIÓN ENTRE TRIALS VECINOS)
# =============================================================================
_SIGNALS_CACHE: Dict[str, pl.DataFrame] = {}
_SIGNALS_CACHE_MAX = 8

# Intervalo de limpieza periódica (cada N trials)
_CLEANUP_INTERVAL = int(os.environ.get("MODELOX_CLEANUP_INTERVAL", "100"))


def _cache_signals(key: str, signals: pl.DataFrame) -> None:
    """Cachea señales para reutilización en vecinos."""
    global _SIGNALS_CACHE
    if len(_SIGNALS_CACHE) >= _SIGNALS_CACHE_MAX:
        _SIGNALS_CACHE.pop(next(iter(_SIGNALS_CACHE)))
    _SIGNALS_CACHE[key] = signals


def _get_cached_signals(key: str) -> Optional[pl.DataFrame]:
    """Obtiene señales cacheadas."""
    return _SIGNALS_CACHE.get(key)


def clear_all_caches() -> None:
    """Limpia todos los caches del sistema."""
    global _SIGNALS_CACHE
    _SIGNALS_CACHE.clear()
    gc.collect()


def periodic_cleanup(trial_number: int, force: bool = False) -> None:
    """
    Limpieza periódica cada N trials para mantener velocidad constante.
    
    Ejecuta:
    1. Limpia cachés de señales
    2. Fuerza garbage collection
    3. En macOS, intenta liberar memoria comprimida
    
    Args:
        trial_number: Número del trial actual
        force: Si True, ejecuta limpieza sin importar el intervalo
    """
    if not force and trial_number % _CLEANUP_INTERVAL != 0:
        return
    
    if trial_number == 0:
        return  # Skip primer trial
    
    # Limpiar cachés
    clear_all_caches()
    
    # Triple GC para liberar referencias cíclicas
    gc.collect()
    gc.collect()
    gc.collect()
    
    # En macOS, liberar memoria si es posible
    import platform
    if platform.system() == "Darwin":
        try:
            import subprocess
            subprocess.run(['purge'], capture_output=True, timeout=2)
        except:
            pass


# =============================================================================
# 2. CONFIGURACIÓN
# =============================================================================

@dataclass(frozen=True)
class OptunaConfig:
    """
    Configuración de Optuna para el runner.
    
    NOTA: El sampler (CMA/TPE) se configura en general/configuracion.py
          con la variable OPTUNA_SAMPLER.
    """
    seed: Optional[int] = None
    n_jobs: int = 1
    storage: Optional[str] = None
    study_name_prefix: str = "MODELOX"
    sampler: str = "CMA"  # Valor recibido desde configuracion.py




# =============================================================================
# 4. HELPERS
# =============================================================================

def create_study_for_strategy(
    *,
    cfg: OptunaConfig,
    strategy_name: str,
    activo: Optional[str] = None,
) -> optuna.study.Study:
    """
    Crea estudio Optuna con el sampler configurado.
    
    Delega la creación al módulo optimizers que contiene
    la lógica específica de cada sampler (CMA-ES o TPE).
    
    Ver: modelox/optimizers/cma.py y modelox/optimizers/tpe.py
    """
    return create_study(
        sampler=cfg.sampler,
        strategy_name=strategy_name,
        activo=activo,
        seed=cfg.seed,
        study_name_prefix=cfg.study_name_prefix,
        storage=cfg.storage,
    )


# =============================================================================
# 5. COMPONENTES DEL PIPELINE
# =============================================================================

@dataclass
class DataLoader:
    """Maneja la carga de datos."""
    
    @staticmethod
    def load_data(file_path: str) -> pl.DataFrame:
        if file_path.endswith(".parquet"):
            df = pl.read_parquet(file_path)
        elif file_path.endswith(".csv"):
            df = pl.read_csv(file_path)
        elif file_path.endswith(".feather") or file_path.endswith(".arrow"):
            df = pl.read_ipc(file_path)
        else:
            raise ValueError(f"Formato no soportado: {file_path}")
        
        if "timestamp" not in df.columns and "datetime" in df.columns:
            df = df.rename({"datetime": "timestamp"})
        
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame debe tener columna 'timestamp' o 'datetime'")
        
        return df


@dataclass
class SignalGenerator:
    """Ejecuta la estrategia y retorna DataFrame con señales."""
    
    @staticmethod
    def generate_signals(
        df: pl.DataFrame,
        strategy: Strategy,
        params: Dict[str, Any],
        df_by_timeframe: Optional[Dict[str, pl.DataFrame]] = None,
    ) -> pl.DataFrame:
        base_tf = normalize_timeframe_to_suffix(params.get("__timeframe_base", "1m"))
        
        if hasattr(strategy, "get_required_timeframes") and callable(strategy.get_required_timeframes):
            required_tfs = strategy.get_required_timeframes(params)
            if required_tfs and df_by_timeframe:
                df = prepare_multitimeframe_data(
                    df, required_tfs, base_tf=base_tf, anti_lookahead=True,
                )
        
        signals_df = strategy.generate_signals(df, params)
        
        # OPTIMIZACIÓN: Solo añadir columnas si no existen
        cols = signals_df.columns
        if "signal_long" not in cols:
            signals_df = signals_df.with_columns(pl.lit(False).alias("signal_long"))
        if "signal_short" not in cols:
            signals_df = signals_df.with_columns(pl.lit(False).alias("signal_short"))
        
        return signals_df


@dataclass
class BacktestEngine:
    """Ejecuta backtest y retorna métricas."""
    
    # Cache de BacktestParams para evitar recreación
    _params_cache: Dict[int, BacktestParams] = field(default_factory=dict)
    
    @staticmethod
    def run_backtest(
        df: pl.DataFrame,
        signals: pl.DataFrame,
        config: BacktestConfig,
        params: Dict[str, Any],
        strategy: Strategy,
        df_1m: Optional[pl.DataFrame] = None,
        signals_1m: Optional[pl.DataFrame] = None,
    ) -> Tuple[pl.DataFrame, List[float], Dict[str, Any]]:
        """
        Ejecuta backtest y retorna métricas.
        
        IMPORTANTE - SALIDAS EN 1M:
            Si df_1m es proporcionado y el timeframe de entrada > 1m,
            las salidas (SL/TP/Trailing/Custom) se evalúan usando velas
            de 1 minuto para máxima precisión.
        
        Args:
            df: DataFrame con datos OHLCV del timeframe de entrada
            signals: DataFrame con señales de entrada/salida
            config: Configuración del backtest
            params: Parámetros del trial
            strategy: Estrategia
            df_1m: DataFrame con datos OHLCV de 1 minuto (para salidas precisas)
            signals_1m: DataFrame con señales de salida en 1m (opcional)
        
        Returns:
            Tuple[trades_df, equity_curve, metrics]
        """
        from modelox.core.types import suffix_to_minutes
        
        backtest_params = BacktestParams.from_config_and_params(config, params)
        timeframe = params.get("__timeframe_base", "1m")
        timeframe_minutes = suffix_to_minutes(timeframe)
        
        trades_df, equity_curve = calculate_performance_vectorized_numba(
            df=df,
            signals=signals,
            params=backtest_params,
            strategy=strategy,
            df_1m=df_1m,
            signals_1m=signals_1m,
            timeframe_minutes=timeframe_minutes,
        )
        
        metrics: Dict[str, Any]
        if not trades_df.is_empty():
            # Extraer rango real del periodo desde el DataFrame de datos
            # para que trades_por_dia use el rango completo, no solo trades
            _period_start = None
            _period_end = None
            if "timestamp" in df.columns and len(df) > 0:
                try:
                    _period_start = df["timestamp"][0]
                    _period_end = df["timestamp"][-1]
                except Exception:
                    pass

            metrics = resumen_metricas(
                trades_df,
                saldo_inicial=config.saldo_inicial,
                equity_curve=equity_curve,
                period_start=_period_start,
                period_end=_period_end,
                timeframe=timeframe,
            )
        else:
            metrics = {}
        
        return trades_df, equity_curve, metrics


# =============================================================================
# 6. RUNNER PRINCIPAL DE OPTIMIZACIÓN
# =============================================================================

@dataclass
class OptimizationRunner:
    """
    Runner de Optimización Bayesiana con soporte para CMA-ES y TPE.
    
    SAMPLERS:
    - CMA-ES: Covariance Matrix Adaptation Evolution Strategy
              - Aprende de los scores para adaptar la búsqueda
              - Ideal para scoring institucional multiplicativo
              - Favorece regiones estables (mesetas de parámetros)
    - TPE: Tree-structured Parzen Estimator (clásico de Optuna)
    
    Soporta:
    - Scoring institucional multiplicativo (PSR, DSR, K-Ratio, etc.)
    - Datos futuros con rango distinto por trial (FuturoTrialDataProvider)
    """
    
    config: BacktestConfig
    n_trials: int
    reporters: Sequence[Reporter]
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    activo: Optional[str] = None
    
    # Proveedor de datos futuros (opcional)
    futuro_data_provider: Optional[Any] = None
    
    # Data Augmentation (entrenamiento robusto)
    augmenter: Optional[Any] = None
    
    # Estado interno
    _last_study: Optional[optuna.study.Study] = None
    
    def _get_score_func(self) -> Callable:
        """
        Retorna la función de scoring correspondiente al sampler elegido.
        
        - CMA → score_cma (scoring institucional con PSR/DSR/SAM)
        - TPE → score_tpe (scoring simple exploratorio)
        - GT  → score_gt  (GT-Score anti-overfitting con interceptor topológico)
        """
        sampler_type = self.optuna.sampler.upper() if self.optuna.sampler else "CMA"
        if sampler_type == "TPE":
            return score_tpe
        if sampler_type == "GT":
            return score_gt
        if sampler_type in ("ML", "MLFOREST", "ML_FOREST"):
            return score_ml
        if sampler_type == "QMC":
            return score_qmc
        if sampler_type == "HYBRID":
            return score_hybrid
        return score_cma  # Default: CMA
    
    def optimize_strategies(
        self,
        *,
        df: pl.DataFrame,
        strategies: Sequence[Strategy],
        df_by_timeframe: Optional[Dict[str, pl.DataFrame]] = None,
        base_timeframe: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Optimiza una o más estrategias."""
        results: Dict[str, Any] = {}
        
        for strat in strategies:
            study = self._optimize_one(
                df=df,
                strategy=strat,
                df_by_timeframe=df_by_timeframe,
                base_timeframe=base_timeframe,
            )
            results[strat.name] = study
            self._last_study = study
            
            for reporter in self.reporters:
                if hasattr(reporter, "on_strategy_end"):
                    try:
                        reporter.on_strategy_end(strat.name, study)
                    except Exception:
                        pass
        
        return results
    
    def _optimize_one(
        self,
        *,
        df: pl.DataFrame,
        strategy: Strategy,
        df_by_timeframe: Optional[Dict[str, pl.DataFrame]] = None,
        base_timeframe: Optional[str] = None,
    ) -> optuna.study.Study:
        """Optimiza una estrategia."""
        from modelox.core.types import suffix_to_minutes
        
        base_tf = normalize_timeframe_to_suffix(base_timeframe or "1m")
        df_map = df_by_timeframe.copy() if df_by_timeframe else {base_tf: df}
        df_base = df_map.get(base_tf, df)
        
        # ================================================================
        # ASEGURAR DATOS DE 1M PARA SALIDAS PRECISAS
        # ================================================================
        # Si el timeframe base > 1m y no tenemos datos de 1m en el cache,
        # intentamos cargarlos para usarlos en las salidas
        base_tf_minutes = suffix_to_minutes(base_tf)
        if base_tf_minutes > 1 and "1m" not in df_map:
            # Intentar cargar datos de 1m si hay un futuro_data_provider
            if self.futuro_data_provider is not None:
                try:
                    # El provider cargará los datos de 1m cuando se necesiten
                    # No los precargamos aquí para no consumir memoria innecesariamente
                    pass
                except Exception:
                    pass
        
        objective = self._create_single_objective(df_base, df_map, strategy, base_tf)
        
        sampler_type = self.optuna.sampler.upper() if self.optuna.sampler else "CMA"
        
        if sampler_type == "HYBRID":
            import math
            import optuna
            n_qmc = int(math.ceil(self.n_trials / 2))
            n_tpe = self.n_trials - n_qmc
            
            from optuna.samplers import QMCSampler, TPESampler
            import re
            
            # Use provided storage or create a file-based storage for sharing between samplers
            if self.optuna.storage:
                storage = self.optuna.storage
            else:
                import os
                db_path = os.path.join(os.getcwd(), "optuna_hybrid.db")
                storage = f"sqlite:///{db_path}"
            
            # Generar nombre del estudio unificado para HYBRID
            parts = [self.optuna.study_name_prefix, str(strategy.name), "HYBRID"]
            if self.activo:
                parts.append(str(self.activo))
            
            s = "_".join(parts).strip().lower()
            s = re.sub(r'[^a-z0-9]+', '_', s)
            study_name = s.strip('_')[:50]
            
            # Phase 1: QMC
            sampler_qmc = QMCSampler(seed=self.optuna.seed, warn_independent_sampling=False)
            study_qmc = optuna.create_study(
                direction="maximize",
                sampler=sampler_qmc,
                study_name=study_name,
                storage=storage,
                load_if_exists=True,
            )
            
            if n_qmc > 0:
                study_qmc.optimize(
                    objective,
                    n_trials=n_qmc,
                    n_jobs=int(getattr(self.optuna, "n_jobs", 1)),
                    gc_after_trial=True,
                    catch=(Exception,),
                )
            
            # Phase 2: TPE
            sampler_tpe = TPESampler(seed=self.optuna.seed, n_startup_trials=0, multivariate=True)
            study_tpe = optuna.create_study(
                direction="maximize",
                sampler=sampler_tpe,
                study_name=study_name,
                storage=storage,
                load_if_exists=True,
            )
            
            if n_tpe > 0:
                study_tpe.optimize(
                    objective,
                    n_trials=n_tpe,
                    n_jobs=int(getattr(self.optuna, "n_jobs", 1)),
                    gc_after_trial=True,
                    catch=(Exception,),
                )
                
            return study_tpe
        else:
            study = create_study_for_strategy(
                cfg=self.optuna, 
                strategy_name=strategy.name, 
                activo=self.activo,
            )
            
            study.optimize(
                objective,
                n_trials=int(self.n_trials),
                n_jobs=int(getattr(self.optuna, "n_jobs", 1)),
                gc_after_trial=True,
                catch=(Exception,),
            )
            
            return study
    
    def _prepare_params(
        self,
        trial: optuna.trial.Trial,
        strategy: Strategy,
        base_tf: str,
    ) -> Dict[str, Any]:
        """Prepara parámetros para un trial."""
        params_puros = strategy.suggest_params(trial)
        params_rt = dict(params_puros)
        
        # Inyectar valores de configuración
        params_rt["__activo"] = self.activo
        params_rt["__saldo_inicial"] = float(self.config.saldo_inicial)
        params_rt["__saldo_operativo_max"] = float(self.config.saldo_operativo_max)
        
        params_rt["__comision_pct"] = float(self.config.comision_pct)
        params_rt["__comision_sides"] = int(self.config.comision_sides)
        params_rt["__saldo_usado"] = float(self.config.saldo_usado)
        params_rt["__apalancamiento_max"] = float(self.config.apalancamiento_max)
        params_rt["__strategy_exit_enabled"] = bool(getattr(strategy, "SALIDAS_PERSONALIZADAS", False))
        
        # Resolver configuración de salida
        exit_settings = resolve_exit_settings_for_trial(trial=trial, config=self.config)
        params_rt["__exit_type"] = exit_settings.exit_type
        params_rt["__exit_sl_pct"] = exit_settings.sl_pct
        params_rt["__exit_tp_pct"] = exit_settings.tp_pct
        params_rt["__exit_trail_act_pct"] = exit_settings.trail_act_pct
        params_rt["__exit_trail_dist_pct"] = exit_settings.trail_dist_pct
        
        # Aliases para compatibilidad
        params_rt["exit_type"] = exit_settings.exit_type
        params_rt["exit_sl_pct"] = exit_settings.sl_pct
        params_rt["exit_tp_pct"] = exit_settings.tp_pct
        params_rt["exit_trail_act_pct"] = exit_settings.trail_act_pct
        params_rt["exit_trail_dist_pct"] = exit_settings.trail_dist_pct
        
        # Timeframes
        entry_tf = normalize_timeframe_to_suffix(getattr(strategy, "timeframe_entry", None) or base_tf)
        exit_tf = normalize_timeframe_to_suffix(getattr(strategy, "timeframe_exit", None) or base_tf)
        
        params_rt["__timeframe_base"] = base_tf
        params_rt["__timeframe_entry"] = entry_tf
        params_rt["__timeframe_exit"] = exit_tf
        
        return params_rt
    
    def _create_single_objective(
        self,
        df_base: pl.DataFrame,
        df_map: Dict[str, pl.DataFrame],
        strategy: Strategy,
        base_tf: str,
    ) -> Callable[[optuna.trial.Trial], float]:
        """Crea función objetivo para TPE (Single-Objective)."""
        
        def objective(trial: optuna.trial.Trial) -> float:
            try:
                t0_total = time.perf_counter()
                
                # LIMPIEZA PERIÓDICA para mantener velocidad constante
                periodic_cleanup(trial.number)
                
                params_rt = self._prepare_params(trial, strategy, base_tf)
                
                # INYECTAR SAMPLER UTILIZADO (Para visualización de HYBRID QMC/TPE)
                sampler_name = type(trial.study.sampler).__name__
                if sampler_name == "QMCSampler":
                    params_rt["__sampler_used"] = "QMC"
                elif sampler_name == "TPESampler":
                    params_rt["__sampler_used"] = "TPE"
                else:
                    params_rt["__sampler_used"] = sampler_name
                
                entry_tf = params_rt["__timeframe_entry"]
                
                # ================================================================
                # DATOS FUTUROS CON RANGO DISTINTO POR TRIAL
                # ================================================================
                if self.futuro_data_provider is not None:
                    # Obtener datos únicos para este trial
                    try:
                        timeframes_needed = list(set([base_tf, entry_tf]))
                        df_map_trial = self.futuro_data_provider.get_trial_data_multiframe(
                            trial_number=trial.number,
                            timeframes=timeframes_needed,
                            verbose=(trial.number % 50 == 0),  # Verbose cada 50 trials
                        )
                        df_entry = df_map_trial.get(entry_tf, df_map_trial.get(base_tf))
                        df_base_trial = df_map_trial.get(base_tf, df_entry)
                        
                        # Guardar rango de meses en params para display
                        if hasattr(self.futuro_data_provider, 'last_range_months') and self.futuro_data_provider.last_range_months:
                            params_rt["__rango_meses"] = self.futuro_data_provider.last_range_months
                    except Exception as e:
                        # Fallback a datos originales
                        df_map_trial = df_map
                        df_entry = df_map.get(entry_tf, df_base)
                        df_base_trial = df_base
                else:
                    df_map_trial = df_map
                    df_entry = df_map.get(entry_tf, df_base)
                    df_base_trial = df_base
                
                # ================================================================
                # OBTENER DATOS DE 1M (Para Augmentation y Salidas)
                # ================================================================
                from modelox.core.types import suffix_to_minutes
                entry_tf_minutes = suffix_to_minutes(entry_tf)
                
                df_1m_for_exits = df_map_trial.get("1m")
                
                # Intentar cargar 1m si no está en el cache, lo necesitamos si hay exits tick-tick o augmentation
                if df_1m_for_exits is None and self.futuro_data_provider is not None:
                    if entry_tf_minutes > 1 or self.augmenter is not None:
                        try:
                            df_1m_data = self.futuro_data_provider.get_trial_data_multiframe(
                                trial_number=trial.number,
                                timeframes=["1m"],
                                verbose=False,
                            )
                            df_1m_for_exits = df_1m_data.get("1m")
                        except Exception:
                            pass
                
                # ================================================================
                # DATOS DEL TRIAL Y DATA AUGMENTATION (COHERENCIA FRACTAL)
                # ================================================================
                df_trial = df_entry
                
                if self.augmenter is not None:
                    if df_1m_for_exits is not None:
                        try:
                            # 1. Perturbar ÚNICAMENTE el dataframe atómico de 1 minuto
                            df_1m_aug = self.augmenter.augment(df_1m_for_exits.clone().lazy())
                            
                            # 2. Asignarlo para que el BacktestEngine use los mismos precios en Salidas
                            df_1m_for_exits = df_1m_aug
                            
                            # 3. Reconstruir df_trial (ej. 5m) a partir del 1m perturbado
                            if entry_tf == "1m":
                                df_trial = df_1m_aug
                            else:
                                df_trial = df_1m_aug.group_by_dynamic("timestamp", every=entry_tf).agg(
                                    pl.col("open").first(),
                                    pl.col("high").max(),
                                    pl.col("low").min(),
                                    pl.col("close").last(),
                                    pl.col("volume").sum()
                                )
                        except Exception as e:
                            # Fallback silencioso a datos originales
                            pass
                    else:
                        # Si por alguna razón no logramos obtener 1m, perturbamos df_trial directamente (fallback)
                        try:
                            df_trial = self.augmenter.augment(df_trial.clone().lazy())
                        except Exception:
                            pass
                
                # Generar señales
                t1_signals = time.perf_counter()
                signals_df = SignalGenerator.generate_signals(df_trial, strategy, params_rt, df_map_trial)
                t2_signals = time.perf_counter()
                
                # ================================================================
                # GENERAR SEÑALES DE SALIDA EN 1M (Tick a Tick precise exits)
                # ================================================================
                signals_1m_for_exits = None
                if entry_tf_minutes > 1 and df_1m_for_exits is not None:
                    if hasattr(strategy, "generate_exit_signals_1m") and callable(strategy.generate_exit_signals_1m):
                        try:
                            signals_1m_for_exits = strategy.generate_exit_signals_1m(df_1m_for_exits, params_rt)
                        except Exception:
                            signals_1m_for_exits = None
                
                # Ejecutar backtest con datos de 1m para salidas
                t1_backtest = time.perf_counter()
                trades_df, equity_curve, metrics = BacktestEngine.run_backtest(
                    df_trial, signals_df, self.config, params_rt, strategy,
                    df_1m=df_1m_for_exits,
                    signals_1m=signals_1m_for_exits,
                )
                t2_backtest = time.perf_counter()
                
                if trades_df.is_empty():
                    # Handle empty trades as valid result but poor score
                    metrics = {"roi": -100.0, "sharpe": -999.0, "sqn": -999.0, "profit_factor": 0.0}
                    score = -9999.0
                else:
                    trial.set_user_attr("metricas", metrics)
                    # Usar scoring correspondiente al sampler elegido
                    score_func = self._get_score_func()
                    score = float(score_func(metrics, trial=trial))
                
                t_total = time.perf_counter() - t0_total
                
                if _TIMINGS_VERBOSE and (trial.number % _TIMINGS_PRINT_EVERY == 0):
                    print(
                        f"  ⏱ TRIAL {trial.number:3d} │ "
                        f"signals {(t2_signals - t1_signals)*1000:6.1f}ms │ "
                        f"backtest {(t2_backtest - t1_backtest)*1000:6.1f}ms │ "
                        f"total {t_total*1000:6.1f}ms │ "
                        f"trades {len(trades_df):5d}"
                    )
                
                # Crear artifacts
                # ⚠ CRITICAL: usar df_trial (perturbado) para que los precios
                # OHLC en gráficos/excel coincidan con los trades ejecutados.
                # ANTES: usaba df_base_trial/df_base → DATA LEAKAGE (markers desalineados)
                df_signals_for_artifacts = None
                for reporter in self.reporters:
                    if hasattr(reporter, "needs_dataframe") and reporter.needs_dataframe(score):
                        ohlc_cols = ["timestamp", "open", "high", "low", "close", "volume"]
                        base_cols = [c for c in ohlc_cols if c in df_trial.columns]
                        signal_cols = [c for c in signals_df.columns if c not in base_cols]
                        df_signals_for_artifacts = df_trial.select(base_cols).hstack(
                            signals_df.select(signal_cols)
                        )
                        break
                
                # Calcular rango de fechas real del trial (para plots dinámicos)
                _trial_date_range = None
                if "timestamp" in df_trial.columns:
                    try:
                        _ts = df_trial["timestamp"]
                        _t0 = str(_ts[0])[:10]
                        _t1 = str(_ts[-1])[:10]
                        _trial_date_range = (_t0, _t1)
                    except Exception:
                        pass

                artifacts = TrialArtifacts(
                    strategy_name=strategy.name,
                    trial_number=trial.number,
                    params=params_rt,
                    params_reporting=params_rt,
                    score=score,
                    metrics=metrics,
                    df_signals=df_signals_for_artifacts,
                    trades=trades_df.to_pandas() if not trades_df.is_empty() else None,
                    equity_curve=equity_curve,
                    indicators_used=params_rt.get("__indicators_used", []),
                    trial_date_range=_trial_date_range,
                )
                
                for reporter in self.reporters:
                    reporter.on_trial_end(artifacts)
                
                return score

            except Exception as e:
                # ERROR HANDLING: Report failure and continue
                print(f"\n❌ TRIAL {trial.number} FAILED: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Create dummy artifacts for reporting
                dummy_metrics = {"roi": 0.0, "sharpe": 0.0, "sqn": 0.0}
                dummy_artifacts = TrialArtifacts(
                    strategy_name=strategy.name,
                    trial_number=trial.number,
                    params={},
                    params_reporting={},
                    score=-9999.0,
                    metrics=dummy_metrics,
                    df_signals=None,
                    trades=None,
                    equity_curve=[],
                    indicators_used=[],
                    trial_date_range=None,
                )
                
                for reporter in self.reporters:
                    reporter.on_trial_end(dummy_artifacts)
                
                return -9999.0
        
        return objective


# =============================================================================
# FUNCIONES DE LIMPIEZA DE RECURSOS
# =============================================================================

def cleanup_parallel_resources():
    """
    Limpia todos los recursos de optimización.
    
    Llamar al final de la optimización para liberar:
    - Caches de señales
    - Garbage collector
    """
    clear_all_caches()
    gc.collect()


# =============================================================================
# EJECUCIÓN DE EXIT TYPE - Función auxiliar para ejecutar.py
# =============================================================================

def run_single_exit_type(
    *,
    exit_type: str,
    strategy: Strategy,
    strategy_name: str,
    strategy_safe: str,
    activo: str,
    df_filtrado: pl.DataFrame,
    tf_cache: dict,
    timeframe_base: int,
    cfg: BacktestConfig,
    tf_display: str,
    archivo_data: str,
    periodo_datos: str,
    # Configuración de optimización
    n_trials: int,
    optuna_sampler: str = "CMA",
    # Datos sintéticos/futuros
    synthetic_mode: bool = False,
    synthetic_years: int = 0,
    futuro_data_provider: Optional[Any] = None,  # FuturoTrialDataProvider
    # Rutas de salida
    excel_dir: str = None,
    graficos_dir: str = None,
    # Opciones
    usar_excel: bool = True,
    generar_plots: bool = True,
    max_archivos: int = 5,
    # Configuración de gráficos (rango binario)
    grafica_rango_personalizado: bool = False,
    grafica_fecha_inicio: str = "2025-01-01",
    grafica_fecha_fin: str = "2025-02-10",
    # Funciones auxiliares
    resolve_archivo_data_tf_func = None,
    fecha_inicio: str = None,
    fecha_fin: str = None,
    # Callbacks visuales
    mostrar_cabecera_func = None,
    mostrar_fin_func = None,
    # Reporters personalizados
    reporters: list = None,
    logger = None,
) -> None:
    """
    Ejecuta optimización para un único tipo de salida (pnl_fixed o pnl_trailing).
    
    Esta función encapsula toda la lógica de:
    1. Configurar el exit_type en BacktestConfig
    2. Detectar capacidades de la estrategia
    3. Mostrar header
    4. Configurar reporters
    5. Ejecutar runner
    6. Mostrar resultados
    7. Limpiar recursos
    
    Args:
        exit_type: "pnl_fixed" o "pnl_trailing"
        strategy: Objeto estrategia con suggest_params y generate_signals
        strategy_name: Nombre de la estrategia
        strategy_safe: Nombre sanitizado para rutas
        activo: Activo siendo optimizado
        df_filtrado: DataFrame con datos filtrados por fecha
        tf_cache: Cache de DataFrames por timeframe
        timeframe_base: Timeframe base en minutos
        cfg: BacktestConfig con parámetros de backtest
        tf_display: String de timeframe para mostrar
        archivo_data: Ruta al archivo de datos
        periodo_datos: String con el periodo (ej: "2021-01-01 -> 2024-01-01")
        n_trials: Número de trials de optimización
        optuna_sampler: "CMA" o "TPE"
        excel_dir: Directorio para guardar Excel
        graficos_dir: Directorio para guardar gráficos
        usar_excel: Si generar reportes Excel
        generar_plots: Si generar gráficos HTML
        max_archivos: Máximo de archivos a guardar
        grafica_rango_personalizado: True=rango manual, False=últimos 2 meses auto
        grafica_fecha_inicio: Fecha inicio (solo si grafica_rango_personalizado=True)
        grafica_fecha_fin: Fecha fin (solo si grafica_rango_personalizado=True)
        resolve_archivo_data_tf_func: Función para resolver rutas de TF
        fecha_inicio: Fecha inicio del backtest
        fecha_fin: Fecha fin del backtest
        mostrar_cabecera_func: Función para mostrar cabecera
        mostrar_fin_func: Función para mostrar fin
        reporters: Lista de reporters personalizados (override)
        logger: Logger para mensajes
    """
    from modelox.core.types import nuclear_cleanup
    
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)

    # 1. CONFIGURACIÓN
    cfg_dict = cfg.__dict__.copy()
    cfg_dict["exit_type"] = str(exit_type)
    cfg_updated = BacktestConfig(**cfg_dict)

    # 2. DETECCIÓN DE CAPACIDADES
    try:
        indicadores = list(getattr(strategy, "parametros_optuna", {}).keys())
    except Exception:
        indicadores = []

    try:
        strategy_exit_enabled = bool(
            callable(getattr(strategy, "decide_exit", None))
            and bool(getattr(strategy, "ACTIVAR_SALIDA_PERSONALIZADA", False))
        )
    except Exception:
        strategy_exit_enabled = False

    # 3. MOSTRAR HEADER
    if mostrar_cabecera_func:
        mostrar_cabecera_func(
            activo=activo,
            combo_nombre=strategy_name,
            indicadores=indicadores,
            n_trials=n_trials,
            archivo_data=archivo_data,
            timeframe=tf_display,
            periodo=periodo_datos,
            exit_type=exit_type,
            strategy_exit_enabled=strategy_exit_enabled,
            sampler_type=optuna_sampler,
            synthetic_mode=synthetic_mode,
            synthetic_years=synthetic_years,
        )

    # 4. REPORTEROS
    if reporters is None:
        reporters = []
        
        # Rich reporter siempre
        from visual.rich import ElegantRichReporter
        reporters.append(ElegantRichReporter(
            saldo_inicial=cfg_updated.saldo_inicial,
            activo=activo,
            n_trials_total=n_trials,
        ))

        if usar_excel and excel_dir:
            from visual.excel import ExcelReporter
            reporters.append(ExcelReporter(
                resumen_path=f"{excel_dir}/RESUMEN ID{strategy.combinacion_id}.xlsx",
                trades_base_dir=excel_dir,
                max_archivos=max_archivos,
            ))

        if generar_plots and graficos_dir:
            from visual.grafico import PlotReporter
            
            reporters.append(PlotReporter(
                plot_base=graficos_dir,
                grafica_rango_personalizado=grafica_rango_personalizado,
                grafica_fecha_inicio=grafica_fecha_inicio,
                grafica_fecha_fin=grafica_fecha_fin,
                max_archivos=max_archivos,
                saldo_inicial=cfg_updated.saldo_inicial,
                activo=activo,
            ))

    # 5. RUNNER
    runner = OptimizationRunner(
        config=cfg_updated, 
        n_trials=n_trials, 
        reporters=reporters,
        futuro_data_provider=futuro_data_provider,
    )

    # 5b. DATA AUGMENTATION (ENTRENAMIENTO ROBUSTO)
    try:
        from general.configuracion import (
            ENTRENAMIENTO_ROBUSTO_ACTIVAR,
            MAX_DRIFT_PCT,
            DRIFT_STEP_VOLATILITY,
            RUIDO_MECHAS_PCT,
            RUIDO_VOLUMEN_PCT,
        )
        if ENTRENAMIENTO_ROBUSTO_ACTIVAR:
            from modelox.perturbaciones import HighFrequencyOHLCVAugmenter
            runner.augmenter = HighFrequencyOHLCVAugmenter(
                max_drift_pct=MAX_DRIFT_PCT,
                drift_step_volatility=DRIFT_STEP_VOLATILITY,
                wick_noise_scale=RUIDO_MECHAS_PCT,
                volume_noise_scale=RUIDO_VOLUMEN_PCT,
            )
    except ImportError:
        pass  # Sin perturbaciones si no está disponible

    runner.optuna = OptunaConfig(
        seed=None, 
        n_jobs=1, 
        storage=None,
        sampler=optuna_sampler,
    )

    runner.activo = activo

    try:
        from modelox.core.types import suffix_to_minutes
        
        # Carga diferida de timeframes extra
        entry_tf = getattr(strategy, "timeframe_entry", None) or timeframe_base
        exit_tf = getattr(strategy, "timeframe_exit", None) or timeframe_base
        needed_tfs = [timeframe_base, entry_tf, exit_tf]
        
        # ================================================================
        # SIEMPRE CARGAR DATOS DE 1M PARA SALIDAS PRECISAS
        # ================================================================
        # Si el timeframe base > 1m, necesitamos datos de 1m para evaluar
        # las salidas (SL/TP/Trailing/Custom) con precisión tick-a-tick
        base_tf_minutes = suffix_to_minutes(normalize_timeframe_to_suffix(timeframe_base))
        if base_tf_minutes > 1 and "1m" not in needed_tfs:
            needed_tfs.append("1m")

        for tf in needed_tfs:
            tf_suf = normalize_timeframe_to_suffix(tf)
            if tf_suf in tf_cache:
                continue

            if resolve_archivo_data_tf_func:
                try:
                    from .data import load_data
                    from .types import filter_by_date
                    path_tf = resolve_archivo_data_tf_func(activo, tf)
                    df_tf = load_data(path_tf)
                    if fecha_inicio and fecha_fin:
                        df_tf = filter_by_date(df_tf, fecha_inicio, fecha_fin)
                    tf_cache[tf_suf] = df_tf
                    
                    # Log si cargamos datos de 1m para salidas
                    if tf_suf == "1m" and base_tf_minutes > 1:
                        logger.info(f"✓ Datos de 1m cargados para salidas precisas (TF entrada: {timeframe_base})")
                except Exception as e:
                    if tf_suf == "1m":
                        logger.warning(f"⚠ No se pudieron cargar datos de 1m para salidas: {e}")
                    else:
                        logger.warning(f"No se pudo cargar TF extra {tf}: {e}")

        runner.optimize_strategies(
            df=df_filtrado,
            strategies=[strategy],
            df_by_timeframe=tf_cache,
            base_timeframe=timeframe_base,
        )

        # VISUALIZACIÓN DE RESULTADOS
        if hasattr(runner, "_last_study") and runner._last_study:
            study = runner._last_study
            is_multiobj = len(study.directions) > 1

            try:
                best_trial = None
                best_val = 0.0

                if is_multiobj:
                    pareto_front = sorted(study.best_trials, key=lambda t: t.values[0], reverse=True)
                    if pareto_front:
                        best_trial = pareto_front[0]
                        best_val = best_trial.values[0]
                else:
                    if study.best_trial:
                        best_trial = study.best_trial
                        best_val = study.best_value

                if best_trial and mostrar_fin_func:
                    # Calcular métricas extra para el reporte final
                    try:
                        valid_trials = [t for t in study.trials if t.state.name == "COMPLETE"]
                        roi_vals = [t.user_attrs.get("metricas", {}).get("roi", 0) for t in valid_trials]
                        roi_medio = sum(roi_vals) / len(roi_vals) if roi_vals else 0.0
                        best_roi = best_trial.user_attrs.get("metricas", {}).get("roi", 0.0)
                    except Exception:
                        roi_medio = 0.0
                        best_roi = 0.0

                    # Buscar path del Excel generado
                    excel_path = None
                    for r in reporters:
                        if hasattr(r, "_final_excel_path") and r._final_excel_path:
                            excel_path = r._final_excel_path
                            break

                    mostrar_fin_func(
                        total_trials=len(study.trials),
                        best_score=best_val,
                        best_trial=best_trial.number,
                        estrategia=strategy_name,
                        activo=activo,
                        roi_medio=roi_medio,
                        best_roi=best_roi,
                        excel_report=excel_path,
                    )
            except Exception as e:
                logger.warning(f"No se pudo extraer el mejor trial: {e}")

    except KeyboardInterrupt:
        raise
    except Exception as e:
        logger.error(f"Error en {strategy_name}: {e}")
    finally:
        del runner
        del reporters
        nuclear_cleanup()
        
        # Eliminar base de datos híbrida al acabar la optimización
        if optuna_sampler.upper() == "HYBRID":
            import os
            db_path = os.path.join(os.getcwd(), "optuna_hybrid.db")
            if os.path.exists(db_path):
                try:
                    os.remove(db_path)
                    logger.info("✓ Base de datos Optuna Híbrida eliminada")
                except Exception as e:
                    logger.warning(f"⚠ No se pudo eliminar la base de datos Optuna Híbrida: {e}")

