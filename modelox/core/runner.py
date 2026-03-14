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
     5. PIPELINE                  — SignalGenerator, BacktestEngine
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
    TPEScorer, QMCScorer, HybridScorer,
    score_tpe, score_qmc, score_hybrid,
    create_study,  # Factory para crear estudios
)
from modelox.optimizers.storage import (
    resolve_storage_for_strategy,
    get_database_file_path,
    delete_strategy_database,
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
    create_database: bool = True
    study_name_prefix: str = "MODELOX"
    sampler: str = "CMA"  # Valor recibido desde configuracion.py
    guardar_equity_en_db: bool = False  # True = guarda equity_curve de cada trial (Monte Carlo)




# =============================================================================
# 4. HELPERS
# =============================================================================

def create_study_for_strategy(
    *,
    cfg: OptunaConfig,
    strategy_name: str,
    strategy_id: Optional[int] = None,
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
        strategy_id=strategy_id,
        activo=activo,
        seed=cfg.seed,
        study_name_prefix=cfg.study_name_prefix,
        create_database=cfg.create_database,
    )


# =============================================================================
# 5. COMPONENTES DEL PIPELINE
# =============================================================================


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
    
    # Régimen de mercado (filtro EMA 21/200 en 1D)
    regimen_activo: bool = False
    regimen_modo: str = "true"  # false|true|all|odd
    regimen_tipo: str = "ALCISTA"
    regimen_buy_hold_activo: bool = False
    regimen_buy_hold_saldo: Any = "ALL"
    regimen_df: Optional[pl.DataFrame] = None  # Pre-computado: timestamp, regime, regime_bullish
    regimen_dias_operables: int = 0
    
    # Estado interno
    _last_study: Optional[optuna.study.Study] = None
    
    def _get_score_func(self) -> Callable:
        """
        Retorna la función de scoring correspondiente al sampler elegido.
        
        - TPE → score_tpe
        - QMC → score_qmc
        - HYBRID → score_hybrid (default)
        """
        sampler_type = self.optuna.sampler.upper() if self.optuna.sampler else "HYBRID"
        if sampler_type == "TPE":
            return score_tpe
        if sampler_type == "QMC":
            return score_qmc
        return score_hybrid  # Default: HYBRID
    
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
            
            storage = resolve_storage_for_strategy(
                create_database=bool(self.optuna.create_database),
                strategy_id=getattr(strategy, "combinacion_id", None),
            )
            
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
                strategy_id=getattr(strategy, "combinacion_id", None),
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
        salidas_personalizadas = bool(getattr(strategy, "SALIDAS_PERSONALIZADAS", False))
        params_rt["__strategy_exit_enabled"] = salidas_personalizadas
        
        # Resolver configuración de salida
        if salidas_personalizadas:
            # REGLA: con SALIDAS_PERSONALIZADAS=True, el SL fijo de exits.py
            # se conserva como backstop de emergencia. TP/Trailing se desactivan
            # (999) porque la estrategia gestiona sus propias salidas mediante
            # las columnas exit_long/exit_short. El SL actúa de último recurso
            # ante movimientos extremos que la salida custom no alcanzara a cubrir.
            exit_settings = resolve_exit_settings_for_trial(trial=trial, config=self.config)
            params_rt["__exit_type"] = "FIXED"               # SL fijo, sin trailing
            params_rt["__exit_sl_pct"] = exit_settings.sl_pct  # SL real = emergencia
            params_rt["__exit_tp_pct"] = 999.0              # TP desactivado (custom)
            params_rt["__exit_trail_act_pct"] = 999.0       # Trailing desactivado
            params_rt["__exit_trail_dist_pct"] = 999.0

            params_rt["exit_type"] = "FIXED"
            params_rt["exit_sl_pct"] = exit_settings.sl_pct
            params_rt["exit_tp_pct"] = 999.0
            params_rt["exit_trail_act_pct"] = 999.0
            params_rt["exit_trail_dist_pct"] = 999.0
            params_rt["__exit_time_bars"] = 0
            params_rt["exit_time_bars"] = 0
        else:
            exit_settings = resolve_exit_settings_for_trial(trial=trial, config=self.config)
            params_rt["__exit_type"] = exit_settings.exit_type
            params_rt["__exit_sl_pct"] = exit_settings.sl_pct
            params_rt["__exit_tp_pct"] = exit_settings.tp_pct
            params_rt["__exit_trail_act_pct"] = exit_settings.trail_act_pct
            params_rt["__exit_trail_dist_pct"] = exit_settings.trail_dist_pct
            params_rt["__exit_time_bars"] = exit_settings.time_stop_bars
            
            # Aliases para compatibilidad
            params_rt["exit_type"] = exit_settings.exit_type
            params_rt["exit_sl_pct"] = exit_settings.sl_pct
            params_rt["exit_tp_pct"] = exit_settings.tp_pct
            params_rt["exit_trail_act_pct"] = exit_settings.trail_act_pct
            params_rt["exit_trail_dist_pct"] = exit_settings.trail_dist_pct
            params_rt["exit_time_bars"] = exit_settings.time_stop_bars

        # ATR params (siempre inyectados para trazabilidad/reporting consistente)
        params_rt["__exit_atr_period"] = exit_settings.atr_period
        params_rt["__exit_atr_min_pct"] = exit_settings.atr_min_pct
        params_rt["__exit_atr_max_pct"] = exit_settings.atr_max_pct
        params_rt["__exit_atr_lookback"] = exit_settings.atr_lookback
        params_rt["exit_atr_period"] = exit_settings.atr_period
        params_rt["exit_atr_min_pct"] = exit_settings.atr_min_pct
        params_rt["exit_atr_max_pct"] = exit_settings.atr_max_pct
        params_rt["exit_atr_lookback"] = exit_settings.atr_lookback
        
        # Timeframes
        entry_tf = normalize_timeframe_to_suffix(getattr(strategy, "timeframe_entry", None) or base_tf)
        # time_bars: usa el mismo TF de entrada para contar barras (no 1m)
        exit_type_rt = str(params_rt.get("__exit_type", "FIXED")).upper()
        exit_tf = entry_tf if exit_type_rt == "BARS" else "1m"
        
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
                # SIEMPRE intentamos cargar 1m, incluso si entry_tf_minutes == 1, para tener coherencia
                if df_1m_for_exits is None:
                    if self.futuro_data_provider is not None:
                        try:
                            df_1m_data = self.futuro_data_provider.get_trial_data_multiframe(
                                trial_number=trial.number,
                                timeframes=["1m"],
                                verbose=False,
                            )
                            df_1m_for_exits = df_1m_data.get("1m")
                        except Exception:
                            pass
                    
                    # Fallback SIEMPRE: si el provider no devolvió 1m, usar el cache global
                    # Esto es CRÍTICO para que las salidas en 1m funcionen con TrialDataProvider
                    if df_1m_for_exits is None:
                        df_1m_for_exits = df_map.get("1m")

                # Deshabilitar 1m para salidas si la estrategia gestiona sus propias salidas
                # pero NO implementa generate_exit_signals_1m (fuerza uso del timeframe de entrada)
                salidas_personalizadas = bool(getattr(strategy, "SALIDAS_PERSONALIZADAS", False))
                if salidas_personalizadas:
                    if not (hasattr(strategy, "generate_exit_signals_1m") and callable(strategy.generate_exit_signals_1m)):
                        df_1m_for_exits = None
                
                # ================================================================
                # DATOS DEL TRIAL Y DATA AUGMENTATION (COHERENCIA FRACTAL)
                # ================================================================
                df_trial = df_entry

                # ================================================================
                # ALINEAR SIEMPRE EL 1M AL RANGO REAL DEL TRIAL
                # ================================================================
                # Evita data leakage cuando hay rangos por trial: si usamos el 1m
                # global (dataset completo), las salidas podrían evaluarse fuera de
                # la ventana del trial y desalinear métricas como TRADES/DAY.
                if df_1m_for_exits is not None and "timestamp" in df_trial.columns and len(df_trial) > 0:
                    try:
                        _ts_ini = df_trial["timestamp"][0]
                        _ts_fin = df_trial["timestamp"][-1]
                        df_1m_for_exits = df_1m_for_exits.filter(
                            (pl.col("timestamp") >= _ts_ini) &
                            (pl.col("timestamp") <= _ts_fin)
                        )
                        if len(df_1m_for_exits) == 0:
                            df_1m_for_exits = None
                    except Exception:
                        # Fallback defensivo: mejor no usar 1m que usar 1m desalineado
                        df_1m_for_exits = None
                
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
                # FILTRO DE RÉGIMEN DE MERCADO (EMA 21/200 | MACD 12/26/9 en 1D)
                # ================================================================
                # Régimen se recalcula PER-TRIAL desde df_trial (DESPUÉS de
                # perturbaciones), para que reflejen el precio perturbado.
                if self.regimen_activo:
                    try:
                        from modelox.core.regime import compute_regime_mask_1d, apply_precomputed_regime_filter
                        _reg_ind = getattr(self, "regimen_indicador", "EMA")
                        _reg_mode = str(getattr(self, "regimen_modo", "true")).lower().strip()
                        regime_df_trial = compute_regime_mask_1d(df_trial, indicador=_reg_ind)
                        signals_df, _dias_op = apply_precomputed_regime_filter(
                            signals_df, df_trial, regime_df_trial,
                            regimen_tipo=self.regimen_tipo,
                            regimen_modo=_reg_mode,
                            regimen_buy_hold_activo=self.regimen_buy_hold_activo,
                            regimen_buy_hold_saldo=self.regimen_buy_hold_saldo,
                        )

                        # Modos de régimen sin señales de estrategia:
                        # all/odd → entradas/salidas exclusivamente por régimen.
                        if _reg_mode in {"all", "odd"}:
                            params_rt["__regime_signal_only_mode"] = True
                            # Blindaje extra: desactivar cualquier salida por BARS/TIME STOP
                            # o por configuración previa del trial.
                            params_rt["__exit_type"] = "FIXED"
                            params_rt["exit_type"] = "FIXED"
                            params_rt["__exit_time_bars"] = 0
                            params_rt["exit_time_bars"] = 0
                            params_rt["time_stop_bars"] = 0

                        # Inyectar flags runtime para que el engine ajuste el tamaño
                        # de posición en modo Buy&Hold por régimen.
                        if "regime_buy_hold_mode" in signals_df.columns:
                            try:
                                _bh_mode = bool(signals_df["regime_buy_hold_mode"][0])
                            except Exception:
                                _bh_mode = False

                            if _bh_mode:
                                try:
                                    _bh_all = bool(signals_df["regime_buy_hold_saldo_all"][0])
                                except Exception:
                                    _bh_all = True
                                try:
                                    _bh_saldo = float(signals_df["regime_buy_hold_saldo"][0])
                                except Exception:
                                    _bh_saldo = 0.0

                                params_rt["__regime_buy_hold_mode"] = True
                                params_rt["__regime_buy_hold_saldo_all"] = _bh_all
                                params_rt["__regime_buy_hold_saldo"] = _bh_saldo
                    except Exception:
                        pass  # Fallback silencioso: operar sin filtro
                
                # ================================================================
                # GENERAR SEÑALES DE SALIDA EN 1M (Tick a Tick precise exits)
                # ================================================================
                signals_1m_for_exits = None
                # Generar señales 1m siempre que sea posible y df_1m exista
                if df_1m_for_exits is not None:
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
                
                # AJUSTAR trades_por_dia al régimen operable
                if self.regimen_activo and self.regimen_dias_operables > 0 and not trades_df.is_empty():
                    _n_trades = metrics.get("n_trades", 0) or metrics.get("total_trades", len(trades_df))
                    metrics["trades_por_dia"] = float(_n_trades) / float(self.regimen_dias_operables)
                
                if trades_df.is_empty():
                    # Handle empty trades as valid result but poor score
                    metrics = {"roi": 0.0, "sharpe": -1.0, "sqn": -1.0, "profit_factor": 0.0}
                    score = 0.0
                else:
                    # Persistencia defensiva de attrs Optuna:
                    # si hay errores de SQLite (disk I/O, lock, etc.)
                    # no abortar el trial por metadatos auxiliares.
                    try:
                        trial.set_user_attr("metricas", metrics)
                    except Exception:
                        pass

                    # Guardar equity curve para análisis Monte Carlo
                    if self.optuna.guardar_equity_en_db and equity_curve:
                        try:
                            trial.set_user_attr("equity_curve", [float(x) for x in equity_curve])
                        except Exception:
                            pass
                    
                    # Extraer pnl_neto array
                    if isinstance(trades_df, pl.DataFrame):
                        _pnl_arr = trades_df["pnl_neto"].to_numpy().astype(np.float64)
                    else:
                        _pnl_arr = trades_df["pnl_neto"].to_numpy(dtype=np.float64)

                    # total_days desde timestamps del DataFrame de entrada
                    _total_days = 0.0
                    if "timestamp" in df_trial.columns and len(df_trial) > 0:
                        try:
                            _ts0 = df_trial["timestamp"][0]
                            _ts1 = df_trial["timestamp"][-1]
                            _delta = pl.DataFrame({"s": [_ts0], "e": [_ts1]}).select(
                                ((pl.col("e") - pl.col("s")).dt.total_seconds() / 86400.0).alias("d")
                            )
                            _total_days = max(1.0, float(_delta["d"][0]))
                        except Exception:
                            _total_days = 1.0

                    # AJUSTE DE TOTAL_DAYS POR RÉGIMEN:
                    # Cuando el régimen está activo, usar solo los días operables
                    # para calcular trades_por_dia correctamente.
                    if self.regimen_activo and self.regimen_dias_operables > 0:
                        _total_days = max(1.0, float(self.regimen_dias_operables))

                    score = float(score_qmc(
                        metrics,
                        trial=trial,
                        trades_pnl=_pnl_arr,
                        total_days=_total_days,
                        saldo_inicial=float(self.config.saldo_inicial),
                        trades_df=trades_df,
                    ))
                
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
                    score=0.0,
                    metrics=dummy_metrics,
                    df_signals=None,
                    trades=None,
                    equity_curve=[],
                    indicators_used=[],
                    trial_date_range=None,
                )
                
                for reporter in self.reporters:
                    reporter.on_trial_end(dummy_artifacts)
                
                return 0.0
        
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
    Ejecuta optimización para un único tipo de salida (FIXED, TRAILING, BARS o ATR).

    Esta función encapsula toda la lógica de:
    1. Configurar el exit_type en BacktestConfig
    2. Detectar capacidades de la estrategia
    3. Mostrar header
    4. Configurar reporters
    5. Ejecutar runner
    6. Mostrar resultados
    7. Limpiar recursos
    
    Args:
        exit_type: "FIXED" | "TRAILING" | "BARS" | "ATR"
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

    def _normalize_regimen_mode(v) -> str:
        """Normaliza REGIMEN_ACTIVO a: false | true | all | odd."""
        if isinstance(v, bool):
            return "true" if v else "false"
        s = str(v).strip().lower()
        if s in {"1", "on", "yes", "si", "sí", "true", "t", "activar", "activo"}:
            return "true"
        if s in {"0", "off", "no", "false", "f", "desactivar", "inactivo"}:
            return "false"
        if s in {"all", "todo", "todos", "buyhold", "buy_hold"}:
            return "all"
        if s in {"odd", "regime", "regimen", "régimen"}:
            return "odd"
        return "true"
    
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
    try:
        from general.configuracion import ENTRENAMIENTO_ROBUSTO_ACTIVAR
    except ImportError:
        ENTRENAMIENTO_ROBUSTO_ACTIVAR = False

    try:
        from general.configuracion import (
            REGIMEN_ACTIVO, REGIMEN_TIPO,
            REGIMEN_BUY_HOLD_ACTIVO, REGIMEN_BUY_HOLD_SALDO,
            REGIMEN_INDICADOR,
        )
    except ImportError:
        REGIMEN_ACTIVO = False
        REGIMEN_TIPO = "ALCISTA"
        REGIMEN_BUY_HOLD_ACTIVO = False
        REGIMEN_BUY_HOLD_SALDO = "ALL"
        REGIMEN_INDICADOR = "EMA"

    _regimen_mode = _normalize_regimen_mode(REGIMEN_ACTIVO)
    _regimen_enabled = _regimen_mode != "false"

    # Pre-computar estadísticas del régimen (días operables / total)
    _regimen_dias_operables = 0
    _regimen_dias_totales = 0
    _regimen_df = None  # DataFrame del régimen pre-computado (para reutilizar en objective)
    if _regimen_enabled:
        try:
            from modelox.core.regime import compute_regime_mask_1d
            # Cargar datos 1m COMPLETOS (sin filtro de fecha) para warmup de EMA200
            _df_1m_full = None
            if resolve_archivo_data_tf_func:
                from .data import load_data as _ld_regime
                _path_1m = resolve_archivo_data_tf_func(activo, "1m")
                if os.path.exists(_path_1m):
                    _df_1m_full = _ld_regime(_path_1m)
            
            if _df_1m_full is not None and len(_df_1m_full) > 0:
                # Computar régimen con todos los datos (EMA200/MACD necesita warmup)
                _regimen_df = compute_regime_mask_1d(_df_1m_full, indicador=REGIMEN_INDICADOR)
                
                # Filtrar al periodo del backtest para contar stats
                if fecha_inicio and fecha_fin:
                    from .types import filter_by_date as _fbd_reg
                    _regimen_period = _fbd_reg(_regimen_df, fecha_inicio, fecha_fin)
                else:
                    _regimen_period = _regimen_df
                
                _regimen_dias_totales = len(_regimen_period)

                if _regimen_mode == "all":
                    _regimen_dias_operables = _regimen_dias_totales
                elif _regimen_mode == "odd":
                    _regimen_dias_operables = _regimen_period.filter(
                        pl.col("regime") == "ALCISTA"
                    ).height
                else:
                    _regimen_dias_operables = _regimen_period.filter(
                        pl.col("regime") == REGIMEN_TIPO.upper()
                    ).height
                
                # Asegurar 1m en cache para el backtest
                if "1m" not in tf_cache:
                    _df_1m_for_cache = tf_cache.get("1m")
                    if _df_1m_for_cache is None and fecha_inicio and fecha_fin:
                        _df_1m_for_cache = _fbd_reg(_df_1m_full, fecha_inicio, fecha_fin)
                    tf_cache["1m"] = _df_1m_for_cache if _df_1m_for_cache is not None else _df_1m_full
                
                del _df_1m_full  # Liberar memoria
        except Exception as _e:
            logger.warning(f"⚠ No se pudo computar régimen: {_e}")

    if mostrar_cabecera_func:
        # time_bars: salidas en el mismo TF de entrada (cuenta barras, no usa 1m)
        _tf_exit_display = tf_display if str(exit_type).upper() == "BARS" else "1m"
        
        mostrar_cabecera_func(
            activo=activo,
            combo_nombre=strategy_name,
            indicadores=indicadores,
            n_trials=n_trials,
            strategy_id=getattr(strategy, "combinacion_id", None),
            archivo_data=archivo_data,
            timeframe=tf_display,
            timeframe_exit=_tf_exit_display,
            periodo=periodo_datos,
            exit_type=exit_type,
            strategy_exit_enabled=strategy_exit_enabled,
            sampler_type=optuna_sampler,
            synthetic_mode=synthetic_mode,
            synthetic_years=synthetic_years,
            perturbacion=ENTRENAMIENTO_ROBUSTO_ACTIVAR,
            regimen_activo=_regimen_enabled,
            regimen_tipo=REGIMEN_TIPO,
            regimen_dias_operables=_regimen_dias_operables,
            regimen_dias_totales=_regimen_dias_totales,
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
            try:
                from general.configuracion import (
                    FECHA_INICIO as _FI, FECHA_FIN as _FF,
                    COMPARATIVA_PERIODOS_ACTIVAR as _CPA, COMPARATIVA_PERIODO as _CP,
                    CARPETA_DATOS as _CDAT, FORMATO_DATOS as _FMT,
                )
            except Exception:
                _FI = _FF = None; _CDAT = "datos"; _FMT = "feather"
                _CPA = True; _CP = "1mes"
            reporters.append(ExcelReporter(
                resumen_path=f"{excel_dir}/RESUMEN ID{strategy.combinacion_id}.xlsx",
                trades_base_dir=excel_dir,
                max_archivos=max_archivos,
                fecha_inicio=_FI,
                fecha_fin=_FF,
                datos_dir=_CDAT,
                formato_datos=_FMT,
                comparativa_periodos_activar=_CPA,
                comparativa_periodo=_CP,
                regimen_activo=_regimen_enabled,
                regimen_tipo=REGIMEN_TIPO,
                regimen_buy_hold_activo=bool(REGIMEN_BUY_HOLD_ACTIVO),
                regimen_buy_hold_saldo=str(REGIMEN_BUY_HOLD_SALDO),
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
    # Reinicio limpio opcional de DB Optuna (evita reutilizar estudios previos)
    try:
        from general.configuracion import OPTUNA_RESET_DB_ON_START, OPTUNA_CREATE_DATABASE, GUARDAR_EQUITY_EN_DB
    except ImportError:
        OPTUNA_RESET_DB_ON_START = False
        OPTUNA_CREATE_DATABASE = True
        GUARDAR_EQUITY_EN_DB = False

    # El reinicio de la DB ahora se hace a nivel global en ejecutar.py
    # para evitar borrar el progreso cuando se corren múltiples activos o salidas.

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
            OPTUNA_CREATE_DATABASE,
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
        create_database=bool(locals().get("OPTUNA_CREATE_DATABASE", True)),
        sampler=optuna_sampler,
        guardar_equity_en_db=bool(locals().get("GUARDAR_EQUITY_EN_DB", False)),
    )

    runner.activo = activo

    # 5c. CONFIGURAR RÉGIMEN DE MERCADO
    runner.regimen_activo = _regimen_enabled
    runner.regimen_modo = _regimen_mode
    runner.regimen_tipo = REGIMEN_TIPO
    runner.regimen_indicador = REGIMEN_INDICADOR
    runner.regimen_buy_hold_activo = bool(REGIMEN_BUY_HOLD_ACTIVO)
    runner.regimen_buy_hold_saldo = REGIMEN_BUY_HOLD_SALDO
    runner.regimen_df = _regimen_df
    runner.regimen_dias_operables = _regimen_dias_operables

    try:
        from modelox.core.types import suffix_to_minutes
        
        # Carga diferida de timeframes extra
        entry_tf = getattr(strategy, "timeframe_entry", None) or timeframe_base
        # FORZAR salidas SIEMPRE en 1m para máxima precisión
        exit_tf = "1m"
        needed_tfs = [timeframe_base, entry_tf, exit_tf]
        
        # ================================================================
        # SIEMPRE CARGAR DATOS DE 1M PARA SALIDAS PRECISAS
        # ================================================================
        # SIEMPRE necesitamos datos de 1m para evaluar las salidas (SL/TP/Trailing/Custom)
        # con precisión tick-a-tick, incluso si el timeframe base es 1m.
        if "1m" not in needed_tfs:
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
                        valid_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
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
                        total_trials=len(study.get_trials(deepcopy=False)),
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
        
        # Base de datos Optuna Híbrida conservada para su uso en optuna-dashboard
        if optuna_sampler.upper() == "HYBRID":
            db_path = get_database_file_path("optuna_hybrid.db", migrate_from_root=True)
            if os.path.exists(db_path):
                logger.info("✓ Base de datos Optuna Híbrida conservada en: " + db_path)

