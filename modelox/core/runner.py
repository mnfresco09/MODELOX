"""
================================================================================
MODELOX/CORE/RUNNER.PY — RUNNER PRINCIPAL DE OPTIMIZACIÓN
================================================================================

PROPÓSITO:
    Orquesta la optimización bayesiana con Optuna, soportando múltiples
    samplers (CMA-ES, TPE, GT, ML, QMC) y perturbación de datos.

CONTENIDO:
     1. CACHÉ DE SEÑALES          — Reutilización entre trials vecinos
     2. CONFIGURACIÓN             — OptunaConfig, PerturbationConfig
     3. PERTURBACIÓN              — Validación/coherencia OHLCV, kernel Numba
     4. HELPERS                   — create_study_for_strategy
     5. PIPELINE                  — DataLoader, SignalGenerator, BacktestEngine
     6. RUNNER PRINCIPAL          — OptimizationRunner (optimize_strategies)

MODOS DE OPERACIÓN:
    1. NORMAL:       Cada trial usa datos ORIGINALES.
    2. PERTURBACIÓN: Cada trial usa datos PERTURBADOS (validates robustness).

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
    CMAScorer, TPEScorer, GTScorer, MLForestScorer, QMCScorer,
    score_cma, score_tpe, score_gt, score_ml, score_qmc,
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


@dataclass
class PerturbationConfig:
    """
    Configuración de perturbación de datos.
    
    NOTA: Los valores se configuran en general/configuracion.py
          con las variables PERTURBACION_*.
    """
    enabled: bool = False
    method: str = "returns_perturbation"
    noise_factor: float = 0.3
    seed: Optional[int] = 42
    verify_perturbation: bool = True


# =============================================================================
# 3. PERTURBACIÓN (VALIDACIÓN/COHERENCIA OHLCV + KERNEL NUMBA)
# =============================================================================

def _validate_ohlcv_coherence(df: pl.DataFrame) -> Tuple[bool, str]:
    """
    Valida que los datos OHLCV sean coherentes.
    
    Reglas:
    - High >= max(Open, Close)
    - Low <= min(Open, Close)
    - Low <= High
    - Todos los precios > 0
    
    Returns:
        (is_valid, error_message)
    """
    open_arr = df["open"].to_numpy()
    high_arr = df["high"].to_numpy()
    low_arr = df["low"].to_numpy()
    close_arr = df["close"].to_numpy()
    
    max_oc = np.maximum(open_arr, close_arr)
    min_oc = np.minimum(open_arr, close_arr)
    
    # Verificar coherencia
    high_valid = np.all(high_arr >= max_oc - 1e-10)
    low_valid = np.all(low_arr <= min_oc + 1e-10)
    hl_valid = np.all(low_arr <= high_arr + 1e-10)
    positive = np.all(open_arr > 0) and np.all(high_arr > 0) and np.all(low_arr > 0) and np.all(close_arr > 0)
    
    if not high_valid:
        return False, "High < max(Open, Close) en algunas velas"
    if not low_valid:
        return False, "Low > min(Open, Close) en algunas velas"
    if not hl_valid:
        return False, "Low > High en algunas velas"
    if not positive:
        return False, "Precios negativos o cero detectados"
    
    return True, "OK"


def _ensure_ohlcv_coherence(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Asegura coherencia OHLCV después de perturbación.
    
    Reglas aplicadas:
    1. High >= max(Open, Close)
    2. Low <= min(Open, Close)
    3. Low <= High
    4. Todos los precios > min_price
    """
    min_price = 0.01
    
    # Asegurar precios positivos
    open_arr = np.maximum(open_arr, min_price)
    high_arr = np.maximum(high_arr, min_price)
    low_arr = np.maximum(low_arr, min_price)
    close_arr = np.maximum(close_arr, min_price)
    
    # Asegurar coherencia
    max_oc = np.maximum(open_arr, close_arr)
    min_oc = np.minimum(open_arr, close_arr)
    
    high_arr = np.maximum(high_arr, max_oc)
    low_arr = np.minimum(low_arr, min_oc)
    low_arr = np.minimum(low_arr, high_arr)
    
    return open_arr, high_arr, low_arr, close_arr


# =============================================================================
# 3b. KERNEL NUMBA PARA PERTURBACIÓN RÁPIDA
# =============================================================================
try:
    from numba import njit
    
    @njit(cache=True, fastmath=True)
    def _perturb_returns_numba(
        close_arr: np.ndarray,
        noise: np.ndarray,
    ) -> np.ndarray:
        """Kernel Numba para reconstruir precios desde retornos perturbados."""
        n = len(close_arr)
        log_returns = np.empty(n - 1, dtype=np.float64)
        
        # Calcular log returns
        for i in range(n - 1):
            log_returns[i] = np.log(close_arr[i + 1] / close_arr[i])
        
        # Añadir ruido y reconstruir
        perturbed_returns = log_returns + noise
        
        new_close = np.empty(n, dtype=np.float64)
        new_close[0] = close_arr[0]
        cumsum = 0.0
        for i in range(n - 1):
            cumsum += perturbed_returns[i]
            new_close[i + 1] = close_arr[0] * np.exp(cumsum)
        
        return new_close
    
    _NUMBA_PERTURB_AVAILABLE = True
except Exception:
    _NUMBA_PERTURB_AVAILABLE = False


def perturb_returns_professional(
    df: pl.DataFrame,
    noise_factor: float = 0.3,
    seed: Optional[int] = None,
) -> pl.DataFrame:
    """
    PERTURBACIÓN PROFESIONAL DE RETORNOS (Método Quant Estándar)
    
    OPTIMIZADO: Usa kernel Numba para cálculos intensivos.
    """
    rng = np.random.default_rng(seed)
    
    # Extraer arrays originales (views, no copias)
    close_arr = df["close"].to_numpy()
    n = len(close_arr)
    
    if n < 10:
        return df
    
    # Calcular volatilidad
    log_returns = np.diff(np.log(np.maximum(close_arr, 1e-10)))
    volatility = np.std(log_returns)
    
    if volatility < 1e-10:
        return df
    
    # Generar ruido
    noise_std = volatility * noise_factor
    noise = rng.normal(0, noise_std, len(log_returns))
    
    # Reconstruir close (usar Numba si disponible)
    if _NUMBA_PERTURB_AVAILABLE:
        new_close = _perturb_returns_numba(close_arr.astype(np.float64), noise)
    else:
        perturbed_returns = log_returns + noise
        new_close = np.zeros(n)
        new_close[0] = close_arr[0]
        new_close[1:] = close_arr[0] * np.exp(np.cumsum(perturbed_returns))
    
    # Escalar OHLC
    scale = new_close / np.maximum(close_arr, 1e-10)
    
    open_arr = df["open"].to_numpy() * scale
    high_arr = df["high"].to_numpy() * scale
    low_arr = df["low"].to_numpy() * scale
    
    # Asegurar coherencia
    new_open, new_high, new_low, new_close = _ensure_ohlcv_coherence(
        open_arr, high_arr, low_arr, new_close
    )
    
    result = df.with_columns([
        pl.Series("open", new_open),
        pl.Series("high", new_high),
        pl.Series("low", new_low),
        pl.Series("close", new_close),
    ])
    
    return result


def apply_perturbation(
    df: pl.DataFrame,
    config: PerturbationConfig,
    trial_number: int,
) -> Tuple[pl.DataFrame, int, Dict[str, Any]]:
    """
    Aplica perturbación a los datos según la configuración.
    
    Returns:
        (df_perturbado, seed_usado, info_perturbacion)
    """
    if not config.enabled:
        return df, 0, {"perturbation_applied": False}
    
    # Semilla única por trial
    seed = (config.seed or 42) + trial_number
    
    # Guardar estadísticas originales para verificación
    original_close_mean = float(df["close"].mean())
    original_close_std = float(df["close"].std())
    
    # Aplicar perturbación (único método: returns_perturbation)
    df_perturbed = perturb_returns_professional(df, config.noise_factor, seed)
    
    # Verificar que la perturbación se aplicó
    perturbed_close_mean = float(df_perturbed["close"].mean())
    perturbed_close_std = float(df_perturbed["close"].std())
    
    # Calcular diferencia relativa
    mean_diff_pct = abs(perturbed_close_mean - original_close_mean) / original_close_mean * 100
    
    info = {
        "perturbation_applied": True,
        "method": "returns_perturbation",
        "seed": seed,
        "noise_factor": config.noise_factor,
        "original_close_mean": original_close_mean,
        "perturbed_close_mean": perturbed_close_mean,
        "mean_diff_pct": mean_diff_pct,
        "original_close_std": original_close_std,
        "perturbed_close_std": perturbed_close_std,
    }
    
    # Verificar coherencia OHLCV
    is_valid, error_msg = _validate_ohlcv_coherence(df_perturbed)
    info["ohlcv_valid"] = is_valid
    if not is_valid:
        info["ohlcv_error"] = error_msg
        # Si no es válido, intentar corregir
        open_arr = df_perturbed["open"].to_numpy()
        high_arr = df_perturbed["high"].to_numpy()
        low_arr = df_perturbed["low"].to_numpy()
        close_arr = df_perturbed["close"].to_numpy()
        
        open_arr, high_arr, low_arr, close_arr = _ensure_ohlcv_coherence(
            open_arr, high_arr, low_arr, close_arr
        )
        
        df_perturbed = df_perturbed.with_columns([
            pl.Series("open", open_arr),
            pl.Series("high", high_arr),
            pl.Series("low", low_arr),
            pl.Series("close", close_arr),
        ])
        info["ohlcv_corrected"] = True
    
    return df_perturbed, seed, info


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
    ) -> Tuple[pl.DataFrame, List[float], Dict[str, Any]]:
        backtest_params = BacktestParams.from_config_and_params(config, params)
        timeframe = params.get("__timeframe_base", "1m")
        
        trades_df, equity_curve = calculate_performance_vectorized_numba(
            df=df, signals=signals, params=backtest_params, strategy=strategy,
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
    - Backtesting normal (sin perturbación)
    - Perturbación de datos para validación
    - Scoring institucional multiplicativo (PSR, DSR, K-Ratio, etc.)
    - Datos futuros con rango distinto por trial (FuturoTrialDataProvider)
    """
    
    config: BacktestConfig
    n_trials: int
    reporters: Sequence[Reporter]
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    activo: Optional[str] = None
    
    # Configuración de perturbación
    perturbation_config: PerturbationConfig = field(default_factory=PerturbationConfig)
    
    # Proveedor de datos futuros (opcional)
    futuro_data_provider: Optional[Any] = None
    
    # Estado interno
    _last_study: Optional[optuna.study.Study] = None
    _perturbation_stats: Dict[str, Any] = field(default_factory=dict)
    
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
            
            # Mostrar resumen de perturbación si estaba habilitada
            if self.perturbation_config.enabled:
                self._show_perturbation_summary()
            
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
        base_tf = normalize_timeframe_to_suffix(base_timeframe or "1m")
        df_map = df_by_timeframe or {base_tf: df}
        df_base = df_map.get(base_tf, df)
        
        # Reset stats de perturbación
        self._perturbation_stats = {
            "enabled": self.perturbation_config.enabled,
            "method": "returns_perturbation",  # Único método soportado
            "trials_perturbed": 0,
            "mean_diff_pcts": [],
        }
        
        objective = self._create_single_objective(df_base, df_map, strategy, base_tf)
        
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
            t0_total = time.perf_counter()
            
            # LIMPIEZA PERIÓDICA para mantener velocidad constante
            periodic_cleanup(trial.number)
            
            params_rt = self._prepare_params(trial, strategy, base_tf)
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
            # PERTURBACIÓN DE DATOS (SOLO SI NO HAY FUTURO PROVIDER)
            # ================================================================
            # Si usamos FuturoTrialDataProvider, cada trial ya tiene datos
            # diferentes, por lo que NO aplicamos perturbación adicional
            df_trial = df_entry
            df_map_perturbed = df_map_trial  # Por defecto, usar el del trial
            perturb_info = {"perturbation_applied": False}
            perturb_seed = 0
            
            # Solo perturbar si NO hay futuro_data_provider activo
            if self.perturbation_config.enabled and self.futuro_data_provider is None:
                # Calcular semilla única para este trial
                base_seed = self.perturbation_config.seed or 42
                perturb_seed = base_seed + trial.number
                
                # Perturbar TODOS los timeframes con la misma semilla
                df_map_perturbed = {}
                for tf_key, tf_df in df_map_trial.items():
                    perturbed_df, _, tf_perturb_info = apply_perturbation(
                        tf_df, self.perturbation_config, trial.number
                    )
                    df_map_perturbed[tf_key] = perturbed_df
                    
                    # Guardar info del timeframe de entrada
                    if tf_key == entry_tf:
                        df_trial = perturbed_df
                        perturb_info = tf_perturb_info
                
                self._perturbation_stats["trials_perturbed"] += 1
                if "mean_diff_pct" in perturb_info:
                    diffs = self._perturbation_stats["mean_diff_pcts"]
                    diffs.append(perturb_info["mean_diff_pct"])
                    if len(diffs) > 100:
                        self._perturbation_stats["mean_diff_pcts"] = diffs[-100:]
            
            # Generar señales
            t1_signals = time.perf_counter()
            signals_df = SignalGenerator.generate_signals(df_trial, strategy, params_rt, df_map_perturbed)
            t2_signals = time.perf_counter()
            
            # Ejecutar backtest
            t1_backtest = time.perf_counter()
            trades_df, equity_curve, metrics = BacktestEngine.run_backtest(
                df_trial, signals_df, self.config, params_rt, strategy,
            )
            t2_backtest = time.perf_counter()
            
            if trades_df.is_empty():
                return 0.0
            
            trial.set_user_attr("metricas", metrics)
            # Usar scoring correspondiente al sampler elegido
            score_func = self._get_score_func()
            score = float(score_func(metrics, trial=trial))
            
            t_total = time.perf_counter() - t0_total
            
            if _TIMINGS_VERBOSE and (trial.number % _TIMINGS_PRINT_EVERY == 0):
                perturb_str = f" [P]" if perturb_info.get("perturbation_applied", False) else ""
                print(
                    f"  ⏱ TRIAL {trial.number:3d}{perturb_str} │ "
                    f"signals {(t2_signals - t1_signals)*1000:6.1f}ms │ "
                    f"backtest {(t2_backtest - t1_backtest)*1000:6.1f}ms │ "
                    f"total {t_total*1000:6.1f}ms │ "
                    f"trades {len(trades_df):5d}"
                )
            
            # Crear artifacts
            df_signals_for_artifacts = None
            df_for_artifacts = df_base_trial if self.futuro_data_provider is not None else df_base
            for reporter in self.reporters:
                if hasattr(reporter, "needs_dataframe") and reporter.needs_dataframe(score):
                    ohlc_cols = ["timestamp", "open", "high", "low", "close", "volume"]
                    base_cols = [c for c in ohlc_cols if c in df_for_artifacts.columns]
                    signal_cols = [c for c in signals_df.columns if c not in base_cols]
                    df_signals_for_artifacts = df_for_artifacts.select(base_cols).hstack(
                        signals_df.select(signal_cols)
                    )
                    break
            
            # Calcular rango de fechas real del trial (para plots dinámicos)
            _trial_date_range = None
            if self.futuro_data_provider is not None and "timestamp" in df_base_trial.columns:
                try:
                    _ts = df_base_trial["timestamp"]
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
                trades=trades_df.to_pandas(),
                equity_curve=equity_curve,
                indicators_used=params_rt.get("__indicators_used", []),
                perturbado=perturb_info.get("perturbation_applied", False),
                perturb_seed=perturb_seed if perturb_info.get("perturbation_applied", False) else None,
                trial_date_range=_trial_date_range,
            )
            
            for reporter in self.reporters:
                reporter.on_trial_end(artifacts)
            
            return score
        
        return objective
    
    def _show_perturbation_summary(self):
        """Muestra resumen de perturbación al final."""
        if not self._perturbation_stats.get("enabled", False):
            return
        
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            
            console = Console()
            
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Métrica", style="white")
            table.add_column("Valor", justify="right", style="green")
            
            table.add_row("Método", self._perturbation_stats.get("method", "unknown"))
            table.add_row("Trials Perturbados", str(self._perturbation_stats.get("trials_perturbed", 0)))
            
            diffs = self._perturbation_stats.get("mean_diff_pcts", [])
            if diffs:
                avg_diff = np.mean(diffs)
                table.add_row("Divergencia Promedio", f"{avg_diff:.2f}%")
            
            panel = Panel(
                table,
                title="📊 RESUMEN DE PERTURBACIÓN DE DATOS",
                border_style="blue",
            )
            
            console.print()
            console.print(panel)
            
        except Exception:
            pass  # No fallar si Rich no está disponible


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
    perturbacion_activar: bool = False,
    perturbacion_config: dict = None,
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
    fecha_inicio_plot: str = "2025-01-01",
    fecha_fin_plot: str = "2025-01-20",
    # Configuración de gráficos (subrango dentro del trial)
    plot_meses_duracion: int = 2,
    plot_ubicacion_aleatoria: bool = True,
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
        perturbacion_activar: Si activar perturbación de datos
        perturbacion_config: Dict con configuración de perturbación
        excel_dir: Directorio para guardar Excel
        graficos_dir: Directorio para guardar gráficos
        usar_excel: Si generar reportes Excel
        generar_plots: Si generar gráficos HTML
        max_archivos: Máximo de archivos a guardar
        fecha_inicio_plot: Fecha inicio para plots
        fecha_fin_plot: Fecha fin para plots
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
            perturbacion_activar=perturbacion_activar,
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
                resumen_path=f"{excel_dir}/RESUMEN.xlsx",
                trades_base_dir=excel_dir,
                max_archivos=max_archivos,
            ))

        if generar_plots and graficos_dir:
            from visual.grafico import PlotReporter
            
            reporters.append(PlotReporter(
                plot_base=graficos_dir,
                fecha_inicio_plot=fecha_inicio_plot,
                fecha_fin_plot=fecha_fin_plot,
                plot_meses_duracion=plot_meses_duracion,
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

    runner.optuna = OptunaConfig(
        seed=None, 
        n_jobs=1, 
        storage=None,
        sampler=optuna_sampler,
    )

    # Configurar perturbación
    perturbacion_config = perturbacion_config or {}
    runner.perturbation_config = PerturbationConfig(
        enabled=perturbacion_activar,
        noise_factor=float(perturbacion_config.get("noise_scale", 0.5)),
        seed=int(perturbacion_config.get("seed", 42)),
    )

    runner.activo = activo

    try:
        # Carga diferida de timeframes extra
        entry_tf = getattr(strategy, "timeframe_entry", None) or timeframe_base
        exit_tf = getattr(strategy, "timeframe_exit", None) or timeframe_base
        needed_tfs = [timeframe_base, entry_tf, exit_tf]

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
                    mostrar_fin_func(
                        total_trials=len(study.trials),
                        best_score=best_val,
                        best_trial=best_trial.number,
                        estrategia=strategy_name,
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


# =============================================================================
# ALIAS DE COMPATIBILIDAD
# =============================================================================

# Mantener el nombre anterior para compatibilidad
MonteCarloRunner = OptimizationRunner
MonteCarloConfig = PerturbationConfig
