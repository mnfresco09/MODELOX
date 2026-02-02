"""
# =============================================================================
#
#     ██████╗ ██╗   ██╗███╗   ██╗███╗   ██╗███████╗██████╗
#     ██╔══██╗██║   ██║████╗  ██║████╗  ██║██╔════╝██╔══██╗
#     ██████╔╝██║   ██║██╔██╗ ██║██╔██╗ ██║█████╗  ██████╔╝
#     ██╔══██╗██║   ██║██║╚██╗██║██║╚██╗██║██╔══╝  ██╔══██╗
#     ██║  ██║╚██████╔╝██║ ╚████║██║ ╚████║███████╗██║  ██║
#     ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝
#
#     RUNNER.PY - ORQUESTADOR PRINCIPAL DE OPTIMIZACIÓN
#
# =============================================================================
#
#     MODOS DE OPERACIÓN:
#     - NORMAL: Cada trial usa datos originales
#     - PERTURBACIÓN: Cada trial usa datos perturbados (anti-overfitting)
#
#     SAMPLERS:
#     - "CMA": CMA-ES (recomendado para scoring institucional)
#     - "TPE": Tree-structured Parzen Estimator (clásico)
#
#     PERTURBACIONES DISPONIBLES:
#     - returns_perturbation: Ruido calibrado a volatilidad (RECOMENDADO)
#     - block_bootstrap: Mantiene autocorrelación
#     - stationary_bootstrap: Politis & Romano 1994
#     - returns_shuffle: Rompe autocorrelación
#
# =============================================================================
"""

from __future__ import annotations

import gc
import os
import re
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import optuna
import polars as pl
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import TPESampler, CmaEsSampler

# GPSampler (Gaussian Process) - Optuna v4.0+
try:
    from optuna.samplers import GPSampler
    _GP_AVAILABLE = True
except ImportError:
    _GP_AVAILABLE = False

# BoTorchSampler alternativo (requiere botorch instalado)
try:
    from optuna_integration import BoTorchSampler
    _BOTORCH_AVAILABLE = True
except ImportError:
    _BOTORCH_AVAILABLE = False

from .engine import BacktestParams, calculate_performance_vectorized_numba
from .metrics import resumen_metricas
from .scoring import (
    score_optuna, 
    set_study_for_scorer,
)
from .types import (
    BacktestConfig,
    Reporter,
    Strategy,
    TrialArtifacts,
    normalize_timeframe_to_suffix,
)
from .data import prepare_multitimeframe_data
from .exits import resolve_exit_settings_for_trial


# =============================================================================
# 1. CONFIGURACIÓN GLOBAL
# =============================================================================

# SILENCIAR WARNINGS DE OPTUNA
warnings.filterwarnings("ignore", category=ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# VARIABLES DE ENTORNO PARA DEBUG
_TIMINGS_VERBOSE: bool = os.environ.get("MODELOX_TIMINGS_VERBOSE", "0") in {"1", "true", "True", "YES", "yes"}
_TIMINGS_PRINT_EVERY: int = int(os.environ.get("MODELOX_TIMINGS_PRINT_EVERY", "1"))
_CLEANUP_INTERVAL: int = int(os.environ.get("MODELOX_CLEANUP_INTERVAL", "100"))


# =============================================================================
# 2. CACHÉ DE SEÑALES PARA REUTILIZACIÓN
# =============================================================================

_SIGNALS_CACHE: Dict[str, pl.DataFrame] = {}
_SIGNALS_CACHE_MAX: int = 8


def _cache_signals(key: str, signals: pl.DataFrame) -> None:
    """GUARDA SEÑALES EN CACHÉ PARA REUTILIZACIÓN."""
    global _SIGNALS_CACHE
    if len(_SIGNALS_CACHE) >= _SIGNALS_CACHE_MAX:
        _SIGNALS_CACHE.pop(next(iter(_SIGNALS_CACHE)))
    _SIGNALS_CACHE[key] = signals


def _get_cached_signals(key: str) -> Optional[pl.DataFrame]:
    """OBTIENE SEÑALES DEL CACHÉ."""
    return _SIGNALS_CACHE.get(key)


def clear_all_caches() -> None:
    """LIMPIA TODOS LOS CACHÉS DEL SISTEMA."""
    global _SIGNALS_CACHE
    _SIGNALS_CACHE.clear()
    gc.collect()


def periodic_cleanup(trial_number: int, force: bool = False) -> None:
    """LIMPIEZA PERIÓDICA CADA N TRIALS.
    
    EJECUTA:
    - Limpia cachés de señales
    - Fuerza garbage collection
    - En macOS, libera memoria comprimida
    """
    if not force and trial_number % _CLEANUP_INTERVAL != 0:
        return
    
    if trial_number == 0:
        return
    
    clear_all_caches()
    
    gc.collect()
    gc.collect()
    gc.collect()
    
    import platform
    if platform.system() == "Darwin":
        try:
            import subprocess
            subprocess.run(['purge'], capture_output=True, timeout=2)
        except Exception:
            pass


# =============================================================================
# 3. DATACLASSES DE CONFIGURACIÓN
# =============================================================================

@dataclass(frozen=True)
class OptunaConfig:
    """CONFIGURACIÓN DE OPTUNA.
    
    ATRIBUTOS:
    - seed: Semilla para reproducibilidad
    - n_jobs: Trials paralelos (1=secuencial, -1=todos los cores)
    - sampler: Algoritmo de muestreo:
        * "CMA" (recomendado) - CMA-ES: Estrategia evolutiva adaptativa
        * "TPE" - Tree-Parzen Estimator: Bayesiano con árboles
        * "GP" - Gaussian Process: Optimización Bayesiana clásica (Optuna v4.0+)
                 Equilibra exploración/explotación vía incertidumbre del modelo GP
                 Mejor para espacios pequeños (<20 dimensiones)
        * "BOTORCH" - GP con backend BoTorch (requiere pip install botorch optuna-integration)
                 Soporta restricciones complejas y optimización multiobjetivo
    """
    seed: Optional[int] = None
    n_jobs: int = 1
    storage: Optional[str] = None
    study_name_prefix: str = "MODELOX"
    sampler: str = "CMA"


@dataclass
class PerturbationConfig:
    """CONFIGURACIÓN DEL SISTEMA DE PERTURBACIÓN.
    
    MÉTODOS:
    - returns_perturbation: Ruido gaussiano calibrado (RECOMENDADO)
    - block_bootstrap: Mantiene autocorrelación
    - stationary_bootstrap: Politis & Romano 1994
    - returns_shuffle: Shuffle completo
    """
    enabled: bool = False
    method: str = "returns_perturbation"
    noise_factor: float = 0.3
    block_size: int = 100
    seed: Optional[int] = 42
    verify_perturbation: bool = True


# =============================================================================
# 4. FUNCIONES DE PERTURBACIÓN PROFESIONALES
# =============================================================================

def _validate_ohlcv_coherence(df: pl.DataFrame) -> Tuple[bool, str]:
    """VALIDA COHERENCIA OHLCV.
    
    REGLAS:
    - High >= max(Open, Close)
    - Low <= min(Open, Close)
    - Low <= High
    - Todos los precios > 0
    """
    open_arr = df["open"].to_numpy()
    high_arr = df["high"].to_numpy()
    low_arr = df["low"].to_numpy()
    close_arr = df["close"].to_numpy()
    
    max_oc = np.maximum(open_arr, close_arr)
    min_oc = np.minimum(open_arr, close_arr)
    
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
    """ASEGURA COHERENCIA OHLCV DESPUÉS DE PERTURBACIÓN."""
    min_price = 0.01
    
    open_arr = np.maximum(open_arr, min_price)
    high_arr = np.maximum(high_arr, min_price)
    low_arr = np.maximum(low_arr, min_price)
    close_arr = np.maximum(close_arr, min_price)
    
    max_oc = np.maximum(open_arr, close_arr)
    min_oc = np.minimum(open_arr, close_arr)
    
    high_arr = np.maximum(high_arr, max_oc)
    low_arr = np.minimum(low_arr, min_oc)
    low_arr = np.minimum(low_arr, high_arr)
    
    return open_arr, high_arr, low_arr, close_arr


# =============================================================================
# 5. KERNEL NUMBA PARA PERTURBACIÓN
# =============================================================================

try:
    from numba import njit
    
    @njit(cache=True, fastmath=True)
    def _perturb_returns_numba(
        close_arr: np.ndarray,
        noise: np.ndarray,
    ) -> np.ndarray:
        """KERNEL NUMBA: RECONSTRUYE PRECIOS DESDE RETORNOS PERTURBADOS."""
        n = len(close_arr)
        log_returns = np.empty(n - 1, dtype=np.float64)
        
        for i in range(n - 1):
            log_returns[i] = np.log(close_arr[i + 1] / close_arr[i])
        
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
    """PERTURBACIÓN PROFESIONAL DE RETORNOS (MÉTODO QUANT ESTÁNDAR)."""
    rng = np.random.default_rng(seed)
    
    close_arr = df["close"].to_numpy()
    n = len(close_arr)
    
    if n < 10:
        return df
    
    log_returns = np.diff(np.log(np.maximum(close_arr, 1e-10)))
    volatility = np.std(log_returns)
    
    if volatility < 1e-10:
        return df
    
    noise_std = volatility * noise_factor
    noise = rng.normal(0, noise_std, len(log_returns))
    
    if _NUMBA_PERTURB_AVAILABLE:
        new_close = _perturb_returns_numba(close_arr.astype(np.float64), noise)
    else:
        perturbed_returns = log_returns + noise
        new_close = np.zeros(n)
        new_close[0] = close_arr[0]
        new_close[1:] = close_arr[0] * np.exp(np.cumsum(perturbed_returns))
    
    scale = new_close / np.maximum(close_arr, 1e-10)
    
    open_arr = df["open"].to_numpy() * scale
    high_arr = df["high"].to_numpy() * scale
    low_arr = df["low"].to_numpy() * scale
    
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


def perturb_block_bootstrap(
    df: pl.DataFrame,
    block_size: int = 100,
    seed: Optional[int] = None,
) -> pl.DataFrame:
    """
    BLOCK BOOTSTRAP sobre retornos.
    
    BLOCK BOOTSTRAP SOBRE RETORNOS.
    
    PRESERVA AUTOCORRELACIÓN DE CORTO PLAZO DENTRO DE BLOQUES.
    """
    rng = np.random.default_rng(seed)
    
    open_arr = df["open"].to_numpy().astype(np.float64)
    high_arr = df["high"].to_numpy().astype(np.float64)
    low_arr = df["low"].to_numpy().astype(np.float64)
    close_arr = df["close"].to_numpy().astype(np.float64)
    n = len(close_arr)
    
    if n < block_size * 2:
        # Fallback a perturbación de retornos si muy pocos datos
        return perturb_returns_professional(df, 0.3, seed)
    
    # Calcular retornos y ratios OHLC
    log_returns = np.diff(np.log(np.maximum(close_arr, 1e-10)))
    n_returns = len(log_returns)
    
    # Calcular ratios para preservar forma de velas
    open_ratio = open_arr / np.maximum(close_arr, 1e-10)
    high_ratio = high_arr / np.maximum(close_arr, 1e-10)
    low_ratio = low_arr / np.maximum(close_arr, 1e-10)
    
    # Número de bloques disponibles
    n_blocks = max(1, n_returns - block_size + 1)
    
    # Muestrear bloques
    sampled_indices = []
    while len(sampled_indices) < n_returns:
        block_start = rng.integers(0, n_blocks)
        for j in range(block_size):
            if len(sampled_indices) >= n_returns:
                break
            idx = block_start + j
            if idx < n_returns:
                sampled_indices.append(idx)
    
    sampled_indices = np.array(sampled_indices[:n_returns])
    
    # Reconstruir precios
    sampled_returns = log_returns[sampled_indices]
    new_close = np.zeros(n)
    new_close[0] = close_arr[0]
    new_close[1:] = close_arr[0] * np.exp(np.cumsum(sampled_returns))
    
    # Reconstruir OHLC con ratios
    new_open = np.zeros(n)
    new_high = np.zeros(n)
    new_low = np.zeros(n)
    
    new_open[0] = new_close[0] * open_ratio[0]
    new_high[0] = new_close[0] * high_ratio[0]
    new_low[0] = new_close[0] * low_ratio[0]
    
    for i in range(1, n):
        orig_idx = min(sampled_indices[i - 1] + 1, n - 1)
        new_open[i] = new_close[i] * open_ratio[orig_idx]
        new_high[i] = new_close[i] * high_ratio[orig_idx]
        new_low[i] = new_close[i] * low_ratio[orig_idx]
    
    # Suavizar gaps entre bloques
    for i in range(1, n):
        gap = new_open[i] / new_close[i - 1] if new_close[i - 1] > 0 else 1.0
        if gap > 1.05 or gap < 0.95:
            new_open[i] = new_close[i - 1] * (1 + (gap - 1) * 0.2)
    
    # Asegurar coherencia
    new_open, new_high, new_low, new_close = _ensure_ohlcv_coherence(
        new_open, new_high, new_low, new_close
    )
    
    return df.with_columns([
        pl.Series("open", new_open),
        pl.Series("high", new_high),
        pl.Series("low", new_low),
        pl.Series("close", new_close),
    ])


def perturb_returns_shuffle(
    df: pl.DataFrame,
    seed: Optional[int] = None,
) -> pl.DataFrame:
    """SHUFFLE DE RETORNOS (ROMPE AUTOCORRELACIÓN COMPLETAMENTE)."""
    rng = np.random.default_rng(seed)
    
    close_arr = df["close"].to_numpy().astype(np.float64)
    n = len(close_arr)
    
    # Calcular retornos
    log_returns = np.diff(np.log(np.maximum(close_arr, 1e-10)))
    
    # Shuffle completo
    rng.shuffle(log_returns)
    
    # Reconstruir
    new_close = np.zeros(n)
    new_close[0] = close_arr[0]
    new_close[1:] = close_arr[0] * np.exp(np.cumsum(log_returns))
    
    # Escalar OHLC
    scale = new_close / np.maximum(close_arr, 1e-10)
    
    open_arr = df["open"].to_numpy() * scale
    high_arr = df["high"].to_numpy() * scale
    low_arr = df["low"].to_numpy() * scale
    
    open_arr, high_arr, low_arr, new_close = _ensure_ohlcv_coherence(
        open_arr, high_arr, low_arr, new_close
    )
    
    return df.with_columns([
        pl.Series("open", open_arr),
        pl.Series("high", high_arr),
        pl.Series("low", low_arr),
        pl.Series("close", new_close),
    ])


def apply_perturbation(
    df: pl.DataFrame,
    config: PerturbationConfig,
    trial_number: int,
) -> Tuple[pl.DataFrame, int, Dict[str, Any]]:
    """APLICA PERTURBACIÓN A LOS DATOS SEGÚN LA CONFIGURACIÓN."""
    if not config.enabled:
        return df, 0, {"perturbation_applied": False}
    
    # Semilla única por trial
    seed = (config.seed or 42) + trial_number
    
    # Guardar estadísticas originales para verificación
    original_close_mean = float(df["close"].mean())
    original_close_std = float(df["close"].std())
    
    # Aplicar método de perturbación
    method = config.method.lower()
    
    if method == "returns_perturbation":
        df_perturbed = perturb_returns_professional(df, config.noise_factor, seed)
    elif method == "block_bootstrap":
        df_perturbed = perturb_block_bootstrap(df, config.block_size, seed)
    elif method in ("stationary_bootstrap", "returns_shuffle"):
        df_perturbed = perturb_returns_shuffle(df, seed)
    else:
        # Fallback al método recomendado
        df_perturbed = perturb_returns_professional(df, config.noise_factor, seed)
    
    # Verificar que la perturbación se aplicó
    perturbed_close_mean = float(df_perturbed["close"].mean())
    perturbed_close_std = float(df_perturbed["close"].std())
    
    # Calcular diferencia relativa
    mean_diff_pct = abs(perturbed_close_mean - original_close_mean) / original_close_mean * 100
    
    info = {
        "perturbation_applied": True,
        "method": method,
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
# 6. HELPERS Y UTILIDADES
# =============================================================================

def _slug(s: str) -> str:
    """GENERA UN SLUG VÁLIDO PARA NOMBRES DE ESTUDIO."""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "study"


def create_study_for_strategy(
    *,
    cfg: OptunaConfig,
    strategy_name: str,
    activo: Optional[str] = None,
) -> optuna.study.Study:
    """CREA ESTUDIO OPTUNA CON CMA-ES O TPE."""
    parts = [str(cfg.study_name_prefix), str(strategy_name)]
    if activo:
        parts.append(str(activo))
    study_name = _slug("_".join(parts))
    
    sampler_type = cfg.sampler.upper() if cfg.sampler else "CMA"
    
    if sampler_type == "GP" and _GP_AVAILABLE:
        # GPSampler: Procesos Gaussianos (Optuna v4.0+)
        # Modela la función objetivo como un proceso gaussiano
        # Equilibra exploración (alta incertidumbre) y explotación (alta predicción)
        sampler = GPSampler(
            seed=cfg.seed,
            n_startup_trials=10,  # Trials aleatorios iniciales para construir el modelo
        )
    elif sampler_type == "GP" and not _GP_AVAILABLE:
        import warnings
        warnings.warn("GPSampler no disponible (requiere Optuna v4.0+). Usando CmaEsSampler.")
        sampler = CmaEsSampler(
            seed=cfg.seed,
            n_startup_trials=10,
            warn_independent_sampling=False,
            consider_pruned_trials=False,
        )
    elif sampler_type == "BOTORCH" and _BOTORCH_AVAILABLE:
        # BoTorchSampler: GP avanzado con backend BoTorch
        # Usa Expected Improvement como función de adquisición
        sampler = BoTorchSampler(
            seed=cfg.seed,
            n_startup_trials=10,
        )
    elif sampler_type == "BOTORCH" and not _BOTORCH_AVAILABLE:
        if _GP_AVAILABLE:
            import warnings
            warnings.warn("BoTorchSampler no disponible. Usando GPSampler.")
            sampler = GPSampler(seed=cfg.seed, n_startup_trials=10)
        else:
            import warnings
            warnings.warn("BoTorchSampler no disponible. Usando CmaEsSampler.")
            sampler = CmaEsSampler(
                seed=cfg.seed,
                n_startup_trials=10,
                warn_independent_sampling=False,
                consider_pruned_trials=False,
            )
    elif sampler_type == "CMA":
        sampler = CmaEsSampler(
            seed=cfg.seed,
            n_startup_trials=10,
            warn_independent_sampling=False,
            consider_pruned_trials=False,
        )
    else:  # TPE o cualquier otro
        sampler = TPESampler(
            seed=cfg.seed,
            multivariate=True,
            group=True,
        )
    
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
        storage=None,
        load_if_exists=False,
    )
    
    set_study_for_scorer(study)
    
    return study


# =============================================================================
# 7. COMPONENTES DEL PIPELINE
# =============================================================================

@dataclass
class DataLoader:
    """MANEJA LA CARGA DE DATOS."""
    
    @staticmethod
    def load_data(file_path: str) -> pl.DataFrame:
        """CARGA DATOS DESDE ARCHIVO."""
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
    """EJECUTA ESTRATEGIA Y RETORNA SEÑALES."""
    
    @staticmethod
    def generate_signals(
        df: pl.DataFrame,
        strategy: Strategy,
        params: Dict[str, Any],
        df_by_timeframe: Optional[Dict[str, pl.DataFrame]] = None,
    ) -> pl.DataFrame:
        """GENERA SEÑALES DE TRADING."""
        base_tf = normalize_timeframe_to_suffix(params.get("__timeframe_base", "1h"))
        
        if hasattr(strategy, "get_required_timeframes") and callable(strategy.get_required_timeframes):
            required_tfs = strategy.get_required_timeframes(params)
            if required_tfs and df_by_timeframe:
                df = prepare_multitimeframe_data(
                    df, required_tfs, base_tf=base_tf, anti_lookahead=True,
                )
        
        signals_df = strategy.generate_signals(df, params)
        
        cols = signals_df.columns
        if "signal_long" not in cols:
            signals_df = signals_df.with_columns(pl.lit(False).alias("signal_long"))
        if "signal_short" not in cols:
            signals_df = signals_df.with_columns(pl.lit(False).alias("signal_short"))
        
        return signals_df


@dataclass
class BacktestEngine:
    """EJECUTA BACKTEST Y RETORNA MÉTRICAS."""
    
    _params_cache: Dict[int, BacktestParams] = field(default_factory=dict)
    
    @staticmethod
    def run_backtest(
        df: pl.DataFrame,
        signals: pl.DataFrame,
        config: BacktestConfig,
        params: Dict[str, Any],
        strategy: Strategy,
    ) -> Tuple[pl.DataFrame, List[float], Dict[str, Any]]:
        """EJECUTA UN BACKTEST COMPLETO."""
        backtest_params = BacktestParams.from_config_and_params(config, params)
        timeframe = params.get("__timeframe_base", "1h")
        
        trades_df, equity_curve = calculate_performance_vectorized_numba(
            df=df, signals=signals, params=backtest_params, strategy=strategy,
        )
        
        period_start = None
        period_end = None
        if "timestamp" in df.columns and df.height > 0:
            period_start = df["timestamp"].min()
            period_end = df["timestamp"].max()
        
        metrics: Dict[str, Any]
        if not trades_df.is_empty():
            metrics = resumen_metricas(
                trades_df,
                saldo_inicial=config.saldo_inicial,
                equity_curve=equity_curve,
                timeframe=timeframe,
                period_start=period_start,
                period_end=period_end,
            )
        else:
            metrics = {}
        
        return trades_df, equity_curve, metrics


# =============================================================================
# 8. RUNNER PRINCIPAL DE OPTIMIZACIÓN
# =============================================================================

@dataclass
class OptimizationRunner:
    """RUNNER DE OPTIMIZACIÓN BAYESIANA (CMA-ES / TPE)."""
    
    config: BacktestConfig
    n_trials: int
    reporters: Sequence[Reporter]
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    activo: Optional[str] = None
    
    # Configuración de perturbación
    perturbation_config: PerturbationConfig = field(default_factory=PerturbationConfig)
    
    # Estado interno
    _last_study: Optional[optuna.study.Study] = None
    _perturbation_stats: Dict[str, Any] = field(default_factory=dict)
    
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
        base_tf = normalize_timeframe_to_suffix(base_timeframe or "1h")
        df_map = df_by_timeframe or {base_tf: df}
        df_base = df_map.get(base_tf, df)
        
        # Reset stats de perturbación
        self._perturbation_stats = {
            "enabled": self.perturbation_config.enabled,
            "method": self.perturbation_config.method,
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
        
        # QTY_MAX_ACTIVO: Optimizar si está habilitado, o usar valor fijo
        if self.config.optimize_qty_max_activo:
            qty_min, qty_max, qty_step = self.config.qty_max_activo_range
            qty_optimized = trial.suggest_float(
                "qty_max_activo", 
                qty_min, 
                qty_max, 
                step=qty_step
            )
            params_rt["__qty_max_activo"] = qty_optimized
            params_rt["qty_max_activo"] = qty_optimized  # Alias para reporting
        else:
            params_rt["__qty_max_activo"] = float(self.config.qty_max_activo)
        
        params_rt["__comision_pct"] = float(self.config.comision_pct)
        params_rt["__comision_sides"] = int(self.config.comision_sides)
        params_rt["__saldo_usado"] = float(self.config.saldo_usado)
        params_rt["__apalancamiento_max"] = float(self.config.apalancamiento_max)
        params_rt["__strategy_exit_enabled"] = bool(getattr(strategy, "SALIDAS_PERSONALIZADAS", False))
        
        # Resolver configuración de salida
        exit_settings = resolve_exit_settings_for_trial(
            trial=trial,
            config=self.config,
            allow_custom_exits=bool(getattr(strategy, "SALIDAS_PERSONALIZADAS", False)),
        )
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
            df_entry = df_map.get(entry_tf, df_base)
            
            # ================================================================
            # PERTURBACIÓN DE DATOS (COHERENTE PARA TODOS LOS TIMEFRAMES)
            # ================================================================
            df_trial = df_entry
            df_map_perturbed = df_map  # Por defecto, usar el original
            perturb_info = {"perturbation_applied": False}
            perturb_seed = 0
            
            if self.perturbation_config.enabled:
                # Calcular semilla única para este trial
                base_seed = self.perturbation_config.seed or 42
                perturb_seed = base_seed + trial.number
                
                # Perturbar TODOS los timeframes con la misma semilla
                df_map_perturbed = {}
                for tf_key, tf_df in df_map.items():
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
            
            # Score basado en calidad (sin test de vecindario)
            score = float(score_optuna(metrics))
            
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
            for reporter in self.reporters:
                if hasattr(reporter, "needs_dataframe") and reporter.needs_dataframe(score):
                    ohlc_cols = ["timestamp", "open", "high", "low", "close", "volume"]
                    base_cols = [c for c in ohlc_cols if c in df_base.columns]
                    signal_cols = [c for c in signals_df.columns if c not in base_cols]
                    df_signals_for_artifacts = df_base.select(base_cols).hstack(
                        signals_df.select(signal_cols)
                    )
                    break
            
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
                neighborhood_result=None,
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
