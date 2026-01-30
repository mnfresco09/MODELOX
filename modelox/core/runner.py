"""modelox/core/runner.py

Runner Principal Unificado con Soporte para:
- Backtesting Normal (sin perturbación)
- Perturbación de Datos para Validación
- NSGA-II (Multi-Objetivo) y TPE (Single-Objetivo)
- Análisis de Vecindario (Neighborhood Fitness Aggregation)

ARQUITECTURA:
=============
El runner soporta DOS modos de operación:

1. MODO NORMAL (perturbation_enabled=False):
   - Cada trial usa los datos ORIGINALES
   - Optuna optimiza parámetros buscando el mejor rendimiento
   - Análisis de vecindario opcional para validar estabilidad

2. MODO PERTURBACIÓN (perturbation_enabled=True):
   - Cada trial usa datos PERTURBADOS con una semilla única
   - Valida que la estrategia funcione en múltiples escenarios
   - Detecta overfitting (si solo funciona con datos exactos)

ANÁLISIS DE VECINDARIO (NEIGHBORHOOD FITNESS):
==============================================
Evalúa estabilidad variando parámetros en un entorno local:
- Genera K vecinos con perturbación gaussiana de parámetros
- Ejecuta K+1 backtests (original + vecinos)
- Score = μ - λ·σ (media penalizada por varianza)
- Detecta "picos de aguja" vs "mesetas estables"

IMPORTANTE: Los análisis de vecindario SIEMPRE usan los MISMOS datos que el
trial original (df_trial). Esto significa que:
- Si el trial usa datos ORIGINALES → vecinos usan datos ORIGINALES
- Si el trial usa datos PERTURBADOS → vecinos usan ESOS MISMOS datos perturbados

Esta consistencia es CRÍTICA para validar estabilidad de forma correcta.

MÉTODOS DE PERTURBACIÓN PROFESIONALES:
======================================
- "returns_perturbation": Perturba retornos con ruido calibrado a volatilidad (RECOMENDADO)
- "block_bootstrap": Block bootstrap sobre retornos (mantiene autocorrelación)  
- "stationary_bootstrap": Politis & Romano 1994
- "returns_shuffle": Shuffle de retornos (rompe autocorrelación)
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
from optuna.samplers import NSGAIISampler, TPESampler

from .engine import BacktestParams, calculate_performance_vectorized_numba
from .metrics import resumen_metricas
# v7.0: Scoring unificado - todo está en scoring.py ahora
from .scoring import (
    score_optuna, 
    nsga2_objectives, 
    score_quality_only,
    run_neighborhood_analysis,
    NeighborhoodConfig,
    NeighborhoodResult,
    DEFAULT_NEIGHBORHOOD_CONFIG,
)
from .types import (
    BacktestConfig,
    Reporter,
    Strategy,
    TrialArtifacts,
    normalize_timeframe_to_suffix,
)
from .data import load_data, prepare_multitimeframe_data
from .exits import resolve_exit_settings_for_trial

# Silenciar warnings experimentales de Optuna
warnings.filterwarnings("ignore", category=ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Debug timings
_TIMINGS_VERBOSE = os.environ.get("MODELOX_TIMINGS_VERBOSE", "0") in {"1", "true", "True", "YES", "yes"}
_TIMINGS_PRINT_EVERY = int(os.environ.get("MODELOX_TIMINGS_PRINT_EVERY", "1"))

# =============================================================================
# OPTIMIZACIÓN: Cache de señales base para vecinos
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
# CONFIGURACIÓN
# =============================================================================

@dataclass(frozen=True)
class OptunaConfig:
    """
    Configuración de Optuna.
    
    PARALELIZACIÓN:
    ===============
    n_jobs controla el número de trials paralelos de Optuna.
    
    - n_jobs=1: Secuencial (default, más estable)
    - n_jobs=2+: Paralelo (más rápido, requiere más RAM)
    - n_jobs=-1: Usar todos los cores disponibles
    
    IMPORTANTE: Si usas n_jobs>1, asegúrate de que tu sistema
    tenga suficiente RAM (~500MB por worker extra).
    
    VARIABLES DE ENTORNO RELACIONADAS:
    - MODELOX_NEIGHBORHOOD_PARALLEL=1: Paralelizar análisis de vecinos (default ON)
    - MODELOX_NEIGHBORHOOD_WORKERS=6: Workers para vecinos (default 6)
    """
    seed: Optional[int] = None
    n_jobs: int = 1  # Número de trials paralelos de Optuna
    storage: Optional[str] = None
    study_name_prefix: str = "MODELOX"
    sampler: str = "tpe"  # "tpe" o "nsga2"
    use_nsga2: bool = False  # Usar NSGA-II multi-objetivo


@dataclass
class PerturbationConfig:
    """
    Configuración del Sistema de Perturbación de Datos.
    
    Cuando enabled=True, cada trial usa datos perturbados con una semilla única.
    Esto permite validar que la estrategia no esté sobreajustada a los datos exactos.
    
    MÉTODOS DISPONIBLES:
    - "returns_perturbation": Añade ruido gaussiano calibrado a retornos (RECOMENDADO)
    - "block_bootstrap": Block bootstrap sobre retornos
    - "stationary_bootstrap": Politis & Romano 1994
    - "returns_shuffle": Shuffle completo de retornos
    
    noise_factor: Intensidad del ruido (0.0 = sin ruido, 1.0 = ruido igual a volatilidad)
    block_size: Tamaño de bloque para métodos de bootstrap
    """
    enabled: bool = False
    method: str = "returns_perturbation"  # Método por defecto (el más profesional)
    noise_factor: float = 0.3  # 30% de la volatilidad como ruido
    block_size: int = 100  # Tamaño de bloque para bootstrap
    seed: Optional[int] = 42  # Semilla base (cada trial suma su número)
    
    # Verificación de que la perturbación se aplicó
    verify_perturbation: bool = True  # Si True, verifica que los datos cambien


# =============================================================================
# FUNCIONES DE PERTURBACIÓN PROFESIONALES
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
# KERNEL NUMBA PARA PERTURBACIÓN RÁPIDA
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


def perturb_block_bootstrap(
    df: pl.DataFrame,
    block_size: int = 100,
    seed: Optional[int] = None,
) -> pl.DataFrame:
    """
    BLOCK BOOTSTRAP sobre retornos.
    
    En lugar de reorganizar precios directamente (que causa saltos),
    este método:
    1. Calcula retornos logarítmicos
    2. Divide en bloques de tamaño fijo
    3. Muestrea bloques con reemplazo
    4. Reconstruye precios desde los retornos muestreados
    
    Preserva autocorrelación de corto plazo dentro de bloques.
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
    """
    SHUFFLE de retornos (rompe autocorrelación completamente).
    
    Mantiene la distribución de retornos pero cambia el orden temporal.
    Útil para verificar si la estrategia depende de patrones temporales.
    """
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
# HELPERS
# =============================================================================

def _slug(s: str) -> str:
    """Genera un slug válido para nombres de estudio."""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "study"


def create_study_for_strategy(
    *,
    cfg: OptunaConfig,
    strategy_name: str,
    activo: Optional[str] = None,
    use_trinity_objectives: bool = False,
) -> optuna.study.Study:
    """
    Crea estudio Optuna con soporte para NSGA-II.
    
    Args:
        cfg: Configuración de Optuna
        strategy_name: Nombre de la estrategia
        activo: Nombre del activo (opcional)
        use_trinity_objectives: Si True, usa 3 objetivos (Trinidad del paper)
            - Robust_DSR: MAXIMIZAR
            - Worst_Case_CVaR: MINIMIZAR
            - Equity_Stability_R2: MAXIMIZAR
    """
    parts = [str(cfg.study_name_prefix), str(strategy_name)]
    if activo:
        parts.append(str(activo))
    study_name = _slug("_".join(parts))
    
    use_nsga2 = cfg.use_nsga2 or str(cfg.sampler).lower() == "nsga2"
    
    if use_nsga2:
        sampler = NSGAIISampler(seed=cfg.seed)
        
        if use_trinity_objectives:
            # TRINIDAD DE OBJETIVOS (Neighborhood Fitness Aggregation)
            # 1. Robust_DSR: MAXIMIZAR (mayor es mejor)
            # 2. Worst_Case_CVaR: MINIMIZAR (menor es mejor)
            # 3. Equity_Stability_R2: MAXIMIZAR (mayor es mejor)
            directions = ["maximize", "minimize", "maximize"]
        else:
            # Modo legacy: 2 objetivos
            directions = ["maximize", "minimize"]  # quality ↑, drawdown ↓
        
        return optuna.create_study(
            directions=directions,
            sampler=sampler,
            study_name=study_name,
            storage=None,
            load_if_exists=False,
        )
    else:
        sampler = TPESampler(seed=cfg.seed, multivariate=True, group=True)
        return optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=study_name,
            storage=None,
            load_if_exists=False,
        )


# =============================================================================
# COMPONENTES DEL PIPELINE
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
        base_tf = normalize_timeframe_to_suffix(params.get("__timeframe_base", "1h"))
        
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
        timeframe = params.get("__timeframe_base", "1h")
        
        trades_df, equity_curve = calculate_performance_vectorized_numba(
            df=df, signals=signals, params=backtest_params, strategy=strategy,
        )
        
        metrics: Dict[str, Any]
        if not trades_df.is_empty():
            metrics = resumen_metricas(
                trades_df,
                saldo_inicial=config.saldo_inicial,
                equity_curve=equity_curve,
                timeframe=timeframe,
            )
        else:
            metrics = {}
        
        return trades_df, equity_curve, metrics


# =============================================================================
# RUNNER PRINCIPAL UNIFICADO
# =============================================================================

@dataclass
class OptimizationRunner:
    """
    Runner de Optimización Unificado.
    
    Soporta:
    - Backtesting normal (sin perturbación)
    - Perturbación de datos para validación
    - NSGA-II (Multi-Objetivo) y TPE (Single-Objetivo)
    - Análisis de Vecindario (Neighborhood Fitness Aggregation)
    """
    
    config: BacktestConfig
    n_trials: int
    reporters: Sequence[Reporter]
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    activo: Optional[str] = None
    
    # Configuración de perturbación
    perturbation_config: PerturbationConfig = field(default_factory=PerturbationConfig)
    
    # Configuración de Agregación de Fitness Vecinal
    neighborhood_config: Optional[NeighborhoodConfig] = None
    neighborhood_enabled: bool = True  # Activado por defecto
    
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
        
        use_nsga2 = self.optuna.use_nsga2 or str(self.optuna.sampler).lower() == "nsga2"
        
        if use_nsga2:
            objective = self._create_nsga2_objective(df_base, df_map, strategy, base_tf)
        else:
            objective = self._create_single_objective(df_base, df_map, strategy, base_tf)
        
        # Determinar si usar Trinidad de objetivos
        # (solo cuando NSGA-II + neighborhood están activos)
        use_trinity = use_nsga2 and self.neighborhood_enabled
        
        study = create_study_for_strategy(
            cfg=self.optuna, 
            strategy_name=strategy.name, 
            activo=self.activo,
            use_trinity_objectives=use_trinity,
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
    
    def _create_nsga2_objective(
        self,
        df_base: pl.DataFrame,
        df_map: Dict[str, pl.DataFrame],
        strategy: Strategy,
        base_tf: str,
    ) -> Callable[[optuna.trial.Trial], Tuple[float, ...]]:
        """
        Crea función objetivo para NSGA-II (Multi-Objetivo).
        
        MODOS DE OPERACIÓN:
        
        1. MODO NEIGHBORHOOD (neighborhood_enabled=True):
           - Retorna TRINIDAD DE OBJETIVOS: (Robust_DSR, Worst_Case_CVaR, Equity_R2)
           - Robust_DSR: MAXIMIZAR (Sharpe penalizado por vecindario y trials)
           - Worst_Case_CVaR: MINIMIZAR (peor caso de riesgo de cola)
           - Equity_R2: MAXIMIZAR (estabilidad de curva de equity)
        
        2. MODO LEGACY (neighborhood_enabled=False):
           - Retorna 2 objetivos: (quality, drawdown)
           - quality: MAXIMIZAR (score de calidad)
           - drawdown: MINIMIZAR (máximo drawdown)
        """
        
        def objective(trial: optuna.trial.Trial) -> Tuple[float, ...]:
            # LIMPIEZA PERIÓDICA para mantener velocidad constante
            periodic_cleanup(trial.number)
            
            params_rt = self._prepare_params(trial, strategy, base_tf)
            entry_tf = params_rt["__timeframe_entry"]
            df_entry = df_map.get(entry_tf, df_base)
            
            # ================================================================
            # PERTURBACIÓN DE DATOS (COHERENTE PARA TODOS LOS TIMEFRAMES)
            # ================================================================
            # Cuando la perturbación está habilitada, TODOS los dataframes
            # en df_map deben perturbarse con la MISMA semilla para mantener
            # coherencia temporal entre timeframes.
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
                    # Limitar a últimos 100 valores para evitar memory bloat
                    diffs = self._perturbation_stats["mean_diff_pcts"]
                    diffs.append(perturb_info["mean_diff_pct"])
                    if len(diffs) > 100:
                        self._perturbation_stats["mean_diff_pcts"] = diffs[-100:]
            
            # Generar señales (usa df_map_perturbed para multi-timeframe)
            signals_df = SignalGenerator.generate_signals(df_trial, strategy, params_rt, df_map_perturbed)
            
            # Ejecutar backtest
            trades_df, equity_curve, metrics = BacktestEngine.run_backtest(
                df_trial, signals_df, self.config, params_rt, strategy,
            )
            
            if trades_df.is_empty():
                if self.neighborhood_enabled:
                    # Trinidad: (Robust_DSR min, CVaR max, R2 min)
                    return (0.0, 100.0, 0.0)
                else:
                    return (0.1, 100.0)
            
            trial.set_user_attr("metricas", metrics)
            
            # Objetivos NSGA-II (legacy, se pueden sobrescribir)
            quality, drawdown = nsga2_objectives(metrics)
            score = float(score_optuna(metrics))
            
            # ================================================================
            # SISTEMA DE EVALUACIÓN DE VECINDARIO
            # ================================================================
            neighborhood_result: Optional[NeighborhoodResult] = None
            
            # Helpers para backtest
            # IMPORTANTE: Usar SIEMPRE el mismo df_trial y df_map_perturbed
            # para garantizar coherencia cuando hay perturbación.
            def _generate_signals_helper(_df, strat, params):
                return SignalGenerator.generate_signals(df_trial, strat, params, df_map_perturbed)
            
            def _run_backtest_helper(_df, sigs, cfg, params, strat):
                return BacktestEngine.run_backtest(df_trial, sigs, cfg, params, strat)
            
            # ================================================================
            # NEIGHBORHOOD FITNESS - TRINIDAD DE OBJETIVOS
            # ================================================================
            if self.neighborhood_enabled:
                neighborhood_cfg = self.neighborhood_config or DEFAULT_NEIGHBORHOOD_CONFIG
                
                if neighborhood_cfg.enabled:
                    neighborhood_result = run_neighborhood_analysis(
                        strategy=strategy,
                        df=df_trial,
                        params=params_rt,
                        original_metrics=metrics,
                        original_score=score,
                        equity_curve=equity_curve,
                        config=self.config,
                        neighborhood_config=neighborhood_cfg,
                        trial_number=trial.number,
                        run_backtest_fn=_run_backtest_helper,
                        generate_signals_fn=_generate_signals_helper,
                    )
                    
                    trial.set_user_attr("neighborhood_result", neighborhood_result.to_dict())
                    
                    # Marcar info del vecindario
                    params_rt["__neighborhood_score"] = neighborhood_result.aggregated_score
                    params_rt["__robust_dsr"] = neighborhood_result.robust_dsr
                    params_rt["__worst_cvar"] = neighborhood_result.worst_case_cvar
                    params_rt["__equity_r2"] = neighborhood_result.equity_stability_r2
                    
                    # Score agregado para reporting
                    score = neighborhood_result.aggregated_score
            
            # ================================================================
            
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
            
            neighborhood_dict = neighborhood_result.to_dict() if neighborhood_result else None
            
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
                neighborhood_result=neighborhood_dict,
            )
            
            for reporter in self.reporters:
                reporter.on_trial_end(artifacts)
            
            # ================================================================
            # RETORNAR OBJETIVOS SEGÚN MODO
            # ================================================================
            if self.neighborhood_enabled and neighborhood_result:
                # TRINIDAD DE OBJETIVOS:
                # 1. Robust_DSR: MAXIMIZAR (mayor es mejor)
                # 2. Worst_Case_CVaR: MINIMIZAR (menor es mejor)
                # 3. Equity_Stability_R2: MAXIMIZAR (mayor es mejor)
                return (
                    neighborhood_result.robust_dsr,
                    neighborhood_result.worst_case_cvar,
                    neighborhood_result.equity_stability_r2,
                )
            else:
                # Modo legacy: 2 objetivos
                return (quality, drawdown)
        
        return objective
    
    def _create_single_objective(
        self,
        df_base: pl.DataFrame,
        df_map: Dict[str, pl.DataFrame],
        strategy: Strategy,
        base_tf: str,
    ) -> Callable[[optuna.trial.Trial], float]:
        """Crea función objetivo para TPE (Single-Objective) con análisis de vecindario."""
        
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
            # Cuando la perturbación está habilitada, TODOS los dataframes
            # en df_map deben perturbarse con la MISMA semilla para mantener
            # coherencia temporal entre timeframes.
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
                    # Limitar a últimos 100 valores para evitar memory bloat
                    diffs = self._perturbation_stats["mean_diff_pcts"]
                    diffs.append(perturb_info["mean_diff_pct"])
                    if len(diffs) > 100:
                        self._perturbation_stats["mean_diff_pcts"] = diffs[-100:]
            
            # Generar señales (usa df_map_perturbed para multi-timeframe)
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
            score = float(score_optuna(metrics))
            
            # ================================================================
            # NEIGHBORHOOD FITNESS AGGREGATION
            # ================================================================
            # Evalúa robustez mediante variación de parámetros:
            #    - Genera K vecinos con perturbación gaussiana de parámetros
            #    - Ejecuta K+1 backtests (original + vecinos)
            #    - Score = μ - λ·σ (media penalizada por varianza)
            #    - Detecta "picos de aguja" vs "mesetas"
            # ================================================================
            
            neighborhood_result: Optional[NeighborhoodResult] = None
            
            # Helpers para backtest
            # IMPORTANTE: Usar SIEMPRE el mismo df_trial y df_map_perturbed
            # para garantizar coherencia cuando hay perturbación.
            def _generate_signals_helper(_df, strat, params):
                return SignalGenerator.generate_signals(df_trial, strat, params, df_map_perturbed)
            
            def _run_backtest_helper(_df, sigs, cfg, params, strat):
                return BacktestEngine.run_backtest(df_trial, sigs, cfg, params, strat)
            
            # ================================================================
            # NEIGHBORHOOD FITNESS AGGREGATION
            # ================================================================
            if self.neighborhood_enabled:
                neighborhood_cfg = self.neighborhood_config or DEFAULT_NEIGHBORHOOD_CONFIG
                
                if neighborhood_cfg.enabled:
                    # CRÍTICO: Usar df_trial (mismos datos del trial, perturbados o no)
                    neighborhood_result = run_neighborhood_analysis(
                        strategy=strategy,
                        df=df_trial,
                        params=params_rt,
                        original_metrics=metrics,
                        original_score=score,
                        equity_curve=equity_curve,
                        config=self.config,
                        neighborhood_config=neighborhood_cfg,
                        trial_number=trial.number,
                        run_backtest_fn=_run_backtest_helper,
                        generate_signals_fn=_generate_signals_helper,
                    )
                    
                    trial.set_user_attr("neighborhood_result", neighborhood_result.to_dict())
                    
                    # USAR SCORE AGREGADO: μ - λ·σ
                    score = neighborhood_result.aggregated_score
                    
                    # Marcar info del vecindario en params para reporting
                    params_rt["__neighborhood_score"] = neighborhood_result.aggregated_score
                    params_rt["__neighborhood_mean"] = neighborhood_result.mean_score
                    params_rt["__neighborhood_std"] = neighborhood_result.std_score
                    params_rt["__neighborhood_n_tested"] = neighborhood_result.n_neighbors_tested
                    params_rt["__robust_dsr"] = neighborhood_result.robust_dsr
            
            # ================================================================
            
            t_total = time.perf_counter() - t0_total
            
            if _TIMINGS_VERBOSE and (trial.number % _TIMINGS_PRINT_EVERY == 0):
                perturb_str = f" [P]" if perturb_info.get("perturbation_applied", False) else ""
                neighb_info = ""
                if neighborhood_result:
                    neighb_info = (
                        f" │ neighb {neighborhood_result.n_neighbors_successful}/{neighborhood_result.n_neighbors_tested} "
                        f"μ={neighborhood_result.mean_score:.1f} σ={neighborhood_result.std_score:.1f} "
                        f"→{neighborhood_result.aggregated_score:.1f} "
                        f"{neighborhood_result.execution_time_ms:.0f}ms"
                    )
                print(
                    f"  ⏱ TRIAL {trial.number:3d}{perturb_str} │ "
                    f"signals {(t2_signals - t1_signals)*1000:6.1f}ms │ "
                    f"backtest {(t2_backtest - t1_backtest)*1000:6.1f}ms │ "
                    f"total {t_total*1000:6.1f}ms │ "
                    f"trades {len(trades_df):5d}{neighb_info}"
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
            
            neighborhood_dict = neighborhood_result.to_dict() if neighborhood_result else None
            
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
                neighborhood_result=neighborhood_dict,
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
    Limpia todos los recursos de paralelización.
    
    Llamar al final de la optimización para liberar:
    - Pool de threads de análisis de vecindario
    - Caches de señales
    - Garbage collector
    """
    try:
        from .scoring import shutdown_neighbor_pool
        shutdown_neighbor_pool()
    except Exception:
        pass
    
    clear_all_caches()
    gc.collect()
# =============================================================================
# ALIAS DE COMPATIBILIDAD
# =============================================================================

# Mantener el nombre anterior para compatibilidad
MonteCarloRunner = OptimizationRunner
MonteCarloConfig = PerturbationConfig
