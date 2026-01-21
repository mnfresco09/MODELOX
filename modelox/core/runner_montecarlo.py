"""
Monte Carlo Optimization Runner con NSGA-II (Multi-Objetivo)

CONCEPTO:
=========
- N_TRIALS = número de mercados sintéticos únicos
- Cada trial evalúa parámetros en un mercado diferente
- NSGA-II optimiza DOS objetivos simultáneamente:
  1. MAXIMIZAR: Calidad/Rentabilidad (score_quality_only)
  2. MINIMIZAR: Drawdown (riesgo)
  
Esto encuentra el FRENTE DE PARETO: las mejores estrategias para cada
nivel de riesgo aceptable.
"""

from __future__ import annotations

import os
import time
import gc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable

import warnings
import numpy as np
import polars as pl
import optuna
from optuna.samplers import NSGAIISampler, TPESampler
from optuna.exceptions import ExperimentalWarning

from .engine import BacktestParams, calculate_performance_vectorized_numba
from .metrics import resumen_metricas
from .scoring import score_quality_only, nsga2_objectives, score_optuna
from .types import BacktestConfig, Reporter, Strategy, TrialArtifacts, normalize_timeframe_to_suffix
from .exits import resolve_exit_settings_for_trial
from .runner import SignalGenerator, BacktestEngine

# Silenciar warnings
warnings.filterwarnings("ignore", category=ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

@dataclass
class MonteCarloConfig:
    """Configuración para optimización Monte Carlo con NSGA-II."""
    n_trials: int = 100              # Número de trials = mercados sintéticos
    noise_pct: float = 0.5           # % de ruido gaussiano (legacy)
    noise_range: float = 100.0       # Rango de variación en unidades monetarias (±100 por defecto)
    block_size: int = 1440           # Tamaño de bloques para bootstrap
    method: str = "monetary"         # "noise", "monetary", "mixed", "block_bootstrap"
    seed: int | None = 42            # Semilla base
    use_nsga2: bool = True           # Usar NSGA-II (multi-objetivo)


# =============================================================================
# GENERADORES DE MERCADOS SINTÉTICOS
# =============================================================================

def add_monetary_noise(df: pl.DataFrame, noise_range: float = 100.0, seed: int | None = None) -> pl.DataFrame:
    """
    Genera mercado sintético con DIVERGENCIA ACUMULATIVA SUAVE.
    
    MÉTODO:
    1. Toma la variación REAL de cada vela: delta_real = close - open
    2. Añade un pequeño paso aleatorio (random walk): delta_extra
    3. Reconstruye precios acumulativamente desde el precio inicial
    
    Resultado: Los precios divergen GRADUALMENTE del original.
    - Al inicio: casi idéntico al original
    - Al final: completamente diferente
    - Sin saltos bruscos entre velas consecutivas
    
    noise_range: Controla la intensidad de la divergencia.
                 Mayor valor = divergencia más rápida.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    
    # Extraer arrays originales
    open_arr = df["open"].to_numpy().astype(np.float64)
    high_arr = df["high"].to_numpy().astype(np.float64)
    low_arr = df["low"].to_numpy().astype(np.float64)
    close_arr = df["close"].to_numpy().astype(np.float64)
    
    # =========================================================================
    # PASO 1: Calcular variaciones REALES de cada vela
    # =========================================================================
    # Variación intra-vela (forma de la vela)
    delta_intra = close_arr - open_arr  # Positivo = vela verde, Negativo = roja
    
    # Variación inter-vela (salto entre velas)
    delta_inter = np.zeros(n)
    delta_inter[1:] = open_arr[1:] - close_arr[:-1]  # Gap entre cierre anterior y apertura actual
    
    # =========================================================================
    # PASO 2: Añadir RANDOM WALK a las variaciones
    # =========================================================================
    # step_size pequeño para divergencia gradual
    # Con n=10000 velas, después de 1000 velas el drift será ~noise_range
    step_size = noise_range / np.sqrt(n) * 2
    
    # Random walk: pequeños pasos que se ACUMULAN
    steps = rng.uniform(-step_size, step_size, n)
    cumulative_offset = np.cumsum(steps)  # Offset acumulado
    
    # También añadir pequeña variación al delta intra-vela (±5% del step)
    intra_noise = rng.uniform(-step_size * 0.3, step_size * 0.3, n)
    delta_intra_noisy = delta_intra + intra_noise
    
    # =========================================================================
    # PASO 3: Reconstruir precios ACUMULATIVAMENTE
    # =========================================================================
    new_open = np.zeros(n)
    new_close = np.zeros(n)
    new_high = np.zeros(n)
    new_low = np.zeros(n)
    
    # Primera vela: partir del precio original + offset inicial
    new_open[0] = open_arr[0] + cumulative_offset[0]
    new_close[0] = new_open[0] + delta_intra_noisy[0]
    
    # Calcular high/low relativos al open/close original
    high_above_max = high_arr[0] - max(open_arr[0], close_arr[0])
    low_below_min = min(open_arr[0], close_arr[0]) - low_arr[0]
    new_high[0] = max(new_open[0], new_close[0]) + high_above_max
    new_low[0] = min(new_open[0], new_close[0]) - low_below_min
    
    # Resto de velas: construir secuencialmente
    for i in range(1, n):
        # Open = Close anterior + gap original + offset acumulado diferencial
        original_gap = delta_inter[i]
        new_open[i] = new_close[i-1] + original_gap + (cumulative_offset[i] - cumulative_offset[i-1])
        
        # Close = Open + variación intra-vela modificada
        new_close[i] = new_open[i] + delta_intra_noisy[i]
        
        # High y Low: mantener la "sombra" proporcional
        high_above = high_arr[i] - max(open_arr[i], close_arr[i])
        low_below = min(open_arr[i], close_arr[i]) - low_arr[i]
        new_high[i] = max(new_open[i], new_close[i]) + high_above
        new_low[i] = min(new_open[i], new_close[i]) - low_below
    
    # =========================================================================
    # PASO 4: Asegurar coherencia y precios positivos
    # =========================================================================
    min_price = 0.01
    new_open = np.maximum(new_open, min_price)
    new_high = np.maximum(new_high, min_price)
    new_low = np.maximum(new_low, min_price)
    new_close = np.maximum(new_close, min_price)
    
    # Asegurar High >= max(O,C) y Low <= min(O,C)
    max_oc = np.maximum(new_open, new_close)
    min_oc = np.minimum(new_open, new_close)
    new_high = np.maximum(new_high, max_oc)
    new_low = np.minimum(new_low, min_oc)
    new_low = np.minimum(new_low, new_high)
    
    result = df.with_columns([
        pl.Series("open", new_open),
        pl.Series("high", new_high),
        pl.Series("low", new_low),
        pl.Series("close", new_close),
    ])
    
    # Variación de volumen (±20%)
    vol_noise = rng.uniform(0.8, 1.2, n)
    result = result.with_columns(
        (pl.col("volume") * pl.Series(vol_noise)).alias("volume")
    )
    
    return result


def add_gaussian_noise(df: pl.DataFrame, noise_pct: float = 0.05, seed: int | None = None) -> pl.DataFrame:
    """
    Añade ruido gaussiano a los precios OHLCV manteniendo coherencia de velas.
    
    IMPORTANTE: Asegura que después de la perturbación:
    - High >= max(Open, Close)
    - Low <= min(Open, Close)
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    noise_factor = noise_pct / 100.0
    
    # Extraer arrays
    open_arr = df["open"].to_numpy().astype(np.float64)
    high_arr = df["high"].to_numpy().astype(np.float64)
    low_arr = df["low"].to_numpy().astype(np.float64)
    close_arr = df["close"].to_numpy().astype(np.float64)
    
    # Generar ruido independiente para cada precio
    noise_open = rng.normal(0, noise_factor, n)
    noise_high = rng.normal(0, noise_factor, n)
    noise_low = rng.normal(0, noise_factor, n)
    noise_close = rng.normal(0, noise_factor, n)
    
    # Aplicar ruido
    open_noisy = open_arr * (1 + noise_open)
    high_noisy = high_arr * (1 + noise_high)
    low_noisy = low_arr * (1 + noise_low)
    close_noisy = close_arr * (1 + noise_close)
    
    # CORREGIR COHERENCIA OHLCV:
    # High debe ser >= max(Open, Close)
    # Low debe ser <= min(Open, Close)
    max_oc = np.maximum(open_noisy, close_noisy)
    min_oc = np.minimum(open_noisy, close_noisy)
    
    high_noisy = np.maximum(high_noisy, max_oc)
    low_noisy = np.minimum(low_noisy, min_oc)
    
    # Asegurar que low <= high
    low_noisy = np.minimum(low_noisy, high_noisy)
    
    result = df.with_columns([
        pl.Series("open", open_noisy),
        pl.Series("high", high_noisy),
        pl.Series("low", low_noisy),
        pl.Series("close", close_noisy),
    ])
    
    # Ruido en volumen
    vol_noise = rng.normal(1.0, noise_factor * 2, n)
    vol_noise = np.maximum(vol_noise, 0.1)
    result = result.with_columns(
        (pl.col("volume") * pl.Series(vol_noise)).alias("volume")
    )
    
    return result


def block_bootstrap(df: pl.DataFrame, block_size: int = 1440, seed: int | None = None) -> pl.DataFrame:
    """
    Bootstrap por bloques - SOLO para datos estacionarios.
    
    ADVERTENCIA: NO usar con datos de crypto/acciones con tendencia.
    Esto reorganiza bloques temporales, lo cual crea discontinuidades
    si el precio ha cambiado significativamente entre épocas.
    
    Para Monte Carlo en trading, usar método "noise" en su lugar.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    
    if n < block_size * 2:
        return add_gaussian_noise(df, 0.3, seed)
    
    n_blocks = n // block_size
    if n_blocks < 2:
        return add_gaussian_noise(df, 0.3, seed)
    
    blocks_needed = (n + block_size - 1) // block_size
    block_indices = rng.integers(0, n_blocks, size=blocks_needed)
    
    indices = []
    for bi in block_indices:
        start = bi * block_size
        end = min(start + block_size, n_blocks * block_size)
        indices.extend(range(start, end))
        if len(indices) >= n:
            break
    
    indices = indices[:n]
    if len(indices) < n:
        indices = indices + indices[:n - len(indices)]
    
    indices = np.array(indices[:n])
    
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    available_cols = [c for c in ohlcv_cols if c in df.columns]
    
    df_ohlcv = df.select([
        pl.col(c).gather(pl.Series(indices)).alias(c) 
        for c in available_cols
    ])
    
    if "timestamp" in df.columns:
        result = df.select("timestamp").hstack(df_ohlcv)
    else:
        result = df_ohlcv
    
    return result


def generate_synthetic_market(
    df: pl.DataFrame,
    method: str,
    noise_pct: float = 0.5,
    noise_range: float = 100.0,
    block_size: int = 1440,
    seed: int | None = None
) -> pl.DataFrame:
    """
    Genera un mercado sintético único.
    
    MÉTODOS PROFESIONALES (recomendados):
    - "stationary_bootstrap": Politis & Romano 1994 - ESTÁNDAR DE LA INDUSTRIA
    - "block_returns": Block bootstrap sobre retornos (mantiene autocorrelación)
    - "returns_shuffle": Shuffle de retornos (simple pero efectivo)
    
    Métodos legacy:
    - "monetary": Divergencia acumulativa (menos realista)
    - "noise": Ruido gaussiano porcentual (no recomendado)
    """
    if method == "stationary_bootstrap":
        # MÉTODO PROFESIONAL: Politis & Romano 1994
        return stationary_bootstrap(df, block_size, seed)
    elif method == "block_returns":
        # Block bootstrap sobre retornos
        return block_bootstrap_returns(df, block_size, seed)
    elif method == "returns_shuffle":
        # Shuffle simple de retornos
        return shuffle_returns(df, seed)
    elif method == "monetary":
        return add_monetary_noise(df, noise_range, seed)
    elif method == "noise":
        return add_gaussian_noise(df, noise_pct, seed)
    elif method == "mixed":
        rng = np.random.default_rng(seed)
        range_multiplier = rng.uniform(0.5, 1.5)
        return add_monetary_noise(df, noise_range * range_multiplier, seed)
    elif method == "block_bootstrap":
        return block_bootstrap(df, block_size, seed)
    else:
        raise ValueError(f"Método desconocido: {method}")


def stationary_bootstrap(df: pl.DataFrame, avg_block_size: int = 100, seed: int | None = None) -> pl.DataFrame:
    """
    PERTURBACIÓN PROFESIONAL DE RETORNOS (Método Quant Real)
    
    En lugar de reorganizar bloques (que crea saltos), este método:
    1. MANTIENE la secuencia temporal original intacta
    2. PERTURBA cada retorno con ruido calibrado a la volatilidad real
    3. RECONSTRUYE precios de forma continua
    
    El ruido es proporcional a la volatilidad del activo:
    - Activos volátiles (BTC): ruido más grande en términos absolutos
    - Activos estables: ruido más pequeño
    
    Resultado: Mercados sintéticos REALISTAS que mantienen la estructura
    pero divergen gradualmente del original.
    
    avg_block_size: NO USADO (mantenido por compatibilidad)
    """
    rng = np.random.default_rng(seed)
    
    # Extraer arrays originales
    open_arr = df["open"].to_numpy().astype(np.float64)
    high_arr = df["high"].to_numpy().astype(np.float64)
    low_arr = df["low"].to_numpy().astype(np.float64)
    close_arr = df["close"].to_numpy().astype(np.float64)
    n = len(close_arr)
    
    # =========================================================================
    # PASO 1: Calcular retornos logarítmicos y su volatilidad
    # =========================================================================
    log_returns = np.diff(np.log(np.maximum(close_arr, 1e-10)))
    n_returns = len(log_returns)
    
    # Volatilidad real del activo (desviación estándar de retornos)
    volatility = np.std(log_returns)
    
    # =========================================================================
    # PASO 2: Perturbar retornos con ruido CALIBRADO
    # =========================================================================
    # El ruido es proporcional a la volatilidad real
    # Factor 0.3-0.5 = perturba sin destruir la estructura
    noise_factor = 0.4
    noise = rng.normal(0, volatility * noise_factor, n_returns)
    
    # Retornos perturbados = originales + ruido calibrado
    perturbed_returns = log_returns + noise
    
    # =========================================================================
    # PASO 3: Reconstruir precios CLOSE de forma continua
    # =========================================================================
    initial_price = close_arr[0]
    new_close = np.zeros(n)
    new_close[0] = initial_price
    new_close[1:] = initial_price * np.exp(np.cumsum(perturbed_returns))
    
    # =========================================================================
    # PASO 4: Reconstruir OHLC manteniendo estructura de cada vela
    # =========================================================================
    # Calcular el ratio de escala para cada vela
    # Esto ajusta toda la vela proporcionalmente
    scale = new_close / np.maximum(close_arr, 1e-10)
    
    new_open = open_arr * scale
    new_high = high_arr * scale
    new_low = low_arr * scale
    
    # =========================================================================
    # PASO 5: Asegurar coherencia OHLCV
    # =========================================================================
    # High debe ser >= max(Open, Close)
    # Low debe ser <= min(Open, Close)
    max_oc = np.maximum(new_open, new_close)
    min_oc = np.minimum(new_open, new_close)
    new_high = np.maximum(new_high, max_oc)
    new_low = np.minimum(new_low, min_oc)
    new_low = np.minimum(new_low, new_high)
    
    # Asegurar precios positivos
    min_price = 0.01
    new_open = np.maximum(new_open, min_price)
    new_high = np.maximum(new_high, min_price)
    new_low = np.maximum(new_low, min_price)
    new_close = np.maximum(new_close, min_price)
    
    result = df.with_columns([
        pl.Series("open", new_open),
        pl.Series("high", new_high),
        pl.Series("low", new_low),
        pl.Series("close", new_close),
    ])
    
    return result


def block_bootstrap_returns(df: pl.DataFrame, block_size: int = 100, seed: int | None = None) -> pl.DataFrame:
    """
    BLOCK BOOTSTRAP PROFESIONAL - Bloques de tamaño fijo.
    
    Similar a stationary bootstrap pero más predecible.
    Mantiene estructura de velas y evita saltos/dispersión.
    """
    rng = np.random.default_rng(seed)
    
    # Extraer arrays
    open_arr = df["open"].to_numpy().astype(np.float64)
    high_arr = df["high"].to_numpy().astype(np.float64)
    low_arr = df["low"].to_numpy().astype(np.float64)
    close_arr = df["close"].to_numpy().astype(np.float64)
    n = len(close_arr)
    
    # Retornos y ratios
    close_returns = np.diff(np.log(np.maximum(close_arr, 1e-10)))
    n_returns = len(close_returns)
    
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
    
    # Reconstruir close
    new_returns = close_returns[sampled_indices]
    initial_price = close_arr[0]
    new_close = np.zeros(n)
    new_close[0] = initial_price
    new_close[1:] = initial_price * np.exp(np.cumsum(new_returns))
    
    # Reconstruir OHLC
    new_open = np.zeros(n)
    new_high = np.zeros(n)
    new_low = np.zeros(n)
    
    new_open[0] = new_close[0] * open_ratio[0]
    new_high[0] = new_close[0] * high_ratio[0]
    new_low[0] = new_close[0] * low_ratio[0]
    
    for i in range(1, n):
        orig_idx = min(sampled_indices[i-1] + 1, n - 1)
        new_open[i] = new_close[i] * open_ratio[orig_idx]
        new_high[i] = new_close[i] * high_ratio[orig_idx]
        new_low[i] = new_close[i] * low_ratio[orig_idx]
    
    # Suavizar gaps
    for i in range(1, n):
        gap_ratio = new_open[i] / new_close[i-1] if new_close[i-1] > 0 else 1.0
        if gap_ratio > 1.05 or gap_ratio < 0.95:
            new_open[i] = new_close[i-1] * (1 + (gap_ratio - 1) * 0.2)
    
    # Coherencia OHLCV
    max_oc = np.maximum(new_open, new_close)
    min_oc = np.minimum(new_open, new_close)
    new_high = np.maximum(new_high, max_oc * 1.001)
    new_low = np.minimum(new_low, min_oc * 0.999)
    new_low = np.minimum(new_low, new_high)
    
    min_price = 0.01
    new_open = np.maximum(new_open, min_price)
    new_high = np.maximum(new_high, min_price)
    new_low = np.maximum(new_low, min_price)
    new_close = np.maximum(new_close, min_price)
    
    result = df.with_columns([
        pl.Series("open", new_open),
        pl.Series("high", new_high),
        pl.Series("low", new_low),
        pl.Series("close", new_close),
    ])
    
    return result


def _reconstruct_ohlcv_from_close(df: pl.DataFrame, old_close: np.ndarray, new_close: np.ndarray) -> pl.DataFrame:
    """
    Reconstruye OHLCV manteniendo la FORMA de cada vela.
    
    La forma de la vela (proporción open/high/low respecto al close)
    se preserva, pero escalada al nuevo nivel de precios.
    """
    n = len(new_close)
    
    old_open = df["open"].to_numpy().astype(np.float64)
    old_high = df["high"].to_numpy().astype(np.float64)
    old_low = df["low"].to_numpy().astype(np.float64)
    
    # Calcular ratios originales (forma de la vela)
    # Evitar división por cero
    old_close_safe = np.where(old_close == 0, 1, old_close)
    
    open_ratio = old_open / old_close_safe
    high_ratio = old_high / old_close_safe
    low_ratio = old_low / old_close_safe
    
    # Aplicar ratios al nuevo close
    new_open = new_close * open_ratio
    new_high = new_close * high_ratio
    new_low = new_close * low_ratio
    
    # Asegurar coherencia OHLCV
    min_price = 0.01
    new_open = np.maximum(new_open, min_price)
    new_high = np.maximum(new_high, min_price)
    new_low = np.maximum(new_low, min_price)
    new_close = np.maximum(new_close, min_price)
    
    max_oc = np.maximum(new_open, new_close)
    min_oc = np.minimum(new_open, new_close)
    new_high = np.maximum(new_high, max_oc)
    new_low = np.minimum(new_low, min_oc)
    new_low = np.minimum(new_low, new_high)
    
    result = df.with_columns([
        pl.Series("open", new_open),
        pl.Series("high", new_high),
        pl.Series("low", new_low),
        pl.Series("close", new_close),
    ])
    
    return result


def shuffle_returns(df: pl.DataFrame, seed: int | None = None) -> pl.DataFrame:
    """
    Genera mercado sintético shuffleando los retornos.
    
    Mantiene la distribución estadística de retornos pero cambia la secuencia.
    El precio inicial se preserva y se reconstruye la serie desde ahí.
    """
    rng = np.random.default_rng(seed)
    
    close = df["close"].to_numpy().astype(np.float64)
    
    # Calcular retornos logarítmicos
    log_returns = np.diff(np.log(close))
    
    # Shuffle de retornos
    rng.shuffle(log_returns)
    
    # Reconstruir precios desde el precio inicial
    initial_price = close[0]
    new_close = np.zeros(len(close))
    new_close[0] = initial_price
    new_close[1:] = initial_price * np.exp(np.cumsum(log_returns))
    
    # Calcular ratios para aplicar a OHLV
    ratio = new_close / np.where(close == 0, 1, close)
    
    result = df.with_columns([
        (pl.col("open") * pl.Series(ratio)).alias("open"),
        (pl.col("high") * pl.Series(ratio)).alias("high"),
        (pl.col("low") * pl.Series(ratio)).alias("low"),
        pl.Series("close", new_close),
    ])
    
    # Corregir coherencia OHLCV
    open_arr = result["open"].to_numpy()
    high_arr = result["high"].to_numpy()
    low_arr = result["low"].to_numpy()
    close_arr = result["close"].to_numpy()
    
    max_oc = np.maximum(open_arr, close_arr)
    min_oc = np.minimum(open_arr, close_arr)
    high_arr = np.maximum(high_arr, max_oc)
    low_arr = np.minimum(low_arr, min_oc)
    low_arr = np.minimum(low_arr, high_arr)
    
    result = result.with_columns([
        pl.Series("high", high_arr),
        pl.Series("low", low_arr),
    ])
    
    return result


# =============================================================================
# MONTE CARLO RUNNER CON NSGA-II
# =============================================================================

class MonteCarloRunner:
    """
    Runner de optimización Monte Carlo con NSGA-II (Multi-Objetivo).
    
    CADA TRIAL USA UN MERCADO SINTÉTICO DIFERENTE.
    """
    
    def __init__(
        self,
        strategy: Strategy,
        config: BacktestConfig,
        mc_config: MonteCarloConfig,
        reporters: List[Reporter] | None = None,
        df: pl.DataFrame | None = None,
    ):
        self.strategy = strategy
        self.config = config
        self.mc_config = mc_config
        self.reporters = reporters or []
        
        self._df_base = df
        self._synthetic_markets: Dict[int, pl.DataFrame] = {}
        
        # Estado de optimización
        self.best_trial: TrialArtifacts | None = None
        self.all_trials: List[TrialArtifacts] = []
        self.profitable_count = 0
        self.pareto_front: List[TrialArtifacts] = []
        
        self.activo = "ASSET"
    
    def _get_synthetic_market(self, trial_number: int) -> pl.DataFrame:
        """Obtiene el mercado sintético único para un trial."""
        if trial_number not in self._synthetic_markets:
            seed = (self.mc_config.seed or 42) + trial_number
            
            self._synthetic_markets[trial_number] = generate_synthetic_market(
                self._df_base,
                method=self.mc_config.method,
                noise_pct=self.mc_config.noise_pct,
                noise_range=self.mc_config.noise_range,
                block_size=self.mc_config.block_size,
                seed=seed
            )
        
        return self._synthetic_markets[trial_number]
    
    def _create_nsga2_objective(self) -> Callable[[optuna.Trial], tuple[float, float]]:
        """
        Crea función objetivo para NSGA-II (Multi-Objetivo).
        
        Retorna: (quality_to_maximize, drawdown_to_minimize)
        """
        strategy = self.strategy
        config = self.config
        
        def objective(trial: optuna.Trial) -> tuple[float, float]:
            t0 = time.perf_counter()
            
            # Mercado sintético único para este trial
            synthetic_df = self._get_synthetic_market(trial.number)
            
            # Sugerir parámetros
            params = strategy.suggest_params(trial)
            
            # Resolver configuración de salida
            exit_settings = resolve_exit_settings_for_trial(trial=trial, config=config)
            
            # Inyectar parámetros runtime
            params_rt = dict(params)
            params_rt["__saldo_inicial"] = float(config.saldo_inicial)
            params_rt["__saldo_operativo_max"] = float(config.saldo_operativo_max)
            params_rt["__qty_max_activo"] = float(config.qty_max_activo)
            params_rt["__comision_pct"] = float(config.comision_pct)
            params_rt["__comision_sides"] = int(config.comision_sides)
            params_rt["__saldo_usado"] = float(config.saldo_usado)
            params_rt["__apalancamiento_max"] = float(config.apalancamiento_max)
            params_rt["__strategy_exit_enabled"] = bool(getattr(strategy, "SALIDAS_PERSONALIZADAS", False))
            params_rt["__activo"] = self.activo  # Para ExcelReporter
            
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
            
            # Generar señales
            try:
                signals_df = SignalGenerator.generate_signals(
                    synthetic_df,
                    strategy,
                    params_rt,
                    {},
                )
            except Exception:
                return (0.1, 100.0)  # (min quality, max drawdown)
            
            if signals_df is None or len(signals_df) == 0:
                return (0.1, 100.0)
            
            # Ejecutar backtest
            try:
                trades_df, equity_curve, metrics = BacktestEngine.run_backtest(
                    synthetic_df,
                    signals_df,
                    config,
                    params_rt,
                    strategy,
                )
            except Exception:
                return (0.1, 100.0)
            
            if trades_df.is_empty():
                return (0.1, 100.0)
            
            # Calcular objetivos NSGA-II
            quality, drawdown = nsga2_objectives(metrics)
            
            # Score combinado para reporting (usa el sistema tradicional)
            score = float(score_optuna(metrics))
            
            elapsed = time.perf_counter() - t0
            
            # Crear artifacts - incluir params_rt completo para reporters
            params_display = {k: v for k, v in params_rt.items() if not k.startswith("__")}
            params_display["__activo"] = self.activo  # Incluir activo para ExcelReporter
            
            # Generar df_signals para PlotReporter si el score es bueno
            df_signals_for_plot = None
            for reporter in self.reporters:
                if hasattr(reporter, "needs_dataframe") and reporter.needs_dataframe(score):
                    ohlc_cols = ["timestamp", "open", "high", "low", "close", "volume"]
                    base_cols = [c for c in ohlc_cols if c in synthetic_df.columns]
                    signal_cols = [c for c in signals_df.columns if c not in base_cols]
                    df_signals_for_plot = synthetic_df.select(base_cols).hstack(
                        signals_df.select(signal_cols)
                    )
                    break
            
            artifact = TrialArtifacts(
                strategy_name=f"{strategy.name} [MC #{trial.number}:{self.mc_config.method}]",
                trial_number=trial.number,
                params=params_rt,
                params_reporting=params_display,
                score=score,
                metrics=metrics,
                df_signals=df_signals_for_plot,
                trades=trades_df.to_pandas() if hasattr(trades_df, 'to_pandas') else trades_df,
                equity_curve=equity_curve,
                indicators_used=params_rt.get("__indicators_used", []),
                perturbado=True,
                perturb_seed=(self.mc_config.seed or 42) + trial.number,
            )
            
            self.all_trials.append(artifact)
            
            # Actualizar mejor (por score combinado)
            is_best = False
            if self.best_trial is None or score > self.best_trial.score:
                self.best_trial = artifact
                is_best = True
            
            # Contar rentables
            pnl = metrics.get("pnl_neto", 0)
            if pnl is None:
                pnl = 0
            if pnl > 0:
                self.profitable_count += 1
            
            # Notificar reporters
            self._notify_reporters(artifact, is_best)
            
            # Retornar objetivos: (quality, drawdown)
            return (quality, drawdown)
        
        return objective
    
    def _create_single_objective(self) -> Callable[[optuna.Trial], float]:
        """
        Crea función objetivo single-objective (TPE tradicional).
        """
        strategy = self.strategy
        config = self.config
        
        def objective(trial: optuna.Trial) -> float:
            t0 = time.perf_counter()
            
            synthetic_df = self._get_synthetic_market(trial.number)
            params = strategy.suggest_params(trial)
            exit_settings = resolve_exit_settings_for_trial(trial=trial, config=config)
            
            params_rt = dict(params)
            params_rt["__saldo_inicial"] = float(config.saldo_inicial)
            params_rt["__saldo_operativo_max"] = float(config.saldo_operativo_max)
            params_rt["__qty_max_activo"] = float(config.qty_max_activo)
            params_rt["__comision_pct"] = float(config.comision_pct)
            params_rt["__comision_sides"] = int(config.comision_sides)
            params_rt["__saldo_usado"] = float(config.saldo_usado)
            params_rt["__apalancamiento_max"] = float(config.apalancamiento_max)
            params_rt["__strategy_exit_enabled"] = bool(getattr(strategy, "SALIDAS_PERSONALIZADAS", False))
            
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
            
            try:
                signals_df = SignalGenerator.generate_signals(
                    synthetic_df, strategy, params_rt, {},
                )
            except Exception:
                return float("-inf")
            
            if signals_df is None or len(signals_df) == 0:
                return float("-inf")
            
            try:
                trades_df, equity_curve, metrics = BacktestEngine.run_backtest(
                    synthetic_df, signals_df, config, params_rt, strategy,
                )
            except Exception:
                return float("-inf")
            
            if trades_df.is_empty():
                return float("-inf")
            
            score = float(score_optuna(metrics))
            
            params_display = {k: v for k, v in params_rt.items() if not k.startswith("__")}
            
            artifact = TrialArtifacts(
                strategy_name=f"{strategy.name} [MC #{trial.number}:{self.mc_config.method}]",
                trial_number=trial.number,
                params=params_rt,
                params_reporting=params_display,
                score=score,
                metrics=metrics,
                df_signals=None,
                trades=trades_df.to_pandas() if hasattr(trades_df, 'to_pandas') else trades_df,
                equity_curve=equity_curve,
                indicators_used=params_rt.get("__indicators_used", []),
                perturbado=True,
                perturb_seed=(self.mc_config.seed or 42) + trial.number,
            )
            
            self.all_trials.append(artifact)
            
            is_best = False
            if self.best_trial is None or score > self.best_trial.score:
                self.best_trial = artifact
                is_best = True
            
            pnl = metrics.get("pnl_neto", 0)
            if pnl is None:
                pnl = 0
            if pnl > 0:
                self.profitable_count += 1
            
            self._notify_reporters(artifact, is_best)
            
            return score
        
        return objective
    
    def _notify_reporters(self, artifact: TrialArtifacts, is_best: bool):
        """Notifica a todos los reporters del nuevo trial."""
        for reporter in self.reporters:
            if hasattr(reporter, "on_trial_end"):
                try:
                    reporter.on_trial_end(artifact)
                except Exception as e:
                    pass  # No romper el flujo
    
    def run(self) -> Dict[str, Any]:
        """
        Ejecuta la optimización Monte Carlo.
        """
        if self._df_base is None:
            raise ValueError("No se proporcionó DataFrame de datos")
        
        # Reset estado
        self.best_trial = None
        self.all_trials = []
        self.profitable_count = 0
        self._synthetic_markets = {}
        self.pareto_front = []
        
        # Notificar inicio
        for reporter in self.reporters:
            if hasattr(reporter, "on_strategy_start"):
                try:
                    reporter.on_strategy_start(
                        f"{self.strategy.name} [Monte Carlo]",
                        self.mc_config.n_trials
                    )
                except Exception:
                    pass
        
        # Crear estudio según modo
        if self.mc_config.use_nsga2:
            # NSGA-II Multi-Objetivo
            sampler = NSGAIISampler(seed=self.mc_config.seed)
            study = optuna.create_study(
                directions=["maximize", "minimize"],  # quality ↑, drawdown ↓
                sampler=sampler,
            )
            objective = self._create_nsga2_objective()
        else:
            # TPE Single-Objective
            sampler = TPESampler(seed=self.mc_config.seed)
            study = optuna.create_study(
                direction="maximize",
                sampler=sampler,
            )
            objective = self._create_single_objective()
        
        # Ejecutar optimización
        start_time = time.perf_counter()
        
        study.optimize(
            objective,
            n_trials=self.mc_config.n_trials,
            show_progress_bar=False,
        )
        
        total_time = time.perf_counter() - start_time
        
        # Extraer frente de Pareto si usamos NSGA-II
        if self.mc_config.use_nsga2:
            self._extract_pareto_front(study)
        
        # Mostrar resumen de robustez
        self._show_robustness_summary()
        
        # Calcular robustez
        robustness_pct = (self.profitable_count / self.mc_config.n_trials) * 100 if self.mc_config.n_trials > 0 else 0
        
        # Resultado
        result = {
            "strategy_name": f"{self.strategy.name} [MC: {robustness_pct:.1f}% robustez]",
            "best_trial": self.best_trial,
            "all_trials": self.all_trials,
            "pareto_front": self.pareto_front,
            "robustness_pct": robustness_pct,
            "profitable_count": self.profitable_count,
            "total_trials": self.mc_config.n_trials,
            "total_time": total_time,
            "mc_config": self.mc_config,
            "study": study,
        }
        
        # Notificar fin a reporters
        for reporter in self.reporters:
            if hasattr(reporter, "on_strategy_end"):
                try:
                    reporter.on_strategy_end(self.strategy.name, study)
                except Exception:
                    pass
        
        # Limpiar cache
        self._synthetic_markets = {}
        gc.collect()
        
        return result
    
    def _extract_pareto_front(self, study: optuna.Study):
        """Extrae los trials del frente de Pareto."""
        try:
            pareto_trials = study.best_trials  # Trials en el frente de Pareto
            pareto_numbers = {t.number for t in pareto_trials}
            self.pareto_front = [t for t in self.all_trials if t.trial_number in pareto_numbers]
        except Exception:
            self.pareto_front = []
    
    def _show_robustness_summary(self):
        """Muestra resumen de robustez Monte Carlo."""
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        
        console = Console()
        
        if not self.all_trials:
            return
        
        # Estadísticas
        n_total = len(self.all_trials)
        n_profitable = self.profitable_count
        robustness = (n_profitable / n_total) * 100 if n_total > 0 else 0
        
        rois = []
        scores = []
        drawdowns = []
        for t in self.all_trials:
            roi = t.metrics.get("roi", 0)
            if roi is None:
                roi = 0
            rois.append(roi)
            scores.append(t.score)
            dd = t.metrics.get("drawdown", 0)
            if dd is None:
                dd = 0
            drawdowns.append(dd)
        
        avg_roi = np.mean(rois) if rois else 0
        std_roi = np.std(rois) if rois else 0
        avg_score = np.mean(scores) if scores else 0
        avg_dd = np.mean(drawdowns) if drawdowns else 0
        
        # Tabla principal
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Métrica", style="white")
        table.add_column("Valor", justify="right", style="green")
        table.add_column("Interpretación", style="dim")
        
        sampler_name = "NSGA-II" if self.mc_config.use_nsga2 else "TPE"
        table.add_row("Sampler", sampler_name, "Multi-Objetivo" if self.mc_config.use_nsga2 else "Single-Objetivo")
        table.add_row("Total Trials", str(n_total), f"= {n_total} mercados sintéticos únicos")
        table.add_row("Rentables", f"{n_profitable}/{n_total}", f"({robustness:.1f}%)")
        
        if robustness >= 70:
            rob_style = "[bold green]"
        elif robustness >= 50:
            rob_style = "[bold yellow]"
        else:
            rob_style = "[bold red]"
        
        table.add_row("ROBUSTEZ", f"{rob_style}{robustness:.1f}%[/]", "≥70% = Robusto")
        table.add_row("", "", "")
        table.add_row("ROI Promedio", f"+{avg_roi:.2f}%" if avg_roi > 0 else f"{avg_roi:.2f}%", "Todos los trials")
        table.add_row("ROI Std Dev", f"{std_roi:.2f}%", "Menor = Más estable")
        table.add_row("Drawdown Promedio", f"{avg_dd:.2f}%", "Menor = Mejor")
        table.add_row("Score Promedio", f"{avg_score:.3f}", "")
        
        # Frente de Pareto
        if self.pareto_front:
            table.add_row("", "", "")
            table.add_row("[bold cyan]FRENTE DE PARETO[/]", f"{len(self.pareto_front)} trials", "Óptimos multi-objetivo")
        
        panel = Panel(
            table,
            title=f"📊 MONTE CARLO ROBUSTNESS ({sampler_name}) - Un mercado único por trial",
            border_style="blue",
        )
        
        console.print()
        console.print(panel)
        
        # Veredicto
        if robustness >= 70:
            verdict = "[bold green]✓ ESTRATEGIA ROBUSTA[/] - Funciona bien en múltiples condiciones de mercado"
        elif robustness >= 50:
            verdict = "[bold yellow]⚠ ROBUSTEZ MODERADA[/] - Resultados mixtos, considerar ajustes"
        else:
            verdict = "[bold red]✗ ESTRATEGIA FRÁGIL[/] - Alto riesgo de overfitting, revisar parámetros"
        
        console.print(Panel(verdict, title="VEREDICTO", border_style="white"))
        
        # Top 5 trials
        sorted_trials = sorted(self.all_trials, key=lambda t: t.score, reverse=True)[:5]
        
        top_table = Table(title="TOP 5 TRIALS", show_header=True, header_style="bold")
        top_table.add_column("#", justify="right")
        top_table.add_column("Score", justify="right")
        top_table.add_column("ROI", justify="right")
        top_table.add_column("Win%", justify="right")
        top_table.add_column("DD%", justify="right")
        top_table.add_column("Trades")
        top_table.add_column("Método")
        
        for t in sorted_trials:
            # CORRECCIÓN: usar las claves correctas de métricas
            roi_val = t.metrics.get("roi", 0)
            if roi_val is None:
                roi_val = 0
            
            winrate_val = t.metrics.get("winrate", 0)  # NO "win_rate"
            if winrate_val is None:
                winrate_val = 0
            
            dd_val = t.metrics.get("drawdown", 0)  # NO "max_drawdown"
            if dd_val is None:
                dd_val = 0
            
            trades_val = t.metrics.get("total_trades", 0)
            if trades_val is None:
                trades_val = t.metrics.get("n_trades", 0)
            if trades_val is None:
                trades_val = 0
            
            top_table.add_row(
                str(t.trial_number),
                f"{t.score:.2f}",
                f"+{roi_val:.1f}%" if roi_val > 0 else f"{roi_val:.1f}%",
                f"{winrate_val:.1f}%",
                f"{dd_val:.1f}%",
                str(trades_val),
                self.mc_config.method,
            )
        
        console.print()
        console.print(top_table)
        
        # Si hay frente de Pareto, mostrarlo
        if self.pareto_front and len(self.pareto_front) > 0:
            console.print()
            pareto_table = Table(title="🏆 FRENTE DE PARETO (Óptimos Multi-Objetivo)", show_header=True, header_style="bold magenta")
            pareto_table.add_column("#", justify="right")
            pareto_table.add_column("Quality", justify="right")
            pareto_table.add_column("DD%", justify="right")
            pareto_table.add_column("ROI", justify="right")
            pareto_table.add_column("Win%", justify="right")
            pareto_table.add_column("Trades")
            
            # Ordenar por quality descendente
            sorted_pareto = sorted(self.pareto_front, key=lambda t: t.score, reverse=True)[:10]
            
            for t in sorted_pareto:
                roi_val = t.metrics.get("roi", 0) or 0
                winrate_val = t.metrics.get("winrate", 0) or 0
                dd_val = t.metrics.get("drawdown", 0) or 0
                trades_val = t.metrics.get("total_trades", 0) or t.metrics.get("n_trades", 0) or 0
                quality = score_quality_only(t.metrics)
                
                pareto_table.add_row(
                    str(t.trial_number),
                    f"{quality:.2f}",
                    f"{dd_val:.1f}%",
                    f"+{roi_val:.1f}%" if roi_val > 0 else f"{roi_val:.1f}%",
                    f"{winrate_val:.1f}%",
                    str(trades_val),
                )
            
            console.print(pareto_table)
