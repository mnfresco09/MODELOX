from __future__ import annotations
from typing import Any, Dict
import polars as pl
import numpy as np
from numba import jit
from .ESTRATEGIA_BASE import EstrategiaBase


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES NUMBA JIT (COMPILADAS A VELOCIDAD C)
# ══════════════════════════════════════════════════════════════════════════════

@jit(nopython=True, cache=True)
def _hurst_rs_numba(returns: np.ndarray, min_periods: int) -> float:
    """
    NÚCLEO DEL CÁLCULO HURST CON NUMBA JIT (VELOCIDAD C)
    
    OPTIMIZACIONES:
    - Compilado a código máquina con Numba
    - Solo 3-4 lags en lugar de todos los posibles (suficiente para regresión)
    - Loops optimizados para cache CPU
    """
    n = len(returns)
    
    if n < 16:  # Mínimo absoluto
        return np.nan
    
    # Generar solo 3-4 lags estratégicos (suficiente para regresión lineal)
    # Espaciados logarítmicamente para mejor cobertura
    max_k = int(np.floor(np.log2(n)))
    if max_k < 3:
        return np.nan
    
    # Lags: aproximadamente 4, 8, 16, 32... (potencias de 2)
    num_lags = min(4, max_k - 1)
    if num_lags < min_periods:
        return np.nan
    
    lags = np.empty(num_lags, dtype=np.int64)
    for i in range(num_lags):
        k = 2 + i
        lags[i] = int(2 ** k)
        if lags[i] >= n:
            return np.nan
    
    # Arrays para R/S
    rs_values = np.empty(num_lags, dtype=np.float64)
    log_lags = np.empty(num_lags, dtype=np.float64)
    
    valid_count = 0
    
    # Calcular R/S para cada lag
    for lag_idx in range(num_lags):
        lag = lags[lag_idx]
        num_segments = n // lag
        
        if num_segments < 1:
            continue
        
        rs_sum = 0.0
        rs_count = 0
        
        # Procesar cada segmento
        for seg_idx in range(num_segments):
            start = seg_idx * lag
            end = start + lag
            
            # Media del segmento
            mean_val = 0.0
            for j in range(start, end):
                mean_val += returns[j]
            mean_val /= lag
            
            # Desviaciones acumuladas y varianza
            cumdev_max = -np.inf
            cumdev_min = np.inf
            cumdev = 0.0
            var_sum = 0.0
            
            for j in range(start, end):
                dev = returns[j] - mean_val
                cumdev += dev
                if cumdev > cumdev_max:
                    cumdev_max = cumdev
                if cumdev < cumdev_min:
                    cumdev_min = cumdev
                var_sum += dev * dev
            
            # Rango R
            R = cumdev_max - cumdev_min
            
            # Desviación estándar S
            S = np.sqrt(var_sum / (lag - 1)) if lag > 1 else 1.0
            
            # R/S válido
            if S > 1e-10 and R > 1e-10:
                rs_sum += R / S
                rs_count += 1
        
        if rs_count > 0:
            rs_values[valid_count] = rs_sum / rs_count
            log_lags[valid_count] = np.log(float(lag))
            valid_count += 1
    
    if valid_count < min_periods:
        return np.nan
    
    # Recortar arrays
    rs_values = rs_values[:valid_count]
    log_lags = log_lags[:valid_count]
    log_rs = np.log(rs_values)
    
    # REGRESIÓN LINEAL MANUAL (más rápida que polyfit en Numba)
    # y = a + b*x  =>  b = cov(x,y) / var(x)
    mean_x = 0.0
    mean_y = 0.0
    for i in range(valid_count):
        mean_x += log_lags[i]
        mean_y += log_rs[i]
    mean_x /= valid_count
    mean_y /= valid_count
    
    cov = 0.0
    var_x = 0.0
    for i in range(valid_count):
        dx = log_lags[i] - mean_x
        dy = log_rs[i] - mean_y
        cov += dx * dy
        var_x += dx * dx
    
    if var_x < 1e-10:
        return np.nan
    
    H = cov / var_x
    
    # Clip a [0, 1]
    if H < 0.0:
        H = 0.0
    elif H > 1.0:
        H = 1.0
    
    return H


@jit(nopython=True, cache=True)
def _interpolate_nans(arr: np.ndarray) -> np.ndarray:
    """
    INTERPOLACIÓN LINEAL DE NaNs (NUMBA JIT)
    
    Rellena valores NaN con interpolación lineal entre valores válidos.
    """
    n = len(arr)
    result = arr.copy()
    
    # Encontrar primer valor válido
    first_valid = -1
    for i in range(n):
        if not np.isnan(arr[i]):
            first_valid = i
            break
    
    if first_valid == -1:
        return result  # Todo NaN
    
    # Llenar valores antes del primer válido
    for i in range(first_valid):
        result[i] = arr[first_valid]
    
    # Interpolación lineal
    last_valid_idx = first_valid
    last_valid_val = arr[first_valid]
    
    for i in range(first_valid + 1, n):
        if not np.isnan(arr[i]):
            # Interpolar entre last_valid y current
            if i > last_valid_idx + 1:
                span = i - last_valid_idx
                for j in range(last_valid_idx + 1, i):
                    weight = (j - last_valid_idx) / span
                    result[j] = last_valid_val * (1 - weight) + arr[i] * weight
            last_valid_idx = i
            last_valid_val = arr[i]
    
    # Llenar valores después del último válido
    for i in range(last_valid_idx + 1, n):
        result[i] = last_valid_val
    
    return result


# ══════════════════════════════════════════════════════════════════════════════
# ESTRATEGIA: HURST EXPONENT (ID 3) - ANÁLISIS R/S (RESCALED RANGE)
# ══════════════════════════════════════════════════════════════════════════════

class StrategyHurstExponent(EstrategiaBase):
    """
    ESTRATEGIA BASADA EN EL EXPONENTE DE HURST
    
    El Exponente de Hurst (H) es una medida estadística que cuantifica la 
    "memoria" de una serie temporal y determina si el precio se encuentra en:
    - H < 0.5: Reversión a la media (mean-reverting)
    - H = 0.5: Caminata aleatoria (random walk)
    - H > 0.5: Tendencia persistente (trending)
    
    FUNDAMENTOS MATEMÁTICOS (R/S Analysis):
    ========================================
    Para una serie de rendimientos logarítmicos, el proceso divide la serie
    en sub-períodos de longitud n. Para cada sub-período:
    
    1. Calcular la desviación acumulada respecto a la media
    2. El rango (R) es la diferencia entre el valor máximo y mínimo
    3. Se divide por la desviación estándar (S) del período
    4. La relación fundamental: E[R/S] = C × n^H
    5. Aplicando logaritmos: log(R/S) = log(C) + H × log(n)
    6. La pendiente de la regresión lineal nos da H
    
    LÓGICA DE ENTRADA:
    ==================
    - LONG:  Cuando H cruza de <= 0.65 a > 0.65 (tendencia alcista persistente)
    - SHORT: Cuando H cruza de >= 0.35 a < 0.35 (reversión bajista fuerte)
    
    Esto permite:
    - Detectar cuando el mercado pasa de aleatorio/reversión a tendencia alcista
    - Detectar cuando el mercado entra en reversión bajista fuerte
    - Rechazar la Hipótesis del Mercado Eficiente cuando H ≠ 0.5
    """

    combinacion_id = 3
    name = "HURST_EXPONENT"
    SALIDAS_PERSONALIZADAS = False

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        """
        ESPACIO DE BÚSQUEDA PARA OPTUNA
        
        Parámetros:
        -----------
        window_size: Tamaño de la ventana para calcular el Hurst Exponent
                     (número de barras a considerar en el análisis R/S)
        
        min_periods: Número mínimo de sub-períodos para el análisis R/S
                     (debe ser >= 4 para tener suficientes puntos de regresión)
        
        threshold_offset: Desviación simétrica desde 0.5 para los umbrales
                         - long_threshold = 0.5 + threshold_offset
                         - short_threshold = 0.5 - threshold_offset
                         Ejemplo: offset=0.15 → long=0.65, short=0.35
        """
        window_size = trial.suggest_int("window_size", 50, 500, step=10)
        min_periods = trial.suggest_int("min_periods", 4, 12, step=1)
        threshold_offset = trial.suggest_float("threshold_offset", 0.05, 0.25, step=0.01)
        
        # Calcular thresholds simétricos con respecto a 0.5
        long_threshold = 0.5 + threshold_offset
        short_threshold = 0.5 - threshold_offset
        
        return {
            "window_size": window_size,
            "min_periods": min_periods,
            "threshold_offset": threshold_offset,
            "long_threshold": long_threshold,
            "short_threshold": short_threshold,
        }

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        """
        GENERADOR DE SEÑALES BASADO EN HURST EXPONENT
        
        OPTIMIZACIÓN DE VELOCIDAD:
        - Minimizar conversiones Polars <-> Numpy
        - Una sola conversión para calcular Hurst
        - Todo lo demás en Polars lazy evaluation
        - Un solo collect() al final (en finalize_signals)
        """
        # 1. INICIALIZACIÓN
        self._init_params_metadata(params)
        self._require_columns(df, ["timestamp", "close"])

        # Extracción de parámetros
        window_size = params["window_size"]
        min_periods = params["min_periods"]
        long_threshold = params["long_threshold"]
        short_threshold = params["short_threshold"]

        # Configuración de Metadata
        params["__warmup_bars"] = window_size + 50
        params["__indicators_used"] = ["hurst_exp"]
        params["__indicator_bounds"] = {
            "hurst_exp": {
                "lower": 0.0,
                "upper": 1.0,
                "mid": 0.5,
            }
        }
        params["__indicator_specs"] = {
            "hurst_exp": {
                "color": "#00FFFF",
                "type": "line",
                "panel": "indicator",
            }
        }

        # 2. CÁLCULO OPTIMIZADO DEL HURST EXPONENT
        # ----------------------------------------------------------------------
        # Conversión única y mínima a numpy para el cálculo R/S
        # (Este análisis estadístico complejo es más eficiente en numpy)
        close_prices = df["close"].to_numpy()
        hurst_values = self._calculate_rolling_hurst_ultra_fast(
            prices=close_prices,
            window_size=window_size,
            min_periods=min_periods
        )
        
        # 3. CONSTRUCCIÓN DE SEÑALES (TODO EN POLARS LAZY)
        # ----------------------------------------------------------------------
        q = df.lazy()
        
        # Añadir Hurst Exponent como literal (evita joins innecesarios)
        q = q.with_columns([
            pl.lit(hurst_values).alias("hurst_exp")
        ])
        
        # Hurst del período anterior para detectar cruces
        q = q.with_columns([
            pl.col("hurst_exp").shift(1).alias("hurst_exp_prev")
        ])
        
        # 4. LÓGICA DE SEÑALES (CRUCES SIMÉTRICOS)
        # ----------------------------------------------------------------------
        # LONG: H cruza hacia arriba del threshold (tendencia persistente)
        long_cond = (
            (pl.col("hurst_exp") > long_threshold) &
            (pl.col("hurst_exp_prev") <= long_threshold) &
            pl.col("hurst_exp").is_not_null() &
            pl.col("hurst_exp_prev").is_not_null()
        )
        
        # SHORT: H cruza hacia abajo del threshold (reversión fuerte)
        short_cond = (
            (pl.col("hurst_exp") < short_threshold) &
            (pl.col("hurst_exp_prev") >= short_threshold) &
            pl.col("hurst_exp").is_not_null() &
            pl.col("hurst_exp_prev").is_not_null()
        )
        
        # 5. APLICAR SEÑALES
        # ----------------------------------------------------------------------
        q = q.with_columns([
            self._as_bool(long_cond).alias("signal_long"),
            self._as_bool(short_cond).alias("signal_short"),
        ])
        
        # 6. RETORNO (UN SOLO COLLECT EN finalize_signals)
        # ----------------------------------------------------------------------
        return self.finalize_signals(q, keep_cols=["hurst_exp"])

    @staticmethod
    def _calculate_rolling_hurst_ultra_fast(
        prices: np.ndarray,
        window_size: int,
        min_periods: int = 4
    ) -> np.ndarray:
        """
        VERSIÓN ULTRA-RÁPIDA DEL CÁLCULO DE HURST CON NUMBA JIT
        
        OPTIMIZACIONES EXTREMAS:
        - Numba JIT compilation para velocidad C
        - Cálculo cada N barras (no cada barra)
        - Interpolación lineal para barras intermedias
        - Menos lags pero suficientes para precisión
        
        Parámetros:
        -----------
        prices: Array de precios
        window_size: Tamaño de la ventana deslizante
        min_periods: Mínimo número de sub-períodos
        
        Retorna:
        --------
        Array con valores del Hurst Exponent
        """
        n = len(prices)
        hurst_values = np.full(n, np.nan, dtype=np.float64)
        
        # Pre-calcular log returns (UNA SOLA VEZ)
        log_returns = np.diff(np.log(prices))
        
        # OPTIMIZACIÓN CRÍTICA: Calcular Hurst cada N barras en lugar de cada barra
        # Esto reduce el número de cálculos drasticamente
        stride = max(10, window_size // 20)  # Calcular cada 10 barras o 5% de la ventana
        
        # Calcular en puntos clave
        calculation_points = list(range(window_size, n, stride))
        if calculation_points[-1] != n - 1:
            calculation_points.append(n - 1)  # Asegurar el último punto
        
        for i in calculation_points:
            window_returns = log_returns[i - window_size:i]
            h = _hurst_rs_numba(window_returns, min_periods)
            if not np.isnan(h):
                hurst_values[i] = h
        
        # Interpolación lineal para barras intermedias
        hurst_values = _interpolate_nans(hurst_values)
        
        return hurst_values
