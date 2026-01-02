from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import polars as pl

# ==========================================
# HELPER: ALMA (Arnaud Legoux Moving Average)
# ==========================================
def _calculate_alma_weights(window: int, offset: float, sigma: float) -> np.ndarray:
    """Genera los pesos gaussianos para el filtro ALMA."""
    m = offset * (window - 1)
    s = window / sigma
    k = np.arange(window)
    weights = np.exp(-((k - m) ** 2) / (2 * s * s))
    return weights / weights.sum()

def _apply_alma(series: pl.Series, window: int, offset: float = 0.85, sigma: float = 6.0) -> pl.Series:
    """Aplica ALMA usando convolución numpy."""
    arr = series.to_numpy()
    if len(arr) < window:
        return pl.Series(np.full(len(arr), np.nan))
    
    weights = _calculate_alma_weights(window, offset, sigma)
    
    # Convolución modo 'valid' devuelve N - W + 1
    # Invertimos pesos porque convolve es una suma rotada
    result_valid = np.convolve(arr, weights[::-1], mode='valid')
    
    # Rellenamos el inicio con NaNs (warmup)
    pad = np.full(window - 1, np.nan)
    full_result = np.concatenate([pad, result_valid])
    
    return pl.Series(full_result)

# ==========================================
# HELPER: ADX (Average Directional Index)
# ==========================================
def _add_adx(df: pl.DataFrame, length: int, smooth: int) -> pl.DataFrame:
    """Calcula el ADX y lo añade al DataFrame."""
    # True Range
    high = pl.col("high")
    low = pl.col("low")
    close = pl.col("close")
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pl.max_horizontal(tr1, tr2, tr3)
    
    # Directional Movement
    up = high - high.shift(1)
    down = low.shift(1) - low
    
    plus_dm = pl.when((up > down) & (up > 0)).then(up).otherwise(0.0)
    minus_dm = pl.when((down > up) & (down > 0)).then(down).otherwise(0.0)
    
    # Suavizado (RMA / Wilder's Smoothing es standard para ADX)
    # RMA(x, n) = EWM(x, alpha=1/n)
    atr = tr.ewm_mean(alpha=1.0/length, min_periods=length)
    smooth_plus = plus_dm.ewm_mean(alpha=1.0/length, min_periods=length)
    smooth_minus = minus_dm.ewm_mean(alpha=1.0/length, min_periods=length)
    
    di_plus = (smooth_plus / atr) * 100
    di_minus = (smooth_minus / atr) * 100
    
    dx = (di_plus - di_minus).abs() / (di_plus + di_minus) * 100
    adx = dx.ewm_mean(alpha=1.0/smooth, min_periods=smooth)
    
    return df.with_columns(adx.alias("adx"))

# ==========================================
# ESTRATEGIA ID 10
# ==========================================
class Strategy10MomentumFusion:
    """
    Estrategia Momentum Fusion (NormPrice + Vol + ADX)
    
    LÓGICA:
    1. Calcula Precio Normalizado (-2 a +2) en ventana rolling.
    2. Calcula Volumen Normalizado (0.5 a 1.5) en ventana rolling.
    3. Fusión = PrecioNorm * VolNorm.
    4. Señal = ALMA(Fusión).
    5. Filtro = ADX > Umbral.
    
    ENTRADAS:
    - LONG: Señal cruza 0 hacia ARRIBA + ADX fuerte + ADX subiendo.
    - SHORT: Señal cruza 0 hacia ABAJO + ADX fuerte.
    """
    
    combinacion_id = 11
    name = "MOMENTUM_FUSION_ADX_ALMA"
    
    # Configuración por defecto para Optuna
    parametros_optuna: Dict[str, Any] = {
        "len_price": (20, 100, 5),
        "len_vol": (20, 100, 5),
        "alma_len": (5, 30, 1),
        "adx_len": (7, 21, 1),
        "adx_thr": (15, 30, 1),
    }

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        return {
            "len_price": trial.suggest_int("len_price", 20, 80, step=5),
            "len_vol": trial.suggest_int("len_vol", 20, 80, step=5),
            "alma_len": trial.suggest_int("alma_len", 8, 25),
            "adx_len": trial.suggest_int("adx_len", 7, 21),
            "adx_smooth": trial.suggest_int("adx_smooth", 7, 14),
            "adx_thr": trial.suggest_int("adx_thr", 15, 35),
        }

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        # 1. Recuperar parámetros
        len_price = int(params.get("len_price", 50))
        len_vol = int(params.get("len_vol", 50))
        alma_len = int(params.get("alma_len", 15))
        adx_len = int(params.get("adx_len", 14))
        adx_smooth = int(params.get("adx_smooth", 14))
        adx_thr = float(params.get("adx_thr", 20.0))

        # 2. Definir Warmup (Crítico para que el Engine no opere antes de tiempo)
        # Necesitamos el máximo histórico requerido
        max_lookback = max(len_price, len_vol, adx_len + adx_smooth) + alma_len + 10
        params["__warmup_bars"] = max_lookback

        # 3. Configuración Gráfica (Metadata para visual/grafico.py)
        params["__indicators_used"] = ["fusion_signal"]
        params["__indicator_bounds"] = {
            "fusion_signal": {"hi": 2.0, "lo": -2.0, "mid": 0.0}
        }
        
        # Opcional: Especificaciones avanzadas si el sistema lo soporta
        params["__indicator_specs"] = {
            "fusion_signal": {
                "panel": "sub",
                "name": "Trinity Fusion (ALMA)",
                "precision": 2
            }
        }

        # -----------------------------------------------------------
        # A. CÁLCULO DE ADX
        # -----------------------------------------------------------
        df = _add_adx(df, length=adx_len, smooth=adx_smooth)
        
        # -----------------------------------------------------------
        # B. NORMALIZACIÓN DE PRECIO (-2 a +2)
        # -----------------------------------------------------------
        # Rolling min/max
        p_max = pl.col("close").rolling_max(window_size=len_price, min_periods=len_price)
        p_min = pl.col("close").rolling_min(window_size=len_price, min_periods=len_price)
        p_denom = p_max - p_min
        
        # Evitar división por cero
        norm_price = -2.0 + ((pl.col("close") - p_min) * 4.0 / pl.when(p_denom == 0).then(1.0).otherwise(p_denom))
        
        # -----------------------------------------------------------
        # C. FACTOR DE VOLUMEN (0.5 a 1.5)
        # -----------------------------------------------------------
        # Usamos SMA(3) del volumen primero como suavizado ligero
        v_smooth_raw = pl.col("volume").rolling_mean(window_size=3, min_periods=1)
        
        v_max = v_smooth_raw.rolling_max(window_size=len_vol, min_periods=len_vol)
        v_min = v_smooth_raw.rolling_min(window_size=len_vol, min_periods=len_vol)
        v_denom = v_max - v_min
        
        # Normalización volumen
        norm_vol = 0.5 + ((v_smooth_raw - v_min) * 1.0 / pl.when(v_denom == 0).then(1.0).otherwise(v_denom))
        
        # Protección si el activo no tiene volumen (Forex/CFD) -> Factor neutro 1.0
        # Chequeamos si la columna volumen es todo null o ceros, o lo manejamos fila a fila
        # Si volume es null, norm_vol será null. Lo reemplazamos por 1.0
        norm_vol = norm_vol.fill_null(1.0)

        # -----------------------------------------------------------
        # D. FUSIÓN Y ALMA
        # -----------------------------------------------------------
        fusion_raw_expr = norm_price * norm_vol
        
        # Añadimos columna temporal para calcular ALMA sobre ella
        df = df.with_columns(fusion_raw_expr.alias("__fusion_raw"))
        
        # Calculamos ALMA sobre la columna __fusion_raw usando el helper numpy
        fusion_series = df["__fusion_raw"]
        alma_series = _apply_alma(fusion_series, window=alma_len, offset=0.85, sigma=6.0)
        
        df = df.with_columns([
            alma_series.alias("fusion_signal"),
            # Limpieza de columnas temporales no es estrictamente necesaria pero ordenado
        ])
        
        # -----------------------------------------------------------
        # E. LÓGICA DE SEÑALES
        # -----------------------------------------------------------
        sig = pl.col("fusion_signal")
        adx = pl.col("adx")
        
        # Condiciones base
        cross_up = (sig > 0) & (sig.shift(1) <= 0)
        cross_down = (sig < 0) & (sig.shift(1) >= 0)
        
        adx_strong = adx > adx_thr
        adx_rising = adx > adx.shift(1)
        
        # LONG: Cruce Up + ADX > Thr + ADX Subiendo
        signal_long = cross_up & adx_strong & adx_rising
        
        # SHORT: Cruce Down + ADX > Thr (No requerimos subiendo explícitamente en short según tu spec, solo fuerza)
        signal_short = cross_down & adx_strong
        
        return df.with_columns([
            signal_long.fill_null(False).alias("signal_long"),
            signal_short.fill_null(False).alias("signal_short")
        ])
    