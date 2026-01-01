from __future__ import annotations
from typing import Any, Dict, Tuple
import math
import numpy as np
import polars as pl

# ==========================================
# 1. MOTOR MATEMÁTICO: SAVITZKY-GOLAY CAUSAL (CORREGIDO)
# ==========================================
# NOTA: Esta versión elimina el Lookahead Bias. Solo mira hacia atrás.

def _savgol_coeffs_causal(window: int, poly: int, deriv: int) -> np.ndarray:
    """
    Calcula coeficientes SG asimétricos (causales).
    El punto de evaluación es t=0 (el último dato), y la ventana se extiende hacia el pasado.
    """
    # El eje x va desde -(window-1) hasta 0. Ej para window=5: [-4, -3, -2, -1, 0]
    x = np.arange(-(window - 1), 1, dtype=np.float64)
    
    # Matriz de Vandermonde
    A = np.vander(x, N=poly + 1, increasing=True)
    
    # Pseudoinversa
    pinv = np.linalg.pinv(A)
    
    # El coeficiente que nos interesa es para evaluar en t=0 (el final de la ventana)
    coeff = pinv[deriv] * math.factorial(deriv)
    
    # Invertimos para usar con np.convolve, ya que la convolución invierte el kernel
    return coeff[::-1]

def _savgol_filter_causal(y: np.ndarray, window: int, poly: int, deriv: int = 0) -> np.ndarray:
    """
    Aplica el filtro SG de manera causal.
    Los primeros (window-1) valores serán NaN porque no hay suficiente historia.
    """
    n = len(y)
    if window > n:
        # Si la ventana es mayor que los datos, devolvemos todo NaNs o ceros
        return np.full(n, np.nan)

    coeffs = _savgol_coeffs_causal(window, poly, deriv)
    
    # 'valid' devuelve array de tamaño N - W + 1. 
    # Es el resultado de aplicar la ventana solo donde cabe completa.
    result_valid = np.convolve(y, coeffs, mode='valid')
    
    # Rellenamos el inicio con NaNs para mantener la longitud original del array
    # Esto representa el periodo de "warm-up" donde no hay suficientes velas pasadas.
    pad_len = window - 1
    pad = np.full(pad_len, np.nan)
    
    return np.concatenate([pad, result_valid])

# ==========================================
# 2. MOTOR MATEMÁTICO: ROLLING OLS (Vectorizado)
# ==========================================
def _rolling_ols_metrics(y: np.ndarray, window: int):
    """
    Calcula Beta, R2, SER y Y_hat vectorizado.
    Maneja NaNs de entrada propagándolos.
    """
    n = len(y)
    
    # Si hay NaNs en la entrada (por el filtro SG causal), debemos tener cuidado.
    # Esta implementación asume que los NaNs están al principio.
    
    x = np.arange(window, dtype=np.float64)
    
    sum_x = x.sum()
    sum_x2 = (x**2).sum()
    n_w = float(window)
    denom = n_w * sum_x2 - sum_x**2
    
    # Sliding Window View
    # Nota: Si y contiene NaNs, las sumas resultarán en NaN, lo cual es correcto (no hay señal).
    y_strided = np.lib.stride_tricks.sliding_window_view(y, window_shape=window)
    
    sum_y = y_strided.sum(axis=1)
    sum_xy = (y_strided * x).sum(axis=1)
    sum_y2 = (y_strided**2).sum(axis=1)
    
    # Beta
    beta = (n_w * sum_xy - sum_x * sum_y) / denom
    
    # Alpha
    alpha = (sum_y - beta * sum_x) / n_w
    
    # Predicción (Y_hat) en el último punto
    y_hat_last = alpha + beta * (window - 1)
    
    # Estadísticas
    sse = sum_y2 - alpha*sum_y - beta*sum_xy
    sse = np.maximum(sse, 0) # Clip flotante negativo
    sst = sum_y2 - (sum_y**2)/n_w
    
    with np.errstate(divide='ignore', invalid='ignore'):
        r2 = 1 - (sse / sst)
        ser = np.sqrt(sse / (window - 2))
    
    # Limpieza de NaNs/Infs generados por división por cero
    r2 = np.nan_to_num(r2, nan=0.0)
    ser = np.nan_to_num(ser, nan=0.0)
    
    # Padding inicial
    pad = np.full(window - 1, np.nan)
    
    return (
        np.concatenate([pad, beta]),
        np.concatenate([pad, r2]),
        np.concatenate([pad, ser]),
        np.concatenate([pad, y_hat_last])
    )

# ==========================================
# 3. ESTRATEGIA: CINEMÁTICA SIMÉTRICA (ID 2)
# ==========================================

class Strategy2KinematicDivergence:
    """
    Estrategia basada en Física de Mercado (Cinemática) + Estadística (OLS).
    VERSIÓN CORREGIDA: Usa filtros causales para evitar Lookahead Bias.
    """
    
    combinacion_id = 2
    name = "KINEMATIC DIVERGENCE (SG+OLS) [CAUSAL]"

    parametros_optuna: Dict[str, Any] = {
        "sg_window": (7, 35, 2),    
        "sg_poly": (2, 4, 1),       
        "ols_window": (30, 100, 5), 
        "r2_min": (0.60, 0.90, 0.05), 
        "z_thr": (1.2, 2.5, 0.1),   
    }

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        sg_window = trial.suggest_int("sg_window", 7, 35, step=2)
        sg_poly = trial.suggest_int("sg_poly", 2, min(4, sg_window - 2))
        ols_window = trial.suggest_int("ols_window", 30, 100, step=5)
        r2_min = trial.suggest_float("r2_min", 0.60, 0.90, step=0.05)
        z_thr = trial.suggest_float("z_thr", 1.2, 2.5, step=0.1)

        return {
            "sg_window": sg_window,
            "sg_poly": sg_poly,
            "ols_window": ols_window,
            "r2_min": r2_min,
            "z_thr": z_thr,
        }

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        sg_window = int(params.get("sg_window", 21))
        sg_poly = int(params.get("sg_poly", 3))
        ols_window = int(params.get("ols_window", 50))
        r2_min = float(params.get("r2_min", 0.7))
        z_thr = float(params.get("z_thr", 1.5))

        # Ajuste de seguridad
        if sg_window % 2 == 0: sg_window += 1
        if sg_poly >= sg_window: sg_poly = sg_window - 2

        # Metadata
        params["__indicators_used"] = ["velocity", "acceleration", "beta", "r2", "z_score"]
        params["__indicator_bounds"] = {
            "velocity": {"mid": 0.0},
            "acceleration": {"mid": 0.0},
            "z_score": {"hi": z_thr, "lo": -z_thr, "mid": 0.0}
        }

        close = df["close"].to_numpy().astype(np.float64)

        # -----------------------------------------------------------
        # A. PRE-PROCESAMIENTO CINEMÁTICO (CAUSAL)
        # -----------------------------------------------------------
        # Usamos _savgol_filter_causal para no mirar al futuro
        
        # P_smooth: Posición suavizada
        p_smooth = _savgol_filter_causal(close, window=sg_window, poly=sg_poly, deriv=0)
        # Velocity
        v = _savgol_filter_causal(close, window=sg_window, poly=sg_poly, deriv=1)
        # Acceleration
        a = _savgol_filter_causal(close, window=sg_window, poly=sg_poly, deriv=2)

        # Rellenar NaNs resultantes del filtro (los primeros N datos)
        # Es preferible dejarlos como NaN para no operar, pero si el sistema requiere float:
        p_smooth = np.nan_to_num(p_smooth, nan=close[0]) # Fallback al primer precio
        v = np.nan_to_num(v, nan=0.0)
        a = np.nan_to_num(a, nan=0.0)

        # -----------------------------------------------------------
        # B. ANÁLISIS ESTRUCTURAL (Rolling OLS)
        # -----------------------------------------------------------
        beta, r2, ser, y_hat_smooth = _rolling_ols_metrics(p_smooth, window=ols_window)

        # -----------------------------------------------------------
        # C. TRIGGER ESTADÍSTICO
        # -----------------------------------------------------------
        resid = close - y_hat_smooth
        ser_safe = np.where(ser == 0, 1e-9, ser)
        z_score = resid / ser_safe
        
        # Limpieza final de NaNs en z_score (generados por el warm-up del OLS)
        z_score = np.nan_to_num(z_score, nan=0.0)

        # -----------------------------------------------------------
        # D. LÓGICA DE DECISIÓN
        # -----------------------------------------------------------
        bull_regime = (beta > 0) & (r2 > r2_min)
        bear_regime = (beta < 0) & (r2 > r2_min)

        positive_inertia = a > 0
        negative_inertia = a < 0

        oversold = z_score < -z_thr
        overbought = z_score > z_thr

        # Señales Crudas
        raw_long = bull_regime & positive_inertia & oversold
        raw_short = bear_regime & negative_inertia & overbought

        # Disparadores (Flanco de subida)
        n = len(close)
        sig_long = np.zeros(n, dtype=bool)
        sig_short = np.zeros(n, dtype=bool)

        sig_long[1:] = raw_long[1:] & (~raw_long[:-1])
        sig_short[1:] = raw_short[1:] & (~raw_short[:-1])

        return df.with_columns([
            pl.Series("p_smooth", p_smooth).cast(pl.Float64),
            pl.Series("velocity", v).cast(pl.Float64),
            pl.Series("acceleration", a).cast(pl.Float64),
            pl.Series("beta", beta).cast(pl.Float64),
            pl.Series("r2", r2).cast(pl.Float64),
            pl.Series("z_score", z_score).cast(pl.Float64),
            pl.Series("signal_long", sig_long).cast(pl.Boolean),
            pl.Series("signal_short", sig_short).cast(pl.Boolean),
        ])