from __future__ import annotations
from typing import Any, Dict
import numpy as np
import polars as pl

def _rolling_linreg_slope(y: np.ndarray, window: int) -> np.ndarray:
    """Calcula la pendiente (Slope) de la Regresión Lineal Rolling."""
    n = len(y)
    x = np.arange(window, dtype=np.float64)
    sum_x = x.sum()
    sum_x2 = (x**2).sum()
    denom = float(window) * sum_x2 - sum_x**2
    if denom == 0.0:
        return np.full(n, np.nan)

    # Rolling view
    y_strided = np.lib.stride_tricks.sliding_window_view(y, window_shape=window)
    sum_y = y_strided.sum(axis=1)
    sum_xy = (y_strided * x).sum(axis=1)
    
    # Beta (Pendiente)
    beta = (float(window) * sum_xy - sum_x * sum_y) / denom
    
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, beta])

class StrategyZScorePullback:
    """
    ESTRATEGIA: Z-SCORE RECOVERY + LINREG TREND
    
    Lógica (Reincorporación a Tendencia):
    - LONG: 
        1. Z-Score cruza -2.0 hacia ARRIBA (Recuperación desde sobreventa).
        2. La Regresión Lineal tiene pendiente POSITIVA (Tendencia Alcista).
        
    - SHORT: 
        1. Z-Score cruza +2.0 hacia ABAJO (Corrección desde sobrecompra).
        2. La Regresión Lineal tiene pendiente NEGATIVA (Tendencia Bajista).
    """
    
    combinacion_id = 9
    name = "Z-SCORE TREND PULLBACK"

    parametros_optuna: Dict[str, Any] = {
        "ema_len": (10, 50, 5),          # Suavizado del Z-Score
        "z_lookback": (20, 100, 10),     # Ventana estadística
        "linreg_len": (50, 200, 10),     # Tendencia de fondo
        "z_threshold": (1.5, 3.0, 0.1),  # Gatillo (ej. 2.0)
    }

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        return {
            "ema_len": trial.suggest_int("ema_len", 10, 50, step=5),
            "z_lookback": trial.suggest_int("z_lookback", 20, 100, step=10),
            "linreg_len": trial.suggest_int("linreg_len", 50, 200, step=10),
            "z_threshold": trial.suggest_float("z_threshold", 1.5, 3.0, step=0.1),
        }

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        ema_len = int(params.get("ema_len", 20))
        z_lookback = int(params.get("z_lookback", 50))
        linreg_len = int(params.get("linreg_len", 100))
        thr = float(params.get("z_threshold", 2.0))

        # Metadata Visual
        params["__indicators_used"] = ["z_score", "linreg_slope"]
        params["__indicator_bounds"] = {"z_score": {"hi": thr, "lo": -thr, "mid": 0.0}}

        # 1. Z-SCORE
        ema = df["close"].ewm_mean(span=ema_len, min_periods=ema_len)
        z_mean = ema.rolling_mean(window_size=z_lookback, min_periods=z_lookback)
        z_std = ema.rolling_std(window_size=z_lookback, min_periods=z_lookback)
        z_score = (ema - z_mean) / z_std
        z_score = z_score.fill_nan(0.0)

        # 2. LINEAR REGRESSION SLOPE (Tendencia)
        close_np = df["close"].to_numpy()
        slope_vals = _rolling_linreg_slope(close_np, linreg_len)
        slope_series = pl.Series("linreg_slope", slope_vals).fill_nan(0.0)

        # 3. SEÑALES CORREGIDAS
        
        # LONG: Cruce -2 hacia ARRIBA + Slope > 0
        z_cross_up = (z_score.shift(1) < -thr) & (z_score > -thr)
        trend_up = slope_series > 0
        signal_long = z_cross_up & trend_up

        # SHORT: Cruce +2 hacia ABAJO + Slope < 0
        z_cross_down = (z_score.shift(1) > thr) & (z_score < thr)
        trend_down = slope_series < 0
        signal_short = z_cross_down & trend_down

        return df.with_columns([
            z_score.alias("z_score"),
            slope_series.alias("linreg_slope"),
            signal_long.fill_null(False).alias("signal_long"),
            signal_short.fill_null(False).alias("signal_short")
        ])