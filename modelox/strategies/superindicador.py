from __future__ import annotations
from typing import Any, Dict
import numpy as np
import polars as pl

def _rolling_linreg_price(y: np.ndarray, window: int) -> np.ndarray:
    """Calcula el precio estimado (Lineal Regression) en el último punto."""
    n = len(y)
    x = np.arange(window, dtype=np.float64)
    sum_x = x.sum()
    sum_x2 = (x**2).sum()
    n_w = float(window)
    denom = n_w * sum_x2 - sum_x**2
    
    # Rolling view
    y_strided = np.lib.stride_tricks.sliding_window_view(y, window_shape=window)
    sum_y = y_strided.sum(axis=1)
    sum_xy = (y_strided * x).sum(axis=1)
    
    beta = (n_w * sum_xy - sum_x * sum_y) / denom
    alpha = (sum_y - beta * sum_x) / n_w
    
    # Y_hat en el punto actual (t = window - 1)
    y_hat = alpha + beta * (window - 1)
    
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, y_hat])

class StrategyZScoreLinReg:
    """
    ESTRATEGIA: Z-SCORE + LINEAR REGRESSION MEAN REVERSION
    
    Reglas:
    LONG: 
      1. Precio < Linear Regression (Estamos 'baratos').
      2. Z-Score (del EMA) cruza -2.0 hacia arriba (Gatillo de rebote).
      
    SHORT:
      1. Precio > Linear Regression (Estamos 'caros').
      2. Z-Score (del EMA) cruza +2.0 hacia abajo.
    """
    
    combinacion_id = 7
    name = "Z-SCORE + LINREG REVERSION"

    parametros_optuna: Dict[str, Any] = {
        "ema_len": (10, 50, 5),          # Suavizado del precio
        "z_lookback": (20, 100, 10),     # Ventana del Z-Score
        "linreg_len": (50, 200, 10),     # Tendencia central (Regresión)
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
        z_lookback = int(params.get("z_lookback", 30))
        linreg_len = int(params.get("linreg_len", 100))
        thr = float(params.get("z_threshold", 2.0))

        # Configuración Visual
        params["__indicators_used"] = ["z_score", "linreg"]
        params["__indicator_bounds"] = {"z_score": {"hi": thr, "lo": -thr, "mid": 0.0}}
        params["__indicator_specs"] = {
            "linreg": {"panel": "overlay", "color": "orange"}
        }

        # 1. CALCULAR Z-SCORE DE LA EMA (Como en tu script original)
        ema = df["close"].ewm_mean(span=ema_len, min_periods=ema_len)
        z_mean = ema.rolling_mean(window_size=z_lookback, min_periods=z_lookback)
        z_std = ema.rolling_std(window_size=z_lookback, min_periods=z_lookback)
        z_score = (ema - z_mean) / z_std
        z_score = z_score.fill_nan(0.0)

        # 2. CALCULAR LINEAR REGRESSION (Ubicación)
        close_np = df["close"].to_numpy()
        linreg_vals = _rolling_linreg_price(close_np, linreg_len)
        linreg_series = pl.Series("linreg", linreg_vals).fill_nan(close_np[0])

        # 3. LÓGICA DE SEÑALES
        
        # CONDICIÓN DE UBICACIÓN
        # Precio barato (Long) o caro (Short)
        below_reg = df["close"] < linreg_series
        above_reg = df["close"] > linreg_series

        # CONDICIÓN DE GATILLO (Z-Score Cross)
        # Cruce hacia arriba de -Threshold
        z_cross_up = (z_score.shift(1) < -thr) & (z_score > -thr)
        
        # Cruce hacia abajo de +Threshold
        z_cross_down = (z_score.shift(1) > thr) & (z_score < thr)

        # SEÑAL FINAL
        signal_long = below_reg & z_cross_up
        signal_short = above_reg & z_cross_down

        return df.with_columns([
            z_score.alias("z_score"),
            linreg_series.alias("linreg"),
            signal_long.fill_null(False).alias("signal_long"),
            signal_short.fill_null(False).alias("signal_short")
        ])