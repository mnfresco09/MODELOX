from __future__ import annotations

from typing import Any, Dict

import numpy as np
import polars as pl


def _zscore_alma(values: np.ndarray, *, window: int, sigma: float, offset: float) -> np.ndarray:
    """Z-Score con ALMA (media y desviación ponderadas gausianas)."""

    n = int(values.shape[0])
    out = np.full(n, np.nan, dtype=np.float64)
    if window < 2 or n < window:
        return out

    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 6.0
    if not np.isfinite(offset):
        offset = 0.85
    offset = float(np.clip(offset, 0.0, 1.0))

    m = offset * (window - 1)
    s = window / sigma
    if s <= 0:
        return out

    i = np.arange(window, dtype=np.float64)
    w = np.exp(-((i - m) ** 2) / (2.0 * (s**2)))
    w_sum = float(w.sum())
    if not np.isfinite(w_sum) or w_sum == 0.0:
        return out
    W = w / w_sum

    for idx in range(window - 1, n):
        win = values[idx - window + 1 : idx + 1]
        if np.isnan(win).any():
            continue
        mu = float(np.dot(W, win))
        var = float(np.dot(W, (win - mu) ** 2))
        if not np.isfinite(var) or var <= 0.0:
            out[idx] = 0.0
            continue
        sd = float(np.sqrt(var))
        out[idx] = (float(values[idx]) - mu) / sd

    return out


def _linreg_slope(values: np.ndarray, *, window: int) -> np.ndarray:
    """Pendiente de regresión lineal rolling (x=0..window-1)."""

    n = int(values.shape[0])
    out = np.full(n, np.nan, dtype=np.float64)
    if window < 2 or n < window:
        return out

    x = np.arange(window, dtype=np.float64)
    sum_x = float(x.sum())
    sum_x2 = float((x * x).sum())
    denom = window * sum_x2 - (sum_x * sum_x)
    if denom == 0.0:
        return out

    for idx in range(window - 1, n):
        win = values[idx - window + 1 : idx + 1]
        if np.isnan(win).any():
            continue
        sum_y = float(win.sum())
        sum_xy = float(np.dot(x, win))
        out[idx] = (window * sum_xy - sum_x * sum_y) / denom

    return out


class Strategy1ZScoreSlope:
    """Z-SCORE + SLOPE

    Reglas (según tu descripción):
    - Define umbral simétrico de ZScore: +thr / -thr, con thr en [1.5..2.0].

    LONG:
    1) Cuando ZScore cruza hacia arriba +thr:
       - si en ese punto la tendencia era creciente (slope > 0), se arma el long.
    2) Tras el cruce, se espera a que el slope cambie a decreciente (cruce a <0).
    3) Se confirma la entrada LONG en el cambio de slope siempre que ZScore > 0.

    SHORT (inverso):
    1) Cuando ZScore cruza hacia abajo -thr:
       - si en ese punto la tendencia era decreciente (slope < 0), se arma el short.
    2) Tras el cruce, se espera a que el slope cambie a creciente (cruce a >0).
    3) Se confirma la entrada SHORT en el cambio de slope siempre que ZScore < 0.
    """

    combinacion_id = 1
    name = "Z-SCORE + SLOPE"

    parametros_optuna: Dict[str, Any] = {
        "z_window": (20, 80, 1),
        "z_sigma": (4.0, 9.0, 0.5),
        "z_offset": (0.60, 0.95, 0.05),
        "z_thr": (1.5, 2.0, 0.1),
        "slope_window": (10, 80, 1),
    }

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        z_window = int(trial.suggest_int("z_window", 20, 80))
        z_sigma = float(trial.suggest_float("z_sigma", 4.0, 9.0, step=0.5))
        z_offset = float(trial.suggest_float("z_offset", 0.60, 0.95, step=0.05))
        z_thr = float(trial.suggest_float("z_thr", 1.5, 2.0, step=0.1))
        slope_window = int(trial.suggest_int("slope_window", 10, 80))
        return {
            "z_window": z_window,
            "z_sigma": z_sigma,
            "z_offset": z_offset,
            "z_thr": z_thr,
            "slope_window": slope_window,
        }

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        z_window = max(2, int(params.get("z_window", 50)))
        z_sigma = float(params.get("z_sigma", 6.0))
        z_offset = float(params.get("z_offset", 0.85))
        z_thr = float(params.get("z_thr", 1.8))
        slope_window = max(2, int(params.get("slope_window", 20)))

        params["__warmup_bars"] = max(z_window, slope_window) + 10

        # Bounds/specs para plot (por trial)
        params["__indicator_bounds"] = {
            "zscore_alma": {"hi": z_thr, "lo": -z_thr, "mid": 0.0},
            "slope": {"mid": 0.0},
        }
        params["__indicator_specs"] = {
            "zscore_alma": {
                "panel": "sub",
                "type": "line",
                "name": f"ZScore ALMA ({z_window}, σ={z_sigma:g}, off={z_offset:g})",
                "precision": 3,
            },
            "slope": {
                "panel": "sub",
                "type": "line",
                "name": f"Slope ({slope_window})",
                "precision": 6,
            },
        }

        params["__indicators_used"] = ["zscore_alma", "slope"]

        close = np.asarray(df["close"].to_numpy(), dtype=np.float64)
        z = _zscore_alma(close, window=z_window, sigma=z_sigma, offset=z_offset)
        sl = _linreg_slope(close, window=slope_window)
        df = df.with_columns(
            [
                pl.Series("zscore_alma", z).cast(pl.Float64),
                pl.Series("slope", sl).cast(pl.Float64),
            ]
        )

        # Señales (state machine)
        z = np.asarray(df["zscore_alma"].to_numpy(), dtype=np.float64)
        sl = np.asarray(df["slope"].to_numpy(), dtype=np.float64)
        n = len(z)

        sig_long = np.zeros(n, dtype=np.bool_)
        sig_short = np.zeros(n, dtype=np.bool_)

        armed_long = False
        armed_short = False

        for i in range(1, n):
            if not (np.isfinite(z[i]) and np.isfinite(z[i - 1]) and np.isfinite(sl[i]) and np.isfinite(sl[i - 1])):
                continue

            # --- LONG arm: z crosses above +thr and slope is rising at cross ---
            z_cross_up = (z[i - 1] <= z_thr) and (z[i] > z_thr)
            if z_cross_up and (sl[i] > 0.0):
                armed_long = True

            # --- SHORT arm: z crosses below -thr and slope is falling at cross ---
            z_cross_dn = (z[i - 1] >= -z_thr) and (z[i] < -z_thr)
            if z_cross_dn and (sl[i] < 0.0):
                armed_short = True

            # --- Confirm LONG: slope turns down and z stays > 0 ---
            if armed_long:
                slope_turn_down = (sl[i - 1] >= 0.0) and (sl[i] < 0.0)
                if slope_turn_down and (z[i] > 0.0):
                    sig_long[i] = True
                    armed_long = False

            # --- Confirm SHORT: slope turns up and z stays < 0 ---
            if armed_short:
                slope_turn_up = (sl[i - 1] <= 0.0) and (sl[i] > 0.0)
                if slope_turn_up and (z[i] < 0.0):
                    sig_short[i] = True
                    armed_short = False

        return df.with_columns(
            [
                pl.Series("signal_long", sig_long).cast(pl.Boolean),
                pl.Series("signal_short", sig_short).cast(pl.Boolean),
            ]
        )
