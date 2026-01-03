from __future__ import annotations

from typing import Any, Dict

import numpy as np
import polars as pl


def _alma_numpy(values: np.ndarray, period: int, offset: float = 0.85, sigma: float = 6.0) -> np.ndarray:
    """ALMA (Arnaud Legoux Moving Average) en numpy.

    Implementación consistente con otras estrategias del repo.
    """
    n = len(values)
    alma = np.full(n, np.nan, dtype=np.float64)
    if period <= 1 or n == 0:
        return alma

    m = offset * (period - 1)
    s = period / float(sigma)

    weights = np.zeros(period, dtype=np.float64)
    for i in range(period):
        weights[i] = np.exp(-((i - m) ** 2) / (2.0 * s * s))
    wsum = float(weights.sum())
    if wsum != 0.0:
        weights = weights / wsum

    # Vectorizado: sliding_window_view + producto punto.
    # Esto es crítico para performance con 50k-100k velas.
    try:
        windows = np.lib.stride_tricks.sliding_window_view(values, window_shape=period)
        out = windows @ weights
        # Respetar NaNs (si existieran)
        if np.isnan(values).any():
            nan_mask = np.isnan(windows).any(axis=1)
            out = out.astype(np.float64, copy=False)
            out[nan_mask] = np.nan
        alma[period - 1 :] = out
    except Exception:
        # Fallback seguro (lento) solo si algo falla con strides.
        for i in range(period - 1, n):
            window = values[i - period + 1 : i + 1]
            if not np.isnan(window).any():
                alma[i] = float(np.sum(window * weights))

    return alma


class StrategyZScoreNormalizacion:
    """ESTRATEGIA: Z-SCORE + NORMALIZACIÓN (ALMA)

    Reglas (según especificación del usuario):

    Parámetros Optuna:
    - Umbral Z simétrico: z_thr ∈ [1.5, 2.5] step 0.2  -> rangos: [-z_thr, +z_thr]
    - Ventana Z-score: z_window ∈ [20, 100] step 5
    - ALMA window: alma_window ∈ [20, 100] step 5
    - Normalización window: norm_window ∈ [20, 100] step 5
    - Umbral del indicador normalizado simétrico: norm_thr ∈ [2.0, 3.0] step 0.1 -> [-norm_thr, +norm_thr]
    - Ventana de espera: wait_bars ∈ [1, 4] step 1

    Indicadores:
    - z_score: Z-score de close
    - alma: ALMA(close)
    - norm_price: normalización tipo z-score del (close - alma), recortado a [-3, 3]

    Entradas LONG:
    1) Activación: z_score cruza el rango bajo hacia arriba (cruza -z_thr)
    2) Se arma una ventana de espera de `wait_bars`
    3) Dentro de esa ventana: norm_price cruza el rango bajo hacia arriba (cruza -norm_thr)
    4) En la vela donde (3) se cumple: si z_score < 0 -> signal_long

    Inverso para SHORT:
    1) Activación: z_score cruza el rango alto hacia abajo (cruza +z_thr)
    2) Ventana de espera
    3) Dentro: norm_price cruza el rango alto hacia abajo (cruza +norm_thr)
    4) En la vela donde (3) se cumple: si z_score > 0 -> signal_short

    Nota:
    - Se garantiza simetría de rangos en cada trial usando un solo parámetro por umbral.
    """

    combinacion_id = 11
    name = "Z_SCORE + NORMALIZACION"

    parametros_optuna: Dict[str, Any] = {
        "z_window": (10, 100, 5),
        "z_thr": (1.3, 3.0, 0.1),
        "alma_window": (5, 100, 5),
        "norm_window": (50, 100, 5),
        "norm_thr": (2.0, 3.0, 0.1),
        "wait_bars": (1, 10, 1),
    }

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        return {
            "z_window": trial.suggest_int("z_window", 10, 100, step=5),
            "z_thr": trial.suggest_float("z_thr", 1.3, 3.0, step=0.1),
            "alma_window": trial.suggest_int("alma_window", 5, 100, step=5),
            "norm_window": trial.suggest_int("norm_window", 50, 100, step=5),
            "norm_thr": trial.suggest_float("norm_thr", 2.0, 3.0, step=0.1),
            "wait_bars": trial.suggest_int("wait_bars", 1, 10, step=1),
        }

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        z_window = int(params.get("z_window", 50))
        z_thr = float(params.get("z_thr", 2.0))
        alma_window = int(params.get("alma_window", 50))
        norm_window = int(params.get("norm_window", 50))
        norm_thr = float(params.get("norm_thr", 2.5))
        wait_bars = int(params.get("wait_bars", 2))

        z_thr = float(np.clip(z_thr, 1.5, 2.5))
        norm_thr = float(np.clip(norm_thr, 2.0, 3.0))
        wait_bars = int(max(1, min(4, wait_bars)))

        # Rangos simétricos
        z_lo = -z_thr
        z_hi = z_thr
        n_lo = -norm_thr
        n_hi = norm_thr

        # Warmup suficiente para rolling std + alma
        warmup = int(max(z_window, alma_window, norm_window) + 5)
        params["__warmup_bars"] = warmup

        # Indicadores a plotear
        params["__indicators_used"] = ["z_score", "norm_price", "alma"]
        params["__indicator_bounds"] = {
            "z_score": {"hi": z_hi, "lo": z_lo, "mid": 0.0},
            "norm_price": {"hi": n_hi, "lo": n_lo, "mid": 0.0},
        }
        # Especificar explícitamente qué panel usar para cada indicador
        params["__indicator_specs"] = {
            "z_score": {"panel": "sub", "name": "Z-SCORE", "color": "#60a5fa"},
            "norm_price": {"panel": "sub", "name": "NORM", "color": "#f472b6"},
            "alma": {"panel": "overlay", "name": "ALMA", "color": "#fbbf24"},
        }

        # --- Z-SCORE de close (Polars rolling) ---
        close = pl.col("close")
        z_mean = close.rolling_mean(window_size=z_window, min_periods=z_window)
        z_std = close.rolling_std(window_size=z_window, min_periods=z_window)
        z_score = ((close - z_mean) / z_std).fill_nan(None)

        # --- ALMA + normalización del precio alrededor de ALMA ---
        close_np = df.select(pl.col("close")).to_numpy().flatten().astype(np.float64)
        alma_np = _alma_numpy(close_np, alma_window)
        alma_s = pl.Series("alma", alma_np)

        diff = (pl.col("close") - alma_s)
        d_mean = diff.rolling_mean(window_size=norm_window, min_periods=norm_window)
        d_std = diff.rolling_std(window_size=norm_window, min_periods=norm_window)
        norm_raw = ((diff - d_mean) / d_std).fill_nan(None)

        # Recortar entre -3 y 3
        norm_price = norm_raw.clip(-3.0, 3.0)

        # Materializar indicadores en df
        df2 = df.with_columns(
            [
                z_score.alias("z_score"),
                alma_s.alias("alma"),
                norm_price.alias("norm_price"),
            ]
        )

        # Señales: necesitamos estado (ventana de espera), hacemos loop numpy rápido.
        z_np = df2.select(pl.col("z_score")).to_numpy().flatten()
        n_np = df2.select(pl.col("norm_price")).to_numpy().flatten()
        n = len(z_np)

        signal_long = np.zeros(n, dtype=bool)
        signal_short = np.zeros(n, dtype=bool)

        pending_long_until = -1
        pending_short_until = -1

        for i in range(1, n):
            if i < warmup:
                continue

            z_prev = z_np[i - 1]
            z_cur = z_np[i]
            n_prev = n_np[i - 1]
            n_cur = n_np[i]

            if np.isnan(z_prev) or np.isnan(z_cur) or np.isnan(n_prev) or np.isnan(n_cur):
                continue

            # 1) Activación LONG: z cruza -z_thr hacia arriba
            if (z_prev < z_lo) and (z_cur >= z_lo):
                pending_long_until = i + wait_bars

            # 1) Activación SHORT: z cruza +z_thr hacia abajo
            if (z_prev > z_hi) and (z_cur <= z_hi):
                pending_short_until = i + wait_bars

            # 3) Condición dentro de la ventana LONG: norm cruza -norm_thr hacia arriba
            if pending_long_until >= i:
                if (n_prev < n_lo) and (n_cur >= n_lo):
                    # 4) si al cumplirse (3), z_score < 0 => long
                    if z_cur < 0.0:
                        signal_long[i] = True
                        pending_long_until = -1

            # 3) Condición dentro de la ventana SHORT: norm cruza +norm_thr hacia abajo
            if pending_short_until >= i:
                if (n_prev > n_hi) and (n_cur <= n_hi):
                    if z_cur > 0.0:
                        signal_short[i] = True
                        pending_short_until = -1

            # Expirar ventanas
            if pending_long_until != -1 and i > pending_long_until:
                pending_long_until = -1
            if pending_short_until != -1 and i > pending_short_until:
                pending_short_until = -1

        df2 = df2.with_columns(
            [
                pl.Series("signal_long", signal_long).alias("signal_long"),
                pl.Series("signal_short", signal_short).alias("signal_short"),
            ]
        )

        # Nunca dejar nulls en señales
        df2 = df2.with_columns(
            [
                pl.col("signal_long").fill_null(False),
                pl.col("signal_short").fill_null(False),
            ]
        )

        return df2
