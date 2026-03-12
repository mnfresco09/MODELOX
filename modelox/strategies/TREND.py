from __future__ import annotations
"""
================================================================================
MODELOX/STRATEGIES/ID2.PY — OPTIMIZADO NUMBA
================================================================================
ID        : 2
NOMBRE    : RLOG — Regresión Lineal OLS con Filtro R²
MERCADO   : Crypto (BTC, ETH, etc.)
TIMEFRAME : 1H
--------------------------------------------------------------------------------
OPTIMIZACIÓN NUMBA JIT:
  El loop OLS rolling se compila con @njit(cache=True) de Numba.
  - Primera llamada: ~1-2s de compilación (una sola vez por sesión)
  - Llamadas siguientes: velocidad C puro, 20-100x más rápido que Python
  - Misma lógica exacta, mismos outputs, misma precisión numérica
  - cache=True: persiste compilación entre sesiones en disco
  - Fallback automático a Python puro si Numba no está instalado

LÓGICA DE ENTRADA:
  - Precio medio OHLC = (O+H+L+C) / 4
  - OLS rolling → β₁ (pendiente), R², σ_residuos, línea ajustada
  - Canal ± dev_mult * σ
  - SEÑAL LONG  : β₁_norm > slope_threshold AND R² ≥ r2_min  [FLANCO]
  - SEÑAL SHORT : β₁_norm < -slope_threshold AND R² ≥ r2_min [FLANCO]

RANGOS OPTUNA (2 parámetros):
  ols_window : int   [6, 50]      step=2
  r2_min     : float [0.70, 0.95] step=0.05

PARÁMETROS FIJOS:
  dev_multiplier=1.5 · r2_smooth=1 · slope_threshold=0.02
================================================================================
"""

from typing import Any, Dict, List
import polars as pl
import numpy as np

# ── Numba: importación defensiva con fallback ──────────────────────────────────
try:
    from numba import njit as _njit
    _NUMBA_AVAILABLE = True
except ImportError:
    def _njit(*args, **kwargs):          # type: ignore[misc]
        def _deco(fn):
            return fn
        return _deco if (args and callable(args[0])) else _deco
    _NUMBA_AVAILABLE = False

from modelox.strategies.ESTRATEGIA_BASE import EstrategiaBase


# ==============================================================================
# NÚCLEO OLS COMPILADO — nivel módulo para que Numba lo compile UNA sola vez
# ==============================================================================
@_njit(cache=True)
def _ols_numba(
    arr:       np.ndarray,
    w:         int,
    x_mean:    float,
    x_var:     float,
    out_beta1: np.ndarray,
    out_r2:    np.ndarray,
    out_sigma: np.ndarray,
    out_y_hat: np.ndarray,
    out_y_bar: np.ndarray,
) -> None:
    """
    Loop OLS rolling compilado a C via Numba @njit.
    Modifica los 5 arrays de output in-place — cero allocations extra.
    x_mean y x_var se reciben precalculados (constantes para ventana fija).
    """
    n_tot = len(arr)

    for i in range(w - 1, n_tot):

        # ── Media de y en ventana ──────────────────────────────────────────
        y_sum = 0.0
        for k in range(w):
            y_sum += arr[i - w + 1 + k]
        y_mean = y_sum / w

        # ── β₁ = Σ(xᵢ-x̄)(yᵢ-ȳ) / Σ(xᵢ-x̄)² ──────────────────────────
        cov_xy = 0.0
        for k in range(w):
            cov_xy += (k - x_mean) * (arr[i - w + 1 + k] - y_mean)
        beta1 = cov_xy / x_var
        beta0 = y_mean - beta1 * x_mean

        # ── Valor ajustado en vela actual (x = w-1) ───────────────────────
        y_hat = beta0 + beta1 * (w - 1)

        # ── R² y σ_residuos ────────────────────────────────────────────────
        ss_res = 0.0
        ss_tot = 0.0
        for k in range(w):
            y_k    = arr[i - w + 1 + k]
            y_pred = beta0 + beta1 * k
            ss_res += (y_k - y_pred) ** 2
            ss_tot += (y_k - y_mean) ** 2

        if ss_tot == 0.0:
            r2 = 0.0
        else:
            r2 = 1.0 - ss_res / ss_tot
            if r2 < 0.0:
                r2 = 0.0
            elif r2 > 1.0:
                r2 = 1.0

        denom = w - 2
        sigma = (ss_res / denom) ** 0.5 if denom > 0 else 0.0

        out_beta1[i] = beta1
        out_r2[i]    = r2
        out_sigma[i] = sigma
        out_y_hat[i] = y_hat
        out_y_bar[i] = y_mean


class EstrategiaID2RLOG(EstrategiaBase):
    """
    RLOG — Regresión Lineal OLS con Filtro R²
    Loop OLS compilado con Numba @njit(cache=True) para máximo rendimiento.
    Fallback automático a Python puro si Numba no está disponible.
    """

    combinacion_id: int          = 3
    name: str                    = "TREND"
    SALIDAS_PERSONALIZADAS: bool = False

    # ==========================================================================
    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        return {
            "ols_window": trial.suggest_int  ("ols_window", 20, 120,   step=1),
            "r2_min":     trial.suggest_float("r2_min",     0.30, 0.80, step=0.01),
        }

    def get_required_timeframes(self, params: Dict[str, Any]) -> List[str]:
        return []

    # ==========================================================================
    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:

        self._init_params_metadata(params)
        self._require_columns(df, ["timestamp", "open", "high", "low", "close"])

        ols_window   = int  (params.get("ols_window",      50))
        r2_min       = float(params.get("r2_min",          0.75))
        dev_mult     = 1.5
        slope_thresh = float(params.get("slope_threshold", 0.02))
        r2_smooth    = 1

        params["__warmup_bars"]      = ols_window + r2_smooth + 1
        params["__indicators_used"]  = [
            "precio_medio", "ols_linea", "ols_upper", "ols_lower",
            "r2_raw", "r2_suav", "beta1_norm_pct",
        ]
        params["__indicator_specs"]  = {
            "precio_medio":   {"panel": "main", "color": "#AAAAAA", "tipo": "line"},
            "ols_linea":      {"panel": "main", "color": "#FFD600", "tipo": "line"},
            "ols_upper":      {"panel": "main", "color": "#42A5F5", "tipo": "line"},
            "ols_lower":      {"panel": "main", "color": "#42A5F5", "tipo": "line"},
            "r2_suav":        {"panel": "sub1", "color": "#00E676", "tipo": "line"},
            "r2_raw":         {"panel": "sub1", "color": "#00897B", "tipo": "line"},
            "beta1_norm_pct": {"panel": "sub2", "color": "#FF9800", "tipo": "histogram"},
        }
        params["__indicator_bounds"] = {
            "r2_suav":        {"lo": 0.0,  "hi": 1.0,  "mid": r2_min},
            "r2_raw":         {"lo": 0.0,  "hi": 1.0,  "mid": r2_min},
            "beta1_norm_pct": {"lo": None, "hi": None,  "mid": 0.0},
        }

        # ── FASE A: Precio medio OHLC ──────────────────────────────────────────
        q = df.lazy().with_columns([
            ((pl.col("open") + pl.col("high") + pl.col("low") + pl.col("close")) / 4.0)
            .alias("precio_medio")
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE B — OLS rolling via map_batches + Numba JIT
        # x_mean y x_var se precalculan UNA sola vez fuera del batch (constantes)
        # _ols_numba modifica arrays in-place → cero allocations dentro del loop
        # ══════════════════════════════════════════════════════════════════════
        w      = ols_window
        x      = np.arange(w, dtype=np.float64)
        x_mean = float(x.mean())
        x_var  = float(((x - x_mean) ** 2).sum())

        def _ols_batch(s: pl.Series) -> pl.Series:
            arr   = s.to_numpy(allow_copy=True).astype(np.float64)
            n_tot = len(arr)

            out_beta1 = np.full(n_tot, np.nan)
            out_r2    = np.full(n_tot, np.nan)
            out_sigma = np.full(n_tot, np.nan)
            out_y_hat = np.full(n_tot, np.nan)
            out_y_bar = np.full(n_tot, np.nan)

            _ols_numba(
                arr, w, x_mean, x_var,
                out_beta1, out_r2, out_sigma, out_y_hat, out_y_bar,
            )

            return pl.Series(
                name   = "_ols",
                values = [
                    {
                        "beta1":  float(out_beta1[i]),
                        "r2_raw": float(out_r2[i]),
                        "sigma":  float(out_sigma[i]),
                        "y_hat":  float(out_y_hat[i]),
                        "y_bar":  float(out_y_bar[i]),
                    }
                    for i in range(n_tot)
                ],
            )

        q = q.with_columns([
            pl.col("precio_medio")
            .map_batches(
                _ols_batch,
                return_dtype=pl.Struct({
                    "beta1":  pl.Float64,
                    "r2_raw": pl.Float64,
                    "sigma":  pl.Float64,
                    "y_hat":  pl.Float64,
                    "y_bar":  pl.Float64,
                }),
            )
            .alias("_ols")
        ])

        # ── FASE C: Desempaquetar struct ───────────────────────────────────────
        q = q.with_columns([
            pl.col("_ols").struct.field("beta1") .alias("_beta1"),
            pl.col("_ols").struct.field("r2_raw").alias("r2_raw"),
            pl.col("_ols").struct.field("sigma") .alias("_sigma"),
            pl.col("_ols").struct.field("y_hat") .alias("ols_linea"),
            pl.col("_ols").struct.field("y_bar") .alias("_y_bar"),
        ]).drop("_ols")

        # ── FASE D: Canal ± dev_mult * σ ───────────────────────────────────────
        q = q.with_columns([
            (pl.col("ols_linea") + dev_mult * pl.col("_sigma")).alias("ols_upper"),
            (pl.col("ols_linea") - dev_mult * pl.col("_sigma")).alias("ols_lower"),
        ])

        # ── FASE E: Pendiente normalizada ──────────────────────────────────────
        q = q.with_columns([
            pl.when(pl.col("_y_bar").abs() > 1e-12)
            .then(pl.col("_beta1") / pl.col("_y_bar") * 100.0)
            .otherwise(0.0)
            .alias("beta1_norm_pct")
        ])

        # ── FASE F: Suavizado R² ───────────────────────────────────────────────
        q = q.with_columns([
            pl.col("r2_raw")
            .rolling_mean(window_size=max(r2_smooth, 1))
            .alias("r2_suav")
        ])

        # ── FASE G: Condiciones base ───────────────────────────────────────────
        q = q.with_columns([
            (
                (pl.col("r2_suav")        >= r2_min) &
                (pl.col("beta1_norm_pct") >  slope_thresh)
            ).fill_null(False).alias("_cond_long"),

            (
                (pl.col("r2_suav")        >= r2_min) &
                (pl.col("beta1_norm_pct") < -slope_thresh)
            ).fill_null(False).alias("_cond_short"),
        ])

        # ── FASE H: Flancos ────────────────────────────────────────────────────
        q = q.with_columns([
            self._as_bool(
                pl.col("_cond_long") &
                ~pl.col("_cond_long").shift(1).fill_null(False)
            ).alias("signal_long"),

            self._as_bool(
                pl.col("_cond_short") &
                ~pl.col("_cond_short").shift(1).fill_null(False) &
                ~pl.col("_cond_long")
            ).alias("signal_short"),
        ])

        # ── FASE I: finalize_signals ───────────────────────────────────────────
        return self.finalize_signals(
            q,
            keep_cols=[
                "precio_medio",
                "ols_linea",
                "ols_upper",
                "ols_lower",
                "r2_raw",
                "r2_suav",
                "beta1_norm_pct",
            ],
        )