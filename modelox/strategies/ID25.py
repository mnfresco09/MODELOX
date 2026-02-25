from __future__ import annotations
"""
================================================================================
MODELOX/STRATEGIES/ID25.PY
================================================================================
ID        : 25
NOMBRE    : RSIOLS — RSI Zona Simétrica + OLS Pendiente Dirección
MERCADO   : Crypto
TIMEFRAME : El que el sistema pase (sin timeframe fijo)
--------------------------------------------------------------------------------
CONCEPTO:
  Combinación de dos perspectivas complementarias sobre la tendencia:
  - RSI zona simétrica: momentum del precio en la dirección correcta
  - OLS pendiente normalizada: la tendencia tiene dirección lineal real

  Sin R² (menos restrictivo que ID2) — busca más señales sacrificando
  algo de calidad estadística a cambio de mayor frecuencia.

  Ventanas independientes (anti-overfitting):
  - RSI fijo en 14 (Wilder clásico, no se toca)
  - OLS Optuna [6,42] step=4 (ventana reactiva)

--------------------------------------------------------------------------------
LÓGICA DE ENTRADA:
  RSI ZONA SIMÉTRICA (Wilder, period=14, fijo):
    rsi_mid ∈ [35,50] — 1 parámetro controla ambos lados
    Long  válido: RSI > rsi_mid          (ej. rsi_mid=40 → RSI > 40)
    Short válido: RSI < (100 - rsi_mid)  (ej. rsi_mid=40 → RSI < 60)

  OLS PENDIENTE (ventana Optuna, sin R²):
    precio_medio = (O+H+L+C)/4
    β₁_norm = (β₁ / y_bar) * 100  (pendiente normalizada en %)
    Long  válido: β₁_norm >  0.02  (tendencia alcista con inclinación)
    Short válido: β₁_norm < -0.02  (tendencia bajista con inclinación)

  SEÑAL LONG  : RSI > rsi_mid AND β₁_norm > +0.02  [FLANCO]
  SEÑAL SHORT : RSI < (100-rsi_mid) AND β₁_norm < -0.02  [FLANCO]

SALIDAS:
  SALIDAS_PERSONALIZADAS = False

RANGOS OPTUNA (2 parámetros):
  ols_window : int [6,  42] step=4
  rsi_mid    : int [35, 50] step=5  (simétrico)

PARÁMETROS FIJOS:
  rsi_period      : 14    (Wilder clásico)
  slope_threshold : 0.02  (consistencia con ID2)

TIMEFRAME: Sin timeframe fijo — usa el que el sistema pase.

IMPLEMENTACIÓN:
  - OLS rolling via map_batches + Numba @njit (mismo patrón ID2)
  - RSI via ewm_mean Wilder (método base)
  - 100% Polars vectorial excepto loop OLS
  - 1 solo .collect() al final en finalize_signals()
================================================================================
"""

from typing import Any, Dict, List, Optional
import polars as pl
import numpy as np

try:
    from numba import njit as _njit
    _NUMBA_AVAILABLE = True
except ImportError:
    def _njit(*args, **kwargs):          # type: ignore[misc]
        def _deco(fn): return fn
        return _deco if (args and callable(args[0])) else _deco
    _NUMBA_AVAILABLE = False

from modelox.strategies.ESTRATEGIA_BASE import EstrategiaBase


# ==============================================================================
# NÚCLEO OLS — solo β₁ y y_bar (sin R² ni sigma — más ligero que ID2)
# ==============================================================================
@_njit(cache=True)
def _ols_beta1_25(
    arr:       np.ndarray,
    w:         int,
    x_mean:    float,
    x_var:     float,
    out_beta1: np.ndarray,
    out_y_bar: np.ndarray,
) -> None:
    """
    OLS rolling ligero — solo β₁ y y_bar, sin R² ni sigma.
    Más rápido que ID2 al calcular menos outputs.
    Modifica arrays in-place.
    """
    n_tot = len(arr)
    for i in range(w - 1, n_tot):
        y_sum = 0.0
        for k in range(w):
            y_sum += arr[i - w + 1 + k]
        y_mean = y_sum / w
        cov_xy = 0.0
        for k in range(w):
            cov_xy += (k - x_mean) * (arr[i - w + 1 + k] - y_mean)
        out_beta1[i] = cov_xy / x_var
        out_y_bar[i] = y_mean


class EstrategiaID25RSIOLS(EstrategiaBase):
    """
    RSIOLS — RSI Zona Simétrica + OLS Pendiente sin R².
    Dos perspectivas complementarias: momentum (RSI) + dirección lineal (OLS).
    Más permisivo que ID2 al no exigir R² alto — mayor frecuencia de señales.
    """

    combinacion_id: int          = 25
    name: str                    = "RSIOLS"
    SALIDAS_PERSONALIZADAS: bool = False
    timeframe_entry: Optional[str] = None
    timeframe_exit:  Optional[str] = None

    # ==========================================================================
    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        """
        2 parámetros Optuna.
        Fijos: rsi_period=14, slope_threshold=0.02
        Total combinaciones: 10 x 4 = 40 — espacio mínimo.
        """
        return {
            "ols_window": trial.suggest_int("ols_window", 22,  40, step=2),
            "rsi_mid":    trial.suggest_int("rsi_mid",   35,  45, step=5),
        }

    def get_required_timeframes(self, params: Dict[str, Any]) -> List[str]:
        return []

    # ==========================================================================
    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        """
        FASES:
          A) Precio medio OHLC
          B) OLS rolling via Numba → β₁_norm (pendiente normalizada)
          C) RSI Wilder(14) fijo
          D) Condiciones base: β₁_norm umbral AND RSI zona simétrica
          E) Flancos
          F) finalize_signals (1 collect)
        """

        self._init_params_metadata(params)
        self._require_columns(df, ["timestamp", "open", "high", "low", "close"])

        ols_window      = int(params.get("ols_window", 20))
        rsi_mid         = int(params.get("rsi_mid",    40))
        rsi_period      = 14    # FIJO: Wilder clásico
        slope_threshold = 0.02  # FIJO: consistencia con ID2

        rsi_long_min  = float(rsi_mid)
        rsi_short_max = float(100 - rsi_mid)

        params["__warmup_bars"]     = ols_window + rsi_period + 2
        params["__indicators_used"] = [
            "precio_medio", "beta1_norm_pct", "rsi"
        ]
        params["__indicator_specs"] = {
            "precio_medio":   {"panel": "main", "color": "#AAAAAA", "tipo": "line"},
            "beta1_norm_pct": {"panel": "sub1", "color": "#FF9800", "tipo": "histogram"},
            "rsi":            {"panel": "sub2", "color": "#00E676", "tipo": "line"},
        }
        params["__indicator_bounds"] = {
            "beta1_norm_pct": {"lo": None,         "hi": None,          "mid": 0.0},
            "rsi":            {"lo": rsi_long_min,  "hi": rsi_short_max, "mid": 50.0},
        }

        # ── FASE A: Precio medio OHLC ──────────────────────────────────────────
        q = df.lazy().with_columns([
            ((pl.col("open") + pl.col("high") + pl.col("low") + pl.col("close")) / 4.0)
            .alias("precio_medio")
        ])

        # ── FASE B: OLS rolling via Numba (solo β₁, sin R² ni sigma) ──────────
        w      = ols_window
        x      = np.arange(w, dtype=np.float64)
        x_mean = float(x.mean())
        x_var  = float(((x - x_mean) ** 2).sum())

        def _ols_batch(s: pl.Series) -> pl.Series:
            arr       = s.to_numpy(allow_copy=True).astype(np.float64)
            n_tot     = len(arr)
            out_beta1 = np.full(n_tot, np.nan)
            out_y_bar = np.full(n_tot, np.nan)
            _ols_beta1_25(arr, w, x_mean, x_var, out_beta1, out_y_bar)
            return pl.Series(
                name   = "_ols",
                values = [
                    {"beta1": float(out_beta1[i]), "y_bar": float(out_y_bar[i])}
                    for i in range(n_tot)
                ],
            )

        q = q.with_columns([
            pl.col("precio_medio")
            .map_batches(
                _ols_batch,
                return_dtype=pl.Struct({"beta1": pl.Float64, "y_bar": pl.Float64}),
            )
            .alias("_ols")
        ])

        q = q.with_columns([
            pl.col("_ols").struct.field("beta1").alias("_beta1"),
            pl.col("_ols").struct.field("y_bar").alias("_y_bar"),
        ]).drop("_ols")

        q = q.with_columns([
            pl.when(pl.col("_y_bar").abs() > 1e-12)
            .then(pl.col("_beta1") / pl.col("_y_bar") * 100.0)
            .otherwise(0.0)
            .alias("beta1_norm_pct")
        ])

        # ── FASE C: RSI Wilder(14) fijo ────────────────────────────────────────
        q = q.with_columns([
            self.rsi_expr(close=pl.col("close"), length=rsi_period).alias("rsi")
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE D — Condiciones base (dos capas independientes)
        #
        # Long:  β₁_norm > +0.02 (OLS alcista) AND RSI > rsi_mid (momentum ok)
        # Short: β₁_norm < -0.02 (OLS bajista) AND RSI < 100-rsi_mid
        #
        # Sin R²: más señales que ID2, pero cada señal tiene dirección
        # estadísticamente confirmada por dos indicadores distintos
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            (
                (pl.col("beta1_norm_pct") >  slope_threshold) &
                (pl.col("rsi")           >  rsi_long_min) &
                pl.col("beta1_norm_pct").is_not_null()
            ).fill_null(False).alias("_cond_long"),

            (
                (pl.col("beta1_norm_pct") < -slope_threshold) &
                (pl.col("rsi")           <  rsi_short_max) &
                pl.col("beta1_norm_pct").is_not_null()
            ).fill_null(False).alias("_cond_short"),
        ])

        # ── FASE E: Flancos ────────────────────────────────────────────────────
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

        # ── FASE F: finalize_signals ───────────────────────────────────────────
        return self.finalize_signals(
            q,
            keep_cols=[
                "precio_medio",
                "beta1_norm_pct",
                "rsi",
            ],
        )