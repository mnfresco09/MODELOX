from __future__ import annotations
"""
================================================================================
MODELOX/STRATEGIES/ID18.PY
================================================================================
ID        : 18
NOMBRE    : OLSREV — Mean Reversion OLS + Z-score Residuos + Divergencia RSI
MERCADO   : Crypto
TIMEFRAME : El que el sistema pase (sin timeframe fijo)
--------------------------------------------------------------------------------
CONCEPTO:
  Extensión natural de ID2 (RLOG) pero en modo mean reversion.
  Misma base OLS + R², pero en lugar de seguir la tendencia,
  opera la REVERSIÓN AL CANAL cuando el precio se sobreextiende.

  EDGE — 3 condiciones simultáneas:
    1. R² alto → la tendencia OLS es coherente (el canal es válido)
    2. Z-score residuos ≥ 2.0σ → el precio está sobreextendido fuera del canal
    3. Divergencia RSI → el RSI ya está girando mientras el precio aún está lejos
       Esta es la confirmación de que la reversión YA EMPEZÓ

--------------------------------------------------------------------------------
LÓGICA DE ENTRADA:
  BASE OLS (idéntica a ID2):
    - precio_medio = (O+H+L+C) / 4
    - OLS rolling → β₁, R², σ_residuos, línea ajustada (y_hat), media (y_bar)
    - Residuo = precio_medio - y_hat  (desviación del precio respecto al canal)
    - Z_residuo = residuo / σ_residuos

  CAPA 1 — FILTRO CALIDAD (R²):
    - R² ≥ r2_min → el canal OLS es estadísticamente válido
    - Sin R² alto, la reversión no tiene referencia fiable

  CAPA 2 — SOBREEXTENSIÓN (Z-score residuos, fijo ±2σ):
    - Long  activado: Z_residuo ≤ -2.0 (precio muy por debajo del canal)
    - Short activado: Z_residuo ≥ +2.0 (precio muy por encima del canal)

  CAPA 3 — DIVERGENCIA RSI (confirmación de giro):
    - RSI 14 (Wilder, fijo)
    - Long  confirmado: precio en mínimo extremo PERO RSI > (100-rsi_threshold)
      → RSI ya subió mientras precio sigue abajo = divergencia alcista
    - Short confirmado: precio en máximo extremo PERO RSI < rsi_threshold
      → RSI ya bajó mientras precio sigue arriba = divergencia bajista
    - Umbral simétrico: 1 parámetro controla ambos lados

  SEÑAL LONG  : R² ≥ r2_min AND Z_res ≤ -2.0 AND RSI > (100-rsi_threshold)
                Solo en FLANCO. Mutuamente excluyente con SHORT.

  SEÑAL SHORT : R² ≥ r2_min AND Z_res ≥ +2.0 AND RSI < rsi_threshold
                Solo en FLANCO. Mutuamente excluyente con LONG.

SALIDAS:
  SALIDAS_PERSONALIZADAS = False

RANGOS OPTUNA (3 parámetros):
  ols_window    : int   [10, 60]    step=5
  r2_min        : float [0.60, 0.85] step=0.05
  rsi_threshold : int   [60, 75]    step=5   (simétrico)

PARÁMETROS FIJOS:
  z_score_threshold : 2.0  (±2σ clásico)
  rsi_period        : 14   (Wilder clásico)

TIMEFRAME: Sin timeframe fijo — usa el que el sistema pase.

IMPLEMENTACIÓN:
  - OLS rolling via map_batches + Numba @njit (mismo núcleo que ID2)
  - RSI via ewm_mean (Wilder-like) 100% Polars
  - Z-score residuos 100% Polars vectorial
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
# NÚCLEO OLS COMPILADO — mismo patrón ID2, adaptado para residuos
# Outputs: beta1, r2, sigma, y_hat (línea ajustada en vela actual), y_bar
# ==============================================================================
@_njit(cache=True)
def _ols_rev_numba(
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
    Loop OLS rolling compilado a C via Numba @njit(cache=True).
    Idéntico al núcleo de ID2 — misma precisión, mismos outputs.
    Modifica arrays in-place. x_mean y x_var precalculados fuera.
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
        beta1 = cov_xy / x_var
        beta0 = y_mean - beta1 * x_mean
        y_hat = beta0 + beta1 * (w - 1)

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
            if r2 < 0.0: r2 = 0.0
            if r2 > 1.0: r2 = 1.0

        denom = w - 2
        sigma = (ss_res / denom) ** 0.5 if denom > 0 else 0.0

        out_beta1[i] = beta1
        out_r2[i]    = r2
        out_sigma[i] = sigma
        out_y_hat[i] = y_hat
        out_y_bar[i] = y_mean


class EstrategiaID18OLSREV(EstrategiaBase):
    """
    OLSREV — Mean Reversion sobre canal OLS con filtro R² + divergencia RSI.
    Entra cuando el precio se sobreextiende estadísticamente fuera del canal
    OLS coherente (R² alto) y el RSI ya confirma el giro con divergencia.
    """

    combinacion_id: int          = 18
    name: str                    = "OLSREV"
    SALIDAS_PERSONALIZADAS: bool = False
    timeframe_entry: Optional[str] = None
    timeframe_exit:  Optional[str] = None

    # ==========================================================================
    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        """
        3 parámetros Optuna.
        Fijos: z_score_threshold=2.0, rsi_period=14
        Umbral RSI simétrico: Long si RSI > (100-thr), Short si RSI < thr
        Total combinaciones: 11 x 6 x 4 = 264 — controlado.
        """
        return {
            "ols_window":    trial.suggest_int  ("ols_window",    35,   35,  step=2),
            "r2_min":        trial.suggest_float("r2_min",        0.75, 0.75, step=0.01),
            "rsi_threshold": trial.suggest_int  ("rsi_threshold", 60,   60,  step=5),
        }

    def get_required_timeframes(self, params: Dict[str, Any]) -> List[str]:
        return []

    # ==========================================================================
    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        """
        FASES:
          A) Precio medio OHLC
          B) OLS rolling via map_batches + Numba JIT → β₁, R², σ, y_hat, y_bar
          C) Residuo = precio - y_hat → Z_residuo = residuo / σ
          D) RSI Wilder(14) via ewm_mean
          E) Condiciones: R² ≥ r2_min AND Z_res extremo AND divergencia RSI
          F) Flancos
          G) finalize_signals (1 collect)
        """

        self._init_params_metadata(params)
        self._require_columns(df, ["timestamp", "open", "high", "low", "close"])

        ols_window    = int  (params.get("ols_window",    30))
        r2_min        = float(params.get("r2_min",        0.70))
        rsi_threshold = int  (params.get("rsi_threshold", 70))
        z_thresh      = 2.0   # FIJO: ±2σ clásico
        rsi_period    = 14    # FIJO: Wilder clásico

        # Umbral RSI simétrico
        rsi_long_min  = float(100 - rsi_threshold)  # Long si RSI > este (divergencia alcista)
        rsi_short_max = float(rsi_threshold)         # Short si RSI < este (divergencia bajista)

        params["__warmup_bars"]     = ols_window + rsi_period + 4
        params["__indicators_used"] = [
            "precio_medio", "ols_linea", "ols_upper", "ols_lower",
            "z_residuo", "r2_raw", "rsi"
        ]
        params["__indicator_specs"] = {
            "precio_medio": {"panel": "main", "color": "#AAAAAA", "tipo": "line"},
            "ols_linea":    {"panel": "main", "color": "#FFD600", "tipo": "line"},
            "ols_upper":    {"panel": "main", "color": "#FF1744", "tipo": "line"},
            "ols_lower":    {"panel": "main", "color": "#00E676", "tipo": "line"},
            "z_residuo":    {"panel": "sub1", "color": "#AB47BC", "tipo": "line"},
            "r2_raw":       {"panel": "sub2", "color": "#00BCD4", "tipo": "line"},
            "rsi":          {"panel": "sub3", "color": "#FF9800", "tipo": "line"},
        }
        params["__indicator_bounds"] = {
            "z_residuo": {"lo": -z_thresh, "hi": z_thresh,     "mid": 0.0},
            "r2_raw":    {"lo": 0.0,       "hi": 1.0,          "mid": r2_min},
            "rsi":       {"lo": rsi_long_min, "hi": rsi_short_max, "mid": 50.0},
        }

        # ── FASE A: Precio medio OHLC ──────────────────────────────────────────
        q = df.lazy().with_columns([
            ((pl.col("open") + pl.col("high") + pl.col("low") + pl.col("close")) / 4.0)
            .alias("precio_medio")
        ])

        # ── FASE B: OLS rolling via Numba JIT ─────────────────────────────────
        w      = ols_window
        x      = np.arange(w, dtype=np.float64)
        x_mean = float(x.mean())
        x_var  = float(((x - x_mean) ** 2).sum())

        def _ols_batch(s: pl.Series) -> pl.Series:
            arr       = s.to_numpy(allow_copy=True).astype(np.float64)
            n_tot     = len(arr)
            out_beta1 = np.full(n_tot, np.nan)
            out_r2    = np.full(n_tot, np.nan)
            out_sigma = np.full(n_tot, np.nan)
            out_y_hat = np.full(n_tot, np.nan)
            out_y_bar = np.full(n_tot, np.nan)
            _ols_rev_numba(arr, w, x_mean, x_var,
                           out_beta1, out_r2, out_sigma, out_y_hat, out_y_bar)
            return pl.Series(
                name   = "_ols",
                values = [
                    {
                        "r2":    float(out_r2[i]),
                        "sigma": float(out_sigma[i]),
                        "y_hat": float(out_y_hat[i]),
                    }
                    for i in range(n_tot)
                ],
            )

        q = q.with_columns([
            pl.col("precio_medio")
            .map_batches(
                _ols_batch,
                return_dtype=pl.Struct({
                    "r2":    pl.Float64,
                    "sigma": pl.Float64,
                    "y_hat": pl.Float64,
                }),
            )
            .alias("_ols")
        ])

        q = q.with_columns([
            pl.col("_ols").struct.field("r2")   .alias("r2_raw"),
            pl.col("_ols").struct.field("sigma") .alias("_sigma"),
            pl.col("_ols").struct.field("y_hat") .alias("ols_linea"),
        ]).drop("_ols")

        # Canal visual ±2σ
        q = q.with_columns([
            (pl.col("ols_linea") + z_thresh * pl.col("_sigma")).alias("ols_upper"),
            (pl.col("ols_linea") - z_thresh * pl.col("_sigma")).alias("ols_lower"),
        ])

        # ── FASE C: Residuo normalizado (Z_residuo) ────────────────────────────
        # Residuo = precio - y_hat (desviación del precio respecto al canal OLS)
        # Z_residuo = residuo / σ  (cuántas sigmas fuera del canal)
        # Z > +2 → precio muy por encima del canal → candidato Short
        # Z < -2 → precio muy por debajo del canal → candidato Long
        q = q.with_columns([
            pl.when(pl.col("_sigma").abs() > 1e-12)
            .then((pl.col("precio_medio") - pl.col("ols_linea")) / pl.col("_sigma"))
            .otherwise(0.0)
            .alias("z_residuo")
        ])

        # ── FASE D: RSI Wilder(14) ─────────────────────────────────────────────
        q = q.with_columns([
            self.rsi_expr(close=pl.col("close"), length=rsi_period).alias("rsi")
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE E — Condiciones base con triple filtro
        #
        # LONG (precio sobreextendido abajo + RSI ya girando arriba):
        #   - R² ≥ r2_min          → canal OLS válido
        #   - Z_residuo ≤ -2.0     → precio ≥ 2σ por debajo del canal
        #   - RSI > rsi_long_min   → RSI ya subió (divergencia alcista)
        #     Ejemplo rsi_threshold=70: Long si RSI > 30
        #     El precio está abajo pero RSI ya subió → giro confirmado
        #
        # SHORT (precio sobreextendido arriba + RSI ya girando abajo):
        #   - R² ≥ r2_min          → canal OLS válido
        #   - Z_residuo ≥ +2.0     → precio ≥ 2σ por encima del canal
        #   - RSI < rsi_short_max  → RSI ya bajó (divergencia bajista)
        #     Ejemplo rsi_threshold=70: Short si RSI < 70
        #     El precio está arriba pero RSI ya bajó → giro confirmado
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            (
                (pl.col("r2_raw")    >= r2_min) &
                (pl.col("z_residuo") <= -z_thresh) &
                (pl.col("rsi")       >  rsi_long_min) &
                pl.col("r2_raw").is_not_null()
            ).fill_null(False).alias("_cond_long"),

            (
                (pl.col("r2_raw")    >= r2_min) &
                (pl.col("z_residuo") >= z_thresh) &
                (pl.col("rsi")       <  rsi_short_max) &
                pl.col("r2_raw").is_not_null()
            ).fill_null(False).alias("_cond_short"),
        ])

        # ── FASE F: Flancos ────────────────────────────────────────────────────
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

        # ── FASE G: finalize_signals ───────────────────────────────────────────
        return self.finalize_signals(
            q,
            keep_cols=[
                "precio_medio",
                "ols_linea",
                "ols_upper",
                "ols_lower",
                "z_residuo",
                "r2_raw",
                "rsi",
            ],
        )