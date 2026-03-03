from __future__ import annotations
"""
================================================================================
MODELOX/STRATEGIES/ID23.PY
================================================================================
ID        : 23
NOMBRE    : ZTREND — Z-score Retornos + ADX Confirmación Tendencia
MERCADO   : Crypto
TIMEFRAME : El que el sistema pase (sin timeframe fijo)
--------------------------------------------------------------------------------
CONCEPTO:
  Z-score de retornos logarítmicos mide si el retorno actual es
  estadísticamente extremo respecto a su distribución reciente.
  Cuando cruza ±umbral en la dirección de la tendencia (ADX confirma),
  es una señal de inicio de movimiento con fuerza real.

  DIFERENCIA CON ID22 (ZREV):
    ID22: Z cruza de VUELTA el umbral → reversión
    ID23: Z cruza el umbral de IDA    → seguimiento de tendencia

  ANTI-WHIPSAW:
    Umbral elevado (≥1.0, Optuna hasta 2.0) + ADX > threshold eliminan
    señales falsas sin necesidad de doble umbral ni filtros complejos.

--------------------------------------------------------------------------------
LÓGICA DE ENTRADA:
  RETORNO LOGARÍTMICO:
    r_i = ln(close_i / close_{i-1})

  Z-SCORE ROLLING:
    z = (r - mean(r, z_window)) / std(r, z_window)

  CRUCE DE UMBRAL (señal de tendencia):
    z_prev < +threshold AND z_actual ≥ +threshold → impulso alcista → Long
    z_prev > -threshold AND z_actual ≤ -threshold → impulso bajista → Short

  ADX CONFIRMACIÓN (Wilder, period=14, fijo):
    ADX > adx_threshold → tendencia activa → señal válida
    ADX ≤ adx_threshold → mercado lateral → ignorar señal Z

  SEÑAL LONG  : z cruza ≥ +z_threshold AND ADX > adx_threshold  [FLANCO]
  SEÑAL SHORT : z cruza ≤ -z_threshold AND ADX > adx_threshold  [FLANCO]

SALIDAS:
  SALIDAS_PERSONALIZADAS = False

RANGOS OPTUNA (3 parámetros):
  z_window      : int   [20, 100] step=10
  z_threshold   : float [1.0, 2.0] step=0.25
  adx_threshold : int   [20, 35]  step=5

PARÁMETROS FIJOS:
  adx_period : 14  (Wilder clásico)

TIMEFRAME: Sin timeframe fijo — usa el que el sistema pase.

IMPLEMENTACIÓN:
  - Z-score 100% Polars vectorial (rolling_mean + rolling_std)
  - ADX via Numba @njit (mismo patrón ID14)
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
# NÚCLEO ADX COMPILADO — Wilder, period=14
# Mismo algoritmo que ID14, renombrado para evitar colisión de cache
# ==============================================================================
@_njit(cache=True)
def _adx_numba_23(
    high:     np.ndarray,
    low:      np.ndarray,
    close:    np.ndarray,
    period:   int,
    out_adx:  np.ndarray,
) -> None:
    """
    ADX Wilder compilado a C via Numba @njit(cache=True).
    Modifica out_adx in-place.
    """
    n   = len(close)
    com = float(period - 1)

    tr_arr  = np.zeros(n)
    dm_p    = np.zeros(n)
    dm_m    = np.zeros(n)

    for i in range(1, n):
        hl   = high[i]  - low[i]
        hpc  = abs(high[i]  - close[i-1])
        lpc  = abs(low[i]   - close[i-1])
        tr_arr[i] = max(hl, hpc, lpc)

        h_diff = high[i]  - high[i-1]
        l_diff = low[i-1] - low[i]
        dm_p[i] = h_diff if (h_diff > l_diff and h_diff > 0.0) else 0.0
        dm_m[i] = l_diff if (l_diff > h_diff and l_diff > 0.0) else 0.0

    # EWM Wilder (com = period-1)
    atr  = np.zeros(n)
    sdp  = np.zeros(n)
    sdm  = np.zeros(n)
    alpha = 1.0 / (com + 1.0)

    atr[0] = tr_arr[0]
    sdp[0] = dm_p[0]
    sdm[0] = dm_m[0]

    for i in range(1, n):
        atr[i] = alpha * tr_arr[i] + (1.0 - alpha) * atr[i-1]
        sdp[i] = alpha * dm_p[i]   + (1.0 - alpha) * sdp[i-1]
        sdm[i] = alpha * dm_m[i]   + (1.0 - alpha) * sdm[i-1]

    di_p = np.zeros(n)
    di_m = np.zeros(n)
    dx   = np.zeros(n)

    for i in range(n):
        if atr[i] > 1e-12:
            di_p[i] = sdp[i] / atr[i] * 100.0
            di_m[i] = sdm[i] / atr[i] * 100.0
        dsum = di_p[i] + di_m[i]
        if dsum > 1e-12:
            dx[i] = abs(di_p[i] - di_m[i]) / dsum * 100.0

    # ADX = EWM(DX)
    adx_v = dx[0]
    out_adx[0] = adx_v
    for i in range(1, n):
        adx_v = alpha * dx[i] + (1.0 - alpha) * adx_v
        out_adx[i] = adx_v


class EstrategiaID23ZTREND(EstrategiaBase):
    """
    ZTREND — Seguimiento de tendencia via cruce Z-score retornos ±umbral
    confirmado por ADX > threshold (tendencia activa).
    Distinto de ID22 (reversión): aquí el cruce es de IDA, no de vuelta.
    """

    combinacion_id: int          = 23
    name: str                    = "ZTREND"
    SALIDAS_PERSONALIZADAS: bool = False
    timeframe_entry: Optional[str] = None
    timeframe_exit:  Optional[str] = None

    # ==========================================================================
    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        """
        3 parámetros Optuna.
        Fijo: adx_period=14 (Wilder clásico)
        Total combinaciones: 9 x 5 x 4 = 180 — controlado.
        """
        return {
            "z_window":      trial.suggest_int  ("z_window",      30,  30, step=10),
            "z_threshold":   trial.suggest_float("z_threshold",   1.75, 1.75, step=0.25),
            "adx_threshold": trial.suggest_int  ("adx_threshold", 25,  25,  step=5),
        }

    def get_required_timeframes(self, params: Dict[str, Any]) -> List[str]:
        return []

    # ==========================================================================
    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        """
        FASES:
          A) Retorno logarítmico
          B) Z-score rolling (Polars vectorial)
          C) ADX Wilder(14) via map_batches + Numba
          D) Cruce Z de IDA: z_prev < threshold AND z_actual ≥ threshold
          E) Condiciones base: cruce Z AND ADX > threshold
          F) Flancos
          G) finalize_signals (1 collect)
        """

        self._init_params_metadata(params)
        self._require_columns(df, ["timestamp", "high", "low", "close"])

        z_window      = int  (params.get("z_window",      50))
        z_threshold   = float(params.get("z_threshold",   1.5))
        adx_threshold = int  (params.get("adx_threshold", 25))
        adx_period    = 14    # FIJO: Wilder clásico

        params["__warmup_bars"]     = z_window + adx_period * 3 + 2
        params["__indicators_used"] = ["log_return", "z_score", "adx"]
        params["__indicator_specs"] = {
            "log_return": {"panel": "sub1", "color": "#78909C", "tipo": "line"},
            "z_score":    {"panel": "sub2", "color": "#AB47BC", "tipo": "line"},
            "adx":        {"panel": "sub3", "color": "#FFD600", "tipo": "line"},
        }
        params["__indicator_bounds"] = {
            "z_score": {"lo": -z_threshold, "hi": z_threshold,      "mid": 0.0},
            "adx":     {"lo": 0.0,          "hi": 100.0,            "mid": float(adx_threshold)},
        }

        # ── FASE A: Retorno logarítmico ────────────────────────────────────────
        q = df.lazy().with_columns([
            (pl.col("close") / pl.col("close").shift(1))
            .log(base=2.718281828)
            .alias("log_return")
        ])

        # ── FASE B: Z-score rolling (100% Polars vectorial) ───────────────────
        q = q.with_columns([
            pl.col("log_return").rolling_mean(window_size=z_window).alias("_r_mean"),
            pl.col("log_return").rolling_std (window_size=z_window).alias("_r_std"),
        ])

        q = q.with_columns([
            pl.when(pl.col("_r_std").abs() > 1e-12)
            .then((pl.col("log_return") - pl.col("_r_mean")) / pl.col("_r_std"))
            .otherwise(0.0)
            .alias("z_score")
        ])

        # ── FASE C: ADX Wilder(14) via map_batches + Numba ────────────────────
        period = adx_period

        def _adx_batch(s: pl.Series) -> pl.Series:
            # s es struct con high, low, close
            arr   = s.to_numpy(allow_copy=True)
            n_tot = len(arr)

            # Compatibilidad robusta entre versiones de Polars:
            # cada fila de struct puede venir como dict/Row/tuple.
            def _get3(row):
                try:
                    # caso dict-like: {'high':..., 'low':..., 'close':...}
                    return row["high"], row["low"], row["close"]
                except Exception:
                    # fallback posicional
                    return row[0], row[1], row[2]

            vals = [_get3(r) for r in arr]
            high_arr  = np.array([v[0] for v in vals], dtype=np.float64)
            low_arr   = np.array([v[1] for v in vals], dtype=np.float64)
            close_arr = np.array([v[2] for v in vals], dtype=np.float64)
            out_adx   = np.zeros(n_tot)
            _adx_numba_23(high_arr, low_arr, close_arr, period, out_adx)
            return pl.Series(name="adx", values=out_adx.tolist())

        q = q.with_columns([
            pl.struct(["high", "low", "close"])
            .map_batches(_adx_batch, return_dtype=pl.Float64)
            .alias("adx")
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE D — Cruce Z de IDA (seguimiento tendencia)
        #
        # Long:  z_prev < +threshold AND z_actual ≥ +threshold
        #   → retorno acaba de entrar en zona alcista extrema
        #   → inicio de impulso estadísticamente significativo hacia arriba
        #
        # Short: z_prev > -threshold AND z_actual ≤ -threshold
        #   → retorno acaba de entrar en zona bajista extrema
        #   → inicio de impulso estadísticamente significativo hacia abajo
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            pl.col("z_score").shift(1).fill_null(0.0).alias("_z_prev")
        ])

        q = q.with_columns([
            (
                (pl.col("_z_prev") <  z_threshold) &
                (pl.col("z_score") >= z_threshold)
            ).fill_null(False).alias("_cruce_long"),

            (
                (pl.col("_z_prev") > -z_threshold) &
                (pl.col("z_score") <= -z_threshold)
            ).fill_null(False).alias("_cruce_short"),
        ])

        # ── FASE E: Condiciones base: cruce Z AND ADX confirma tendencia ───────
        q = q.with_columns([
            (
                pl.col("_cruce_long") &
                (pl.col("adx") > float(adx_threshold))
            ).fill_null(False).alias("_cond_long"),

            (
                pl.col("_cruce_short") &
                (pl.col("adx") > float(adx_threshold))
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
        # Validación defensiva del esquema antes de collect para evitar KeyError opaco
        try:
            # Polars LazyFrame: schema es un dict nombre->dtype
            schema_cols = list(q.schema.keys())
            required = {"timestamp", "signal_long", "signal_short", "log_return", "z_score", "adx"}
            missing = [c for c in required if c not in schema_cols]
            if missing:
                raise ValueError(
                    f"ID23 finalize_signals: faltan columnas antes de collect: {missing}. "
                    f"Schema actual: {schema_cols}"
                )
        except Exception:
            # Re-lanzar con contexto específico de la estrategia
            raise

        return self.finalize_signals(
            q,
            keep_cols=[
                "log_return",
                "z_score",
                "adx",
            ],
        )