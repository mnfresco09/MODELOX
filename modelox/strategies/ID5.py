from __future__ import annotations
"""
================================================================================
MODELOX/STRATEGIES/ID5.PY
================================================================================
ID        : 5
NOMBRE    : MAMA — MESA Adaptive Moving Average + Confirmación Z-score Diferencial
MERCADO   : Crypto (BTC, ETH, etc.)
TIMEFRAME : 1H
--------------------------------------------------------------------------------
LÓGICA DE ENTRADA:
  CAPA 1 — CRUCE MAMA / FAMA (Ehlers):
    - MAMA via Transformada de Hilbert discreta (Ehlers, Cybernetic Analysis 2004)
    - FAMA = Following Adaptive Moving Average (señal interna de MAMA)
    - Cruce alcista : MAMA cruza FAMA hacia arriba → candidato Long
    - Cruce bajista : MAMA cruza FAMA hacia abajo → candidato Short

  CAPA 2 — CONFIRMACIÓN Z-score del diferencial:
    - diff = MAMA - FAMA
    - z = (diff - mean(diff, z_window)) / std(diff, z_window)
    - Long confirmado  : cruce alcista AND z >= +z_threshold
    - Short confirmado : cruce bajista AND z <= -z_threshold
    - Filtra cruces débiles sin momentum estadístico real

  SEÑAL LONG  : cruce MAMA↑FAMA AND z >= +z_threshold
                Solo en FLANCO. Mutuamente excluyente con SHORT.

  SEÑAL SHORT : cruce MAMA↓FAMA AND z <= -z_threshold
                Solo en FLANCO. Mutuamente excluyente con LONG.

SALIDAS:
  SALIDAS_PERSONALIZADAS = False
  → SL / TP controlados 100% por exits.py (engine global)

RANGOS OPTUNA (3 parámetros — balance entre expresividad y anti-overfitting):
  fast_limit  : float [0.2, 0.8]  step=0.1   (7 valores — adaptabilidad MAMA)
  z_threshold : float [1.0, 2.0]  step=0.25  (5 valores — exigencia confirmación)
  z_window    : int   [20, 100]   step=10    (9 valores — ventana Z-score)

PARÁMETROS FIJOS:
  slow_limit  : 0.05  (Ehlers clásico — suelo mínimo de adaptación)

IMPLEMENTACIÓN:
  - MAMA/FAMA via map_batches + numpy (algoritmo Ehlers iterativo completo)
  - Z-score del diferencial 100% Polars vectorial (rolling_mean + rolling_std)
  - 1 solo .collect() al final en finalize_signals()
================================================================================
"""

from typing import Any, Dict, List
import polars as pl
import numpy as np

try:
    from numba import njit as _njit
except ImportError:
    def _njit(*args, **kwargs):          # type: ignore[misc]
        def _deco(fn): return fn
        return _deco if (args and callable(args[0])) else _deco

from modelox.strategies.ESTRATEGIA_BASE import EstrategiaBase

@_njit(cache=True)
def _mama_fama_numba(
    price: np.ndarray,
    fl: float,
    sl: float,
    mama_out: np.ndarray,
    fama_out: np.ndarray,
) -> None:
    n = len(price)
    smooth    = np.zeros(n)
    detrender = np.zeros(n)
    q1        = np.zeros(n)
    i1        = np.zeros(n)
    q2        = np.zeros(n)
    i2        = np.zeros(n)
    re_v      = np.zeros(n)
    im_v      = np.zeros(n)
    per       = np.zeros(n)
    sper      = np.zeros(n)
    phase     = np.zeros(n)
    mama_v    = price[0]
    fama_v    = price[0]

    for i in range(6, n):
        smooth[i] = (4*price[i] + 3*price[i-1] + 2*price[i-2] + price[i-3]) / 10.0
        c = 0.075 * per[i-1] + 0.54
        detrender[i] = (0.0962*smooth[i] + 0.5769*smooth[i-2] - 0.5769*smooth[i-4] - 0.0962*smooth[i-6]) * c
        q1[i] = (0.0962*detrender[i] + 0.5769*detrender[i-2] - 0.5769*detrender[i-4] - 0.0962*detrender[i-6]) * c
        i1[i] = detrender[i-3]
        ji = (0.0962*i1[i] + 0.5769*i1[i-2] - 0.5769*i1[i-4] - 0.0962*i1[i-6]) * c
        jq = (0.0962*q1[i] + 0.5769*q1[i-2] - 0.5769*q1[i-4] - 0.0962*q1[i-6]) * c
        i2_raw = i1[i] - jq
        q2_raw = q1[i] + ji
        i2[i] = 0.2*i2_raw + 0.8*i2[i-1]
        q2[i] = 0.2*q2_raw + 0.8*q2[i-1]
        re_v[i] = 0.2*(i2[i]*i2[i-1] + q2[i]*q2[i-1]) + 0.8*re_v[i-1]
        im_v[i] = 0.2*(i2[i]*q2[i-1] - q2[i]*i2[i-1]) + 0.8*im_v[i-1]
        if im_v[i] != 0.0 and re_v[i] != 0.0:
            per[i] = 360.0 / (57.29578 * (im_v[i] / re_v[i]))
        else:
            per[i] = per[i-1]
        if per[i] > 1.5*per[i-1]: per[i] = 1.5*per[i-1]
        if per[i] < 0.67*per[i-1]: per[i] = 0.67*per[i-1]
        if per[i] < 6.0: per[i] = 6.0
        if per[i] > 50.0: per[i] = 50.0
        sper[i] = 0.33*per[i] + 0.67*sper[i-1]
        if i1[i] != 0.0:
            phase[i] = 57.29578 * (q1[i] / i1[i])
        else:
            phase[i] = phase[i-1]
        delta_phase = phase[i-1] - phase[i]
        if delta_phase < 1.0: delta_phase = 1.0
        alpha = fl / delta_phase
        if alpha < sl: alpha = sl
        if alpha > fl: alpha = fl
        mama_v = alpha*price[i] + (1.0-alpha)*mama_v
        fama_v = 0.5*alpha*mama_v + (1.0-0.5*alpha)*fama_v
        mama_out[i] = mama_v
        fama_out[i] = fama_v



class EstrategiaID5MAMA(EstrategiaBase):
    """
    MAMA — MESA Adaptive Moving Average (Ehlers) cruzando FAMA,
    confirmado por Z-score del diferencial para filtrar cruces por ruido.
    """

    # ── Identidad ──────────────────────────────────────────────────────────────
    combinacion_id: int = 5
    name: str           = "MAMA"

    # ── Salidas: engine global vía exits.py ────────────────────────────────────
    SALIDAS_PERSONALIZADAS: bool = False

    # ── Timeframe ──────────────────────────────────────────────────────────────

    # ==========================================================================
    # ESPACIO DE BÚSQUEDA OPTUNA — 3 parámetros
    # ==========================================================================
    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        """
        3 parámetros Optuna.
        Fijo: slow_limit=0.05 (Ehlers clásico, suelo mínimo de adaptación)
        Total combinaciones aprox: 7 x 5 x 9 = 315 — manejable sin overfitting grave
        """
        return {
            "fast_limit":  trial.suggest_float("fast_limit",  0.2,  0.8, step=0.1),
            "z_threshold": trial.suggest_float("z_threshold", 1.0,  2.0, step=0.25),
            "z_window":    trial.suggest_int  ("z_window",    20,   100, step=10),
        }

    # ==========================================================================
    # TIMEFRAMES EXTRA (MTF) — no requeridos
    # ==========================================================================
    def get_required_timeframes(self, params: Dict[str, Any]) -> List[str]:
        return []

    # ==========================================================================
    # GENERATE SIGNALS — POLARS VECTORIAL, 1 COLLECT
    # ==========================================================================
    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        """
        FASES:
          A) MAMA + FAMA via map_batches + numpy (Ehlers Hilbert Transform)
          B) Diferencial MAMA - FAMA
          C) Z-score rolling del diferencial (Polars vectorial)
          D) Cruce MAMA/FAMA: cambio de signo del diferencial
          E) Confirmación: cruce AND z con signo correcto >= z_threshold
          F) Condiciones base Long/Short mutuamente excluyentes
          G) Flancos: señal solo en cambio False→True
          H) finalize_signals (1 collect)
        """

        # ── Init ───────────────────────────────────────────────────────────────
        self._init_params_metadata(params)
        self._require_columns(df, ["timestamp", "close"])

        # ── Parámetros con defaults defensivos ────────────────────────────────
        fast_limit  = float(params.get("fast_limit",  0.5))
        z_threshold = float(params.get("z_threshold", 1.5))
        z_window    = int  (params.get("z_window",    30))
        slow_limit  = 0.05  # FIJO: Ehlers clásico

        # ── Metadata para reporter / plots ────────────────────────────────────
        params["__warmup_bars"]     = z_window + 32 + 2  # 32 warmup Hilbert mínimo
        params["__indicators_used"] = ["mama", "fama", "diff_mf", "z_diff"]
        params["__indicator_specs"] = {
            "mama":    {"panel": "main", "color": "#00E676", "tipo": "line"},
            "fama":    {"panel": "main", "color": "#FF1744", "tipo": "line"},
            "diff_mf": {"panel": "sub1", "color": "#FF9800", "tipo": "histogram"},
            "z_diff":  {"panel": "sub2", "color": "#AB47BC", "tipo": "line"},
        }
        params["__indicator_bounds"] = {
            "diff_mf": {"lo": None,         "hi": None,        "mid": 0.0},
            "z_diff":  {"lo": -z_threshold,  "hi": z_threshold, "mid": 0.0},
        }

        # ══════════════════════════════════════════════════════════════════════
        # FASE A — MAMA + FAMA via map_batches + numpy
        # Algoritmo de John Ehlers (Cybernetic Analysis for Stocks and Futures, 2004)
        # Transformada de Hilbert discreta de 4 componentes para estimar el
        # periodo dominante del ciclo y adaptar alpha en cada barra.
        # ══════════════════════════════════════════════════════════════════════
        fl = fast_limit
        sl = slow_limit

        def _mama_fama_batch(s: pl.Series) -> pl.Series:
            price = s.to_numpy(allow_copy=True).astype(np.float64)
            n     = len(price)

            mama_out = np.full(n, np.nan)
            fama_out = np.full(n, np.nan)

            # Buffers de estado (Ehlers usa los últimos valores, no arrays completos)
            smooth     = np.zeros(n)
            detrender  = np.zeros(n)
            q1         = np.zeros(n)
            i1         = np.zeros(n)
            q2         = np.zeros(n)
            i2         = np.zeros(n)
            re_v       = np.zeros(n)
            im_v       = np.zeros(n)
            per        = np.zeros(n)
            sper       = np.zeros(n)
            phase      = np.zeros(n)

            mama_v = price[0]
            fama_v = price[0]

            for i in range(6, n):
                # ── Paso 1: Suavizado WMA de 4 velas ──────────────────────
                smooth[i] = (
                    4 * price[i] +
                    3 * price[i-1] +
                    2 * price[i-2] +
                        price[i-3]
                ) / 10.0

                # ── Paso 2: Detrender (Hilbert 4-tap) ─────────────────────
                detrender[i] = (
                    0.0962 * smooth[i] +
                    0.5769 * smooth[i-2] -
                    0.5769 * smooth[i-4] -
                    0.0962 * smooth[i-6] if i >= 6 else 0.0
                ) * (0.075 * per[i-1] + 0.54)

                # ── Paso 3: Componentes en cuadratura ─────────────────────
                q1[i] = (
                    0.0962 * detrender[i] +
                    0.5769 * detrender[i-2] -
                    0.5769 * detrender[i-4] -
                    0.0962 * detrender[i-6] if i >= 6 else 0.0
                ) * (0.075 * per[i-1] + 0.54)

                i1[i] = detrender[i-3] if i >= 3 else 0.0

                # ── Paso 4: Avance de fase 90° ────────────────────────────
                ji = (
                    0.0962 * i1[i] +
                    0.5769 * i1[i-2] -
                    0.5769 * i1[i-4] -
                    0.0962 * i1[i-6] if i >= 6 else 0.0
                ) * (0.075 * per[i-1] + 0.54)

                jq = (
                    0.0962 * q1[i] +
                    0.5769 * q1[i-2] -
                    0.5769 * q1[i-4] -
                    0.0962 * q1[i-6] if i >= 6 else 0.0
                ) * (0.075 * per[i-1] + 0.54)

                # ── Paso 5: Rotar 45° ─────────────────────────────────────
                i2_raw = i1[i] - jq
                q2_raw = q1[i] + ji

                # ── Paso 6: Suavizado con EMA 0.2 ────────────────────────
                i2[i] = 0.2 * i2_raw + 0.8 * i2[i-1]
                q2[i] = 0.2 * q2_raw + 0.8 * q2[i-1]

                # ── Paso 7: Discriminador de fase ────────────────────────
                re_v[i] = 0.2 * (i2[i] * i2[i-1] + q2[i] * q2[i-1]) + 0.8 * re_v[i-1]
                im_v[i] = 0.2 * (i2[i] * q2[i-1] - q2[i] * i2[i-1]) + 0.8 * im_v[i-1]

                # ── Paso 8: Periodo dominante ─────────────────────────────
                if im_v[i] != 0.0 and re_v[i] != 0.0:
                    per[i] = 360.0 / np.degrees(np.arctan(im_v[i] / re_v[i]))
                else:
                    per[i] = per[i-1]

                per[i] = np.clip(per[i], 0.67 * per[i-1], 1.5 * per[i-1])
                per[i] = np.clip(per[i], 6.0, 50.0)
                sper[i] = 0.33 * per[i] + 0.67 * sper[i-1]

                # ── Paso 9: Fase y delta de fase ─────────────────────────
                if i1[i] != 0.0:
                    phase[i] = np.degrees(np.arctan(q1[i] / i1[i]))
                else:
                    phase[i] = phase[i-1]

                delta_phase = phase[i-1] - phase[i]
                delta_phase = max(delta_phase, 1.0)

                # ── Paso 10: Alpha adaptativo ─────────────────────────────
                alpha = fl / delta_phase
                alpha = np.clip(alpha, sl, fl)

                # ── Paso 11: MAMA y FAMA ──────────────────────────────────
                mama_v = alpha * price[i] + (1.0 - alpha) * mama_v
                fama_v = 0.5 * alpha * mama_v + (1.0 - 0.5 * alpha) * fama_v

                mama_out[i] = mama_v
                fama_out[i] = fama_v

            return pl.Series(
                name   = "_mf",
                values = [
                    {"mama": float(mama_out[i]), "fama": float(fama_out[i])}
                    for i in range(n)
                ],
            )

        q = df.lazy().with_columns([
            pl.col("close")
            .map_batches(
                _mama_fama_batch,
                return_dtype=pl.Struct({
                    "mama": pl.Float64,
                    "fama": pl.Float64,
                }),
            )
            .alias("_mf")
        ])

        # Desempaquetar struct
        q = q.with_columns([
            pl.col("_mf").struct.field("mama").alias("mama"),
            pl.col("_mf").struct.field("fama").alias("fama"),
        ]).drop("_mf")

        # ══════════════════════════════════════════════════════════════════════
        # FASE B — Diferencial MAMA - FAMA
        # Positivo: MAMA encima de FAMA (tendencia alcista)
        # Negativo: MAMA debajo de FAMA (tendencia bajista)
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            (pl.col("mama") - pl.col("fama")).alias("diff_mf")
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE C — Z-score rolling del diferencial (100% Polars vectorial)
        # z = (diff - mean(diff, z_window)) / std(diff, z_window)
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            pl.col("diff_mf").rolling_mean(window_size=z_window).alias("_diff_mean"),
            pl.col("diff_mf").rolling_std (window_size=z_window).alias("_diff_std"),
        ])

        q = q.with_columns([
            pl.when(pl.col("_diff_std").abs() > 1e-12)
            .then((pl.col("diff_mf") - pl.col("_diff_mean")) / pl.col("_diff_std"))
            .otherwise(0.0)
            .alias("z_diff")
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE D — Cruce MAMA/FAMA: cambio de signo del diferencial
        # Cruce alcista: diff anterior <= 0 AND diff actual > 0
        # Cruce bajista: diff anterior >= 0 AND diff actual < 0
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            pl.col("diff_mf").shift(1).fill_null(0.0).alias("_diff_prev")
        ])

        q = q.with_columns([
            (
                (pl.col("_diff_prev") <= 0.0) &
                (pl.col("diff_mf")    >  0.0)
            ).fill_null(False).alias("_cruce_alcista"),

            (
                (pl.col("_diff_prev") >= 0.0) &
                (pl.col("diff_mf")    <  0.0)
            ).fill_null(False).alias("_cruce_bajista"),
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE E — Confirmación: cruce AND z con signo correcto >= z_threshold
        # Long:  cruce alcista AND z_diff >= +z_threshold
        # Short: cruce bajista AND z_diff <= -z_threshold
        # El signo del z_diff coincide siempre con el del diferencial
        # → exclusividad garantizada por diseño
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            (
                pl.col("_cruce_alcista") &
                (pl.col("z_diff") >= z_threshold)
            ).fill_null(False).alias("_cond_long"),

            (
                pl.col("_cruce_bajista") &
                (pl.col("z_diff") <= -z_threshold)
            ).fill_null(False).alias("_cond_short"),
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE F — Flancos: señal solo en cambio False→True
        # El cruce ya es por definición un evento puntual (1 vela),
        # pero aplicamos el patrón estándar de flanco para consistencia
        # con el resto de estrategias del sistema.
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            self._as_bool(
                pl.col("_cond_long") &
                ~pl.col("_cond_long").shift(1).fill_null(False)
            ).alias("signal_long"),

            self._as_bool(
                pl.col("_cond_short") &
                ~pl.col("_cond_short").shift(1).fill_null(False) &
                ~pl.col("_cond_long")  # guardia explícita de exclusividad
            ).alias("signal_short"),
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE G — finalize_signals (1 collect + validación contrato)
        # Columnas internas (_diff_mean, _diff_std, _diff_prev,
        # _cruce_*, _cond_*) excluidas via keep_cols explícito
        # ══════════════════════════════════════════════════════════════════════
        return self.finalize_signals(
            q,
            keep_cols=[
                "mama",
                "fama",
                "diff_mf",
                "z_diff",
            ],
        )