from __future__ import annotations
"""
================================================================================
MODELOX/STRATEGIES/ID28.PY
================================================================================
ID        : 28
NOMBRE    : BBSQUEEZE — Bollinger Squeeze Breakout + Percentil Histórico
MERCADO   : Crypto
TIMEFRAME : El que el sistema pase (sin timeframe fijo)
--------------------------------------------------------------------------------
CONCEPTO:
  El squeeze de Bollinger Bands ocurre cuando la volatilidad se comprime
  por debajo de su nivel histórico bajo — el mercado acumula energía.
  Cuando las bandas se expanden de nuevo, el precio explota en una dirección
  con momentum real. Operar el breakout en la dirección de la explosión.

  DEFINICIÓN DE SQUEEZE:
    ancho_banda = BB_upper - BB_lower
    squeeze activo: ancho_banda < percentil(pct_squeeze, bb_window)
    → el ancho actual está en el X% más bajo de su historia reciente

  SEÑAL: Squeeze se DESACTIVA (bandas se expanden) Y precio rompe banda
    → La compresión terminó Y el precio ya eligió dirección

--------------------------------------------------------------------------------
LÓGICA DE ENTRADA:
  BOLLINGER BANDS (bb_mult=2.0 fijo):
    sma       = rolling_mean(close, bb_window)
    sigma     = rolling_std(close, bb_window)
    BB_upper  = sma + 2.0 * sigma
    BB_lower  = sma - 2.0 * sigma
    bb_width  = BB_upper - BB_lower  (ancho absoluto de banda)
    bb_width_pct = bb_width / sma * 100  (ancho normalizado %)

  SQUEEZE (percentil histórico, misma ventana):
    pct_threshold = percentil(bb_width_pct, pct_squeeze, bb_window)
    squeeze_on  = bb_width_pct ≤ pct_threshold  (comprimido)
    squeeze_off = squeeze_on anterior AND NOT squeeze_on actual
                = primer vela de expansión tras compresión

  DIRECCIÓN DEL BREAKOUT:
    Long:  precio rompe BB_upper en la misma vela que sale del squeeze
    Short: precio rompe BB_lower en la misma vela que sale del squeeze

  SEÑAL LONG  : squeeze_off AND close > BB_upper  [FLANCO]
  SEÑAL SHORT : squeeze_off AND close < BB_lower  [FLANCO]

SALIDAS:
  SALIDAS_PERSONALIZADAS = False

RANGOS OPTUNA (2 parámetros):
  bb_window   : int [10, 50] step=5
  pct_squeeze : int [10, 30] step=5  (percentil máximo para squeeze)

PARÁMETROS FIJOS:
  bb_mult : 2.0  (clásico ±2σ)
  ventana percentil = bb_window (anti-overfitting)

TIMEFRAME: Sin timeframe fijo — usa el que el sistema pase.

IMPLEMENTACIÓN:
  - BB 100% Polars vectorial (rolling_mean + rolling_std)
  - Percentil rolling via map_batches + numpy (numpy.percentile sobre ventana)
  - Squeeze via comparación vectorial + shift(1) para flanco de salida
  - 1 solo .collect() al final en finalize_signals()
================================================================================
"""

from typing import Any, Dict, List, Optional
import polars as pl
import numpy as np

from modelox.strategies.ESTRATEGIA_BASE import EstrategiaBase


class EstrategiaID28BBSQUEEZE(EstrategiaBase):
    """
    BBSQUEEZE — Bollinger Bands Squeeze Breakout.
    Opera el momento exacto en que el mercado sale de compresión de volatilidad
    y el precio elige dirección rompiendo la banda correspondiente.
    """

    combinacion_id: int          = 28
    name: str                    = "BBSQUEEZE"
    SALIDAS_PERSONALIZADAS: bool = False
    timeframe_entry: Optional[str] = None
    timeframe_exit:  Optional[str] = None

    # ==========================================================================
    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        """
        2 parámetros Optuna (máximo anti-overfitting).
        Fijos: bb_mult=2.0, ventana_pct=bb_window
        Total combinaciones: 9 x 5 = 45 — espacio mínimo.
        """
        return {
            "bb_window":   trial.suggest_int("bb_window",   10, 50, step=5),
            "pct_squeeze": trial.suggest_int("pct_squeeze", 10, 30, step=5),
        }

    def get_required_timeframes(self, params: Dict[str, Any]) -> List[str]:
        return []

    # ==========================================================================
    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        """
        FASES:
          A) BB: sma, sigma, upper, lower, width normalizado
          B) Percentil histórico del ancho de banda (rolling numpy.percentile)
          C) Squeeze on/off + flanco de salida (squeeze_off)
          D) Condiciones base: squeeze_off AND precio rompe banda
          E) Flancos
          F) finalize_signals (1 collect)
        """

        self._init_params_metadata(params)
        self._require_columns(df, ["timestamp", "close"])

        bb_window   = int(params.get("bb_window",   20))
        pct_squeeze = int(params.get("pct_squeeze", 20))
        bb_mult     = 2.0   # FIJO

        params["__warmup_bars"]     = bb_window * 3 + 2
        params["__indicators_used"] = [
            "bb_upper", "bb_lower", "bb_mid", "bb_width_pct", "pct_threshold"
        ]
        params["__indicator_specs"] = {
            "bb_upper":     {"panel": "main", "color": "#FF1744", "tipo": "line"},
            "bb_lower":     {"panel": "main", "color": "#00E676", "tipo": "line"},
            "bb_mid":       {"panel": "main", "color": "#FFD600", "tipo": "line"},
            "bb_width_pct": {"panel": "sub1", "color": "#AB47BC", "tipo": "line"},
            "pct_threshold":{"panel": "sub1", "color": "#FF9800", "tipo": "line"},
        }
        params["__indicator_bounds"] = {
            "bb_width_pct": {"lo": 0.0, "hi": None, "mid": None},
        }

        q = df.lazy()

        # ── FASE A: Bollinger Bands ────────────────────────────────────────────
        q = q.with_columns([
            pl.col("close").rolling_mean(window_size=bb_window).alias("bb_mid"),
            pl.col("close").rolling_std (window_size=bb_window).alias("_sigma"),
        ])

        q = q.with_columns([
            (pl.col("bb_mid") + bb_mult * pl.col("_sigma")).alias("bb_upper"),
            (pl.col("bb_mid") - bb_mult * pl.col("_sigma")).alias("bb_lower"),
        ])

        # Ancho normalizado % del precio medio — robusto entre activos
        q = q.with_columns([
            pl.when(pl.col("bb_mid").abs() > 1e-12)
            .then(
                (pl.col("bb_upper") - pl.col("bb_lower")) /
                pl.col("bb_mid") * 100.0
            )
            .otherwise(0.0)
            .alias("bb_width_pct")
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE B — Percentil histórico del ancho de banda (rolling)
        # numpy.percentile sobre ventana deslizante → umbral de squeeze
        # pct_threshold = percentil(pct_squeeze, bb_window) del ancho
        # Si bb_width_pct ≤ pct_threshold → banda en zona históricamente baja
        # ══════════════════════════════════════════════════════════════════════
        window  = bb_window
        pct_val = pct_squeeze

        def _pct_threshold_batch(s: pl.Series) -> pl.Series:
            arr    = s.to_numpy(allow_copy=True).astype(np.float64)
            n_tot  = len(arr)
            result = np.full(n_tot, np.nan)
            for i in range(window - 1, n_tot):
                sub = arr[i - window + 1 : i + 1]
                # Ignorar NaN en la ventana
                valid = sub[~np.isnan(sub)]
                if len(valid) > 0:
                    result[i] = np.percentile(valid, pct_val)
            return pl.Series(name="pct_threshold", values=result.tolist())

        q = q.with_columns([
            pl.col("bb_width_pct")
            .map_batches(_pct_threshold_batch, return_dtype=pl.Float64)
            .alias("pct_threshold")
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE C — Squeeze on/off + flanco de salida
        #
        # squeeze_on:  ancho actual ≤ umbral percentil → comprimido
        # squeeze_off: squeeze_on anterior AND NOT squeeze_on actual
        #              = exactamente la primera vela de expansión
        #              Es el momento de máxima probabilidad de breakout
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            (
                pl.col("bb_width_pct") <= pl.col("pct_threshold")
            ).fill_null(False).alias("_squeeze_on")
        ])

        q = q.with_columns([
            pl.col("_squeeze_on").shift(1).fill_null(False).alias("_squeeze_on_prev")
        ])

        # Flanco de salida del squeeze: estaba comprimido Y ahora no
        q = q.with_columns([
            (
                pl.col("_squeeze_on_prev") &
                ~pl.col("_squeeze_on")
            ).fill_null(False).alias("_squeeze_off")
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE D — Condiciones base
        #
        # Long:  squeeze_off AND close > BB_upper
        #   → bandas se expanden Y precio ya rompió hacia arriba
        #   → breakout alcista confirmado en el mismo momento
        #
        # Short: squeeze_off AND close < BB_lower
        #   → bandas se expanden Y precio ya rompió hacia abajo
        #   → breakout bajista confirmado en el mismo momento
        #
        # Mutuamente excluyentes: close no puede estar > upper y < lower
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            (
                pl.col("_squeeze_off") &
                (pl.col("close") > pl.col("bb_upper")) &
                pl.col("pct_threshold").is_not_null()
            ).fill_null(False).alias("_cond_long"),

            (
                pl.col("_squeeze_off") &
                (pl.col("close") < pl.col("bb_lower")) &
                pl.col("pct_threshold").is_not_null()
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
                "bb_upper",
                "bb_lower",
                "bb_mid",
                "bb_width_pct",
                "pct_threshold",
            ],
        )