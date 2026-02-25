from __future__ import annotations
"""
================================================================================
MODELOX/STRATEGIES/ID20.PY
================================================================================
ID        : 20
NOMBRE    : CCIBREAK — CCI Breakout Lambert + Filtro Pendiente EMA
MERCADO   : Crypto
TIMEFRAME : El que el sistema pase (sin timeframe fijo)
--------------------------------------------------------------------------------
CONCEPTO:
  CCI (Commodity Channel Index) de Donald Lambert mide la desviación del
  precio respecto a su media estadística en unidades de desviación media.

  Fórmula:
    precio_típico = (H + L + C) / 3
    CCI = (precio_típico - SMA(PT, n)) / (0.015 * MAD(PT, n))
    MAD = mean absolute deviation (desviación media absoluta)

  LÓGICA LAMBERT CLÁSICA:
    CCI > +threshold → precio muy por encima de su media → impulso alcista → Long
    CCI < -threshold → precio muy por debajo de su media → impulso bajista → Short

  FILTRO DE CALIDAD (pendiente EMA normalizada, menos lag que R²):
    EMA(close, cci_window) — misma ventana que CCI (anti-overfitting)
    slope = (EMA_actual - EMA_anterior) / EMA_anterior * 100
    Long  válido : slope >  slope_threshold (EMA subiendo con inclinación)
    Short válido : slope < -slope_threshold (EMA bajando con inclinación)
    Elimina señales CCI donde el precio no tiene dirección clara

--------------------------------------------------------------------------------
SEÑAL LONG  : CCI > +cci_threshold AND slope_ema >  +0.02
              Solo en FLANCO. Mutuamente excluyente con SHORT.

SEÑAL SHORT : CCI < -cci_threshold AND slope_ema < -0.02
              Solo en FLANCO. Mutuamente excluyente con LONG.

SALIDAS:
  SALIDAS_PERSONALIZADAS = False

RANGOS OPTUNA (2 parámetros — anti-overfitting):
  cci_window    : int [14, 40]  step=2
  cci_threshold : int [100, 200] step=25

PARÁMETROS FIJOS:
  ema_window      : cci_window  (misma ventana — anti-overfitting)
  slope_threshold : 0.02        (mismo que ID2 — consistencia de colección)
  lambert_const   : 0.015       (constante Lambert clásica)

TIMEFRAME: Sin timeframe fijo — usa el que el sistema pase.

IMPLEMENTACIÓN:
  - CCI 100% Polars vectorial (rolling_mean + rolling_mean de |desv|)
  - EMA via ewm_mean nativo Polars
  - Pendiente normalizada via shift(1) Polars
  - 1 solo .collect() al final en finalize_signals()
================================================================================
"""

from typing import Any, Dict, List, Optional
import polars as pl
import numpy as np

from modelox.strategies.ESTRATEGIA_BASE import EstrategiaBase


class EstrategiaID20CCIBREAK(EstrategiaBase):
    """
    CCIBREAK — CCI Breakout clásico Lambert con filtro de pendiente EMA.
    Entra en impulsos estadísticamente fuertes (CCI > umbral) confirmados
    por la inclinación de la EMA (tendencia con dirección real, bajo lag).
    """

    combinacion_id: int          = 20
    name: str                    = "CCIBREAK"
    SALIDAS_PERSONALIZADAS: bool = False
    timeframe_entry: Optional[str] = None
    timeframe_exit:  Optional[str] = None

    # ==========================================================================
    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        """
        2 parámetros Optuna (anti-overfitting máximo).
        Fijos: ema_window=cci_window, slope_threshold=0.02, lambert=0.015
        Total combinaciones: 14 x 5 = 70 — espacio muy controlado.
        """
        return {
            "cci_window":    trial.suggest_int("cci_window",    14, 40,  step=2),
            "cci_threshold": trial.suggest_int("cci_threshold", 150, 200, step=10),
        }

    def get_required_timeframes(self, params: Dict[str, Any]) -> List[str]:
        return []

    # ==========================================================================
    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        """
        FASES:
          A) Precio típico = (H+L+C)/3
          B) CCI = (PT - SMA(PT,n)) / (0.015 * MAD(PT,n))
          C) EMA(close, n) + pendiente normalizada
          D) Condiciones base: CCI breakout AND pendiente EMA válida
          E) Flancos
          F) finalize_signals (1 collect)
        """

        self._init_params_metadata(params)
        self._require_columns(df, ["timestamp", "high", "low", "close"])

        cci_window    = int(params.get("cci_window",    20))
        cci_threshold = int(params.get("cci_threshold", 100))
        ema_window    = cci_window   # FIJO: misma ventana
        slope_thresh  = 0.02         # FIJO: mismo que ID2
        lambert_const = 0.015        # FIJO: Lambert clásico

        params["__warmup_bars"]     = cci_window * 2 + 2
        params["__indicators_used"] = [
            "precio_tipico", "cci", "ema_val", "slope_ema"
        ]
        params["__indicator_specs"] = {
            "precio_tipico": {"panel": "main", "color": "#AAAAAA", "tipo": "line"},
            "ema_val":       {"panel": "main", "color": "#FFD600", "tipo": "line"},
            "cci":           {"panel": "sub1", "color": "#AB47BC", "tipo": "line"},
            "slope_ema":     {"panel": "sub2", "color": "#FF9800", "tipo": "histogram"},
        }
        params["__indicator_bounds"] = {
            "cci":       {"lo": -float(cci_threshold), "hi": float(cci_threshold), "mid": 0.0},
            "slope_ema": {"lo": -slope_thresh,          "hi": slope_thresh,         "mid": 0.0},
        }

        # ── FASE A: Precio típico ──────────────────────────────────────────────
        q = df.lazy().with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0)
            .alias("precio_tipico")
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE B — CCI 100% Polars vectorial
        #
        # CCI = (PT - SMA(PT, n)) / (0.015 * MAD(PT, n))
        # MAD = mean(|PT - SMA(PT, n)|, n)  — desviación media absoluta
        #
        # Polars nativo: rolling_mean para SMA, rolling_mean de |desv| para MAD
        # Constante Lambert 0.015 hace que CCI oscile normalmente en ±100
        # cuando el precio sigue distribución normal
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            pl.col("precio_tipico")
            .rolling_mean(window_size=cci_window)
            .alias("_sma_pt")
        ])

        q = q.with_columns([
            (pl.col("precio_tipico") - pl.col("_sma_pt")).abs()
            .alias("_abs_dev")
        ])

        q = q.with_columns([
            pl.col("_abs_dev")
            .rolling_mean(window_size=cci_window)
            .alias("_mad")
        ])

        q = q.with_columns([
            pl.when(pl.col("_mad").abs() > 1e-12)
            .then(
                (pl.col("precio_tipico") - pl.col("_sma_pt")) /
                (lambert_const * pl.col("_mad"))
            )
            .otherwise(0.0)
            .alias("cci")
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE C — EMA + pendiente normalizada (filtro de calidad bajo lag)
        #
        # EMA(close, cci_window) — misma ventana que CCI
        # slope = (EMA_i - EMA_{i-1}) / EMA_{i-1} * 100  (% de cambio)
        # slope >  0.02 → EMA subiendo con inclinación suficiente → Long válido
        # slope < -0.02 → EMA bajando con inclinación suficiente → Short válido
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            pl.col("close")
            .ewm_mean(span=ema_window, adjust=False)
            .alias("ema_val")
        ])

        q = q.with_columns([
            pl.col("ema_val").shift(1).alias("_ema_prev")
        ])

        q = q.with_columns([
            pl.when(pl.col("_ema_prev").abs() > 1e-12)
            .then(
                (pl.col("ema_val") - pl.col("_ema_prev")) /
                pl.col("_ema_prev") * 100.0
            )
            .otherwise(0.0)
            .alias("slope_ema")
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE D — Condiciones base
        # Long:  CCI rompe por encima del umbral (impulso alcista estadístico)
        #        AND EMA subiendo con inclinación suficiente (tendencia real)
        # Short: CCI rompe por debajo del umbral negativo (impulso bajista)
        #        AND EMA bajando con inclinación suficiente
        # Mutuamente excluyentes: CCI no puede ser > +thr y < -thr simultáneo
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            (
                (pl.col("cci")       >  float(cci_threshold)) &
                (pl.col("slope_ema") >  slope_thresh) &
                pl.col("cci").is_not_null()
            ).fill_null(False).alias("_cond_long"),

            (
                (pl.col("cci")       < -float(cci_threshold)) &
                (pl.col("slope_ema") < -slope_thresh) &
                pl.col("cci").is_not_null()
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
                "precio_tipico",
                "cci",
                "ema_val",
                "slope_ema",
            ],
        )