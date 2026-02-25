from __future__ import annotations
"""
================================================================================
MODELOX/STRATEGIES/ID21.PY
================================================================================
ID        : 21
NOMBRE    : BBEMA — Bollinger Bands Breakout + EMA Dirección
MERCADO   : Crypto
TIMEFRAME : El que el sistema pase (sin timeframe fijo)
--------------------------------------------------------------------------------
CONCEPTO:
  Bollinger Bands miden la volatilidad relativa del precio respecto a su
  media. Cuando el precio rompe la banda superior/inferior, está en territorio
  estadísticamente extremo — potencial inicio de tendencia fuerte.

  La EMA sobre la misma ventana confirma la dirección:
  precio sobre EMA = tendencia alcista, precio bajo EMA = tendencia bajista.

  EDGE: Rotura de banda + EMA alineada = breakout con dirección confirmada.

--------------------------------------------------------------------------------
LÓGICA DE ENTRADA:
  BOLLINGER BANDS:
    - SMA  = rolling_mean(close, bb_window)
    - σ    = rolling_std(close, bb_window)
    - BB_upper = SMA + 2.0 * σ
    - BB_lower = SMA - 2.0 * σ
    - bb_mult = 2.0 (fijo, clásico)

  EMA DIRECCIÓN (misma ventana, anti-overfitting):
    - EMA = ewm_mean(close, span=bb_window)
    - close > EMA → tendencia alcista
    - close < EMA → tendencia bajista

  SEÑAL LONG  : close > BB_upper AND close > EMA   [FLANCO]
  SEÑAL SHORT : close < BB_lower AND close < EMA   [FLANCO]

SALIDAS:
  SALIDAS_PERSONALIZADAS = False

RANGOS OPTUNA (1 parámetro — máximo anti-overfitting):
  bb_window : int [10, 50] step=5

PARÁMETROS FIJOS:
  bb_mult    : 2.0         (clásico ±2σ)
  ema_window : bb_window   (misma ventana)

TIMEFRAME: Sin timeframe fijo — usa el que el sistema pase.

IMPLEMENTACIÓN:
  - 100% Polars vectorial (rolling_mean, rolling_std, ewm_mean nativos)
  - Cero loops, cero Numba necesario
  - 1 solo .collect() al final en finalize_signals()
================================================================================
"""

from typing import Any, Dict, List, Optional
import polars as pl

from modelox.strategies.ESTRATEGIA_BASE import EstrategiaBase


class EstrategiaID21BBEMA(EstrategiaBase):
    """
    BBEMA — Bollinger Bands Breakout + EMA Dirección.
    Entra cuando el precio rompe la banda estadística (±2σ) con la EMA
    confirmando la dirección. Simple, clásico y 100% vectorial.
    """

    combinacion_id: int          = 21
    name: str                    = "BBEMA"
    SALIDAS_PERSONALIZADAS: bool = False
    timeframe_entry: Optional[str] = None
    timeframe_exit:  Optional[str] = None

    # ==========================================================================
    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        """
        1 parámetro Optuna (máximo anti-overfitting).
        Fijos: bb_mult=2.0, ema_window=bb_window
        Total combinaciones: 9 — espacio mínimo absoluto.
        """
        return {
            "bb_window": trial.suggest_int("bb_window", 10, 50, step=5),
        }

    def get_required_timeframes(self, params: Dict[str, Any]) -> List[str]:
        return []

    # ==========================================================================
    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        """
        FASES:
          A) SMA + σ rolling → BB_upper / BB_lower
          B) EMA dirección (misma ventana)
          C) Condiciones base: close rompe banda AND EMA confirma
          D) Flancos
          E) finalize_signals (1 collect)
        """

        self._init_params_metadata(params)
        self._require_columns(df, ["timestamp", "close"])

        bb_window  = int(params.get("bb_window", 20))
        bb_mult    = 2.0          # FIJO: clásico ±2σ
        ema_window = bb_window    # FIJO: misma ventana

        params["__warmup_bars"]     = bb_window * 2 + 2
        params["__indicators_used"] = [
            "bb_upper", "bb_lower", "bb_mid", "ema_val"
        ]
        params["__indicator_specs"] = {
            "bb_upper": {"panel": "main", "color": "#FF1744", "tipo": "line"},
            "bb_lower": {"panel": "main", "color": "#00E676", "tipo": "line"},
            "bb_mid":   {"panel": "main", "color": "#FFD600", "tipo": "line"},
            "ema_val":  {"panel": "main", "color": "#42A5F5", "tipo": "line"},
        }
        params["__indicator_bounds"] = {}

        q = df.lazy()

        # ── FASE A: Bollinger Bands ────────────────────────────────────────────
        q = q.with_columns([
            pl.col("close").rolling_mean(window_size=bb_window).alias("bb_mid"),
            pl.col("close").rolling_std (window_size=bb_window).alias("_bb_std"),
        ])

        q = q.with_columns([
            (pl.col("bb_mid") + bb_mult * pl.col("_bb_std")).alias("bb_upper"),
            (pl.col("bb_mid") - bb_mult * pl.col("_bb_std")).alias("bb_lower"),
        ])

        # ── FASE B: EMA dirección ──────────────────────────────────────────────
        q = q.with_columns([
            pl.col("close")
            .ewm_mean(span=ema_window, adjust=False)
            .alias("ema_val")
        ])

        # ── FASE C: Condiciones base ───────────────────────────────────────────
        # Long:  close rompe BB superior AND close sobre EMA (alcista)
        # Short: close rompe BB inferior AND close bajo EMA (bajista)
        # Mutuamente excluyentes: close no puede estar > upper y < lower a la vez
        q = q.with_columns([
            (
                (pl.col("close") > pl.col("bb_upper")) &
                (pl.col("close") > pl.col("ema_val")) &
                pl.col("bb_upper").is_not_null()
            ).fill_null(False).alias("_cond_long"),

            (
                (pl.col("close") < pl.col("bb_lower")) &
                (pl.col("close") < pl.col("ema_val")) &
                pl.col("bb_lower").is_not_null()
            ).fill_null(False).alias("_cond_short"),
        ])

        # ── FASE D: Flancos ────────────────────────────────────────────────────
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

        # ── FASE E: finalize_signals ───────────────────────────────────────────
        return self.finalize_signals(
            q,
            keep_cols=[
                "bb_upper",
                "bb_lower",
                "bb_mid",
                "ema_val",
            ],
        )