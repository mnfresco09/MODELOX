from __future__ import annotations
"""
================================================================================
MODELOX/STRATEGIES/ID27.PY
================================================================================
ID        : 27
NOMBRE    : MACDBAND — MACD Cruce + RSI Zona Banda + Rotura EMA Reciente
MERCADO   : Crypto
TIMEFRAME : El que el sistema pase (sin timeframe fijo)
--------------------------------------------------------------------------------
CONCEPTO:
  Triple confirmación de tendencia con momentum moderado:

  1. MACD cruce (trigger): cambio de momentum confirmado
  2. RSI en zona banda (filtro calidad):
     - Long:  RSI entre 50 y 70 → momentum alcista sin sobrecompra
     - Short: RSI entre 30 y 50 → momentum bajista sin sobreventa
     Evita entrar en extremos donde el movimiento puede agotarse
  3. Rotura EMA reciente (contexto): el precio cruzó la EMA
     en las últimas 1-4 velas → tendencia con participación fresca

  EDGE: El RSI en banda media es más selectivo que RSI > 50 solo.
  Filtra entradas en mercados sobreextendidos y en mercados sin momentum.
  La rotura EMA reciente garantiza que el contexto de tendencia es nuevo,
  no una tendencia vieja que puede estar agotándose.

--------------------------------------------------------------------------------
LÓGICA DE ENTRADA:
  MACD (mismo esquema anti-colisión ID24):
    fast ∈ [8,16], slow ∈ [18,30] → fast < slow garantizado
    signal = 9 (fijo, Gerald Appel clásico)
    Cruce alcista: histograma pasa de ≤0 a >0
    Cruce bajista: histograma pasa de ≥0 a <0

  RSI ZONA BANDA (Wilder, period=14, fijo):
    Long  válido: 50 < RSI ≤ 70  (momentum alcista moderado)
    Short válido: 30 ≤ RSI < 50  (momentum bajista moderado)
    Fijos: sin parámetros Optuna extra

  ROTURA EMA RECIENTE (ventana Optuna):
    EMA = ewm_mean(close, span=ma_window)
    Rotura alcista reciente: close cruzó EMA hacia arriba en últimas N=4 velas
    Rotura bajista reciente: close cruzó EMA hacia abajo en últimas N=4 velas
    Fijo N=4: ocurre entre vela actual y las 3 anteriores

  SEÑAL LONG  : cruce MACD alcista AND 50 < RSI ≤ 70 AND rotura EMA alcista reciente
  SEÑAL SHORT : cruce MACD bajista AND 30 ≤ RSI < 50 AND rotura EMA bajista reciente

SALIDAS:
  SALIDAS_PERSONALIZADAS = False

RANGOS OPTUNA (3 parámetros):
  fast      : int [8,  16] step=2
  slow      : int [18, 30] step=2
  ma_window : int [20,100] step=10

PARÁMETROS FIJOS:
  signal     : 9   (Gerald Appel clásico)
  rsi_period : 14  (Wilder clásico)
  rsi_lo_long  : 50, rsi_hi_long  : 70  (banda Long)
  rsi_lo_short : 30, rsi_hi_short : 50  (banda Short)
  rotura_velas : 4  (rotura en últimas 4 velas)

TIMEFRAME: Sin timeframe fijo — usa el que el sistema pase.

IMPLEMENTACIÓN:
  - MACD 100% Polars vectorial (ewm_mean encadenado)
  - RSI via método base (ewm_mean Wilder)
  - Rotura EMA: rolling_max/min de señal de cruce en ventana N=4
  - 1 solo .collect() al final en finalize_signals()
================================================================================
"""

from typing import Any, Dict, List, Optional
import polars as pl

from modelox.strategies.ESTRATEGIA_BASE import EstrategiaBase


class EstrategiaID27MACDBAND(EstrategiaBase):
    """
    MACDBAND — MACD Cruce + RSI Zona Banda + Rotura EMA Reciente.
    Triple confirmación: trigger (MACD) + calidad momentum (RSI banda)
    + contexto tendencia fresco (rotura EMA en últimas 4 velas).
    """

    combinacion_id: int          = 27
    name: str                    = "MACDBAND"
    SALIDAS_PERSONALIZADAS: bool = False
    timeframe_entry: Optional[str] = None
    timeframe_exit:  Optional[str] = None

    # ==========================================================================
    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        """
        3 parámetros Optuna.
        fast máx=16, slow mín=18 → fast < slow garantizado por diseño.
        Fijos: signal=9, rsi_period=14, bandas RSI, N=4 velas rotura
        Total combinaciones: 5 x 7 x 9 = 315 — controlado.
        """
        return {
            "fast":      trial.suggest_int("fast",      14,  14, step=1),
            "slow":      trial.suggest_int("slow",     35,  35, step=1),
            "ma_window": trial.suggest_int("ma_window", 50, 50, step=5),
        }

    def get_required_timeframes(self, params: Dict[str, Any]) -> List[str]:
        return []

    # ==========================================================================
    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        """
        FASES:
          A) MACD line + signal line + histograma
          B) RSI Wilder(14)
          C) EMA contexto (ma_window)
          D) Rotura EMA reciente: cruce en últimas 4 velas
          E) Cruce MACD (cambio signo histograma)
          F) Condiciones base: cruce AND RSI banda AND rotura reciente
          G) Flancos
          H) finalize_signals (1 collect)
        """

        self._init_params_metadata(params)
        self._require_columns(df, ["timestamp", "close"])

        fast       = int(params.get("fast",       12))
        slow       = int(params.get("slow",       26))
        ma_window  = int(params.get("ma_window",  50))
        signal     = 9    # FIJO
        rsi_period = 14   # FIJO
        n_rotura   = 4    # FIJO: rotura en últimas 4 velas

        # Bandas RSI fijas
        rsi_lo_long  = 50.0
        rsi_hi_long  = 70.0
        rsi_lo_short = 30.0
        rsi_hi_short = 50.0

        params["__warmup_bars"]     = slow + signal + ma_window + rsi_period + 4
        params["__indicators_used"] = ["macd_line", "signal_line", "histogram", "rsi", "ema_ctx"]
        params["__indicator_specs"] = {
            "ema_ctx":     {"panel": "main", "color": "#FFD600", "tipo": "line"},
            "macd_line":   {"panel": "sub1", "color": "#00E676", "tipo": "line"},
            "signal_line": {"panel": "sub1", "color": "#FF1744", "tipo": "line"},
            "histogram":   {"panel": "sub1", "color": "#78909C", "tipo": "histogram"},
            "rsi":         {"panel": "sub2", "color": "#FF9800", "tipo": "line"},
        }
        params["__indicator_bounds"] = {
            "histogram": {"lo": None,        "hi": None,        "mid": 0.0},
            "rsi":       {"lo": rsi_lo_short, "hi": rsi_hi_long, "mid": 50.0},
        }

        q = df.lazy()

        # ── FASE A: MACD ───────────────────────────────────────────────────────
        q = q.with_columns([
            pl.col("close").ewm_mean(span=fast,   adjust=False).alias("_ema_fast"),
            pl.col("close").ewm_mean(span=slow,   adjust=False).alias("_ema_slow"),
        ])

        q = q.with_columns([
            (pl.col("_ema_fast") - pl.col("_ema_slow")).alias("macd_line")
        ])

        q = q.with_columns([
            pl.col("macd_line").ewm_mean(span=signal, adjust=False).alias("signal_line")
        ])

        q = q.with_columns([
            (pl.col("macd_line") - pl.col("signal_line")).alias("histogram")
        ])

        # ── FASE B: RSI Wilder(14) ─────────────────────────────────────────────
        q = q.with_columns([
            self.rsi_expr(close=pl.col("close"), length=rsi_period).alias("rsi")
        ])

        # ── FASE C: EMA contexto ───────────────────────────────────────────────
        q = q.with_columns([
            pl.col("close").ewm_mean(span=ma_window, adjust=False).alias("ema_ctx")
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE D — Rotura EMA reciente (últimas N=4 velas)
        #
        # Cruce alcista puntual: close_prev ≤ ema_prev AND close > ema_ctx
        # Cruce bajista puntual: close_prev ≥ ema_prev AND close < ema_ctx
        #
        # Rotura reciente: rolling_max del cruce puntual en ventana N=4
        # Si cualquiera de las últimas 4 velas tuvo cruce → rotura reciente = True
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            pl.col("close")  .shift(1).alias("_close_prev"),
            pl.col("ema_ctx").shift(1).alias("_ema_ctx_prev"),
        ])

        # Cruce puntual de EMA
        q = q.with_columns([
            (
                (pl.col("_close_prev") <= pl.col("_ema_ctx_prev")) &
                (pl.col("close")       >  pl.col("ema_ctx"))
            ).fill_null(False).cast(pl.Int32).alias("_cruce_ema_alc"),

            (
                (pl.col("_close_prev") >= pl.col("_ema_ctx_prev")) &
                (pl.col("close")       <  pl.col("ema_ctx"))
            ).fill_null(False).cast(pl.Int32).alias("_cruce_ema_baj"),
        ])

        # Rotura reciente: algún cruce en las últimas N=4 velas
        q = q.with_columns([
            (
                pl.col("_cruce_ema_alc")
                .rolling_max(window_size=n_rotura) >= 1
            ).fill_null(False).alias("_rotura_alc"),

            (
                pl.col("_cruce_ema_baj")
                .rolling_max(window_size=n_rotura) >= 1
            ).fill_null(False).alias("_rotura_baj"),
        ])

        # ── FASE E: Cruce MACD via cambio signo histograma ────────────────────
        q = q.with_columns([
            pl.col("histogram").shift(1).fill_null(0.0).alias("_hist_prev")
        ])

        q = q.with_columns([
            (
                (pl.col("_hist_prev") <= 0.0) &
                (pl.col("histogram")  >  0.0)
            ).fill_null(False).alias("_cruce_macd_long"),

            (
                (pl.col("_hist_prev") >= 0.0) &
                (pl.col("histogram")  <  0.0)
            ).fill_null(False).alias("_cruce_macd_short"),
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE F — Condiciones base: triple confirmación
        #
        # LONG:
        #   - Cruce MACD alcista (trigger)
        #   - RSI entre 50 y 70 (momentum alcista, no sobrecomprado)
        #   - Rotura EMA alcista en últimas 4 velas (contexto fresco)
        #
        # SHORT:
        #   - Cruce MACD bajista (trigger)
        #   - RSI entre 30 y 50 (momentum bajista, no sobrevendido)
        #   - Rotura EMA bajista en últimas 4 velas (contexto fresco)
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            (
                pl.col("_cruce_macd_long") &
                (pl.col("rsi") >  rsi_lo_long) &
                (pl.col("rsi") <= rsi_hi_long) &
                pl.col("_rotura_alc") &
                pl.col("rsi").is_not_null()
            ).fill_null(False).alias("_cond_long"),

            (
                pl.col("_cruce_macd_short") &
                (pl.col("rsi") >= rsi_lo_short) &
                (pl.col("rsi") <  rsi_hi_short) &
                pl.col("_rotura_baj") &
                pl.col("rsi").is_not_null()
            ).fill_null(False).alias("_cond_short"),
        ])

        # ── FASE G: Flancos ────────────────────────────────────────────────────
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

        # ── FASE H: finalize_signals ───────────────────────────────────────────
        return self.finalize_signals(
            q,
            keep_cols=[
                "macd_line",
                "signal_line",
                "histogram",
                "rsi",
                "ema_ctx",
            ],
        )