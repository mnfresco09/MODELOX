from __future__ import annotations
"""
================================================================================
MODELOX/STRATEGIES/ID26.PY
================================================================================
ID        : 26
NOMBRE    : NATRTREND — ATR Normalizado Z-score + Retorno Acumulado Dirección
MERCADO   : Crypto
TIMEFRAME : El que el sistema pase (sin timeframe fijo)
--------------------------------------------------------------------------------
CONCEPTO:
  ATR/precio (ATR normalizado) mide la volatilidad relativa al nivel del precio.
  Robusto entre activos: BTC a 60k y una altcoin a 0.001 tienen la misma escala.

  Z-score del ATR normalizado detecta cuándo la volatilidad es anómalamente
  alta respecto a su propia historia reciente — señal de movimiento con fuerza.

  Retorno acumulado rolling determina la dirección: si el precio ha subido
  más de lo que ha bajado en la ventana → momentum alcista, y viceversa.

  EDGE: Volatilidad anómala + momentum acumulado en la misma dirección =
  movimiento con participación real, no ruido aleatorio.

--------------------------------------------------------------------------------
LÓGICA DE ENTRADA:
  ATR NORMALIZADO (Wilder, period=14, fijo):
    ATR      = Wilder ATR(14)
    natr     = ATR / close * 100  (% del precio)

  Z-SCORE ATR NORMALIZADO:
    z_natr   = (natr - mean(natr, vol_window)) / std(natr, vol_window)
    z_natr ≥ z_atr_threshold → volatilidad anómalamente alta → operar

  RETORNO ACUMULADO ROLLING (misma ventana):
    ret_acum = close / close.shift(vol_window) - 1
    ret_acum > 0 → momentum alcista → Long
    ret_acum < 0 → momentum bajista → Short

  SEÑAL LONG  : z_natr ≥ z_atr_threshold AND ret_acum > 0  [FLANCO]
  SEÑAL SHORT : z_natr ≥ z_atr_threshold AND ret_acum < 0  [FLANCO]

SALIDAS:
  SALIDAS_PERSONALIZADAS = False

RANGOS OPTUNA (2 parámetros):
  vol_window      : int   [20, 80]  step=10
  z_atr_threshold : float [0.5,2.0] step=0.25

PARÁMETROS FIJOS:
  atr_period : 14  (Wilder clásico)
  Misma ventana para Z-score y retorno acumulado (anti-overfitting)

TIMEFRAME: Sin timeframe fijo — usa el que el sistema pase.

IMPLEMENTACIÓN:
  - ATR Wilder via ewm_mean nativo Polars (com=13)
  - NATR y Z-score 100% Polars vectorial
  - Retorno acumulado via shift(vol_window)
  - 1 solo .collect() al final en finalize_signals()
================================================================================
"""

from typing import Any, Dict, List, Optional
import polars as pl

from modelox.strategies.ESTRATEGIA_BASE import EstrategiaBase


class EstrategiaID26NATRTREND(EstrategiaBase):
    """
    NATRTREND — ATR Normalizado Z-score como filtro de volatilidad anómala
    + Retorno Acumulado Rolling como confirmador de dirección.
    Simple, robusto entre activos y 100% vectorial.
    """

    combinacion_id: int          = 26
    name: str                    = "NATRTREND"
    SALIDAS_PERSONALIZADAS: bool = False
    timeframe_entry: Optional[str] = None
    timeframe_exit:  Optional[str] = None

    # ==========================================================================
    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        """
        2 parámetros Optuna (máximo anti-overfitting).
        Fijos: atr_period=14, vol_window compartida para Z y retorno
        Total combinaciones: 7 x 7 = 49 — espacio mínimo.
        """
        return {
            "vol_window":      trial.suggest_int  ("vol_window",      40,  40, step=1),
            "z_atr_threshold": trial.suggest_float("z_atr_threshold", 2.5, 2.5, step=0.1),
        }

    def get_required_timeframes(self, params: Dict[str, Any]) -> List[str]:
        return []

    # ==========================================================================
    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        """
        FASES:
          A) True Range + ATR Wilder(14)
          B) NATR = ATR / close * 100
          C) Z-score NATR rolling
          D) Retorno acumulado rolling (misma ventana)
          E) Condiciones base: z_natr ≥ umbral AND dirección retorno
          F) Flancos
          G) finalize_signals (1 collect)
        """

        self._init_params_metadata(params)
        self._require_columns(df, ["timestamp", "high", "low", "close"])

        vol_window      = int  (params.get("vol_window",      40))
        z_atr_threshold = float(params.get("z_atr_threshold", 1.0))
        atr_period      = 14   # FIJO: Wilder clásico

        params["__warmup_bars"]     = vol_window * 2 + atr_period + 2
        params["__indicators_used"] = ["natr", "z_natr", "ret_acum"]
        params["__indicator_specs"] = {
            "natr":    {"panel": "sub1", "color": "#FFD600", "tipo": "line"},
            "z_natr":  {"panel": "sub2", "color": "#FF9800", "tipo": "line"},
            "ret_acum":{"panel": "sub3", "color": "#00E676", "tipo": "histogram"},
        }
        params["__indicator_bounds"] = {
            "z_natr":   {"lo": None,             "hi": None, "mid": z_atr_threshold},
            "ret_acum": {"lo": None,             "hi": None, "mid": 0.0},
        }

        q = df.lazy()

        # ══════════════════════════════════════════════════════════════════════
        # FASE A — True Range + ATR Wilder(14)
        # TR  = max(H-L, |H-Cprev|, |L-Cprev|)
        # ATR = EWM(TR, com=atr_period-1) — equivale a Wilder smoothing
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            pl.col("close").shift(1).alias("_close_prev")
        ])

        q = q.with_columns([
            pl.max_horizontal(
                pl.col("high") - pl.col("low"),
                (pl.col("high") - pl.col("_close_prev")).abs(),
                (pl.col("low")  - pl.col("_close_prev")).abs(),
            ).alias("_tr")
        ])

        q = q.with_columns([
            pl.col("_tr")
            .ewm_mean(com=atr_period - 1, adjust=False)
            .alias("_atr")
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE B — NATR = ATR / close * 100
        # Normaliza la volatilidad por el nivel del precio
        # Comparable entre activos de distinta escala
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            pl.when(pl.col("close").abs() > 1e-12)
            .then(pl.col("_atr") / pl.col("close") * 100.0)
            .otherwise(0.0)
            .alias("natr")
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE C — Z-score NATR rolling
        # z_natr = (natr - mean(natr, vol_window)) / std(natr, vol_window)
        # z_natr ≥ threshold → volatilidad anómalamente alta → operar
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            pl.col("natr").rolling_mean(window_size=vol_window).alias("_natr_mean"),
            pl.col("natr").rolling_std (window_size=vol_window).alias("_natr_std"),
        ])

        q = q.with_columns([
            pl.when(pl.col("_natr_std").abs() > 1e-12)
            .then((pl.col("natr") - pl.col("_natr_mean")) / pl.col("_natr_std"))
            .otherwise(0.0)
            .alias("z_natr")
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE D — Retorno acumulado rolling (misma ventana)
        # ret_acum = close / close.shift(vol_window) - 1
        # > 0 → precio subió en la ventana → momentum alcista → Long
        # < 0 → precio bajó en la ventana → momentum bajista → Short
        # Misma ventana que Z-score: anti-overfitting
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            pl.col("close").shift(vol_window).alias("_close_lag")
        ])

        q = q.with_columns([
            pl.when(pl.col("_close_lag").abs() > 1e-12)
            .then(pl.col("close") / pl.col("_close_lag") - 1.0)
            .otherwise(0.0)
            .alias("ret_acum")
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE E — Condiciones base
        # Long:  volatilidad anómala AND momentum alcista acumulado
        # Short: volatilidad anómala AND momentum bajista acumulado
        # La volatilidad anómala confirma que el movimiento tiene fuerza real
        # El retorno acumulado determina la dirección sin indicadores extra
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            (
                (pl.col("z_natr")   >= z_atr_threshold) &
                (pl.col("ret_acum") >  0.0) &
                pl.col("z_natr").is_not_null()
            ).fill_null(False).alias("_cond_long"),

            (
                (pl.col("z_natr")   >= z_atr_threshold) &
                (pl.col("ret_acum") <  0.0) &
                pl.col("z_natr").is_not_null()
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
                "natr",
                "z_natr",
                "ret_acum",
            ],
        )