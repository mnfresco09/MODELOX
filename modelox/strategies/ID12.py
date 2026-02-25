from __future__ import annotations
"""
================================================================================
MODELOX/STRATEGIES/ID12.PY
================================================================================
ID        : 12
NOMBRE    : BREAKVOL — Rotura de Rango por Expansión de Sigma + ATR Expanding
MERCADO   : Crypto
TIMEFRAME : El que el sistema pase (sin timeframe fijo)
--------------------------------------------------------------------------------
LÓGICA DE ENTRADA:
  DETECCIÓN DE RANGO COMPRIMIDO:
    - r_i = ln(close_i / close_{i-1})
    - sigma = std(r, base_window)              (volatilidad realizada actual)
    - sigma_mean = mean(sigma, sigma_mean_window) (nivel medio de volatilidad)
    - Rango comprimido cuando: sigma < sigma_mean (vol por debajo de su media)

  ROTURA DEL RANGO:
    - ratio_sigma = sigma / sigma_mean
    - Rotura cuando: ratio_sigma >= 1.5 (sigma supera 1.5x su media)
    - La aceleración relativa de volatilidad ES la rotura
    - Confirma que el mercado salió del estado de compresión

  CONFIRMACIÓN ATR EXPANDING:
    - ATR(14) Wilder = volatilidad absoluta del precio
    - atr_mean = rolling_mean(ATR, base_window)
    - ATR expanding: ATR > atr_mean
    - Doble confirmación: sigma relativa Y ATR absoluto ambos expandiendo

  DIRECCIÓN:
    - precio_medio = (O+H+L+C) / 4
    - sma_precio = rolling_mean(precio_medio, base_window)
    - Long  : precio_medio > sma_precio (precio por encima de su media)
    - Short : precio_medio < sma_precio (precio por debajo de su media)

  SEÑAL LONG  : ratio_sigma >= 1.5 AND ATR > atr_mean AND precio > sma
                Solo en FLANCO. Mutuamente excluyente con SHORT.

  SEÑAL SHORT : ratio_sigma >= 1.5 AND ATR > atr_mean AND precio < sma
                Solo en FLANCO. Mutuamente excluyente con LONG.

SALIDAS:
  SALIDAS_PERSONALIZADAS = False
  → SL / TP controlados 100% por exits.py (engine global)

RANGOS OPTUNA (2 parámetros):
  base_window        : int [10, 50]  step=5   (ventana sigma + ATR + SMA precio)
  sigma_mean_window  : int [20, 100] step=10  (ventana media de sigma — umbral)

PARÁMETROS FIJOS:
  sigma_ratio : 1.5  (umbral de expansión relativa — anti-overfitting)
  atr_period  : 14   (ATR Wilder clásico)

TIMEFRAME:
  Sin timeframe fijo — usa el que el sistema pase.

IMPLEMENTACIÓN:
  - 100% Polars vectorial (Zero Pandas, Zero loops por fila)
  - ATR via ewm_mean (Wilder-like) Polars nativo
  - Sigma via rolling_std nativo Polars
  - 1 solo .collect() al final en finalize_signals()
================================================================================
"""

from typing import Any, Dict, List
import polars as pl

from modelox.strategies.ESTRATEGIA_BASE import EstrategiaBase


class EstrategiaID12BREAKVOL(EstrategiaBase):
    """
    BREAKVOL — Detecta roturas de rango comprimido mediante la aceleración
    relativa de sigma (ratio sigma/sigma_mean >= 1.5) confirmada por ATR
    expanding. Dirección determinada por precio vs su media rolling.
    """

    # ── Identidad ──────────────────────────────────────────────────────────────
    combinacion_id: int = 12
    name: str           = "BREAKVOL"

    # ── Salidas: engine global vía exits.py ────────────────────────────────────
    SALIDAS_PERSONALIZADAS: bool = False

    # ── Timeframe: None — usa el que el sistema pase ───────────────────────────

    # ==========================================================================
    # ESPACIO DE BÚSQUEDA OPTUNA — 2 parámetros
    # ==========================================================================
    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        """
        2 parámetros Optuna (anti-overfitting).
        Fijos: sigma_ratio=1.5, atr_period=14
        Total combinaciones: 9 x 9 = 81 — espacio muy controlado.
        """
        return {
            "base_window":       trial.suggest_int("base_window",       10,  50, step=5),
            "sigma_mean_window": trial.suggest_int("sigma_mean_window", 20, 100, step=10),
        }

    # ==========================================================================
    # TIMEFRAMES EXTRA (MTF) — no requeridos
    # ==========================================================================
    def get_required_timeframes(self, params: Dict[str, Any]) -> List[str]:
        return []

    # ==========================================================================
    # GENERATE SIGNALS — 100% POLARS VECTORIAL, 1 COLLECT
    # ==========================================================================
    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        """
        FASES:
          A) Retorno logarítmico + precio medio OHLC
          B) Sigma rolling (volatilidad realizada)
          C) Media de sigma (umbral dinámico de compresión)
          D) Ratio sigma / sigma_mean (detector de rotura)
          E) ATR Wilder(14) + media rolling (confirmador)
          F) SMA del precio medio (determinador de dirección)
          G) Condiciones base Long/Short
          H) Flancos: señal solo en cambio False→True
          I) finalize_signals (1 collect)
        """

        # ── Init ───────────────────────────────────────────────────────────────
        self._init_params_metadata(params)
        self._require_columns(df, ["timestamp", "open", "high", "low", "close"])

        # ── Parámetros ────────────────────────────────────────────────────────
        base_w       = int  (params.get("base_window",       20))
        sigma_mean_w = int  (params.get("sigma_mean_window", 50))
        sigma_ratio  = 1.5   # FIJO: umbral de expansión relativa
        atr_period   = 14    # FIJO: ATR Wilder clásico

        # ── Metadata para reporter / plots ────────────────────────────────────
        params["__warmup_bars"]     = sigma_mean_w + base_w + atr_period + 2
        params["__indicators_used"] = [
            "sigma", "sigma_mean", "ratio_sigma", "atr", "atr_mean", "sma_precio"
        ]
        params["__indicator_specs"] = {
            "sigma":       {"panel": "sub1", "color": "#FF9800", "tipo": "line"},
            "sigma_mean":  {"panel": "sub1", "color": "#FFD600", "tipo": "line"},
            "ratio_sigma": {"panel": "sub2", "color": "#AB47BC", "tipo": "line"},
            "atr":         {"panel": "sub3", "color": "#00BCD4", "tipo": "line"},
            "atr_mean":    {"panel": "sub3", "color": "#0088AA", "tipo": "line"},
            "sma_precio":  {"panel": "main", "color": "#AAAAAA", "tipo": "line"},
        }
        params["__indicator_bounds"] = {
            "sigma":       {"lo": 0.0,         "hi": None, "mid": None},
            "ratio_sigma": {"lo": 0.0,         "hi": None, "mid": sigma_ratio},
            "atr":         {"lo": 0.0,         "hi": None, "mid": None},
        }

        # ══════════════════════════════════════════════════════════════════════
        # FASE A — Retorno logarítmico + precio medio OHLC
        # ══════════════════════════════════════════════════════════════════════
        q = df.lazy().with_columns([
            (pl.col("close") / pl.col("close").shift(1))
            .log(base=2.718281828)
            .alias("log_return"),

            ((pl.col("open") + pl.col("high") + pl.col("low") + pl.col("close")) / 4.0)
            .alias("precio_medio"),
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE B — Sigma rolling: std(log_return, base_window)
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            pl.col("log_return")
            .rolling_std(window_size=base_w)
            .alias("sigma")
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE C — Media de sigma (umbral dinámico)
        # sigma < sigma_mean → mercado en compresión
        # sigma > sigma_mean → mercado expandiendo
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            pl.col("sigma")
            .rolling_mean(window_size=sigma_mean_w)
            .alias("sigma_mean")
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE D — Ratio sigma / sigma_mean
        # ratio >= 1.5 → sigma supera 1.5x su media = rotura de compresión
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            pl.when(pl.col("sigma_mean").abs() > 1e-12)
            .then(pl.col("sigma") / pl.col("sigma_mean"))
            .otherwise(0.0)
            .alias("ratio_sigma")
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE E — ATR Wilder(14) + media rolling (confirmador)
        # True Range = max(H-L, |H-Cprev|, |L-Cprev|)
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
            .ewm_mean(com=atr_period - 1, min_periods=atr_period)
            .alias("atr")
        ])

        q = q.with_columns([
            pl.col("atr")
            .rolling_mean(window_size=base_w)
            .alias("atr_mean")
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE F — SMA del precio medio (determinador de dirección)
        # precio_medio > sma_precio → mercado por encima de su media → Long
        # precio_medio < sma_precio → mercado por debajo de su media → Short
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            pl.col("precio_medio")
            .rolling_mean(window_size=base_w)
            .alias("sma_precio")
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE G — Condiciones base (mutuamente excluyentes)
        # Rotura: ratio_sigma >= 1.5 AND ATR > atr_mean
        # Long:  rotura AND precio_medio > sma_precio
        # Short: rotura AND precio_medio < sma_precio
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            (
                (pl.col("ratio_sigma")  >= sigma_ratio) &
                (pl.col("atr")          >  pl.col("atr_mean")) &
                (pl.col("precio_medio") >  pl.col("sma_precio")) &
                pl.col("sigma_mean").is_not_null() &
                pl.col("atr_mean").is_not_null()
            ).fill_null(False).alias("_cond_long"),

            (
                (pl.col("ratio_sigma")  >= sigma_ratio) &
                (pl.col("atr")          >  pl.col("atr_mean")) &
                (pl.col("precio_medio") <  pl.col("sma_precio")) &
                pl.col("sigma_mean").is_not_null() &
                pl.col("atr_mean").is_not_null()
            ).fill_null(False).alias("_cond_short"),
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE H — Flancos: señal solo en cambio False→True
        # ══════════════════════════════════════════════════════════════════════
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

        # ══════════════════════════════════════════════════════════════════════
        # FASE I — finalize_signals (1 collect + validación contrato)
        # ══════════════════════════════════════════════════════════════════════
        return self.finalize_signals(
            q,
            keep_cols=[
                "sigma",
                "sigma_mean",
                "ratio_sigma",
                "atr",
                "atr_mean",
                "sma_precio",
            ],
        )