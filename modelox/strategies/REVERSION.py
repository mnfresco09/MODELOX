from __future__ import annotations
"""
================================================================================
MODELOX / STRATEGIES / REVERSION.py
================================================================================
ID        : 2
NOMBRE    : REVERSION — RSI Cross Reversion
MERCADO   : Crypto / cualquier activo
TIMEFRAME : El que el sistema pase (sin timeframe fijo)
--------------------------------------------------------------------------------
CONCEPTO:
  Estrategia de reversión a la media basada en cruces del RSI desde zonas
  extremas. Cuando el RSI estaba en zona de sobreventa y cruza hacia arriba
  la banda inferior → probable rebote alcista (LONG). Cuando estaba en zona
  de sobrecompra y cruza hacia abajo la banda superior → probable corrección
  bajista (SHORT).

  BANDAS SIMÉTRICAS (un único parámetro):
    rsi_upper ∈ [60, 80]  →  banda superior = rsi_upper
                            banda inferior = 100 - rsi_upper
    Ejemplo: rsi_upper=70  →  bandas [30, 70]
    Ejemplo: rsi_upper=65  →  bandas [35, 65]
    Ejemplo: rsi_upper=75  →  bandas [25, 75]

  SEÑAL LONG  (cruce alcista desde sobreventa):
    RSI[t-1] < lower_threshold  AND  RSI[t] >= lower_threshold

  SEÑAL SHORT (cruce bajista desde sobrecompra):
    RSI[t-1] > upper_threshold  AND  RSI[t] <= upper_threshold

--------------------------------------------------------------------------------
PARÁMETROS OPTUNA:
  rsi_window  : int   [5, 30]    — período del RSI
  rsi_upper   : float [60, 80]   — banda superior (inferior = 100 - rsi_upper)

SALIDAS: estándar (motor de exits)
================================================================================
"""

from typing import Any, Dict, List, Optional
import polars as pl

from modelox.strategies.ESTRATEGIA_BASE import EstrategiaBase


class EstrategiaID2REVERSION(EstrategiaBase):
    """
    REVERSION — RSI Cross Reversion.
    Entradas en cruce del RSI desde zonas extremas simétricas.
    """

    combinacion_id: int          = 2
    name: str                    = "REVERSION"
    SALIDAS_PERSONALIZADAS: bool = False
    timeframe_entry: Optional[str] = None
    timeframe_exit:  Optional[str] = None

    # ──────────────────────────────────────────────────────────────────────────
    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        return {
            "rsi_window": trial.suggest_int  ("rsi_window", 14,  14, step=2),
            "rsi_upper":  trial.suggest_float("rsi_upper",  65, 65, step=1),
        }

    def get_required_timeframes(self, params: Dict[str, Any]) -> List[str]:
        return []

    # ──────────────────────────────────────────────────────────────────────────
    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        """
        FASES:
          A) Calcular RSI con la ventana parametrizada
          B) Derivar bandas simétricas desde rsi_upper
          C) Detectar cruces (flanco de subida / bajada)
          D) finalize_signals
        """
        self._init_params_metadata(params)
        self._require_columns(df, ["timestamp", "close"])

        rsi_window   = int  (params.get("rsi_window", 14))
        rsi_upper    = float(params.get("rsi_upper",  70.0))
        rsi_lower    = 100.0 - rsi_upper          # simétrico

        params["__warmup_bars"]     = rsi_window * 3 + 4
        params["__indicators_used"] = ["rsi"]
        params["__indicator_specs"] = {
            "rsi": {"panel": "sub1", "color": "#F59E0B", "tipo": "line"},
        }
        params["__indicator_bounds"] = {
            "rsi": {"lo": rsi_lower, "hi": rsi_upper, "mid": 50.0},
        }

        q = df.lazy()

        # ── FASE A: RSI ───────────────────────────────────────────────────────
        q = q.with_columns([
            self.rsi_expr(close=pl.col("close"), length=rsi_window).alias("rsi")
        ])

        # ── FASE B: RSI anterior (shift 1) ───────────────────────────────────
        q = q.with_columns([
            pl.col("rsi").shift(1).alias("_rsi_prev")
        ])

        # ── FASE C: Cruces ────────────────────────────────────────────────────
        # LONG:  rsi_prev < lower  AND  rsi >= lower   (cruce alcista)
        # SHORT: rsi_prev > upper  AND  rsi <= upper   (cruce bajista)
        q = q.with_columns([
            self._as_bool(
                (pl.col("_rsi_prev") < rsi_lower) &
                (pl.col("rsi") >= rsi_lower)       &
                pl.col("rsi").is_not_null()        &
                pl.col("_rsi_prev").is_not_null()
            ).alias("signal_long"),

            self._as_bool(
                (pl.col("_rsi_prev") > rsi_upper) &
                (pl.col("rsi") <= rsi_upper)       &
                pl.col("rsi").is_not_null()        &
                pl.col("_rsi_prev").is_not_null()
            ).alias("signal_short"),
        ])

        return self.finalize_signals(q, keep_cols=["rsi"])
