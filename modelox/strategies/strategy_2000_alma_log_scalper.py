from __future__ import annotations

"""Strategy 2000 — ALMA Log-Scalper (Momentum Breakout + Trend Filter)

Variables:
- A5  (trigger): ALMA length 5
- A25 (filter):  ALMA length 25
- A60 (trend):   ALMA length 60

ENTRY
LONG:
- Structure: A5 > A25 and A25 > A60
- Trigger: close crosses above A5

SHORT:
- Structure: A5 < A25 and A25 < A60
- Trigger: close crosses below A5

EXIT
- LONG: close < A25
- SHORT: close > A25

Nota: exits usan la propia cinta como stop dinámico.
"""

from typing import Any, Dict, Optional

import numpy as np
import polars as pl
from numba import njit

from logic.indicators import IndicadorFactory
from modelox.core.types import ExitDecision
from modelox.strategies.indicator_specs import cfg_alma


@njit(cache=True)
def _exit_alma_mid_numba(
    entry_idx: int,
    close: np.ndarray,
    alma_mid: np.ndarray,
    side_flag: int,
    max_bars: int,
) -> int:
    """Return first exit index based on close vs alma_mid.

    LONG exits when close < alma_mid.
    SHORT exits when close > alma_mid.
    """
    n = len(close)
    last_allowed = min(n - 1, entry_idx + max_bars)

    for i in range(entry_idx + 1, last_allowed + 1):
        c = close[i]
        m = alma_mid[i]
        if np.isnan(c) or np.isnan(m):
            continue

        if side_flag == 1:
            if c < m:
                return i
        else:
            if c > m:
                return i

    return last_allowed


class Strategy2000ALMALogScalper:
    combinacion_id = 2000
    name = "ALMA_LogScalper"

    __indicators_used = ["alma_5", "alma_25", "alma_60"]

    @classmethod
    def get_indicators_used(cls):
        return cls.__indicators_used

    # Para cabecera de ejecutar.py
    parametros_optuna = {
        # Se optimizan con restricción jerárquica (fast < mid < slow)
        # Nota: la restricción se aplica en suggest_params con límites dinámicos.
        "alma_slow": (30, 120, 5),
        "alma_mid": (10, 60, 1),
        "alma_fast": (3, 20, 1),
        "alma_offset": (0.50, 0.95, 0.05),
        "alma_sigma": (2.0, 10.0, 0.5),
    }

    TIMEOUT_BARS = 260

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        # Optimizables con alineación jerárquica obligatoria:
        # fast < mid < slow
        # (Evitamos el "ruido" de cruces constantes).
        if trial is None:
            return {
                "alma_fast": 5,
                "alma_mid": 25,
                "alma_slow": 60,
                "alma_offset": 0.85,
                "alma_sigma": 7.0,
            }

        alma_slow = int(trial.suggest_int("alma_slow", 30, 120, step=5))

        mid_hi = max(11, min(60, alma_slow - 2))
        alma_mid = int(trial.suggest_int("alma_mid", 10, mid_hi, step=1))

        fast_hi = max(4, min(20, alma_mid - 1))
        alma_fast = int(trial.suggest_int("alma_fast", 3, fast_hi, step=1))

        alma_offset = float(trial.suggest_float("alma_offset", 0.50, 0.95, step=0.05))
        alma_sigma = float(trial.suggest_float("alma_sigma", 2.0, 10.0, step=0.5))

        return {
            "alma_fast": alma_fast,
            "alma_mid": alma_mid,
            "alma_slow": alma_slow,
            "alma_offset": alma_offset,
            "alma_sigma": alma_sigma,
        }

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        params["__indicators_used"] = self.get_indicators_used()

        a_fast = int(params.get("alma_fast", 5))
        a_mid = int(params.get("alma_mid", 25))
        a_slow = int(params.get("alma_slow", 60))
        offset = float(params.get("alma_offset", 0.85))
        sigma = float(params.get("alma_sigma", 7.0))

        params["__warmup_bars"] = max(a_fast, a_mid, a_slow) + 10

        # IndicadorFactory no admite el mismo key repetido; calculamos las 3 ALMAs en 3 pasadas.
        df = IndicadorFactory.procesar(
            df,
            {"alma": cfg_alma(length=a_fast, offset=offset, sigma=sigma, col="close", out="alma_5")},
        )
        df = IndicadorFactory.procesar(
            df,
            {"alma": cfg_alma(length=a_mid, offset=offset, sigma=sigma, col="close", out="alma_25")},
        )
        df = IndicadorFactory.procesar(
            df,
            {"alma": cfg_alma(length=a_slow, offset=offset, sigma=sigma, col="close", out="alma_60")},
        )

        close = pl.col("close")
        a5 = pl.col("alma_5")
        a25 = pl.col("alma_25")
        a60 = pl.col("alma_60")

        # Structure filter
        structure_long = (a5 > a25) & (a25 > a60)
        structure_short = (a5 < a25) & (a25 < a60)

        # Trigger: close crosses A5
        cross_up = (close > a5) & (close.shift(1) <= a5.shift(1))
        cross_down = (close < a5) & (close.shift(1) >= a5.shift(1))

        signal_long = structure_long & cross_up
        signal_short = structure_short & cross_down

        return df.with_columns(
            [
                signal_long.fill_null(False).alias("signal_long"),
                signal_short.fill_null(False).alias("signal_short"),
            ]
        )

    def decide_exit(
        self,
        df: pl.DataFrame,
        params: Dict[str, Any],
        entry_idx: int,
        entry_price: float,
        side: str,
        *,
        saldo_apertura: float,
    ) -> Optional[ExitDecision]:
        n = len(df)
        if entry_idx >= n - 1:
            return None

        is_long = side.upper() == "LONG"
        is_short = side.upper() == "SHORT"
        if not (is_long or is_short):
            return ExitDecision(exit_idx=min(entry_idx + 1, n - 1), reason="UNKNOWN_SIDE")

        if "alma_25" not in df.columns:
            return ExitDecision(exit_idx=min(entry_idx + 1, n - 1), reason="NO_ALMA_25")

        close_arr = df["close"].to_numpy().astype(np.float64)
        alma_mid_arr = df["alma_25"].to_numpy().astype(np.float64)

        max_bars = min(self.TIMEOUT_BARS, n - entry_idx - 1)
        exit_idx = _exit_alma_mid_numba(
            int(entry_idx),
            close_arr,
            alma_mid_arr,
            1 if is_long else -1,
            int(max_bars),
        )

        reason = "ALMA25_LOSS" if is_long else "ALMA25_RECLAIM"
        return ExitDecision(exit_idx=int(exit_idx), reason=reason)
