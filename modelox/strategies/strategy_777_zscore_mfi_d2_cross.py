from __future__ import annotations

"""Strategy 777 — ZScore + MFI 2nd Derivative Cross

REGLAS DE ENTRADA
- SHORT:
  - zscore > z_threshold  (z_threshold optimizable 1.5–2.5)
  - mfi_d2 cruza 0 hacia abajo: (mfi_d2[t-1] >= 0) y (mfi_d2[t] < 0)
  - (y zscore sigue > z_threshold en la vela del cruce)

- LONG (inversa):
  - zscore < -z_threshold
  - mfi_d2 cruza 0 hacia arriba: (mfi_d2[t-1] <= 0) y (mfi_d2[t] > 0)

SALIDAS
- Trailing stop basado en ATR (actualiza vela a vela, solo favorablemente)
- Stop loss de emergencia (% fijo desde entrada)

Nota: El motor usa `close[exit_idx]` como precio de salida.
"""

from typing import Any, Dict, Optional

import numpy as np
import polars as pl
from numba import njit

from logic.indicators import IndicadorFactory
from modelox.core.types import ExitDecision
from modelox.strategies.indicator_specs import cfg_atr, cfg_mfi, cfg_mfi_d2, cfg_zscore


class Strategy777ZScoreMFID2Cross:
    combinacion_id = 777
    name = "ZScore_MFI_D2_Cross"

    __indicators_used = ["zscore", "mfi_d2", "atr"]

    @classmethod
    def get_indicators_used(cls):
        return cls.__indicators_used

    parametros_optuna = {
        "z_length": (20, 60, 5),
        "z_threshold": (1.5, 2.5, 0.1),
        "mfi_period": (14, 14, 1),
        "atr_period": (10, 20, 2),
        "trailing_atr_mult": (1.0, 4.0, 0.2),
        "emergency_sl_pct": (3.0, 20.0, 1.0),
    }

    TIMEOUT_BARS = 220

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        return {
            "z_length": int(trial.suggest_int("z_length", 20, 60, step=5)),
            "z_threshold": float(trial.suggest_float("z_threshold", 1.5, 2.5, step=0.1)),
            "mfi_period": 14,
            "atr_period": int(trial.suggest_int("atr_period", 10, 20, step=2)),
            "trailing_atr_mult": float(trial.suggest_float("trailing_atr_mult", 1.0, 4.0, step=0.2)),
            "emergency_sl_pct": float(trial.suggest_float("emergency_sl_pct", 3.0, 20.0, step=1.0)),
        }

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        params["__indicators_used"] = self.get_indicators_used()

        z_length = int(params.get("z_length", 40))
        z_threshold = float(params.get("z_threshold", 2.0))
        mfi_period = int(params.get("mfi_period", 14))
        atr_period = int(params.get("atr_period", 14))

        params["__warmup_bars"] = max(max(z_length, mfi_period, atr_period) + 6, 60)

        ind_config = {
            "atr": cfg_atr(period=atr_period, out="atr"),
            "mfi": cfg_mfi(period=mfi_period, out="mfi"),
            "mfi_d2": cfg_mfi_d2(mfi_col="mfi", out="mfi_d2"),
            "zscore": cfg_zscore(col="close", window=z_length, out="zscore"),
        }
        df = IndicadorFactory.procesar(df, ind_config)

        z = pl.col("zscore")
        d2 = pl.col("mfi_d2")

        cross_down = (d2.shift(1) >= 0) & (d2 < 0)
        cross_up = (d2.shift(1) <= 0) & (d2 > 0)

        signal_short = (z > z_threshold) & cross_down
        signal_long = (z < -z_threshold) & cross_up

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

        trailing_atr_mult = float(params.get("trailing_atr_mult", 2.0))
        emergency_sl_pct = float(params.get("emergency_sl_pct", 10.0))

        close_arr = df["close"].to_numpy().astype(np.float64)
        high_arr = df["high"].to_numpy().astype(np.float64)
        low_arr = df["low"].to_numpy().astype(np.float64)
        atr_arr = df["atr"].to_numpy().astype(np.float64)

        max_bars = min(self.TIMEOUT_BARS, n - entry_idx - 1)

        exit_idx, reason_code = _decide_exit_trailing_sl(
            entry_idx,
            float(entry_price),
            close_arr,
            high_arr,
            low_arr,
            atr_arr,
            float(trailing_atr_mult),
            float(emergency_sl_pct),
            1 if is_long else -1,
            int(max_bars),
        )

        reason_map = {
            0: None,
            1: "TRAILING_STOP",
            2: "EMERGENCY_SL",
            3: "TIME_EXIT",
        }

        if exit_idx >= 0 and reason_code in reason_map and reason_map[reason_code] is not None:
            return ExitDecision(exit_idx=int(exit_idx), reason=str(reason_map[reason_code]))

        return ExitDecision(exit_idx=min(entry_idx + max_bars, n - 1), reason="TIME_EXIT")


@njit(cache=True, fastmath=True)
def _decide_exit_trailing_sl(
    entry_idx: int,
    entry_price: float,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    trailing_atr_mult: float,
    emergency_sl_pct: float,
    side_flag: int,
    max_bars: int,
) -> tuple[int, int]:
    """Trailing stop basado en ATR + SL de emergencia."""

    n = len(close)
    last_allowed = min(n - 1, entry_idx + max_bars)

    # SL de emergencia
    emergency_factor = emergency_sl_pct / 100.0
    if side_flag == 1:
        emergency_sl = entry_price * (1.0 - emergency_factor)
    else:
        emergency_sl = entry_price * (1.0 + emergency_factor)

    # Trailing
    best_price = entry_price
    atr0 = atr[entry_idx]
    if np.isnan(atr0):
        atr0 = 0.0

    if side_flag == 1:
        trailing_stop = entry_price - atr0 * trailing_atr_mult
    else:
        trailing_stop = entry_price + atr0 * trailing_atr_mult

    for i in range(entry_idx + 1, last_allowed + 1):
        hi = high[i]
        lo = low[i]
        cl = close[i]
        a = atr[i]

        if np.isnan(hi) or np.isnan(lo) or np.isnan(cl):
            continue
        if np.isnan(a):
            a = 0.0

        # Emergency SL first
        if side_flag == 1:
            if lo <= emergency_sl:
                return i, 2
        else:
            if hi >= emergency_sl:
                return i, 2

        # Update trailing only in favorable direction
        if side_flag == 1:
            if cl > best_price:
                best_price = cl
                new_trailing = best_price - a * trailing_atr_mult
                if new_trailing > trailing_stop:
                    trailing_stop = new_trailing
            if lo <= trailing_stop:
                return i, 1
        else:
            if cl < best_price:
                best_price = cl
                new_trailing = best_price + a * trailing_atr_mult
                if new_trailing < trailing_stop:
                    trailing_stop = new_trailing
            if hi >= trailing_stop:
                return i, 1

    return last_allowed, 3
