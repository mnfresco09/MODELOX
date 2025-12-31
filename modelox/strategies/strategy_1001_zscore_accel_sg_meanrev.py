from __future__ import annotations

"""Strategy 1001 — ZScore + Acceleration (SG Sim) Mean Reversion (Intradía)

INDICADORES
A) Oscilador de Aceleración (SG Simulado, normalizado)
  s1 = WMA(close, 29)
  s2 = WMA(s1, 29)
  s3 = WMA(s2, 29)
  accel_raw = s1 - 2*s2 + s3
  accel_norm = 3 * tanh(zscore(accel_raw))  -> rango [-3, +3]

B) Z-Score del precio
  Z = (close - SMA(close,20)) / std(close,20)

ENTRADAS
LONG:
  - Setup: zscore < -1.5
  - Trigger: accel_norm cruza -2.0 hacia arriba  (prev < -2.0 AND curr > -2.0)

SHORT:
  - Setup: zscore > +1.5
  - Trigger: accel_norm cruza +2.0 hacia abajo  (prev > +2.0 AND curr < +2.0)

SALIDAS
A) SL Emergencia: fijo 1.0% desde entrada
B) Trailing stop ATR (14) con:
   - Activación cuando el precio se mueve a favor >= 3*ATR
   - Una vez activo: stop persigue a distancia 3*ATR, sin retroceso (ratchet)

Nota: el motor usa close[exit_idx] como precio de salida.
"""

from typing import Any, Dict, Optional

import numpy as np
import polars as pl
from numba import njit

from logic.indicators import IndicadorFactory
from modelox.core.types import ExitDecision
from modelox.strategies.indicator_specs import cfg_accel_sg, cfg_atr, cfg_zscore


class Strategy1001ZScoreAccelSGMeanReversion:
    combinacion_id = 1001
    name = "ZScore_AccelSG_MeanReversion"

    __indicators_used = ["zscore", "accel_sg", "atr"]

    @classmethod
    def get_indicators_used(cls):
        return cls.__indicators_used

    # Parámetros: según especificación (principalmente fijos)
    parametros_optuna = {
        # Se mantienen fijos para respetar la definición del sistema.
        # (Si quieres optimizarlos luego, los abrimos.)
    }

    # Constantes del sistema
    Z_WINDOW = 20
    Z_SETUP = 1.5

    ACC_WMA_PERIOD = 29
    ACC_Z_WINDOW = 20
    ACC_LEVEL = 2.0

    ATR_PERIOD = 14
    ATR_MULT = 3.0
    ACTIVATE_ATR_MULT = 3.0

    EMERGENCY_SL_PCT = 1.0

    TIMEOUT_BARS = 260

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        # Estrategia determinista (parámetros fijos por definición)
        return {
            "z_window": int(self.Z_WINDOW),
            "z_setup": float(self.Z_SETUP),
            "acc_wma_period": int(self.ACC_WMA_PERIOD),
            "acc_z_window": int(self.ACC_Z_WINDOW),
            "acc_level": float(self.ACC_LEVEL),
            "atr_period": int(self.ATR_PERIOD),
            "atr_mult": float(self.ATR_MULT),
            "activate_atr_mult": float(self.ACTIVATE_ATR_MULT),
            "emergency_sl_pct": float(self.EMERGENCY_SL_PCT),
        }

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        params["__indicators_used"] = self.get_indicators_used()

        z_window = int(params.get("z_window", self.Z_WINDOW))
        z_setup = float(params.get("z_setup", self.Z_SETUP))

        acc_wma_period = int(params.get("acc_wma_period", self.ACC_WMA_PERIOD))
        acc_z_window = int(params.get("acc_z_window", self.ACC_Z_WINDOW))
        acc_level = float(params.get("acc_level", self.ACC_LEVEL))

        atr_period = int(params.get("atr_period", self.ATR_PERIOD))

        params["__warmup_bars"] = max(max(z_window, acc_wma_period, acc_z_window, atr_period) + 20, 120)

        ind_config = {
            "zscore": cfg_zscore(col="close", window=z_window, out="zscore"),
            "accel_sg": cfg_accel_sg(
                wma_period=acc_wma_period,
                z_window=acc_z_window,
                col="close",
                out="accel_sg",
            ),
            "atr": cfg_atr(period=atr_period, out="atr"),
        }
        df = IndicadorFactory.procesar(df, ind_config)

        z = pl.col("zscore")
        acc = pl.col("accel_sg")

        # Setup
        setup_long = z < (-z_setup)
        setup_short = z > z_setup

        # Trigger crosses
        cross_up_minus = (acc.shift(1) < (-acc_level)) & (acc > (-acc_level))
        cross_down_plus = (acc.shift(1) > acc_level) & (acc < acc_level)

        signal_long = setup_long & cross_up_minus
        signal_short = setup_short & cross_down_plus

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

        atr_mult = float(params.get("atr_mult", self.ATR_MULT))
        activate_mult = float(params.get("activate_atr_mult", self.ACTIVATE_ATR_MULT))
        sl_pct = float(params.get("emergency_sl_pct", self.EMERGENCY_SL_PCT))

        close_arr = df["close"].to_numpy().astype(np.float64)
        high_arr = df["high"].to_numpy().astype(np.float64)
        low_arr = df["low"].to_numpy().astype(np.float64)
        atr_arr = df["atr"].to_numpy().astype(np.float64)

        max_bars = min(self.TIMEOUT_BARS, n - entry_idx - 1)
        exit_idx, code = _exit_logic_numba(
            entry_idx,
            float(entry_price),
            close_arr,
            high_arr,
            low_arr,
            atr_arr,
            float(sl_pct) / 100.0,
            float(activate_mult),
            float(atr_mult),
            1 if is_long else -1,
            int(max_bars),
        )

        if exit_idx < 0:
            return None

        if code == 1:
            return ExitDecision(exit_idx=int(exit_idx), reason="EMERGENCY_SL")
        if code == 2:
            return ExitDecision(exit_idx=int(exit_idx), reason="TRAILING_ATR")
        return ExitDecision(exit_idx=int(exit_idx), reason="TIME_EXIT")


@njit(cache=True, fastmath=True)
def _exit_logic_numba(
    entry_idx: int,
    entry_price: float,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    sl_frac: float,
    activate_mult: float,
    trail_mult: float,
    side_flag: int,
    max_bars: int,
) -> tuple[int, int]:
    """Exit logic:
    - Emergency SL (static)
    - ATR trailing activated after move >= activate_mult * ATR

    code: 1=SL, 2=Trailing, 3=Time
    """

    n = len(close)
    last_allowed = min(n - 1, entry_idx + max_bars)

    if side_flag == 1:
        sl_price = entry_price * (1.0 - sl_frac)
        best = entry_price
        trailing_active = False
        stop = 0.0

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
            if lo <= sl_price:
                return i, 1

            if hi > best:
                best = hi

            if not trailing_active:
                if (best - entry_price) >= activate_mult * a:
                    trailing_active = True
                    stop = best - trail_mult * a
            else:
                sb = best - trail_mult * a
                if sb > stop:
                    stop = sb
                if lo <= stop:
                    return i, 2

        return last_allowed, 3

    else:
        sl_price = entry_price * (1.0 + sl_frac)
        best = entry_price
        trailing_active = False
        stop = 0.0

        for i in range(entry_idx + 1, last_allowed + 1):
            hi = high[i]
            lo = low[i]
            cl = close[i]
            a = atr[i]
            if np.isnan(hi) or np.isnan(lo) or np.isnan(cl):
                continue
            if np.isnan(a):
                a = 0.0

            if hi >= sl_price:
                return i, 1

            if lo < best:
                best = lo

            if not trailing_active:
                if (entry_price - best) >= activate_mult * a:
                    trailing_active = True
                    stop = best + trail_mult * a
            else:
                sb = best + trail_mult * a
                if sb < stop:
                    stop = sb
                if hi >= stop:
                    return i, 2

        return last_allowed, 3
