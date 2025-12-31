from __future__ import annotations

"""Strategy 997 — MFI Accel Event + ZScore Hook (Priority: Evento → Contexto)

ENTRADAS (evaluación estricta por pasos; vectorizado equivale a AND encadenado):

LONG
  Paso 1 (Trigger): Acc cruza 0 hacia arriba
    (Acc[t-1] < 0) AND (Acc[t] > 0)
  Paso 2 (Zona): Z[t] < -Umbral_Z
  Paso 3 (Gancho): Z[t] > Z[t-1]

SHORT
  Paso 1 (Trigger): Acc cruza 0 hacia abajo
    (Acc[t-1] > 0) AND (Acc[t] < 0)
  Paso 2 (Zona): Z[t] > Umbral_Z
  Paso 3 (Gancho): Z[t] < Z[t-1]

SALIDAS (se ejecuta la primera que ocurra):
A) SL de Emergencia (estático desde entrada):
   LONG: entry*(1-SL%)  |  SHORT: entry*(1+SL%)
B) Trailing Stop Ratchet (dinámico, sin retroceso):
   StopBase:
     LONG: close[t] - ATR[t]*mult
     SHORT: close[t] + ATR[t]*mult
   Ratchet:
     LONG: stop[t] = max(stop[t-1], stopBase[t])
     SHORT: stop[t] = min(stop[t-1], stopBase[t])

Nota: El motor usa close[exit_idx] como precio de salida.
"""

from typing import Any, Dict, Optional

import numpy as np
import polars as pl
from numba import njit

from logic.indicators import IndicadorFactory
from modelox.core.types import ExitDecision
from modelox.strategies.indicator_specs import cfg_atr, cfg_mfi, cfg_mfi_d2, cfg_zscore


class Strategy997MFIAccelZScore:
    combinacion_id = 997
    name = "MFIAccel_ZScore_EventHook"

    __indicators_used = ["zscore", "mfi_d2", "atr"]

    @classmethod
    def get_indicators_used(cls):
        return cls.__indicators_used

    # Rangos Optuna solicitados
    parametros_optuna = {
        "Periodo_MFI": (8, 30, 2),
        "Periodo_ZScore": (10, 50, 2),
        "Umbral_Z": (1.5, 2.5, 0.1),
        "SL_Emergencia_%": (5.0, 20.0, 1.0),
        # Config_ATR (según spec): periodo fijo 14, mult por defecto 2.0
        # Se deja como parámetro fijo (no optimizable) para reproducibilidad.
    }

    ATR_PERIOD = 14
    ATR_MULT_DEFAULT = 2.0
    TIMEOUT_BARS = 240

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        periodo_mfi = trial.suggest_int("Periodo_MFI", 8, 30, step=2)
        periodo_z = trial.suggest_int("Periodo_ZScore", 10, 50, step=2)
        umbral_z = trial.suggest_float("Umbral_Z", 1.5, 2.5, step=0.1)
        sl_pct = trial.suggest_float("SL_Emergencia_%", 5.0, 20.0, step=1.0)

        return {
            "Periodo_MFI": int(periodo_mfi),
            "Periodo_ZScore": int(periodo_z),
            "Umbral_Z": float(umbral_z),
            "SL_Emergencia_%": float(sl_pct),
            "ATR_period": int(self.ATR_PERIOD),
            "ATR_mult": float(self.ATR_MULT_DEFAULT),
        }

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        params["__indicators_used"] = self.get_indicators_used()

        periodo_mfi = int(params.get("Periodo_MFI", 14))
        periodo_z = int(params.get("Periodo_ZScore", 20))
        umbral_z = float(params.get("Umbral_Z", 2.0))

        atr_period = int(params.get("ATR_period", self.ATR_PERIOD))
        atr_mult = float(params.get("ATR_mult", self.ATR_MULT_DEFAULT))
        params["ATR_period"] = atr_period
        params["ATR_mult"] = atr_mult

        # Warm-up robusto (MFI + Z + ATR + derivada)
        max_win = max(periodo_mfi, periodo_z, atr_period)
        params["__warmup_bars"] = max(max_win + 10, 80)

        ind_config = {
            "atr": cfg_atr(period=atr_period, out="atr"),
            "mfi": cfg_mfi(period=periodo_mfi, out="mfi"),
            "mfi_d2": cfg_mfi_d2(mfi_col="mfi", out="mfi_d2"),
            "zscore": cfg_zscore(col="close", window=periodo_z, out="zscore"),
        }
        df = IndicadorFactory.procesar(df, ind_config)

        z = pl.col("zscore")
        acc = pl.col("mfi_d2")

        # Paso 1: Trigger (evento)
        trigger_long = (acc.shift(1) < 0) & (acc > 0)
        trigger_short = (acc.shift(1) > 0) & (acc < 0)

        # Paso 2: Zona (contexto)
        zone_long = z < (-1.0 * umbral_z)
        zone_short = z > umbral_z

        # Paso 3: Gancho (hook)
        hook_long = z > z.shift(1)
        hook_short = z < z.shift(1)

        # Evaluación estricta por orden == AND encadenado
        signal_long = trigger_long & zone_long & hook_long
        signal_short = trigger_short & zone_short & hook_short

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

        sl_pct = float(params.get("SL_Emergencia_%", 10.0))
        atr_mult = float(params.get("ATR_mult", self.ATR_MULT_DEFAULT))

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
            float(atr_mult),
            1 if is_long else -1,
            int(max_bars),
        )

        if exit_idx < 0:
            return None

        # code: 1=SL, 2=Trailing, 3=Time
        if code == 1:
            return ExitDecision(exit_idx=int(exit_idx), reason="EMERGENCY_SL")
        if code == 2:
            return ExitDecision(exit_idx=int(exit_idx), reason="TRAILING_RATCHET")
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
    atr_mult: float,
    side_flag: int,
    max_bars: int,
) -> tuple[int, int]:
    """Salida por SL estático + trailing ATR ratchet.

    Returns:
        (exit_idx, code)
        code: 1=SL, 2=Trailing, 3=Time
    """

    n = len(close)
    last_allowed = min(n - 1, entry_idx + max_bars)

    if side_flag == 1:
        sl_price = entry_price * (1.0 - sl_frac)
        a0 = atr[entry_idx]
        if np.isnan(a0):
            a0 = 0.0
        stop = close[entry_idx] - a0 * atr_mult

        for i in range(entry_idx + 1, last_allowed + 1):
            lo = low[i]
            hi = high[i]
            cl = close[i]
            a = atr[i]
            if np.isnan(lo) or np.isnan(hi) or np.isnan(cl):
                continue
            if np.isnan(a):
                a = 0.0

            # A) Emergency SL (primero)
            if lo <= sl_price:
                return i, 1

            # B) Trailing ratchet
            sb = cl - a * atr_mult
            if sb > stop:
                stop = sb
            if lo <= stop:
                return i, 2

        return last_allowed, 3

    else:
        sl_price = entry_price * (1.0 + sl_frac)
        a0 = atr[entry_idx]
        if np.isnan(a0):
            a0 = 0.0
        stop = close[entry_idx] + a0 * atr_mult

        for i in range(entry_idx + 1, last_allowed + 1):
            lo = low[i]
            hi = high[i]
            cl = close[i]
            a = atr[i]
            if np.isnan(lo) or np.isnan(hi) or np.isnan(cl):
                continue
            if np.isnan(a):
                a = 0.0

            # A) Emergency SL (primero)
            if hi >= sl_price:
                return i, 1

            # B) Trailing ratchet
            sb = cl + a * atr_mult
            if sb < stop:
                stop = sb
            if hi >= stop:
                return i, 2

        return last_allowed, 3
