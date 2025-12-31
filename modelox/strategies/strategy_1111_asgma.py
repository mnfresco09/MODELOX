from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import polars as pl
from numba import njit

from logic.indicators import IndicadorFactory
from modelox.core.types import ExitDecision
from modelox.strategies.indicator_specs import cfg_asgma, cfg_atr, cfg_chande_mo, cfg_rsi


@njit(cache=True, fastmath=True)
def _exit_logic_numba(
    entry_idx: int,
    entry_price: float,
    close: np.ndarray,
    atr: np.ndarray,
    sl_frac: float,
    trail_mult: float,
    side_flag: int,
    max_bars: int,
) -> tuple[int, int]:
    """Exit logic:
    - Emergency SL (static)
    - ATR trailing (ratchet)

    code: 1=SL, 2=TRAIL, 3=TIME
    """
    n = len(close)
    last_allowed = min(n - 1, entry_idx + max_bars)

    if side_flag == 1:
        sl_price = entry_price * (1.0 - sl_frac)
        trail = -1e308
        for i in range(entry_idx + 1, last_allowed + 1):
            c = close[i]
            a = atr[i]
            if np.isnan(c) or np.isnan(a) or a <= 0:
                continue
            trail = max(trail, c - trail_mult * a)
            stop = max(sl_price, trail)
            if c <= stop:
                return i, 1 if c <= sl_price else 2
        return last_allowed, 3

    # short
    sl_price = entry_price * (1.0 + sl_frac)
    trail = 1e308
    for i in range(entry_idx + 1, last_allowed + 1):
        c = close[i]
        a = atr[i]
        if np.isnan(c) or np.isnan(a) or a <= 0:
            continue
        trail = min(trail, c + trail_mult * a)
        stop = min(sl_price, trail)
        if c >= stop:
            return i, 1 if c >= sl_price else 2
    return last_allowed, 3


class Strategy1111ASGMA:
    combinacion_id = 1111
    name = "ASGMA_Crossover_ATR"

    parametros_optuna = {
        "smooth": (1, 5, 1),
        "alma_len": (10, 60, 1),
        "alma_offset": (0.5, 0.95, 0.05),
        "alma_sigma": (2.0, 10.0, 1.0),
        "gma_len": (5, 40, 1),
        "volatility_period": (10, 60, 1),
        "sigma_fixed": (0.5, 3.0, 0.5),
        "gma_ema": (3, 20, 1),
        "rsi_period": (5, 30, 1),
        "chande_len": (5, 30, 1),
        "atr_period": (7, 30, 1),
        "atr_mult": (1.0, 6.0, 0.5),
        "emergency_sl_pct": (0.5, 5.0, 0.5),
    }

    __indicators_used = ["asgma_avpchange", "asgma_gma", "rsi", "chande_mo", "atr"]

    @classmethod
    def get_indicators_used(cls):
        return cls.__indicators_used

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        return {
            # ASGMA
            "smooth": trial.suggest_int("smooth", 1, 5),
            "alma_len": trial.suggest_int("alma_len", 10, 60),
            "alma_offset": trial.suggest_float("alma_offset", 0.5, 0.95),
            "alma_sigma": trial.suggest_float("alma_sigma", 2.0, 10.0),
            "gma_len": trial.suggest_int("gma_len", 5, 40),
            "adaptive": trial.suggest_categorical("adaptive", [True, False]),
            "volatility_period": trial.suggest_int("volatility_period", 10, 60),
            "sigma_fixed": trial.suggest_float("sigma_fixed", 0.5, 3.0),
            "gma_ema": trial.suggest_int("gma_ema", 3, 20),
            # Pine-filter helpers (computed for analysis/plot)
            "rsi_period": trial.suggest_int("rsi_period", 5, 30),
            "chande_len": trial.suggest_int("chande_len", 5, 30),
            # Exits
            "atr_period": trial.suggest_int("atr_period", 7, 30),
            "atr_mult": trial.suggest_float("atr_mult", 1.0, 6.0),
            "emergency_sl_pct": trial.suggest_float("emergency_sl_pct", 0.5, 5.0),
        }

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        params["__indicators_used"] = self.get_indicators_used()

        smooth = int(params.get("smooth", 1))
        alma_len = int(params.get("alma_len", 25))
        alma_offset = float(params.get("alma_offset", 0.85))
        alma_sigma = float(params.get("alma_sigma", 7.0))
        gma_len = int(params.get("gma_len", 14))
        adaptive = bool(params.get("adaptive", True))
        volatility_period = int(params.get("volatility_period", 20))
        sigma_fixed = float(params.get("sigma_fixed", 1.0))
        gma_ema = int(params.get("gma_ema", 7))

        rsi_period = int(params.get("rsi_period", 14))
        chande_len = int(params.get("chande_len", 9))
        atr_period = int(params.get("atr_period", 14))

        params["__warmup_bars"] = max(alma_len, gma_len, volatility_period, rsi_period, chande_len, atr_period) + 50

        ind_config = {
            "asgma": cfg_asgma(
                col="close",
                smooth=smooth,
                alma_len=alma_len,
                alma_offset=alma_offset,
                alma_sigma=alma_sigma,
                gma_len=gma_len,
                adaptive=adaptive,
                volatility_period=volatility_period,
                sigma_fixed=sigma_fixed,
                gma_ema=gma_ema,
                out_avpchange="asgma_avpchange",
                out_gma="asgma_gma",
            ),
            "rsi": cfg_rsi(period=rsi_period, col="close", out="rsi"),
            "chande_mo": cfg_chande_mo(length=chande_len, col="close", out="chande_mo"),
            "atr": cfg_atr(period=atr_period, out="atr"),
        }

        df = IndicadorFactory.procesar(df, ind_config)

        avp = pl.col("asgma_avpchange")
        gma = pl.col("asgma_gma")

        # Pine:
        # buySignal  = crossover(avpchange, gma)
        # sellSignal = crossunder(avpchange, gma)
        crossover = (avp > gma) & (avp.shift(1) <= gma.shift(1))
        crossunder = (avp < gma) & (avp.shift(1) >= gma.shift(1))

        return df.with_columns(
            [
                crossover.fill_null(False).alias("signal_long"),
                crossunder.fill_null(False).alias("signal_short"),
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

        atr_mult = float(params.get("atr_mult", 3.0))
        sl_pct = float(params.get("emergency_sl_pct", 1.0))

        close_arr = df["close"].to_numpy().astype(np.float64)
        atr_arr = df["atr"].to_numpy().astype(np.float64) if "atr" in df.columns else np.full(n, np.nan)

        max_bars = min(260, n - entry_idx - 1)
        exit_idx, code = _exit_logic_numba(
            int(entry_idx),
            float(entry_price),
            close_arr,
            atr_arr,
            float(sl_pct) / 100.0,
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
