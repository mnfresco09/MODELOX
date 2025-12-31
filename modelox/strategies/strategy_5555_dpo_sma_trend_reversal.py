from __future__ import annotations

"""Strategy 5555 — DPO_SMA_Trend_Reversal

Objetivo:
- Detectar giros de tendencia usando pivote (V / V invertida) en SMA.
- Confirmar el giro con N velas consecutivas.
- Validar la fuerza del giro con aceleración relativa del DPO.
- Gestionar salida con SL de emergencia + trailing ATR.
- Aplicar cooldown tras salida usando el motor (block_velas_after_exit).

ENTRADA LONG (secuencia estricta):
1) Giro SMA (V) en k: SMA[k-2] > SMA[k-1] < SMA[k]
2) Confirmación: SMA sube consecutivamente durante n_confirm velas desde k
3) Impulso DPO al finalizar confirmación (t):
   (DPO[t] - DPO[t-lookback]) / abs(DPO[t-lookback]) >= dpo_inc_pct

ENTRADA SHORT:
1) Giro SMA (V invertida) en k: SMA[k-2] < SMA[k-1] > SMA[k]
2) Confirmación: SMA baja consecutivamente durante n_confirm velas desde k
3) Impulso DPO al finalizar confirmación (t):
   (DPO[t] - DPO[t-lookback]) / abs(DPO[t-lookback]) <= -dpo_inc_pct

NOTA UNIDADES:
- dpo_inc_pct se interpreta como porcentaje (ej. 1.5 => 1.5%).
"""

from typing import Any, Dict, Optional

import numpy as np
import polars as pl
from numba import njit

from logic.indicators import IndicadorFactory
from modelox.core.types import ExitDecision
from modelox.strategies.indicator_specs import cfg_atr, cfg_dpo, cfg_sma


@njit(cache=True, fastmath=True)
def _exit_atr_trail_numba(
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
    - Trailing ATR (ratchet)

    code: 1=EMERGENCY_SL, 2=TRAIL_ATR, 3=TIME_EXIT
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


class Strategy5555DPOSMATrendReversal:
    combinacion_id = 5555
    name = "DPO_SMA_Trend_Reversal"

    __indicators_used = ["sma", "dpo", "atr"]

    @classmethod
    def get_indicators_used(cls):
        return cls.__indicators_used

    parametros_optuna = {
        "sma_period": (40, 150, 5),
        "n_confirm": (2, 4, 1),
        "dpo_period": (18, 30, 1),
        "dpo_lookback": (5, 10, 1),
        "dpo_inc_pct": (0.75, 3.0, 0.25),
        "n_cooldown": (6, 30, 2),
        # Exits
        "atr_period": (10, 20, 2),
        "atr_trailing_mult": (1.0, 5.0, 0.25),
        "emergency_sl_pct": (3.0, 20.0, 1.0),
    }

    TIMEOUT_BARS = 320

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        return {
            "sma_period": int(trial.suggest_int("sma_period", 40, 150, step=5)),
            "n_confirm": int(trial.suggest_int("n_confirm", 2, 4, step=1)),
            "dpo_period": int(trial.suggest_int("dpo_period", 18, 30, step=1)),
            "dpo_lookback": int(trial.suggest_int("dpo_lookback", 5, 10, step=1)),
            "dpo_inc_pct": float(trial.suggest_float("dpo_inc_pct", 0.75, 3.0, step=0.25)),
            "n_cooldown": int(trial.suggest_int("n_cooldown", 6, 30, step=2)),
            "atr_period": int(trial.suggest_int("atr_period", 10, 20, step=2)),
            "atr_trailing_mult": float(trial.suggest_float("atr_trailing_mult", 1.0, 5.0, step=0.25)),
            "emergency_sl_pct": float(trial.suggest_float("emergency_sl_pct", 3.0, 20.0, step=1.0)),
        }

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        params["__indicators_used"] = self.get_indicators_used()

        sma_period = int(params.get("sma_period", 80))
        n_confirm = int(params.get("n_confirm", 3))
        dpo_period = int(params.get("dpo_period", 20))
        dpo_lookback = int(params.get("dpo_lookback", 7))
        dpo_inc_pct = float(params.get("dpo_inc_pct", 1.5))
        n_cooldown = int(params.get("n_cooldown", 12))
        atr_period = int(params.get("atr_period", 14))

        # Cooldown is enforced by the engine in generate_trades
        params["block_velas_after_exit"] = int(n_cooldown)

        # Warmup: SMA + DPO + ATR + lookback + confirmation window
        params["__warmup_bars"] = max(sma_period, dpo_period, atr_period) + max(dpo_lookback, 10) + (n_confirm + 3)

        ind_config = {
            "sma": cfg_sma(period=sma_period, col="close", out="sma"),
            "dpo": cfg_dpo(period=dpo_period, col="close", out="dpo"),
            "atr": cfg_atr(period=atr_period, out="atr"),
        }
        df = IndicadorFactory.procesar(df, ind_config)

        sma = pl.col("sma")
        dpo = pl.col("dpo")

        # 1) Pivot detection at k
        v_long_k = (sma.shift(2) > sma.shift(1)) & (sma > sma.shift(1))
        v_short_k = (sma.shift(2) < sma.shift(1)) & (sma < sma.shift(1))

        # 2) Confirmation from k (strictly monotonic for n_confirm bars)
        conf_up_parts = []
        conf_down_parts = []
        for j in range(max(1, n_confirm) - 1):
            conf_up_parts.append(sma.shift(-(j + 1)) > sma.shift(-j))
            conf_down_parts.append(sma.shift(-(j + 1)) < sma.shift(-j))

        conf_up = pl.lit(True)
        for expr in conf_up_parts:
            conf_up = conf_up & expr

        conf_down = pl.lit(True)
        for expr in conf_down_parts:
            conf_down = conf_down & expr

        # entry index is t = k + (n_confirm - 1)
        entry_base_long = (v_long_k & conf_up).shift(n_confirm - 1)
        entry_base_short = (v_short_k & conf_down).shift(n_confirm - 1)

        # 3) DPO impulse at t: relative change vs lookback
        dpo_past = dpo.shift(dpo_lookback)
        denom = dpo_past.abs() + 1e-12
        rel_change = (dpo - dpo_past) / denom

        thr = (dpo_inc_pct / 100.0)
        impulse_long = rel_change >= thr
        impulse_short = rel_change <= (-thr)

        signal_long = entry_base_long & impulse_long
        signal_short = entry_base_short & impulse_short

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

        if "atr" not in df.columns:
            return ExitDecision(exit_idx=min(entry_idx + 1, n - 1), reason="NO_ATR")

        atr_mult = float(params.get("atr_trailing_mult", 2.5))
        sl_pct = float(params.get("emergency_sl_pct", 10.0))

        close_arr = df["close"].to_numpy().astype(np.float64)
        atr_arr = df["atr"].to_numpy().astype(np.float64)

        max_bars = min(self.TIMEOUT_BARS, n - entry_idx - 1)
        exit_idx, code = _exit_atr_trail_numba(
            int(entry_idx),
            float(entry_price),
            close_arr,
            atr_arr,
            float(sl_pct) / 100.0,
            float(atr_mult),
            1 if is_long else -1,
            int(max_bars),
        )

        if code == 1:
            return ExitDecision(exit_idx=int(exit_idx), reason="EMERGENCY_SL")
        if code == 2:
            return ExitDecision(exit_idx=int(exit_idx), reason="TRAILING_ATR")
        return ExitDecision(exit_idx=int(exit_idx), reason="TIME_EXIT")
