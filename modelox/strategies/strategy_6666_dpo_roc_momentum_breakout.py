from __future__ import annotations

"""Strategy 6666 — DPO_ROC_Momentum_Breakout

Estrategia Momentum / Ruptura de Volatilidad.

Indicadores
- DPO: desviación de tendencia (en unidades de precio)
- ROC: velocidad del cambio del precio (%)
- ATR: para normalizar DPO y para trailing

Nota importante sobre escalas (adaptación de DPO):
- Nuestro DPO es un diferencial en unidades de precio: DPO = close - SMA_shifted.
- Para que los umbrales sean robustos entre activos (BTC vs GOLD vs SPX),
  usamos DPO normalizado por ATR: dpo_n = dpo / atr.
  Así, umbrales como 0.8–3.0 significan “desviación de 0.8–3.0 ATR”.

Entradas
LONG:
- dpo_n > umbral_dpo
- roc > umbral_roc

SHORT:
- dpo_n < -umbral_dpo
- roc < -umbral_roc

Salidas
A) Take Profit por PnL: salir si PnL neto >= tp_pnl
B) Stop Loss por PnL: salir si PnL neto <= -sl_pnl
C) Protección: si en algún momento PnL neto >= lock_trigger_pnl, mover el stop a PnL neto = lock_stop_pnl

"""

from typing import Any, Dict, Optional

import numpy as np
import polars as pl
from numba import njit

from logic.indicators import IndicadorFactory
from modelox.core.types import ExitDecision
from modelox.strategies.indicator_specs import cfg_atr, cfg_dpo, cfg_roc


@njit(cache=True, fastmath=True)
def _exit_logic_numba(
    entry_idx: int,
    entry_price: float,
    close: np.ndarray,
    qty: float,
    comision_pct: float,
    comision_sides: int,
    tp_pnl: float,
    sl_pnl: float,
    lock_trigger_pnl: float,
    lock_stop_pnl: float,
    side_flag: int,
    max_bars: int,
) -> tuple[int, int]:
    """Exit logic PnL-based.

    Net PnL is estimated as if exiting at close[i], including commissions.

    code: 1=TP_PNL, 2=SL_PNL, 3=LOCK_SL_PNL, 4=TIME_EXIT
    """

    n = len(close)
    last_allowed = min(n - 1, entry_idx + max_bars)

    lock_active = False

    # Entry commission follows simulate_trades() convention:
    # - If comision_sides >= 2: charge entry commission
    # - Exit commission is always charged
    c_ent = (qty * entry_price * comision_pct) if comision_sides >= 2 else 0.0

    for i in range(entry_idx + 1, last_allowed + 1):
        c = close[i]
        if np.isnan(c):
            continue

        # Gross PnL in quote currency
        bruto = (c - entry_price) * qty if side_flag == 1 else (entry_price - c) * qty

        # Exit commission at current price
        c_ext = qty * c * comision_pct
        neto = bruto - c_ent - c_ext

        # Take Profit
        if neto >= tp_pnl:
            return i, 1

        # Activate lock once profit threshold touched
        if (not lock_active) and neto >= lock_trigger_pnl:
            lock_active = True

        # Locked stop (after trigger)
        if lock_active and neto <= lock_stop_pnl:
            return i, 3

        # Hard stop loss
        if neto <= -sl_pnl:
            return i, 2

    return last_allowed, 4


class Strategy6666DPOROCMomentumBreakout:
    combinacion_id = 6666
    name = "DPO_ROC_Momentum_Breakout"

    __indicators_used = ["dpo", "roc", "atr"]

    @classmethod
    def get_indicators_used(cls):
        return cls.__indicators_used

    parametros_optuna = {
        "dpo_period": (10, 30, 1),
        "roc_period": (5, 21, 1),
        # Entry intensity thresholds
        # NOTE: symmetric thresholds for long/short entries.
        "umbral_dpo": (0.5, 3.0, 0.1),  # en ATRs
        "umbral_roc": (0.1, 2.5, 0.1),  # %
        # Exits (PnL neto, en moneda de la cuenta)
        "tp_pnl": (10.0, 60.0, 1.0),
        "sl_pnl": (5.0, 40.0, 1.0),
        "lock_trigger_pnl": (1.0, 30.0, 0.5),
        "lock_stop_pnl": (0.0, 20.0, 0.5),
        # ATR is still used for DPO normalization in entries.
        "atr_period": (14, 14, 1),
    }

    TIMEOUT_BARS = 320

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        # Exit params with ordering constraints:
        # - tp_pnl must be > lock_trigger_pnl
        # - lock_trigger_pnl must be > lock_stop_pnl
        # - sl_pnl is a positive loss magnitude
        tp_pnl = float(trial.suggest_float("tp_pnl", 10.0, 60.0, step=1.0))
        sl_pnl = float(trial.suggest_float("sl_pnl", 5.0, 40.0, step=1.0))

        lock_trigger_hi = min(30.0, tp_pnl - 1.0)
        if lock_trigger_hi < 1.0:
            lock_trigger_hi = 1.0
        lock_trigger_pnl = float(
            trial.suggest_float("lock_trigger_pnl", 1.0, lock_trigger_hi, step=0.5)
        )

        lock_stop_hi = min(20.0, lock_trigger_pnl - 0.5)
        if lock_stop_hi < 0.0:
            lock_stop_hi = 0.0
        lock_stop_pnl = float(
            trial.suggest_float("lock_stop_pnl", 0.0, lock_stop_hi, step=0.5)
        )

        return {
            "dpo_period": int(trial.suggest_int("dpo_period", 10, 30, step=1)),
            "roc_period": int(trial.suggest_int("roc_period", 5, 21, step=1)),
            # Symmetric entries: one threshold used for both sides (sign-flipped).
            "umbral_dpo": float(trial.suggest_float("umbral_dpo", 0.5, 3.0, step=0.1)),
            "umbral_roc": float(trial.suggest_float("umbral_roc", 0.1, 2.5, step=0.1)),
            # Exits
            "tp_pnl": tp_pnl,
            "sl_pnl": sl_pnl,
            "lock_trigger_pnl": lock_trigger_pnl,
            "lock_stop_pnl": lock_stop_pnl,
            "atr_period": 14,
        }

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        params["__indicators_used"] = self.get_indicators_used()

        dpo_period = int(params.get("dpo_period", 20))
        roc_period = int(params.get("roc_period", 12))
        atr_period = int(params.get("atr_period", 14))

        # Entry thresholds are symmetric by requirement.
        # Backwards-compatible: if older param names exist, we still accept them.
        umbral_dpo = float(
            params.get(
                "umbral_dpo",
                params.get("umbral_long_dpo", params.get("umbral_short_dpo", 1.2)),
            )
        )
        umbral_roc = float(
            params.get(
                "umbral_roc",
                params.get("umbral_long_roc", params.get("umbral_short_roc", 0.6)),
            )
        )

        # Warmup: DPO uses SMA(period) + shift, plus ATR/ROC
        params["__warmup_bars"] = max(dpo_period + (dpo_period // 2 + 1), atr_period, roc_period) + 10

        ind_config = {
            "dpo": cfg_dpo(period=dpo_period, col="close", out="dpo"),
            "roc": cfg_roc(period=roc_period, col="close", out="roc"),
            "atr": cfg_atr(period=atr_period, out="atr"),
        }
        df = IndicadorFactory.procesar(df, ind_config)

        dpo = pl.col("dpo")
        roc = pl.col("roc")
        atr = pl.col("atr")

        dpo_n = dpo / (atr + 1e-12)

        signal_long = (dpo_n > umbral_dpo) & (roc > umbral_roc)
        signal_short = (dpo_n < (-umbral_dpo)) & (roc < (-umbral_roc))

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

        # PnL-based exits (net of commissions; sizing aligned with engine).
        tp_pnl = float(params.get("tp_pnl", 20.0))
        sl_pnl = float(params.get("sl_pnl", 10.0))
        lock_trigger_pnl = float(params.get("lock_trigger_pnl", 5.0))
        lock_stop_pnl = float(params.get("lock_stop_pnl", 2.0))

        # Sizing inputs injected by runner (see modelox/core/runner.py)
        saldo_operativo_max = float(params.get("__saldo_operativo_max", saldo_apertura))
        apalancamiento = float(params.get("__apalancamiento", 1.0))
        qty_max_activo = float(params.get("__qty_max_activo", 1e308))
        comision_pct = float(params.get("__comision_pct", 0.0))
        comision_sides = int(params.get("__comision_sides", 2))

        stake = float(saldo_apertura)
        if saldo_operativo_max > 0:
            stake = min(stake, saldo_operativo_max)

        # qty per simulate_trades() convention
        qty = (stake * apalancamiento) / float(entry_price) if entry_price > 0 else 0.0
        if qty > qty_max_activo:
            qty = float(qty_max_activo)

        close_arr = df["close"].to_numpy().astype(np.float64)

        max_bars = min(self.TIMEOUT_BARS, n - entry_idx - 1)
        exit_idx, code = _exit_logic_numba(
            int(entry_idx),
            float(entry_price),
            close_arr,
            float(qty),
            float(comision_pct),
            int(comision_sides),
            float(tp_pnl),
            float(sl_pnl),
            float(lock_trigger_pnl),
            float(lock_stop_pnl),
            1 if is_long else -1,
            int(max_bars),
        )

        if code == 1:
            return ExitDecision(exit_idx=int(exit_idx), reason="TP_PNL")
        if code == 2:
            return ExitDecision(exit_idx=int(exit_idx), reason="SL_PNL")
        if code == 3:
            return ExitDecision(exit_idx=int(exit_idx), reason="LOCK_SL_PNL")
        return ExitDecision(exit_idx=int(exit_idx), reason="TIME_EXIT")
