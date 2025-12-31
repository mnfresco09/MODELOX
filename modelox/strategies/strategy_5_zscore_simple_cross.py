from __future__ import annotations

"""
Strategy 5 — Z-Score Simple Cross

Estrategia 100% determinista basada en cruces de Z-Score:

LÓGICA DE ENTRADA:
  LONG:
    - Z-Score cruza +z_threshold hacia arriba
    - (z_score[t-1] <= +z_threshold) Y (z_score[t] > +z_threshold)

  SHORT:
    - Z-Score cruza -z_threshold hacia abajo
    - (z_score[t-1] >= -z_threshold) Y (z_score[t] < -z_threshold)

LÓGICA DE SALIDA:
  - Stop Loss de emergencia: fijo en % desde entrada (3%-20%)
  - Take Profit: trailing stop basado en ATR (ajuste vela a vela)

Sin confirmaciones adicionales, solo cruces puros del Z-Score.
"""

from typing import Any, Dict, Optional

import numpy as np
import polars as pl
from numba import njit

from logic.indicators import IndicadorFactory
from modelox.core.types import ExitDecision
from modelox.strategies.indicator_specs import cfg_zscore, cfg_atr


class Strategy5ZScoreSimpleCross:
    """Z-Score Simple Cross con trailing stop."""

    # Identificación única
    combinacion_id = 5
    name = "ZScore_Simple_Cross"

    # Indicadores para el plot
    __indicators_used = ["zscore", "atr"]

    @classmethod
    def get_indicators_used(cls):
        return cls.__indicators_used

    # Espacio de parámetros Optuna
    parametros_optuna = {
        # Z-Score
        "z_length": (10, 60, 5),                   # ventana Z-Score
        "z_threshold": (0.5, 2.5, 0.1),            # umbral de cruce (simétrico ±)
        # Trailing & SL
        "atr_period": (10, 20, 2),                 # período ATR
        "trailing_atr_mult": (1.0, 4.0, 0.2),      # múltiplo ATR para trailing
        "emergency_sl_pct": (3.0, 20.0, 1.0),      # SL de emergencia (%)
    }

    TIMEOUT_BARS = 200

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        z_length = trial.suggest_int("z_length", 10, 60, step=5)
        z_threshold = trial.suggest_float("z_threshold", 0.5, 2.5, step=0.1)
        atr_period = trial.suggest_int("atr_period", 10, 20, step=2)
        trailing_atr_mult = trial.suggest_float("trailing_atr_mult", 1.0, 4.0, step=0.2)
        emergency_sl_pct = trial.suggest_float("emergency_sl_pct", 3.0, 20.0, step=1.0)

        return {
            "z_length": int(z_length),
            "z_threshold": float(z_threshold),
            "atr_period": int(atr_period),
            "trailing_atr_mult": float(trailing_atr_mult),
            "emergency_sl_pct": float(emergency_sl_pct),
        }

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        # Para reporting/plot modular
        params["__indicators_used"] = self.get_indicators_used()

        z_length = int(params.get("z_length", 20))
        z_threshold = float(params.get("z_threshold", 1.0))
        atr_period = int(params.get("atr_period", 14))

        # Warm-up robusto
        max_win = max(z_length, atr_period)
        params["__warmup_bars"] = max(max_win + 5, 50)

        # 1) Calcular indicadores
        ind_config = {
            "zscore": cfg_zscore(col="close", window=z_length, out="zscore"),
            "atr": cfg_atr(period=atr_period, out="atr"),
        }
        df = IndicadorFactory.procesar(df, ind_config)

        # 2) Detectar cruces con Numba
        z_arr = df["zscore"].to_numpy().astype(np.float64)

        signal_long, signal_short = _detect_cross_signals(
            z_arr,
            float(z_threshold),
        )

        return df.with_columns([
            pl.Series("signal_long", signal_long),
            pl.Series("signal_short", signal_short),
        ])

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
        """Trailing stop + SL de emergencia."""

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

        max_bars = min(self.TIMEOUT_BARS, len(df) - entry_idx - 1)

        exit_idx, reason_code = _decide_exit_trailing_sl(
            entry_idx,
            entry_price,
            close_arr,
            high_arr,
            low_arr,
            atr_arr,
            trailing_atr_mult,
            emergency_sl_pct,
            1 if is_long else -1,
            max_bars,
        )

        reason_map = {
            0: None,
            1: "TRAILING_STOP",
            2: "EMERGENCY_SL",
            3: "TIME_EXIT",
        }

        if exit_idx >= 0 and reason_code in reason_map and reason_map[reason_code] is not None:
            return ExitDecision(exit_idx=exit_idx, reason=reason_map[reason_code])

        fallback_idx = min(entry_idx + max_bars, n - 1)
        return ExitDecision(exit_idx=fallback_idx, reason="TIME_EXIT")


@njit(cache=True, fastmath=True)
def _detect_cross_signals(
    z: np.ndarray,
    z_threshold: float,
) -> tuple:
    """Detecta cruces simples del Z-Score.

    LONG: z cruza +z_threshold hacia arriba
      (z[t-1] <= +z_threshold) AND (z[t] > +z_threshold)

    SHORT: z cruza -z_threshold hacia abajo
      (z[t-1] >= -z_threshold) AND (z[t] < -z_threshold)
    """

    n = len(z)
    signal_long = np.zeros(n, dtype=np.bool_)
    signal_short = np.zeros(n, dtype=np.bool_)

    for i in range(1, n):
        z_prev = z[i - 1]
        z_curr = z[i]

        if np.isnan(z_prev) or np.isnan(z_curr):
            continue

        # LONG: cruce hacia arriba del umbral positivo
        if z_prev <= z_threshold and z_curr > z_threshold:
            signal_long[i] = True

        # SHORT: cruce hacia abajo del umbral negativo
        if z_prev >= -z_threshold and z_curr < -z_threshold:
            signal_short[i] = True

    return signal_long, signal_short


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
) -> tuple:
    """Trailing stop basado en ATR + SL de emergencia."""

    n = len(close)
    last_allowed = min(n - 1, entry_idx + max_bars)

    # SL de emergencia
    emergency_factor = emergency_sl_pct / 100.0
    if side_flag == 1:
        emergency_level = entry_price * (1.0 - emergency_factor)
    else:
        emergency_level = entry_price * (1.0 + emergency_factor)

    # Inicializar trailing stop
    if side_flag == 1:
        best_price = entry_price
        trailing_stop_level = entry_price - trailing_atr_mult * atr[entry_idx] if not np.isnan(atr[entry_idx]) else entry_price * 0.95
    else:
        best_price = entry_price
        trailing_stop_level = entry_price + trailing_atr_mult * atr[entry_idx] if not np.isnan(atr[entry_idx]) else entry_price * 1.05

    for i in range(entry_idx + 1, last_allowed + 1):
        c = close[i]
        h = high[i]
        l = low[i]
        atr_val = atr[i]

        if np.isnan(c) or np.isnan(atr_val):
            continue

        # 1) Emergency SL
        if side_flag == 1:
            if l <= emergency_level:
                return i, 2
        else:
            if h >= emergency_level:
                return i, 2

        # 2) Actualizar trailing stop vela a vela
        if side_flag == 1:
            if h > best_price:
                best_price = h
                new_trailing = best_price - trailing_atr_mult * atr_val
                if new_trailing > trailing_stop_level:
                    trailing_stop_level = new_trailing
        else:
            if l < best_price:
                best_price = l
                new_trailing = best_price + trailing_atr_mult * atr_val
                if new_trailing < trailing_stop_level:
                    trailing_stop_level = new_trailing

        # 3) Trailing stop
        if side_flag == 1:
            if l <= trailing_stop_level:
                return i, 1
        else:
            if h >= trailing_stop_level:
                return i, 1

    # Timeout
    if last_allowed > entry_idx:
        return last_allowed, 3

    return -1, 0
