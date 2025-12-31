from __future__ import annotations

"""
Strategy 4 — Z-Score + MACD Reversal

Estrategia 100% determinista basada en:
- Z-Score extremo (sobreventa/sobrecompra estadística)
- Reversión del MACD (N velas consecutivas creciendo/decreciendo)

LÓGICA DE ENTRADA:

  LONG:
    1) Z-Score < z_threshold_long (ej: -2.0, rango [-2.5, -1.5])
    2) MACD < 0 (zona negativa)
    3) MACD empieza a revertirse: crece durante N velas consecutivas (N=2-4)
    4) Cuando el MACD cumple las N velas, Z-Score debe seguir siendo < z_threshold_long

  SHORT:
    1) Z-Score > z_threshold_short (ej: +2.0, rango [+1.5, +2.5])
    2) MACD > 0 (zona positiva)
    3) MACD empieza a revertirse: decrece durante N velas consecutivas (N=2-4)
    4) Cuando el MACD cumple las N velas, Z-Score debe seguir siendo > z_threshold_short

LÓGICA DE SALIDA:
  - Stop Loss de emergencia: fijo en % desde entrada (5%-15%)
  - Take Profit: trailing stop basado en ATR (ajuste vela a vela)

Sin interpretaciones, todo determinista bar-a-bar.
"""

from typing import Any, Dict, Optional

import numpy as np
import polars as pl
from numba import njit

from logic.indicators import IndicadorFactory
from modelox.core.types import ExitDecision
from modelox.strategies.indicator_specs import cfg_zscore, cfg_macd, cfg_atr


class Strategy4ZScoreMacdReversal:
    """Z-Score + MACD Reversal con trailing stop."""

    # Identificación única
    combinacion_id = 4
    name = "ZScore_MACD_Reversal"

    # Indicadores para el plot
    __indicators_used = ["zscore", "macd", "macd_signal", "macd_hist", "atr"]

    @classmethod
    def get_indicators_used(cls):
        return cls.__indicators_used

    # Espacio de parámetros Optuna (rangos relajados y simétricos)
    parametros_optuna = {
        # Z-Score (simétrico para LONG/SHORT)
        "z_length": (10, 60, 5),                   # ventana Z-Score (más amplio)
        "z_threshold": (1.0, 2.5, 0.1),            # umbral simétrico (±)
        # MACD (rangos más amplios)
        "macd_fast": (6, 20, 2),                   # EMA rápida MACD
        "macd_slow": (18, 40, 2),                  # EMA lenta MACD
        "macd_signal": (5, 15, 1),                 # señal MACD
        # Reversión (más flexible)
        "reversal_bars": (2, 6, 1),                # velas consecutivas de reversión (2-6)
        # Trailing & SL
        "atr_period": (10, 20, 2),                 # período ATR
        "trailing_atr_mult": (1.0, 4.0, 0.2),      # múltiplo ATR para trailing (más amplio)
        "emergency_sl_pct": (3.0, 20.0, 1.0),      # SL de emergencia (más rango)
    }

    TIMEOUT_BARS = 200

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        z_length = trial.suggest_int("z_length", 10, 60, step=5)
        z_threshold = trial.suggest_float("z_threshold", 1.0, 3.0, step=0.1)
        
        macd_fast = trial.suggest_int("macd_fast", 6, 20, step=2)
        macd_slow = trial.suggest_int("macd_slow", 18, 40, step=2)
        macd_signal = trial.suggest_int("macd_signal", 5, 15, step=1)
        
        # Asegurar fast < slow con margen mayor
        if macd_fast >= macd_slow:
            macd_slow = macd_fast + 6
        
        reversal_bars = trial.suggest_int("reversal_bars", 2, 6, step=1)
        atr_period = trial.suggest_int("atr_period", 10, 20, step=2)
        trailing_atr_mult = trial.suggest_float("trailing_atr_mult", 1.0, 4.0, step=0.2)
        emergency_sl_pct = trial.suggest_float("emergency_sl_pct", 3.0, 20.0, step=1.0)

        return {
            "z_length": int(z_length),
            "z_threshold": float(z_threshold),
            "macd_fast": int(macd_fast),
            "macd_slow": int(macd_slow),
            "macd_signal": int(macd_signal),
            "reversal_bars": int(reversal_bars),
            "atr_period": int(atr_period),
            "trailing_atr_mult": float(trailing_atr_mult),
            "emergency_sl_pct": float(emergency_sl_pct),
        }

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        # Para reporting/plot modular
        params["__indicators_used"] = self.get_indicators_used()

        z_length = int(params.get("z_length", 20))
        z_threshold = float(params.get("z_threshold", 2.0))
        # Simétrico: -z_threshold para LONG, +z_threshold para SHORT
        z_thr_long = -abs(z_threshold)
        z_thr_short = abs(z_threshold)
        macd_fast = int(params.get("macd_fast", 12))
        macd_slow = int(params.get("macd_slow", 26))
        macd_signal = int(params.get("macd_signal", 9))
        reversal_bars = int(params.get("reversal_bars", 3))
        atr_period = int(params.get("atr_period", 14))

        # Warm-up robusto
        max_win = max(z_length, macd_slow, atr_period)
        params["__warmup_bars"] = max(max_win + 5, 50)

        # 1) Calcular indicadores
        ind_config = {
            "zscore": cfg_zscore(col="close", window=z_length, out="zscore"),
            "macd": cfg_macd(
                fast=macd_fast,
                slow=macd_slow,
                signal=macd_signal,
                col="close",
                out_macd="macd",
                out_signal="macd_signal",
                out_hist="macd_hist",
            ),
            "atr": cfg_atr(period=atr_period, out="atr"),
        }
        df = IndicadorFactory.procesar(df, ind_config)

        # 2) Detectar señales con kernel Numba (necesitamos arrays)
        z_arr = df["zscore"].to_numpy().astype(np.float64)
        macd_arr = df["macd"].to_numpy().astype(np.float64)
        n = len(df)

        signal_long, signal_short = _detect_reversal_signals(
            z_arr,
            macd_arr,
            float(z_thr_long),
            float(z_thr_short),
            int(reversal_bars),
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
        """Trailing stop + SL de emergencia (igual que Strategy 3)."""

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
def _detect_reversal_signals(
    z: np.ndarray,
    macd: np.ndarray,
    z_thr_long: float,
    z_thr_short: float,
    reversal_bars: int,
) -> tuple:
    """Detecta señales LONG/SHORT basadas en Z-Score extremo y reversión de MACD.

    LONG:
      - Z < z_thr_long (ej: -2.0)
      - MACD < 0
      - MACD sube durante reversal_bars velas consecutivas
      - Z sigue siendo < z_thr_long cuando se completa la reversión

    SHORT:
      - Z > z_thr_short (ej: +2.0)
      - MACD > 0
      - MACD baja durante reversal_bars velas consecutivas
      - Z sigue siendo > z_thr_short cuando se completa la reversión
    """

    n = len(z)
    signal_long = np.zeros(n, dtype=np.bool_)
    signal_short = np.zeros(n, dtype=np.bool_)

    # Necesitamos lookback para contar velas de reversión
    lookback = max(reversal_bars, 5)

    for i in range(lookback, n):
        z_t = z[i]
        macd_t = macd[i]

        if np.isnan(z_t) or np.isnan(macd_t):
            continue

        # === SEÑAL LONG ===
        # 1) Condición base: Z extremo negativo y MACD en zona negativa
        if z_t < z_thr_long and macd_t < 0.0:
            # 2) Verificar reversión: MACD sube durante reversal_bars velas
            reversal_ok = True
            for j in range(1, reversal_bars + 1):
                idx_curr = i - j + 1
                idx_prev = i - j
                if idx_prev < 0 or idx_curr < 0:
                    reversal_ok = False
                    break
                macd_curr = macd[idx_curr]
                macd_prev = macd[idx_prev]
                if np.isnan(macd_curr) or np.isnan(macd_prev):
                    reversal_ok = False
                    break
                if macd_curr <= macd_prev:  # debe subir estrictamente
                    reversal_ok = False
                    break
            
            # 3) Z debe seguir siendo extremo en esta vela (i)
            if reversal_ok and z_t < z_thr_long:
                signal_long[i] = True

        # === SEÑAL SHORT ===
        # 1) Condición base: Z extremo positivo y MACD en zona positiva
        if z_t > z_thr_short and macd_t > 0.0:
            # 2) Verificar reversión: MACD baja durante reversal_bars velas
            reversal_ok = True
            for j in range(1, reversal_bars + 1):
                idx_curr = i - j + 1
                idx_prev = i - j
                if idx_prev < 0 or idx_curr < 0:
                    reversal_ok = False
                    break
                macd_curr = macd[idx_curr]
                macd_prev = macd[idx_prev]
                if np.isnan(macd_curr) or np.isnan(macd_prev):
                    reversal_ok = False
                    break
                if macd_curr >= macd_prev:  # debe bajar estrictamente
                    reversal_ok = False
                    break
            
            # 3) Z debe seguir siendo extremo en esta vela (i)
            if reversal_ok and z_t > z_thr_short:
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
    """Trailing stop basado en ATR + SL de emergencia (reutilizado de Strategy 3)."""

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
