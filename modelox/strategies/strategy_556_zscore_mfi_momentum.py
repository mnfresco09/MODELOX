"""
═══════════════════════════════════════════════════════════════════════════════
Strategy 556: Z-Score + MFI Momentum Sync
═══════════════════════════════════════════════════════════════════════════════

CONCEPTO:
---------
Estrategia de reversión que combina Z-Score y MFI con sincronización de momentum.
Busca extremos en Z-Score y MFI que estén revirtiendo simultáneamente durante
el mismo número de velas consecutivas.

SEÑALES DE ENTRADA:
-------------------
LONG:
  1. Z-Score < -z_threshold (entre -0.5 y -1.5)
  2. Z-Score subiendo durante N velas consecutivas (2-4)
  3. MFI < mfi_threshold (típicamente 50)
  4. MFI subiendo durante N velas consecutivas (2-4)
  5. N debe ser el mismo para Z-Score y MFI

SHORT:
  1. Z-Score > +z_threshold (entre +0.5 y +1.5)
  2. Z-Score bajando durante N velas consecutivas (2-4)
  3. MFI > mfi_threshold (típicamente 50)
  4. MFI bajando durante N velas consecutivas (2-4)
  5. N debe ser el mismo para Z-Score y MFI

SALIDAS:
--------
- Trailing Stop: ATR-based, actualiza vela a vela solo favorablemente
- Stop Loss de Emergencia: Porcentaje fijo (3%-20%)

PARÁMETROS OPTIMIZABLES:
------------------------
- z_length: Período del Z-Score [10-60]
- z_threshold: Umbral del Z-Score [0.5-1.5]
- mfi_period: Período del MFI [10-30]
- mfi_threshold: Umbral del MFI [40-60]
- momentum_bars: Velas consecutivas de momentum [2-4]
- trailing_atr_mult: Multiplicador ATR para trailing [1.5-4.0]
- trailing_atr_period: Período ATR para trailing [10-20]
- emergency_sl_pct: Stop loss de emergencia [3.0-20.0]

═══════════════════════════════════════════════════════════════════════════════
"""

import polars as pl
import numpy as np
from numba import njit
from typing import Dict, Any

from modelox.strategies.indicator_specs import (
    cfg_zscore,
    cfg_mfi,
    cfg_atr,
)
from logic.indicators import IndicadorFactory


class Strategy556ZScoreMFIMomentum:
    """Strategy 556: Z-Score + MFI con sincronización de momentum."""

    combinacion_id = 556
    name = "ZScore_MFI_Momentum_Sync"
    __warmup_bars = 80
    __indicators_used = ["zscore", "mfi", "atr"]

    @staticmethod
    def parametros_optuna() -> Dict[str, Dict[str, Any]]:
        """Define rangos de optimización para Optuna."""
        return {
            "z_length": {"low": 10, "high": 60, "step": 5},
            "z_threshold": {"low": 0.5, "high": 1.5, "step": 0.1},
            "mfi_period": {"low": 10, "high": 30, "step": 2},
            "mfi_threshold": {"low": 40.0, "high": 60.0, "step": 2.0},
            "momentum_bars": {"low": 2, "high": 4, "step": 1},
            "trailing_atr_mult": {"low": 1.5, "high": 4.0, "step": 0.25},
            "trailing_atr_period": {"low": 10, "high": 20, "step": 2},
            "emergency_sl_pct": {"low": 3.0, "high": 20.0, "step": 1.0},
        }

    @staticmethod
    def suggest_params(trial) -> Dict[str, Any]:
        """Sugiere parámetros para un trial de Optuna."""
        ranges = Strategy556ZScoreMFIMomentum.parametros_optuna()
        return {
            "z_length": trial.suggest_int("z_length", **ranges["z_length"]),
            "z_threshold": trial.suggest_float("z_threshold", **ranges["z_threshold"]),
            "mfi_period": trial.suggest_int("mfi_period", **ranges["mfi_period"]),
            "mfi_threshold": trial.suggest_float("mfi_threshold", **ranges["mfi_threshold"]),
            "momentum_bars": trial.suggest_int("momentum_bars", **ranges["momentum_bars"]),
            "trailing_atr_mult": trial.suggest_float("trailing_atr_mult", **ranges["trailing_atr_mult"]),
            "trailing_atr_period": trial.suggest_int("trailing_atr_period", **ranges["trailing_atr_period"]),
            "emergency_sl_pct": trial.suggest_float("emergency_sl_pct", **ranges["emergency_sl_pct"]),
        }

    @staticmethod
    def generate_signals(df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        """
        Genera señales de entrada basadas en sincronización de momentum Z-Score/MFI.

        Returns:
            DataFrame con columnas 'signal' (1=LONG, -1=SHORT, 0=neutro)
        """
        z_length = int(params["z_length"])
        z_threshold = float(params["z_threshold"])
        mfi_period = int(params["mfi_period"])
        mfi_threshold = float(params["mfi_threshold"])
        momentum_bars = int(params["momentum_bars"])

        # Warm-up robusto
        max_win = max(z_length, mfi_period)
        params["__warmup_bars"] = max(max_win + momentum_bars + 5, 50)

        # 1) Calcular indicadores
        ind_config = {
            "zscore": cfg_zscore(col="close", window=z_length, out="zscore"),
            "mfi": cfg_mfi(period=mfi_period, out="mfi"),
        }
        df = IndicadorFactory.procesar(df, ind_config)

        # 2) Detectar señales con Numba
        z_arr = df["zscore"].to_numpy().astype(np.float64)
        mfi_arr = df["mfi"].to_numpy().astype(np.float64)

        signals = _detect_zscore_mfi_momentum_signals(
            z_arr,
            mfi_arr,
            z_threshold,
            mfi_threshold,
            momentum_bars,
        )

        df = df.with_columns(pl.Series("signal", signals, dtype=pl.Int8))
        return df

    @staticmethod
    def decide_exit(
        df: pl.DataFrame,
        signal_bar: int,
        direction: int,
        entry_price: float,
        params: Dict[str, Any],
    ) -> tuple[int, float, str]:
        """
        Decide salida mediante trailing stop ATR + stop loss de emergencia.

        Args:
            df: DataFrame completo
            signal_bar: Índice de la barra de entrada
            direction: 1 (LONG) o -1 (SHORT)
            entry_price: Precio de entrada
            params: Parámetros de la estrategia

        Returns:
            (exit_bar, exit_price, exit_reason)
        """
        trailing_atr_mult = float(params["trailing_atr_mult"])
        trailing_atr_period = int(params["trailing_atr_period"])
        emergency_sl_pct = float(params["emergency_sl_pct"])

        # Calcular ATR si no existe
        if "atr" not in df.columns:
            ind_config = {
                "atr": cfg_atr(period=trailing_atr_period, out="atr"),
            }
            df = IndicadorFactory.procesar(df, ind_config)

        close_arr = df["close"].to_numpy().astype(np.float64)
        high_arr = df["high"].to_numpy().astype(np.float64)
        low_arr = df["low"].to_numpy().astype(np.float64)
        atr_arr = df["atr"].to_numpy().astype(np.float64)

        exit_bar, exit_price, reason_code = _decide_exit_trailing_sl(
            close_arr,
            high_arr,
            low_arr,
            atr_arr,
            signal_bar,
            direction,
            entry_price,
            trailing_atr_mult,
            emergency_sl_pct / 100.0,
        )

        reason_map = {0: "trailing_stop", 1: "emergency_sl", 2: "end_of_data"}
        return exit_bar, exit_price, reason_map.get(reason_code, "unknown")


# ══════════════════════════════════════════════════════════════════════════════
# NUMBA KERNELS
# ══════════════════════════════════════════════════════════════════════════════


@njit
def _is_rising_for_n_bars(arr: np.ndarray, idx: int, n: int) -> bool:
    """
    Verifica si un indicador está subiendo durante N velas consecutivas.
    
    Args:
        arr: Array del indicador
        idx: Índice actual
        n: Número de velas a verificar
        
    Returns:
        True si arr[idx] > arr[idx-1] > arr[idx-2] > ... > arr[idx-n]
    """
    if idx < n:
        return False
    
    for i in range(n):
        if np.isnan(arr[idx - i]) or np.isnan(arr[idx - i - 1]):
            return False
        if arr[idx - i] <= arr[idx - i - 1]:
            return False
    
    return True


@njit
def _is_falling_for_n_bars(arr: np.ndarray, idx: int, n: int) -> bool:
    """
    Verifica si un indicador está bajando durante N velas consecutivas.
    
    Args:
        arr: Array del indicador
        idx: Índice actual
        n: Número de velas a verificar
        
    Returns:
        True si arr[idx] < arr[idx-1] < arr[idx-2] < ... < arr[idx-n]
    """
    if idx < n:
        return False
    
    for i in range(n):
        if np.isnan(arr[idx - i]) or np.isnan(arr[idx - i - 1]):
            return False
        if arr[idx - i] >= arr[idx - i - 1]:
            return False
    
    return True


@njit
def _detect_zscore_mfi_momentum_signals(
    z_arr: np.ndarray,
    mfi_arr: np.ndarray,
    z_threshold: float,
    mfi_threshold: float,
    momentum_bars: int,
) -> np.ndarray:
    """
    Detecta señales cuando Z-Score y MFI están en extremos y revierten
    simultáneamente durante N velas consecutivas.

    LONG:
      - Z-Score < -z_threshold
      - Z-Score subiendo durante momentum_bars velas
      - MFI < mfi_threshold
      - MFI subiendo durante momentum_bars velas

    SHORT:
      - Z-Score > +z_threshold
      - Z-Score bajando durante momentum_bars velas
      - MFI > mfi_threshold
      - MFI bajando durante momentum_bars velas

    Returns:
        Array de señales: 1 (LONG), -1 (SHORT), 0 (neutro)
    """
    n = len(z_arr)
    signals = np.zeros(n, dtype=np.int8)

    for i in range(momentum_bars, n):
        z = z_arr[i]
        mfi = mfi_arr[i]

        # Validación de datos
        if np.isnan(z) or np.isnan(mfi):
            continue

        # === LONG ===
        # 1. Z-Score en extremo inferior
        if z < -z_threshold:
            # 2. Z-Score subiendo durante momentum_bars velas
            z_rising = _is_rising_for_n_bars(z_arr, i, momentum_bars)
            
            if z_rising:
                # 3. MFI en extremo inferior
                if mfi < mfi_threshold:
                    # 4. MFI subiendo durante momentum_bars velas
                    mfi_rising = _is_rising_for_n_bars(mfi_arr, i, momentum_bars)
                    
                    if mfi_rising:
                        signals[i] = 1
                        continue

        # === SHORT ===
        # 1. Z-Score en extremo superior
        if z > z_threshold:
            # 2. Z-Score bajando durante momentum_bars velas
            z_falling = _is_falling_for_n_bars(z_arr, i, momentum_bars)
            
            if z_falling:
                # 3. MFI en extremo superior
                if mfi > mfi_threshold:
                    # 4. MFI bajando durante momentum_bars velas
                    mfi_falling = _is_falling_for_n_bars(mfi_arr, i, momentum_bars)
                    
                    if mfi_falling:
                        signals[i] = -1

    return signals


@njit
def _decide_exit_trailing_sl(
    close_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    atr_arr: np.ndarray,
    entry_bar: int,
    direction: int,
    entry_price: float,
    trailing_atr_mult: float,
    emergency_sl_pct: float,
) -> tuple[int, float, int]:
    """
    Salida mediante trailing stop ATR + stop loss de emergencia.

    El trailing stop se actualiza vela a vela pero solo se mueve favorablemente.

    Returns:
        (exit_bar, exit_price, reason_code)
        reason_code: 0=trailing_stop, 1=emergency_sl, 2=end_of_data
    """
    n = len(close_arr)
    
    # Stop loss de emergencia
    if direction == 1:  # LONG
        emergency_sl = entry_price * (1.0 - emergency_sl_pct)
    else:  # SHORT
        emergency_sl = entry_price * (1.0 + emergency_sl_pct)

    # Inicializar trailing stop
    best_price = entry_price
    
    if direction == 1:  # LONG
        trailing_stop = entry_price - atr_arr[entry_bar] * trailing_atr_mult
    else:  # SHORT
        trailing_stop = entry_price + atr_arr[entry_bar] * trailing_atr_mult

    # Iterar vela a vela
    for i in range(entry_bar + 1, n):
        close = close_arr[i]
        high = high_arr[i]
        low = low_arr[i]
        atr = atr_arr[i]

        if np.isnan(close) or np.isnan(atr):
            continue

        # Verificar stop loss de emergencia
        if direction == 1:  # LONG
            if low <= emergency_sl:
                return i, emergency_sl, 1
        else:  # SHORT
            if high >= emergency_sl:
                return i, emergency_sl, 1

        # Actualizar best_price
        if direction == 1:  # LONG
            if close > best_price:
                best_price = close
                # Actualizar trailing stop (solo sube)
                new_trailing = best_price - atr * trailing_atr_mult
                if new_trailing > trailing_stop:
                    trailing_stop = new_trailing
        else:  # SHORT
            if close < best_price:
                best_price = close
                # Actualizar trailing stop (solo baja)
                new_trailing = best_price + atr * trailing_atr_mult
                if new_trailing < trailing_stop:
                    trailing_stop = new_trailing

        # Verificar trailing stop
        if direction == 1:  # LONG
            if low <= trailing_stop:
                return i, trailing_stop, 0
        else:  # SHORT
            if high >= trailing_stop:
                return i, trailing_stop, 0

    # Fin de datos - salir al último close
    return n - 1, close_arr[n - 1], 2
