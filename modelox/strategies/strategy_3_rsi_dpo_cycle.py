"""
================================================================================
Strategy 11: MACD_RSI_Sync_Native_Robust
================================================================================
Estrategia Scalping 5m con Motor Matemático BLINDADO.

CORRECCIÓN CRÍTICA:
- El código original de 'indicators.py' fallaba si el dataframe tenía NaNs al inicio.
- Se ha implementado una versión 'Robust' de EMA, RSI y MACD que busca 
  el primer dato válido antes de calcular.
- MATEMÁTICA: Idéntica a la original (RSI Wilder, MACD EMA).

LÓGICA:
- Sincronización de cruces en ventana de 1-4 velas.
- Salida Híbrida (TP/SL Fijo o RSI Target).

Author: MODELOX Quant Team
Date: 2025-12-29
ID: 11
================================================================================
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import polars as pl
from numba import njit

from modelox.core.types import ExitDecision

# =============================================================================
# MOTOR DE INDICADORES BLINDADO (Robust vs NaNs)
# =============================================================================

@njit(cache=True, fastmath=True)
def _ema_robust(src: np.ndarray, period: int) -> np.ndarray:
    """
    EMA que no se rompe si los datos empiezan con NaN.
    """
    n = len(src)
    ema = np.full(n, np.nan, dtype=np.float64)
    if n < period: return ema
    
    alpha = 2.0 / (period + 1.0)
    
    # 1. Encontrar donde empiezan los datos reales
    start_idx = 0
    while start_idx < n and np.isnan(src[start_idx]):
        start_idx += 1
        
    # Verificar si quedan suficientes datos tras saltar los NaNs
    if n - start_idx < period:
        return ema
        
    # 2. Inicializar (Seed) con SMA
    total = 0.0
    for i in range(period):
        total += src[start_idx + i]
    
    # El índice donde termina la SMA inicial
    current_idx = start_idx + period - 1
    ema[current_idx] = total / period
    
    # 3. Calcular el resto
    for i in range(current_idx + 1, n):
        val = src[i]
        # Si encontramos un hueco en medio, mantenemos el valor anterior
        if np.isnan(val):
            ema[i] = ema[i-1]
        else:
            ema[i] = alpha * val + (1.0 - alpha) * ema[i - 1]
            
    return ema

@njit(cache=True, fastmath=True)
def _macd_robust(
    close: np.ndarray, fast: int, slow: int, signal: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MACD usando las EMAs robustas.
    Retorna: (macd_line, signal_line, histogram)
    """
    n = len(close)
    macd_line = np.full(n, np.nan, dtype=np.float64)
    signal_line = np.full(n, np.nan, dtype=np.float64)
    histogram = np.full(n, np.nan, dtype=np.float64)
    
    # Usar EMA Robust
    ema_fast = _ema_robust(close, fast)
    ema_slow = _ema_robust(close, slow)
    
    # MACD Line
    for i in range(n):
        if not np.isnan(ema_fast[i]) and not np.isnan(ema_slow[i]):
            macd_line[i] = ema_fast[i] - ema_slow[i]
    
    # Signal Line (EMA sobre MACD Line)
    # Replicamos lógica EMA robusta sobre la línea MACD resultante
    start_idx = 0
    while start_idx < n and np.isnan(macd_line[start_idx]):
        start_idx += 1
        
    if n - start_idx >= signal:
        alpha_sig = 2.0 / (signal + 1.0)
        
        # Seed Signal
        total = 0.0
        for i in range(signal):
            total += macd_line[start_idx + i]
        
        curr = start_idx + signal - 1
        signal_line[curr] = total / signal
        
        # Loop Signal
        for i in range(curr + 1, n):
            val = macd_line[i]
            if np.isnan(val):
                signal_line[i] = signal_line[i-1]
            else:
                signal_line[i] = alpha_sig * val + (1.0 - alpha_sig) * signal_line[i - 1]

    # Histogram
    for i in range(n):
        if not np.isnan(macd_line[i]) and not np.isnan(signal_line[i]):
            histogram[i] = macd_line[i] - signal_line[i]
            
    return macd_line, signal_line, histogram

@njit(cache=True, fastmath=True)
def _rsi_robust(close: np.ndarray, period: int) -> np.ndarray:
    """RSI Wilder Blindado."""
    n = len(close)
    rsi_out = np.full(n, np.nan, dtype=np.float64)
    
    # Encontrar inicio válido
    start_idx = 0
    while start_idx < n and np.isnan(close[start_idx]):
        start_idx += 1
        
    if n - start_idx < period + 1:
        return rsi_out
    
    # Calcular Deltas iniciales
    gains = 0.0
    losses = 0.0
    
    # Loop de inicialización (SMA de ganancias/perdidas)
    # Empezamos desde start_idx + 1 comparando con el anterior
    for i in range(start_idx + 1, start_idx + period + 1):
        delta = close[i] - close[i - 1]
        if delta > 0:
            gains += delta
        else:
            losses -= delta # delta es negativo, restamos para hacerlo positivo
            
    avg_gain = gains / period
    avg_loss = losses / period
    
    idx = start_idx + period
    if avg_loss == 0:
        rsi_out[idx] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi_out[idx] = 100.0 - 100.0 / (1.0 + rs)
        
    # Smoothing Wilder
    for i in range(idx + 1, n):
        delta = close[i] - close[i - 1]
        
        # Si hay hueco, saltamos o mantenemos el previo
        if np.isnan(delta): 
            continue
            
        if delta > 0:
            up = delta
            down = 0.0
        else:
            up = 0.0
            down = -delta
            
        avg_gain = (avg_gain * (period - 1) + up) / period
        avg_loss = (avg_loss * (period - 1) + down) / period
        
        if avg_loss == 0:
            rsi_out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_out[i] = 100.0 - 100.0 / (1.0 + rs)
            
    return rsi_out

@njit(cache=True, fastmath=True)
def _atr_robust(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    atr = np.full(n, np.nan, dtype=np.float64)
    
    start_idx = 0
    while start_idx < n and np.isnan(close[start_idx]):
        start_idx += 1
        
    if n - start_idx < period: return atr
    
    tr = np.zeros(n, dtype=np.float64)
    # Llenar TR
    for i in range(start_idx + 1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, max(hc, lc))
        
    # SMA Inicial
    total = 0.0
    for i in range(period):
        total += tr[start_idx + 1 + i]
    
    curr = start_idx + period
    atr[curr] = total / period
    
    for i in range(curr + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        
    return atr


# =============================================================================
# LÓGICA DE SINCRONIZACIÓN
# =============================================================================

@njit(cache=True, fastmath=True)
def _generate_sync_signals(
    rsi: np.ndarray,
    macd_hist: np.ndarray,
    rsi_low: float,
    rsi_high: float,
    window: int,
    warmup: int
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    
    n = len(rsi)
    sig_long = np.zeros(n, dtype=np.bool_)
    sig_short = np.zeros(n, dtype=np.bool_)
    
    last_rsi_up = -999
    last_macd_up = -999
    last_rsi_down = -999
    last_macd_down = -999
    
    c_rsi = 0
    c_macd = 0
    
    for i in range(warmup, n):
        if np.isnan(rsi[i]) or np.isnan(macd_hist[i]) or np.isnan(rsi[i-1]):
            continue
            
        # --- LONG ---
        if rsi[i-1] <= rsi_low and rsi[i] > rsi_low:
            last_rsi_up = i
            c_rsi += 1
            
        if macd_hist[i-1] <= 0 and macd_hist[i] > 0:
            last_macd_up = i
            c_macd += 1
            
        # Sync Check
        if (i - last_rsi_up <= window) and (i - last_macd_up <= window):
            # Validar estado actual
            if rsi[i] > rsi_low and macd_hist[i] > 0:
                if not sig_long[i-1]:
                    sig_long[i] = True
                    last_rsi_up = -999
                    last_macd_up = -999

        # --- SHORT ---
        if rsi[i-1] >= rsi_high and rsi[i] < rsi_high:
            last_rsi_down = i
            
        if macd_hist[i-1] >= 0 and macd_hist[i] < 0:
            last_macd_down = i
            
        if (i - last_rsi_down <= window) and (i - last_macd_down <= window):
            if rsi[i] < rsi_high and macd_hist[i] < 0:
                if not sig_short[i-1]:
                    sig_short[i] = True
                    last_rsi_down = -999
                    last_macd_down = -999
                    
    return sig_long, sig_short, c_rsi, c_macd


# =============================================================================
# ESTRATEGIA
# =============================================================================

class Strategy11MACDRSISync:
    # Modular indicator declaration for plotting and reporting
    # Update this list with all indicator columns used by this strategy
    __indicators_used = ["macd_hist", "rsi", "atr"]
    combinacion_id = 15
    name = "MACD_RSI_Sync_Robust"
    
    TIMEOUT_BARS = 300
    
    parametros_optuna = {
        "macd_fast": (6, 6, 1),
        "macd_slow": (13, 13, 1),
        "macd_sig": (4, 4, 1),
        "rsi_period": (7, 7, 1),
        "rsi_low": (25, 45, 2),
        "rsi_high": (55, 75, 2),
        "sync_window": (1, 5, 1),
        "rsi_target_long": (60, 80, 5),
        "rsi_target_short": (20, 40, 5),
        "atr_period": (14, 14, 1),
        "sl_atr": (1.0, 2.5, 0.1),
        "rr_ratio": (1.5, 3.0, 0.25),
    }
    
    @classmethod
    def get_indicators_used(cls):
        return cls.__indicators_used

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        return {
            "macd_fast": 6, "macd_slow": 13, "macd_sig": 4, 
            "rsi_period": 7,
            "rsi_low": trial.suggest_int("rsi_low", 25, 45),
            "rsi_high": trial.suggest_int("rsi_high", 55, 75),
            "sync_window": trial.suggest_int("sync_window", 1, 5),
            "rsi_target_long": trial.suggest_int("rsi_target_long", 60, 80),
            "rsi_target_short": trial.suggest_int("rsi_target_short", 20, 40),
            "atr_period": 14,
            "sl_atr": trial.suggest_float("sl_atr", 1.0, 2.5, step=0.1),
            "rr_ratio": trial.suggest_float("rr_ratio", 1.5, 3.0, step=0.25),
        }
    
    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        # For modular reporting/plotting, always set params_reporting with indicators used
        params["__indicators_used"] = self.get_indicators_used()
        high = np.ascontiguousarray(df["high"].to_numpy().astype(np.float64))
        low = np.ascontiguousarray(df["low"].to_numpy().astype(np.float64))
        close = np.ascontiguousarray(df["close"].to_numpy().astype(np.float64))
        
        # 1. Indicadores (Robustos)
        _, _, macd_hist = _macd_robust(close, int(params["macd_fast"]), int(params["macd_slow"]), int(params["macd_sig"]))
        rsi = _rsi_robust(close, int(params["rsi_period"]))
        atr = _atr_robust(high, low, close, int(params["atr_period"]))
        
        warmup = int(params["macd_slow"]) + 20
        
        # 2. Señales
        sig_long, sig_short, c_rsi, c_macd = _generate_sync_signals(
            rsi, macd_hist,
            float(params["rsi_low"]), float(params["rsi_high"]),
            int(params["sync_window"]),
            warmup
        )
        
        if sig_long.sum() + sig_short.sum() == 0:
             # Ahora deberíamos ver valores reales, no NaNs
            min_h = np.nanmin(macd_hist) if not np.all(np.isnan(macd_hist)) else "NaN"
            print(f"[DEBUG] No Trades. HistMin: {min_h}. Events: RSI={c_rsi}, MACD={c_macd}")

        return df.with_columns([
            pl.Series("signal_long", sig_long),
            pl.Series("signal_short", sig_short),
            pl.Series("macd_hist", macd_hist),
            pl.Series("rsi", rsi),
            pl.Series("atr", atr),
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
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        rsi = df["rsi"].to_numpy().astype(np.float64)
        atr_arr = df["atr"].to_numpy().astype(np.float64)
        n = len(high)
        
        atr_val = atr_arr[entry_idx]
        if np.isnan(atr_val) or atr_val <= 0: atr_val = entry_price * 0.005
            
        sl_dist = atr_val * float(params["sl_atr"])
        tp_dist = sl_dist * float(params["rr_ratio"])
        
        target_long = float(params["rsi_target_long"])
        target_short = float(params["rsi_target_short"])
        
        if side.upper() == "LONG":
            stop_loss = entry_price - sl_dist
            take_profit = entry_price + tp_dist
        else:
            stop_loss = entry_price + sl_dist
            take_profit = entry_price - tp_dist
            
        end_idx = min(entry_idx + self.TIMEOUT_BARS, n)
        
        for i in range(entry_idx + 1, end_idx):
            if np.isnan(high[i]): continue
            
            curr_h = high[i]
            curr_l = low[i]
            curr_rsi = rsi[i]
            
            if side.upper() == "LONG":
                if curr_l <= stop_loss: return ExitDecision(exit_idx=i, reason="FIXED_SL")
                if curr_h >= take_profit: return ExitDecision(exit_idx=i, reason="FIXED_TP")
                if not np.isnan(curr_rsi) and curr_rsi >= target_long: return ExitDecision(exit_idx=i, reason="RSI_TARGET_HIT")
            else:
                if curr_h >= stop_loss: return ExitDecision(exit_idx=i, reason="FIXED_SL")
                if curr_l <= take_profit: return ExitDecision(exit_idx=i, reason="FIXED_TP")
                if not np.isnan(curr_rsi) and curr_rsi <= target_short: return ExitDecision(exit_idx=i, reason="RSI_TARGET_HIT")
                    
        return ExitDecision(exit_idx=end_idx - 1, reason="TIMEOUT")

Strategy = Strategy11MACDRSISync