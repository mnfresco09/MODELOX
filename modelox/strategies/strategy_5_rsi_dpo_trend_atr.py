"""
================================================================================
Strategy 10: MACD_RSI_Momentum_FixedRR
================================================================================
5-Minute Scalping Strategy: RSI Mean Reversion + MACD Momentum Confirmation.

INDICATORS (Optimized for 5m):
- MACD: Fast=8, Slow=21, Signal=5 (Fibonacci Sequence).
- RSI: Period 14.

ENTRY LOGIC:
- LONG:
    1. RSI Crosses UP the Oversold Threshold (e.g., 30).
    2. MACD Histogram is growing (increasing) for 2 consecutive bars.
- SHORT:
    1. RSI Crosses DOWN the Overbought Threshold (e.g., 70).
    2. MACD Histogram is shrinking (decreasing) for 2 consecutive bars.

EXIT LOGIC:
- Fixed Risk:Reward Ratio based on ATR at entry.
- SL: 1.0 - 2.0 ATR.
- TP: SL * Ratio (1.75 - 3.0).

Author: MODELOX Senior Quant Engineer
Date: 2025-12-29
ID: 10
================================================================================
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import polars as pl
from numba import njit

from modelox.core.types import ExitDecision


# ============================================================================
# NUMBA: INDICATORS (MACD, RSI, ATR)
# ============================================================================

@njit(cache=True, fastmath=True)
def _ema_numba(values: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""
    n = len(values)
    ema = np.full(n, np.nan, dtype=np.float64)
    if n < period: return ema
    
    alpha = 2.0 / (period + 1.0)
    
    # Initialize with SMA
    sma_first = np.mean(values[:period])
    ema[period - 1] = sma_first
    
    for i in range(period, n):
        ema[i] = (values[i] * alpha) + (ema[i - 1] * (1.0 - alpha))
        
    return ema

@njit(cache=True, fastmath=True)
def _macd_numba(close: np.ndarray, fast: int, slow: int, sig: int) -> np.ndarray:
    """Returns MACD Histogram only (MacdLine - SignalLine)."""
    ema_fast = _ema_numba(close, fast)
    ema_slow = _ema_numba(close, slow)
    
    n = len(close)
    macd_line = np.full(n, np.nan, dtype=np.float64)
    
    start_idx = max(fast, slow)
    for i in range(start_idx, n):
        macd_line[i] = ema_fast[i] - ema_slow[i]
        
    # Signal Line
    # We need to compute EMA of the valid macd_line part
    # Construct a clean array for signal calculation
    valid_macd = macd_line[start_idx:]
    
    if len(valid_macd) < sig:
        return np.full(n, np.nan, dtype=np.float64) # Not enough data
        
    # Calculate Signal on the valid slice
    signal_slice = _ema_numba(valid_macd, sig)
    
    # Reconstruct Histogram aligned with original array
    # The signal_slice starts corresponding to index 'start_idx'
    for i in range(len(signal_slice)):
        global_idx = start_idx + i
        if not np.isnan(signal_slice[i]):
            # Return Histogram directly
            # Note: signal_slice[i] aligns with macd_line[global_idx]
            # But _ema_numba output is padded.
            # Correct approach for aligned arrays:
            pass
            
    # Re-do specific histogram calculation loop for safety alignment
    hist = np.full(n, np.nan, dtype=np.float64)
    
    # We need signal line on the full array, ignoring NaNs
    # This is tricky in pure Numba without pandas shift, so we do manual loop
    
    # 1. MACD Line
    curr_ema_fast = np.nan
    curr_ema_slow = np.nan
    alpha_fast = 2.0 / (fast + 1.0)
    alpha_slow = 2.0 / (slow + 1.0)
    
    # Pre-calc SMA for init
    # (Simplified for speed: usually EMA converges quickly, 
    # but for precision we use the previous function logic)
    
    # Let's use the array approach, it's safer
    # Re-calculate Signal EMA on the padded MACD line
    # We treat NaN as skip
    
    signal_line = np.full(n, np.nan, dtype=np.float64)
    
    # Find first valid MACD index
    first_valid = -1
    for i in range(n):
        if not np.isnan(macd_line[i]):
            first_valid = i
            break
            
    if first_valid != -1 and (n - first_valid) >= sig:
        # Calculate Signal EMA starting from first_valid
        alpha_sig = 2.0 / (sig + 1.0)
        
        # Init with SMA of first 'sig' macd values
        sum_val = 0.0
        for i in range(sig):
            sum_val += macd_line[first_valid + i]
        
        sig_start_idx = first_valid + sig - 1
        signal_line[sig_start_idx] = sum_val / sig
        
        for i in range(sig_start_idx + 1, n):
            signal_line[i] = (macd_line[i] * alpha_sig) + (signal_line[i - 1] * (1.0 - alpha_sig))
            
    # Final Histogram
    for i in range(n):
        if not np.isnan(macd_line[i]) and not np.isnan(signal_line[i]):
            hist[i] = macd_line[i] - signal_line[i]
            
    return hist

@njit(cache=True, fastmath=True)
def _rsi_numba(close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    rsi = np.full(n, np.nan, dtype=np.float64)
    if n < period + 1: return rsi
    
    # Changes
    deltas = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        deltas[i] = close[i] - close[i-1]
        
    # First avg gain/loss
    gain_sum = 0.0
    loss_sum = 0.0
    for i in range(1, period + 1):
        d = deltas[i]
        if d > 0: gain_sum += d
        else: loss_sum += -d
        
    avg_gain = gain_sum / period
    avg_loss = loss_sum / period
    
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))
        
    # Wilder Smoothing
    for i in range(period + 1, n):
        d = deltas[i]
        gain = d if d > 0 else 0.0
        loss = -d if d < 0 else 0.0
        
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
            
    return rsi

@njit(cache=True, fastmath=True)
def _atr_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    atr = np.full(n, np.nan, dtype=np.float64)
    if n < period: return atr
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, max(hc, lc))
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


# ============================================================================
# SIGNAL GENERATION
# ============================================================================

@njit(cache=True, fastmath=True)
def _generate_momentum_signals(
    rsi: np.ndarray, 
    macd_hist: np.ndarray, 
    rsi_threshold: float,
    warmup: int
):
    n = len(rsi)
    signal_long = np.zeros(n, dtype=np.bool_)
    signal_short = np.zeros(n, dtype=np.bool_)
    
    # Symmetric thresholds
    # e.g., if threshold is 30: Long < 30, Short > 70
    thresh_low = rsi_threshold
    thresh_high = 100.0 - rsi_threshold
    
    for i in range(warmup, n):
        if np.isnan(rsi[i]) or np.isnan(macd_hist[i]): continue
        
        # --- LONG LOGIC ---
        # 1. RSI Crossover: Was <= 30, Now > 30
        rsi_cross_up = rsi[i-1] <= thresh_low and rsi[i] > thresh_low
        
        # 2. MACD Growth (2 candles): H[i] > H[i-1] > H[i-2]
        macd_growing = macd_hist[i] > macd_hist[i-1] and macd_hist[i-1] > macd_hist[i-2]
        
        if rsi_cross_up and macd_growing:
            signal_long[i] = True
            
        # --- SHORT LOGIC ---
        # 1. RSI Crossover: Was >= 70, Now < 70
        rsi_cross_down = rsi[i-1] >= thresh_high and rsi[i] < thresh_high
        
        # 2. MACD Decline (2 candles): H[i] < H[i-1] < H[i-2]
        macd_declining = macd_hist[i] < macd_hist[i-1] and macd_hist[i-1] < macd_hist[i-2]
        
        if rsi_cross_down and macd_declining:
            signal_short[i] = True
            
    return signal_long, signal_short


# ============================================================================
# STRATEGY CLASS
# ============================================================================

class Strategy10MACDRSIFixed:
    # Modular indicator declaration for plotting and reporting
    # Update this list with all indicator columns used by this strategy
    __indicators_used = ["macd_hist", "rsi", "atr"]
    combinacion_id = 10
    name = "MACD_RSI_Momentum_Fixed"
    
    TIMEOUT_BARS = 300
    
    parametros_optuna = {
        # MACD (Fibonacci 5m setup by default)
        "macd_fast": (8, 8, 1),
        "macd_slow": (21, 21, 1),
        "macd_sig": (5, 5, 1),
        
        # RSI
        "rsi_period": (14, 14, 1),
        "rsi_threshold": (30, 60, 1),   # Trigger level (User requested range)
        
        # Exit (Fixed RR)
        "atr_period": (14, 14, 1),
        "sl_atr": (1.0, 2.0, 0.1),
        "rr_ratio": (1.75, 3.0, 0.25),
    }
    
    @classmethod
    def get_indicators_used(cls):
        return cls.__indicators_used

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        return {
            "macd_fast": 8,
            "macd_slow": 21,
            "macd_sig": 5,
            "rsi_period": 14,
            "rsi_threshold": trial.suggest_int("rsi_threshold", 30, 60),
            "atr_period": 14,
            "sl_atr": trial.suggest_float("sl_atr", 1.0, 2.0, step=0.1),
            "rr_ratio": trial.suggest_float("rr_ratio", 1.75, 3.0, step=0.25),
        }
    
    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        # For modular reporting/plotting, always set params_reporting with indicators used
        params["__indicators_used"] = self.get_indicators_used()
        high = np.ascontiguousarray(df["high"].to_numpy().astype(np.float64))
        low = np.ascontiguousarray(df["low"].to_numpy().astype(np.float64))
        close = np.ascontiguousarray(df["close"].to_numpy().astype(np.float64))
        
        # Indicators
        macd_hist = _macd_numba(
            close, 
            int(params["macd_fast"]), 
            int(params["macd_slow"]), 
            int(params["macd_sig"])
        )
        rsi = _rsi_numba(close, int(params["rsi_period"]))
        atr = _atr_numba(high, low, close, int(params["atr_period"]))
        
        warmup = int(params["macd_slow"]) + int(params["macd_sig"]) + 2
        
        # Signals
        sig_long, sig_short = _generate_momentum_signals(
            rsi, macd_hist,
            float(params["rsi_threshold"]),
            warmup
        )
        
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
        """Fixed RR Exit"""
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        atr_arr = df["atr"].to_numpy().astype(np.float64)
        n = len(high)
        
        atr_val = atr_arr[entry_idx]
        if np.isnan(atr_val) or atr_val <= 0: atr_val = entry_price * 0.01
            
        sl_dist = atr_val * float(params["sl_atr"])
        tp_dist = sl_dist * float(params["rr_ratio"])
        
        if side.upper() == "LONG":
            stop_loss = entry_price - sl_dist
            take_profit = entry_price + tp_dist
        else:
            stop_loss = entry_price + sl_dist
            take_profit = entry_price - tp_dist
            
        end_idx = min(entry_idx + self.TIMEOUT_BARS, n)
        
        for i in range(entry_idx + 1, end_idx):
            current_high = high[i]
            current_low = low[i]
            
            if side.upper() == "LONG":
                if current_low <= stop_loss: return ExitDecision(exit_idx=i, reason="FIXED_SL")
                if current_high >= take_profit: return ExitDecision(exit_idx=i, reason="FIXED_TP")
            else:
                if current_high >= stop_loss: return ExitDecision(exit_idx=i, reason="FIXED_SL")
                if current_low <= take_profit: return ExitDecision(exit_idx=i, reason="FIXED_TP")
                    
        return ExitDecision(exit_idx=end_idx - 1, reason="TIMEOUT")

Strategy = Strategy10MACDRSIFixed