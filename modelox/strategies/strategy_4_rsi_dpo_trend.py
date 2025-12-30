"""
================================================================================
Strategy 7: DPO_ZScore_Cycle_FixedRR
================================================================================
Synchronized Mean Reversion Strategy (DPO + Z-Score)

CONCEPT:
Reversion to mean based on two different cycle methodologies:
1. DPO: Detrended cycles (Price vs SMA displacement).
2. Z-Score: Statistical deviation (Standard Deviations from Mean).

STRUCTURE:
- DPO: Normalized [-10, 10]. Logic: Extreme -> Mid.
- Z-Score: 5 Symmetric Zones.
    [+] Extreme (> 2.0)
    [+] Mid (1.0 to 2.0) -> TRIGGER ZONE SHORT
    [=] Neutral (-1.0 to 1.0)
    [-] Mid (-2.0 to -1.0) -> TRIGGER ZONE LONG
    [-] Extreme (< -2.0)

ENTRY LOGIC (Window 3-6 bars):
- LONG: Both must be in Extreme Low (Zone 5 / RIB). One crosses to Mid, second confirms.
- SHORT: Both must be in Extreme High (Zone 1 / RSA). One crosses to Mid, second confirms.

EXIT: Fixed Risk:Reward (SL 1-2 ATR, RR 1.75-3.0).

Author: MODELOX Senior Quant Engineer
Date: 2025-12-29
ID: 7
================================================================================
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import polars as pl
from numba import njit

from modelox.core.types import ExitDecision


# ============================================================================
# NUMBA: DPO INDICATOR (Normalized)
# ============================================================================

@njit(cache=True, fastmath=True)
def _dpo_raw_numba(close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    dpo = np.full(n, np.nan, dtype=np.float64)
    if n < period: return dpo
    
    # Calculate SMA
    sma = np.full(n, np.nan, dtype=np.float64)
    total = 0.0
    for i in range(period): total += close[i]
    sma[period - 1] = total / period
    for i in range(period, n):
        total = total - close[i - period] + close[i]
        sma[i] = total / period
    
    # DPO Calculation: Price - Shifted SMA
    shift = period // 2 + 1
    for i in range(shift + period - 1, n):
        sma_shifted = sma[i - shift]
        if not np.isnan(sma_shifted):
            dpo[i] = close[i] - sma_shifted
    return dpo

@njit(cache=True, fastmath=True)
def _normalize_dpo(dpo: np.ndarray) -> np.ndarray:
    """Normalize DPO to [-10, 10] using 95th percentile."""
    n = len(dpo)
    normalized = np.full(n, np.nan, dtype=np.float64)
    valid_abs = np.zeros(n, dtype=np.float64)
    count = 0
    for i in range(n):
        if not np.isnan(dpo[i]):
            valid_abs[count] = abs(dpo[i])
            count += 1
    if count < 10: return normalized
    
    sorted_abs = np.sort(valid_abs[:count])
    p95_idx = int(count * 0.95)
    if p95_idx >= count: p95_idx = count - 1
    p95_value = sorted_abs[p95_idx]
    
    if p95_value < 1e-10: return normalized
    
    scale = 10.0 / p95_value
    for i in range(n):
        if not np.isnan(dpo[i]):
            val = dpo[i] * scale
            if val > 10.0: val = 10.0
            elif val < -10.0: val = -10.0
            normalized[i] = val
    return normalized

def _dpo_normalized(close: np.ndarray, period: int) -> np.ndarray:
    return _normalize_dpo(_dpo_raw_numba(close, period))


# ============================================================================
# NUMBA: Z-SCORE INDICATOR
# ============================================================================

@njit(cache=True, fastmath=True)
def _zscore_numba(close: np.ndarray, period: int) -> np.ndarray:
    """
    Z-Score = (Close - SMA) / StdDev
    Essentially the normalized distance from the mean.
    """
    n = len(close)
    zscore = np.full(n, np.nan, dtype=np.float64)
    if n < period: return zscore
    
    # We calculate Rolling Mean and Rolling StdDev manually for speed
    for i in range(period - 1, n):
        # Slice window
        sum_val = 0.0
        sum_sq = 0.0
        for j in range(i - period + 1, i + 1):
            val = close[j]
            sum_val += val
            sum_sq += val * val
            
        mean = sum_val / period
        variance = (sum_sq / period) - (mean * mean)
        
        if variance > 1e-10:
            std_dev = np.sqrt(variance)
            zscore[i] = (close[i] - mean) / std_dev
        else:
            zscore[i] = 0.0
            
    return zscore


# ============================================================================
# NUMBA: ATR (For Exit Logic)
# ============================================================================

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
# SIGNAL GENERATION (Synchronized Logic: DPO + Z-Score)
# ============================================================================

@njit(cache=True, fastmath=True)
def _generate_signals_dpo_zscore(
    dpo: np.ndarray, 
    zscore: np.ndarray, 
    dpo_ext: float, 
    z_ext: float, 
    z_mid: float, 
    min_w: int, 
    max_w: int, 
    warmup: int
):
    n = len(dpo)
    signal_long = np.zeros(n, dtype=np.bool_)
    signal_short = np.zeros(n, dtype=np.bool_)
    
    # --- DPO LEVELS (Symmetric 3 distance) ---
    # RSA (High): dpo_ext | RSM (Mid): dpo_ext - 3
    dpo_rsa = dpo_ext
    dpo_rsm = dpo_ext - 3.0
    # RIB (Low): -dpo_ext | RIM (Mid): -dpo_ext + 3
    dpo_rib = -dpo_ext
    dpo_rim = -dpo_ext + 3.0
    
    # --- Z-SCORE LEVELS (5 Parts) ---
    # Zone 1 (High Ext): > z_ext
    # Zone 2 (High Mid): z_mid to z_ext
    # Zone 3 (Neutral): -z_mid to z_mid
    # Zone 4 (Low Mid): -z_ext to -z_mid
    # Zone 5 (Low Ext): < -z_ext
    
    # State tracking
    trig_long_idx = -1
    trig_short_idx = -1
    which_long = 0  # 1=DPO, 2=ZScore
    which_short = 0
    
    for i in range(warmup, n):
        if np.isnan(dpo[i]) or np.isnan(zscore[i]): continue
        
        # =========================
        # LONG LOGIC (Bottom up)
        # =========================
        # We look for transition from Extreme Low to Mid Low
        
        # DPO Trigger: Was in RIB (< -8), crosses into RIM (> -8)
        dpo_cross_up = dpo[i-1] <= dpo_rib and dpo[i] > dpo_rib
        
        # Z-Score Trigger: Was in Zone 5 (< -z_ext), crosses into Zone 4 (> -z_ext)
        z_cross_up = zscore[i-1] <= -z_ext and zscore[i] > -z_ext
        
        # 1. Activate Trigger
        if trig_long_idx == -1:
            if dpo_cross_up:
                trig_long_idx = i
                which_long = 1 # DPO fired first
            elif z_cross_up:
                trig_long_idx = i
                which_long = 2 # ZScore fired first
        
        # 2. Check Confirmation Window
        else: # Trigger active
            bars_passed = i - trig_long_idx
            
            if bars_passed > max_w:
                trig_long_idx = -1 # Timeout
            elif bars_passed >= min_w:
                # Check confirmation from the OTHER indicator
                confirmed = False
                if which_long == 1: # Waiting for ZScore
                    if z_cross_up: confirmed = True
                    # Optional: Check if it's already in Zone 4 coming from 5
                    elif zscore[i-1] <= -z_ext and zscore[i] > -z_ext: confirmed = True
                
                else: # Waiting for DPO
                    if dpo_cross_up: confirmed = True
                
                if confirmed:
                    signal_long[i] = True
                    trig_long_idx = -1 # Reset
        
        # =========================
        # SHORT LOGIC (Top down)
        # =========================
        # We look for transition from Extreme High to Mid High
        
        # DPO Trigger: Was in RSA (> 8), crosses down to RSM (< 8)
        dpo_cross_down = dpo[i-1] >= dpo_rsa and dpo[i] < dpo_rsa
        
        # Z-Score Trigger: Was in Zone 1 (> z_ext), crosses down to Zone 2 (< z_ext)
        z_cross_down = zscore[i-1] >= z_ext and zscore[i] < z_ext
        
        # 1. Activate Trigger
        if trig_short_idx == -1:
            if dpo_cross_down:
                trig_short_idx = i
                which_short = 1
            elif z_cross_down:
                trig_short_idx = i
                which_short = 2
        
        # 2. Check Confirmation Window
        else:
            bars_passed = i - trig_short_idx
            if bars_passed > max_w:
                trig_short_idx = -1
            elif bars_passed >= min_w:
                confirmed = False
                if which_short == 1: # Waiting for ZScore
                    if z_cross_down: confirmed = True
                else: # Waiting for DPO
                    if dpo_cross_down: confirmed = True
                
                if confirmed:
                    signal_short[i] = True
                    trig_short_idx = -1
                    
    return signal_long, signal_short


# ============================================================================
# STRATEGY CLASS
# ============================================================================

class Strategy7DPOZScoreFixedRR:
    # Modular indicator declaration for plotting and reporting
    # Update this list with all indicator columns used by this strategy
    __indicators_used = ["dpo", "zscore", "atr"]
    combinacion_id = 7
    name = "DPO_ZScore_FixedRR"
    
    TIMEOUT_BARS = 300
    
    parametros_optuna = {
        # Indicators
        "dpo_period": (14, 28, 1),
        "z_period": (14, 50, 2),            # Z-Score usually 20-50
        
        # Levels
        "dpo_extreme": (5.0, 9.0, 0.5),     # RSA (e.g., 8.0)
        "z_extreme": (2.0, 3.0, 0.1),       # Zone 1 threshold (e.g., 2.0 StdDev)
        "z_mid_dist": (0.5, 1.5, 0.1),      # Distance to define Zone 2 width
        
        # Window
        "min_window": (2, 5, 1),
        "max_window": (4, 8, 1),
        
        # Exit (Fixed RR)
        "atr_period": (14, 14, 1),
        "sl_atr": (1.0, 2.0, 0.1),
        "rr_ratio": (1.75, 3.0, 0.25),
    }
    
    @classmethod
    def get_indicators_used(cls):
        return cls.__indicators_used

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        min_w = trial.suggest_int("min_window", 2, 5)
        max_w = trial.suggest_int("max_window", min_w + 1, 9)
        
        return {
            "dpo_period": trial.suggest_int("dpo_period", 14, 28),
            "z_period": trial.suggest_int("z_period", 14, 50),
            "dpo_extreme": trial.suggest_float("dpo_extreme", 5.0, 9.0, step=0.5),
            "z_extreme": trial.suggest_float("z_extreme", 1.8, 3.0, step=0.1),
            "z_mid_dist": trial.suggest_float("z_mid_dist", 0.5, 1.5, step=0.1),
            "min_window": min_w,
            "max_window": max_w,
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
        dpo = _dpo_normalized(close, int(params["dpo_period"]))
        zscore = _zscore_numba(close, int(params["z_period"]))
        atr = _atr_numba(high, low, close, int(params["atr_period"]))
        
        warmup = max(int(params["dpo_period"])*2, int(params["z_period"])+1)
        
        # Z-Score Mid level derivation
        # Example: z_extreme=2.0, z_mid_dist=1.0 -> z_mid threshold is 1.0
        # So Zone 2 is between 1.0 and 2.0
        z_mid_thresh = float(params["z_extreme"]) - float(params["z_mid_dist"])
        
        sig_long, sig_short = _generate_signals_dpo_zscore(
            dpo, zscore,
            float(params["dpo_extreme"]),
            float(params["z_extreme"]),
            z_mid_thresh,
            int(params["min_window"]),
            int(params["max_window"]),
            warmup
        )
        
        return df.with_columns([
            pl.Series("signal_long", sig_long),
            pl.Series("signal_short", sig_short),
            pl.Series("dpo", dpo),
            pl.Series("zscore", zscore),
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
        """
        Fixed TP/SL based on ATR at entry.
        """
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
                if current_low <= stop_loss:
                    return ExitDecision(exit_idx=i, reason="FIXED_SL")
                if current_high >= take_profit:
                    return ExitDecision(exit_idx=i, reason="FIXED_TP")
            else:
                if current_high >= stop_loss:
                    return ExitDecision(exit_idx=i, reason="FIXED_SL")
                if current_low <= take_profit:
                    return ExitDecision(exit_idx=i, reason="FIXED_TP")
                    
        return ExitDecision(exit_idx=end_idx - 1, reason="TIMEOUT")

Strategy = Strategy7DPOZScoreFixedRR