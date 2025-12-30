"""
================================================================================
Strategy 9: LuxAlgo_GPR_FixedRR
================================================================================
Machine Learning: Gaussian Process Regression (GPR) [LuxAlgo Port]

CONCEPT:
A non-parametric Bayesian approach to regression. It uses a Radial Basis Function
(RBF) kernel to smooth price data and forecast future movements based on a 
training window.

MATH OPTIMIZATION:
Instead of recalculating the Kernel Matrix inversion on every bar (extremely slow),
we pre-calculate the 'Weight Vector' since X-coordinates (time indices) are relative
and constant. We then apply these weights as a Dot Product over the price window.

ENTRY LOGIC:
- We calculate the GPR-smoothed value for the current bar (t) and the forecast (t+1).
- LONG: Forecast(t+1) > Smoothed(t) (Projected Upward Trend).
- SHORT: Forecast(t+1) < Smoothed(t) (Projected Downward Trend).

EXIT LOGIC:
- Fixed Risk:Reward Ratio based on ATR at entry.

Author: MODELOX Senior Quant Engineer
Date: 2025-12-29
ID: 9
================================================================================
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import polars as pl
from numba import njit

from modelox.core.types import ExitDecision


# ============================================================================
# NUMBA: GPR KERNEL MATH (Linear Algebra)
# ============================================================================

@njit(cache=True, fastmath=True)
def _rbf_kernel(x1: float, x2: float, length: float) -> float:
    """Radial Basis Function Kernel."""
    return np.exp(-((x1 - x2)**2) / (2.0 * length**2))

@njit(cache=True, fastmath=True)
def _compute_gpr_weights(window: int, length: float, sigma: float) -> np.ndarray:
    """
    Pre-calculates the GPR Weight Vector.
    This replaces the expensive matrix operations inside the loop.
    
    Returns a vector of shape (2, window). 
    Row 0: Weights to get Current Smoothed Value (at index window-1).
    Row 1: Weights to get Next Forecast Value (at index window).
    """
    # 1. Build Training Identity (X_train = 0..window-1)
    # We construct the Kernel Matrix K (window x window)
    K = np.zeros((window, window), dtype=np.float64)
    for i in range(window):
        for j in range(window):
            K[i, j] = _rbf_kernel(float(i), float(j), length)
            if i == j:
                K[i, j] += sigma * sigma  # Add noise variance (Ridge)

    # 2. Invert K (Using Pseudo-Inverse or Solve)
    # K_inv = (K + sigma^2 I)^-1
    # Since K is symmetric positive-definite, pinv is safe.
    K_inv = np.linalg.pinv(K)

    # 3. Build Query Vector (K_star)
    # We want to predict for:
    # Point A: The current bar (index = window - 1) -> To see current fit
    # Point B: The next bar (index = window) -> To see forecast direction
    
    # K_star is shape (window, 2)
    K_star = np.zeros((window, 2), dtype=np.float64)
    
    current_idx = float(window - 1)
    next_idx = float(window)
    
    for i in range(window):
        # Weights for current time fit
        K_star[i, 0] = _rbf_kernel(float(i), current_idx, length)
        # Weights for next step forecast
        K_star[i, 1] = _rbf_kernel(float(i), next_idx, length)

    # 4. Compute Final Weights (Weights = K_star.T @ K_inv)
    # Shape: (2, window)
    weights = K_star.T @ K_inv
    
    return weights

@njit(cache=True, fastmath=True)
def _apply_gpr_rolling(close: np.ndarray, window: int, length: float, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies the GPR weights over the price array.
    Returns: (Current_Fit_Array, Next_Forecast_Array)
    """
    n = len(close)
    current_fit = np.full(n, np.nan, dtype=np.float64)
    next_forecast = np.full(n, np.nan, dtype=np.float64)
    
    if n < window:
        return current_fit, next_forecast

    # 1. Pre-calculate weights (The heavy lifting)
    # weights[0] is for current fit, weights[1] is for forecast
    weights = _compute_gpr_weights(window, length, sigma)
    w_fit = weights[0]
    w_fcast = weights[1]
    
    # 2. Rolling Window Application
    # LuxAlgo Logic: ytrain = close - mean.
    # Prediction = (Weights @ (Close_Window - Mean))
    
    for i in range(window - 1, n):
        # Extract window
        window_slice = close[i - window + 1 : i + 1]
        
        # Calculate Mean (SMA)
        mean_val = np.mean(window_slice)
        
        # Center data (Y - Mean)
        centered_y = window_slice - mean_val
        
        # Apply Weights (Dot Product)
        # result = weights @ y_centered
        val_fit = 0.0
        val_fcast = 0.0
        
        for j in range(window):
            val_fit += w_fit[j] * centered_y[j]
            val_fcast += w_fcast[j] * centered_y[j]
            
        # Add Mean back to get absolute price
        current_fit[i] = val_fit + mean_val
        next_forecast[i] = val_fcast + mean_val
        
    return current_fit, next_forecast

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
def _generate_gpr_signals(
    fit: np.ndarray, 
    forecast: np.ndarray, 
    warmup: int
):
    n = len(fit)
    signal_long = np.zeros(n, dtype=np.bool_)
    signal_short = np.zeros(n, dtype=np.bool_)
    
    # Need at least 1 previous bar valid
    start_idx = warmup + 1
    
    for i in range(start_idx, n):
        if np.isnan(fit[i]) or np.isnan(forecast[i]):
            continue
            
        # Logic: Compare Forecast(t+1) with Fit(t)
        # If Forecast is higher than current fit, the curve is pointing UP.
        
        slope = forecast[i] - fit[i]
        
        # Optional: Compare with previous slope to detect change
        # prev_slope = forecast[i-1] - fit[i-1]
        
        # Simple Logic:
        # Long if predicting Up
        if slope > 0:
            # Entry trigger: If previous was down or flat
            if (forecast[i-1] - fit[i-1]) <= 0:
                signal_long[i] = True
        
        # Short if predicting Down
        elif slope < 0:
            # Entry trigger: If previous was up or flat
            if (forecast[i-1] - fit[i-1]) >= 0:
                signal_short[i] = True
                
    return signal_long, signal_short


# ============================================================================
# STRATEGY CLASS
# ============================================================================

class Strategy9LuxAlgoGPR:
    # Modular indicator declaration for plotting and reporting
    # Update this list with all indicator columns used by this strategy
    __indicators_used = ["gpr_fit", "gpr_forecast", "atr"]
    combinacion_id = 9
    name = "LuxAlgo_GPR_FixedRR"
    
    TIMEOUT_BARS = 300
    
    parametros_optuna = {
        # GPR Parameters
        "window": (50, 200, 10),     # Training Window
        "length": (10.0, 50.0, 5.0), # RBF Smoothness (Sigma in Kernel)
        "sigma": (0.01, 0.1, 0.01),  # Noise variance
        
        # Exit Params (Fixed RR)
        "atr_period": (14, 14, 1),
        "sl_atr": (1.0, 2.0, 0.1),
        "rr_ratio": (1.75, 3.0, 0.25),
    }
    
    @classmethod
    def get_indicators_used(cls):
        return cls.__indicators_used

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        return {
            "window": trial.suggest_int("window", 50, 200, step=10),
            "length": trial.suggest_float("length", 10.0, 50.0, step=5.0),
            "sigma": trial.suggest_float("sigma", 0.01, 0.1, step=0.01),
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
        
        window = int(params["window"])
        length = float(params["length"])
        sigma = float(params["sigma"])
        
        # 1. Calculate GPR Curves
        # Returns current smoothed line and next step forecast
        fit_line, forecast_line = _apply_gpr_rolling(close, window, length, sigma)
        
        atr = _atr_numba(high, low, close, int(params["atr_period"]))
        
        warmup = window
        
        # 2. Generate Signals based on Slope
        sig_long, sig_short = _generate_gpr_signals(fit_line, forecast_line, warmup)
        
        return df.with_columns([
            pl.Series("signal_long", sig_long),
            pl.Series("signal_short", sig_short),
            pl.Series("gpr_fit", fit_line),
            pl.Series("gpr_forecast", forecast_line),
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
        FIXED EXIT LOGIC
        """
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        atr_arr = df["atr"].to_numpy().astype(np.float64)
        n = len(high)
        
        atr_val = atr_arr[entry_idx]
        if np.isnan(atr_val) or atr_val <= 0: 
            atr_val = entry_price * 0.01
            
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

Strategy = Strategy9LuxAlgoGPR