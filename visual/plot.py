"""  
================================================================================
MODELOX - Dynamic Multi-Strategy Trading Charts (v6.1 UNIFIED WARMUP)
================================================================================
100% dynamic indicator detection - Compatible with any strategy (1, 2, N...).

ARCHITECTURE v6.1 - ZERO-LAG + GLOBAL WARMUP:

1. SINGLE AUTHORITATIVE TIMESTAMP ARRAY (SATA)
   - ts_q = authoritative timestamps from candle data
   - ALL data (indicators, markers) references ts_q indices
   - Eliminates any possibility of desynchronization

2. StrictAlignmentMapper CLASS
   - Maps indicator values to ts_q using O(1) hash lookup
   - Handles NaN values as None (gaps in line series)
   - Guarantees: indicator[i] corresponds to candle[i]

3. GLOBAL WARM-UP PERIOD (v6.1 - DATA SLICING)
   - Detects max period from all *_period params (mfi_period, atr_period, etc.)
   - SLICES all data (candles + volume + indicators) from warmup point
   - Chart visually starts at candle[max_warmup]
   - Trade markers within warmup period are filtered out
   - Result: Clean, synchronized chart where everything starts together

4. CENTRALIZED TIMESTAMP NORMALIZATION
   - _normalize_timestamps_to_unix() is the ONLY timestamp converter
   - Ensures consistent datetime64 -> Unix seconds conversion
   - Supports any datetime64 precision (ns, us, ms, s)

5. CROSS-PANEL MARKER MIRRORING
   - Markers use exact candle timestamps (no approximation)
   - Markers appear on price chart AND indicator panels
   - Zero pixel deviation between chart elements

6. DYNAMIC OPTUNA PARAMETER BINDING
   - Reference lines read from trial params (overbought, oversold)
   - Panel names include period (e.g., "MFI (14)")
   - Entry/exit levels visualized as horizontal lines

PERFORMANCE:
- orjson for 10x faster JSON serialization
- Vectorized numpy operations (no row-by-row iteration)
- Data quantization (2-4 decimal precision)
- Unix Int64 timestamps (minimal payload)
- Polars-first processing

================================================================================
"""

from __future__ import annotations

import os
import re
from typing import Optional, List, Union, Dict, Any, TYPE_CHECKING

import numpy as np

# Ultra-fast JSON serialization
try:
    import orjson
    HAS_ORJSON = True
    def _dumps(obj: dict) -> str:
        return orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY).decode('utf-8')
except ImportError:
    import json
    HAS_ORJSON = False
    def _dumps(obj: dict) -> str:
        return json.dumps(obj, separators=(',', ':'), default=str)

# Polars support
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

import pandas as pd

if TYPE_CHECKING:
    DataFrameType = Union[pd.DataFrame, "pl.DataFrame"]


# =============================================================================
# INDICATOR DETECTION CONFIGURATION (Refactored v3.0)
# =============================================================================

# Color map for consistent indicator colors across all strategies
INDICATOR_COLORS = {
    # Oscillators (sub-panels)
    "rsi": "#60a5fa",        # Blue
    "stc": "#f472b6",        # Pink
    "chop": "#fb923c",       # Orange
    "dpo": "#22c55e",        # Green (DPO)
    "adx": "#a78bfa",        # Purple
    # Strategy 388 colors
    "rsi14": "#f59e0b",       # Orange
    "rsi9": "#10b981",        # Green
    "wavetrend": "#8b5cf6",   # Violet
    "cci": "#ec4899",         # Pink
    "ml_prediction": "#06b6d4",# Cyan
    "mfi": "#34d399",        # Emerald
    "zscore": "#fbbf24",     # Amber
    "macd": "#60a5fa",       # Blue
    "macd_hist": "#60a5fa",  # Blue
    "macd_signal": "#fbbf24", # Amber
    
    # NEW: Stochastic & ROC
    "stoch_k": "#f472b6",    # Pink
    "stoch_d": "#a78bfa",    # Purple
    "roc": "#fb923c",        # Orange
    
    # NEW: ML Learning / Lorentzian
    "ml_vote": "#22d3ee",    # Cyan
    "zscore_kalman": "#fbbf24",  # Amber
    "hma_slope": "#ec4899",  # Pink
    "kalman_dist": "#a78bfa",  # Purple
    "slope_signal": "#f97316",  # Orange
    
    # Overlays (price panel)
    "ema": "#fbbf24",        # Amber (generic EMA)
    "ema_200": "#fbbf24",    # Amber
    "ema200_mtf": "#a855f7", # Violet
    "ema_base": "#f97316",   # Orange (Mean Reversion base EMA)
    "sma": "#94a3b8",        # Gray
    "wma": "#60a5fa",        # Blue
    "vwma": "#22d3ee",       # Cyan
    "vwap_session": "#38bdf8", # Sky
    "supertrend": "#10b981", # Emerald
    "ema_50": "#f97316",     # Orange
    "ema_20": "#84cc16",     # Lime
    "hma": "#ec4899",        # Pink
    "donchian_hi": "#22c55e", # Green
    "donchian_lo": "#ef4444", # Red

    # ALMA family (Strategy 2000)
    "alma": "#fbbf24",
    "alma_5": "#22c55e",   # Green
    "alma_25": "#fbbf24",  # Amber
    "alma_60": "#ef4444",  # Red
    
    # Nadaraya-Watson
    "nw_baseline": "#a78bfa", # Purple
    "nw_upper": "#22c55e",    # Green
    "nw_lower": "#ef4444",    # Red
    
    # SSL Hull
    "ssl_baseline": "#a78bfa", # Purple
    "ssl_line": "#22c55e",     # Green
    "ssl_trend": "#fbbf24",    # Amber
    # SSL Hybrid Advanced
    "ssl_upper": "#22c55e",    # Green
    "ssl_lower": "#ef4444",    # Red
    "ssl1_line": "#f472b6",   # Pink
    "ssl2_line": "#60a5fa",   # Blue
    "ssl_exit_line": "#fbbf24", # Amber
    "ssl_hlv": "#a78bfa",     # Purple
    "ssl_candle_violation": "#ef4444", # Red
    
    # TTM Squeeze
    "squeeze": "#f472b6",    # Pink
    "squeeze_mom": "#60a5fa", # Blue
    "bb_basis": "#94a3b8",   # Gray
    "bb_upper": "#94a3b8",   # Gray
    "bb_lower": "#94a3b8",   # Gray
    "kc_basis": "#fbbf24",   # Amber
    "kc_upper": "#fbbf24",   # Amber
    "kc_lower": "#fbbf24",   # Amber
    
    # ATR
    "atr": "#fb923c",        # Orange
    "atr_p90": "#ef4444",    # Red (ATR Percentile 90 threshold)
    
    # Laguerre RSI
    "laguerre_rsi": "#a78bfa", # Purple
    
    # KAMA
    "kama": "#22d3ee",       # Cyan
    "er": "#34d399",         # Emerald (Efficiency Ratio)
    "er_kaufman": "#34d399", # Emerald
    
    # Elder Ray
    "elder_ema": "#fbbf24",  # Amber
    "bull_power": "#22c55e", # Green
    "bear_power": "#ef4444", # Red
    
    # Kalman
    "kalman": "#a855f7",     # Violet
    
    # LinReg
    "linreg_slope": "#fb923c", # Orange
    "lrs": "#fb923c",          # Orange (raw slope)
    "lrs_smooth": "#22d3ee",    # Cyan (smoothed slope)
    "lrs_signal": "#94a3b8",    # Gray (signal line)
    
    # ADX components
    "plus_di": "#22c55e",    # Green
    "minus_di": "#ef4444",   # Red
    
    # Kinematics (Derivatives)
    "velocity": "#06b6d4",      # Cyan
    "acceleration": "#f43f5e",  # Rose
    "hma_velocity": "#06b6d4",  # Cyan
    "hma_acceleration": "#f43f5e", # Rose
    "hma_fast_velocity": "#06b6d4",  # Cyan
    "hma_slow_velocity": "#0ea5e9",  # Sky
    "hma_fast_acceleration": "#f43f5e", # Rose
}

# Overlay indicators (values in price range, drawn on main chart)
OVERLAY_INDICATORS = {
    # Moving Averages
    "ema", "ema_200", "ema200_mtf", "ema_50", "ema_20", "ema_base",
    "sma", "wma", "hma", "vwma", "vwap_session",
    "alma", "alma_5", "alma_25", "alma_60",
    "kama", "kalman",
    
    # Trend
    "supertrend",
    
    # Bands/Channels
    "donchian_hi", "donchian_lo",
    "bb_basis", "bb_upper", "bb_lower",
    "kc_basis", "kc_upper", "kc_lower",
    "nw_baseline", "nw_upper", "nw_lower",
    "ssl_baseline", "ssl_line",
    # SSL Hybrid Advanced overlay lines
    "ssl_upper", "ssl_lower", "ssl1_line", "ssl2_line", "ssl_exit_line",
    
    # Elder Ray
    "elder_ema",
}

# Sub-panel oscillators (bounded 0-100 or unbounded, separate panel)
OSCILLATOR_INDICATORS = {
    # Classic Oscillators
    "rsi", "stc", "chop", "adx", "mfi", "zscore", "dpo", "rsi14", "rsi9", "wavetrend", "cci", "ml_prediction",
    
    # Strategy 388 Lorentzian indicators
    "rsi14", "rsi9", "wavetrend", "cci", "ml_prediction",
    
    # NEW: Stochastic & ROC
    "stoch_k", "stoch_d", "roc",
    
    # NEW: ML Learning / Lorentzian
    "ml_vote", "zscore_kalman", "hma_slope", "kalman_dist", "slope_signal",
    
    # MACD components
    "macd", "macd_hist", "macd_signal",
    
    # Efficiency/Trend
    "er", "er_kaufman", "efficiency_ratio",
    "linreg_slope",
    "lrs", "lrs_smooth", "lrs_signal",  # LRS Advanced
    
    # Advanced Oscillators
    "laguerre_rsi",
    "mfi_slope",
    
    # Elder Ray
    "bull_power", "bear_power",
    
    # SSL / Trend direction
    "ssl_trend", "ssl_hlv", "ssl_candle_violation",
    
    # TTM Squeeze
    "squeeze", "squeeze_mom",
    
    # ADX components
    "plus_di", "minus_di",
    
    # ATR (can be sub-panel for visibility)
    "atr", "atr_p90",
    
    # Kinematics (Derivatives)
    "velocity", "acceleration",
    "hma_velocity", "hma_acceleration",
    "hma_fast_velocity", "hma_slow_velocity", "hma_fast_acceleration",
}

# Reference lines for bounded oscillators
OSCILLATOR_BOUNDS = {
    "dpo": {"hi": 10, "lo": -10, "mid": 0},
    "rsi": {"hi": 70, "lo": 30, "mid": 50},
    "stc": {"hi": 75, "lo": 25, "mid": 50},
    "chop": {"hi": 61.8, "lo": 38.2, "mid": 50},
    "mfi": {"hi": 80, "lo": 20, "mid": 50},
    "stoch_k": {"hi": 80, "lo": 20, "mid": 50},
    "stoch_d": {"hi": 80, "lo": 20, "mid": 50},
    "laguerre_rsi": {"hi": 0.8, "lo": 0.2, "mid": 0.5},
    "er": {"hi": 0.6, "lo": 0.2, "mid": 0.4},
    "er_kaufman": {"hi": 0.6, "lo": 0.2, "mid": 0.4},
    "adx": {"hi": 40, "lo": 20, "mid": 25},
    # Strategy 388 bounds
    "rsi14": {"hi": 70, "lo": 30, "mid": 50},
    "rsi9": {"hi": 70, "lo": 30, "mid": 50},
    "wavetrend": {"hi": 60, "lo": -60, "mid": 0},
    "cci": {"hi": 100, "lo": -100, "mid": 0},
    "ml_prediction": {"hi": 8, "lo": -8, "mid": 0},
    "plus_di": {"hi": 40, "lo": 20, "mid": 25},
    "minus_di": {"hi": 40, "lo": 20, "mid": 25},
    # ML Learning / Lorentzian bounds
    "ml_vote": {"hi": 0.5, "lo": -0.5, "mid": 0},
    "zscore_kalman": {"hi": 2.0, "lo": -2.0, "mid": 0},
    "zscore": {"hi": 3.0, "lo": -3.0, "mid": 0},  # Z-Score bounds
    "atr_p90": {},  # ATR Percentile 90 - no bounds (same panel as ATR)
    
    # Kinematics - no bounds (unbounded oscillators)
    "velocity": {"mid": 0},
    "acceleration": {"mid": 0},
    "hma_velocity": {"mid": 0},
    "hma_acceleration": {"mid": 0},
    "hma_fast_velocity": {"mid": 0},
    "hma_slow_velocity": {"mid": 0},
    "hma_fast_acceleration": {"mid": 0},
}

# MACD-style indicators (need special rendering with histogram)
MACD_STYLE_INDICATORS = {"macd", "macd_hist", "macd_signal"}

# Histogram-style oscillators (colored bars like momentum)
HISTOGRAM_INDICATORS = {
    "macd_hist", "squeeze_mom", "bull_power", "bear_power",
    "roc",  # ROC can be rendered as histogram
}


# =============================================================================
# VECTORIZED DATA PREPARATION (v6.0 - ZERO-LAG ARCHITECTURE)
# =============================================================================
# KEY PRINCIPLE: Single Authoritative Timestamp Array (SATA)
# All data (candles, indicators, markers) MUST reference the same ts_q array.
# This eliminates any possibility of desynchronization.
# =============================================================================

def _normalize_timestamps_to_unix(timestamps: np.ndarray) -> np.ndarray:
    """
    Convert any timestamp format to Unix Epoch seconds (Int64).
    This is the ONLY function that should perform timestamp conversion.
    
    Supports:
    - datetime64[ns], datetime64[s], datetime64[ms], etc.
    - Float timestamps (assumed to be Unix seconds)
    - Already Int64 timestamps
    
    Returns:
        np.ndarray[np.int64] - Unix seconds
    """
    if np.issubdtype(timestamps.dtype, np.datetime64):
        # Convert any datetime64 variant to seconds precision, then to int64
        return timestamps.astype('datetime64[s]').astype(np.int64)
    elif np.issubdtype(timestamps.dtype, np.floating):
        return np.floor(timestamps).astype(np.int64)
    elif np.issubdtype(timestamps.dtype, np.integer):
        return timestamps.astype(np.int64)
    else:
        # Fallback: try to parse as datetime64
        return timestamps.astype('datetime64[s]').astype(np.int64)


def _prepare_ohlcv_vectorized(
    timestamps: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: Optional[np.ndarray] = None,
    price_precision: int = 2,
) -> tuple:
    """
    Vectorized OHLCV preparation for Lightweight Charts.
    
    RETURNS the authoritative timestamp array (ts_q) that ALL other
    data (indicators, markers) MUST reference for perfect alignment.
    """
    # CRITICAL: Sanitize NaN/Inf values BEFORE any conversion
    opens = np.nan_to_num(opens, nan=0.0, posinf=0.0, neginf=0.0)
    highs = np.nan_to_num(highs, nan=0.0, posinf=0.0, neginf=0.0)
    lows = np.nan_to_num(lows, nan=0.0, posinf=0.0, neginf=0.0)
    closes = np.nan_to_num(closes, nan=0.0, posinf=0.0, neginf=0.0)
    if volumes is not None:
        volumes = np.nan_to_num(volumes, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize timestamps to Unix seconds using central function
    timestamps = _normalize_timestamps_to_unix(timestamps)
    
    # Remove duplicates and ensure strict ordering
    unique_ts, unique_indices = np.unique(timestamps, return_index=True)
    if len(unique_ts) < len(timestamps):
        timestamps = unique_ts
        opens = opens[unique_indices]
        highs = highs[unique_indices]
        lows = lows[unique_indices]
        closes = closes[unique_indices]
        if volumes is not None:
            volumes = volumes[unique_indices]
    
    # Ensure sorted (should already be, but guarantee it)
    sort_indices = np.argsort(timestamps)
    if not np.all(sort_indices == np.arange(len(timestamps))):
        timestamps = timestamps[sort_indices]
        opens = opens[sort_indices]
        highs = highs[sort_indices]
        lows = lows[sort_indices]
        closes = closes[sort_indices]
        if volumes is not None:
            volumes = volumes[sort_indices]
    
    # Quantize prices
    factor = 10 ** price_precision
    opens = np.round(opens * factor).astype(np.int64)
    highs = np.round(highs * factor).astype(np.int64)
    lows = np.round(lows * factor).astype(np.int64)
    closes = np.round(closes * factor).astype(np.int64)
    
    if volumes is not None:
        volumes = np.round(volumes).astype(np.int64)
    
    return timestamps, opens, highs, lows, closes, volumes, factor


class StrictAlignmentMapper:
    """
    ZERO-LAG ALIGNMENT ENGINE (v6.0)
    
    Maps indicator values to the authoritative candle timestamp array.
    Guarantees that indicator[i] corresponds EXACTLY to candle[i].
    
    Architecture:
    1. ts_q (authoritative) = timestamps from candle data after filtering/dedup
    2. indicator_ts = timestamps from indicator calculation source
    3. This class creates a mapping: indicator_ts -> ts_q indices
    4. Result: indicator values aligned to ts_q with None for gaps
    
    This eliminates any possibility of lag or shift between indicators and candles.
    """
    
    def __init__(self, authoritative_timestamps: np.ndarray):
        """
        Initialize with the authoritative timestamp array (ts_q from candles).
        
        Args:
            authoritative_timestamps: np.ndarray[np.int64] - Unix seconds from candle data
        """
        self.ts_q = authoritative_timestamps
        self.n = len(authoritative_timestamps)
        # O(1) lookup: timestamp -> index in ts_q
        self.ts_to_idx = {int(ts): i for i, ts in enumerate(authoritative_timestamps)}
    
    def align(self, indicator_timestamps: np.ndarray, indicator_values: np.ndarray) -> list:
        """
        Align indicator values to the authoritative timestamp array.
        
        Args:
            indicator_timestamps: np.ndarray[np.int64] - Timestamps from indicator source
            indicator_values: np.ndarray[np.float64] - Indicator values
            
        Returns:
            list[float|None] - Values aligned to ts_q, with None for gaps/NaN
        """
        # Initialize with None for all authoritative timestamps
        aligned = [None] * self.n
        
        # Ensure timestamps are Unix seconds
        if not np.issubdtype(indicator_timestamps.dtype, np.integer):
            indicator_timestamps = _normalize_timestamps_to_unix(indicator_timestamps)
        
        # Map each indicator value to its corresponding ts_q index
        for i, ts in enumerate(indicator_timestamps):
            ts_int = int(ts)
            if ts_int in self.ts_to_idx:
                val = indicator_values[i]
                # Only assign if value is valid (not NaN/Inf)
                if not (np.isnan(val) or np.isinf(val)):
                    aligned[self.ts_to_idx[ts_int]] = float(val)
        
        return aligned
    
    def quantize(self, aligned_values: list, precision: int = 4) -> tuple:
        """
        Quantize aligned values for JSON serialization.
        
        Args:
            aligned_values: list[float|None] from align()
            precision: Decimal places for quantization
            
        Returns:
            (quantized_list, factor)
        """
        factor = 10 ** precision
        quantized = [
            int(round(v * factor)) if v is not None else None 
            for v in aligned_values
        ]
        return quantized, factor
    
    def count_valid(self, aligned_values: list) -> int:
        """Count non-None values in aligned list."""
        return sum(1 for v in aligned_values if v is not None)


def _prepare_indicator_vectorized_aligned(
    candle_timestamps: np.ndarray,
    indicator_values: np.ndarray,
    precision: int = 4
) -> tuple:
    """
    Vectorized indicator preparation with STRICT 1:1 ALIGNMENT.
    
    DEPRECATED: Use StrictAlignmentMapper for new code.
    Kept for backwards compatibility.
    """
    if len(indicator_values) == 0:
        return None, None, None
    
    # Ensure arrays have same length
    if len(candle_timestamps) != len(indicator_values):
        min_len = min(len(candle_timestamps), len(indicator_values))
        candle_timestamps = candle_timestamps[:min_len]
        indicator_values = indicator_values[:min_len]
    
    factor = 10 ** precision
    
    # Convert values: NaN -> None, valid -> quantized int
    values_list = []
    valid_count = 0
    
    for val in indicator_values:
        if np.isnan(val) or np.isinf(val):
            values_list.append(None)
        else:
            values_list.append(int(np.round(val * factor)))
            valid_count += 1
    
    if valid_count == 0:
        return None, None, None
    
    return candle_timestamps, values_list, factor


def _prepare_indicator_vectorized(
    timestamps: np.ndarray,
    values: np.ndarray,
    precision: int = 4
) -> tuple:
    """
    LEGACY: Vectorized indicator preparation (filters NaN - causes desync).
    DEPRECATED: Use StrictAlignmentMapper instead.
    """
    mask = ~np.isnan(values)
    ts_clean = timestamps[mask]
    vals_clean = values[mask]
    
    if len(vals_clean) == 0:
        return None, None, None
    
    factor = 10 ** precision
    vals_quantized = np.round(vals_clean * factor).astype(np.int64)
    
    return ts_clean, vals_quantized, factor


# =============================================================================
# GLOBAL WARM-UP PERIOD (v6.1)
# =============================================================================
# All indicators must "wake up" at the same candle to ensure visual consistency.
# This eliminates the staggered start caused by different indicator periods.
# =============================================================================

def _detect_max_warmup_period(params: Optional[Dict[str, Any]], min_warmup: int = 1) -> int:
    """
    Detect the maximum warm-up period from all Optuna parameters.
    
    Scans for any key ending with '_period', '_length', '_len', '_window'
    and returns the maximum value found.
    
    Args:
        params: Optuna trial parameters dict
        min_warmup: Minimum warmup period (default: 1)
        
    Returns:
        int: Maximum warm-up period (at least min_warmup)
    
    Example:
        params = {'mfi_period': 14, 'atr_period': 20, 'ema_length': 50}
        -> Returns 50 (ema_length is the max)
    """
    if not params:
      return min_warmup

    # Explicit override from strategy (fully modular):
    # if a strategy sets params["__warmup_bars"], respect it.
    override = params.get("__warmup_bars") if isinstance(params, dict) else None
    try:
      if override is not None:
        ov = int(override)
        if ov > 0:
          return max(min_warmup, ov)
    except (ValueError, TypeError):
      pass
    
    period_suffixes = ('_period', '_length', '_len', '_window')
    max_period = min_warmup
    
    for key, value in params.items():
        key_lower = key.lower()
        # Check if key ends with a period-related suffix
        if any(key_lower.endswith(suffix) for suffix in period_suffixes):
            try:
                period_val = int(value)
                if period_val > max_period:
                    max_period = period_val
            except (ValueError, TypeError):
                pass
    
    return max_period


def _apply_warmup_mask(aligned_values: list, warmup_period: int) -> list:
    """
    Apply warm-up mask to aligned indicator values.
    
    Sets the first `warmup_period` values to None, regardless of their original value.
    This ensures all indicators visually start at the same candle.
    
    Args:
        aligned_values: list[float|None] - Values aligned to ts_q
        warmup_period: int - Number of initial values to mask as None
        
    Returns:
        list[float|None] - Values with warmup period masked
    """
    if warmup_period <= 0:
        return aligned_values
    
    # Create a copy to avoid modifying the original
    masked = aligned_values.copy()
    
    # Mask the first `warmup_period` values
    for i in range(min(warmup_period, len(masked))):
        masked[i] = None
    
    return masked


def _extract_indicator_params_from_optuna(params: Optional[Dict[str, Any]], indicator_name: str) -> Dict[str, Any]:
    """
    Extract indicator-specific parameters from Optuna trial params.
    
    100% MODULAR: Dynamically compares column names with param keys.
    
    Searches for keys like:
    - '{indicator}_overbought', '{indicator}_oversold', '{indicator}_period'
    - 'overbought_{indicator}', 'oversold_{indicator}', 'period_{indicator}'
    - 'entry_long_{indicator}', 'exit_short_{indicator}' (strategy levels)
    - Generic 'overbought', 'oversold', 'period' if only one oscillator
    
    Returns dict with 'hi', 'lo', 'mid', 'period', 'entry_long', 'entry_short', 'exit_long', 'exit_short' if found.
    """
    if not params:
        return {}
    
    result = {}
    ind_lower = indicator_name.lower()
    
    # === PERIOD DETECTION ===
    period_patterns = [
        f"{ind_lower}_period", f"period_{ind_lower}", f"{ind_lower}_length",
        f"length_{ind_lower}", f"{ind_lower}_len", f"{ind_lower}_window",
        f"window_{ind_lower}", "period", "length"
    ]
    for key in period_patterns:
        if key in params and params[key] is not None:
            try:
                result['period'] = int(params[key])
                break
            except (ValueError, TypeError):
                pass
    
    # === OVERBOUGHT/OVERSOLD DETECTION ===
    overbought_patterns = [
        f"{ind_lower}_overbought", f"overbought_{ind_lower}", f"{ind_lower}_ob",
        f"{ind_lower}_upper", f"upper_{ind_lower}", f"{ind_lower}_hi",
        f"{ind_lower}_threshold_hi", "overbought", "ob_level", "upper_threshold"
    ]
    oversold_patterns = [
        f"{ind_lower}_oversold", f"oversold_{ind_lower}", f"{ind_lower}_os",
        f"{ind_lower}_lower", f"lower_{ind_lower}", f"{ind_lower}_lo",
        f"{ind_lower}_threshold_lo", "oversold", "os_level", "lower_threshold"
    ]
    
    for key in overbought_patterns:
        if key in params and params[key] is not None:
            try:
                result['hi'] = float(params[key])
                break
            except (ValueError, TypeError):
                pass
    
    for key in oversold_patterns:
        if key in params and params[key] is not None:
            try:
                result['lo'] = float(params[key])
                break
            except (ValueError, TypeError):
                pass
    
    # === STRATEGY ENTRY/EXIT LEVELS (Pendulum Logic) ===
    # Entry Long: when indicator reaches this level, enter long
    entry_long_patterns = [
        f"entry_long_{ind_lower}", f"{ind_lower}_entry_long", f"entry_long",
        f"long_entry_{ind_lower}", f"{ind_lower}_long_entry", "entry_level_long"
    ]
    entry_short_patterns = [
        f"entry_short_{ind_lower}", f"{ind_lower}_entry_short", f"entry_short",
        f"short_entry_{ind_lower}", f"{ind_lower}_short_entry", "entry_level_short"
    ]
    exit_long_patterns = [
        f"exit_long_{ind_lower}", f"{ind_lower}_exit_long", f"exit_long",
        f"long_exit_{ind_lower}", f"{ind_lower}_long_exit", "exit_level_long"
    ]
    exit_short_patterns = [
        f"exit_short_{ind_lower}", f"{ind_lower}_exit_short", f"exit_short",
        f"short_exit_{ind_lower}", f"{ind_lower}_short_exit", "exit_level_short"
    ]
    
    for key in entry_long_patterns:
        if key in params and params[key] is not None:
            try:
                result['entry_long'] = float(params[key])
                break
            except (ValueError, TypeError):
                pass
    
    for key in entry_short_patterns:
        if key in params and params[key] is not None:
            try:
                result['entry_short'] = float(params[key])
                break
            except (ValueError, TypeError):
                pass
    
    for key in exit_long_patterns:
        if key in params and params[key] is not None:
            try:
                result['exit_long'] = float(params[key])
                break
            except (ValueError, TypeError):
                pass
    
    for key in exit_short_patterns:
        if key in params and params[key] is not None:
            try:
                result['exit_short'] = float(params[key])
                break
            except (ValueError, TypeError):
                pass
    
    # Calculate mid if we have hi and lo
    if 'hi' in result and 'lo' in result:
        result['mid'] = (result['hi'] + result['lo']) / 2
    
    return result


def _generate_dynamic_combo(params: Optional[Dict[str, Any]], strategy_name: str = "") -> str:
    """
    Generate a dynamic combo string summarizing key trial parameters.
    
    Format: STRATEGY | P:14 OB:75 OS:25 | EntryL:20 ExitS:80
    
    This allows identifying winning configurations at a glance in YouTube/TikTok shorts.
    """
    if not params:
        return strategy_name or "TRIAL"
    
    parts = []
    
    # Strategy name first
    if strategy_name:
        parts.append(strategy_name.upper())
    
    # Extract key parameters
    period_keys = [k for k in params.keys() if 'period' in k.lower() or 'length' in k.lower()]
    level_keys = [k for k in params.keys() if any(x in k.lower() for x in ['overbought', 'oversold', 'ob', 'os', 'entry', 'exit', 'threshold'])]
    
    param_parts = []
    
    # Periods (P:)
    for key in sorted(period_keys)[:2]:  # Max 2 periods
        val = params[key]
        if val is not None:
            # Extract indicator name from key
            ind_name = key.replace('_period', '').replace('period_', '').replace('_length', '').upper()[:3]
            param_parts.append(f"{ind_name}:{int(val)}")
    
    # Levels (OB/OS/Entry/Exit)
    for key in sorted(level_keys)[:4]:  # Max 4 levels
        val = params[key]
        if val is not None:
            key_lower = key.lower()
            if 'overbought' in key_lower or '_ob' in key_lower:
                param_parts.append(f"OB:{val:.0f}" if isinstance(val, float) else f"OB:{val}")
            elif 'oversold' in key_lower or '_os' in key_lower:
                param_parts.append(f"OS:{val:.0f}" if isinstance(val, float) else f"OS:{val}")
            elif 'entry_long' in key_lower:
                param_parts.append(f"EL:{val:.0f}" if isinstance(val, float) else f"EL:{val}")
            elif 'entry_short' in key_lower:
                param_parts.append(f"ES:{val:.0f}" if isinstance(val, float) else f"ES:{val}")
            elif 'exit_long' in key_lower:
                param_parts.append(f"XL:{val:.0f}" if isinstance(val, float) else f"XL:{val}")
            elif 'exit_short' in key_lower:
                param_parts.append(f"XS:{val:.0f}" if isinstance(val, float) else f"XS:{val}")
    
    if param_parts:
        parts.append(' '.join(param_parts[:6]))  # Max 6 params for readability
    
    return ' | '.join(parts) if parts else strategy_name or "TRIAL"


def _detect_indicators(
    df_cols: set, 
    price_range: tuple,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Auto-detect indicators from DataFrame columns.
    Categorizes into overlays vs sub-panels based on indicator type.
    
    NEW: Reads Optuna params to dynamically set bounds and panel names.
    
    Args:
        df_cols: Set of column names in DataFrame
        price_range: (min_price, max_price) tuple for overlay detection
        params: Optuna trial parameters dict for dynamic bounds
        
    Returns:
        Dict with 'overlays' and 'sub_panels' lists
    """
    # Columns to skip (base OHLCV data)
    skip_cols = {"timestamp", "open", "high", "low", "close", "volume", 
                 "signal_long", "signal_short", "_session_day"}
    
    overlays = []
    sub_panels = []
    
    for col in df_cols:
        if col in skip_cols or col.startswith("_"):
            continue
            
        # Check if overlay (price-range indicator)
        if col in OVERLAY_INDICATORS or col.endswith("_upper") or col.endswith("_lower"):
            overlays.append({
                "col": col,
                "color": INDICATOR_COLORS.get(col, "#fbbf24"),
                "type": "line"
            })
        # Check if oscillator (sub-panel indicator)
        elif col in OSCILLATOR_INDICATORS:
            # Get default bounds
            default_bounds = OSCILLATOR_BOUNDS.get(col, None)
            
            # Extract dynamic params from Optuna trial
            optuna_params = _extract_indicator_params_from_optuna(params, col)
            
            # Merge: Optuna params override defaults
            if default_bounds:
                merged_bounds = default_bounds.copy()
                if 'hi' in optuna_params:
                    merged_bounds['hi'] = optuna_params['hi']
                if 'lo' in optuna_params:
                    merged_bounds['lo'] = optuna_params['lo']
                if 'mid' in optuna_params:
                    merged_bounds['mid'] = optuna_params['mid']
            else:
                merged_bounds = optuna_params.copy() if optuna_params else {}
            
            # === ADD STRATEGY ENTRY/EXIT LEVELS TO BOUNDS ===
            # These will be drawn as reference lines on the oscillator panel
            if 'entry_long' in optuna_params:
                merged_bounds['entry_long'] = optuna_params['entry_long']
            if 'entry_short' in optuna_params:
                merged_bounds['entry_short'] = optuna_params['entry_short']
            if 'exit_long' in optuna_params:
                merged_bounds['exit_long'] = optuna_params['exit_long']
            if 'exit_short' in optuna_params:
                merged_bounds['exit_short'] = optuna_params['exit_short']
            
            # === DPO SYMMETRIC TRIGGER LINES ===
            # dpo_trigger: +value for LONG, -value for SHORT
            if col == 'dpo' and 'dpo_trigger' in optuna_params:
                trigger = optuna_params['dpo_trigger']
                merged_bounds['dpo_long'] = trigger       # +trigger (LONG entry)
                merged_bounds['dpo_short'] = -trigger     # -trigger (SHORT entry)
            if col == 'dpo' and 'dpo_exit' in optuna_params:
                exit_val = optuna_params['dpo_exit']
                # Mirrored: LONG exits at dpo_exit, SHORT exits at -dpo_exit
                merged_bounds['dpo_exit_long'] = exit_val    # LONG exit (e.g. -0.75)
                merged_bounds['dpo_exit_short'] = -exit_val  # SHORT exit (e.g. +0.75)
            
            # === ADX THRESHOLD LINE ===
            # adx_threshold: minimum ADX for entry (same for LONG and SHORT)
            if col == 'adx' and 'adx_threshold' in optuna_params:
                merged_bounds['adx_threshold'] = optuna_params['adx_threshold']
            
            # === RSI ENTRY LEVELS ===
            # rsi_long: RSI level for LONG entry (cross UP)
            # rsi_short: RSI level for SHORT entry (cross DOWN)
            if col == 'rsi' and 'rsi_long' in optuna_params:
                merged_bounds['rsi_long'] = optuna_params['rsi_long']
            if col == 'rsi' and 'rsi_short' in optuna_params:
                merged_bounds['rsi_short'] = optuna_params['rsi_short']
            
            # === DPO CYCLE LEVELS (Strategy 6) ===
            # dpo_extreme: normalized DPO level for zones [-10, +10]
            # RSA = +extreme, RSM = +extreme-3, 0 = neutral, RIM = -extreme+3, RIB = -extreme
            if col == 'dpo' and params and 'dpo_extreme' in params:
                dpo_ext = params['dpo_extreme']
                merged_bounds['dpo_rsa'] = dpo_ext          # Upper High (+8) - Euphoria
                merged_bounds['dpo_rsm'] = dpo_ext - 3.0    # Upper Mid (+5) - Distribution
                merged_bounds['dpo_zero'] = 0.0             # Neutral line
                merged_bounds['dpo_rim'] = -dpo_ext + 3.0   # Lower Mid (-5) - Accumulation
                merged_bounds['dpo_rib'] = -dpo_ext         # Lower Low (-8) - Panic
            
            # === MFI THRESHOLD LEVELS (Strategy 6) ===
            # mfi_threshold: symmetric around 50 (e.g., 70 -> zones at 70, 50, 30)
            if col == 'mfi' and params and 'mfi_threshold' in params:
                mfi_thresh = params['mfi_threshold']
                merged_bounds['mfi_high'] = mfi_thresh          # High zone (ZA=70) - Overbought
                merged_bounds['mfi_mid'] = 50.0                 # Equilibrium (50)
                merged_bounds['mfi_low'] = 100.0 - mfi_thresh   # Low zone (ZB=30) - Oversold
            
            # === Z-SCORE ENTRY LEVELS (Strategy 13 Mean Reversion) ===
            # z_entry: Z-Score threshold for entry (symmetric for LONG/SHORT)
            if col == 'zscore' and params and 'z_entry' in params:
                z_entry = params['z_entry']
                merged_bounds['z_entry_long'] = -z_entry   # LONG entry (Z < -z_entry)
                merged_bounds['z_entry_short'] = z_entry   # SHORT entry (Z > z_entry)
            if col == 'zscore' and params and 'tp_z_return' in params:
                tp_z = params['tp_z_return']
                merged_bounds['z_tp_long'] = -tp_z   # LONG TP (Z >= -tp_z_return)
                merged_bounds['z_tp_short'] = tp_z   # SHORT TP (Z <= tp_z_return)
            
            # === Z-SCORE RANGE LEVELS (Strategy 13 v2 Mean Reversion) ===
            # z_long_min, z_long_max: Range for LONG entry (negative values)
            # z_short_min, z_short_max: Range for SHORT entry (positive values)
            if col == 'zscore' and params:
                if 'z_long_min' in params:
                    merged_bounds['z_long_min'] = params['z_long_min']   # e.g. -2.5
                if 'z_long_max' in params:
                    merged_bounds['z_long_max'] = params['z_long_max']   # e.g. -1.5
                if 'z_short_min' in params:
                    merged_bounds['z_short_min'] = params['z_short_min'] # e.g. +1.5
                if 'z_short_max' in params:
                    merged_bounds['z_short_max'] = params['z_short_max'] # e.g. +2.5
                # Z-Score entry threshold (Strategy 13 ER version)
                if 'z_entry_threshold' in params:
                    z_thresh = params['z_entry_threshold']
                    merged_bounds['z_entry_long'] = z_thresh         # e.g. -2.0 (LONG entry)
                    merged_bounds['z_entry_short'] = abs(z_thresh)   # e.g. +2.0 (SHORT entry)
            
            # === EFFICIENCY RATIO THRESHOLD (Strategy 13 ER) ===
            if col in ('er', 'er_kaufman') and params and 'er_threshold' in params:
                merged_bounds['er_threshold'] = params['er_threshold']  # e.g. 0.3
            
            # Build dynamic panel name with period if available
            panel_name = col.upper()
            if 'period' in optuna_params:
                panel_name = f"{col.upper()} ({optuna_params['period']})"
            
            sub_panels.append({
                "col": col,
                "name": panel_name,
                "color": INDICATOR_COLORS.get(col, "#60a5fa"),
                "type": "histogram" if col in HISTOGRAM_INDICATORS else "line",
                "bounds": merged_bounds,
                "optuna_params": optuna_params  # Store for reference
            })
    
    return {"overlays": overlays, "sub_panels": sub_panels}


# =============================================================================
# DYNAMIC HTML GENERATOR
# =============================================================================

def _generate_hft_html(
    candle_data: dict,
    indicators: dict,
    trades: dict,
    config: dict
) -> str:
    """
    Generate HTML with dynamic sub-panel generation.
    No fixed panel IDs - everything is data-driven.
    
    ENHANCED FEATURES (v4.0):
    - Cross-panel markers: Trade markers on indicator sub-panels
    - Dot-style centered markers (circles, inBar position)
    - Dynamic Optuna parameter sync for bounds/labels
    """
    
    candles_json = _dumps(candle_data)
    indicators_json = _dumps(indicators)
    trades_json = _dumps(trades)
    
    activo = str(config.get("activo", ""))
    combo = str(config.get("combo", ""))
    total_trades = int(config.get("total_trades", 0))
    winrate = float(config.get("winrate", 0))
    pnl_neto = float(config.get("pnl_neto", 0))
    pnl_class = "pos" if pnl_neto >= 0 else "neg"
    score = float(config.get("score", 0))
    
    html = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>MODELOX - Dynamic Chart</title>
<script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
<script>
// Fallback check - if library didn't load, show error
window.addEventListener('DOMContentLoaded', function() {
  if (typeof LightweightCharts === 'undefined') {
    document.body.innerHTML = '<div style="color:#ef4444;padding:40px;font-family:system-ui;text-align:center;"><h2>Error: Could not load chart library</h2><p>CDN may be blocked. Check your internet connection.</p></div>';
  }
});
</script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:100%;height:100%;background:#0b1220;font-family:'SF Pro Display',system-ui,-apple-system,sans-serif;overflow:hidden;touch-action:none}
.c{display:flex;flex-direction:column;height:100vh;padding:8px;gap:4px}
.h{display:flex;justify-content:space-between;align-items:center;padding:8px 16px;background:linear-gradient(180deg,rgba(15,23,42,.98) 0%,rgba(15,23,42,.92) 100%);border-radius:6px;border:1px solid rgba(148,163,184,.1)}
.h .a{color:#22d3ee;font-size:18px;font-weight:700;letter-spacing:-.5px}
.h .t{color:#94a3b8;font-size:12px;font-weight:500}
.h .info{display:flex;gap:20px;align-items:center}
.h .stat{display:flex;flex-direction:column;align-items:flex-end}
.h .stat-label{color:#64748b;font-size:9px;text-transform:uppercase;letter-spacing:.5px}
.h .stat-val{color:#e2e8f0;font-size:13px;font-weight:600}
.h .stat-val.pos{color:#22c55e}
.h .stat-val.neg{color:#ef4444}
.p{flex:1;display:flex;flex-direction:column;gap:3px;min-height:0}
.m{position:relative;min-height:280px;border-radius:4px;overflow:hidden}
.sub{position:relative;min-height:60px;border-radius:4px;overflow:hidden}
.l{position:absolute;top:6px;left:10px;z-index:100;background:rgba(15,23,42,.92);color:#e2e8f0;font-size:10px;font-weight:700;padding:3px 10px;border-radius:4px;border:1px solid rgba(148,163,184,.1);letter-spacing:.3px;text-transform:uppercase}
#tt{position:fixed;display:none;background:linear-gradient(180deg,rgba(15,23,42,.98) 0%,rgba(10,18,32,.99) 100%);border:1px solid rgba(71,85,105,.6);border-radius:10px;padding:14px 18px;color:#e2e8f0;font-size:12px;z-index:999999;pointer-events:none;min-width:220px;backdrop-filter:blur(12px);box-shadow:0 8px 32px rgba(0,0,0,.6)}
.tt-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid rgba(148,163,184,.15)}
.tt-type{font-weight:700;font-size:13px;text-transform:uppercase;letter-spacing:.5px}
.tt-type.long{color:#3b82f6}
.tt-type.short{color:#a855f7}
.tt-badge{padding:2px 8px;border-radius:4px;font-size:9px;font-weight:600;text-transform:uppercase}
.tt-badge.win{background:rgba(34,197,94,.2);color:#22c55e}
.tt-badge.loss{background:rgba(239,68,68,.2);color:#ef4444}
.tt-row{display:flex;justify-content:space-between;align-items:center;margin:6px 0}
.tt-label{color:#94a3b8;font-size:11px}
.tt-val{font-weight:600;font-size:12px;font-family:'SF Mono',ui-monospace,monospace}
.tt-val.pos{color:#22c55e}
.tt-val.neg{color:#ef4444}
.tt-pnl{margin-top:10px;padding-top:8px;border-top:1px solid rgba(148,163,184,.15);display:flex;justify-content:space-between;align-items:center}
.tt-pnl-label{color:#94a3b8;font-size:11px;font-weight:600}
.tt-pnl-val{font-size:16px;font-weight:700;font-family:'SF Mono',ui-monospace,monospace}
#ohlc{position:absolute;top:6px;right:10px;z-index:100;display:flex;gap:12px;background:rgba(15,23,42,.92);padding:4px 12px;border-radius:4px;border:1px solid rgba(148,163,184,.1);font-size:11px;font-family:'SF Mono',ui-monospace,monospace}
.ohlc-item{display:flex;gap:4px}
.ohlc-label{color:#64748b}
.ohlc-val{font-weight:600}
.ohlc-val.up{color:#22c55e}
.ohlc-val.down{color:#ef4444}
.tv-zoom-container{position:absolute;bottom:20px;left:50%;transform:translateX(-50%);z-index:10001;display:flex;gap:6px}
.tv-zoom-btn{width:32px;height:32px;background:rgba(15,23,42,.85);border:1px solid rgba(148,163,184,.25);border-radius:50%;color:#94a3b8;font-size:16px;font-weight:600;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all .15s;backdrop-filter:blur(4px)}
.tv-zoom-btn:hover{background:rgba(30,41,59,.95);color:#e2e8f0;border-color:#60a5fa}
</style>
</head>
<body>
<div class="c">
<div class="h">
<div style="display:flex;align-items:center;gap:16px">
<span class="a">ACTIVO_PLACEHOLDER</span>
<span class="t">COMBO_PLACEHOLDER</span>
</div>
<div class="info">
<div class="stat"><span class="stat-label">Trades</span><span class="stat-val">TRADES_PLACEHOLDER</span></div>
<div class="stat"><span class="stat-label">Win Rate</span><span class="stat-val">WINRATE_PLACEHOLDER%</span></div>
<div class="stat"><span class="stat-label">PnL Neto</span><span class="stat-val PNL_CLASS_PLACEHOLDER">$PNL_PLACEHOLDER</span></div>
<div class="stat"><span class="stat-label">Score</span><span class="stat-val pos">SCORE_PLACEHOLDER</span></div>
</div>
</div>
<div class="p" id="ct"></div>
</div>
<div id="tt"></div>

<script>
(function(){
'use strict';

try {

const D=CANDLES_JSON_PLACEHOLDER;
const I=INDICATORS_JSON_PLACEHOLDER;
const T=TRADES_JSON_PLACEHOLDER;

// Validate data loaded correctly
if (!D || !D.t || D.t.length === 0) {
  console.error('No candle data available');
  document.body.innerHTML = '<div style="color:#fbbf24;padding:40px;font-family:system-ui;text-align:center;"><h2>No data to display</h2><p>No candle data was generated for this trial.</p></div>';
  return;
}

const dq=(v,f)=>v/f;
const ct=document.getElementById('ct');
const charts=[];
let syncingCharts=false;

const baseOpts={
layout:{background:{type:'solid',color:'#0b1220'},textColor:'#94a3b8',fontSize:10,fontFamily:"'SF Pro Display',system-ui"},
grid:{vertLines:{color:'rgba(148,163,184,.04)'},horzLines:{color:'rgba(148,163,184,.04)'}},
crosshair:{mode:LightweightCharts.CrosshairMode.Normal,vertLine:{color:'rgba(34,211,238,.7)',width:1,style:LightweightCharts.LineStyle.Dashed,labelBackgroundColor:'#1e293b',labelVisible:true},horzLine:{color:'rgba(34,211,238,.5)',width:1,style:LightweightCharts.LineStyle.Dashed,labelBackgroundColor:'#1e293b',labelVisible:true}},
timeScale:{borderColor:'rgba(148,163,184,.1)',timeVisible:true,secondsVisible:false,rightOffset:8,barSpacing:8,minBarSpacing:2,fixLeftEdge:false,fixRightEdge:false,lockVisibleTimeRangeOnResize:true,autoScale:true,visible:false},
rightPriceScale:{borderColor:'rgba(148,163,184,.1)',scaleMargins:{top:.1,bottom:.1},autoScale:true,alignLabels:true,borderVisible:true,entireTextOnly:false},
// Desactivamos zoom con rueda/pinch; se permite zoom sólo
// arrastrando ejes (time/price) y con los botones + y -.
handleScale:{axisPressedMouseMove:{time:true,price:true},mouseWheel:false,pinch:false},
// Mantenemos el scroll horizontal/vertical para desplazarse por el gráfico.
handleScroll:{mouseWheel:true,pressedMouseMove:true,horzTouchDrag:true,vertTouchDrag:true},
kineticScroll:{touch:true,mouse:true},
localization:{
  // Force UTC display for consistent time across all browsers
  timeFormatter:(ts)=>{
    const d=new Date(ts*1000);
    const mo=String(d.getUTCMonth()+1).padStart(2,'0');
    const dy=String(d.getUTCDate()).padStart(2,'0');
    const hr=String(d.getUTCHours()).padStart(2,'0');
    const mn=String(d.getUTCMinutes()).padStart(2,'0');
    return mo+'/'+dy+' '+hr+':'+mn;
  },
  dateFormat:'yyyy-MM-dd'
}
};

// === DYNAMIC PANEL CREATION ===
function mkPanel(id,lbl,isMain,heightPct){
try{
const p=document.createElement('div');
p.className=isMain?'m':'sub';
p.id=id;
p.style.flex=isMain?'6':'1.2';
const l=document.createElement('div');
l.className='l';
l.textContent=lbl;
p.appendChild(l);

if(isMain){
  const ohlc=document.createElement('div');
  ohlc.id='ohlc';
  ohlc.innerHTML='<div class="ohlc-item"><span class="ohlc-label">T</span><span class="ohlc-val" id="tv" style="color:#22d3ee">-</span></div><div class="ohlc-item"><span class="ohlc-label">O</span><span class="ohlc-val" id="ov">-</span></div><div class="ohlc-item"><span class="ohlc-label">H</span><span class="ohlc-val" id="hv">-</span></div><div class="ohlc-item"><span class="ohlc-label">L</span><span class="ohlc-val" id="lv">-</span></div><div class="ohlc-item"><span class="ohlc-label">C</span><span class="ohlc-val" id="cv">-</span></div>';
  p.appendChild(ohlc);
  const zoomDiv=document.createElement('div');
  zoomDiv.className='tv-zoom-container';
  zoomDiv.innerHTML='<button class="tv-zoom-btn" id="zoomIn" title="Zoom In">+</button><button class="tv-zoom-btn" id="zoomOut" title="Zoom Out">-</button>';
  p.appendChild(zoomDiv);
}
ct.appendChild(p);

const opts={...baseOpts,width:p.clientWidth,height:p.clientHeight};
// Don't set timeScale visible here - will be set in SINGLE TIMESCALE section
const ch=LightweightCharts.createChart(p,opts);

charts.push({ch,p,id,label:lbl});
return ch;
}catch(e){
console.error('Failed to create panel:',lbl,e);
return null;
}
}

// === CALCULATE PANEL HEIGHTS ===
const numSubPanels=(I.sub_panels?I.sub_panels.length:0)+(D.vol&&D.vol.length>0?1:0);
const mainFlex=6;
const subFlex=numSubPanels>0?1.2:0;

// === MAIN PRICE CHART (Marker Stability Optimized) ===
try {
  const mc=mkPanel('mc','PRECIO',true);
  if(!mc)throw new Error('Main chart creation failed');
  
  // Configure candlestick series with options that prevent marker recalculation during scroll/zoom
  const cs=mc.addCandlestickSeries({
    upColor:'#22c55e',
    downColor:'#ef4444',
    borderUpColor:'#16a34a',
    borderDownColor:'#dc2626',
    wickUpColor:'#22c55e',
    wickDownColor:'#ef4444',
    priceFormat:{type:'price',precision:2,minMove:0.01},
    // Disable dynamic updates that trigger price scale recalculation
    lastValueVisible:false,
    priceLineVisible:false
  });

  const f=D.f;
  const cData=[];
  
  // Build candle timestamp Set for O(1) marker validation
  const candleTimeSet=new Set();
  
  if(D.t && D.t.length>0){
    for(let i=0;i<D.t.length;i++){
      const t=D.t[i];
      cData.push({time:t,open:dq(D.o[i],f),high:dq(D.h[i],f),low:dq(D.l[i],f),close:dq(D.c[i],f)});
      candleTimeSet.add(t);
    }
    cs.setData(cData);

    // Stable trade markers on candle chart.
    // Guardamos el array y lo re-aplicamos en cada cambio de rango visible
    // para evitar cualquier parpadeo durante scroll/zoom.
    if(T.m && T.m.length>0){
      try{
        const candleMarkers=T.m.map(m=>({
          time:m.time,
          position:m.position||'inBar',
          color:m.color,
          shape:m.shape||'circle',
          text:m.text||'',
          size:m.size||2
        }));
        cs.setMarkers(candleMarkers);
        mc.timeScale().subscribeVisibleTimeRangeChange(()=>{
          cs.setMarkers(candleMarkers);
        });
      }catch(e){console.warn('Markers error:',e);}    
    }
  }

  // === OVERLAY INDICATORS (on main chart) - NULL-AWARE ===
  if(I.overlays && Array.isArray(I.overlays)){
    I.overlays.forEach(ov=>{
      try{
        if(ov.t&&ov.t.length>0&&ov.v&&ov.v.length>0){
          const ls=mc.addLineSeries({color:ov.color||'#fbbf24',lineWidth:1.5,priceLineVisible:false,lastValueVisible:true,crosshairMarkerVisible:false,lineStyle:0});
          // STRICT ALIGNMENT: Filter out null values but maintain timestamp alignment
          const ovData=[];
          for(let i=0;i<ov.t.length;i++){
            if(ov.v[i]!==null){
              ovData.push({time:ov.t[i],value:dq(ov.v[i],ov.f)});
            }
          }
          if(ovData.length>0)ls.setData(ovData);
        }
      }catch(e){console.warn('Overlay error:',e);}
    });
  }
} catch(e) {
  console.error('CRITICAL: Main chart failed:',e);
}

// === VOLUME PANEL ===
try {
  if(D.vol&&D.vol.length>0&&D.t&&D.t.length>0){
    const vc=mkPanel('vc','VOLUMEN',false);
    if(vc){
      vc.priceScale('right').applyOptions({autoScale:true});
      const vs=vc.addHistogramSeries({priceFormat:{type:'volume'},priceLineVisible:false,lastValueVisible:false});
      const vData=D.t.map((t,i)=>({time:t,value:D.vol[i],color:D.c[i]>=D.o[i]?'rgba(34,197,94,.5)':'rgba(239,68,68,.5)'}));
      vs.setData(vData);
    }
  }
} catch(e) {
  console.warn('Volume panel error:',e);
}

// === DYNAMIC SUB-PANELS (auto-generated from detected indicators) ===
if(I.sub_panels && Array.isArray(I.sub_panels)){
  I.sub_panels.forEach((panel,idx)=>{
    try{
      if(!panel.data||!panel.data.t||panel.data.t.length===0)return;
      
      const panelId='sp_'+idx;
      const panelLabel=panel.name.toUpperCase();
      const pc=mkPanel(panelId,panelLabel,false);
      if(!pc)return;
      
      pc.priceScale('right').applyOptions({scaleMargins:{top:.1,bottom:.1},autoScale:true});
      
      // Histogram or Line based on indicator type
      // Build timestamp set for this panel for marker validation
      // Use candle timestamps as reference since indicators are now strictly aligned
      const panelTimeSet=new Set(D.t);
      let mainSeries=null;
      
      // STRICT ALIGNMENT: Filter out null values but maintain timestamp alignment
      const validData=[];
      for(let i=0;i<panel.data.t.length;i++){
        if(panel.data.v[i]!==null){
          validData.push({time:panel.data.t[i],value:dq(panel.data.v[i],panel.data.f)});
        }
      }
      if(validData.length===0)return;
      
      if(panel.type==='histogram'){
        const hs=pc.addHistogramSeries({priceLineVisible:false,lastValueVisible:false});
        const hData=validData.map(d=>({
          time:d.time,
          value:d.value,
          color:d.value>=0?'rgba(34,197,94,.7)':'rgba(239,68,68,.7)'
        }));
        hs.setData(hData);
        mainSeries=hs;
      }else{
        const ls=pc.addLineSeries({color:panel.color||'#60a5fa',lineWidth:2,priceLineVisible:false,lastValueVisible:true});
        ls.setData(validData);
        mainSeries=ls;
        
        // Reference lines for bounded oscillators (DYNAMIC FROM OPTUNA)
        // Use validData timestamps for reference lines (already filtered)
        if(panel.bounds){
          const b=panel.bounds;
          // Overbought line (red dashed)
          if(b.hi!==undefined){
            const hiLine=pc.addLineSeries({color:'rgba(239,68,68,.6)',lineWidth:1.5,lineStyle:2,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            hiLine.setData(validData.map(d=>({time:d.time,value:b.hi})));
          }
          // Oversold line (green dashed)
          if(b.lo!==undefined){
            const loLine=pc.addLineSeries({color:'rgba(34,197,94,.6)',lineWidth:1.5,lineStyle:2,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            loLine.setData(validData.map(d=>({time:d.time,value:b.lo})));
          }
          // Midline (gray dotted)
          if(b.mid!==undefined){
            const midLine=pc.addLineSeries({color:'rgba(148,163,184,.3)',lineWidth:1,lineStyle:1,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            midLine.setData(validData.map(d=>({time:d.time,value:b.mid})));
          }
          // === STRATEGY ENTRY/EXIT LEVELS (Pendulum Visualization) ===
          // Entry Long level (cyan solid)
          if(b.entry_long!==undefined){
            const entryLongLine=pc.addLineSeries({color:'rgba(34,211,238,.8)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            entryLongLine.setData(validData.map(d=>({time:d.time,value:b.entry_long})));
          }
          // Entry Short level (magenta solid)
          if(b.entry_short!==undefined){
            const entryShortLine=pc.addLineSeries({color:'rgba(236,72,153,.8)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            entryShortLine.setData(validData.map(d=>({time:d.time,value:b.entry_short})));
          }
          // Exit Long level (orange dotted)
          if(b.exit_long!==undefined){
            const exitLongLine=pc.addLineSeries({color:'rgba(251,146,60,.7)',lineWidth:1.5,lineStyle:1,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            exitLongLine.setData(validData.map(d=>({time:d.time,value:b.exit_long})));
          }
          // Exit Short level (amber dotted)
          if(b.exit_short!==undefined){
            const exitShortLine=pc.addLineSeries({color:'rgba(251,191,36,.7)',lineWidth:1.5,lineStyle:1,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            exitShortLine.setData(validData.map(d=>({time:d.time,value:b.exit_short})));
          }
          // === DPO SYMMETRIC TRIGGER LINES ===
          // DPO Long entry level (cyan solid)
          if(b.dpo_long!==undefined){
            const dpoLongLine=pc.addLineSeries({color:'rgba(34,211,238,.8)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            dpoLongLine.setData(validData.map(d=>({time:d.time,value:b.dpo_long})));
          }
          // DPO Short entry level (magenta solid)
          if(b.dpo_short!==undefined){
            const dpoShortLine=pc.addLineSeries({color:'rgba(236,72,153,.8)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            dpoShortLine.setData(validData.map(d=>({time:d.time,value:b.dpo_short})));
          }
          // DPO Long exit level (orange dashed)
          if(b.dpo_exit_long!==undefined){
            const dpoExitLongLine=pc.addLineSeries({color:'rgba(251,146,60,.7)',lineWidth:1.5,lineStyle:2,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            dpoExitLongLine.setData(validData.map(d=>({time:d.time,value:b.dpo_exit_long})));
          }
          // DPO Short exit level (amber dashed)
          if(b.dpo_exit_short!==undefined){
            const dpoExitShortLine=pc.addLineSeries({color:'rgba(251,191,36,.7)',lineWidth:1.5,lineStyle:2,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            dpoExitShortLine.setData(validData.map(d=>({time:d.time,value:b.dpo_exit_short})));
          }
          // === ADX THRESHOLD LINE ===
          // ADX minimum threshold (yellow solid)
          if(b.adx_threshold!==undefined){
            const adxThreshLine=pc.addLineSeries({color:'rgba(250,204,21,.9)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            adxThreshLine.setData(validData.map(d=>({time:d.time,value:b.adx_threshold})));
          }
          // === RSI ENTRY LEVELS ===
          // RSI LONG entry level (cyan solid)
          if(b.rsi_long!==undefined){
            const rsiLongLine=pc.addLineSeries({color:'rgba(34,211,238,.8)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            rsiLongLine.setData(validData.map(d=>({time:d.time,value:b.rsi_long})));
          }
          // RSI SHORT entry level (magenta solid)
          if(b.rsi_short!==undefined){
            const rsiShortLine=pc.addLineSeries({color:'rgba(236,72,153,.8)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            rsiShortLine.setData(validData.map(d=>({time:d.time,value:b.rsi_short})));
          }
          // === DPO CYCLE ZONE LEVELS (Strategy 6) ===
          // DPO RSA - Upper High extreme (red solid) - EUPHORIA zone above
          if(b.dpo_rsa!==undefined){
            const dpoRsaLine=pc.addLineSeries({color:'rgba(239,68,68,.9)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            dpoRsaLine.setData(validData.map(d=>({time:d.time,value:b.dpo_rsa})));
          }
          // DPO RSM - Upper Mid zone (orange dashed) - DISTRIBUTION zone
          if(b.dpo_rsm!==undefined){
            const dpoRsmLine=pc.addLineSeries({color:'rgba(251,146,60,.7)',lineWidth:1.5,lineStyle:2,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            dpoRsmLine.setData(validData.map(d=>({time:d.time,value:b.dpo_rsm})));
          }
          // DPO ZERO - Neutral line (white/gray dashed)
          if(b.dpo_zero!==undefined){
            const dpoZeroLine=pc.addLineSeries({color:'rgba(148,163,184,.6)',lineWidth:1,lineStyle:1,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            dpoZeroLine.setData(validData.map(d=>({time:d.time,value:b.dpo_zero})));
          }
          // DPO RIM - Lower Mid zone (cyan dashed) - ACCUMULATION zone
          if(b.dpo_rim!==undefined){
            const dpoRimLine=pc.addLineSeries({color:'rgba(34,211,238,.7)',lineWidth:1.5,lineStyle:2,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            dpoRimLine.setData(validData.map(d=>({time:d.time,value:b.dpo_rim})));
          }
          // DPO RIB - Lower Low extreme (green solid) - PANIC zone below
          if(b.dpo_rib!==undefined){
            const dpoRibLine=pc.addLineSeries({color:'rgba(34,197,94,.9)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            dpoRibLine.setData(validData.map(d=>({time:d.time,value:b.dpo_rib})));
          }
          // === MFI THRESHOLD ZONE LEVELS (Strategy 6) ===
          // MFI High zone threshold (magenta solid) - OVERBOUGHT zone above
          if(b.mfi_high!==undefined){
            const mfiHighLine=pc.addLineSeries({color:'rgba(236,72,153,.9)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            mfiHighLine.setData(validData.map(d=>({time:d.time,value:b.mfi_high})));
          }
          // MFI Mid equilibrium (white/gray dashed) - 50 line
          if(b.mfi_mid!==undefined){
            const mfiMidLine=pc.addLineSeries({color:'rgba(148,163,184,.6)',lineWidth:1,lineStyle:1,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            mfiMidLine.setData(validData.map(d=>({time:d.time,value:b.mfi_mid})));
          }
          // MFI Low zone threshold (cyan solid) - OVERSOLD zone below
          if(b.mfi_low!==undefined){
            const mfiLowLine=pc.addLineSeries({color:'rgba(34,211,238,.9)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            mfiLowLine.setData(validData.map(d=>({time:d.time,value:b.mfi_low})));
          }
          // === Z-SCORE RANGE LEVELS (Strategy 13 v2 Mean Reversion) ===
          // Z-Score LONG range (negative values) - green band
          if(b.z_long_min!==undefined){
            const zLongMinLine=pc.addLineSeries({color:'rgba(34,197,94,.9)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            zLongMinLine.setData(validData.map(d=>({time:d.time,value:b.z_long_min})));
          }
          if(b.z_long_max!==undefined){
            const zLongMaxLine=pc.addLineSeries({color:'rgba(34,197,94,.6)',lineWidth:1.5,lineStyle:2,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            zLongMaxLine.setData(validData.map(d=>({time:d.time,value:b.z_long_max})));
          }
          // Z-Score SHORT range (positive values) - red band
          if(b.z_short_min!==undefined){
            const zShortMinLine=pc.addLineSeries({color:'rgba(239,68,68,.6)',lineWidth:1.5,lineStyle:2,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            zShortMinLine.setData(validData.map(d=>({time:d.time,value:b.z_short_min})));
          }
          if(b.z_short_max!==undefined){
            const zShortMaxLine=pc.addLineSeries({color:'rgba(239,68,68,.9)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            zShortMaxLine.setData(validData.map(d=>({time:d.time,value:b.z_short_max})));
          }
          // Z-Score entry/TP levels (legacy v1 support)
          if(b.z_entry_long!==undefined){
            const zEntryLongLine=pc.addLineSeries({color:'rgba(34,211,238,.8)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            zEntryLongLine.setData(validData.map(d=>({time:d.time,value:b.z_entry_long})));
          }
          if(b.z_entry_short!==undefined){
            const zEntryShortLine=pc.addLineSeries({color:'rgba(236,72,153,.8)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            zEntryShortLine.setData(validData.map(d=>({time:d.time,value:b.z_entry_short})));
          }
          if(b.z_tp_long!==undefined){
            const zTpLongLine=pc.addLineSeries({color:'rgba(251,146,60,.7)',lineWidth:1.5,lineStyle:1,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            zTpLongLine.setData(validData.map(d=>({time:d.time,value:b.z_tp_long})));
          }
          if(b.z_tp_short!==undefined){
            const zTpShortLine=pc.addLineSeries({color:'rgba(251,191,36,.7)',lineWidth:1.5,lineStyle:1,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            zTpShortLine.setData(validData.map(d=>({time:d.time,value:b.z_tp_short})));
          }
          // === Z-SCORE RANGE LEVELS (Strategy 13 v2 Mean Reversion) ===
          // Z-Score LONG range (negative values) - green band
          if(b.z_long_min!==undefined){
            const zLongMinLine=pc.addLineSeries({color:'rgba(34,197,94,.9)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            zLongMinLine.setData(validData.map(d=>({time:d.time,value:b.z_long_min})));
          }
          if(b.z_long_max!==undefined){
            const zLongMaxLine=pc.addLineSeries({color:'rgba(34,197,94,.6)',lineWidth:1.5,lineStyle:2,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            zLongMaxLine.setData(validData.map(d=>({time:d.time,value:b.z_long_max})));
          }
          // Z-Score SHORT range (positive values) - red band
          if(b.z_short_min!==undefined){
            const zShortMinLine=pc.addLineSeries({color:'rgba(239,68,68,.6)',lineWidth:1.5,lineStyle:2,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            zShortMinLine.setData(validData.map(d=>({time:d.time,value:b.z_short_min})));
          }
          if(b.z_short_max!==undefined){
            const zShortMaxLine=pc.addLineSeries({color:'rgba(239,68,68,.9)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            zShortMaxLine.setData(validData.map(d=>({time:d.time,value:b.z_short_max})));
          }
          // Z-Score entry/TP levels (legacy v1 support)
          if(b.z_entry_long!==undefined){
            const zEntryLongLine=pc.addLineSeries({color:'rgba(34,211,238,.8)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            zEntryLongLine.setData(validData.map(d=>({time:d.time,value:b.z_entry_long})));
          }
          if(b.z_entry_short!==undefined){
            const zEntryShortLine=pc.addLineSeries({color:'rgba(236,72,153,.8)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            zEntryShortLine.setData(validData.map(d=>({time:d.time,value:b.z_entry_short})));
          }
          if(b.z_tp_long!==undefined){
            const zTpLongLine=pc.addLineSeries({color:'rgba(251,146,60,.7)',lineWidth:1.5,lineStyle:1,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            zTpLongLine.setData(validData.map(d=>({time:d.time,value:b.z_tp_long})));
          }
          if(b.z_tp_short!==undefined){
            const zTpShortLine=pc.addLineSeries({color:'rgba(251,191,36,.7)',lineWidth:1.5,lineStyle:1,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            zTpShortLine.setData(validData.map(d=>({time:d.time,value:b.z_tp_short})));
          }
          // === EFFICIENCY RATIO THRESHOLD (Strategy 13 ER) ===
          if(b.er_threshold!==undefined){
            const erThreshLine=pc.addLineSeries({color:'rgba(250,204,21,.9)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            erThreshLine.setData(validData.map(d=>({time:d.time,value:b.er_threshold})));
          }
        }
        
        // Zero line for unbounded oscillators (like MACD, zscore)
        if(panel.zero_line){
          const zl=pc.addLineSeries({color:'rgba(148,163,184,.3)',lineWidth:1,lineStyle:1,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
          zl.setData(validData.map(d=>({time:d.time,value:0})));
        }
      }
      
      // Sub-panels ya muestran la información del precio/indicador;
      // REPINTADO ROBUSTO DE MARCADORES EN LOS SUB-PANELES
      // Dibujamos los mismos marcadores de trade también sobre el indicador
      // principal del panel para que nunca desaparezcan al hacer scroll.
      if(mainSeries && T.m && T.m.length>0){
        try{
          const panelMarkers=T.m.map(m=>({
            time:m.time,
            position:m.position||'inBar',
            color:m.color,
            shape:m.shape||'circle',
            text:m.text||'',
            size:m.size||2
          }));
          mainSeries.setMarkers(panelMarkers);
          pc.timeScale().subscribeVisibleTimeRangeChange(()=>{
            mainSeries.setMarkers(panelMarkers);
          });
        }catch(e){console.warn('Sub-panel markers error:',panel.name,e);}
      }
    }catch(e){
      console.warn('Sub-panel error:',panel.name,e);
    }
  });
}

// === SINGLE TIMESCALE: Only LAST panel shows time axis ===
try {
  if(charts.length>1){
    // Hide time axis on all panels except the last one
    for(let i=0;i<charts.length-1;i++){
      if(charts[i].ch)charts[i].ch.timeScale().applyOptions({visible:false});
    }
    // Show time axis only on the last (bottom) panel
    charts[charts.length-1].ch.timeScale().applyOptions({visible:true});
  }else if(charts.length===1){
    charts[0].ch.timeScale().applyOptions({visible:true});
  }
} catch(e) {
  console.warn('TimeScale config error:',e);
}

// === CHART SYNCHRONIZATION ===
try {
  if(charts.length>1){
    const masterTS=charts[0].ch.timeScale();
    charts.forEach(({ch},idx)=>{
      try{
        if(idx===0||!ch)return;
        const slaveTS=ch.timeScale();
        masterTS.subscribeVisibleLogicalRangeChange(range=>{
          try{
            if(syncingCharts||!range)return;
            syncingCharts=true;
            slaveTS.setVisibleLogicalRange(range);
            syncingCharts=false;
          }catch(e){syncingCharts=false;}
        });
        slaveTS.subscribeVisibleLogicalRangeChange(range=>{
          try{
            if(syncingCharts||!range)return;
            syncingCharts=true;
            masterTS.setVisibleLogicalRange(range);
            syncingCharts=false;
          }catch(e){syncingCharts=false;}
        });
      }catch(e){console.warn('Sync error:',e);}
    });
  }
} catch(e) {
  console.warn('Sync setup error:',e);
}

// === CROSSHAIR SYNC ACROSS ALL PANELS (Improved) ===
try {
  if(charts.length>1){
    // Store first series of each chart for crosshair sync
    const chartSeries=charts.map(({ch})=>{
      try{
        // Get first series from each chart for price reference
        return ch._private__seriesMap?Array.from(ch._private__seriesMap.values())[0]:null;
      }catch(e){return null;}
    });
    
    let syncing=false;
    
    const syncCrosshair=(sourceIdx,param)=>{
      if(syncing)return;
      syncing=true;
      
      charts.forEach(({ch},idx)=>{
        if(idx===sourceIdx||!ch)return;
        try{
          if(param.time){
            // Move crosshair to same time on all charts
            const series=chartSeries[idx];
            if(series){
              const data=series.data();
              if(data&&data.length>0){
                // Find the bar at this time
                const bar=data.find(d=>d.time===param.time);
                if(bar&&bar.value!==undefined){
                  ch.setCrosshairPosition(bar.value,param.time,series);
                }else if(bar&&bar.close!==undefined){
                  ch.setCrosshairPosition(bar.close,param.time,series);
                }
              }
            }
          }else{
            ch.clearCrosshairPosition();
          }
        }catch(e){/* Ignore crosshair errors */}
      });
      
      syncing=false;
    };
    
    charts.forEach(({ch},idx)=>{
      if(ch)ch.subscribeCrosshairMove(param=>syncCrosshair(idx,param));
    });
  }
} catch(e) {
  console.warn('Crosshair sync error:',e);
}

// === TRADE MAP FOR TOOLTIPS ===
const globalTradeMap={};
if(T.i&&Array.isArray(T.i)){
  T.i.forEach(t=>{
    if(t&&t.time!==undefined)globalTradeMap[t.time]=t;
  });
}

// === TOOLTIP SYSTEM (Unified across all panels) ===
try {
  const tt=document.getElementById('tt');
  const candleMap={};
  if(D.t&&D.t.length>0){
    const f=D.f||100;
    for(let i=0;i<D.t.length;i++){
      candleMap[D.t[i]]={time:D.t[i],open:D.o[i]/f,high:D.h[i]/f,low:D.l[i]/f,close:D.c[i]/f};
    }
  }
  
  // Format Unix timestamp to readable date (UTC to avoid timezone confusion)
  const formatTime=(ts)=>{
    try{
      const d=new Date(ts*1000);
      // Use UTC methods to prevent browser timezone conversion
      const mo=String(d.getUTCMonth()+1).padStart(2,'0');
      const dy=String(d.getUTCDate()).padStart(2,'0');
      const hr=String(d.getUTCHours()).padStart(2,'0');
      const mn=String(d.getUTCMinutes()).padStart(2,'0');
      return mo+'/'+dy+' '+hr+':'+mn+' UTC';
    }catch(e){return '-';}
  };
  
  // Handle crosshair move on ALL panels to update unified tooltip
  const handleCrosshair=(param)=>{
    try{
      if(!param.time){
        if(tt)tt.style.display='none';
        return;
      }
      
      // Update time display in header
      const tvEl=document.getElementById('tv');
      if(tvEl)tvEl.textContent=formatTime(param.time);
      
      // Check for trade at this time
      const tr=globalTradeMap[param.time];
      if(tr&&tr.pnl!==undefined&&param.point){
        const isWin=tr.pnl>=0;
        const pnlSign=isWin?'+':'';
        const pnlCls=isWin?'pos':'neg';
        const typeCls=tr.type==='LONG'?'long':'short';
        const badgeCls=isWin?'win':'loss';
        
        tt.innerHTML='<div class="tt-header"><span class="tt-type '+typeCls+'">'+tr.type+'</span><span class="tt-badge '+badgeCls+'">'+(isWin?'WIN':'LOSS')+'</span></div><div class="tt-row"><span class="tt-label">Entry</span><span class="tt-val">'+(tr.ep?tr.ep.toFixed(2):'-')+'</span></div><div class="tt-row"><span class="tt-label">Exit</span><span class="tt-val">'+(tr.xp?tr.xp.toFixed(2):'-')+'</span></div><div class="tt-row"><span class="tt-label">Qty</span><span class="tt-val">'+(tr.qty?tr.qty.toFixed(4):'-')+'</span></div><div class="tt-row"><span class="tt-label">Comm</span><span class="tt-val">$'+(tr.comm?tr.comm.toFixed(2):'0')+'</span></div><div class="tt-pnl"><span class="tt-pnl-label">PnL Neto</span><span class="tt-pnl-val '+pnlCls+'">'+pnlSign+'$'+(tr.pnl?tr.pnl.toFixed(2):'0')+'</span></div>';
        tt.style.display='block';
        const maxX=window.innerWidth-260;
        const maxY=window.innerHeight-200;
        tt.style.left=Math.min(param.point.x+15,maxX)+'px';
        tt.style.top=Math.min(param.point.y+15,maxY)+'px';
      }else{
        tt.style.display='none';
        // Update OHLC values from candle data
        const candle=candleMap[param.time];
        if(candle){
          const isUp=candle.close>=candle.open;
          const cls=isUp?'up':'down';
          ['ov','hv','lv','cv'].forEach((id,i)=>{
            const el=document.getElementById(id);
            if(el){
              const vals=[candle.open,candle.high,candle.low,candle.close];
              el.textContent=vals[i].toFixed(2);
              el.className='ohlc-val '+cls;
            }
          });
        }
      }
    }catch(e){console.warn('Tooltip error:',e);}
  };
  
  // Subscribe to ALL charts for unified tooltip
  charts.forEach(({ch})=>{
    if(ch)ch.subscribeCrosshairMove(handleCrosshair);
  });
} catch(e) {
  console.warn('Tooltip setup error:',e);
}

// === RESIZE OBSERVER ===
try {
  const ro=new ResizeObserver(()=>{
    charts.forEach(({ch,p})=>{
      if(ch&&p)ch.applyOptions({width:p.clientWidth,height:p.clientHeight});
    });
  });
  charts.forEach(({p})=>{if(p)ro.observe(p);});
} catch(e) {
  console.warn('Resize observer error:',e);
}

// === AUTO-FIT & KEYBOARD SHORTCUTS ===
try {
  setTimeout(()=>{
    // Show BEGINNING of data (first 200 bars visible)
    charts.forEach(({ch})=>{
      if(ch){
        const ts=ch.timeScale();
        // Set visible range to first 200 bars
        ts.setVisibleLogicalRange({from:0,to:200});
      }
    });
  },150);
  
  document.addEventListener('keydown',e=>{
    if(e.key==='f'||e.key==='F')charts.forEach(({ch})=>{if(ch)ch.timeScale().fitContent();});
    if(e.key==='r'||e.key==='R')charts.forEach(({ch})=>{if(ch)ch.timeScale().resetTimeScale();});
    // Press 'h' for home (beginning)
    if(e.key==='h'||e.key==='H')charts.forEach(({ch})=>{if(ch)ch.timeScale().setVisibleLogicalRange({from:0,to:200});});
    // Press 'e' for end
    if(e.key==='e'||e.key==='E')charts.forEach(({ch})=>{if(ch)ch.timeScale().scrollToRealTime();});
  });
} catch(e) {
  console.warn('Shortcuts error:',e);
}

// === ZOOM CONTROLS ===
try {
  const zoomInBtn=document.getElementById('zoomIn');
  const zoomOutBtn=document.getElementById('zoomOut');
  
  if(zoomInBtn){
    zoomInBtn.addEventListener('click',()=>{
      charts.forEach(({ch})=>{
        if(ch){
          const ts=ch.timeScale();
          const spacing=(ts.options().barSpacing||8)+2;
          ts.applyOptions({barSpacing:Math.min(spacing,50)});
        }
      });
    });
  }
  if(zoomOutBtn){
    zoomOutBtn.addEventListener('click',()=>{
      charts.forEach(({ch})=>{
        if(ch){
          const ts=ch.timeScale();
          const spacing=(ts.options().barSpacing||8)-2;
          ts.applyOptions({barSpacing:Math.max(spacing,1)});
        }
      });
    });
  }
} catch(e) {
  console.warn('Zoom controls error:',e);
}

} catch(globalError) {
  console.error('Chart initialization failed:', globalError);
  document.body.innerHTML = '<div style="color:#ef4444;padding:40px;font-family:system-ui;text-align:center;"><h2>Chart Error</h2><p>' + globalError.message + '</p><p>Open browser console for details.</p></div>';
}

})();
</script>
</body>
</html>'''
    
    # Replace placeholders
    html = html.replace('ACTIVO_PLACEHOLDER', activo)
    html = html.replace('COMBO_PLACEHOLDER', combo)
    html = html.replace('TRADES_PLACEHOLDER', str(total_trades))
    html = html.replace('WINRATE_PLACEHOLDER', str(round(winrate, 1)))
    html = html.replace('PNL_CLASS_PLACEHOLDER', pnl_class)
    html = html.replace('PNL_PLACEHOLDER', str(round(pnl_neto, 2)))
    html = html.replace('SCORE_PLACEHOLDER', str(round(score, 2)))
    html = html.replace('CANDLES_JSON_PLACEHOLDER', candles_json)
    html = html.replace('INDICATORS_JSON_PLACEHOLDER', indicators_json)
    html = html.replace('TRADES_JSON_PLACEHOLDER', trades_json)
    
    return html


# =============================================================================
# MAIN PLOT FUNCTION
# =============================================================================

def plot_trades(
    df: DataFrameType,
    df_trades: DataFrameType,
    plot_base: str,
    fecha_inicio_plot: str,
    fecha_fin_plot: str,
    trial_number: int,
    params: dict,
    score: float,
    combo: str,
    metrics: Optional[dict] = None,
    equity_curve: Optional[list] = None,
    saldo_inicial: float = 300.0,
    max_archivos: int = 5,
    activo: Optional[str] = None,
):
    """
    Generate ultra-fast trading chart HTML with dynamic indicator detection.
    
    Automatically detects indicators from DataFrame columns and generates
    appropriate panels (overlays on price, oscillators as sub-panels).
    """
    
    # ================== EXTRACT NUMPY ARRAYS ==================
    if HAS_POLARS and isinstance(df, pl.DataFrame):
        timestamps = df["timestamp"].to_numpy() if "timestamp" in df.columns else df.to_pandas().index.values
        opens = df["open"].to_numpy().astype(np.float64)
        highs = df["high"].to_numpy().astype(np.float64)
        lows = df["low"].to_numpy().astype(np.float64)
        closes = df["close"].to_numpy().astype(np.float64)
        volumes = df["volume"].to_numpy().astype(np.float64) if "volume" in df.columns else None
        df_cols = set(df.columns)
    else:
        df_pd = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
        if not isinstance(df_pd.index, pd.DatetimeIndex):
            if "timestamp" in df_pd.columns:
                df_pd = df_pd.set_index("timestamp")
        timestamps = df_pd.index.values
        opens = df_pd["open"].values.astype(np.float64)
        highs = df_pd["high"].values.astype(np.float64)
        lows = df_pd["low"].values.astype(np.float64)
        closes = df_pd["close"].values.astype(np.float64)
        volumes = df_pd["volume"].values.astype(np.float64) if "volume" in df_pd.columns else None
        df_cols = set(df_pd.columns)
    
    # ================== DATE FILTERING ==================
    start_pd = pd.to_datetime(fecha_inicio_plot, utc=True)
    end_pd = pd.to_datetime(fecha_fin_plot, utc=True)
    start = np.datetime64(start_pd.tz_localize(None))
    end = np.datetime64(end_pd.tz_localize(None))
    
    if np.issubdtype(timestamps.dtype, np.datetime64):
        ts_compare = timestamps
    else:
        ts_compare = timestamps.astype('datetime64[ns]')
    
    mask = (ts_compare >= start) & (ts_compare <= end)
    
    timestamps = timestamps[mask]
    opens = opens[mask]
    highs = highs[mask]
    lows = lows[mask]
    closes = closes[mask]
    if volumes is not None:
        volumes = volumes[mask]
    
    if len(timestamps) == 0:
        return
    
    # ================== BANKRUPTCY CUTOFF ==================
    saldo_minimo_operativo = 5.0
    if equity_curve and len(equity_curve) > 0:
        eq_arr = np.array(equity_curve, dtype=np.float64)
        bankruptcy_indices = np.where(eq_arr <= saldo_minimo_operativo)[0]
        if len(bankruptcy_indices) > 0:
            bankruptcy_idx = int(bankruptcy_indices[0])
            if bankruptcy_idx < len(eq_arr) - 1:
                ratio = (bankruptcy_idx + 1) / len(eq_arr)
                candle_cutoff = max(1, min(int(len(timestamps) * ratio), len(timestamps)))
                timestamps = timestamps[:candle_cutoff]
                opens = opens[:candle_cutoff]
                highs = highs[:candle_cutoff]
                lows = lows[:candle_cutoff]
                closes = closes[:candle_cutoff]
                if volumes is not None:
                    volumes = volumes[:candle_cutoff]
                equity_curve = equity_curve[:bankruptcy_idx + 1]
    
    # ================== PREPARE OHLCV ==================
    ts_q_full, o_q_full, h_q_full, l_q_full, c_q_full, vol_q_full, price_factor = _prepare_ohlcv_vectorized(
        timestamps, opens, highs, lows, closes, volumes
    )
    
    # ================== GLOBAL WARM-UP PERIOD (v6.1) ==================
    # Detect the maximum period from all *_period params BEFORE slicing data
    # This ensures candles + indicators + markers all start at the same point
    max_warmup = _detect_max_warmup_period(params, min_warmup=1)
    
    # Clamp warmup to valid range (leave at least 10 candles visible)
    max_warmup = min(max_warmup, len(ts_q_full) - 10)
    max_warmup = max(0, max_warmup)  # Ensure non-negative
    
    print(f"[PLOT v6.1] Global Warm-up: {max_warmup} bars (chart starts at candle {max_warmup})")
    
    # SLICE ALL DATA FROM WARMUP POINT - Everything synchronized
    ts_q = ts_q_full[max_warmup:]
    o_q = o_q_full[max_warmup:]
    h_q = h_q_full[max_warmup:]
    l_q = l_q_full[max_warmup:]
    c_q = c_q_full[max_warmup:]
    vol_q = vol_q_full[max_warmup:] if vol_q_full is not None else None
    
    # Get the warmup threshold timestamp for trade marker filtering
    warmup_threshold_ts = int(ts_q[0]) if len(ts_q) > 0 else 0
    
    candle_data = {
        "t": ts_q.tolist(),
        "o": o_q.tolist(),
        "h": h_q.tolist(),
        "l": l_q.tolist(),
        "c": c_q.tolist(),
        "f": int(price_factor)
    }
    if vol_q is not None:
        candle_data["vol"] = vol_q.tolist()
    
    # ================== ZERO-LAG INDICATOR ALIGNMENT (v6.0) ==================
    # Architecture: Single Authoritative Timestamp Array (SATA)
    # ts_q is the ONLY source of truth for all timestamp alignment.
    # All indicator values are mapped to ts_q indices using StrictAlignmentMapper.
    
    # Step 1: Convert source DataFrame to pandas with UTC index
    if HAS_POLARS and isinstance(df, pl.DataFrame):
        df_pd_full = df.to_pandas()
    else:
        df_pd_full = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    
    if not isinstance(df_pd_full.index, pd.DatetimeIndex):
        if "timestamp" in df_pd_full.columns:
            df_pd_full = df_pd_full.set_index("timestamp")
    
    # Ensure index is UTC-aware for consistent timestamp conversion
    if df_pd_full.index.tz is None:
        df_pd_full.index = df_pd_full.index.tz_localize("UTC")
    elif str(df_pd_full.index.tz) != 'UTC':
        df_pd_full.index = df_pd_full.index.tz_convert("UTC")
    
    # Step 2: Extract source timestamps as Unix seconds
    # Use the centralized _normalize_timestamps_to_unix for consistency
    source_ts_raw = df_pd_full.index.tz_localize(None).values  # Remove TZ for datetime64 conversion
    source_timestamps = _normalize_timestamps_to_unix(source_ts_raw)
    
    # Step 3: Initialize the StrictAlignmentMapper with authoritative timestamps
    # This is the KEY component for zero-lag alignment
    aligner = StrictAlignmentMapper(ts_q)
    
    # Step 4: Create aligned source mask (which source rows exist in ts_q)
    source_mask = np.array([int(ts) in aligner.ts_to_idx for ts in source_timestamps])
    df_aligned = df_pd_full[source_mask].copy()
    aligned_source_ts = source_timestamps[source_mask]
    
    # Verify alignment
    alignment_match = len(df_aligned) == len(ts_q)
    # print(f"[PLOT v6.0] Candles: {len(ts_q)}, Aligned Source: {len(df_aligned)}, Perfect Match: {alignment_match}")
    
    if not alignment_match:
        # Detailed debug for misalignment
        missing_in_source = len(ts_q) - len(df_aligned)
        print(f"[PLOT WARN] Missing {missing_in_source} timestamps in source DataFrame. Using mapping fallback.")
    
    # Detect indicators from columns (with Optuna params for dynamic bounds)
    price_range = (float(closes.min()), float(closes.max()))
    detected = _detect_indicators(set(df_aligned.columns), price_range, params)
    
    # Lazy loading check - only use indicators if present in params
    indicators_used = params.get("__indicators_used", None) if params else None
    
    indicators = {"overlays": [], "sub_panels": []}
    
    # ================== PROCESS OVERLAYS (ZERO-LAG, PRE-SLICED) ==================
    for overlay_cfg in detected["overlays"]:
        col = overlay_cfg["col"]
        if col not in df_aligned.columns:
            continue
        # Lazy check
        if indicators_used is not None and col not in indicators_used:
            continue
        
        # Extract values and align using the mapper
        vals = df_aligned[col].values.astype(np.float64)
        aligned_vals = aligner.align(aligned_source_ts, vals)
        
        # Only add if we have valid data
        valid_count = aligner.count_valid(aligned_vals)
        if valid_count > 0:
            # Quantize for JSON efficiency (overlays use price precision)
            quantized, factor = aligner.quantize(aligned_vals, precision=2)
            
            indicators["overlays"].append({
                "t": ts_q.tolist(),  # AUTHORITATIVE timestamps
                "v": quantized,
                "f": int(factor),
                "color": overlay_cfg["color"]
            })
    
    # ================== PROCESS SUB-PANELS / OSCILLATORS (ZERO-LAG, PRE-SLICED) ==================
    for panel_cfg in detected["sub_panels"]:
        col = panel_cfg["col"]
        if col not in df_aligned.columns:
            continue
        # Lazy check
        if indicators_used is not None and col not in indicators_used:
            continue
        
        # Extract values and align using the mapper
        vals = df_aligned[col].values.astype(np.float64)
        aligned_vals = aligner.align(aligned_source_ts, vals)
        
        # Only add if we have valid data
        valid_count = aligner.count_valid(aligned_vals)
        if valid_count > 0:
            # Use dynamic name from detection (includes period if found in params)
            panel_name = panel_cfg.get("name", col.upper())
            
            # Quantize (higher precision for MACD/zscore, standard for others)
            precision = 4 if col in {"macd", "macd_hist", "macd_signal", "zscore", "zscore_kalman"} else 2
            quantized, factor = aligner.quantize(aligned_vals, precision=precision)
            
            indicators["sub_panels"].append({
                "name": panel_name,
                "type": panel_cfg["type"],
                "color": panel_cfg["color"],
                "bounds": panel_cfg.get("bounds"),  # Already merged with Optuna params
                "zero_line": col in {"macd", "macd_hist", "zscore", "zscore_kalman"},
                "data": {
                    "t": ts_q.tolist(),  # AUTHORITATIVE timestamps - ZERO-LAG GUARANTEED
                    "v": quantized,
                    "f": int(factor)
                }
            })
    
    # ================== TRADE MARKERS (Temporal Snapping + Warmup Filter) ==================
    # Use np.searchsorted to snap trade timestamps to exact candle timestamps
    # This prevents marker disappearance during scroll/zoom
    # WARMUP FILTER: Trades within the warmup period are not displayed
    trades = {"m": [], "i": []}
    max_valid_ts = int(ts_q[-1]) if len(ts_q) > 0 else None
    
    # Build efficient lookup structure for candle timestamps
    candle_ts_set = set(ts_q.tolist()) if len(ts_q) > 0 else set()
    
    def _snap_to_candle(trade_ts: int, candle_timestamps: np.ndarray) -> int:
        """
        Vectorized temporal snapping using binary search.
        Snaps trade timestamp to the nearest previous/equal candle timestamp.
        
        Uses np.searchsorted with side='right' to find insertion point,
        then subtracts 1 to get the candle at or before the trade time.
        """
        if len(candle_timestamps) == 0:
            return trade_ts
        
        # Find insertion point (index where trade_ts would be inserted to maintain order)
        idx = np.searchsorted(candle_timestamps, trade_ts, side='right')
        
        # Clamp to valid range and get the candle at or before trade time
        idx = max(0, min(idx - 1, len(candle_timestamps) - 1))
        
        return int(candle_timestamps[idx])
    
    if df_trades is not None:
        if HAS_POLARS and isinstance(df_trades, pl.DataFrame):
            trades_df = df_trades.to_pandas()
        else:
            trades_df = df_trades.copy() if isinstance(df_trades, pd.DataFrame) else pd.DataFrame(df_trades)
        
        if not trades_df.empty:
            entry_times_dt = pd.to_datetime(trades_df["entry_time"], utc=True, errors="coerce")
            exit_times_dt = pd.to_datetime(trades_df["exit_time"], utc=True, errors="coerce")
            
            valid_mask = entry_times_dt.notna()
            trades_df = trades_df[valid_mask].copy()
            entry_times_dt = entry_times_dt[valid_mask]
            exit_times_dt = exit_times_dt[valid_mask]
            
            if len(trades_df) > 0 and len(ts_q) > 0:
                entry_times_np = entry_times_dt.values.astype('datetime64[s]')
                exit_times_np = exit_times_dt.values.astype('datetime64[s]')
                
                entry_timestamps = np.floor(entry_times_np.astype(np.int64)).astype(np.int64)
                exit_timestamps = np.floor(exit_times_np.astype(np.int64)).astype(np.int64)
                
                start_ts = int(start_pd.timestamp())
                end_ts = max_valid_ts if max_valid_ts else int(end_pd.timestamp())
                mask = (entry_timestamps >= start_ts) & (entry_timestamps <= end_ts)
                
                trades_df = trades_df[mask].copy()
                entry_timestamps = entry_timestamps[mask]
                exit_timestamps = exit_timestamps[mask]
                
                # Vectorized snapping: snap all timestamps at once
                # Use searchsorted to find the nearest candle for each trade
                entry_snap_indices = np.searchsorted(ts_q, entry_timestamps, side='right') - 1
                entry_snap_indices = np.clip(entry_snap_indices, 0, len(ts_q) - 1)
                snapped_entry_ts = ts_q[entry_snap_indices]
                
                exit_snap_indices = np.searchsorted(ts_q, exit_timestamps, side='right') - 1
                exit_snap_indices = np.clip(exit_snap_indices, 0, len(ts_q) - 1)
                snapped_exit_ts = ts_q[exit_snap_indices]
                
                for i, (idx, row) in enumerate(trades_df.iterrows()):
                    trade_type = str(row.get("type", "")).upper()
                    
                    # Use snapped timestamps that exactly match candle times
                    et = int(snapped_entry_ts[i])
                    xt = int(snapped_exit_ts[i]) if not np.isnan(exit_timestamps[i]) else None
                    
                    ep = float(row.get("entry_price", 0))
                    xp = float(row.get("exit_price", 0)) if pd.notna(row.get("exit_price")) else None
                    pnl = float(row.get("pnl_neto", 0))
                    
                    # WARMUP FILTER: Skip trades that occur within the global warmup period
                    # This ensures no visual inconsistencies with indicator start points
                    if et < warmup_threshold_ts:
                        continue  # Trade is within warmup period, skip marker
                    
                    # Verify snapped timestamp is in valid range
                    if et in candle_ts_set:
                        comm = float(row.get("comision_total", row.get("comision", 0))) if pd.notna(row.get("comision_total", row.get("comision"))) else 0.0
                        qty = float(row.get("qty", row.get("cantidad", row.get("size", 0)))) if pd.notna(row.get("qty", row.get("cantidad", row.get("size")))) else 0.0
                        
                        trade_info = {
                            "type": trade_type,
                            "ep": round(ep, 2),
                            "xp": round(xp, 2) if xp else None,
                            "pnl": round(pnl, 2),
                            "comm": round(comm, 2),
                            "qty": round(qty, 6)
                        }
                        
                        # DOT-STYLE CENTERED MARKERS (circles, inBar for precision)
                        entry_color = "#3b82f6" if trade_type == "LONG" else "#a855f7"
                        trades["m"].append({
                            "time": et,
                            "position": "inBar",  # Centered on candle body
                            "color": entry_color,
                            "shape": "circle",   # Clean dot style
                            "text": "",           # No text for cleaner look
                            "size": 2
                        })
                        trades["i"].append({"time": et, **trade_info})
                        
                        if xt is not None and xp and xt in candle_ts_set:
                            # Exit marker: white circle
                            trades["m"].append({
                                "time": xt,
                                "position": "inBar",  # Centered on candle body
                                "color": "#ffffff",
                                "shape": "circle",    # Clean dot style
                                "text": "",
                                "size": 1             # Smaller for exits
                            })
                            trades["i"].append({"time": xt, **trade_info})
                
                trades["m"].sort(key=lambda x: x["time"])
                trades["i"].sort(key=lambda x: x["time"])
    
    # ================== CONFIG ==================
    total_trades = 0
    winrate = 0.0
    pnl_neto = 0.0
    
    if metrics:
        total_trades = int(metrics.get("total_trades", metrics.get("num_trades", 0)))
        winrate = float(metrics.get("win_rate", metrics.get("winrate", 0))) * 100 if metrics.get("win_rate", metrics.get("winrate", 0)) <= 1 else float(metrics.get("win_rate", metrics.get("winrate", 0)))
        pnl_neto = float(metrics.get("pnl_neto", metrics.get("net_pnl", 0)))
    elif df_trades is not None:
        if HAS_POLARS and isinstance(df_trades, pl.DataFrame):
            total_trades = len(df_trades)
            pnl_col = "pnl_neto" if "pnl_neto" in df_trades.columns else "pnl"
            if pnl_col in df_trades.columns:
                pnl_neto = float(df_trades[pnl_col].sum())
                wins = (df_trades[pnl_col] > 0).sum()
                winrate = (wins / total_trades * 100) if total_trades > 0 else 0
        else:
            trades_df_for_stats = df_trades if isinstance(df_trades, pd.DataFrame) else pd.DataFrame(df_trades)
            total_trades = len(trades_df_for_stats)
            pnl_col = "pnl_neto" if "pnl_neto" in trades_df_for_stats.columns else "pnl"
            if pnl_col in trades_df_for_stats.columns:
                pnl_neto = float(trades_df_for_stats[pnl_col].sum())
                wins = (trades_df_for_stats[pnl_col] > 0).sum()
                winrate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    config = {
        "activo": str(activo).upper() if activo else "",
        "combo": _generate_dynamic_combo(params, combo) if params else combo,
        "score": score,
        "trial": trial_number,
        "total_trades": total_trades,
        "winrate": winrate,
        "pnl_neto": pnl_neto
    }
    
    # ================== GENERATE HTML ==================
    html = _generate_hft_html(candle_data, indicators, trades, config)
    
    # ================== SAVE FILE ==================
    os.makedirs(plot_base, exist_ok=True)
    
    # Sanitize combo name for filename (remove special chars, limit length)
    combo_safe = re.sub(r"[^a-zA-Z0-9_-]", "_", combo or "STRATEGY")[:30]
    filename = f"TRIAL-{trial_number}_SC-{score:.2f}_{combo_safe}.html"
    filepath = os.path.join(plot_base, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)
    
    # ================== CLEANUP ==================
    if max_archivos > 0:
        _cleanup_old_plots(plot_base, max_archivos)


def _cleanup_old_plots(plot_base: str, max_archivos: int):
    """Remove old plot files, keeping only the best scores."""
    try:
        all_files = [f for f in os.listdir(plot_base) if f.endswith(".html") and f.startswith("TRIAL-")]
        files_with_scores = []
        for f in all_files:
            # Match new format: TRIAL-{n}_SC-{score}_{combo}.html
            match = re.search(r"TRIAL-\d+_SC-([\d.]+)_.*\.html", f)
            if match:
                try:
                    files_with_scores.append((f, float(match.group(1))))
                except ValueError:
                    continue
        
        files_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        if len(files_with_scores) > max_archivos:
            for f, _ in files_with_scores[max_archivos:]:
                old_path = os.path.join(plot_base, f)
                if os.path.exists(old_path):
                    os.remove(old_path)
    except Exception:
        pass
    # Sanitize combo name for filename (remove special chars, limit length)
    combo_safe = re.sub(r"[^a-zA-Z0-9_-]", "_", combo or "STRATEGY")[:30]
    filename = f"TRIAL-{trial_number}_SC-{score:.2f}_{combo_safe}.html"
    filepath = os.path.join(plot_base, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)
    
    # ================== CLEANUP ==================
    if max_archivos > 0:
        _cleanup_old_plots(plot_base, max_archivos)


def _cleanup_old_plots(plot_base: str, max_archivos: int):
    """Remove old plot files, keeping only the best scores."""
    try:
        all_files = [f for f in os.listdir(plot_base) if f.endswith(".html") and f.startswith("TRIAL-")]
        files_with_scores = []
        for f in all_files:
            # Match new format: TRIAL-{n}_SC-{score}_{combo}.html
            match = re.search(r"TRIAL-\d+_SC-([\d.]+)_.*\.html", f)
            if match:
                try:
                    files_with_scores.append((f, float(match.group(1))))
                except ValueError:
                    continue
        
        files_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        if len(files_with_scores) > max_archivos:
            for f, _ in files_with_scores[max_archivos:]:
                old_path = os.path.join(plot_base, f)
                if os.path.exists(old_path):
                    os.remove(old_path)
    except Exception:
        pass
        files_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        if len(files_with_scores) > max_archivos:
            for f, _ in files_with_scores[max_archivos:]:
                old_path = os.path.join(plot_base, f)
                if os.path.exists(old_path):
                    os.remove(old_path)
    except Exception:
        pass
