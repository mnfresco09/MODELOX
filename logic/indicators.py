"""
================================================================================
MODELOX - High-Performance Technical Indicators Engine (Refactored)
================================================================================
Optimized for 5-minute OHLCV data processing with institutional-grade performance.

Architecture:
- Numba @njit for ALL iterative algorithms (unified approach)
- Polars native only for simple vectorized operations
- Zero silent errors: all divisions protected against zero/NaN
- Consistent output: all indicators return same-length DataFrame

Refactoring Notes:
- Removed Polars duplicates (rsi, ema, mfi, adx) - Numba versions only
- Consolidated macd (single function, 3 outputs)
- Added: Stochastic (%K, %D), ROC
- Clean Factory registry

Author: MODELOX Quant Team
Version: 3.0.0 (Refactored - No Duplicates)
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import math
import numpy as np
import polars as pl
from numba import njit


# =============================================================================
# CONSTANTS
# =============================================================================

_EPSILON = 1e-10  # Minimum denominator to prevent division by zero


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def _has_cols(df: pl.DataFrame, cols: Tuple[str, ...]) -> bool:
    """Check if DataFrame contains all required columns."""
    return all(c in df.columns for c in cols)


def _ensure_timestamp(df: pl.DataFrame, ts_col: str = "timestamp") -> pl.DataFrame:
    """Ensure timestamp column exists and is proper Datetime type."""
    if ts_col not in df.columns:
        # Common alias support: many datasets use 'datetime'
        if "datetime" in df.columns:
            df = df.with_columns(pl.col("datetime").alias(ts_col))
        else:
            raise ValueError(f"Missing temporal column '{ts_col}'.")
    
    schema_type = df.schema.get(ts_col)
    dtype_str = str(schema_type) if schema_type is not None else ""

    # Accept any Datetime variant, including timezone-aware (e.g., Datetime('us', 'UTC')).
    if dtype_str.startswith("Datetime"):
        return df

    # If integer epoch timestamps, convert to UTC Datetime.
    if dtype_str in {"Int64", "Int32", "UInt64", "UInt32"}:
        # Infer epoch unit by magnitude (seconds/ms/us/ns).
        max_abs = df.select(pl.col(ts_col).abs().max()).item()
        if max_abs is None:
            return df.with_columns(pl.lit(None).cast(pl.Datetime("us", "UTC")).alias(ts_col))

        x = float(max_abs)
        # ~2026 epoch magnitudes:
        # s  ~ 1e9,  ms ~ 1e12, us ~ 1e15, ns ~ 1e18
        if x >= 1e17:
            unit = "ns"
        elif x >= 1e14:
            unit = "us"
        elif x >= 1e11:
            unit = "ms"
        else:
            unit = "s"

        return df.with_columns(
            pl.from_epoch(pl.col(ts_col).cast(pl.Int64), time_unit=unit)
            .dt.replace_time_zone("UTC")
            .alias(ts_col)
        )

    # Parse strings to UTC-aware Datetime. Newer Polars requires explicit tz if present in data.
    return df.with_columns(
        pl.col(ts_col)
        .cast(pl.Utf8)
        .str.to_datetime(strict=False, time_zone="UTC")
        .alias(ts_col)
    )


def _alpha_from_period(period: int) -> float:
    """Convert period to EMA alpha: alpha = 2 / (period + 1)."""
    return 2.0 / (float(period) + 1.0)


def _safe_divide(numerator: pl.Expr, denominator: pl.Expr, default: float = 0.0) -> pl.Expr:
    """Safe division for Polars expressions."""
    return (
        pl.when((denominator == 0) | denominator.is_null() | denominator.is_nan())
        .then(pl.lit(default))
        .otherwise(numerator / denominator)
    )


@njit(cache=True, fastmath=True)
def _safe_div_numba(num: float, denom: float, default: float = 0.0) -> float:
    """Safe division for Numba kernels."""
    if abs(denom) < _EPSILON or np.isnan(denom) or np.isinf(denom):
        return default
    return num / denom


# =============================================================================
# ### EMA - EXPONENTIAL MOVING AVERAGE ###
# =============================================================================


@njit(cache=True, fastmath=True)
def _ema_numba(src: np.ndarray, period: int) -> np.ndarray:
    """EMA - Numba optimized."""
    n = len(src)
    ema = np.full(n, np.nan, dtype=np.float64)
    
    if n < period:
        return ema
    
    alpha = 2.0 / (period + 1.0)
    
    # Seed with SMA
    total = 0.0
    for i in range(period):
        total += src[i]
    ema[period - 1] = total / period
    
    # EMA calculation
    for i in range(period, n):
        ema[i] = alpha * src[i] + (1.0 - alpha) * ema[i - 1]
    
    return ema


def ema(
    df: pl.DataFrame,
    *,
    period: int = 20,
    col: str = "close",
    out: str = "ema",
) -> pl.DataFrame:
    """Exponential Moving Average - Numba optimized."""
    if col not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out))
    
    src = np.ascontiguousarray(df[col].to_numpy().astype(np.float64))
    ema_vals = _ema_numba(src, int(period))
    return df.with_columns(pl.Series(out, ema_vals))


# =============================================================================
# ### SMA - SIMPLE MOVING AVERAGE ###
# =============================================================================


@njit(cache=True, fastmath=True)
def _sma_numba(src: np.ndarray, period: int) -> np.ndarray:
    """SMA - Numba optimized with rolling sum."""
    n = len(src)
    sma = np.full(n, np.nan, dtype=np.float64)
    
    if n < period:
        return sma
    
    # First SMA
    total = 0.0
    for i in range(period):
        total += src[i]
    sma[period - 1] = total / period
    
    # Rolling sum for efficiency
    for i in range(period, n):
        total = total - src[i - period] + src[i]
        sma[i] = total / period
    
    return sma


def sma(
    df: pl.DataFrame,
    *,
    period: int = 20,
    col: str = "close",
    out: str = "sma",
) -> pl.DataFrame:
    """Simple Moving Average - Numba optimized."""
    if col not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out))
    
    src = np.ascontiguousarray(df[col].to_numpy().astype(np.float64))
    sma_vals = _sma_numba(src, int(period))
    return df.with_columns(pl.Series(out, sma_vals))


# =============================================================================
# ### WMA - WEIGHTED MOVING AVERAGE ###
# =============================================================================


@njit(cache=True, fastmath=True)
def _wma_numba(src: np.ndarray, period: int) -> np.ndarray:
    """WMA - Numba optimized. Weights decrease linearly."""
    n = len(src)
    wma = np.full(n, np.nan, dtype=np.float64)
    
    if n < period:
        return wma
    
    weight_sum = period * (period + 1) / 2.0
    
    for i in range(period - 1, n):
        weighted_sum = 0.0
        for j in range(period):
            weight = period - j
            weighted_sum += weight * src[i - j]
        wma[i] = weighted_sum / weight_sum
    
    return wma


def wma(
    df: pl.DataFrame,
    *,
    period: int = 20,
    col: str = "close",
    out: str = "wma",
) -> pl.DataFrame:
    """Weighted Moving Average - Numba optimized."""
    if col not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out))
    
    src = np.ascontiguousarray(df[col].to_numpy().astype(np.float64))
    wma_vals = _wma_numba(src, int(period))
    return df.with_columns(pl.Series(out, wma_vals))


# =============================================================================
# ### HMA - HULL MOVING AVERAGE ###
# =============================================================================


@njit(cache=True, fastmath=True)
def _hma_numba(src: np.ndarray, period: int) -> np.ndarray:
    """HMA - Reduced lag while maintaining smoothness."""
    n = len(src)
    hma = np.full(n, np.nan, dtype=np.float64)
    
    half_period = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(float(period))))
    
    if n < period:
        return hma
    
    wma_half = _wma_numba(src, half_period)
    wma_full = _wma_numba(src, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    hma = _wma_numba(diff, sqrt_period)
    return hma


def hma(
    df: pl.DataFrame,
    *,
    period: int = 20,
    col: str = "close",
    out: str = "hma",
) -> pl.DataFrame:
    """Hull Moving Average - Numba optimized."""
    if col not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out))
    
    src = np.ascontiguousarray(df[col].to_numpy().astype(np.float64))
    hma_vals = _hma_numba(src, int(period))
    return df.with_columns(pl.Series(out, hma_vals))


# =============================================================================
# ### ALMA - ARNAUD LEGOUX MOVING AVERAGE ###
# =============================================================================


@njit(cache=True, fastmath=True)
def _alma_numba(src: np.ndarray, length: int, offset: float, sigma: float) -> np.ndarray:
    """ALMA smoothing.

    Uses Gaussian weights centered at m = offset*(length-1) with width sigma.
    """
    n = len(src)
    out = np.full(n, np.nan, dtype=np.float64)
    if length <= 0 or n < length:
        return out

    m = offset * (length - 1)
    s = max(1e-6, float(sigma))
    denom = 2.0 * s * s

    # Precompute weights (fixed for given length/offset/sigma)
    w = np.empty(length, dtype=np.float64)
    wsum = 0.0
    for i in range(length):
        x = float(i) - m
        wi = np.exp(-(x * x) / denom)
        w[i] = wi
        wsum += wi

    for t in range(length - 1, n):
        acc = 0.0
        ok = True
        # align: i=0 refers to oldest in window, i=length-1 newest
        for i in range(length):
            v = src[t - (length - 1 - i)]
            if np.isnan(v):
                ok = False
                break
            acc += v * w[i]
        if not ok or wsum <= 0.0:
            continue
        out[t] = acc / wsum

    return out


def alma(
    df: pl.DataFrame,
    *,
    length: int = 20,
    offset: float = 0.85,
    sigma: float = 7.0,
    col: str = "close",
    out: str = "alma",
) -> pl.DataFrame:
    """ALMA (Arnaud Legoux Moving Average) - Numba optimized."""
    if col not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out))
    src = np.ascontiguousarray(df[col].to_numpy().astype(np.float64))
    vals = _alma_numba(src, int(length), float(offset), float(sigma))
    return df.with_columns(pl.Series(out, vals))


# =============================================================================
# ### CHANDE MOMENTUM OSCILLATOR (CMO) ###
# =============================================================================


@njit(cache=True, fastmath=True)
def _chande_mo_numba(close: np.ndarray, length: int) -> np.ndarray:
    """Chande Momentum Oscillator-style value in [-100, 100]."""
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if length <= 0 or n < length + 1:
        return out

    # diffs
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        a = close[i]
        b = close[i - 1]
        if np.isnan(a) or np.isnan(b):
            continue
        diff[i] = a - b

    for i in range(length, n):
        up = 0.0
        dn = 0.0
        ok = True
        for j in range(i - length + 1, i + 1):
            d = diff[j]
            if np.isnan(d):
                ok = False
                break
            if d >= 0.0:
                up += d
            else:
                dn += -d
        if not ok:
            continue
        denom = up + dn
        if denom < _EPSILON:
            out[i] = 0.0
        else:
            out[i] = 100.0 * (up - dn) / denom
    return out


def chande_mo(
    df: pl.DataFrame,
    *,
    length: int = 9,
    col: str = "close",
    out: str = "chande_mo",
) -> pl.DataFrame:
    """Chande Momentum Oscillator (CMO-style) - Numba optimized."""
    if col not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out))
    src = np.ascontiguousarray(df[col].to_numpy().astype(np.float64))
    vals = _chande_mo_numba(src, int(length))
    return df.with_columns(pl.Series(out, vals))


# =============================================================================
# ### ASGMA - ALMA Smoothed Gaussian Moving Average (oscillator version) ###
# =============================================================================


@njit(cache=True, fastmath=True)
def _rolling_std_numba(src: np.ndarray, window: int) -> np.ndarray:
    n = len(src)
    out = np.full(n, np.nan, dtype=np.float64)
    if window <= 1 or n < window:
        return out
    for i in range(window - 1, n):
        s = 0.0
        ss = 0.0
        ok = True
        for j in range(i - window + 1, i + 1):
            x = src[j]
            if np.isnan(x):
                ok = False
                break
            s += x
            ss += x * x
        if not ok:
            continue
        mean = s / window
        var = ss / window - mean * mean
        out[i] = np.sqrt(var) if var > 0.0 else 0.0
    return out


def stdev(
    df: pl.DataFrame,
    *,
    window: int = 20,
    col: str = "close",
    out: str = "stdev",
) -> pl.DataFrame:
    """Rolling standard deviation (Numba)."""
    if col not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out))
    src = np.ascontiguousarray(df[col].to_numpy().astype(np.float64))
    vals = _rolling_std_numba(src, int(window))
    return df.with_columns(pl.Series(out, vals))


@njit(cache=True)
def _ema_nanaware_numba(src: np.ndarray, period: int) -> np.ndarray:
    """EMA that can start after leading NaNs.

    Seeds with the first window of `period` consecutive finite values.
    If a later value is NaN, it carries forward the previous EMA.
    """
    n = len(src)
    out = np.full(n, np.nan, dtype=np.float64)
    if period <= 0 or n < period:
        return out

    alpha = 2.0 / (period + 1.0)

    start = -1
    for i in range(period - 1, n):
        ok = True
        total = 0.0
        for j in range(i - period + 1, i + 1):
            v = src[j]
            if np.isnan(v):
                ok = False
                break
            total += v
        if ok:
            out[i] = total / period
            start = i
            break

    if start < 0:
        return out

    for i in range(start + 1, n):
        v = src[i]
        prev = out[i - 1]
        if np.isnan(prev):
            continue
        if np.isnan(v):
            out[i] = prev
        else:
            out[i] = alpha * v + (1.0 - alpha) * prev

    return out


@njit(cache=True)
def _asgma_numba(
    close: np.ndarray,
    smooth: int,
    alma_len: int,
    alma_offset: float,
    alma_sigma: float,
    gma_len: int,
    adaptive: int,
    vol_period: int,
    sigma_fixed: float,
    gma_ema: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ASGMA core series.

    avpchange:
      pchange = change(close, smooth)/close * 100
      avpchange = ALMA(pchange, alma_len, alma_offset, alma_sigma)

    gma:
      gaussian-weighted average of midrange(highest+lowest)/2 of avpchange over variable windows,
      optionally with adaptive sigma=stdev(close, vol_period), then EMA(gma, gma_ema)
    """

    n = len(close)
    pchg = np.full(n, np.nan, dtype=np.float64)
    for i in range(smooth, n):
        c = close[i]
        prev = close[i - smooth]
        if np.isnan(c) or np.isnan(prev) or abs(c) < _EPSILON:
            continue
        pchg[i] = (c - prev) / c * 100.0

    avp = _alma_numba(pchg, alma_len, alma_offset, alma_sigma)

    # sigma series
    if adaptive != 0:
        sig = _rolling_std_numba(close, vol_period)
    else:
        sig = np.full(n, float(sigma_fixed), dtype=np.float64)

    gma = np.full(n, np.nan, dtype=np.float64)
    start_t = gma_len - 1
    if adaptive != 0 and vol_period - 1 > start_t:
        start_t = vol_period - 1
    if alma_len - 1 > start_t:
        start_t = alma_len - 1

    for t in range(start_t, n):
        sigma_t = sig[t]
        if np.isnan(sigma_t) or sigma_t < 0.1:
            sigma_t = max(0.1, float(sigma_fixed))

        denom_sigma = 2.0 * sigma_t
        if denom_sigma < 1e-6:
            denom_sigma = 1e-6

        sum_w = 0.0
        acc = 0.0

        # weights depend on i and sigma_t
        for i in range(gma_len):
            # i goes 0..len-1
            x = (float(i) - float(gma_len - 1)) / denom_sigma
            w = np.exp(-(x * x) / 2.0)

            win = i + 1
            hi = -1e308
            lo = 1e308
            ok = True
            for j in range(t - win + 1, t + 1):
                v = avp[j]
                if np.isnan(v):
                    ok = False
                    break
                if v > hi:
                    hi = v
                if v < lo:
                    lo = v
            if not ok:
                continue

            value = (hi + lo) * 0.5
            acc += value * w
            sum_w += w

        if sum_w > 0.0:
            gma[t] = acc / sum_w

    # smooth gma with EMA (NaN-aware)
    gma_sm = _ema_nanaware_numba(gma, int(gma_ema))
    return avp, gma_sm


def asgma(
    df: pl.DataFrame,
    *,
    col: str = "close",
    smooth: int = 1,
    alma_len: int = 25,
    alma_offset: float = 0.85,
    alma_sigma: float = 7.0,
    gma_len: int = 14,
    adaptive: bool = True,
    volatility_period: int = 20,
    sigma_fixed: float = 1.0,
    gma_ema: int = 7,
    out_avpchange: str = "asgma_avpchange",
    out_gma: str = "asgma_gma",
) -> pl.DataFrame:
    """ALMA-smoothed percent change + Gaussian moving average of that signal."""
    if col not in df.columns:
        return df.with_columns([
            pl.lit(None).cast(pl.Float64).alias(out_avpchange),
            pl.lit(None).cast(pl.Float64).alias(out_gma),
        ])

    close = np.ascontiguousarray(df[col].to_numpy().astype(np.float64))
    avp, gma_vals = _asgma_numba(
        close,
        int(smooth),
        int(alma_len),
        float(alma_offset),
        float(alma_sigma),
        int(gma_len),
        1 if adaptive else 0,
        int(volatility_period),
        float(sigma_fixed),
        int(gma_ema),
    )

    return df.with_columns([
        pl.Series(out_avpchange, avp),
        pl.Series(out_gma, gma_vals),
    ])


# =============================================================================
# ### RSI - RELATIVE STRENGTH INDEX ###
# =============================================================================


@njit(cache=True, fastmath=True)
def _rsi_numba(close: np.ndarray, period: int) -> np.ndarray:
    """RSI - Numba optimized with Wilder's smoothing."""
    n = len(close)
    rsi_out = np.full(n, np.nan, dtype=np.float64)
    
    if n < period + 1:
        return rsi_out
    
    gains = np.zeros(n, dtype=np.float64)
    losses = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        delta = close[i] - close[i - 1]
        if delta > 0:
            gains[i] = delta
        else:
            losses[i] = -delta
    
    # First average (SMA)
    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(1, period + 1):
        avg_gain += gains[i]
        avg_loss += losses[i]
    avg_gain /= period
    avg_loss /= period
    
    if avg_loss > _EPSILON:
        rs = avg_gain / avg_loss
        rsi_out[period] = 100.0 - 100.0 / (1.0 + rs)
    else:
        rsi_out[period] = 100.0 if avg_gain > 0 else 50.0
    
    # Wilder's smoothing
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss > _EPSILON:
            rs = avg_gain / avg_loss
            rsi_out[i] = 100.0 - 100.0 / (1.0 + rs)
        else:
            rsi_out[i] = 100.0 if avg_gain > 0 else 50.0
    
    return rsi_out


def rsi(
    df: pl.DataFrame,
    *,
    period: int = 14,
    col: str = "close",
    out: str = "rsi",
) -> pl.DataFrame:
    """Relative Strength Index - Numba optimized."""
    if col not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out))
    
    src = np.ascontiguousarray(df[col].to_numpy().astype(np.float64))
    rsi_vals = _rsi_numba(src, int(period))
    return df.with_columns(pl.Series(out, rsi_vals))


# =============================================================================
# ### STOCHASTIC - %K AND %D ###
# =============================================================================


@njit(cache=True, fastmath=True)
def _stochastic_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    k_period: int, d_period: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stochastic Oscillator - Numba optimized.
    
    %K = 100 * (Close - Lowest Low) / (Highest High - Lowest Low)
    %D = SMA(%K, d_period)
    """
    n = len(close)
    stoch_k = np.full(n, np.nan, dtype=np.float64)
    stoch_d = np.full(n, np.nan, dtype=np.float64)
    
    if n < k_period:
        return stoch_k, stoch_d
    
    # Calculate %K
    for i in range(k_period - 1, n):
        highest = high[i]
        lowest = low[i]
        for j in range(i - k_period + 1, i + 1):
            if high[j] > highest:
                highest = high[j]
            if low[j] < lowest:
                lowest = low[j]
        
        range_hl = highest - lowest
        if range_hl > _EPSILON:
            stoch_k[i] = 100.0 * (close[i] - lowest) / range_hl
        else:
            stoch_k[i] = 50.0  # Neutral when no range
    
    # Calculate %D (SMA of %K)
    if n < k_period + d_period - 1:
        return stoch_k, stoch_d
    
    for i in range(k_period + d_period - 2, n):
        total = 0.0
        count = 0
        for j in range(i - d_period + 1, i + 1):
            if not np.isnan(stoch_k[j]):
                total += stoch_k[j]
                count += 1
        if count > 0:
            stoch_d[i] = total / count
    
    return stoch_k, stoch_d


def stochastic(
    df: pl.DataFrame,
    *,
    k_period: int = 14,
    d_period: int = 3,
    out_k: str = "stoch_k",
    out_d: str = "stoch_d",
) -> pl.DataFrame:
    """Stochastic Oscillator (%K, %D) - Numba optimized."""
    if not _has_cols(df, ("high", "low", "close")):
        return df.with_columns([
            pl.lit(None).cast(pl.Float64).alias(out_k),
            pl.lit(None).cast(pl.Float64).alias(out_d),
        ])
    
    h = np.ascontiguousarray(df["high"].to_numpy().astype(np.float64))
    l = np.ascontiguousarray(df["low"].to_numpy().astype(np.float64))
    c = np.ascontiguousarray(df["close"].to_numpy().astype(np.float64))
    
    k_vals, d_vals = _stochastic_numba(h, l, c, int(k_period), int(d_period))
    
    return df.with_columns([
        pl.Series(out_k, k_vals),
        pl.Series(out_d, d_vals),
    ])


# =============================================================================
# ### ROC - RATE OF CHANGE ###
# =============================================================================


@njit(cache=True, fastmath=True)
def _roc_numba(close: np.ndarray, period: int) -> np.ndarray:
    """
    Rate of Change - Numba optimized.
    
    ROC = ((Close - Close[n]) / Close[n]) * 100
    """
    n = len(close)
    roc = np.full(n, np.nan, dtype=np.float64)
    
    if n <= period:
        return roc
    
    for i in range(period, n):
        prev = close[i - period]
        if abs(prev) > _EPSILON:
            roc[i] = ((close[i] - prev) / prev) * 100.0
        else:
            roc[i] = 0.0
    
    return roc


def roc(
    df: pl.DataFrame,
    *,
    period: int = 12,
    col: str = "close",
    out: str = "roc",
) -> pl.DataFrame:
    """Rate of Change - Numba optimized."""
    if col not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out))
    
    src = np.ascontiguousarray(df[col].to_numpy().astype(np.float64))
    roc_vals = _roc_numba(src, int(period))
    return df.with_columns(pl.Series(out, roc_vals))


# =============================================================================
# ### DPO - DETRENDED PRICE OSCILLATOR ###
# =============================================================================


@njit(cache=True, fastmath=True)
def _dpo_numba(close: np.ndarray, period: int) -> np.ndarray:
    """
    DPO (Detrended Price Oscillator) - Numba optimized.
    
    Formula: DPO = Close - SMA(Close, period) shifted forward by (period/2 + 1)
    
    The DPO removes the trend from price to identify cycles.
    Positive DPO = price above the shifted SMA (bullish cycle)
    Negative DPO = price below the shifted SMA (bearish cycle)
    """
    n = len(close)
    dpo = np.full(n, np.nan, dtype=np.float64)
    
    if n < period:
        return dpo
    
    # Calculate SMA
    sma = np.full(n, np.nan, dtype=np.float64)
    total = 0.0
    for i in range(period):
        total += close[i]
    sma[period - 1] = total / period
    
    for i in range(period, n):
        total = total - close[i - period] + close[i]
        sma[i] = total / period
    
    # Shift amount: period/2 + 1
    shift = period // 2 + 1
    
    # DPO = Close - SMA(shifted forward)
    # Looking back: current price vs SMA from (shift) bars ago
    for i in range(shift + period - 1, n):
        sma_shifted = sma[i - shift]
        if not np.isnan(sma_shifted):
            dpo[i] = close[i] - sma_shifted
    
    return dpo


def dpo(
    df: pl.DataFrame,
    *,
    period: int = 20,
    col: str = "close",
    out: str = "dpo",
) -> pl.DataFrame:
    """
    Detrended Price Oscillator (DPO) - Numba optimized.
    
    Removes the trend from price to show underlying cycles.
    Used for identifying cycle highs/lows independent of trend direction.
    
    Parameters:
    - period: SMA period (typically 20)
    - col: Source column (default: close)
    - out: Output column name
    
    Returns:
    - DPO values (positive = bullish cycle, negative = bearish cycle)
    """
    if col not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out))
    
    src = np.ascontiguousarray(df[col].to_numpy().astype(np.float64))
    dpo_vals = _dpo_numba(src, int(period))
    return df.with_columns(pl.Series(out, dpo_vals))


# =============================================================================
# ### ATR - AVERAGE TRUE RANGE ###
# =============================================================================


@njit(cache=True, fastmath=True)
def _atr_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """ATR - Numba optimized with Wilder's smoothing."""
    n = len(close)
    atr = np.full(n, np.nan, dtype=np.float64)
    
    if n < period:
        return atr
    
    # True Range
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
    
    # First ATR (SMA)
    total = 0.0
    for i in range(period):
        total += tr[i]
    atr[period - 1] = total / period
    
    # Wilder's smoothing
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def atr(
    df: pl.DataFrame,
    *,
    period: int = 14,
    out: str = "atr",
) -> pl.DataFrame:
    """Average True Range - Numba optimized."""
    if not _has_cols(df, ("high", "low", "close")):
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out))
    
    h = np.ascontiguousarray(df["high"].to_numpy().astype(np.float64))
    l = np.ascontiguousarray(df["low"].to_numpy().astype(np.float64))
    c = np.ascontiguousarray(df["close"].to_numpy().astype(np.float64))
    
    atr_vals = _atr_numba(h, l, c, int(period))
    return df.with_columns(pl.Series(out, atr_vals))


# =============================================================================
# ### ADX - AVERAGE DIRECTIONAL INDEX ###
# =============================================================================


@njit(cache=True, fastmath=False)
def _adx_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ADX - Numba optimized Wilder's implementation.
    Returns: (ADX, +DI, -DI)
    """
    n = len(close)
    adx_out = np.full(n, np.nan, dtype=np.float64)
    plus_di = np.full(n, np.nan, dtype=np.float64)
    minus_di = np.full(n, np.nan, dtype=np.float64)
    
    if n < 2 * period:
        return adx_out, plus_di, minus_di
    
    # True Range, +DM, -DM
    tr = np.empty(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
        
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Wilder's Smoothing
    tr_smooth = np.full(n, np.nan, dtype=np.float64)
    plus_dm_smooth = np.full(n, np.nan, dtype=np.float64)
    minus_dm_smooth = np.full(n, np.nan, dtype=np.float64)
    
    tr_sum = 0.0
    plus_sum = 0.0
    minus_sum = 0.0
    
    for i in range(1, period + 1):
        tr_sum += tr[i]
        plus_sum += plus_dm[i]
        minus_sum += minus_dm[i]
    
    tr_smooth[period] = tr_sum
    plus_dm_smooth[period] = plus_sum
    minus_dm_smooth[period] = minus_sum
    
    for i in range(period + 1, n):
        tr_smooth[i] = tr_smooth[i - 1] - tr_smooth[i - 1] / period + tr[i]
        plus_dm_smooth[i] = plus_dm_smooth[i - 1] - plus_dm_smooth[i - 1] / period + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i - 1] - minus_dm_smooth[i - 1] / period + minus_dm[i]
    
    # +DI, -DI
    for i in range(period, n):
        if tr_smooth[i] > _EPSILON:
            plus_di[i] = 100.0 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100.0 * minus_dm_smooth[i] / tr_smooth[i]
        else:
            plus_di[i] = 0.0
            minus_di[i] = 0.0
    
    # DX
    dx = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > _EPSILON:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0.0
    
    # ADX (smoothed DX)
    start_adx = 2 * period - 1
    if start_adx < n:
        dx_sum = 0.0
        for i in range(period, 2 * period):
            if i < n and not np.isnan(dx[i]):
                dx_sum += dx[i]
        adx_out[start_adx] = dx_sum / period
        
        for i in range(start_adx + 1, n):
            if not np.isnan(dx[i]):
                adx_out[i] = (adx_out[i - 1] * (period - 1) + dx[i]) / period
    
    return adx_out, plus_di, minus_di


def adx(
    df: pl.DataFrame,
    *,
    period: int = 14,
    out_adx: str = "adx",
    out_plus_di: str = "plus_di",
    out_minus_di: str = "minus_di",
) -> pl.DataFrame:
    """Average Directional Index - Numba optimized with +DI/-DI."""
    if not _has_cols(df, ("high", "low", "close")):
        return df.with_columns([
            pl.lit(None).cast(pl.Float64).alias(out_adx),
            pl.lit(None).cast(pl.Float64).alias(out_plus_di),
            pl.lit(None).cast(pl.Float64).alias(out_minus_di),
        ])
    
    h = np.ascontiguousarray(df["high"].to_numpy().astype(np.float64))
    l = np.ascontiguousarray(df["low"].to_numpy().astype(np.float64))
    c = np.ascontiguousarray(df["close"].to_numpy().astype(np.float64))
    
    adx_vals, plus_di_vals, minus_di_vals = _adx_numba(h, l, c, int(period))
    
    return df.with_columns([
        pl.Series(out_adx, adx_vals),
        pl.Series(out_plus_di, plus_di_vals),
        pl.Series(out_minus_di, minus_di_vals),
    ])


# =============================================================================
# ### MACD - MOVING AVERAGE CONVERGENCE DIVERGENCE ###
# =============================================================================


@njit(cache=True, fastmath=True)
def _macd_numba(
    close: np.ndarray, fast: int, slow: int, signal: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MACD - Numba optimized (consolidated).
    Returns: (macd_line, signal_line, histogram)
    """
    n = len(close)
    macd_line = np.full(n, np.nan, dtype=np.float64)
    signal_line = np.full(n, np.nan, dtype=np.float64)
    histogram = np.full(n, np.nan, dtype=np.float64)
    
    if n < slow:
        return macd_line, signal_line, histogram
    
    # Calculate EMAs
    ema_fast = _ema_numba(close, fast)
    ema_slow = _ema_numba(close, slow)
    
    # MACD Line
    for i in range(n):
        if not np.isnan(ema_fast[i]) and not np.isnan(ema_slow[i]):
            macd_line[i] = ema_fast[i] - ema_slow[i]
    
    # Signal Line (EMA of MACD)
    alpha_sig = 2.0 / (signal + 1.0)
    start_idx = -1
    for i in range(n):
        if not np.isnan(macd_line[i]):
            start_idx = i
            signal_line[i] = macd_line[i]
            break
    
    if start_idx >= 0:
        for i in range(start_idx + 1, n):
            if np.isnan(macd_line[i]):
                signal_line[i] = signal_line[i - 1]
            else:
                signal_line[i] = alpha_sig * macd_line[i] + (1.0 - alpha_sig) * signal_line[i - 1]
    
    # Histogram
    for i in range(n):
        if not np.isnan(macd_line[i]) and not np.isnan(signal_line[i]):
            histogram[i] = macd_line[i] - signal_line[i]
    
    return macd_line, signal_line, histogram


def macd(
    df: pl.DataFrame,
    *,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    col: str = "close",
    out_macd: str = "macd",
    out_signal: str = "macd_signal",
    out_hist: str = "macd_hist",
) -> pl.DataFrame:
    """
    MACD (Consolidated) - Returns line, signal, and histogram.
    Numba optimized.
    """
    if col not in df.columns:
        return df.with_columns([
            pl.lit(None).cast(pl.Float64).alias(out_macd),
            pl.lit(None).cast(pl.Float64).alias(out_signal),
            pl.lit(None).cast(pl.Float64).alias(out_hist),
        ])
    
    src = np.ascontiguousarray(df[col].to_numpy().astype(np.float64))
    macd_line, signal_line, histogram = _macd_numba(src, int(fast), int(slow), int(signal))
    
    return df.with_columns([
        pl.Series(out_macd, macd_line),
        pl.Series(out_signal, signal_line),
        pl.Series(out_hist, histogram),
    ])


# =============================================================================
# ### MFI - MONEY FLOW INDEX ###
# =============================================================================


@njit(cache=True, fastmath=False)
def _mfi_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int
) -> np.ndarray:
    """MFI - Volume-weighted RSI. Numba optimized."""
    n = len(close)
    mfi = np.full(n, np.nan, dtype=np.float64)
    
    if n < period + 1:
        return mfi
    
    # Typical Price
    tp = np.empty(n, dtype=np.float64)
    for i in range(n):
        tp[i] = (high[i] + low[i] + close[i]) / 3.0
    
    # Money Flow classification
    pos_mf = np.zeros(n, dtype=np.float64)
    neg_mf = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        mf = tp[i] * volume[i]
        if tp[i] > tp[i - 1]:
            pos_mf[i] = mf
        elif tp[i] < tp[i - 1]:
            neg_mf[i] = mf
    
    # Rolling sum and MFI
    for i in range(period, n):
        pos_sum = 0.0
        neg_sum = 0.0
        
        for j in range(i - period + 1, i + 1):
            pos_sum += pos_mf[j]
            neg_sum += neg_mf[j]
        
        if neg_sum < _EPSILON:
            mfi[i] = 100.0
        else:
            mfr = pos_sum / neg_sum
            mfi[i] = 100.0 - 100.0 / (1.0 + mfr)
    
    return mfi


def mfi(
    df: pl.DataFrame,
    *,
    period: int = 14,
    out: str = "mfi",
) -> pl.DataFrame:
    """Money Flow Index - Numba optimized."""
    if not _has_cols(df, ("high", "low", "close")) or "volume" not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out))
    
    h = np.ascontiguousarray(df["high"].to_numpy().astype(np.float64))
    l = np.ascontiguousarray(df["low"].to_numpy().astype(np.float64))
    c = np.ascontiguousarray(df["close"].to_numpy().astype(np.float64))
    v = np.ascontiguousarray(df["volume"].to_numpy().astype(np.float64))
    
    mfi_vals = _mfi_numba(h, l, c, v, int(period))
    return df.with_columns(pl.Series(out, mfi_vals))


# =============================================================================
# ### SUPERTREND ###
# =============================================================================


@njit(cache=True, fastmath=True)
def _supertrend_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int, mult: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Supertrend - Returns (supertrend_values, direction: 1=bull, -1=bear)."""
    n = len(close)
    st = np.full(n, np.nan, dtype=np.float64)
    direction = np.full(n, 0, dtype=np.int8)

    if n <= period:
        return st, direction

    # True Range
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    # ATR with Wilder's Smoothing
    atr = np.full(n, np.nan, dtype=np.float64)
    atr_sum = 0.0
    for i in range(1, period + 1):
        atr_sum += tr[i]
    atr[period] = atr_sum / period
    
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    # Bands
    basic_ub = np.full(n, np.nan, dtype=np.float64)
    basic_lb = np.full(n, np.nan, dtype=np.float64)
    final_ub = np.full(n, np.nan, dtype=np.float64)
    final_lb = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        if not np.isnan(atr[i]):
            hl2 = 0.5 * (high[i] + low[i])
            basic_ub[i] = hl2 + mult * atr[i]
            basic_lb[i] = hl2 - mult * atr[i]

    start = period
    final_ub[start] = basic_ub[start]
    final_lb[start] = basic_lb[start]
    direction[start] = 1
    st[start] = final_lb[start]

    for i in range(start + 1, n):
        if basic_ub[i] < final_ub[i - 1] or close[i - 1] > final_ub[i - 1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i - 1]

        if basic_lb[i] > final_lb[i - 1] or close[i - 1] < final_lb[i - 1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i - 1]

        prev_dir = direction[i - 1]
        if prev_dir == 1:
            direction[i] = -1 if close[i] < final_lb[i] else 1
        else:
            direction[i] = 1 if close[i] > final_ub[i] else -1

        st[i] = final_lb[i] if direction[i] == 1 else final_ub[i]

    return st, direction


def supertrend(
    df: pl.DataFrame,
    *,
    period: int = 10,
    mult: float = 3.0,
    out: str = "supertrend",
    out_dir: str = "supertrend_dir",
) -> pl.DataFrame:
    """Supertrend - Numba optimized."""
    if not _has_cols(df, ("high", "low", "close")):
        return df.with_columns([
            pl.lit(None).cast(pl.Float64).alias(out),
            pl.lit(None).cast(pl.Int16).alias(out_dir),
        ])

    h = np.ascontiguousarray(df["high"].to_numpy().astype(np.float64))
    l = np.ascontiguousarray(df["low"].to_numpy().astype(np.float64))
    c = np.ascontiguousarray(df["close"].to_numpy().astype(np.float64))

    st, direction = _supertrend_numba(h, l, c, int(period), float(mult))
    
    return df.with_columns([
        pl.Series(out, st),
        pl.Series(out_dir, direction.astype(np.int16)),
    ])


# =============================================================================
# ### STC - SCHAFF TREND CYCLE ###
# =============================================================================


@njit(cache=True, fastmath=False)
def _stc_numba(
    close: np.ndarray, fast: int, slow: int, cycle: int, smooth: int
) -> np.ndarray:
    """
    Schaff Trend Cycle - Numba optimized.
    Double stochastic on MACD.
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    
    if n < max(slow, cycle) + 5:
        return out

    # EMAs
    alpha_fast = 2.0 / (fast + 1.0)
    alpha_slow = 2.0 / (slow + 1.0)
    alpha_smooth = 2.0 / (smooth + 1.0)
    
    ema_fast = np.empty(n, dtype=np.float64)
    ema_slow = np.empty(n, dtype=np.float64)
    
    ema_fast[0] = close[0]
    ema_slow[0] = close[0]
    
    for i in range(1, n):
        ema_fast[i] = alpha_fast * close[i] + (1.0 - alpha_fast) * ema_fast[i - 1]
        ema_slow[i] = alpha_slow * close[i] + (1.0 - alpha_slow) * ema_slow[i - 1]

    # MACD
    macd = ema_fast - ema_slow

    # Stochastic of MACD -> K1
    k1 = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(cycle - 1, n):
        lo = macd[i]
        hi = macd[i]
        
        for j in range(i - cycle + 1, i + 1):
            val = macd[j]
            if val < lo:
                lo = val
            if val > hi:
                hi = val
        
        if hi - lo > _EPSILON:
            k1[i] = 100.0 * (macd[i] - lo) / (hi - lo)

    # Smooth K1 -> D1
    d1 = np.full(n, np.nan, dtype=np.float64)
    start_idx = -1
    for i in range(n):
        if not np.isnan(k1[i]):
            start_idx = i
            d1[i] = k1[i]
            break
    
    if start_idx >= 0:
        for i in range(start_idx + 1, n):
            if np.isnan(k1[i]):
                d1[i] = d1[i - 1]
            else:
                d1[i] = alpha_smooth * k1[i] + (1.0 - alpha_smooth) * d1[i - 1]

    # Stochastic of D1 -> K2
    k2 = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(cycle - 1, n):
        if np.isnan(d1[i]):
            continue
            
        lo = d1[i]
        hi = d1[i]
        valid = True
        
        for j in range(i - cycle + 1, i + 1):
            val = d1[j]
            if np.isnan(val):
                valid = False
                break
            if val < lo:
                lo = val
            if val > hi:
                hi = val
        
        if valid and hi - lo > _EPSILON:
            k2[i] = 100.0 * (d1[i] - lo) / (hi - lo)

    # Smooth K2 -> STC
    start_idx = -1
    for i in range(n):
        if not np.isnan(k2[i]):
            start_idx = i
            out[i] = k2[i]
            break
    
    if start_idx >= 0:
        for i in range(start_idx + 1, n):
            if np.isnan(k2[i]):
                out[i] = out[i - 1]
            else:
                out[i] = alpha_smooth * k2[i] + (1.0 - alpha_smooth) * out[i - 1]

    return out


def stc(
    df: pl.DataFrame,
    *,
    fast: int = 23,
    slow: int = 50,
    cycle: int = 10,
    smooth: int = 3,
    col: str = "close",
    out: str = "stc",
) -> pl.DataFrame:
    """Schaff Trend Cycle - Numba optimized."""
    if col not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out))
    
    c = np.ascontiguousarray(df[col].to_numpy().astype(np.float64))
    stc_vals = _stc_numba(c, int(fast), int(slow), int(cycle), int(smooth))
    return df.with_columns(pl.Series(out, stc_vals))


# =============================================================================
# ### BOLLINGER BANDS ###
# =============================================================================


@njit(cache=True, fastmath=True)
def _bollinger_bands_numba(
    close: np.ndarray, period: int, mult: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands - Returns (basis, upper, lower)."""
    n = len(close)
    basis = np.full(n, np.nan, dtype=np.float64)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    if n < period:
        return basis, upper, lower
    
    for i in range(period - 1, n):
        total = 0.0
        for j in range(i - period + 1, i + 1):
            total += close[j]
        sma = total / period
        basis[i] = sma
        
        var_sum = 0.0
        for j in range(i - period + 1, i + 1):
            diff = close[j] - sma
            var_sum += diff * diff
        std = np.sqrt(var_sum / period)
        
        upper[i] = sma + mult * std
        lower[i] = sma - mult * std
    
    return basis, upper, lower


def bollinger_bands(
    df: pl.DataFrame,
    *,
    period: int = 20,
    mult: float = 2.0,
    col: str = "close",
    out_basis: str = "bb_basis",
    out_upper: str = "bb_upper",
    out_lower: str = "bb_lower",
) -> pl.DataFrame:
    """Bollinger Bands - Numba optimized."""
    if col not in df.columns:
        return df.with_columns([
            pl.lit(None).cast(pl.Float64).alias(out_basis),
            pl.lit(None).cast(pl.Float64).alias(out_upper),
            pl.lit(None).cast(pl.Float64).alias(out_lower),
        ])
    
    src = np.ascontiguousarray(df[col].to_numpy().astype(np.float64))
    basis, upper, lower = _bollinger_bands_numba(src, int(period), float(mult))
    
    return df.with_columns([
        pl.Series(out_basis, basis),
        pl.Series(out_upper, upper),
        pl.Series(out_lower, lower),
    ])


# =============================================================================
# ### KELTNER CHANNELS ###
# =============================================================================


@njit(cache=True, fastmath=True)
def _keltner_channels_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int, mult: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keltner Channels - EMA basis with ATR bands."""
    n = len(close)
    basis = np.full(n, np.nan, dtype=np.float64)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    if n < period:
        return basis, upper, lower
    
    # EMA
    ema = _ema_numba(close, period)
    
    # True Range
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
    
    # ATR (Wilder's)
    atr = np.full(n, np.nan, dtype=np.float64)
    atr_sum = 0.0
    for i in range(period):
        atr_sum += tr[i]
    atr[period - 1] = atr_sum / period
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    # Channels
    for i in range(period - 1, n):
        basis[i] = ema[i]
        if not np.isnan(atr[i]):
            upper[i] = ema[i] + mult * atr[i]
            lower[i] = ema[i] - mult * atr[i]
    
    return basis, upper, lower


def keltner_channels(
    df: pl.DataFrame,
    *,
    period: int = 20,
    mult: float = 1.5,
    out_basis: str = "kc_basis",
    out_upper: str = "kc_upper",
    out_lower: str = "kc_lower",
) -> pl.DataFrame:
    """Keltner Channels - Numba optimized."""
    if not _has_cols(df, ("high", "low", "close")):
        return df.with_columns([
            pl.lit(None).cast(pl.Float64).alias(out_basis),
            pl.lit(None).cast(pl.Float64).alias(out_upper),
            pl.lit(None).cast(pl.Float64).alias(out_lower),
        ])
    
    h = np.ascontiguousarray(df["high"].to_numpy().astype(np.float64))
    l = np.ascontiguousarray(df["low"].to_numpy().astype(np.float64))
    c = np.ascontiguousarray(df["close"].to_numpy().astype(np.float64))
    
    basis, upper, lower = _keltner_channels_numba(h, l, c, int(period), float(mult))
    
    return df.with_columns([
        pl.Series(out_basis, basis),
        pl.Series(out_upper, upper),
        pl.Series(out_lower, lower),
    ])


# =============================================================================
# ### TTM SQUEEZE (Consolidated) ###
# =============================================================================


@njit(cache=True, fastmath=True)
def _linreg_momentum_numba(close: np.ndarray, period: int, smooth: int) -> np.ndarray:
    """Linear Regression Momentum for TTM Squeeze histogram."""
    n = len(close)
    mom = np.full(n, np.nan, dtype=np.float64)
    
    if n < period:
        return mom
    
    for i in range(period - 1, n):
        sum_x = 0.0
        sum_y = 0.0
        sum_xy = 0.0
        sum_xx = 0.0
        
        for j in range(period):
            x = float(j)
            y = close[i - period + 1 + j]
            sum_x += x
            sum_y += y
            sum_xy += x * y
            sum_xx += x * x
        
        n_p = float(period)
        denom = n_p * sum_xx - sum_x * sum_x
        
        if abs(denom) > _EPSILON:
            slope = (n_p * sum_xy - sum_x * sum_y) / denom
            intercept = (sum_y - slope * sum_x) / n_p
            linreg_val = intercept + slope * (n_p - 1.0)
            avg = sum_y / n_p
            mom[i] = linreg_val - avg
    
    # EMA smoothing
    if smooth > 1:
        alpha = 2.0 / (smooth + 1.0)
        smoothed = np.full(n, np.nan, dtype=np.float64)
        
        for i in range(n):
            if not np.isnan(mom[i]):
                smoothed[i] = mom[i]
                for k in range(i + 1, n):
                    if np.isnan(mom[k]):
                        smoothed[k] = smoothed[k - 1]
                    else:
                        smoothed[k] = alpha * mom[k] + (1.0 - alpha) * smoothed[k - 1]
                break
        return smoothed
    
    return mom


@njit(cache=True, fastmath=True)
def _squeeze_state_numba(
    bb_upper: np.ndarray, bb_lower: np.ndarray,
    kc_upper: np.ndarray, kc_lower: np.ndarray
) -> np.ndarray:
    """Squeeze state: 1 when BB inside KC."""
    n = len(bb_upper)
    squeeze = np.zeros(n, dtype=np.int8)
    
    for i in range(n):
        if np.isnan(bb_upper[i]) or np.isnan(kc_upper[i]):
            continue
        if bb_lower[i] > kc_lower[i] and bb_upper[i] < kc_upper[i]:
            squeeze[i] = 1
    
    return squeeze


def ttm_squeeze(
    df: pl.DataFrame,
    *,
    bb_period: int = 20,
    bb_mult: float = 2.0,
    kc_mult: float = 1.5,
    mom_period: int = 20,
    mom_smooth: int = 8,
    out_squeeze: str = "squeeze",
    out_momentum: str = "squeeze_mom",
) -> pl.DataFrame:
    """
    TTM Squeeze (Consolidated) - Returns squeeze state and momentum.
    Numba optimized.
    """
    if not _has_cols(df, ("high", "low", "close")):
        return df.with_columns([
            pl.lit(None).cast(pl.Int8).alias(out_squeeze),
            pl.lit(None).cast(pl.Float64).alias(out_momentum),
        ])
    
    h = np.ascontiguousarray(df["high"].to_numpy().astype(np.float64))
    l = np.ascontiguousarray(df["low"].to_numpy().astype(np.float64))
    c = np.ascontiguousarray(df["close"].to_numpy().astype(np.float64))
    
    # Bollinger Bands
    bb_basis, bb_upper, bb_lower = _bollinger_bands_numba(c, int(bb_period), float(bb_mult))
    
    # Keltner Channels
    kc_basis, kc_upper, kc_lower = _keltner_channels_numba(h, l, c, int(bb_period), float(kc_mult))
    
    # Squeeze state
    squeeze = _squeeze_state_numba(bb_upper, bb_lower, kc_upper, kc_lower)
    
    # Momentum
    momentum = _linreg_momentum_numba(c, int(mom_period), int(mom_smooth))
    
    return df.with_columns([
        pl.Series(out_squeeze, squeeze),
        pl.Series(out_momentum, momentum),
    ])


# =============================================================================
# ### DONCHIAN CHANNELS ###
# =============================================================================


@njit(cache=True, fastmath=True)
def _donchian_numba(high: np.ndarray, low: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Donchian Channels - Returns (upper, lower)."""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    if n < window:
        return upper, lower
    
    for i in range(window - 1, n):
        hi = high[i - window + 1]
        lo = low[i - window + 1]
        for j in range(i - window + 2, i + 1):
            if high[j] > hi:
                hi = high[j]
            if low[j] < lo:
                lo = low[j]
        upper[i] = hi
        lower[i] = lo
    
    return upper, lower


def donchian(
    df: pl.DataFrame,
    *,
    window: int = 20,
    out_hi: str = "donchian_hi",
    out_lo: str = "donchian_lo",
) -> pl.DataFrame:
    """Donchian Channels - Numba optimized."""
    if not _has_cols(df, ("high", "low")):
        return df.with_columns([
            pl.lit(None).cast(pl.Float64).alias(out_hi),
            pl.lit(None).cast(pl.Float64).alias(out_lo),
        ])
    
    h = np.ascontiguousarray(df["high"].to_numpy().astype(np.float64))
    l = np.ascontiguousarray(df["low"].to_numpy().astype(np.float64))
    
    upper, lower = _donchian_numba(h, l, int(window))
    
    return df.with_columns([
        pl.Series(out_hi, upper),
        pl.Series(out_lo, lower),
    ])


# =============================================================================
# ### KALMAN FILTER ###
# =============================================================================


@njit(cache=True, fastmath=True)
def _kalman_numba(src: np.ndarray, length: int, Q: float = 0.1, R: float = 0.01) -> np.ndarray:
    """Kalman Filter - BigBeluga implementation."""
    n = len(src)
    kalman = np.full(n, np.nan, dtype=np.float64)
    
    if n < length:
        return kalman
    
    state_estimate = src[0]
    covariance = 1.0
    
    for i in range(n):
        if np.isnan(src[i]):
            continue
        
        measurement = src[i]
        predicted_state = state_estimate
        predicted_covariance = covariance + Q
        kalman_gain = predicted_covariance / (predicted_covariance + R)
        state_estimate = predicted_state + kalman_gain * (measurement - predicted_state)
        covariance = (1.0 - kalman_gain) * predicted_covariance
        kalman[i] = state_estimate
    
    return kalman


def kalman(
    df: pl.DataFrame,
    *,
    length: int = 50,
    col: str = "close",
    out: str = "kalman",
    Q: float = 0.1,
    R: float = 0.01,
) -> pl.DataFrame:
    """Kalman Filter - Numba optimized."""
    if col not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out))
    
    src = np.ascontiguousarray(df[col].to_numpy().astype(np.float64))
    kalman_vals = _kalman_numba(src, int(length), float(Q), float(R))
    return df.with_columns(pl.Series(out, kalman_vals))


# =============================================================================
# ### NADARAYA-WATSON KERNEL REGRESSION ###
# =============================================================================


@njit(cache=True, fastmath=True)
def _nadaraya_watson_numba(
    src: np.ndarray, h: float, lookback: int, mult: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nadaraya-Watson with Gaussian kernel and envelope bands."""
    n = len(src)
    baseline = np.full(n, np.nan, dtype=np.float64)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    if n < lookback:
        return baseline, upper, lower
    
    for i in range(lookback - 1, n):
        sum_weights = 0.0
        sum_weighted_price = 0.0
        start = max(0, i - lookback + 1)
        
        for j in range(start, i + 1):
            if np.isnan(src[j]):
                continue
            u = float(i - j) / h
            weight = np.exp(-0.5 * u * u)
            sum_weights += weight
            sum_weighted_price += weight * src[j]
        
        if sum_weights > _EPSILON:
            baseline[i] = sum_weighted_price / sum_weights
    
    # Bands from residual std
    for i in range(lookback - 1, n):
        if np.isnan(baseline[i]):
            continue
        
        sum_sq = 0.0
        count = 0
        start = max(0, i - lookback + 1)
        
        for j in range(start, i + 1):
            if not np.isnan(src[j]) and not np.isnan(baseline[j]):
                residual = src[j] - baseline[j]
                sum_sq += residual * residual
                count += 1
        
        if count > 0:
            std = np.sqrt(sum_sq / count)
            upper[i] = baseline[i] + mult * std
            lower[i] = baseline[i] - mult * std
    
    return baseline, upper, lower


def nadaraya_watson(
    df: pl.DataFrame,
    *,
    h: float = 8.0,
    lookback: int = 50,
    mult: float = 2.0,
    col: str = "close",
    out_baseline: str = "nw_baseline",
    out_upper: str = "nw_upper",
    out_lower: str = "nw_lower",
) -> pl.DataFrame:
    """Nadaraya-Watson Kernel Regression - Numba optimized."""
    if col not in df.columns:
        return df.with_columns([
            pl.lit(None).cast(pl.Float64).alias(out_baseline),
            pl.lit(None).cast(pl.Float64).alias(out_upper),
            pl.lit(None).cast(pl.Float64).alias(out_lower),
        ])
    
    src = np.ascontiguousarray(df[col].to_numpy().astype(np.float64))
    baseline, upper, lower = _nadaraya_watson_numba(src, float(h), int(lookback), float(mult))
    
    return df.with_columns([
        pl.Series(out_baseline, baseline),
        pl.Series(out_upper, upper),
        pl.Series(out_lower, lower),
    ])


# =============================================================================
# ### LAGUERRE RSI ###
# =============================================================================


@njit(cache=True, fastmath=True)
def _laguerre_rsi_numba(close: np.ndarray, gamma: float) -> np.ndarray:
    """Ehlers Laguerre RSI - Ultra-low lag. Returns 0-1 range."""
    n = len(close)
    lrsi = np.full(n, np.nan, dtype=np.float64)
    
    if n < 4:
        return lrsi
    
    L0_prev = close[0]
    L1_prev = close[0]
    L2_prev = close[0]
    L3_prev = close[0]
    
    g = gamma
    g1 = 1.0 - gamma
    
    for i in range(n):
        p = close[i]
        
        L0 = g1 * p + g * L0_prev
        L1 = -g * L0 + L0_prev + g * L1_prev
        L2 = -g * L1 + L1_prev + g * L2_prev
        L3 = -g * L2 + L2_prev + g * L3_prev
        
        cu = 0.0
        cd = 0.0
        
        if L0 >= L1:
            cu += L0 - L1
        else:
            cd += L1 - L0
        
        if L1 >= L2:
            cu += L1 - L2
        else:
            cd += L2 - L1
        
        if L2 >= L3:
            cu += L2 - L3
        else:
            cd += L3 - L2
        
        if cu + cd > _EPSILON:
            lrsi[i] = cu / (cu + cd)
        else:
            lrsi[i] = 0.5
        
        L0_prev = L0
        L1_prev = L1
        L2_prev = L2
        L3_prev = L3
    
    return lrsi


def laguerre_rsi(
    df: pl.DataFrame,
    *,
    gamma: float = 0.5,
    col: str = "close",
    out: str = "laguerre_rsi",
) -> pl.DataFrame:
    """Ehlers Laguerre RSI - Numba optimized."""
    if col not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out))
    
    src = np.ascontiguousarray(df[col].to_numpy().astype(np.float64))
    lrsi = _laguerre_rsi_numba(src, float(gamma))
    return df.with_columns(pl.Series(out, lrsi))


# =============================================================================
# ### KAMA - KAUFMAN ADAPTIVE MOVING AVERAGE ###
# =============================================================================


@njit(cache=True, fastmath=True)
def _kama_numba(
    src: np.ndarray, period: int, fast_end: int = 2, slow_end: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """KAMA - Returns (KAMA values, Efficiency Ratio)."""
    n = len(src)
    kama = np.full(n, np.nan, dtype=np.float64)
    er = np.full(n, np.nan, dtype=np.float64)
    
    if n < period + 1:
        return kama, er
    
    fast_sc = 2.0 / (fast_end + 1.0)
    slow_sc = 2.0 / (slow_end + 1.0)
    
    kama[period] = src[period]
    
    for i in range(period, n):
        change = abs(src[i] - src[i - period])
        
        volatility = 0.0
        for j in range(i - period + 1, i + 1):
            volatility += abs(src[j] - src[j - 1])
        
        if volatility > _EPSILON:
            er[i] = change / volatility
        else:
            er[i] = 0.0
        
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        
        if i == period:
            kama[i] = src[i]
        else:
            kama[i] = kama[i - 1] + sc * (src[i] - kama[i - 1])
    
    return kama, er


def kama(
    df: pl.DataFrame,
    *,
    period: int = 10,
    fast_end: int = 2,
    slow_end: int = 30,
    col: str = "close",
    out_kama: str = "kama",
    out_er: str = "er",
) -> pl.DataFrame:
    """Kaufman Adaptive Moving Average - Numba optimized."""
    if col not in df.columns:
        return df.with_columns([
            pl.lit(None).cast(pl.Float64).alias(out_kama),
            pl.lit(None).cast(pl.Float64).alias(out_er),
        ])
    
    src = np.ascontiguousarray(df[col].to_numpy().astype(np.float64))
    kama_vals, er_vals = _kama_numba(src, int(period), int(fast_end), int(slow_end))
    
    return df.with_columns([
        pl.Series(out_kama, kama_vals),
        pl.Series(out_er, er_vals),
    ])


# =============================================================================
# ### ELDER-RAY INDEX ###
# =============================================================================


@njit(cache=True, fastmath=True)
def _elder_ray_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Elder-Ray Index - Returns (EMA, Bull Power, Bear Power)."""
    n = len(close)
    ema_out = _ema_numba(close, period)
    bull_power = np.full(n, np.nan, dtype=np.float64)
    bear_power = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        if not np.isnan(ema_out[i]):
            bull_power[i] = high[i] - ema_out[i]
            bear_power[i] = low[i] - ema_out[i]
    
    return ema_out, bull_power, bear_power


def elder_ray(
    df: pl.DataFrame,
    *,
    period: int = 13,
    out_ema: str = "elder_ema",
    out_bull: str = "bull_power",
    out_bear: str = "bear_power",
) -> pl.DataFrame:
    """Elder-Ray Index - Numba optimized."""
    if not _has_cols(df, ("high", "low", "close")):
        return df.with_columns([
            pl.lit(None).cast(pl.Float64).alias(out_ema),
            pl.lit(None).cast(pl.Float64).alias(out_bull),
            pl.lit(None).cast(pl.Float64).alias(out_bear),
        ])
    
    h = np.ascontiguousarray(df["high"].to_numpy().astype(np.float64))
    l = np.ascontiguousarray(df["low"].to_numpy().astype(np.float64))
    c = np.ascontiguousarray(df["close"].to_numpy().astype(np.float64))
    
    ema_vals, bull_vals, bear_vals = _elder_ray_numba(h, l, c, int(period))
    
    return df.with_columns([
        pl.Series(out_ema, ema_vals),
        pl.Series(out_bull, bull_vals),
        pl.Series(out_bear, bear_vals),
    ])


# =============================================================================
# ### SSL HYBRID ADVANCED (Mihkel00 TradingView v6) ###
# =============================================================================


@njit(cache=True)
def _compute_atr_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """
    Compute ATR using Wilder's smoothing.
    Reusable helper for SSL Hybrid.
    """
    n = len(close)
    atr = np.full(n, np.nan, dtype=np.float64)
    
    if n < period:
        return atr
    
    # True Range
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
    
    # First ATR (SMA)
    total = 0.0
    for i in range(period):
        total += tr[i]
    atr[period - 1] = total / period
    
    # Wilder's smoothing
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


@njit(cache=True)
def _ssl_hybrid_advanced_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    # Baseline params
    baseline_period: int,
    baseline_type: int,  # 0=HMA, 1=EMA, 2=SMA, 3=WMA
    channel_mult: float,
    # SSL1 (Trend) params
    ssl1_period: int,
    # SSL2 (Continuation) params 
    ssl2_period: int,
    # Exit Line params
    exit_period: int,
    # ATR params
    atr_period: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
           np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    SSL Hybrid Advanced - Mihkel00 TradingView v6 implementation.
    
    Returns:
        baseline: Main baseline (HMA/EMA/SMA/WMA)
        upper_channel: Baseline + channel_mult * ATR
        lower_channel: Baseline - channel_mult * ATR
        ssl1_line: SSL1 trend line (Hlv logic)
        ssl2_line: SSL2 continuation line (fast SMA)
        exit_line: Exit trigger line (EMA)
        trend_state: 1=bullish, -1=bearish, 0=neutral
        hlv: Hlv state for SSL1 (1 or -1)
        candle_violation: 1 if candle size > ATR (noise warning)
    """
    n = len(close)
    
    # Output arrays
    baseline = np.full(n, np.nan, dtype=np.float64)
    upper_channel = np.full(n, np.nan, dtype=np.float64)
    lower_channel = np.full(n, np.nan, dtype=np.float64)
    ssl1_line = np.full(n, np.nan, dtype=np.float64)
    ssl2_line = np.full(n, np.nan, dtype=np.float64)
    exit_line = np.full(n, np.nan, dtype=np.float64)
    trend_state = np.zeros(n, dtype=np.int8)
    hlv = np.zeros(n, dtype=np.int8)
    candle_violation = np.zeros(n, dtype=np.int8)
    
    if n < max(baseline_period, ssl1_period, ssl2_period, exit_period, atr_period):
        return (baseline, upper_channel, lower_channel, ssl1_line, ssl2_line,
                exit_line, trend_state, hlv, candle_violation)
    
    # ========================================
    # LAYER 1: BASELINE (Multiple MA Types)
    # ========================================
    if baseline_type == 0:  # HMA (default)
        baseline = _hma_numba(close, baseline_period)
    elif baseline_type == 1:  # EMA
        baseline = _ema_numba(close, baseline_period)
    elif baseline_type == 2:  # SMA
        baseline = _sma_numba(close, baseline_period)
    elif baseline_type == 3:  # WMA
        baseline = _wma_numba(close, baseline_period)
    else:
        baseline = _hma_numba(close, baseline_period)  # Default to HMA
    
    # ========================================
    # ATR FOR CHANNELS AND NOISE FILTER
    # ========================================
    atr = _compute_atr_numba(high, low, close, atr_period)
    
    # ========================================
    # LAYER 2: CHANNELS (Baseline ± mult * ATR)
    # ========================================
    for i in range(n):
        if not np.isnan(baseline[i]) and not np.isnan(atr[i]):
            channel_width = channel_mult * atr[i]
            upper_channel[i] = baseline[i] + channel_width
            lower_channel[i] = baseline[i] - channel_width
    
    # ========================================
    # LAYER 3: SSL1 - TREND (Hlv Logic)
    # ========================================
    # EMA of highs and lows
    ema_high = _ema_numba(high, ssl1_period)
    ema_low = _ema_numba(low, ssl1_period)
    
    # Hlv calculation: cross logic
    # If close > EMA(high) -> Hlv = 1
    # If close < EMA(low) -> Hlv = -1
    # Otherwise -> persist previous Hlv
    prev_hlv = 0
    for i in range(n):
        if np.isnan(ema_high[i]) or np.isnan(ema_low[i]):
            hlv[i] = prev_hlv
            continue
        
        if close[i] > ema_high[i]:
            hlv[i] = 1
        elif close[i] < ema_low[i]:
            hlv[i] = -1
        else:
            hlv[i] = prev_hlv
        
        prev_hlv = hlv[i]
    
    # SSL1 Line: uses Hlv to select high/low EMA
    for i in range(n):
        if hlv[i] == 1:
            ssl1_line[i] = ema_low[i]  # Bullish: use low EMA as support
        elif hlv[i] == -1:
            ssl1_line[i] = ema_high[i]  # Bearish: use high EMA as resistance
        else:
            # Neutral: average
            if not np.isnan(ema_high[i]) and not np.isnan(ema_low[i]):
                ssl1_line[i] = (ema_high[i] + ema_low[i]) / 2.0
    
    # ========================================
    # LAYER 4: SSL2 - CONTINUATION (Fast SMA)
    # ========================================
    # Fast SMA of HL2 (midpoint)
    hl2 = np.empty(n, dtype=np.float64)
    for i in range(n):
        hl2[i] = (high[i] + low[i]) / 2.0
    
    ssl2_line = _sma_numba(hl2, ssl2_period)
    
    # ========================================
    # LAYER 5: EXIT LINE (EMA of close)
    # ========================================
    exit_line = _ema_numba(close, exit_period)
    
    # ========================================
    # LAYER 6: TREND STATE (Color Logic)
    # ========================================
    # Bullish: close > upper_channel
    # Bearish: close < lower_channel
    # Neutral: inside channel
    for i in range(n):
        if np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]):
            trend_state[i] = 0
            continue
        
        if close[i] > upper_channel[i]:
            trend_state[i] = 1  # Bullish
        elif close[i] < lower_channel[i]:
            trend_state[i] = -1  # Bearish
        else:
            trend_state[i] = 0  # Neutral
    
    # ========================================
    # LAYER 7: CANDLE SIZE VIOLATION (Noise Filter)
    # ========================================
    # Flag when candle body > ATR (potential false breakout)
    for i in range(n):
        if np.isnan(atr[i]):
            continue
        
        candle_body = abs(close[i] - (high[i] + low[i]) / 2.0) * 2.0  # Approx body
        candle_range = high[i] - low[i]
        
        if candle_range > atr[i]:
            candle_violation[i] = 1
    
    return (baseline, upper_channel, lower_channel, ssl1_line, ssl2_line,
            exit_line, trend_state, hlv, candle_violation)


def ssl_hybrid_advanced(
    df: pl.DataFrame,
    *,
    # Baseline params
    baseline_period: int = 50,
    baseline_type: str = "HMA",  # HMA, EMA, SMA, WMA
    channel_mult: float = 0.2,
    # SSL1 params
    ssl1_period: int = 14,
    # SSL2 params
    ssl2_period: int = 5,
    # Exit Line params
    exit_period: int = 12,
    # ATR params
    atr_period: int = 14,
    # Output column names
    out_baseline: str = "ssl_baseline",
    out_upper: str = "ssl_upper",
    out_lower: str = "ssl_lower",
    out_ssl1: str = "ssl1_line",
    out_ssl2: str = "ssl2_line",
    out_exit: str = "ssl_exit_line",
    out_trend: str = "ssl_trend",
    out_hlv: str = "ssl_hlv",
    out_violation: str = "ssl_candle_violation",
) -> pl.DataFrame:
    """
    SSL Hybrid Advanced - Mihkel00 TradingView v6 implementation.
    
    Multi-layer trend indicator with:
    - Baseline (HMA/EMA/SMA/WMA) with upper/lower channel
    - SSL1: Trend detection via Hlv cross logic
    - SSL2: Fast continuation signal
    - Exit Line: Position exit trigger
    - Candle Violation: Noise/false breakout warning
    
    Args:
        df: Polars DataFrame with OHLC data
        baseline_period: Period for baseline MA (default 50)
        baseline_type: Type of MA - "HMA", "EMA", "SMA", "WMA"
        channel_mult: Multiplier for ATR channel width (default 0.2)
        ssl1_period: Period for SSL1 trend EMA (default 14)
        ssl2_period: Period for SSL2 continuation SMA (default 5)
        exit_period: Period for exit line EMA (default 12)
        atr_period: Period for ATR calculation (default 14)
        
    Returns:
        DataFrame with all SSL Hybrid columns added
    """
    if not _has_cols(df, ("high", "low", "close")):
        return df.with_columns([
            pl.lit(None).cast(pl.Float64).alias(out_baseline),
            pl.lit(None).cast(pl.Float64).alias(out_upper),
            pl.lit(None).cast(pl.Float64).alias(out_lower),
            pl.lit(None).cast(pl.Float64).alias(out_ssl1),
            pl.lit(None).cast(pl.Float64).alias(out_ssl2),
            pl.lit(None).cast(pl.Float64).alias(out_exit),
            pl.lit(None).cast(pl.Int8).alias(out_trend),
            pl.lit(None).cast(pl.Int8).alias(out_hlv),
            pl.lit(None).cast(pl.Int8).alias(out_violation),
        ])
    
    # Convert to contiguous numpy arrays
    h = np.ascontiguousarray(df["high"].to_numpy().astype(np.float64))
    l = np.ascontiguousarray(df["low"].to_numpy().astype(np.float64))
    c = np.ascontiguousarray(df["close"].to_numpy().astype(np.float64))
    
    # Map baseline type string to int
    type_map = {"HMA": 0, "EMA": 1, "SMA": 2, "WMA": 3}
    baseline_type_int = type_map.get(baseline_type.upper(), 0)
    
    # Call Numba kernel
    (baseline, upper, lower, ssl1, ssl2, exit_ln, 
     trend, hlv_arr, violation) = _ssl_hybrid_advanced_numba(
        h, l, c,
        int(baseline_period),
        baseline_type_int,
        float(channel_mult),
        int(ssl1_period),
        int(ssl2_period),
        int(exit_period),
        int(atr_period),
    )
    
    return df.with_columns([
        pl.Series(out_baseline, baseline),
        pl.Series(out_upper, upper),
        pl.Series(out_lower, lower),
        pl.Series(out_ssl1, ssl1),
        pl.Series(out_ssl2, ssl2),
        pl.Series(out_exit, exit_ln),
        pl.Series(out_trend, trend),
        pl.Series(out_hlv, hlv_arr),
        pl.Series(out_violation, violation),
    ])


# Legacy wrapper for backward compatibility
@njit(cache=True)
def _ssl_hybrid_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SSL Hybrid Channel - Legacy simple version."""
    n = len(close)
    baseline = np.full(n, np.nan, dtype=np.float64)
    ssl_line = np.full(n, np.nan, dtype=np.float64)
    trend = np.zeros(n, dtype=np.int8)
    
    if n < period:
        return baseline, ssl_line, trend
    
    # Use HMA as baseline (consistent with advanced version)
    baseline = _hma_numba(close, period)
    
    # ATR
    atr = _compute_atr_numba(high, low, close, period)
    
    # Trend and SSL line
    for i in range(n):
        if np.isnan(baseline[i]) or np.isnan(atr[i]):
            continue
        
        if close[i] > baseline[i]:
            trend[i] = 1
            ssl_line[i] = baseline[i] - atr[i] * 0.2
        else:
            trend[i] = -1
            ssl_line[i] = baseline[i] + atr[i] * 0.2
    
    return baseline, ssl_line, trend


def ssl_hybrid(
    df: pl.DataFrame,
    *,
    period: int = 50,
    out_baseline: str = "ssl_baseline",
    out_line: str = "ssl_line",
    out_trend: str = "ssl_trend",
) -> pl.DataFrame:
    """SSL Hybrid Channel - Simple version (backward compatible)."""
    if not _has_cols(df, ("high", "low", "close")):
        return df.with_columns([
            pl.lit(None).cast(pl.Float64).alias(out_baseline),
            pl.lit(None).cast(pl.Float64).alias(out_line),
            pl.lit(None).cast(pl.Int8).alias(out_trend),
        ])
    
    h = np.ascontiguousarray(df["high"].to_numpy().astype(np.float64))
    l = np.ascontiguousarray(df["low"].to_numpy().astype(np.float64))
    c = np.ascontiguousarray(df["close"].to_numpy().astype(np.float64))
    
    baseline, ssl_line, trend = _ssl_hybrid_numba(h, l, c, int(period))
    
    return df.with_columns([
        pl.Series(out_baseline, baseline),
        pl.Series(out_line, ssl_line),
        pl.Series(out_trend, trend),
    ])


# =============================================================================
# ### LINEAR REGRESSION SLOPE ###
# =============================================================================


@njit(cache=True, fastmath=False)
def _rolling_linreg_slope_numba(y: np.ndarray, window: int) -> np.ndarray:
    """Rolling Linear Regression Slope."""
    n = len(y)
    out = np.full(n, np.nan, dtype=np.float64)
    
    if window <= 1 or n < window:
        return out

    sum_x = (window - 1) * window / 2.0
    sum_x2 = (window - 1) * window * (2 * window - 1) / 6.0
    denom = window * sum_x2 - sum_x * sum_x
    
    if abs(denom) < _EPSILON:
        return out

    for i in range(window - 1, n):
        sum_y = 0.0
        sum_xy = 0.0
        valid = True
        base = i - window + 1
        
        for j in range(window):
            val = y[base + j]
            if np.isnan(val):
                valid = False
                break
            sum_y += val
            sum_xy += j * val
        
        if valid:
            out[i] = (window * sum_xy - sum_x * sum_y) / denom

    return out


@njit(cache=True, fastmath=False)
def _rolling_linreg_value_numba(y: np.ndarray, window: int) -> np.ndarray:
    """Rolling Linear Regression end-point value (yhat at x=window-1)."""
    n = len(y)
    out = np.full(n, np.nan, dtype=np.float64)

    if window <= 1 or n < window:
        return out

    sum_x = (window - 1) * window / 2.0
    sum_x2 = (window - 1) * window * (2 * window - 1) / 6.0
    denom = window * sum_x2 - sum_x * sum_x

    if abs(denom) < _EPSILON:
        return out

    x_end = window - 1.0

    for i in range(window - 1, n):
        sum_y = 0.0
        sum_xy = 0.0
        valid = True
        base = i - window + 1

        for j in range(window):
            val = y[base + j]
            if np.isnan(val):
                valid = False
                break
            sum_y += val
            sum_xy += j * val

        if not valid:
            continue

        slope = (window * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / window
        out[i] = intercept + slope * x_end

    return out


def linreg_value(
    df: pl.DataFrame,
    *,
    window: int = 20,
    col: str = "close",
    out: str = "linreg_value",
) -> pl.DataFrame:
    """Rolling Linear Regression value (end-point) - Numba optimized."""
    if col not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out))

    y = np.ascontiguousarray(df[col].to_numpy().astype(np.float64))
    vals = _rolling_linreg_value_numba(y, int(window))
    return df.with_columns(pl.Series(out, vals))


def linreg_slope(
    df: pl.DataFrame,
    *,
    window: int = 20,
    col: str = "close",
    out: str = "linreg_slope",
) -> pl.DataFrame:
    """Linear Regression Slope - Numba optimized."""
    if col not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out))
    
    y = np.ascontiguousarray(df[col].to_numpy().astype(np.float64))
    slope = _rolling_linreg_slope_numba(y, int(window))
    return df.with_columns(pl.Series(out, slope))


def lrs_advanced(
    df: pl.DataFrame,
    *,
    clen: int = 50,
    slen: int = 10,
    glen: int = 20,
    col: str = "close",
    out_slope: str = "lrs",
    out_smooth: str = "lrs_smooth",
    out_signal: str = "lrs_signal",
) -> pl.DataFrame:
    """
    Advanced Linear Regression Slope (LRS) with smoothing and signal line.
    
    Based on UCSgears concept:
    - clen: Period for slope calculation (20-80)
    - slen: EMA period for smoothing the slope (5-20)
    - glen: SMA period for signal line ('gray line') (10-30)
    
    Returns:
    - lrs: Raw Linear Regression Slope
    - lrs_smooth: EMA-smoothed slope (slrs)
    - lrs_signal: SMA signal line (gray line)
    
    Trading Logic:
    - LONG: lrs_smooth crosses above 0
    - SHORT: lrs_smooth crosses below 0
    - Exit: lrs_smooth crosses lrs_signal in opposite direction
    """
    if col not in df.columns:
        return df.with_columns([
            pl.lit(None).cast(pl.Float64).alias(out_slope),
            pl.lit(None).cast(pl.Float64).alias(out_smooth),
            pl.lit(None).cast(pl.Float64).alias(out_signal),
        ])
    
    # 1. Calculate raw LRS
    y = np.ascontiguousarray(df[col].to_numpy().astype(np.float64))
    slope = _rolling_linreg_slope_numba(y, int(clen))
    
    # 2. Calculate EMA-smoothed slope (handle NaN by using Polars)
    # Convert numpy NaN to Polars null for proper handling
    slope_series = pl.Series("slope", slope).fill_nan(None)
    
    # Use Polars native EMA/SMA which handle nulls properly
    temp_df = pl.DataFrame({"slope": slope_series})
    
    # EMA smoothing
    temp_df = temp_df.with_columns(
        pl.col("slope").ewm_mean(span=slen, ignore_nulls=True).alias("slope_smooth")
    )
    
    # SMA signal line
    temp_df = temp_df.with_columns(
        pl.col("slope_smooth").rolling_mean(window_size=glen).alias("slope_signal")
    )
    
    slope_smooth = temp_df["slope_smooth"].to_numpy()
    slope_signal = temp_df["slope_signal"].to_numpy()
    
    return df.with_columns([
        pl.Series(out_slope, slope),
        pl.Series(out_smooth, slope_smooth),
        pl.Series(out_signal, slope_signal),
    ])


# =============================================================================
# ### VWMA - VOLUME WEIGHTED MOVING AVERAGE ###
# =============================================================================


def vwma(df: pl.DataFrame, *, window: int, out: str = "vwma") -> pl.DataFrame:
    """Volume Weighted Moving Average - Polars native."""
    if "volume" not in df.columns or "close" not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out))
    
    num = (pl.col("close") * pl.col("volume")).rolling_sum(window)
    den = pl.col("volume").rolling_sum(window)
    
    vwma_expr = _safe_divide(num, den, default=float("nan"))
    return df.with_columns(vwma_expr.alias(out))


# =============================================================================
# ### VWAP SESSION ###
# =============================================================================


def vwap_session(
    df: pl.DataFrame, *, out: str = "vwap_session", ts_col: str = "timestamp"
) -> pl.DataFrame:
    """Session VWAP (resets daily) - Polars native."""
    df = _ensure_timestamp(df, ts_col=ts_col)
    if "volume" not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out))

    day = pl.col(ts_col).dt.date().alias("_session_day")
    tp = ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("_tp")
    pv = (pl.col("_tp") * pl.col("volume")).alias("_pv")

    df2 = df.with_columns([day, tp]).with_columns([pv])
    
    cum_pv = pl.col("_pv").cum_sum().over("_session_day")
    cum_vol = pl.col("volume").cum_sum().over("_session_day")
    
    vwap_expr = _safe_divide(cum_pv, cum_vol, default=float("nan")).alias(out)
    
    return df2.with_columns(vwap_expr).drop(["_pv", "_tp"])


# =============================================================================
# ### Z-SCORE ###
# =============================================================================


def zscore(df: pl.DataFrame, *, col: str, window: int, out: str) -> pl.DataFrame:
    """Z-Score (standardized distance from mean) - Polars native."""
    if col not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out))
    
    mean_expr = pl.col(col).rolling_mean(window)
    std_expr = pl.col(col).rolling_std(window)
    
    zscore_expr = _safe_divide(pl.col(col) - mean_expr, std_expr, default=0.0)
    return df.with_columns(zscore_expr.alias(out))


@njit(cache=True)
def _zscore_ema_numba(
    src: np.ndarray,
    period: int,
) -> np.ndarray:
    """
    Z-Score using EMA for both mean and variance - NO LAG version.
    
    Uses exponential smoothing which weights recent data more heavily,
    resulting in a Z-Score that's perfectly aligned with current price.
    
    Formula:
    - EMA_mean = EMA(price, period)
    - EMA_var = EMA((price - EMA_mean)^2, period)  
    - Z = (price - EMA_mean) / sqrt(EMA_var)
    """
    n = len(src)
    zscore = np.full(n, np.nan, dtype=np.float64)
    
    if n < period:
        return zscore
    
    alpha = 2.0 / (period + 1.0)
    
    # Initialize with SMA for first period
    total = 0.0
    for i in range(period):
        total += src[i]
    ema_mean = total / period
    
    # Initialize variance with first period
    var_sum = 0.0
    for i in range(period):
        diff = src[i] - ema_mean
        var_sum += diff * diff
    ema_var = var_sum / period
    
    # First valid Z-Score
    if ema_var > 1e-10:
        zscore[period - 1] = (src[period - 1] - ema_mean) / np.sqrt(ema_var)
    
    # EMA-based Z-Score for remaining bars
    for i in range(period, n):
        # Update EMA mean
        ema_mean = alpha * src[i] + (1.0 - alpha) * ema_mean
        
        # Current deviation from EMA
        deviation = src[i] - ema_mean
        
        # Update EMA variance (exponentially weighted)
        ema_var = alpha * (deviation * deviation) + (1.0 - alpha) * ema_var
        
        # Z-Score: current price vs EMA mean, normalized by EMA std
        if ema_var > 1e-10:
            zscore[i] = deviation / np.sqrt(ema_var)
        else:
            zscore[i] = 0.0
    
    return zscore


def zscore_ema(
    df: pl.DataFrame,
    *,
    col: str = "close",
    period: int = 50,
    out: str = "zscore",
) -> pl.DataFrame:
    """
    Z-Score using EMA - NO LAG version.
    
    Unlike rolling Z-Score, this version uses exponential moving averages
    which respond immediately to price changes without the lag typical
    of simple moving averages.
    
    The result is a Z-Score that's perfectly synchronized with current price:
    - When price spikes up → Z-Score spikes up immediately
    - When price drops → Z-Score drops immediately
    
    Args:
        col: Source column (default: close)
        period: EMA period (default: 50)
        out: Output column name
    """
    if col not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out))
    
    src = np.ascontiguousarray(df[col].to_numpy().astype(np.float64))
    zscore_vals = _zscore_ema_numba(src, int(period))
    return df.with_columns(pl.Series(out, zscore_vals))


# =============================================================================
# ### CHOPPINESS INDEX ###
# =============================================================================


def choppiness_index(
    df: pl.DataFrame, *, period: int = 14, out: str = "chop"
) -> pl.DataFrame:
    """Choppiness Index - Polars native."""
    if not _has_cols(df, ("high", "low", "close")):
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out))

    prev_close = pl.col("close").shift(1)
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )

    atr_sum = tr.rolling_sum(period)
    highest_high = pl.col("high").rolling_max(period)
    lowest_low = pl.col("low").rolling_min(period)
    hl_range = highest_high - lowest_low

    log_period = np.log10(float(period))
    ratio = _safe_divide(atr_sum, hl_range, default=1.0)
    chop_expr = (100.0 * ratio.log(base=10) / log_period).alias(out)

    return df.with_columns(chop_expr)


# =============================================================================
# ### EFFICIENCY RATIO ###
# =============================================================================


def efficiency_ratio(
    df: pl.DataFrame, *, window: int, col: str = "close", out: str = "er_kaufman"
) -> pl.DataFrame:
    """Kaufman's Efficiency Ratio - Polars native."""
    if col not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out))
    
    change = (pl.col(col) - pl.col(col).shift(window)).abs()
    volatility = pl.col(col).diff().abs().rolling_sum(window)
    
    er_expr = _safe_divide(change, volatility, default=0.0)
    return df.with_columns(er_expr.alias(out))


# =============================================================================
# ### MFI SLOPE ###
# =============================================================================


@njit(cache=True, fastmath=False)
def _mfi_slope_numba(mfi: np.ndarray, lookback: int) -> np.ndarray:
    """MFI Slope - Rate of change of MFI."""
    n = len(mfi)
    slope = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(lookback, n):
        if not np.isnan(mfi[i]) and not np.isnan(mfi[i - lookback]):
            slope[i] = mfi[i] - mfi[i - lookback]
    
    return slope


def mfi_slope(
    df: pl.DataFrame,
    *,
    mfi_col: str = "mfi",
    lookback: int = 5,
    out: str = "mfi_slope",
) -> pl.DataFrame:
    """MFI Slope - Numba optimized."""
    if mfi_col not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out))
    
    mfi_arr = np.ascontiguousarray(df[mfi_col].to_numpy().astype(np.float64))
    slope_vals = _mfi_slope_numba(mfi_arr, int(lookback))
    return df.with_columns(pl.Series(out, slope_vals))


# =============================================================================
# ### MFI 2ND DERIVATIVE (MFI D2) ###
# =============================================================================


@njit(cache=True, fastmath=False)
def _mfi_d2_numba(mfi_arr: np.ndarray) -> np.ndarray:
    """Second derivative (discrete) of MFI: d2[t] = mfi[t] - 2*mfi[t-1] + mfi[t-2]."""
    n = len(mfi_arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 3:
        return out
    for i in range(2, n):
        a = mfi_arr[i]
        b = mfi_arr[i - 1]
        c = mfi_arr[i - 2]
        if np.isnan(a) or np.isnan(b) or np.isnan(c):
            continue
        out[i] = a - 2.0 * b + c
    return out


def mfi_d2(
    df: pl.DataFrame,
    *,
    mfi_col: str = "mfi",
    out: str = "mfi_d2",
) -> pl.DataFrame:
    """MFI second derivative (discrete) computed from an existing MFI column."""
    if mfi_col not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out))
    mfi_arr = np.ascontiguousarray(df[mfi_col].to_numpy().astype(np.float64))
    d2_vals = _mfi_d2_numba(mfi_arr)
    return df.with_columns(pl.Series(out, d2_vals))


# =============================================================================
# ### ACCELERATION OSCILLATOR (Savitzky-Golay Simulated) ###
# =============================================================================


@njit(cache=True, fastmath=True)
def _rolling_zscore_numba(src: np.ndarray, window: int) -> np.ndarray:
    """Rolling Z-Score with strict window (NaN if any NaN in window)."""
    n = len(src)
    out = np.full(n, np.nan, dtype=np.float64)
    if window <= 1 or n < window:
        return out

    for i in range(window - 1, n):
        s = 0.0
        ss = 0.0
        ok = True
        for j in range(i - window + 1, i + 1):
            x = src[j]
            if np.isnan(x):
                ok = False
                break
            s += x
            ss += x * x
        if not ok:
            continue
        mean = s / window
        var = ss / window - mean * mean
        if var <= 0.0:
            out[i] = 0.0
        else:
            out[i] = (src[i] - mean) / np.sqrt(var)
    return out


@njit(cache=True, fastmath=True)
def _accel_sg_norm_numba(close: np.ndarray, wma_period: int, z_window: int) -> np.ndarray:
    """Compute normalized acceleration oscillator in [-3, 3].

    Steps:
      s1 = WMA(close, p)
      s2 = WMA(s1, p)
      s3 = WMA(s2, p)
      accel = s1 - 2*s2 + s3
      z = zscore(accel, z_window)
      out = 3 * tanh(z)
    """
    s1 = _wma_numba(close, int(wma_period))
    s2 = _wma_numba(s1, int(wma_period))
    s3 = _wma_numba(s2, int(wma_period))

    n = len(close)
    accel = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        a = s1[i]
        b = s2[i]
        c = s3[i]
        if np.isnan(a) or np.isnan(b) or np.isnan(c):
            continue
        accel[i] = a - 2.0 * b + c

    z = _rolling_zscore_numba(accel, int(z_window))
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        zi = z[i]
        if np.isnan(zi):
            continue
        out[i] = 3.0 * np.tanh(zi)
    return out


def accel_sg(
    df: pl.DataFrame,
    *,
    wma_period: int = 29,
    z_window: int = 20,
    col: str = "close",
    out: str = "accel_sg",
) -> pl.DataFrame:
    """Savitzky-Golay simulated acceleration oscillator (normalized to [-3, 3])."""
    if col not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out))
    src = np.ascontiguousarray(df[col].to_numpy().astype(np.float64))
    vals = _accel_sg_norm_numba(src, int(wma_period), int(z_window))
    return df.with_columns(pl.Series(out, vals))


# =============================================================================
# ### QUANT 4444 - Derivative/Integral/LinReg Oscillator (Normalized [-3,3]) ###
# =============================================================================


@njit(cache=True, fastmath=True)
def _rolling_sum_nanaware(src: np.ndarray, window: int) -> np.ndarray:
    n = len(src)
    out = np.full(n, np.nan, dtype=np.float64)
    if window <= 0 or n < window:
        return out
    for i in range(window - 1, n):
        s = 0.0
        ok = True
        for j in range(i - window + 1, i + 1):
            v = src[j]
            if np.isnan(v):
                ok = False
                break
            s += v
        if ok:
            out[i] = s
    return out


@njit(cache=True, fastmath=True)
def _rolling_mean_nanaware(src: np.ndarray, window: int) -> np.ndarray:
    n = len(src)
    out = np.full(n, np.nan, dtype=np.float64)
    if window <= 0 or n < window:
        return out
    for i in range(window - 1, n):
        s = 0.0
        ok = True
        for j in range(i - window + 1, i + 1):
            v = src[j]
            if np.isnan(v):
                ok = False
                break
            s += v
        if ok:
            out[i] = s / window
    return out


@njit(cache=True, fastmath=True)
def _quant_4444_norm_numba(
    src: np.ndarray,
    len_reg: int,
    len_int: int,
    len_norm: int,
) -> np.ndarray:
    """Oscilador Quant 4444.

    Replica la idea Pine:
    - lin_reg = linreg(src)
    - derivative = lin_reg - lin_reg[1]
    - integral = sum(src - lin_reg, len_int)
    - raw = derivative + integral/len_int
    - zscore(raw, len_norm)
    - clamp(z, -3, 3)
    """
    n = len(src)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out

    lr = _rolling_linreg_value_numba(src, int(len_reg))

    # derivative
    der = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        a = lr[i]
        b = lr[i - 1]
        if np.isnan(a) or np.isnan(b):
            continue
        der[i] = a - b

    # integral over error
    err = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        p = src[i]
        a = lr[i]
        if np.isnan(p) or np.isnan(a):
            continue
        err[i] = p - a
    integ = _rolling_sum_nanaware(err, int(len_int))

    raw = np.full(n, np.nan, dtype=np.float64)
    inv_len_int = 1.0 / float(len_int) if len_int > 0 else 0.0
    for i in range(n):
        d = der[i]
        it = integ[i]
        if np.isnan(d) or np.isnan(it):
            continue
        raw[i] = d + it * inv_len_int

    mean = _rolling_mean_nanaware(raw, int(len_norm))
    dev = _rolling_std_numba(raw, int(len_norm))

    for i in range(n):
        r = raw[i]
        m = mean[i]
        s = dev[i]
        if np.isnan(r) or np.isnan(m) or np.isnan(s):
            continue
        safe = 1.0 if s == 0.0 else s
        z = (r - m) / safe
        if z > 3.0:
            z = 3.0
        elif z < -3.0:
            z = -3.0
        out[i] = z

    return out


def quant_4444(
    df: pl.DataFrame,
    *,
    len_reg: int = 20,
    len_int: int = 14,
    len_norm: int = 100,
    col: str = "close",
    out: str = "quant_4444",
) -> pl.DataFrame:
    """Quant 4444 Oscillator - Normalized to [-3, 3]."""
    if col not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out))
    src = np.ascontiguousarray(df[col].to_numpy().astype(np.float64))
    vals = _quant_4444_norm_numba(src, int(len_reg), int(len_int), int(len_norm))
    return df.with_columns(pl.Series(out, vals))


# =============================================================================
# ### SUPERINDICADOR 9955 - RSI + STOCH + MOM (Z) Oscillator (Clamped) ###
# =============================================================================


@njit(cache=True, fastmath=True)
def _superindicador_9955_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    len_rsi: int,
    len_stoch: int,
    len_mom: int,
    mom_norm: int,
    amplification: float,
    cap: float,
) -> np.ndarray:
    """SuperIndicador 9955.

    Inspirado en Pine:
    - z_rsi   = (rsi(close, len_rsi) - 50) / 10
    - stoch_d = sma(stoch(close, high, low, len_stoch), 3)
    - z_stoch = (stoch_d - 50) / 15
    - mom_raw = close - close[len_mom]
    - z_mom   = (mom_raw - sma(mom_raw, mom_norm)) / stdev(mom_raw, mom_norm)
    - combined = (z_rsi + z_stoch + z_mom) / 3
    - final = clamp(combined * amplification, -cap, cap)
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out

    rsi_vals = _rsi_numba(close, int(len_rsi))
    _, stoch_d = _stochastic_numba(high, low, close, int(len_stoch), 3)

    mom_raw = np.full(n, np.nan, dtype=np.float64)
    if len_mom > 0:
        for i in range(len_mom, n):
            a = close[i]
            b = close[i - len_mom]
            if np.isnan(a) or np.isnan(b):
                continue
            mom_raw[i] = a - b

    mom_mean = _rolling_mean_nanaware(mom_raw, int(mom_norm))
    mom_dev = _rolling_std_numba(mom_raw, int(mom_norm))

    cap_f = float(cap)
    if cap_f <= 0.0:
        cap_f = 2.0

    for i in range(n):
        r = rsi_vals[i]
        sd = stoch_d[i]
        mr = mom_raw[i]
        mm = mom_mean[i]
        ms = mom_dev[i]

        if np.isnan(r) or np.isnan(sd) or np.isnan(mr) or np.isnan(mm) or np.isnan(ms):
            continue

        z_rsi = (r - 50.0) / 10.0
        z_stoch = (sd - 50.0) / 15.0
        safe = 1.0 if ms == 0.0 else ms
        z_mom = (mr - mm) / safe

        combined = (z_rsi + z_stoch + z_mom) / 3.0
        v = combined * float(amplification)

        if v > cap_f:
            v = cap_f
        elif v < -cap_f:
            v = -cap_f

        out[i] = v

    return out


def superindicador_9955(
    df: pl.DataFrame,
    *,
    len_rsi: int = 14,
    len_stoch: int = 14,
    len_mom: int = 10,
    mom_norm: int = 100,
    amplification: float = 1.6,
    cap: float = 3.0,
    out: str = "super_9955",
) -> pl.DataFrame:
    """SuperIndicador 9955 oscillator.

    Requiere columnas: high/low/close.
    """
    if not _has_cols(df, ("high", "low", "close")):
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out))

    h = np.ascontiguousarray(df["high"].to_numpy().astype(np.float64))
    l = np.ascontiguousarray(df["low"].to_numpy().astype(np.float64))
    c = np.ascontiguousarray(df["close"].to_numpy().astype(np.float64))

    vals = _superindicador_9955_numba(
        h,
        l,
        c,
        int(len_rsi),
        int(len_stoch),
        int(len_mom),
        int(mom_norm),
        float(amplification),
        float(cap),
    )

    return df.with_columns(pl.Series(out, vals))


# =============================================================================
# ### EMA200 MTF (Multi-Timeframe) ###
# =============================================================================


def ema200_mtf(
    df: pl.DataFrame,
    *,
    tf_min: int = 5,
    period: int = 200,
    ts_col: str = "timestamp",
    out: str = "ema200_mtf",
) -> pl.DataFrame:
    """Multi-timeframe EMA200 - Polars native."""
    df = _ensure_timestamp(df, ts_col=ts_col)
    alpha = _alpha_from_period(period)

    every = f"{int(tf_min)}m"
    
    mtf = (
        df.sort(ts_col)
        .group_by_dynamic(ts_col, every=every, closed="right", label="right")
        .agg(pl.col("close").last().alias("_close_mtf"))
        .with_columns(pl.col("_close_mtf").ewm_mean(alpha=alpha, adjust=False).alias(out))
        .select([pl.col(ts_col), pl.col(out)])
        .sort(ts_col)
    )

    return df.sort(ts_col).join_asof(mtf, on=ts_col, strategy="backward")


# =============================================================================
# ### VELOCITY (First Derivative) ###
# =============================================================================


@njit(cache=True)
def _velocity_numba(src: np.ndarray, lookback: int) -> np.ndarray:
    """
    Velocity (First Derivative) - Rate of change over lookback period.
    
    Velocity = (src[i] - src[i-lookback]) / lookback
    Normalized by lookback to make comparable across different periods.
    """
    n = len(src)
    velocity = np.full(n, np.nan, dtype=np.float64)
    
    if n <= lookback:
        return velocity
    
    for i in range(lookback, n):
        if not np.isnan(src[i]) and not np.isnan(src[i - lookback]):
            velocity[i] = (src[i] - src[i - lookback]) / lookback
    
    return velocity


def velocity(
    df: pl.DataFrame,
    *,
    col: str = "close",
    lookback: int = 1,
    out: str = "velocity",
) -> pl.DataFrame:
    """
    Velocity (First Derivative) - Numba optimized.
    
    Calculates the rate of change of a series.
    For price: shows momentum/direction.
    For HMA: shows trend strength.
    
    Args:
        col: Source column
        lookback: Periods to look back (1 = simple diff)
        out: Output column name
    """
    if col not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out))
    
    src = np.ascontiguousarray(df[col].to_numpy().astype(np.float64))
    vel_vals = _velocity_numba(src, int(lookback))
    return df.with_columns(pl.Series(out, vel_vals))


# =============================================================================
# ### ACCELERATION (Second Derivative) ###
# =============================================================================


@njit(cache=True)
def _acceleration_numba(src: np.ndarray, lookback: int) -> np.ndarray:
    """
    Acceleration (Second Derivative) - Change in velocity.
    
    First computes velocity, then computes velocity of velocity.
    Acceleration = d(velocity)/dt
    """
    n = len(src)
    acceleration = np.full(n, np.nan, dtype=np.float64)
    
    if n <= 2 * lookback:
        return acceleration
    
    # First derivative (velocity)
    velocity = np.full(n, np.nan, dtype=np.float64)
    for i in range(lookback, n):
        if not np.isnan(src[i]) and not np.isnan(src[i - lookback]):
            velocity[i] = (src[i] - src[i - lookback]) / lookback
    
    # Second derivative (acceleration)
    for i in range(2 * lookback, n):
        if not np.isnan(velocity[i]) and not np.isnan(velocity[i - lookback]):
            acceleration[i] = (velocity[i] - velocity[i - lookback]) / lookback
    
    return acceleration


def acceleration(
    df: pl.DataFrame,
    *,
    col: str = "close",
    lookback: int = 1,
    out: str = "acceleration",
) -> pl.DataFrame:
    """
    Acceleration (Second Derivative) - Numba optimized.
    
    Calculates the rate of change of velocity.
    Positive acceleration = momentum increasing.
    Negative acceleration = momentum decreasing.
    
    Key signals:
    - Accel crosses from - to + : Bullish momentum shift
    - Accel crosses from + to - : Bearish momentum shift
    
    Args:
        col: Source column (usually HMA or close)
        lookback: Periods to look back
        out: Output column name
    """
    if col not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out))
    
    src = np.ascontiguousarray(df[col].to_numpy().astype(np.float64))
    accel_vals = _acceleration_numba(src, int(lookback))
    return df.with_columns(pl.Series(out, accel_vals))


# =============================================================================
# ### HMA VELOCITY & ACCELERATION (Convenience) ###
# =============================================================================


def hma_velocity(
    df: pl.DataFrame,
    *,
    hma_period: int = 20,
    lookback: int = 1,
    col: str = "close",
    out_hma: str = "hma",
    out_velocity: str = "hma_velocity",
) -> pl.DataFrame:
    """
    HMA Velocity - Compute HMA and its first derivative.
    
    Useful for trend direction detection.
    """
    # First compute HMA if not exists
    if out_hma not in df.columns:
        df = hma(df, period=hma_period, col=col, out=out_hma)
    
    # Then compute velocity
    return velocity(df, col=out_hma, lookback=lookback, out=out_velocity)


def hma_acceleration(
    df: pl.DataFrame,
    *,
    hma_period: int = 20,
    lookback: int = 1,
    col: str = "close",
    out_hma: str = "hma",
    out_velocity: str = "hma_velocity",
    out_acceleration: str = "hma_acceleration",
) -> pl.DataFrame:
    """
    HMA Acceleration - Compute HMA and its second derivative.
    
    Useful for momentum shift detection.
    """
    # First compute HMA if not exists
    if out_hma not in df.columns:
        df = hma(df, period=hma_period, col=col, out=out_hma)
    
    # Compute velocity
    df = velocity(df, col=out_hma, lookback=lookback, out=out_velocity)
    
    # Compute acceleration
    return acceleration(df, col=out_hma, lookback=lookback, out=out_acceleration)


# =============================================================================
# ### SUPERGOLAY (Causal Savitzky–Golay Kinematics + Score) ###
# =============================================================================


def _supergolay_causal_coeffs(*, window: int, polyorder: int, deriv: int) -> np.ndarray:
    """Compute one-sided (causal) Savitzky–Golay FIR coefficients.

    The window uses only past data: y[t-window+1 .. t].
    We fit a polynomial of degree `polyorder` over x = [-(window-1), ..., 0]
    and evaluate the `deriv`-th derivative at x=0.
    """
    w = int(window)
    p = int(polyorder)
    d = int(deriv)

    if w < 2:
        raise ValueError("window must be >= 2")
    if p < 1:
        raise ValueError("polyorder must be >= 1")
    if p >= w:
        raise ValueError("polyorder must be < window")
    if d < 0 or d > p:
        raise ValueError("deriv must satisfy 0 <= deriv <= polyorder")

    x = np.arange(-(w - 1), 1, dtype=np.float64)  # [-w+1, ..., 0]
    # Design matrix: [1, x, x^2, ..., x^p]
    A = np.vander(x, N=p + 1, increasing=True)  # (w, p+1)
    pinv = np.linalg.pinv(A)  # (p+1, w)

    coeff = pinv[d].astype(np.float64, copy=False)
    if d > 0:
        coeff = coeff * float(math.factorial(d))
    return np.ascontiguousarray(coeff)


@njit(cache=True, fastmath=True)
def _apply_fir_causal_numba(src: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    """Apply a causal FIR defined on the last `window` samples."""
    n = len(src)
    w = len(coeff)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < w:
        return out

    for i in range(w - 1, n):
        s = 0.0
        ok = True
        base = i - (w - 1)
        for j in range(w):
            v = src[base + j]
            if np.isnan(v) or np.isinf(v):
                ok = False
                break
            s += v * coeff[j]
        if ok:
            out[i] = s
    return out


@njit(cache=True, fastmath=True)
def _rolling_zscore_numba(src: np.ndarray, window: int) -> np.ndarray:
    n = len(src)
    w = int(window)
    out = np.full(n, np.nan, dtype=np.float64)
    if w <= 1 or n < w:
        return out

    buf = np.empty(w, dtype=np.float64)
    nan_count = 0
    s = 0.0
    s2 = 0.0

    for i in range(n):
        x = src[i]
        idx = i % w

        if i >= w:
            old = buf[idx]
            if np.isnan(old) or np.isinf(old):
                nan_count -= 1
            else:
                s -= old
                s2 -= old * old

        buf[idx] = x
        if np.isnan(x) or np.isinf(x):
            nan_count += 1
        else:
            s += x
            s2 += x * x

        if i >= w - 1 and nan_count == 0:
            mean = s / w
            var = (s2 / w) - mean * mean
            if var < 0.0:
                var = 0.0
            std = np.sqrt(var)
            if std > 1e-12:
                out[i] = (x - mean) / std
            else:
                out[i] = 0.0

    return out


@njit(cache=True, fastmath=True)
def _rolling_linreg_resid_std_numba(src: np.ndarray, window: int) -> np.ndarray:
    """Rolling residual std of y ~ a + b*x over x=0..window-1."""
    n = len(src)
    w = int(window)
    out = np.full(n, np.nan, dtype=np.float64)
    if w <= 2 or n < w:
        return out

    sumx = (w - 1) * w / 2.0
    sumxx = (w - 1) * w * (2.0 * w - 1.0) / 6.0
    denom = w * sumxx - sumx * sumx
    if abs(denom) < 1e-12:
        return out

    buf = np.empty(w, dtype=np.float64)
    nan_count = 0

    s0 = 0.0   # sum(y)
    s1 = 0.0   # sum(x*y)
    s2 = 0.0   # sum(y^2)

    # Warm-up fill
    for i in range(w):
        y = src[i]
        buf[i] = y
        if np.isnan(y) or np.isinf(y):
            nan_count += 1
            continue
        s0 += y
        s1 += i * y
        s2 += y * y

    if nan_count == 0:
        b = (w * s1 - sumx * s0) / denom
        a = (s0 - b * sumx) / w
        sse = s2 + w * a * a + b * b * sumxx + 2.0 * a * b * sumx - 2.0 * a * s0 - 2.0 * b * s1
        if sse < 0.0:
            sse = 0.0
        out[w - 1] = np.sqrt(sse / w)

    for i in range(w, n):
        idx = i % w
        old = buf[idx]
        new = src[i]

        old_is_bad = np.isnan(old) or np.isinf(old)
        new_is_bad = np.isnan(new) or np.isinf(new)

        old_val = 0.0 if old_is_bad else old
        new_val = 0.0 if new_is_bad else new

        # Update nan_count
        if old_is_bad:
            nan_count -= 1
        if new_is_bad:
            nan_count += 1

        # O(1) rolling updates for sums
        s0_old = s0
        s1_old = s1
        s2_old = s2

        s1 = s1_old - s0_old + old_val + (w - 1) * new_val
        s0 = s0_old - old_val + new_val
        s2 = s2_old - old_val * old_val + new_val * new_val

        buf[idx] = new

        if nan_count == 0:
            b = (w * s1 - sumx * s0) / denom
            a = (s0 - b * sumx) / w
            sse = s2 + w * a * a + b * b * sumxx + 2.0 * a * b * sumx - 2.0 * a * s0 - 2.0 * b * s1
            if sse < 0.0:
                sse = 0.0
            out[i] = np.sqrt(sse / w)

    return out


def supergolay(
    df: pl.DataFrame,
    *,
    col: str = "close",
    window: int = 21,
    polyorder: int = 3,
    zscore_window: int = 100,
    noise_window: int = 50,
    w_v: float = 0.5,
    w_a: float = 0.4,
    w_noise: float = 0.3,
    out_smooth: str = "supergolay",
    out_score: str = "supergolay_score",
    out_zv: str = "supergolay_zv",
    out_za: str = "supergolay_za",
    out_noise: str = "supergolay_noise",
) -> pl.DataFrame:
    """Causal Savitzky–Golay smoothing on log(close) + derivatives + master score.

    Outputs:
    - out_smooth: exp(smoothed_log_price)
    - out_zv/out_za: rolling z-scores of velocity/acceleration (window=zscore_window)
    - out_noise: rolling residual std of local linear regression on log(close) (window=noise_window)
    - out_score: w_v*Z_v + w_a*Z_a - w_noise*sigma_eps
    """
    if col not in df.columns:
        return df.with_columns(
            [
                pl.lit(None).cast(pl.Float64).alias(out_smooth),
                pl.lit(None).cast(pl.Float64).alias(out_score),
                pl.lit(None).cast(pl.Float64).alias(out_zv),
                pl.lit(None).cast(pl.Float64).alias(out_za),
                pl.lit(None).cast(pl.Float64).alias(out_noise),
            ]
        )

    w = int(window)
    p = int(polyorder)
    zw = int(zscore_window)
    nw = int(noise_window)

    # Prefer odd windows for SG; enforce basic validity.
    if w < 5:
        w = 5
    if w % 2 == 0:
        w += 1
    if p < 2:
        p = 2
    if p > 4:
        p = 4
    if p >= w:
        p = max(2, min(4, w - 1))
    if zw < 20:
        zw = 20
    if nw < 10:
        nw = 10

    close = np.ascontiguousarray(df[col].to_numpy().astype(np.float64))
    logp = np.full(len(close), np.nan, dtype=np.float64)
    for i in range(len(close)):
        c = close[i]
        if np.isfinite(c) and c > 0.0:
            logp[i] = np.log(c)

    c0 = _supergolay_causal_coeffs(window=w, polyorder=p, deriv=0)
    c1 = _supergolay_causal_coeffs(window=w, polyorder=p, deriv=1)
    c2 = _supergolay_causal_coeffs(window=w, polyorder=p, deriv=2)

    log_smooth = _apply_fir_causal_numba(logp, c0)
    v = _apply_fir_causal_numba(logp, c1)
    a = _apply_fir_causal_numba(logp, c2)

    zv = _rolling_zscore_numba(v, zw)
    za = _rolling_zscore_numba(a, zw)
    noise = _rolling_linreg_resid_std_numba(logp, nw)

    score = (float(w_v) * zv) + (float(w_a) * za) - (float(w_noise) * noise)
    smooth_close = np.exp(log_smooth)

    return df.with_columns(
        [
            pl.Series(out_smooth, smooth_close),
            pl.Series(out_zv, zv),
            pl.Series(out_za, za),
            pl.Series(out_noise, noise),
            pl.Series(out_score, score),
        ]
    )


# =============================================================================
# ### INDICATOR FACTORY (CLEAN REGISTRY) ###
# =============================================================================


@dataclass(frozen=True)
class IndicadorFactory:
    """
    Centralized indicator calculation factory.
    
    Usage:
        config = {
            "rsi": {"activo": True, "period": 14, "out": "rsi"},
            "macd": {"activo": True, "fast": 12, "slow": 26, "signal": 9},
        }
        df = IndicadorFactory.procesar(df, config)
    
    Only calculates indicators where "activo" is True.
    """

    @staticmethod
    def procesar(df: pl.DataFrame, config: Dict[str, Dict[str, Any]]) -> pl.DataFrame:
        """Process DataFrame with configured indicators."""
        df = _ensure_timestamp(df, ts_col="timestamp")

        def _wrap(fn: Callable[..., pl.DataFrame]) -> Callable[[pl.DataFrame, Dict[str, Any]], pl.DataFrame]:
            def _inner(d: pl.DataFrame, cfg: Dict[str, Any]) -> pl.DataFrame:
                kwargs = dict(cfg)
                kwargs.pop("activo", None)
                return fn(d, **kwargs)
            return _inner

        # CLEAN REGISTRY - No duplicates, Numba-first
        registry: Dict[str, Callable[[pl.DataFrame, Dict[str, Any]], pl.DataFrame]] = {
            # Moving Averages
            "ema": _wrap(ema),
            "sma": _wrap(sma),
            "wma": _wrap(wma),
            "hma": _wrap(hma),
            "alma": _wrap(alma),
            "stdev": _wrap(stdev),
            "vwma": _wrap(vwma),
            "kama": _wrap(kama),
            "ema200_mtf": _wrap(ema200_mtf),
            
            # Oscillators
            "rsi": _wrap(rsi),
            "stochastic": _wrap(stochastic),
            "roc": _wrap(roc),
            "dpo": _wrap(dpo),
            "mfi": _wrap(mfi),
            "mfi_slope": _wrap(mfi_slope),
            "mfi_d2": _wrap(mfi_d2),
            "accel_sg": _wrap(accel_sg),
            "chande_mo": _wrap(chande_mo),
            "asgma": _wrap(asgma),
            "stc": _wrap(stc),
            "laguerre_rsi": _wrap(laguerre_rsi),
            "choppiness_index": _wrap(choppiness_index),
            "efficiency_ratio": _wrap(efficiency_ratio),
            
            # Trend
            "adx": _wrap(adx),
            "macd": _wrap(macd),
            "supertrend": _wrap(supertrend),
            "linreg_slope": _wrap(linreg_slope),
            "linreg_value": _wrap(linreg_value),
            "lrs_advanced": _wrap(lrs_advanced),
            
            # Volatility / Bands
            "atr": _wrap(atr),
            "bollinger_bands": _wrap(bollinger_bands),
            "keltner_channels": _wrap(keltner_channels),
            "ttm_squeeze": _wrap(ttm_squeeze),
            "donchian": _wrap(donchian),
            
            # Advanced
            "kalman": _wrap(kalman),
            "nadaraya_watson": _wrap(nadaraya_watson),
            "ssl_hybrid": _wrap(ssl_hybrid),
            "ssl_hybrid_advanced": _wrap(ssl_hybrid_advanced),
            "elder_ray": _wrap(elder_ray),
            
            # Session/Price
            "vwap_session": _wrap(vwap_session),
            "zscore": _wrap(zscore),
            "zscore_ema": _wrap(zscore_ema),
            
            # Derivatives (Kinematics)
            "velocity": _wrap(velocity),
            "acceleration": _wrap(acceleration),
            "hma_velocity": _wrap(hma_velocity),
            "hma_acceleration": _wrap(hma_acceleration),

            # Signal Processing / Kinematics (Causal SG)
            "supergolay": _wrap(supergolay),

            # Quant / Custom
            "quant_4444": _wrap(quant_4444),
            "superindicador_9955": _wrap(superindicador_9955),
        }

        # Handle zscore_vwap alias
        config_final: Dict[str, Dict[str, Any]] = dict(config)
        
        if "zscore_vwap" in config_final and config_final["zscore_vwap"].get("activo", False):
            if "vwap_session" not in config_final:
                config_final["vwap_session"] = {"activo": True, "out": "vwap_session"}
            cfg_alias = dict(config_final["zscore_vwap"])
            cfg_alias["col"] = cfg_alias.get("col", "vwap_session")
            cfg_alias["out"] = cfg_alias.get("out", "zscore_vwap")
            cfg_alias["activo"] = True
            config_final["zscore"] = cfg_alias
            config_final.pop("zscore_vwap", None)

        # Execute in sorted order for reproducibility
        for name in sorted(config_final.keys()):
            cfg = config_final[name]
            if not isinstance(cfg, dict) or not cfg.get("activo", False):
                continue
                
            fn = registry.get(name)
            if fn is None:
                raise ValueError(f"Unsupported indicator: '{name}'")

            df = fn(df, cfg)

        return df
