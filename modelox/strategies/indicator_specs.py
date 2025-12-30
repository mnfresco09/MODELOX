from __future__ import annotations

"""
Indicator specs (reutilizables) para estrategias.

Refactored v3.0:
- Removed duplicate specs (macd_hist standalone)
- Added: stochastic, roc
- Clean naming convention

Objective:
- Strategies declare which indicators they use explicitly.
- Avoid config duplication (out/col/params).
- System (Rich/Plot/Export) knows which are active.
"""

from typing import Any, Dict


def cfg_rsi(*, period: int, out: str = "rsi", col: str = "close") -> Dict[str, Any]:
    return {"activo": True, "period": int(period), "col": col, "out": out}


def cfg_macd(
    *,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    col: str = "close",
    out_macd: str = "macd",
    out_signal: str = "macd_signal",
    out_hist: str = "macd_hist",
) -> Dict[str, Any]:
    """
    MACD (Consolidated) - Returns line, signal, and histogram.
    This is the ONLY macd config - no separate macd_hist.
    """
    return {
        "activo": True,
        "fast": int(fast),
        "slow": int(slow),
        "signal": int(signal),
        "col": col,
        "out_macd": out_macd,
        "out_signal": out_signal,
        "out_hist": out_hist,
    }


def cfg_mfi(*, period: int, out: str = "mfi") -> Dict[str, Any]:
    return {"activo": True, "period": int(period), "out": out}


def cfg_ema(*, period: int, out: str = "ema_200", col: str = "close") -> Dict[str, Any]:
    return {"activo": True, "period": int(period), "col": col, "out": out}


def cfg_ema200_mtf(
    *, tf_min: int, period: int = 200, out: str = "ema200_mtf"
) -> Dict[str, Any]:
    return {"activo": True, "tf_min": int(tf_min), "period": int(period), "out": out}


def cfg_vwap_session(*, out: str = "vwap_session") -> Dict[str, Any]:
    return {"activo": True, "out": out}


def cfg_zscore(*, col: str, window: int, out: str) -> Dict[str, Any]:
    return {"activo": True, "col": col, "window": int(window), "out": out}


def cfg_zscore_vwap(*, window: int, out: str = "zscore_vwap") -> Dict[str, Any]:
    return {"activo": True, "window": int(window), "out": out}


def cfg_supertrend(
    *,
    period: int,
    mult: float,
    out: str = "supertrend",
    out_dir: str = "supertrend_dir",
) -> Dict[str, Any]:
    return {
        "activo": True,
        "period": int(period),
        "mult": float(mult),
        "out": out,
        "out_dir": out_dir,
    }


def cfg_vwma(*, window: int, out: str = "vwma") -> Dict[str, Any]:
    return {"activo": True, "window": int(window), "out": out}


def cfg_donchian(
    *, window: int, out_hi: str = "donchian_hi", out_lo: str = "donchian_lo"
) -> Dict[str, Any]:
    return {"activo": True, "window": int(window), "out_hi": out_hi, "out_lo": out_lo}


def cfg_linreg_slope(
    *, window: int, col: str = "close", out: str = "linreg_slope"
) -> Dict[str, Any]:
    return {"activo": True, "window": int(window), "col": col, "out": out}


def cfg_stc(
    *,
    fast: int,
    slow: int,
    cycle: int = 10,
    smooth: int = 3,
    col: str = "close",
    out: str = "stc",
) -> Dict[str, Any]:
    return {
        "activo": True,
        "fast": int(fast),
        "slow": int(slow),
        "cycle": int(cycle),
        "smooth": int(smooth),
        "col": col,
        "out": out,
    }


def cfg_adx(
    *,
    period: int,
    out_adx: str = "adx",
    out_plus_di: str = "plus_di",
    out_minus_di: str = "minus_di",
) -> Dict[str, Any]:
    """ADX - Matches the actual adx() function signature."""
    return {
        "activo": True,
        "period": int(period),
        "out_adx": out_adx,
        "out_plus_di": out_plus_di,
        "out_minus_di": out_minus_di,
    }


def cfg_efficiency_ratio(
    *, window: int, col: str = "close", out: str = "er_kaufman"
) -> Dict[str, Any]:
    return {"activo": True, "window": int(window), "col": col, "out": out}


def cfg_mfi_numba(*, period: int, out: str = "mfi") -> Dict[str, Any]:
    """MFI using Numba-optimized kernel."""
    return {"activo": True, "period": int(period), "out": out}


def cfg_adx_numba(
    *,
    period: int,
    out_adx: str = "adx",
    out_plus_di: str = "plus_di",
    out_minus_di: str = "minus_di",
) -> Dict[str, Any]:
    """ADX using Numba-optimized Wilder's implementation with +DI/-DI."""
    return {
        "activo": True,
        "period": int(period),
        "out_adx": out_adx,
        "out_plus_di": out_plus_di,
        "out_minus_di": out_minus_di,
    }


def cfg_mfi_slope(
    *, mfi_col: str = "mfi", lookback: int = 5, out: str = "mfi_slope"
) -> Dict[str, Any]:
    """MFI Slope for momentum exhaustion detection."""
    return {"activo": True, "mfi_col": mfi_col, "lookback": int(lookback), "out": out}


def cfg_kalman(
    *,
    length: int = 50,
    col: str = "close",
    out: str = "kalman",
    Q: float = 0.1,
    R: float = 0.01,
) -> Dict[str, Any]:
    """Kalman Filter (BigBeluga-style) for price smoothing."""
    return {
        "activo": True,
        "length": int(length),
        "col": col,
        "out": out,
        "Q": float(Q),
        "R": float(R),
    }


def cfg_nadaraya_watson(
    *,
    h: float = 8.0,
    lookback: int = 50,
    mult: float = 2.0,
    col: str = "close",
    out_baseline: str = "nw_baseline",
    out_upper: str = "nw_upper",
    out_lower: str = "nw_lower",
) -> Dict[str, Any]:
    """Nadaraya-Watson Kernel Regression with Gaussian kernel."""
    return {
        "activo": True,
        "h": float(h),
        "lookback": int(lookback),
        "mult": float(mult),
        "col": col,
        "out_baseline": out_baseline,
        "out_upper": out_upper,
        "out_lower": out_lower,
    }


def cfg_hma(
    *,
    period: int = 20,
    col: str = "close",
    out: str = "hma",
) -> Dict[str, Any]:
    """Hull Moving Average for reduced lag smoothing."""
    return {
        "activo": True,
        "period": int(period),
        "col": col,
        "out": out,
    }


def cfg_ssl_hybrid(
    *,
    period: int = 50,
    out_baseline: str = "ssl_baseline",
    out_line: str = "ssl_line",
    out_trend: str = "ssl_trend",
) -> Dict[str, Any]:
    """SSL Hybrid Channel with trend direction (simple version)."""
    return {
        "activo": True,
        "period": int(period),
        "out_baseline": out_baseline,
        "out_line": out_line,
        "out_trend": out_trend,
    }


def cfg_ssl_hybrid_advanced(
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
) -> Dict[str, Any]:
    """
    SSL Hybrid Advanced - Mihkel00 TradingView v6.
    
    Multi-layer trend indicator with:
    - Baseline (HMA/EMA/SMA/WMA) with upper/lower channel
    - SSL1: Trend detection via Hlv cross logic
    - SSL2: Fast continuation signal
    - Exit Line: Position exit trigger
    - Candle Violation: Noise/false breakout warning
    """
    return {
        "activo": True,
        "baseline_period": int(baseline_period),
        "baseline_type": str(baseline_type),
        "channel_mult": float(channel_mult),
        "ssl1_period": int(ssl1_period),
        "ssl2_period": int(ssl2_period),
        "exit_period": int(exit_period),
        "atr_period": int(atr_period),
        "out_baseline": out_baseline,
        "out_upper": out_upper,
        "out_lower": out_lower,
        "out_ssl1": out_ssl1,
        "out_ssl2": out_ssl2,
        "out_exit": out_exit,
        "out_trend": out_trend,
        "out_hlv": out_hlv,
        "out_violation": out_violation,
    }


def cfg_ttm_squeeze(
    *,
    bb_period: int = 20,
    bb_mult: float = 2.0,
    kc_mult: float = 1.5,
    mom_period: int = 20,
    mom_smooth: int = 8,
    out_squeeze: str = "squeeze",
    out_momentum: str = "squeeze_mom",
) -> Dict[str, Any]:
    """TTM Squeeze with momentum histogram."""
    return {
        "activo": True,
        "bb_period": int(bb_period),
        "bb_mult": float(bb_mult),
        "kc_mult": float(kc_mult),
        "mom_period": int(mom_period),
        "mom_smooth": int(mom_smooth),
        "out_squeeze": out_squeeze,
        "out_momentum": out_momentum,
    }


def cfg_laguerre_rsi(
    *,
    gamma: float = 0.5,
    col: str = "close",
    out: str = "laguerre_rsi",
) -> Dict[str, Any]:
    """Ehlers Laguerre RSI for ultra-low lag."""
    return {
        "activo": True,
        "gamma": float(gamma),
        "col": col,
        "out": out,
    }


def cfg_sma(
    *,
    period: int,
    col: str = "close",
    out: str = "sma",
) -> Dict[str, Any]:
    """Simple Moving Average."""
    return {
        "activo": True,
        "period": int(period),
        "col": col,
        "out": out,
    }


def cfg_rsi_numba(
    *,
    period: int = 14,
    col: str = "close",
    out: str = "rsi",
) -> Dict[str, Any]:
    """RSI using Numba-optimized kernel."""
    return {
        "activo": True,
        "period": int(period),
        "col": col,
        "out": out,
    }


def cfg_atr(
    *,
    period: int = 14,
    out: str = "atr",
) -> Dict[str, Any]:
    """Average True Range."""
    return {
        "activo": True,
        "period": int(period),
        "out": out,
    }


def cfg_kama(
    *,
    period: int = 10,
    fast_end: int = 2,
    slow_end: int = 30,
    col: str = "close",
    out_kama: str = "kama",
    out_er: str = "er",
) -> Dict[str, Any]:
    """Kaufman Adaptive Moving Average with Efficiency Ratio."""
    return {
        "activo": True,
        "period": int(period),
        "fast_end": int(fast_end),
        "slow_end": int(slow_end),
        "col": col,
        "out_kama": out_kama,
        "out_er": out_er,
    }


def cfg_elder_ray(
    *,
    period: int = 13,
    out_ema: str = "elder_ema",
    out_bull: str = "bull_power",
    out_bear: str = "bear_power",
) -> Dict[str, Any]:
    """Elder-Ray Index (Bull/Bear Power)."""
    return {
        "activo": True,
        "period": int(period),
        "out_ema": out_ema,
        "out_bull": out_bull,
        "out_bear": out_bear,
    }


# =============================================================================
# NEW INDICATORS (v3.0)
# =============================================================================


def cfg_stochastic(
    *,
    k_period: int = 14,
    d_period: int = 3,
    out_k: str = "stoch_k",
    out_d: str = "stoch_d",
) -> Dict[str, Any]:
    """Stochastic Oscillator (%K, %D)."""
    return {
        "activo": True,
        "k_period": int(k_period),
        "d_period": int(d_period),
        "out_k": out_k,
        "out_d": out_d,
    }


def cfg_roc(
    *,
    period: int = 12,
    col: str = "close",
    out: str = "roc",
) -> Dict[str, Any]:
    """Rate of Change."""
    return {
        "activo": True,
        "period": int(period),
        "col": col,
        "out": out,
    }


def cfg_wma(
    *,
    period: int = 20,
    col: str = "close",
    out: str = "wma",
) -> Dict[str, Any]:
    """Weighted Moving Average."""
    return {
        "activo": True,
        "period": int(period),
        "col": col,
        "out": out,
    }


def cfg_choppiness(
    *,
    period: int = 14,
    out: str = "chop",
) -> Dict[str, Any]:
    """Choppiness Index."""
    return {
        "activo": True,
        "period": int(period),
        "out": out,
    }


def cfg_zscore_vwap(
    *,
    window: int = 20,
    out: str = "zscore_vwap",
) -> Dict[str, Any]:
    """Z-Score of VWAP (requires vwap_session)."""
    return {
        "activo": True,
        "window": int(window),
        "out": out,
    }


def cfg_zscore_ema(
    *,
    col: str = "close",
    period: int = 50,
    out: str = "zscore",
) -> Dict[str, Any]:
    """
    Z-Score using EMA - NO LAG version.
    
    Unlike rolling Z-Score, this uses exponential moving averages
    which respond immediately to price changes.
    The result is perfectly synchronized with current price.
    """
    return {
        "activo": True,
        "col": col,
        "period": int(period),
        "out": out,
    }


# =============================================================================
# KINEMATICS (Derivatives)
# =============================================================================


def cfg_velocity(
    *,
    col: str = "close",
    lookback: int = 1,
    out: str = "velocity",
) -> Dict[str, Any]:
    """
    Velocity (First Derivative) - Rate of change.
    
    Measures momentum direction and strength.
    """
    return {
        "activo": True,
        "col": col,
        "lookback": int(lookback),
        "out": out,
    }


def cfg_acceleration(
    *,
    col: str = "close",
    lookback: int = 1,
    out: str = "acceleration",
) -> Dict[str, Any]:
    """
    Acceleration (Second Derivative) - Change in velocity.
    
    Key for detecting momentum shifts:
    - Accel crosses from - to + : Bullish momentum shift
    - Accel crosses from + to - : Bearish momentum shift
    """
    return {
        "activo": True,
        "col": col,
        "lookback": int(lookback),
        "out": out,
    }


def cfg_hma_velocity(
    *,
    hma_period: int = 20,
    lookback: int = 1,
    col: str = "close",
    out_hma: str = "hma",
    out_velocity: str = "hma_velocity",
) -> Dict[str, Any]:
    """
    HMA Velocity - Hull MA with first derivative.
    
    Useful for trend direction detection with low lag.
    """
    return {
        "activo": True,
        "hma_period": int(hma_period),
        "lookback": int(lookback),
        "col": col,
        "out_hma": out_hma,
        "out_velocity": out_velocity,
    }


def cfg_hma_acceleration(
    *,
    hma_period: int = 20,
    lookback: int = 1,
    col: str = "close",
    out_hma: str = "hma",
    out_velocity: str = "hma_velocity",
    out_acceleration: str = "hma_acceleration",
) -> Dict[str, Any]:
    """
    HMA Acceleration - Hull MA with both derivatives.
    
    Computes:
    - HMA (trend line)
    - HMA Velocity (1st derivative - direction)
    - HMA Acceleration (2nd derivative - momentum shift)
    """
    return {
        "activo": True,
        "hma_period": int(hma_period),
        "lookback": int(lookback),
        "col": col,
        "out_hma": out_hma,
        "out_velocity": out_velocity,
        "out_acceleration": out_acceleration,
    }
