"""
================================================================================
Strategy 11: MACD_RSI_Sync_Explicit
================================================================================
Estrategia Scalping 5m con corrección de NOMBRES y ASIGNACIÓN de MACD.

CAMBIOS:
- Desempaquetado explícito: macd_line, macd_signal, macd_hist.
- Cálculo de seguridad: Si el histograma falla, se recalcula como Line - Signal.
- Lógica de entrada: Cruce de RSI + Cruce de Histograma a positivo/negativo.

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
# MOTOR MATEMÁTICO (Incluye protección contra NaNs iniciales)
# =============================================================================

@njit(cache=True, fastmath=True)
def _ema_safe(src: np.ndarray, period: int) -> np.ndarray:
    """EMA que salta los valores nulos iniciales."""
    n = len(src)
    ema = np.full(n, np.nan, dtype=np.float64)
    
    # 1. Buscar inicio válido
    start_idx = 0
    while start_idx < n and np.isnan(src[start_idx]):
        start_idx += 1
        
    if n - start_idx < period: return ema
    
    alpha = 2.0 / (period + 1.0)
    
    # 2. SMA Inicial
    total = 0.0
    for i in range(period):
        total += src[start_idx + i]
    
    curr = start_idx + period - 1
    ema[curr] = total / period
    
    # 3. EMA Loop
    for i in range(curr + 1, n):
        val = src[i]
        if np.isnan(val):
            ema[i] = ema[i-1]
        else:
            ema[i] = alpha * val + (1.0 - alpha) * ema[i - 1]
            
    return ema

@njit(cache=True, fastmath=True)
def _macd_explicit(
    close: np.ndarray, fast: int, slow: int, signal: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Retorna explícitamente: (MACD_LINE, SIGNAL_LINE, HISTOGRAM)
    """
    n = len(close)
    macd_line = np.full(n, np.nan, dtype=np.float64)
    signal_line = np.full(n, np.nan, dtype=np.float64)
    histogram = np.full(n, np.nan, dtype=np.float64)
    
    # 1. EMAs
    ema_fast = _ema_safe(close, fast)
    ema_slow = _ema_safe(close, slow)
    
    # 2. MACD Line
    for i in range(n):
        if not np.isnan(ema_fast[i]) and not np.isnan(ema_slow[i]):
            macd_line[i] = ema_fast[i] - ema_slow[i]
            
    # 3. Signal Line (EMA sobre MACD Line)
    # Buscamos donde empieza el MACD válido
    start_sig = 0
    while start_sig < n and np.isnan(macd_line[start_sig]):
        start_sig += 1
        
    if n - start_sig >= signal:
        alpha_sig = 2.0 / (signal + 1.0)
        
        # SMA Inicial para Signal
        sum_sig = 0.0
        for i in range(signal):
            sum_sig += macd_line[start_sig + i]
            
        curr_sig = start_sig + signal - 1
        signal_line[curr_sig] = sum_sig / signal
        
        # EMA Loop para Signal
        for i in range(curr_sig + 1, n):
            val = macd_line[i]
            if np.isnan(val):
                signal_line[i] = signal_line[i-1]
            else:
                signal_line[i] = alpha_sig * val + (1.0 - alpha_sig) * signal_line[i - 1]
                
    # 4. Histograma
    for i in range(n):
        if not np.isnan(macd_line[i]) and not np.isnan(signal_line[i]):
            histogram[i] = macd_line[i] - signal_line[i]
            
    return macd_line, signal_line, histogram

@njit(cache=True, fastmath=True)
def _rsi_safe(close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    rsi_out = np.full(n, np.nan, dtype=np.float64)
    
    start = 0
    while start < n and np.isnan(close[start]):
        start += 1
        
    if n - start < period + 1: return rsi_out
    
    gains = 0.0
    losses = 0.0
    
    for i in range(start + 1, start + period + 1):
        delta = close[i] - close[i-1]
        if delta > 0: gains += delta
        else: losses -= delta
        
    avg_gain = gains / period
    avg_loss = losses / period
    
    idx = start + period
    if avg_loss == 0:
        rsi_out[idx] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi_out[idx] = 100.0 - 100.0 / (1.0 + rs)
        
    for i in range(idx + 1, n):
        delta = close[i] - close[i-1]
        if np.isnan(delta): continue
        
        if delta > 0:
            g = delta
            l = 0.0
        else:
            g = 0.0
            l = -delta
            
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
        
        if avg_loss == 0:
            rsi_out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_out[i] = 100.0 - 100.0 / (1.0 + rs)
            
    return rsi_out

@njit(cache=True, fastmath=True)
def _atr_safe(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    atr = np.full(n, np.nan, dtype=np.float64)
    
    start = 0
    while start < n and np.isnan(close[start]):
        start += 1
        
    if n - start < period: return atr
    
    tr = np.zeros(n, dtype=np.float64)
    for i in range(start + 1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, max(hc, lc))
        
    sum_tr = 0.0
    for i in range(period):
        sum_tr += tr[start + 1 + i]
        
    curr = start + period
    atr[curr] = sum_tr / period
    
    for i in range(curr + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        
    return atr

# =============================================================================
# LÓGICA DE SEÑALES (Sincronización)
# =============================================================================

@njit(cache=True, fastmath=True)
def _generate_signals_explicit(
    rsi: np.ndarray,
    macd_hist: np.ndarray,
    rsi_low: float,
    rsi_high: float,
    window: int,
    warmup: int
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    
    n = len(rsi)
    longs = np.zeros(n, dtype=np.bool_)
    shorts = np.zeros(n, dtype=np.bool_)
    
    last_rsi_up = -999
    last_macd_up = -999
    last_rsi_down = -999
    last_macd_down = -999
    
    c_rsi = 0
    c_macd = 0
    
    for i in range(warmup, n):
        if np.isnan(rsi[i]) or np.isnan(macd_hist[i]) or np.isnan(rsi[i-1]):
            continue
            
        # --- LONG EVENTS ---
        # 1. RSI Cruza hacia ARRIBA (Sale de sobreventa o rebota en nivel bajo)
        if rsi[i-1] <= rsi_low and rsi[i] > rsi_low:
            last_rsi_up = i
            c_rsi += 1
            
        # 2. MACD Histograma Cruza hacia ARRIBA (Pasa de negativo a positivo)
        if macd_hist[i-1] <= 0 and macd_hist[i] > 0:
            last_macd_up = i
            c_macd += 1
            
        # CHECK SYNC LONG
        if (i - last_rsi_up <= window) and (i - last_macd_up <= window):
            # Filtro de coherencia: RSI debe seguir alcista y MACD positivo
            if rsi[i] > rsi_low and macd_hist[i] > 0:
                if not longs[i-1]: # Evitar duplicados
                    longs[i] = True
                    # Consumir eventos para no repetir
                    last_rsi_up = -999
                    last_macd_up = -999
                    
        # --- SHORT EVENTS ---
        # 1. RSI Cruza hacia ABAJO (Sale de sobrecompra)
        if rsi[i-1] >= rsi_high and rsi[i] < rsi_high:
            last_rsi_down = i
            
        # 2. MACD Histograma Cruza hacia ABAJO (Pasa de positivo a negativo)
        if macd_hist[i-1] >= 0 and macd_hist[i] < 0:
            last_macd_down = i
            
        # CHECK SYNC SHORT
        if (i - last_rsi_down <= window) and (i - last_macd_down <= window):
            if rsi[i] < rsi_high and macd_hist[i] < 0:
                if not shorts[i-1]:
                    shorts[i] = True
                    last_rsi_down = -999
                    last_macd_down = -999
                    
    return longs, shorts, c_rsi, c_macd

# =============================================================================
# ESTRATEGIA PRINCIPAL
# =============================================================================

class Strategy11MACDRSISync:
    # Modular indicator declaration for plotting and reporting
    # Update this list with all indicator columns used by this strategy
    __indicators_used = ["macd_hist", "rsi", "atr"]
    combinacion_id = 11
    name = "MACD_RSI_Sync_Explicit"
    
    TIMEOUT_BARS = 300
    
    parametros_optuna = {
        # Configuración Fija (Según Imagen)
        "macd_fast": (6, 6, 1),
        "macd_slow": (13, 13, 1),
        "macd_sig": (4, 4, 1),
        "rsi_period": (7, 7, 1),
        
        # Rangos de Entrada
        "rsi_low": (25, 45, 2),
        "rsi_high": (55, 75, 2),
        "sync_window": (1, 5, 1),
        
        # Salida Dinámica
        "rsi_target_long": (65, 80, 5),
        "rsi_target_short": (20, 35, 5),
        
        # Gestión Riesgo
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
            "rsi_target_long": trial.suggest_int("rsi_target_long", 65, 80),
            "rsi_target_short": trial.suggest_int("rsi_target_short", 20, 35),
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
        
        # 1. Calcular Indicadores (Explicit Unpacking)
        macd_line, macd_sig, macd_hist = _macd_explicit(
            close, 
            int(params["macd_fast"]), 
            int(params["macd_slow"]), 
            int(params["macd_sig"])
        )
        
        # FIX DE EMERGENCIA: Si Hist es NaN pero tenemos Line y Signal, recalculamos
        if np.all(np.isnan(macd_hist)) and not np.all(np.isnan(macd_line)):
            print("[AUTO-FIX] Recalculando Histograma manualmente...")
            macd_hist = macd_line - macd_sig
            
        rsi = _rsi_safe(close, int(params["rsi_period"]))
        atr = _atr_safe(high, low, close, int(params["atr_period"]))
        
        # Warmup
        warmup = int(params["macd_slow"]) + 20
        
        # 2. Generar Señales
        sig_long, sig_short, c_rsi, c_macd = _generate_signals_explicit(
            rsi, macd_hist,
            float(params["rsi_low"]), float(params["rsi_high"]),
            int(params["sync_window"]),
            warmup
        )
        
        # DEBUG
        if sig_long.sum() + sig_short.sum() == 0:
            min_h = np.nanmin(macd_hist) if not np.all(np.isnan(macd_hist)) else "NaN"
            # Solo imprimimos una vez para no saturar
            pass 
            
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