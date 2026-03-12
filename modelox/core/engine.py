"""modelox/core/engine.py

Vector engine (Polars + Numba + C) consolidado - ULTRA OPTIMIZADO.

════════════════════════════════════════════════════════════════════════════════
                    SISTEMA DE SALIDAS - MÁXIMA VELOCIDAD
════════════════════════════════════════════════════════════════════════════════

ARQUITECTURA DE SALIDAS:
╔══════════════════════════════════════════════════════════════════════════════╗
║  FLUJO DE CONFIGURACIÓN Y EJECUCIÓN                                          ║
║                                                                              ║
║  exits.py (CONFIGURACIÓN)                                                   ║
║    └── Defaults, rangos, resolve_exit_settings_for_trial()                  ║
║                    │                                                        ║
║                    ▼                                                        ║
║  runner.py (ORQUESTACIÓN)                                                   ║
║    └── Inyecta __exit_* en params por trial                                ║
║    └── Carga datos de 1m para salidas si TF entrada > 1m                   ║
║                    │                                                        ║
║                    ▼                                                        ║
║  engine.py (EJECUCIÓN NUMBA - MÁXIMA VELOCIDAD)  ← ESTE ARCHIVO            ║
║    └── _simulate_trades_sequential(): Kernel Numba optimizado               ║
║    └── SL/TP/Trailing integrados en el kernel (zero-overhead)              ║
║    └── SALIDAS SIEMPRE EN 1M: Precisión tick-a-tick                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

SALIDAS EN TIMEFRAME DE 1 MINUTO:
    Independientemente del timeframe usado para las ENTRADAS (5m, 15m, 1h, etc.),
    las SALIDAS (SL/TP/Trailing/Custom) SIEMPRE se evalúan usando velas de 1 minuto.
    
    Esto garantiza:
    - Precisión casi tick-a-tick sin perder rendimiento
    - El SL se ejecuta en el momento exacto que se toca, no al cierre de la vela
    - Resultados de backtest más realistas y cercanos al trading en vivo

OPTIMIZACIONES APLICADAS:
- Kernel Numba JIT compilado (@njit cache=True, fastmath=True)
- Lógica SL/TP/Trailing INLINE en el kernel (sin llamadas a funciones)
- Extensión C nativa via Cython como alternativa (5-10x más rápido)
- Zero-copy arrays via views de numpy
- Pre-allocación de memoria para trades
- Eliminación de JOINs de Polars
- Edge-trigger vectorizado sin materialización

NOTA: Las funciones decide_exit_pnl_* en exits.py son para DEBUGGING/AUDITORÍA,
      NO se usan en producción. El engine ejecuta su propia lógica Numba.

PARA MÁXIMO RENDIMIENTO:
    cd cp && python setup.py build_ext --inplace
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numba as nb
import numpy as np
import polars as pl

from modelox.core.types import BacktestConfig, Strategy, suffix_to_minutes
from modelox.core.exits import (
    DEFAULT_EXIT_ATR_PERIOD,
    DEFAULT_EXIT_ATR_MIN_PCT,
    DEFAULT_EXIT_ATR_MAX_PCT,
    DEFAULT_EXIT_ATR_LOOKBACK,
)


# Motor: Numba JIT (puro Python, sin extensiones C)



# =============================================================================
# 1. CACHÉ GLOBAL DE ARRAYS (EVITA RECREACIÓN EN CADA BACKTEST)
# =============================================================================
_ARRAY_CACHE: Dict[int, Dict[str, np.ndarray]] = {}
_CACHE_MAX_SIZE = 4


def _get_cached_arrays(df_id: int, df: pl.DataFrame) -> Dict[str, np.ndarray]:
    """Obtiene arrays cacheados o los crea."""
    if df_id in _ARRAY_CACHE:
        return _ARRAY_CACHE[df_id]
    
    # Crear arrays (zero-copy cuando es posible)
    arrays = {
        "close": df["close"].to_numpy(),
        "high": df["high"].to_numpy() if "high" in df.columns else df["close"].to_numpy(),
        "low": df["low"].to_numpy() if "low" in df.columns else df["close"].to_numpy(),
    }
    
    # Limpiar cache si está llena
    if len(_ARRAY_CACHE) >= _CACHE_MAX_SIZE:
        _ARRAY_CACHE.pop(next(iter(_ARRAY_CACHE)))
    
    _ARRAY_CACHE[df_id] = arrays
    return arrays


# =============================================================================
# 2. PARÁMETROS DE EJECUCIÓN
# =============================================================================

@dataclass
class BacktestParams:
    """PARÁMETROS DE EJECUCIÓN PARA UN BACKTEST INDIVIDUAL."""
    saldo_inicial: float
    comision_pct: float
    comision_sides: int
    saldo_minimo_operativo: float
    saldo_usado: float
    apalancamiento_max: float
    exit_type: str
    exit_sl_pct: float
    exit_tp_pct: float
    exit_trail_act_pct: float
    exit_trail_dist_pct: float
    exit_time_bars: int
    time_stop_bars: int
    # ── ATR adaptive ──
    exit_atr_period:   int   = DEFAULT_EXIT_ATR_PERIOD
    exit_atr_min_pct:  float = DEFAULT_EXIT_ATR_MIN_PCT   # % stake mínimo (igual que sl_pct)
    exit_atr_max_pct:  float = DEFAULT_EXIT_ATR_MAX_PCT   # % stake máximo (igual que sl_pct)
    exit_atr_lookback: int   = DEFAULT_EXIT_ATR_LOOKBACK
    # ── Régimen Buy&Hold (runtime, opcional) ──
    regime_buy_hold_mode: bool = False
    regime_buy_hold_saldo_all: bool = True
    regime_buy_hold_saldo: float = 0.0

    @classmethod
    def from_config_and_params(cls, config: BacktestConfig, params: Dict[str, Any]) -> "BacktestParams":
        return cls(
            saldo_inicial=float(config.saldo_inicial),
            comision_pct=float(config.comision_pct),
            comision_sides=int(getattr(config, "comision_sides", 2)),
            saldo_minimo_operativo=float(config.saldo_minimo_operativo),
            saldo_usado=float(getattr(config, "saldo_usado", 75.0)),
            apalancamiento_max=float(getattr(config, "apalancamiento_max", 60.0)),
            exit_type=str(params.get("__exit_type", getattr(config, "exit_type", "FIXED"))),
            exit_sl_pct=float(params.get("__exit_sl_pct", getattr(config, "exit_sl_pct", 0.0))),
            exit_tp_pct=float(params.get("__exit_tp_pct", getattr(config, "exit_tp_pct", 0.0))),
            exit_trail_act_pct=float(params.get("__exit_trail_act_pct", getattr(config, "exit_trail_act_pct", 0.0))),
            exit_trail_dist_pct=float(params.get("__exit_trail_dist_pct", getattr(config, "exit_trail_dist_pct", 0.0))),
            exit_time_bars=int(params.get("__exit_time_bars", getattr(config, "exit_time_bars", 0))),
            time_stop_bars=int(params.get("time_stop_bars", params.get("__exit_time_bars", 0))),
            exit_atr_period=int(params.get("__exit_atr_period", getattr(config, "exit_atr_period", DEFAULT_EXIT_ATR_PERIOD))),
            exit_atr_min_pct=float(params.get("__exit_atr_min_pct", getattr(config, "exit_atr_min_pct", DEFAULT_EXIT_ATR_MIN_PCT))),
            exit_atr_max_pct=float(params.get("__exit_atr_max_pct", getattr(config, "exit_atr_max_pct", DEFAULT_EXIT_ATR_MAX_PCT))),
            exit_atr_lookback=int(params.get("__exit_atr_lookback", getattr(config, "exit_atr_lookback", DEFAULT_EXIT_ATR_LOOKBACK))),
            regime_buy_hold_mode=bool(params.get("__regime_buy_hold_mode", False)),
            regime_buy_hold_saldo_all=bool(params.get("__regime_buy_hold_saldo_all", True)),
            regime_buy_hold_saldo=float(params.get("__regime_buy_hold_saldo", 0.0) or 0.0),
        )


# =============================================================================
# 3. ATR ADAPTIVE — HELPER DE PRE-CÓMPUTO (Python puro, antes de Numba)
# =============================================================================

def _compute_atr_adaptive_distances(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    entry_indices: np.ndarray,
    entry_prices: np.ndarray,
    apalancamiento_max: float,
    atr_period: int,
    atr_min_pct: float,   # % stake mínimo (ej. 20.0) — mismo formato que sl_pct
    atr_max_pct: float,   # % stake máximo (ej. 40.0) — mismo formato que sl_pct
    atr_lookback: int,
) -> tuple:
    """
    Pre-computa distancias SL/TP adaptativas por entrada, en unidades de precio.

    El ATR(14) se usa como INDICADOR DE VOLATILIDAD para determinar qué %
    del stake aplicar. El rango [atr_min_pct, atr_max_pct] es equivalente
    a sl_pct/tp_pct en modo FIXED, pero el valor se ajusta automáticamente.

    Lógica:
        1. ATR(atr_period) Wilder — mide volatilidad absoluta de precios.
        2. ATR relativo = ATR / close — normaliza por nivel de precio.
        3. Rolling min/max sobre atr_lookback barras → vol_rank ∈ [0, 1]:
               vol_rank = 0  → volatilidad baja  → sl_pct = atr_min_pct (ej. 20%)
               vol_rank = 1  → volatilidad alta  → sl_pct = atr_max_pct (ej. 40%)
               vol_rank = 0.5 → volatilidad media → sl_pct ≈ 30%
        4. sl_pct_adaptive = atr_min_pct + (atr_max_pct - atr_min_pct) × vol_rank
           → Mismo concepto que sl_pct en modo FIXED, pero dinámico.
        5. Distancia de precio (equivalente al kernel):
               sl_distance = (sl_pct_adaptive / 100) × entry_price / apalancamiento
           Derivado de: sl_distance = (stake × sl_pct/100) / qty
                        qty = stake × leverage / entry_p
                    → sl_distance = (sl_pct/100) × entry_p / leverage

    Ejemplo con leverage=60, entry_price=50000, vol_rank=0.5:
        sl_pct_adaptive = 20 + (40-20)×0.5 = 30  (30% del stake)
        sl_distance = (30/100) × 50000 / 60 = 250 USD desde precio de entrada

    Returns:
        (sl_dists, tp_dists): arrays float64 de longitud len(entry_indices),
        en unidades de precio (distancia absoluta SL/TP desde el precio de entrada).
    """
    n = len(close)

    # ── True Range (vectorizado) ──────────────────────────────────────────────
    prev_close = np.empty(n)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]

    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
    )

    # ── ATR Wilder EWM (puro numpy, sin pandas) ─────────────────────────────
    # Equivalente a pd.Series(tr).ewm(com=period-1, adjust=False).mean()
    # alpha = 1 / period  (Wilder smoothing)
    alpha = 1.0 / atr_period
    atr = np.empty(n, dtype=np.float64)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i - 1]

    # ── ATR relativo: normaliza volatilidad por nivel de precio ──────────────
    close_safe = np.where(np.abs(close) > 1e-10, close, 1.0)
    atr_pct = atr / close_safe

    # ── Rolling min/max (puro numpy, sin pandas) ─────────────────────────────
    # Equivalente a pd.Series(atr_pct).rolling(lookback, min_periods=1).min()/max()
    roll_min = np.empty(n, dtype=np.float64)
    roll_max = np.empty(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - atr_lookback + 1)
        window = atr_pct[start:i + 1]
        roll_min[i] = np.min(window)
        roll_max[i] = np.max(window)

    rng = roll_max - roll_min
    safe_rng = np.where(rng > 1e-10, rng, 1.0)   # evita división por cero
    vol_rank = np.where(rng > 1e-10, (atr_pct - roll_min) / safe_rng, 0.5)
    vol_rank = np.clip(vol_rank, 0.0, 1.0)

    # ── % del stake adaptativo en [atr_min_pct, atr_max_pct] ─────────────────
    valid_idx = np.clip(entry_indices, 0, n - 1)
    sl_pct_adaptive = atr_min_pct + (atr_max_pct - atr_min_pct) * vol_rank[valid_idx]

    # ── Distancia de precio: (sl_pct/100) × entry_price / leverage ───────────
    lev = max(apalancamiento_max, 1.0)
    dists = (sl_pct_adaptive / 100.0 * entry_prices / lev).astype(np.float64)

    return dists, dists.copy()



# =============================================================================
# 4. KERNEL NUMBA: SIMULACIÓN CON SALIDAS EN 1M
# =============================================================================

@nb.njit(cache=True, fastmath=True, parallel=False)
def _simulate_trades_with_1m_exits(
    # Datos del timeframe de entrada (para señales)
    entry_indices: np.ndarray,
    entry_prices: np.ndarray,
    entry_types: np.ndarray,
    entry_is_buy_hold: np.ndarray,
    entry_timestamps: np.ndarray,  # Timestamps del TF de entrada
    # Datos de 1m para salidas
    close_1m: np.ndarray,
    high_1m: np.ndarray,
    low_1m: np.ndarray,
    timestamps_1m: np.ndarray,  # Timestamps de 1m
    # Parámetros de capital
    saldo_inicial: float,
    fee_rate: float,
    min_op: float,
    apalancamiento_max: float,    # Para trades B&H
    saldo_usado_cfg: float,       # Para trades B&H
    apalancamiento_strat: float,  # Para trades de estrategia
    saldo_usado_strat: float,     # Para trades de estrategia
    # Parámetros de salida
    is_trailing: bool,
    sl_pct: float,
    tp_pct: float,
    trail_act_pct: float,
    trail_dist_pct: float,
    time_stop_bars: int,  # En barras del TF de entrada
    comision_sides: int,
    # Señales de salida personalizadas (en resolución 1m)
    custom_exit_long_1m: np.ndarray,
    custom_exit_short_1m: np.ndarray,
    # Ratio de timeframes
    tf_ratio: int,  # Cuántas velas de 1m por vela del TF de entrada
    # ATR adaptive: distancias por entrada (longitud n_entries o 0 si no aplica)
    atr_sl_dist_arr: np.ndarray,
    atr_tp_dist_arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Kernel Numba optimizado para simular trades con SALIDAS EN 1M.
    
    Las ENTRADAS se detectan en el timeframe configurado (5m, 15m, 1h, etc.)
    pero las SALIDAS (SL/TP/Trailing/Custom) se evalúan vela a vela en 1m
    para máxima precisión.
    
    IMPORTANTE:
    - entry_indices son índices en el TF de entrada
    - Las salidas se buscan en el array de 1m
    - El mapeo se hace multiplicando entry_idx * tf_ratio
    """
    n_entries = len(entry_indices)
    n_bars_1m = len(close_1m)
    
    # Pre-allocate output arrays
    out_entry_idx = np.empty(n_entries, dtype=np.int64)
    out_exit_idx = np.empty(n_entries, dtype=np.int64)  # Índice en 1m
    out_entry_price = np.empty(n_entries, dtype=np.float64)
    out_exit_price = np.empty(n_entries, dtype=np.float64)
    out_side = np.empty(n_entries, dtype=np.int64)
    out_reason = np.empty(n_entries, dtype=np.int32)
    out_qty = np.empty(n_entries, dtype=np.float64)
    out_saldo_usado = np.empty(n_entries, dtype=np.float64)
    out_pnl_neto = np.empty(n_entries, dtype=np.float64)
    out_pnl_pct = np.empty(n_entries, dtype=np.float64)
    out_saldo_antes = np.empty(n_entries, dtype=np.float64)
    out_saldo_despues = np.empty(n_entries, dtype=np.float64)
    out_trail_act_idx = np.full(n_entries, -1, dtype=np.int64)
    out_trail_act_price = np.full(n_entries, np.nan, dtype=np.float64)
    
    current_balance = saldo_inicial
    last_exit_idx_1m = -1  # Último índice de salida en 1m
    trade_count = 0
    
    for i in range(n_entries):
        entry_idx_tf = entry_indices[i]  # Índice en el TF de entrada
        
        # Convertir a índice en 1m (inicio de la vela del TF) usando timestamps para evitar desfases
        entry_ts = entry_timestamps[entry_idx_tf]
        entry_idx_1m = int(np.searchsorted(timestamps_1m, entry_ts))
        if entry_idx_1m >= n_bars_1m:
            entry_idx_1m = n_bars_1m - 1
        
        # Skip si la entrada está antes de la salida del trade anterior
        if entry_idx_1m <= last_exit_idx_1m:
            continue
        
        # STOP si el saldo ya bajó al mínimo operativo
        if current_balance <= min_op:
            break
        
        entry_p = entry_prices[i]
        side = entry_types[i]
        is_bh_trade = entry_is_buy_hold[i]
        
        # Calcular saldo_usado real (B&H usa config BH, estrategia usa config estrategia)
        _saldo_cfg = saldo_usado_cfg if is_bh_trade else saldo_usado_strat
        _apal = apalancamiento_max if is_bh_trade else apalancamiento_strat
        saldo_usado = min(_saldo_cfg, current_balance)

        # Calcular qty dinámica
        volumen_max = saldo_usado * _apal
        qty = volumen_max / entry_p if entry_p > 0 else 0.0
        
        if qty <= 0 or saldo_usado <= 0:
            continue
        
        # Calcular distancias de precio (ATR adaptive o % sobre stake)
        if len(atr_sl_dist_arr) > 0 and not is_bh_trade:
            sl_distance = atr_sl_dist_arr[i]
            tp_distance = atr_tp_dist_arr[i]
        else:
            # Buy&Hold por régimen: SL fijo de seguridad del 25%
            _sl_pct = 25.0 if is_bh_trade else sl_pct
            _tp_pct = 0.0 if is_bh_trade else tp_pct
            sl_distance = (saldo_usado * _sl_pct / 100.0) / qty
            tp_distance = (saldo_usado * _tp_pct / 100.0) / qty
        _trail_act_pct = 0.0 if is_bh_trade else trail_act_pct
        _trail_dist_pct = 0.0 if is_bh_trade else trail_dist_pct
        trail_act_distance = (saldo_usado * _trail_act_pct / 100.0) / qty
        trail_dist_distance = (saldo_usado * _trail_dist_pct / 100.0) / qty

        if side == 1:  # LONG
            sl_price = entry_p - sl_distance
            tp_price = entry_p + tp_distance
            activation_price = entry_p + trail_act_distance
        else:  # SHORT
            sl_price = entry_p + sl_distance
            tp_price = entry_p - tp_distance
            activation_price = entry_p - trail_act_distance

        trailing_active = False
        trailing_level = 0.0

        # Calcular límite de búsqueda en 1m
        # time_stop_bars está en barras del TF de entrada
        search_limit_1m = n_bars_1m
        limit_1m = n_bars_1m
        if (time_stop_bars > 0) and (not is_bh_trade):
            limit_tf = entry_idx_tf + time_stop_bars
            if limit_tf < len(entry_timestamps):
                limit_ts = entry_timestamps[limit_tf]
                limit_1m_idx = int(np.searchsorted(timestamps_1m, limit_ts))
                limit_1m = limit_1m_idx + 1
                
            if limit_1m < search_limit_1m:
                search_limit_1m = limit_1m
        
        exit_idx_1m = -1
        exit_p = 0.0
        exit_reason = 0
        
        # Buscar salida vela a vela en 1m (empezando desde la misma vela de entrada)
        # ya que al entrar en el OPEN se puede tocar SL/TP en la misma vela.
        for curr_1m in range(entry_idx_1m, search_limit_1m):
            h = high_1m[curr_1m]
            low_val = low_1m[curr_1m]
            c = close_1m[curr_1m]
            just_activated = False
            
            # CHECK CUSTOM EXITS FIRST (en resolución 1m)
            if side == 1 and custom_exit_long_1m[curr_1m]:
                exit_idx_1m = curr_1m
                exit_p = c
                exit_reason = 5  # Custom Signal
                break
            if side == -1 and custom_exit_short_1m[curr_1m]:
                exit_idx_1m = curr_1m
                exit_p = c
                exit_reason = 5  # Custom Signal
                break
            
            if is_trailing and (not is_bh_trade):
                if not trailing_active:
                    # Check SL inicial
                    if side == 1 and low_val <= sl_price:
                        exit_idx_1m = curr_1m
                        exit_p = sl_price
                        exit_reason = 1
                        break
                    if side == -1 and h >= sl_price:
                        exit_idx_1m = curr_1m
                        exit_p = sl_price
                        exit_reason = 1
                        break
                    
                    # Check activación del trailing
                    if (side == 1 and h >= activation_price) or (side == -1 and low_val <= activation_price):
                        trailing_active = True
                        just_activated = True
                        out_trail_act_idx[trade_count] = curr_1m
                        out_trail_act_price[trade_count] = activation_price
                        
                        if side == 1:
                            trailing_level = h - trail_dist_distance
                        else:
                            trailing_level = low_val + trail_dist_distance
                
                if trailing_active:
                    if side == 1:
                        new_level = h - trail_dist_distance
                        if new_level > trailing_level:
                            trailing_level = new_level
                        
                        check_val = c if just_activated else low_val
                        
                        if check_val <= trailing_level:
                            exit_idx_1m = curr_1m
                            exit_p = trailing_level
                            exit_reason = 3
                            break
                    else:
                        new_level = low_val + trail_dist_distance
                        if new_level < trailing_level:
                            trailing_level = new_level
                        
                        check_val = c if just_activated else h
                        
                        if check_val >= trailing_level:
                            exit_idx_1m = curr_1m
                            exit_p = trailing_level
                            exit_reason = 3
                            break
            else:
                # Fixed SL/TP mode
                if side == 1:
                    if low_val <= sl_price:
                        exit_idx_1m = curr_1m
                        exit_p = sl_price
                        exit_reason = 1
                        break
                    if (not is_bh_trade) and tp_pct > 0 and h >= tp_price:
                        exit_idx_1m = curr_1m
                        exit_p = tp_price
                        exit_reason = 2
                        break
                else:
                    if h >= sl_price:
                        exit_idx_1m = curr_1m
                        exit_p = sl_price
                        exit_reason = 1
                        break
                    if (not is_bh_trade) and tp_pct > 0 and low_val <= tp_price:
                        exit_idx_1m = curr_1m
                        exit_p = tp_price
                        exit_reason = 2
                        break
        
        # Time stop fallback
        if exit_idx_1m == -1 and time_stop_bars > 0 and (not is_bh_trade):
            final_idx_1m = limit_1m - 1
            if final_idx_1m >= n_bars_1m:
                final_idx_1m = n_bars_1m - 1
            if final_idx_1m > entry_idx_1m:
                exit_idx_1m = final_idx_1m
                exit_p = close_1m[final_idx_1m]
                exit_reason = 4
        
        # End of data fallback
        if exit_idx_1m == -1:
            exit_idx_1m = n_bars_1m - 1
            exit_p = close_1m[exit_idx_1m]
            exit_reason = 0
        
        # Calcular PnL
        if exit_idx_1m < 0:
            continue
        
        last_exit_idx_1m = exit_idx_1m
        
        if side == 1:
            pnl_bruto = (exit_p - entry_p) * qty
        else:
            pnl_bruto = (entry_p - exit_p) * qty
        
        if comision_sides >= 2:
            comision = (entry_p * qty + exit_p * qty) * fee_rate
        else:
            comision = entry_p * qty * fee_rate
        
        pnl_neto = pnl_bruto - comision
        
        saldo_antes = current_balance
        current_balance += pnl_neto
        
        if current_balance < min_op:
            current_balance = min_op
        
        saldo_despues = current_balance

        # IMPORTANTE: el PnL contabilizado debe ser consistente con el cambio
        # real de balance (especialmente cuando se aplica floor de saldo mínimo).
        pnl_neto = saldo_despues - saldo_antes
        pnl_pct = (pnl_neto / saldo_usado * 100) if saldo_usado > 0 else 0.0
        
        # Guardar trade
        out_entry_idx[trade_count] = entry_idx_tf  # Índice en TF de entrada
        out_exit_idx[trade_count] = exit_idx_1m    # Índice en 1m
        out_entry_price[trade_count] = entry_p
        out_exit_price[trade_count] = exit_p
        out_side[trade_count] = side
        out_reason[trade_count] = exit_reason
        out_qty[trade_count] = qty
        out_saldo_usado[trade_count] = saldo_usado
        out_pnl_neto[trade_count] = pnl_neto
        out_pnl_pct[trade_count] = pnl_pct
        out_saldo_antes[trade_count] = saldo_antes
        out_saldo_despues[trade_count] = saldo_despues
        
        trade_count += 1
    
    return (out_entry_idx, out_exit_idx, out_entry_price, out_exit_price,
            out_side, out_reason, out_qty, out_saldo_usado, out_pnl_neto,
            out_pnl_pct, out_saldo_antes, out_saldo_despues,
            out_trail_act_idx, out_trail_act_price, trade_count)


@nb.njit(cache=True, fastmath=True, parallel=False)
def _simulate_trades_sequential(
    entry_indices: np.ndarray,
    entry_prices: np.ndarray,
    entry_types: np.ndarray,
    entry_is_buy_hold: np.ndarray,
    close_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    saldo_inicial: float,
    fee_rate: float,
    min_op: float,
    apalancamiento_max: float,
    saldo_usado_cfg: float,
    apalancamiento_strat: float,  # Para trades de estrategia
    saldo_usado_strat: float,     # Para trades de estrategia
    is_trailing: bool,
    sl_pct: float,
    tp_pct: float,
    trail_act_pct: float,
    trail_dist_pct: float,
    time_stop_bars: int,
    comision_sides: int,
    custom_exit_long: np.ndarray,
    custom_exit_short: np.ndarray,
    # ATR adaptive: distancias por entrada (longitud n_entries o 0 si no aplica)
    atr_sl_dist_arr: np.ndarray,
    atr_tp_dist_arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Kernel Numba optimizado para simular todos los trades secuencialmente.
    Kernel estándar que usa el mismo timeframe para entradas y salidas.

    Para salidas en 1m, usar _simulate_trades_with_1m_exits().
    """
    n_entries = len(entry_indices)
    n_bars = len(close_prices)
    
    # Pre-allocate output arrays
    out_entry_idx = np.empty(n_entries, dtype=np.int64)
    out_exit_idx = np.empty(n_entries, dtype=np.int64)
    out_entry_price = np.empty(n_entries, dtype=np.float64)
    out_exit_price = np.empty(n_entries, dtype=np.float64)
    out_side = np.empty(n_entries, dtype=np.int64)
    out_reason = np.empty(n_entries, dtype=np.int32)
    out_qty = np.empty(n_entries, dtype=np.float64)
    out_saldo_usado = np.empty(n_entries, dtype=np.float64)
    out_pnl_neto = np.empty(n_entries, dtype=np.float64)
    out_pnl_pct = np.empty(n_entries, dtype=np.float64)
    out_saldo_antes = np.empty(n_entries, dtype=np.float64)
    out_saldo_despues = np.empty(n_entries, dtype=np.float64)
    out_trail_act_idx = np.full(n_entries, -1, dtype=np.int64)
    out_trail_act_price = np.full(n_entries, np.nan, dtype=np.float64)
    
    current_balance = saldo_inicial
    last_exit_idx = -1
    trade_count = 0
    
    for i in range(n_entries):
        entry_idx = entry_indices[i]
        
        if entry_idx <= last_exit_idx:
            continue
        
        if current_balance <= min_op:
            break
        
        entry_p = entry_prices[i]
        side = entry_types[i]
        is_bh_trade = entry_is_buy_hold[i]

        # B&H usa config BH, estrategia usa config estrategia
        _saldo_cfg = saldo_usado_cfg if is_bh_trade else saldo_usado_strat
        _apal = apalancamiento_max if is_bh_trade else apalancamiento_strat
        saldo_usado = min(_saldo_cfg, current_balance)
        volumen_max = saldo_usado * _apal
        qty = volumen_max / entry_p if entry_p > 0 else 0.0

        if qty <= 0 or saldo_usado <= 0:
            continue

        # Calcular distancias de precio (ATR adaptive o % sobre stake)
        if len(atr_sl_dist_arr) > 0 and not is_bh_trade:
            sl_distance = atr_sl_dist_arr[i]
            tp_distance = atr_tp_dist_arr[i]
        else:
            # Buy&Hold por régimen: SL fijo de seguridad del 25%
            _sl_pct = 25.0 if is_bh_trade else sl_pct
            _tp_pct = 0.0 if is_bh_trade else tp_pct
            sl_distance = (saldo_usado * _sl_pct / 100.0) / qty
            tp_distance = (saldo_usado * _tp_pct / 100.0) / qty
        _trail_act_pct = 0.0 if is_bh_trade else trail_act_pct
        _trail_dist_pct = 0.0 if is_bh_trade else trail_dist_pct
        trail_act_distance = (saldo_usado * _trail_act_pct / 100.0) / qty
        trail_dist_distance = (saldo_usado * _trail_dist_pct / 100.0) / qty

        if side == 1:
            sl_price = entry_p - sl_distance
            tp_price = entry_p + tp_distance
            activation_price = entry_p + trail_act_distance
        else:
            sl_price = entry_p + sl_distance
            tp_price = entry_p - tp_distance
            activation_price = entry_p - trail_act_distance

        trailing_active = False
        trailing_level = 0.0

        search_limit = n_bars
        if (time_stop_bars > 0) and (not is_bh_trade):
            limit = entry_idx + time_stop_bars + 1
            if limit < search_limit:
                search_limit = limit
        
        exit_idx = -1
        exit_p = 0.0
        exit_reason = 0
        
        # Buscar salida vela a vela (empezando desde la misma vela de entrada)
        # ya que al entrar en el OPEN se puede tocar SL/TP en la misma vela.
        for curr in range(entry_idx, search_limit):
            h = high_prices[curr]
            low_val = low_prices[curr]
            c = close_prices[curr]
            just_activated = False
            
            # Custom exits
            if side == 1 and custom_exit_long[curr]:
                exit_idx = curr
                exit_p = c
                exit_reason = 5
                break
            if side == -1 and custom_exit_short[curr]:
                exit_idx = curr
                exit_p = c
                exit_reason = 5
                break
            
            if is_trailing and (not is_bh_trade):
                if not trailing_active:
                    if side == 1 and low_val <= sl_price:
                        exit_idx = curr
                        exit_p = sl_price
                        exit_reason = 1
                        break
                    if side == -1 and h >= sl_price:
                        exit_idx = curr
                        exit_p = sl_price
                        exit_reason = 1
                        break
                    if (side == 1 and h >= activation_price) or (side == -1 and low_val <= activation_price):
                        trailing_active = True
                        just_activated = True
                        out_trail_act_idx[trade_count] = curr
                        out_trail_act_price[trade_count] = activation_price
                        if side == 1:
                            trailing_level = h - trail_dist_distance
                        else:
                            trailing_level = low_val + trail_dist_distance
                
                if trailing_active:
                    if side == 1:
                        new_level = h - trail_dist_distance
                        if new_level > trailing_level:
                            trailing_level = new_level
                        check_val = c if just_activated else low_val
                        if check_val <= trailing_level:
                            exit_idx = curr
                            exit_p = trailing_level
                            exit_reason = 3
                            break
                    else:
                        new_level = low_val + trail_dist_distance
                        if new_level < trailing_level:
                            trailing_level = new_level
                        check_val = c if just_activated else h
                        if check_val >= trailing_level:
                            exit_idx = curr
                            exit_p = trailing_level
                            exit_reason = 3
                            break
            else:
                if side == 1:
                    if low_val <= sl_price:
                        exit_idx = curr
                        exit_p = sl_price
                        exit_reason = 1
                        break
                    if (not is_bh_trade) and tp_pct > 0 and h >= tp_price:
                        exit_idx = curr
                        exit_p = tp_price
                        exit_reason = 2
                        break
                else:
                    if h >= sl_price:
                        exit_idx = curr
                        exit_p = sl_price
                        exit_reason = 1
                        break
                    if (not is_bh_trade) and tp_pct > 0 and low_val <= tp_price:
                        exit_idx = curr
                        exit_p = tp_price
                        exit_reason = 2
                        break
        
        if exit_idx == -1 and time_stop_bars > 0 and (not is_bh_trade):
            final_idx = entry_idx + time_stop_bars
            if final_idx >= n_bars:
                final_idx = n_bars - 1
            if final_idx > entry_idx:
                exit_idx = final_idx
                exit_p = close_prices[final_idx]
                exit_reason = 4
        
        if exit_idx == -1:
            exit_idx = n_bars - 1
            exit_p = close_prices[exit_idx]
            exit_reason = 0
        
        if exit_idx < 0:
            continue
        
        last_exit_idx = exit_idx
        
        if side == 1:
            pnl_bruto = (exit_p - entry_p) * qty
        else:
            pnl_bruto = (entry_p - exit_p) * qty
        
        if comision_sides >= 2:
            comision = (entry_p * qty + exit_p * qty) * fee_rate
        else:
            comision = entry_p * qty * fee_rate
        
        pnl_neto = pnl_bruto - comision
        
        saldo_antes = current_balance
        current_balance += pnl_neto
        
        if current_balance < min_op:
            current_balance = min_op
        
        saldo_despues = current_balance

        # IMPORTANTE: el PnL contabilizado debe ser consistente con el cambio
        # real de balance (especialmente cuando se aplica floor de saldo mínimo).
        pnl_neto = saldo_despues - saldo_antes
        pnl_pct = (pnl_neto / saldo_usado * 100) if saldo_usado > 0 else 0.0
        
        out_entry_idx[trade_count] = entry_idx
        out_exit_idx[trade_count] = exit_idx
        out_entry_price[trade_count] = entry_p
        out_exit_price[trade_count] = exit_p
        out_side[trade_count] = side
        out_reason[trade_count] = exit_reason
        out_qty[trade_count] = qty
        out_saldo_usado[trade_count] = saldo_usado
        out_pnl_neto[trade_count] = pnl_neto
        out_pnl_pct[trade_count] = pnl_pct
        out_saldo_antes[trade_count] = saldo_antes
        out_saldo_despues[trade_count] = saldo_despues
        
        trade_count += 1
    
    return (out_entry_idx, out_exit_idx, out_entry_price, out_exit_price,
            out_side, out_reason, out_qty, out_saldo_usado, out_pnl_neto,
            out_pnl_pct, out_saldo_antes, out_saldo_despues,
            out_trail_act_idx, out_trail_act_price, trade_count)


# =============================================================================
# 5. FUNCIÓN PRINCIPAL: CALCULATE PERFORMANCE
# =============================================================================

def calculate_performance_vectorized_numba(
    *,
    df: pl.DataFrame,
    signals: pl.DataFrame,
    params: BacktestParams,
    strategy: Strategy,
    df_1m: Optional[pl.DataFrame] = None,
    signals_1m: Optional[pl.DataFrame] = None,
    timeframe_minutes: int = 1,
) -> Tuple[pl.DataFrame, List[float]]:
    """
    Vector engine con gestión de capital realista.
    
    IMPORTANTE - SALIDAS EN 1M:
        Si df_1m es proporcionado y timeframe_minutes > 1, las salidas
        (SL/TP/Trailing/Custom) se evalúan usando velas de 1 minuto para
        máxima precisión, independientemente del timeframe de las entradas.
    
    Args:
        df: DataFrame con datos OHLCV del timeframe de entrada
        signals: DataFrame con señales (signal_long, signal_short, exit_long, exit_short)
        params: Parámetros de backtest
        strategy: Estrategia (para compatibilidad)
        df_1m: DataFrame con datos OHLCV de 1 minuto (opcional, para salidas precisas)
        signals_1m: DataFrame con señales de salida en 1m (opcional)
        timeframe_minutes: Timeframe de entrada en minutos (1, 5, 15, 60, etc.)
    
    Returns:
        Tuple[trades_df, equity_curve]
    """
    # =========================================================================
    # 1) Extraer señales de entrada
    # =========================================================================
    sig_long = signals["signal_long"].fill_null(False)
    sig_short = signals["signal_short"].fill_null(False)
    
    # Custom exits del TF de entrada
    if "exit_long" in signals.columns:
        exit_long = signals["exit_long"].fill_null(False).to_numpy()
    else:
        exit_long = np.zeros(len(signals), dtype=np.bool_)
        
    if "exit_short" in signals.columns:
        exit_short = signals["exit_short"].fill_null(False).to_numpy()
    else:
        exit_short = np.zeros(len(signals), dtype=np.bool_)
    
    # Edge trigger vectorizado
    entry_long = sig_long & ~sig_long.shift(1).fill_null(False)
    entry_short = sig_short & ~sig_short.shift(1).fill_null(False)
    
    entry_mask = entry_long | entry_short
    n_entries = entry_mask.sum()
    
    if n_entries == 0:
        return pl.DataFrame(), [params.saldo_inicial]
    
    # =========================================================================
    # 2) Extraer arrays numpy
    # =========================================================================
    o_arr = df["open"].to_numpy()
    c_arr = df["close"].to_numpy()
    h_arr = df["high"].to_numpy() if "high" in df.columns else c_arr
    l_arr = df["low"].to_numpy() if "low" in df.columns else c_arr
    ts_arr = df["timestamp"]
    
    all_indices = np.arange(df.height, dtype=np.int64)
    entry_mask_np = entry_mask.to_numpy()
    
    # Índices originales del trigger (vela donde se dio la señal)
    signal_indices = all_indices[entry_mask_np]
    
    # Tipos de entrada basados en la vela de señal
    entry_long_np = entry_long.to_numpy()
    signal_types = np.where(entry_long_np[signal_indices], 1, -1).astype(np.int64)
    
    # Flag por entrada: si proviene de señal Buy&Hold por régimen
    if "regime_buy_hold_long" in signals.columns:
        _bh_col = signals["regime_buy_hold_long"].fill_null(False).to_numpy()
        signal_is_bh = _bh_col[signal_indices]
    else:
        signal_is_bh = np.zeros(len(signal_indices), dtype=np.bool_)
    
    # ========== ENTRADA AL INICIO DE LA SIGUIENTE VELA (OPEN) ==========
    # Desplazamos la entrada a la siguiente vela
    entry_indices = signal_indices + 1
    
    # Filtrar entradas que quedarían fuera del array (última vela con señal)
    valid_mask = entry_indices < df.height
    entry_indices = entry_indices[valid_mask]
    entry_types = signal_types[valid_mask]
    entry_is_buy_hold = signal_is_bh[valid_mask]
    
    # Precio de entrada = OPEN de la vela de entrada (la siguiente a la señal)
    entry_prices = o_arr[entry_indices]
    
    # Preparar modos de salida
    is_trailing     = params.exit_type == "TRAILING"
    is_time_bars    = params.exit_type == "BARS"
    is_atr_adaptive = params.exit_type == "ATR"
    
    # Parámetros
    fee_rate = float(params.comision_pct)
    min_op = float(params.saldo_minimo_operativo)
    apalancamiento_max = float(params.apalancamiento_max)
    saldo_usado_cfg = float(params.saldo_usado)

    # Guardar valores originales de estrategia antes de posible override B&H
    apalancamiento_strat = apalancamiento_max
    saldo_usado_strat = saldo_usado_cfg

    # Modo Buy&Hold por régimen (inyectado por runner):
    # - usa 100% del balance (ALL) o saldo fijo capado por balance.
    # - fuerza apalancamiento 1x para simular compra spot del activo.
    bh_mode = bool(getattr(params, "regime_buy_hold_mode", False))
    if bh_mode:
        bh_all = bool(getattr(params, "regime_buy_hold_saldo_all", True))
        bh_saldo = float(getattr(params, "regime_buy_hold_saldo", 0.0) or 0.0)
        saldo_usado_cfg = 1e18 if bh_all else max(0.0, bh_saldo)
        apalancamiento_max = 1.0

    sl_pct = float(params.exit_sl_pct)
    tp_pct = float(params.exit_tp_pct)
    trail_act = float(params.exit_trail_act_pct)
    trail_dist = float(params.exit_trail_dist_pct)
    time_stop = int(params.time_stop_bars)
    time_bars_cfg = int(params.exit_time_bars)

    # Si es modo time_bars: desactivar SL/TP/Trailing y usar time_stop_bars = exit_time_bars
    # IMPORTANTE: sl_pct = 9999.0 (NO 0.0) porque sl_pct=0 → sl_price=entry_price → salida inmediata
    if is_time_bars:
        sl_pct = 9999.0
        tp_pct = 0.0
        trail_act = 0.0
        trail_dist = 0.0
        is_trailing = False
        time_stop = time_bars_cfg if time_bars_cfg > 0 else time_stop

    # Pre-cómputo de distancias ATR adaptive (por entrada, en unidades de precio)
    # Arrays vacíos → kernels usan el modo sl_pct/tp_pct estándar
    _atr_sl_dist = np.empty(0, dtype=np.float64)
    _atr_tp_dist = np.empty(0, dtype=np.float64)
    if is_atr_adaptive:
        _atr_sl_dist, _atr_tp_dist = _compute_atr_adaptive_distances(
            high=h_arr,
            low=l_arr,
            close=c_arr,
            entry_indices=entry_indices,
            entry_prices=entry_prices,
            apalancamiento_max=apalancamiento_strat,  # Siempre el leverage de estrategia, nunca el B&H override
            atr_period=int(getattr(params, "exit_atr_period", DEFAULT_EXIT_ATR_PERIOD)),
            atr_min_pct=float(getattr(params, "exit_atr_min_pct", DEFAULT_EXIT_ATR_MIN_PCT)),
            atr_max_pct=float(getattr(params, "exit_atr_max_pct", DEFAULT_EXIT_ATR_MAX_PCT)),
            atr_lookback=int(getattr(params, "exit_atr_lookback", DEFAULT_EXIT_ATR_LOOKBACK)),
        )
        # sl_pct/tp_pct no se usan para las distancias (se usan los arrays per-entry),
        # pero tp_pct debe ser > 0 para que el kernel evalúe el TP.
        sl_pct = 1.0
        tp_pct = 1.0
        trail_act = 0.0
        trail_dist = 0.0

    comision_sides_int = int(params.comision_sides)
    saldo_inicial = float(params.saldo_inicial)
    
    # =========================================================================
    # 3) Decidir si usar salidas en 1m
    # =========================================================================
    # SIEMPRE usar salidas en 1m si los datos están disponibles, incluso si el TF de entrada es 1m.
    # EXCEPCIÓN: time_bars evalúa el time stop contando barras del TF de entrada,
    # por lo que debe usar el kernel estándar del mismo timeframe.
    use_1m_exits = (df_1m is not None) and not is_time_bars
    
    if use_1m_exits:
        # Usar kernel con salidas en 1m
        c_1m = df_1m["close"].to_numpy()
        h_1m = df_1m["high"].to_numpy() if "high" in df_1m.columns else c_1m
        l_1m = df_1m["low"].to_numpy() if "low" in df_1m.columns else c_1m
        ts_1m = df_1m["timestamp"].to_numpy().view(np.int64)
        ts_entry = df["timestamp"].to_numpy().view(np.int64)
        
        tf_ratio = timeframe_minutes  # Ratio de velas 1m por vela TF
        
        # Señales de salida en 1m (estrategia)
        if signals_1m is not None and "exit_long" in signals_1m.columns:
            exit_long_1m = signals_1m["exit_long"].fill_null(False).to_numpy()
        else:
            exit_long_1m = np.zeros(len(df_1m), dtype=np.bool_)

        if signals_1m is not None and "exit_short" in signals_1m.columns:
            exit_short_1m = signals_1m["exit_short"].fill_null(False).to_numpy()
        else:
            exit_short_1m = np.zeros(len(df_1m), dtype=np.bool_)

        # Propagar señales de salida del TF (régimen, custom) a resolución 1m.
        # CRÍTICO: el filtro de régimen escribe exit_long/exit_short en signals (TF),
        # pero el kernel 1m solo lee exit_long_1m. Sin esto, los trades abiertos
        # en régimen no permitido nunca se cierran → la estrategia se bloquea.
        _tf_idx_per_1m = np.clip(
            np.searchsorted(ts_entry, ts_1m, side='right') - 1,
            0, len(ts_entry) - 1,
        )
        if "exit_long" in signals.columns:
            _tf_ex_l = signals["exit_long"].fill_null(False).to_numpy()
            if _tf_ex_l.any():
                exit_long_1m = exit_long_1m | _tf_ex_l[_tf_idx_per_1m]
        if "exit_short" in signals.columns:
            _tf_ex_s = signals["exit_short"].fill_null(False).to_numpy()
            if _tf_ex_s.any():
                exit_short_1m = exit_short_1m | _tf_ex_s[_tf_idx_per_1m]

        (out_entry_idx, out_exit_idx, out_entry_price, out_exit_price,
         out_side, out_reason, out_qty, out_saldo_usado, out_pnl_neto,
         out_pnl_pct, out_saldo_antes, out_saldo_despues,
         out_trail_act_idx, out_trail_act_price, trade_count) = _simulate_trades_with_1m_exits(
            entry_indices=entry_indices,
            entry_prices=entry_prices,
            entry_types=entry_types,
            entry_is_buy_hold=entry_is_buy_hold,
            entry_timestamps=ts_entry,
            close_1m=c_1m,
            high_1m=h_1m,
            low_1m=l_1m,
            timestamps_1m=ts_1m,
            saldo_inicial=saldo_inicial,
            fee_rate=fee_rate,
            min_op=min_op,
            apalancamiento_max=apalancamiento_max,
            saldo_usado_cfg=saldo_usado_cfg,
            apalancamiento_strat=apalancamiento_strat,
            saldo_usado_strat=saldo_usado_strat,
            is_trailing=is_trailing,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            trail_act_pct=trail_act,
            trail_dist_pct=trail_dist,
            time_stop_bars=time_stop,
            comision_sides=comision_sides_int,
            custom_exit_long_1m=exit_long_1m,
            custom_exit_short_1m=exit_short_1m,
            tf_ratio=tf_ratio,
            atr_sl_dist_arr=_atr_sl_dist,
            atr_tp_dist_arr=_atr_tp_dist,
        )
        
        # Para timestamps de salida, usar df_1m
        ts_arr_exit = df_1m["timestamp"]
    else:
        # Kernel estándar (mismo TF para entradas y salidas)
        (out_entry_idx, out_exit_idx, out_entry_price, out_exit_price,
         out_side, out_reason, out_qty, out_saldo_usado, out_pnl_neto,
         out_pnl_pct, out_saldo_antes, out_saldo_despues,
         out_trail_act_idx, out_trail_act_price, trade_count) = _simulate_trades_sequential(
            entry_indices=entry_indices,
            entry_prices=entry_prices,
            entry_types=entry_types,
            entry_is_buy_hold=entry_is_buy_hold,
            close_prices=c_arr,
            high_prices=h_arr,
            low_prices=l_arr,
            saldo_inicial=saldo_inicial,
            fee_rate=fee_rate,
            min_op=min_op,
            apalancamiento_max=apalancamiento_max,
            saldo_usado_cfg=saldo_usado_cfg,
            apalancamiento_strat=apalancamiento_strat,
            saldo_usado_strat=saldo_usado_strat,
            is_trailing=is_trailing,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            trail_act_pct=trail_act,
            trail_dist_pct=trail_dist,
            time_stop_bars=time_stop,
            comision_sides=comision_sides_int,
            custom_exit_long=exit_long,
            custom_exit_short=exit_short,
            atr_sl_dist_arr=_atr_sl_dist,
            atr_tp_dist_arr=_atr_tp_dist,
        )
        
        ts_arr_exit = ts_arr
    
    if trade_count == 0:
        return pl.DataFrame(), [saldo_inicial]
    
    # =========================================================================
    # 4) Construir DataFrame de trades
    # =========================================================================
    entry_idx_view = out_entry_idx[:trade_count]
    exit_idx_view = out_exit_idx[:trade_count]
    entry_price_view = out_entry_price[:trade_count]
    exit_price_view = out_exit_price[:trade_count]
    side_view = out_side[:trade_count]
    reason_view = out_reason[:trade_count]
    qty_view = out_qty[:trade_count]
    saldo_usado_view = out_saldo_usado[:trade_count]
    pnl_neto_view = out_pnl_neto[:trade_count]
    pnl_pct_view = out_pnl_pct[:trade_count]
    saldo_antes_view = out_saldo_antes[:trade_count]
    saldo_despues_view = out_saldo_despues[:trade_count]
    trail_act_idx_view = out_trail_act_idx[:trade_count]
    trail_act_price_view = out_trail_act_price[:trade_count]
    
    # Flag por trade: si la entrada provino del modo Buy&Hold por régimen
    entry_is_bh_by_idx = np.zeros(df.height, dtype=np.bool_)
    entry_is_bh_by_idx[entry_indices] = entry_is_buy_hold
    is_buy_hold_view = entry_is_bh_by_idx[entry_idx_view]
    
    # PnL bruto y comisión
    pnl_bruto = np.where(
        side_view == 1,
        (exit_price_view - entry_price_view) * qty_view,
        (entry_price_view - exit_price_view) * qty_view
    )
    if comision_sides_int >= 2:
        comision = (entry_price_view * qty_view + exit_price_view * qty_view) * fee_rate
    else:
        comision = entry_price_view * qty_view * fee_rate
    
    trade_type = np.where(side_view == 1, "long", "short")
    
    # Timestamps
    entry_times = ts_arr.gather(pl.Series(entry_idx_view))
    exit_times = ts_arr_exit.gather(pl.Series(exit_idx_view))
    
    trades_df = pl.DataFrame({
        "entry_idx": entry_idx_view,
        "exit_idx": exit_idx_view,
        "entry_price": entry_price_view,
        "exit_price": exit_price_view,
        "side_int": side_view,
        "reason": reason_view,
        "type": trade_type,
        "qty": qty_view,
        "saldo_usado": saldo_usado_view,
        "pnl_bruto": pnl_bruto,
        "comision": comision,
        "pnl_neto": pnl_neto_view,
        "pnl_pct": pnl_pct_view,
        "saldo_antes": saldo_antes_view,
        "saldo_despues": saldo_despues_view,
        "entry_time": entry_times,
        "exit_time": exit_times,
        "trail_act_idx": trail_act_idx_view,
        "trail_act_price": trail_act_price_view,
        "is_buy_hold": is_buy_hold_view,
    })

    # Para exit_type=time_bars, la duración debe representar barras del TF de entrada,
    # no tiempo de reloj (que puede incluir huecos de mercado como overnight/weekend).
    # Esto garantiza que "12 barras en 1H" se refleje como 720 min fijos en métricas.
    if is_time_bars and timeframe_minutes > 0:
        duracion_min = (exit_idx_view - entry_idx_view).astype(np.float64) * float(timeframe_minutes)
        # Seguridad defensiva: evitar negativos por cualquier desalineación de índices.
        duracion_min = np.maximum(duracion_min, 0.0)
        trades_df = trades_df.with_columns(pl.Series("duracion_min", duracion_min))
    
    # Trailing activation timestamps
    if use_1m_exits:
        ts_np = df_1m["timestamp"].to_numpy()
    else:
        ts_np = df["timestamp"].to_numpy()
    
    if np.issubdtype(ts_np.dtype, np.datetime64):
        act_timestamps = np.full(len(trail_act_idx_view), np.datetime64("NaT"), dtype=ts_np.dtype)
        valid_mask = trail_act_idx_view != -1
        if valid_mask.any():
            valid_indices = trail_act_idx_view[valid_mask]
            # Asegurar que los índices estén dentro del rango
            valid_indices = np.clip(valid_indices, 0, len(ts_np) - 1)
            act_timestamps[valid_mask] = ts_np[valid_indices]
        trail_act_times_pl = pl.Series("trail_act_time", act_timestamps)
    else:
        vals = [ts_np[min(i, len(ts_np)-1)] if i != -1 else None for i in trail_act_idx_view]
        trail_act_times_pl = pl.Series("trail_act_time", vals)
    
    trades_df = trades_df.with_columns(trail_act_time=trail_act_times_pl)
    
    equity_curve = list(saldo_despues_view)
    
    return trades_df, equity_curve


# =============================================================================
# 6. WARMUP JIT (PRE-COMPILACIÓN AL IMPORTAR)
# =============================================================================

def _warmup_jit() -> None:
    """Pre-compila funciones Numba con datos dummy para eliminar latencia inicial."""
    dummy_entries = np.array([0, 5, 10], dtype=np.int64)
    dummy_prices = np.array([100.0, 101.0, 102.0], dtype=np.float64)
    dummy_types = np.array([1, -1, 1], dtype=np.int64)
    dummy_close = np.linspace(100, 110, 20, dtype=np.float64)
    dummy_high = dummy_close * 1.01
    dummy_low = dummy_close * 0.99
    dummy_ts = np.arange(20, dtype=np.int64)
    
    # Warmup _simulate_trades_sequential
    try:
        dummy_bool = np.zeros(20, dtype=np.bool_)
        dummy_atr = np.empty(0, dtype=np.float64)  # ATR arrays vacíos (modo no-ATR)
        _simulate_trades_sequential(
            dummy_entries, dummy_prices, dummy_types, np.zeros(3, dtype=np.bool_),
            dummy_close, dummy_high, dummy_low,
            1000.0, 0.001, 100.0, 10.0, 75.0,
            False, 5.0, 10.0, 0.0, 0.0, 0, 2,
            dummy_bool, dummy_bool,
            dummy_atr, dummy_atr
        )
    except Exception:
        pass
    
    # Warmup _simulate_trades_with_1m_exits
    try:
        dummy_bool_1m = np.zeros(100, dtype=np.bool_)
        dummy_close_1m = np.linspace(100, 110, 100, dtype=np.float64)
        dummy_high_1m = dummy_close_1m * 1.01
        dummy_low_1m = dummy_close_1m * 0.99
        dummy_ts_1m = np.arange(100, dtype=np.int64)
        dummy_atr_1m = np.empty(0, dtype=np.float64)  # ATR arrays vacíos (modo no-ATR)
        _simulate_trades_with_1m_exits(
            dummy_entries, dummy_prices, dummy_types, np.zeros(3, dtype=np.bool_), dummy_ts,
            dummy_close_1m, dummy_high_1m, dummy_low_1m, dummy_ts_1m,
            1000.0, 0.001, 100.0, 10.0, 75.0,
            False, 5.0, 10.0, 0.0, 0.0, 0, 2,
            dummy_bool_1m, dummy_bool_1m, 5,
            dummy_atr_1m, dummy_atr_1m
        )
    except Exception:
        pass


# Ejecutar warmup al importar
try:
    _warmup_jit()
except Exception:
    pass
