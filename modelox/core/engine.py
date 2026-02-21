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
    block_velas_after_exit: int
    time_stop_bars: int

    @classmethod
    def from_config_and_params(cls, config: BacktestConfig, params: Dict[str, Any]) -> "BacktestParams":
        return cls(
            saldo_inicial=float(config.saldo_inicial),
            comision_pct=float(config.comision_pct),
            comision_sides=int(getattr(config, "comision_sides", 2)),
            saldo_minimo_operativo=float(config.saldo_minimo_operativo),
            saldo_usado=float(getattr(config, "saldo_usado", 75.0)),
            apalancamiento_max=float(getattr(config, "apalancamiento_max", 60.0)),
            exit_type=str(params.get("__exit_type", getattr(config, "exit_type", "pnl_fixed"))),
            exit_sl_pct=float(params.get("__exit_sl_pct", getattr(config, "exit_sl_pct", 0.0))),
            exit_tp_pct=float(params.get("__exit_tp_pct", getattr(config, "exit_tp_pct", 0.0))),
            exit_trail_act_pct=float(params.get("__exit_trail_act_pct", getattr(config, "exit_trail_act_pct", 0.0))),
            exit_trail_dist_pct=float(params.get("__exit_trail_dist_pct", getattr(config, "exit_trail_dist_pct", 0.0))),
            block_velas_after_exit=int(params.get("block_velas_after_exit", 0)),
            time_stop_bars=int(params.get("time_stop_bars", 0)),
        )


# =============================================================================
# 3. KERNEL NUMBA: SALIDAS (SL/TP/TRAILING) - KERNEL ESTÁNDAR (MISMO TF)
# =============================================================================

@nb.njit(cache=True, fastmath=True)
def find_exits_numba(
    entry_indices,
    entry_prices,
    entry_types,  # 1=Long, -1=Short
    entry_qty,    # qty por trade
    entry_stake,  # stake (saldo usado) por trade
    close_prices,
    high_prices,
    low_prices,
    is_trailing,
    sl_pct,
    tp_pct,
    trail_act_pct,
    trail_dist_pct,
    time_stop_bars,
):
    """Kernel Numba para encontrar salidas con SL/TP basados en % sobre STAKE.

    Los porcentajes (sl_pct, tp_pct, etc.) son sobre el STAKE (margen/saldo usado),
    NO sobre el precio de entrada.

    Fórmulas:
    - LONG:
      - SL_price = entry_price - (stake × sl_pct / 100) / qty
      - TP_price = entry_price + (stake × tp_pct / 100) / qty
    - SHORT:
      - SL_price = entry_price + (stake × sl_pct / 100) / qty
      - TP_price = entry_price - (stake × tp_pct / 100) / qty

    Esto garantiza que al llegar al SL, el PNL sea exactamente -stake × sl_pct%.
    """
    n_entries = len(entry_indices)
    n_bars = len(close_prices)

    exit_indices = np.full(n_entries, -1, dtype=np.int64)
    exit_prices = np.full(n_entries, np.nan, dtype=np.float64)
    exit_reasons = np.zeros(n_entries, dtype=np.int32)  # 1=SL, 2=TP, 3=Trail, 4=Time

    for i in range(n_entries):
        entry_idx = entry_indices[i]
        entry_price = entry_prices[i]
        side = entry_types[i]
        qty = entry_qty[i]
        stake = entry_stake[i]

        if qty <= 0 or stake <= 0:
            continue

        # Calcular precios de SL/TP basados en % sobre stake
        # sl_pct% de pérdida sobre stake = (stake × sl_pct / 100)
        # eso corresponde a un movimiento de precio de (stake × sl_pct / 100) / qty
        sl_distance = (stake * sl_pct / 100.0) / qty
        tp_distance = (stake * tp_pct / 100.0) / qty
        trail_act_distance = (stake * trail_act_pct / 100.0) / qty
        trail_dist_distance = (stake * trail_dist_pct / 100.0) / qty

        if side == 1:  # LONG
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
            activation_price = entry_price + trail_act_distance
        else:  # SHORT
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance
            activation_price = entry_price - trail_act_distance

        trailing_active = False
        trailing_level = 0.0

        search_limit = n_bars
        if time_stop_bars > 0:
            limit = entry_idx + time_stop_bars + 1
            if limit < search_limit:
                search_limit = limit

        for curr in range(entry_idx + 1, search_limit):
            h = high_prices[curr]
            low_val = low_prices[curr]
            c = close_prices[curr]
            just_activated = False

            if is_trailing:
                # Trailing mode: SL inicial + trailing activation
                if not trailing_active:
                    # Check SL inicial primero
                    if side == 1 and low_val <= sl_price:
                        exit_indices[i] = curr
                        exit_prices[i] = sl_price
                        exit_reasons[i] = 1  # SL
                        break
                    if side == -1 and h >= sl_price:
                        exit_indices[i] = curr
                        exit_prices[i] = sl_price
                        exit_reasons[i] = 1  # SL
                        break

                    # Check activación del trailing
                    if (side == 1 and h >= activation_price) or (side == -1 and low_val <= activation_price):
                        trailing_active = True
                        just_activated = True
                        if side == 1:
                            # Trailing level = high - trailing_distance
                            trailing_level = h - trail_dist_distance
                        else:
                            trailing_level = low_val + trail_dist_distance

                if trailing_active:
                    if side == 1:
                        # Actualizar trailing level hacia arriba
                        new_level = h - trail_dist_distance
                        if new_level > trailing_level:
                            trailing_level = new_level
                        
                        # Prevent phantom exit on activation bar by checking Close instead of Low
                        check_val = c if just_activated else low_val

                        # Check hit del trailing
                        if check_val <= trailing_level:
                            exit_indices[i] = curr
                            exit_prices[i] = trailing_level
                            exit_reasons[i] = 3  # Trail
                            break
                    else:
                        # Actualizar trailing level hacia abajo
                        new_level = low_val + trail_dist_distance
                        if new_level < trailing_level:
                            trailing_level = new_level
                        
                        # Prevent phantom exit on activation bar by checking Close instead of High
                        check_val = c if just_activated else h
                        
                        # Check hit del trailing
                        if check_val >= trailing_level:
                            exit_indices[i] = curr
                            exit_prices[i] = trailing_level
                            exit_reasons[i] = 3  # Trail
                            break
            else:
                # Fixed SL/TP mode
                if side == 1:
                    if low_val <= sl_price:
                        exit_indices[i] = curr
                        exit_prices[i] = sl_price
                        exit_reasons[i] = 1  # SL
                        break
                    if tp_pct > 0 and h >= tp_price:
                        exit_indices[i] = curr
                        exit_prices[i] = tp_price
                        exit_reasons[i] = 2  # TP
                        break
                else:
                    if h >= sl_price:
                        exit_indices[i] = curr
                        exit_prices[i] = sl_price
                        exit_reasons[i] = 1  # SL
                        break
                    if tp_pct > 0 and low_val <= tp_price:
                        exit_indices[i] = curr
                        exit_prices[i] = tp_price
                        exit_reasons[i] = 2  # TP
                        break

        # Time stop fallback
        if exit_indices[i] == -1 and time_stop_bars > 0:
            final_idx = entry_idx + time_stop_bars
            if final_idx >= n_bars:
                final_idx = n_bars - 1
            if final_idx > entry_idx:
                exit_indices[i] = final_idx
                exit_prices[i] = close_prices[final_idx]
                exit_reasons[i] = 4  # Time

    return exit_indices, exit_prices, exit_reasons


@nb.njit(cache=True, fastmath=True)
def find_single_exit_numba(
    entry_idx: int,
    entry_price: float,
    side: int,  # 1=Long, -1=Short
    qty: float,
    stake: float,
    close_prices,
    high_prices,
    low_prices,
    is_trailing: bool,
    sl_pct: float,
    tp_pct: float,
    trail_act_pct: float,
    trail_dist_pct: float,
    time_stop_bars: int,
) -> Tuple[int, float, int]:
    """Encuentra la salida para un trade individual con SL/TP basados en % sobre STAKE.

    Returns: (exit_idx, exit_price, exit_reason)
        exit_reason: 1=SL, 2=TP, 3=Trail, 4=Time, 0=EndOfData
    """
    n_bars = len(close_prices)

    if qty <= 0 or stake <= 0:
        return -1, np.nan, 0

    # Calcular distancias de precio basadas en % sobre stake
    sl_distance = (stake * sl_pct / 100.0) / qty
    tp_distance = (stake * tp_pct / 100.0) / qty
    trail_act_distance = (stake * trail_act_pct / 100.0) / qty
    trail_dist_distance = (stake * trail_dist_pct / 100.0) / qty

    if side == 1:  # LONG
        sl_price = entry_price - sl_distance
        tp_price = entry_price + tp_distance
        activation_price = entry_price + trail_act_distance
    else:  # SHORT
        sl_price = entry_price + sl_distance
        tp_price = entry_price - tp_distance
        activation_price = entry_price - trail_act_distance

    trailing_active = False
    trailing_level = 0.0

    search_limit = n_bars
    if time_stop_bars > 0:
        limit = entry_idx + time_stop_bars + 1
        if limit < search_limit:
            search_limit = limit

    for curr in range(entry_idx + 1, search_limit):
        h = high_prices[curr]
        low_val = low_prices[curr]
        c = close_prices[curr]
        just_activated = False

        if is_trailing:
            # Trailing mode: SL inicial + trailing activation
            if not trailing_active:
                # Check SL inicial primero
                if side == 1 and low_val <= sl_price:
                    return curr, sl_price, 1  # SL
                if side == -1 and h >= sl_price:
                    return curr, sl_price, 1  # SL

                # Check activación del trailing
                if (side == 1 and h >= activation_price) or (side == -1 and low_val <= activation_price):
                    trailing_active = True
                    just_activated = True
                    if side == 1:
                        trailing_level = h - trail_dist_distance
                    else:
                        trailing_level = low_val + trail_dist_distance

        if trailing_active:
            if side == 1:
                new_level = h - trail_dist_distance
                if new_level > trailing_level:
                    trailing_level = new_level
                
                # Prevent phantom exit on activation bar by checking Close instead of Low
                check_val = c if just_activated else low_val

                if check_val <= trailing_level:
                    return curr, trailing_level, 3  # Trail
            else:
                new_level = low_val + trail_dist_distance
                if new_level < trailing_level:
                    trailing_level = new_level
                
                # Prevent phantom exit on activation bar by checking Close instead of High
                check_val = c if just_activated else h
                
                if check_val >= trailing_level:
                    return curr, trailing_level, 3  # Trail
        else:
            # Fixed SL/TP mode
            if side == 1:
                if low_val <= sl_price:
                    return curr, sl_price, 1  # SL
                if tp_pct > 0 and h >= tp_price:
                    return curr, tp_price, 2  # TP
            else:
                if h >= sl_price:
                    return curr, sl_price, 1  # SL
                if tp_pct > 0 and low_val <= tp_price:
                    return curr, tp_price, 2  # TP

    # Time stop fallback
    if time_stop_bars > 0:
        final_idx = entry_idx + time_stop_bars
        if final_idx >= n_bars:
            final_idx = n_bars - 1
        if final_idx > entry_idx:
            return final_idx, close_prices[final_idx], 4  # Time

    # End of data
    final_idx = n_bars - 1
    return final_idx, close_prices[final_idx], 0  # EndOfData


# =============================================================================
# 4. KERNEL NUMBA: SIMULACIÓN CON SALIDAS EN 1M
# =============================================================================

@nb.njit(cache=True, fastmath=True, parallel=False)
def _simulate_trades_with_1m_exits(
    # Datos del timeframe de entrada (para señales)
    entry_indices: np.ndarray,
    entry_prices: np.ndarray,
    entry_types: np.ndarray,
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
    apalancamiento_max: float,
    saldo_usado_cfg: float,
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
        
        # Convertir a índice en 1m (inicio de la vela del TF)
        entry_idx_1m = entry_idx_tf * tf_ratio
        
        # Skip si la entrada está antes de la salida del trade anterior
        if entry_idx_1m <= last_exit_idx_1m:
            continue
        
        # STOP si el saldo ya bajó al mínimo operativo
        if current_balance <= min_op:
            break
        
        entry_p = entry_prices[i]
        side = entry_types[i]
        
        # Calcular saldo_usado real
        saldo_usado = min(saldo_usado_cfg, current_balance)
        
        # Calcular qty dinámica
        volumen_max = saldo_usado * apalancamiento_max
        qty = volumen_max / entry_p if entry_p > 0 else 0.0
        
        if qty <= 0 or saldo_usado <= 0:
            continue
        
        # Calcular distancias de precio basadas en % sobre stake
        sl_distance = (saldo_usado * sl_pct / 100.0) / qty
        tp_distance = (saldo_usado * tp_pct / 100.0) / qty
        trail_act_distance = (saldo_usado * trail_act_pct / 100.0) / qty
        trail_dist_distance = (saldo_usado * trail_dist_pct / 100.0) / qty
        
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
        if time_stop_bars > 0:
            limit_1m = entry_idx_1m + (time_stop_bars * tf_ratio) + 1
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
            
            if is_trailing:
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
                    if tp_pct > 0 and h >= tp_price:
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
                    if tp_pct > 0 and low_val <= tp_price:
                        exit_idx_1m = curr_1m
                        exit_p = tp_price
                        exit_reason = 2
                        break
        
        # Time stop fallback
        if exit_idx_1m == -1 and time_stop_bars > 0:
            final_idx_1m = entry_idx_1m + (time_stop_bars * tf_ratio)
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
        pnl_pct = (pnl_neto / saldo_usado * 100) if saldo_usado > 0 else 0.0
        
        saldo_antes = current_balance
        current_balance += pnl_neto
        
        if current_balance < min_op:
            current_balance = min_op
        
        saldo_despues = current_balance
        
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
    close_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    saldo_inicial: float,
    fee_rate: float,
    min_op: float,
    apalancamiento_max: float,
    saldo_usado_cfg: float,
    is_trailing: bool,
    sl_pct: float,
    tp_pct: float,
    trail_act_pct: float,
    trail_dist_pct: float,
    time_stop_bars: int,
    comision_sides: int,
    custom_exit_long: np.ndarray,
    custom_exit_short: np.ndarray,
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
        
        saldo_usado = min(saldo_usado_cfg, current_balance)
        volumen_max = saldo_usado * apalancamiento_max
        qty = volumen_max / entry_p if entry_p > 0 else 0.0
        
        if qty <= 0 or saldo_usado <= 0:
            continue
        
        sl_distance = (saldo_usado * sl_pct / 100.0) / qty
        tp_distance = (saldo_usado * tp_pct / 100.0) / qty
        trail_act_distance = (saldo_usado * trail_act_pct / 100.0) / qty
        trail_dist_distance = (saldo_usado * trail_dist_pct / 100.0) / qty
        
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
        if time_stop_bars > 0:
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
            
            if is_trailing:
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
                    if tp_pct > 0 and h >= tp_price:
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
                    if tp_pct > 0 and low_val <= tp_price:
                        exit_idx = curr
                        exit_p = tp_price
                        exit_reason = 2
                        break
        
        if exit_idx == -1 and time_stop_bars > 0:
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
        pnl_pct = (pnl_neto / saldo_usado * 100) if saldo_usado > 0 else 0.0
        
        saldo_antes = current_balance
        current_balance += pnl_neto
        
        if current_balance < min_op:
            current_balance = min_op
        
        saldo_despues = current_balance
        
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
    
    # ========== ENTRADA AL INICIO DE LA SIGUIENTE VELA (OPEN) ==========
    # Desplazamos la entrada a la siguiente vela
    entry_indices = signal_indices + 1
    
    # Filtrar entradas que quedarían fuera del array (última vela con señal)
    valid_mask = entry_indices < df.height
    entry_indices = entry_indices[valid_mask]
    entry_types = signal_types[valid_mask]
    
    # Precio de entrada = OPEN de la vela de entrada (la siguiente a la señal)
    entry_prices = o_arr[entry_indices]
    
    is_trailing = params.exit_type == "pnl_trailing"
    
    # Parámetros
    fee_rate = float(params.comision_pct)
    min_op = float(params.saldo_minimo_operativo)
    apalancamiento_max = float(params.apalancamiento_max)
    saldo_usado_cfg = float(params.saldo_usado)
    sl_pct = float(params.exit_sl_pct)
    tp_pct = float(params.exit_tp_pct)
    trail_act = float(params.exit_trail_act_pct)
    trail_dist = float(params.exit_trail_dist_pct)
    time_stop = int(params.time_stop_bars)
    comision_sides_int = int(params.comision_sides)
    saldo_inicial = float(params.saldo_inicial)
    
    # =========================================================================
    # 3) Decidir si usar salidas en 1m
    # =========================================================================
    use_1m_exits = (df_1m is not None and timeframe_minutes > 1)
    
    if use_1m_exits:
        # Usar kernel con salidas en 1m
        c_1m = df_1m["close"].to_numpy()
        h_1m = df_1m["high"].to_numpy() if "high" in df_1m.columns else c_1m
        l_1m = df_1m["low"].to_numpy() if "low" in df_1m.columns else c_1m
        ts_1m = df_1m["timestamp"].to_numpy()
        ts_entry = df["timestamp"].to_numpy()
        
        tf_ratio = timeframe_minutes  # Ratio de velas 1m por vela TF
        
        # Señales de salida en 1m
        if signals_1m is not None and "exit_long" in signals_1m.columns:
            exit_long_1m = signals_1m["exit_long"].fill_null(False).to_numpy()
        else:
            exit_long_1m = np.zeros(len(df_1m), dtype=np.bool_)
            
        if signals_1m is not None and "exit_short" in signals_1m.columns:
            exit_short_1m = signals_1m["exit_short"].fill_null(False).to_numpy()
        else:
            exit_short_1m = np.zeros(len(df_1m), dtype=np.bool_)
        
        (out_entry_idx, out_exit_idx, out_entry_price, out_exit_price,
         out_side, out_reason, out_qty, out_saldo_usado, out_pnl_neto,
         out_pnl_pct, out_saldo_antes, out_saldo_despues,
         out_trail_act_idx, out_trail_act_price, trade_count) = _simulate_trades_with_1m_exits(
            entry_indices=entry_indices,
            entry_prices=entry_prices,
            entry_types=entry_types,
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
            close_prices=c_arr,
            high_prices=h_arr,
            low_prices=l_arr,
            saldo_inicial=saldo_inicial,
            fee_rate=fee_rate,
            min_op=min_op,
            apalancamiento_max=apalancamiento_max,
            saldo_usado_cfg=saldo_usado_cfg,
            is_trailing=is_trailing,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            trail_act_pct=trail_act,
            trail_dist_pct=trail_dist,
            time_stop_bars=time_stop,
            comision_sides=comision_sides_int,
            custom_exit_long=exit_long,
            custom_exit_short=exit_short,
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
    })
    
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
        _simulate_trades_sequential(
            dummy_entries, dummy_prices, dummy_types,
            dummy_close, dummy_high, dummy_low,
            1000.0, 0.001, 100.0, 10.0, 75.0,
            False, 5.0, 10.0, 0.0, 0.0, 0, 2,
            dummy_bool, dummy_bool
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
        _simulate_trades_with_1m_exits(
            dummy_entries, dummy_prices, dummy_types, dummy_ts,
            dummy_close_1m, dummy_high_1m, dummy_low_1m, dummy_ts_1m,
            1000.0, 0.001, 100.0, 10.0, 75.0,
            False, 5.0, 10.0, 0.0, 0.0, 0, 2,
            dummy_bool_1m, dummy_bool_1m, 5
        )
    except Exception:
        pass


# Ejecutar warmup al importar
try:
    _warmup_jit()
except Exception:
    pass
