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
║                    │                                                        ║
║                    ▼                                                        ║
║  engine.py (EJECUCIÓN NUMBA - MÁXIMA VELOCIDAD)  ← ESTE ARCHIVO            ║
║    └── _simulate_trades_sequential(): Kernel Numba optimizado               ║
║    └── SL/TP/Trailing integrados en el kernel (zero-overhead)              ║
║    └── Extensión C opcional para aún más velocidad                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

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
from typing import Any, Dict, List, Tuple

import numba as nb
import numpy as np
import polars as pl

from modelox.core.types import BacktestConfig, Strategy


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
    qty_max_activo: float
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
            qty_max_activo=float(config.qty_max_activo),
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
# 3. KERNEL NUMBA: SALIDAS (SL/TP/TRAILING)
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
                        # Check hit del trailing
                        if low_val <= trailing_level:
                            exit_indices[i] = curr
                            exit_prices[i] = trailing_level
                            exit_reasons[i] = 3  # Trail
                            break
                    else:
                        # Actualizar trailing level hacia abajo
                        new_level = low_val + trail_dist_distance
                        if new_level < trailing_level:
                            trailing_level = new_level
                        # Check hit del trailing
                        if h >= trailing_level:
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
                    if side == 1:
                        trailing_level = h - trail_dist_distance
                    else:
                        trailing_level = low_val + trail_dist_distance

            if trailing_active:
                if side == 1:
                    new_level = h - trail_dist_distance
                    if new_level > trailing_level:
                        trailing_level = new_level
                    if low_val <= trailing_level:
                        return curr, trailing_level, 3  # Trail
                else:
                    new_level = low_val + trail_dist_distance
                    if new_level < trailing_level:
                        trailing_level = new_level
                    if h >= trailing_level:
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
    qty_max: float,
    saldo_usado_cfg: float,
    is_trailing: bool,
    sl_pct: float,
    tp_pct: float,
    trail_act_pct: float,
    trail_dist_pct: float,
    time_stop_bars: int,
    comision_sides: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
           np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, int]:
    """
    Kernel Numba optimizado para simular todos los trades secuencialmente.
    Retorna arrays con datos de cada trade exitoso.
    """
    n_entries = len(entry_indices)
    n_bars = len(close_prices)
    
    # Pre-allocate output arrays (máximo = número de entradas)
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
    
    current_balance = saldo_inicial
    last_exit_idx = -1
    trade_count = 0
    
    for i in range(n_entries):
        entry_idx = entry_indices[i]
        
        # Skip si la entrada está antes de la salida del trade anterior
        if entry_idx <= last_exit_idx:
            continue
        
        # STOP si el saldo ya bajó al mínimo operativo
        if current_balance <= min_op:
            break
        
        entry_p = entry_prices[i]
        side = entry_types[i]
        
        # Calcular saldo_usado real
        saldo_usado = min(saldo_usado_cfg, current_balance)
        
        # Calcular qty escalada al saldo disponible
        volumen_max = saldo_usado * apalancamiento_max
        qty_calculated = volumen_max / entry_p if entry_p > 0 else 0.0
        qty = min(qty_max, qty_calculated)
        
        if qty <= 0:
            continue
        
        # ========== Encontrar salida inline (evita overhead de llamada) ==========
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
        
        search_limit = n_bars
        if time_stop_bars > 0:
            limit = entry_idx + time_stop_bars + 1
            if limit < search_limit:
                search_limit = limit
        
        exit_idx = -1
        exit_p = 0.0
        exit_reason = 0
        
        for curr in range(entry_idx + 1, search_limit):
            h = high_prices[curr]
            low_val = low_prices[curr]
            
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
                        if side == 1:
                            trailing_level = h - trail_dist_distance
                        else:
                            trailing_level = low_val + trail_dist_distance
                
                if trailing_active:
                    if side == 1:
                        new_level = h - trail_dist_distance
                        if new_level > trailing_level:
                            trailing_level = new_level
                        if low_val <= trailing_level:
                            exit_idx = curr
                            exit_p = trailing_level
                            exit_reason = 3
                            break
                    else:
                        new_level = low_val + trail_dist_distance
                        if new_level < trailing_level:
                            trailing_level = new_level
                        if h >= trailing_level:
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
        
        # Time stop fallback
        if exit_idx == -1 and time_stop_bars > 0:
            final_idx = entry_idx + time_stop_bars
            if final_idx >= n_bars:
                final_idx = n_bars - 1
            if final_idx > entry_idx:
                exit_idx = final_idx
                exit_p = close_prices[final_idx]
                exit_reason = 4
        
        # End of data
        if exit_idx == -1:
            exit_idx = n_bars - 1
            exit_p = close_prices[exit_idx]
            exit_reason = 0
        
        # ========== Calcular PnL ==========
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
        
        # Guardar trade
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
            out_pnl_pct, out_saldo_antes, out_saldo_despues, trade_count)


# =============================================================================
# 5. FUNCIÓN PRINCIPAL: CALCULATE PERFORMANCE
# =============================================================================

def calculate_performance_vectorized_numba(
    *,
    df: pl.DataFrame,
    signals: pl.DataFrame,
    params: BacktestParams,
    strategy: Strategy,
) -> Tuple[pl.DataFrame, List[float]]:
    """
    Vector engine con gestión de capital realista:
    - Escala qty al saldo disponible (apalancamiento variable)
    - Calcula SL/TP basados en % sobre STAKE (no sobre precio)
    - Detiene el trading cuando saldo <= saldo_minimo_operativo
    - Equity curve refleja el balance real después de cada trade
    
    SALIDAS: Ejecutadas INLINE en kernel Numba para máxima velocidad.
             La configuración viene de exits.py via runner.py.
    
    OPTIMIZADO: JOIN eliminado, usa hstack + filtrado vectorizado directo.
    """

    # =========================================================================
    # 1) OPTIMIZACIÓN: Evitar JOIN costoso - usar select + hstack directo
    # =========================================================================
    # Solo necesitamos signal_long y signal_short de signals
    sig_long = signals["signal_long"].fill_null(False)
    sig_short = signals["signal_short"].fill_null(False)
    
    # Edge trigger vectorizado sin JOIN
    entry_long = sig_long & ~sig_long.shift(1).fill_null(False)
    entry_short = sig_short & ~sig_short.shift(1).fill_null(False)
    
    # Máscara de entradas
    entry_mask = entry_long | entry_short
    n_entries = entry_mask.sum()
    
    if n_entries == 0:
        return pl.DataFrame(), [params.saldo_inicial]
    
    # =========================================================================
    # 2) OPTIMIZACIÓN: Extraer arrays numpy una sola vez (evitar múltiples to_numpy)
    # =========================================================================
    c_arr = df["close"].to_numpy()
    h_arr = df["high"].to_numpy() if "high" in df.columns else c_arr
    l_arr = df["low"].to_numpy() if "low" in df.columns else c_arr
    ts_arr = df["timestamp"]
    
    # Índices de entrada usando arange + filter (más rápido que gather)
    all_indices = np.arange(df.height, dtype=np.int64)
    entry_mask_np = entry_mask.to_numpy()
    entry_indices = all_indices[entry_mask_np]
    
    # Precios y tipos de entrada
    entry_prices = c_arr[entry_indices]
    entry_long_np = entry_long.to_numpy()
    entry_types = np.where(entry_long_np[entry_indices], 1, -1).astype(np.int64)

    is_trailing = params.exit_type == "pnl_trailing"

    # 3) Simular secuencialmente con gestión de capital realista
    fee_rate = float(params.comision_pct)
    min_op = float(params.saldo_minimo_operativo)
    apalancamiento_max = float(params.apalancamiento_max)
    qty_max = float(params.qty_max_activo)
    saldo_usado_cfg = float(params.saldo_usado)

    # Pre-compute params for Numba
    sl_pct = float(params.exit_sl_pct)
    tp_pct = float(params.exit_tp_pct)
    trail_act = float(params.exit_trail_act_pct)
    trail_dist = float(params.exit_trail_dist_pct)
    time_stop = int(params.time_stop_bars)
    comision_sides_int = int(params.comision_sides)
    saldo_inicial = float(params.saldo_inicial)

    # =========================================================================
    # KERNEL OPTIMIZADO: C (si disponible) o Numba (fallback)
    # =========================================================================
    (out_entry_idx, out_exit_idx, out_entry_price, out_exit_price,
     out_side, out_reason, out_qty, out_saldo_usado, out_pnl_neto,
     out_pnl_pct, out_saldo_antes, out_saldo_despues, trade_count) = _simulate_trades_sequential(
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
        qty_max=qty_max,
        saldo_usado_cfg=saldo_usado_cfg,
        is_trailing=is_trailing,
        sl_pct=sl_pct,
        tp_pct=tp_pct,
        trail_act_pct=trail_act,
        trail_dist_pct=trail_dist,
        time_stop_bars=time_stop,
        comision_sides=comision_sides_int,
    )

    if trade_count == 0:
        return pl.DataFrame(), [saldo_inicial]

    # =========================================================================
    # 4) OPTIMIZACIÓN: Slicing directo sin copias innecesarias
    # =========================================================================
    # Vistas sobre arrays (no copian memoria)
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

    # Calcular PnL bruto y comisión vectorizado
    pnl_bruto = np.where(
        side_view == 1,
        (exit_price_view - entry_price_view) * qty_view,
        (entry_price_view - exit_price_view) * qty_view
    )
    if comision_sides_int >= 2:
        comision = (entry_price_view * qty_view + exit_price_view * qty_view) * fee_rate
    else:
        comision = entry_price_view * qty_view * fee_rate

    # Tipo de trade
    trade_type = np.where(side_view == 1, "long", "short")

    # =========================================================================
    # 5) Construir DataFrame optimizado - una sola llamada
    # =========================================================================
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
        "entry_time": ts_arr.gather(pl.Series(entry_idx_view)),
        "exit_time": ts_arr.gather(pl.Series(exit_idx_view)),
    })

    # 6) Equity curve como lista Python (más rápido que .tolist() para arrays pequeños)
    equity_curve = list(saldo_despues_view)

    return trades_df, equity_curve


# =============================================================================
# 6. WARMUP JIT (PRE-COMPILACIÓN AL IMPORTAR)
# =============================================================================

def _warmup_jit() -> None:
    """Pre-compila funciones Numba con datos dummy para eliminar latencia inicial."""
    # Datos dummy mínimos
    dummy_entries = np.array([0, 5, 10], dtype=np.int64)
    dummy_prices = np.array([100.0, 101.0, 102.0], dtype=np.float64)
    dummy_types = np.array([1, -1, 1], dtype=np.int64)
    dummy_close = np.linspace(100, 110, 20, dtype=np.float64)
    dummy_high = dummy_close * 1.01
    dummy_low = dummy_close * 0.99
    
    # Warmup _simulate_trades_sequential
    try:
        _simulate_trades_sequential(
            dummy_entries, dummy_prices, dummy_types,
            dummy_close, dummy_high, dummy_low,
            1000.0, 0.001, 100.0, 10.0, 1.0, 50.0,
            False, 5.0, 10.0, 0.0, 0.0, 0, 2
        )
    except Exception:
        pass
    
    # Warmup _compute_fast_metrics
    try:
        dummy_pnl = np.array([10.0, -5.0, 15.0], dtype=np.float64)
        dummy_pnl_pct = np.array([1.0, -0.5, 1.5], dtype=np.float64)
        dummy_saldo = np.array([1010.0, 1005.0, 1020.0], dtype=np.float64)
        _compute_fast_metrics(dummy_pnl, dummy_pnl_pct, dummy_saldo, 1000.0)
    except Exception:
        pass


# Ejecutar warmup al importar (solo si Numba está disponible)
try:
    _warmup_jit()
except Exception:
    pass
