"""
# =============================================================================
#
#     ███████╗██╗  ██╗██╗████████╗███████╗
#     ██╔════╝╚██╗██╔╝██║╚══██╔══╝██╔════╝
#     █████╗   ╚███╔╝ ██║   ██║   ███████╗
#     ██╔══╝   ██╔██╗ ██║   ██║   ╚════██║
#     ███████╗██╔╝ ██╗██║   ██║   ███████║
#     ╚══════╝╚═╝  ╚═╝╚═╝   ╚═╝   ╚══════╝
#
#     EXITS.PY - SISTEMA DE SALIDAS POR PNL%
#
# =============================================================================
#
#     TIPOS DE SALIDA:
#     - "pnl_fixed": SL/TP fijos por % sobre stake
#     - "pnl_trailing": SL inicial + trailing activado por %
#
#     DEFINICIÓN:
#     PNL_PCT = (pnl_eur / stake) × 100
#     Los parámetros sl_pct, tp_pct, etc. son % respecto al STAKE.
#
# =============================================================================
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np



# =============================================================================
# 1. CONSTANTES POR DEFECTO
# =============================================================================

DEFAULT_EXIT_TYPE: str = "pnl_trailing"
DEFAULT_EXIT_SL_PCT: float = 8.0
DEFAULT_EXIT_TP_PCT: float = 14.0
DEFAULT_EXIT_TRAIL_ACT_PCT: float = 15.0
DEFAULT_EXIT_TRAIL_DIST_PCT: float = 3.0
DEFAULT_OPTIMIZE_EXITS: bool = True

# RANGOS PARA OPTUNA (min, max, step)
DEFAULT_EXIT_SL_PCT_RANGE: tuple = (1.0, 50.0, 1.0)
DEFAULT_EXIT_TP_PCT_RANGE: tuple = (20.0, 40.0, 1.0)
DEFAULT_EXIT_TRAIL_ACT_PCT_RANGE: tuple = (1.0, 50.0, 1.0)
DEFAULT_EXIT_TRAIL_DIST_PCT_RANGE: tuple = (0.5, 20.0, 0.5)


# =============================================================================
# 2. ESTRUCTURAS DE DATOS
# =============================================================================

@dataclass(frozen=True)
class ExitSettings:
    """CONFIGURACIÓN DE SALIDA BASADA EN PNL%."""
    exit_type: str = DEFAULT_EXIT_TYPE
    sl_pct: float = DEFAULT_EXIT_SL_PCT
    tp_pct: float = DEFAULT_EXIT_TP_PCT
    trail_act_pct: float = DEFAULT_EXIT_TRAIL_ACT_PCT
    trail_dist_pct: float = DEFAULT_EXIT_TRAIL_DIST_PCT
    time_stop_bars: int = 0


@dataclass(frozen=True)
class ExitResult:
    """RESULTADO DE UNA SALIDA."""
    exit_idx: int
    exit_price: float
    tipo_salida: str
    sl_distance: float = 0.0


@dataclass(frozen=True)
class IntrabarExit:
    """RESULTADO DE CHEQUEO INTRA-BARRA."""
    triggered: bool
    reason: str = ""
    exit_price: float | None = None


# =============================================================================
# 3. NORMALIZACIÓN DE VALORES
# =============================================================================

def _normalize_exit_values(
    exit_type: str,
    sl_pct: float,
    tp_pct: float,
    trail_act_pct: float,
    trail_dist_pct: float,
) -> tuple:
    """
    NORMALIZA Y VALIDA VALORES DE EXIT
    
    Reglas:
    - Todos positivos (abs)
    - En trailing, tp_pct puede ser 0
    - trail_dist_pct <= trail_act_pct
    """
    sl_pct = abs(sl_pct) if sl_pct != 0 else 1.0
    
    if exit_type == "pnl_trailing":
        tp_pct = abs(tp_pct) if tp_pct != 0 else 0.0
    else:
        tp_pct = abs(tp_pct) if tp_pct != 0 else 1.0
    
    trail_act_pct = abs(trail_act_pct) if trail_act_pct != 0 else 0.5
    trail_dist_pct = abs(trail_dist_pct) if trail_dist_pct != 0 else 0.25
    
    # DISTANCIA NO PUEDE SUPERAR ACTIVACIÓN
    if trail_dist_pct > trail_act_pct:
        trail_dist_pct = trail_act_pct
    
    return float(sl_pct), float(tp_pct), float(trail_act_pct), float(trail_dist_pct)


# =============================================================================
# 4. RESOLUCIÓN DE PARÁMETROS
# =============================================================================

def resolve_exit_settings_for_trial(*, trial: Any, config: Any) -> ExitSettings:
    """
    RESUELVE PARÁMETROS DE SALIDA PARA UN TRIAL
    
    Fuentes (en orden de prioridad):
    1. Optuna sugiere si optimize_exits=True
    2. Config del backtest
    3. Defaults de este archivo
    """
    optimize_exits = bool(getattr(config, "optimize_exits", DEFAULT_OPTIMIZE_EXITS))
    exit_type = str(getattr(config, "exit_type", DEFAULT_EXIT_TYPE)).strip().lower()
    
    # VALORES BASE
    sl_pct = float(getattr(config, "exit_sl_pct", DEFAULT_EXIT_SL_PCT))
    tp_pct = float(getattr(config, "exit_tp_pct", DEFAULT_EXIT_TP_PCT))
    trail_act_pct = float(getattr(config, "exit_trail_act_pct", DEFAULT_EXIT_TRAIL_ACT_PCT))
    trail_dist_pct = float(getattr(config, "exit_trail_dist_pct", DEFAULT_EXIT_TRAIL_DIST_PCT))
    
    if exit_type == "pnl_trailing":
        tp_pct = 0.0
    
    # OPTIMIZACIÓN CON OPTUNA
    if optimize_exits:
        sl_rng = tuple(getattr(config, "exit_sl_pct_range", DEFAULT_EXIT_SL_PCT_RANGE))
        sl_pct = float(trial.suggest_float("exit_sl_pct", sl_rng[0], sl_rng[1], step=sl_rng[2]))
        
        if exit_type in {"pnl_fixed", "all"}:
            tp_rng = tuple(getattr(config, "exit_tp_pct_range", DEFAULT_EXIT_TP_PCT_RANGE))
            tp_pct = float(trial.suggest_float("exit_tp_pct", tp_rng[0], tp_rng[1], step=tp_rng[2]))
        else:
            tp_pct = 0.0
        
        if exit_type in {"pnl_trailing", "percent_trailing", "all"}:
            act_rng = tuple(getattr(config, "exit_trail_act_pct_range", DEFAULT_EXIT_TRAIL_ACT_PCT_RANGE))
            dist_rng = tuple(getattr(config, "exit_trail_dist_pct_range", DEFAULT_EXIT_TRAIL_DIST_PCT_RANGE))
            trail_act_pct = float(trial.suggest_float("exit_trail_act_pct", act_rng[0], act_rng[1], step=act_rng[2]))
            trail_dist_pct = float(trial.suggest_float("exit_trail_dist_pct", dist_rng[0], dist_rng[1], step=dist_rng[2]))
    
    # NORMALIZAR
    sl_pct, tp_pct, trail_act_pct, trail_dist_pct = _normalize_exit_values(
        exit_type, sl_pct, tp_pct, trail_act_pct, trail_dist_pct
    )
    
    return ExitSettings(
        exit_type=str(exit_type),
        sl_pct=sl_pct,
        tp_pct=tp_pct,
        trail_act_pct=trail_act_pct,
        trail_dist_pct=trail_dist_pct,
    )


def exit_settings_from_params(params: Dict[str, Any]) -> ExitSettings:
    """LEE SETTINGS DESDE DICCIONARIO DE PARAMS."""
    exit_type = str(params.get("__exit_type", params.get("exit_type", DEFAULT_EXIT_TYPE))).lower()
    sl_pct = float(params.get("__exit_sl_pct", params.get("exit_sl_pct", DEFAULT_EXIT_SL_PCT)))
    tp_pct = float(params.get("__exit_tp_pct", params.get("exit_tp_pct", DEFAULT_EXIT_TP_PCT)))
    trail_act_pct = float(params.get("__exit_trail_act_pct", params.get("exit_trail_act_pct", DEFAULT_EXIT_TRAIL_ACT_PCT)))
    trail_dist_pct = float(params.get("__exit_trail_dist_pct", params.get("exit_trail_dist_pct", DEFAULT_EXIT_TRAIL_DIST_PCT)))
    
    sl_pct, tp_pct, trail_act_pct, trail_dist_pct = _normalize_exit_values(
        exit_type, sl_pct, tp_pct, trail_act_pct, trail_dist_pct
    )
    
    return ExitSettings(
        exit_type=str(exit_type),
        sl_pct=sl_pct,
        tp_pct=tp_pct,
        trail_act_pct=trail_act_pct,
        trail_dist_pct=trail_dist_pct,
    )


# =============================================================================
# 5. CÁLCULO DE PNL%
# =============================================================================

def calc_pnl_pct(entry_price: float, current_price: float, side: str) -> float:
    """
    CALCULA PNL% DE UN TRADE
    
    LONG:  ((current - entry) / entry) × 100
    SHORT: ((entry - current) / entry) × 100
    """
    if entry_price <= 0:
        return 0.0
    
    s = (side or "").upper()
    if s == "LONG":
        return ((current_price - entry_price) / entry_price) * 100.0
    else:
        return ((entry_price - current_price) / entry_price) * 100.0


# =============================================================================
# 6. CHEQUEO INTRA-BARRA
# =============================================================================

def check_exit_pnl_intrabar(
    entry_price: float,
    high: float,
    low: float,
    close: float,
    side: str,
    settings: ExitSettings,
    max_pnl_pct_reached: float = 0.0,
    trailing_active: bool = False,
) -> IntrabarExit:
    """
    VERIFICA SI SE ACTIVA SALIDA EN UNA BARRA
    
    Revisa SL, TP y trailing según el tipo de exit.
    Retorna IntrabarExit con triggered=True si hay salida.
    """
    if entry_price <= 0:
        return IntrabarExit(triggered=False)
    
    s = (side or "").upper()
    is_long = (s == "LONG")
    
    # CALCULAR PNL% EN EXTREMOS
    if is_long:
        pnl_high = ((high - entry_price) / entry_price) * 100.0
        pnl_low = ((low - entry_price) / entry_price) * 100.0
    else:
        pnl_high = ((entry_price - low) / entry_price) * 100.0
        pnl_low = ((entry_price - high) / entry_price) * 100.0
    
    # CHEQUEAR SL
    if pnl_low <= -settings.sl_pct:
        sl_price = entry_price * (1 - settings.sl_pct / 100.0) if is_long else entry_price * (1 + settings.sl_pct / 100.0)
        return IntrabarExit(triggered=True, reason="SL", exit_price=sl_price)
    
    # CHEQUEAR TP (solo si no es trailing o es fixed)
    if settings.exit_type == "pnl_fixed" and settings.tp_pct > 0:
        if pnl_high >= settings.tp_pct:
            tp_price = entry_price * (1 + settings.tp_pct / 100.0) if is_long else entry_price * (1 - settings.tp_pct / 100.0)
            return IntrabarExit(triggered=True, reason="TP", exit_price=tp_price)
    
    # CHEQUEAR TRAILING
    if settings.exit_type == "pnl_trailing":
        new_max = max(max_pnl_pct_reached, pnl_high)
        
        # ACTIVAR TRAILING
        if not trailing_active and new_max >= settings.trail_act_pct:
            trailing_active = True
        
        # SI TRAILING ACTIVO, CHEQUEAR RETROCESO
        if trailing_active:
            pnl_close = ((close - entry_price) / entry_price) * 100.0 if is_long else ((entry_price - close) / entry_price) * 100.0
            if (new_max - pnl_close) >= settings.trail_dist_pct:
                trail_price = entry_price * (1 + (new_max - settings.trail_dist_pct) / 100.0) if is_long else entry_price * (1 - (new_max - settings.trail_dist_pct) / 100.0)
                return IntrabarExit(triggered=True, reason="TRAIL", exit_price=trail_price)
    
    return IntrabarExit(triggered=False)


# =============================================================================
# 7. FUNCIÓN VECTORIZADA PARA MÚLTIPLES TRADES
# =============================================================================

def find_exits_vectorized(
    entry_indices: np.ndarray,
    entry_prices: np.ndarray,
    entry_types: np.ndarray,
    entry_qty: np.ndarray,
    entry_stake: np.ndarray,
    close_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    settings: ExitSettings,
    time_stop_bars: int = 0,
) -> tuple:
    """
    ENCUENTRA SALIDAS PARA MÚLTIPLES TRADES
    
    Versión Python pura (fallback si no hay Numba).
    Retorna (exit_indices, exit_prices, exit_reasons).
    """
    n_entries = len(entry_indices)
    n_bars = len(close_prices)
    
    exit_indices = np.full(n_entries, -1, dtype=np.int64)
    exit_prices = np.full(n_entries, np.nan, dtype=np.float64)
    exit_reasons = np.zeros(n_entries, dtype=np.int32)
    
    is_trailing = settings.exit_type == "pnl_trailing"
    
    for i in range(n_entries):
        entry_idx = entry_indices[i]
        entry_price = entry_prices[i]
        side = entry_types[i]
        qty = entry_qty[i]
        stake = entry_stake[i]
        
        if qty <= 0 or stake <= 0:
            continue
        
        # CALCULAR DISTANCIAS EN PRECIO
        sl_distance = (stake * settings.sl_pct / 100.0) / qty
        tp_distance = (stake * settings.tp_pct / 100.0) / qty
        trail_act_distance = (stake * settings.trail_act_pct / 100.0) / qty
        trail_dist_distance = (stake * settings.trail_dist_pct / 100.0) / qty
        
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
            search_limit = min(n_bars, entry_idx + time_stop_bars + 1)
        
        for curr in range(entry_idx + 1, search_limit):
            h = high_prices[curr]
            l = low_prices[curr]
            
            if is_trailing:
                # SL INICIAL
                if not trailing_active:
                    if side == 1 and l <= sl_price:
                        exit_indices[i], exit_prices[i], exit_reasons[i] = curr, sl_price, 1
                        break
                    if side == -1 and h >= sl_price:
                        exit_indices[i], exit_prices[i], exit_reasons[i] = curr, sl_price, 1
                        break
                    
                    # ACTIVAR TRAILING
                    if (side == 1 and h >= activation_price) or (side == -1 and l <= activation_price):
                        trailing_active = True
                        trailing_level = h - trail_dist_distance if side == 1 else l + trail_dist_distance
                
                # TRAILING ACTIVO
                if trailing_active:
                    if side == 1:
                        new_level = h - trail_dist_distance
                        if new_level > trailing_level:
                            trailing_level = new_level
                        if l <= trailing_level:
                            exit_indices[i], exit_prices[i], exit_reasons[i] = curr, trailing_level, 3
                            break
                    else:
                        new_level = l + trail_dist_distance
                        if new_level < trailing_level:
                            trailing_level = new_level
                        if h >= trailing_level:
                            exit_indices[i], exit_prices[i], exit_reasons[i] = curr, trailing_level, 3
                            break
            else:
                # FIXED SL/TP
                if side == 1:
                    if l <= sl_price:
                        exit_indices[i], exit_prices[i], exit_reasons[i] = curr, sl_price, 1
                        break
                    if settings.tp_pct > 0 and h >= tp_price:
                        exit_indices[i], exit_prices[i], exit_reasons[i] = curr, tp_price, 2
                        break
                else:
                    if h >= sl_price:
                        exit_indices[i], exit_prices[i], exit_reasons[i] = curr, sl_price, 1
                        break
                    if settings.tp_pct > 0 and l <= tp_price:
                        exit_indices[i], exit_prices[i], exit_reasons[i] = curr, tp_price, 2
                        break
        
        # TIME STOP
        if exit_indices[i] == -1 and time_stop_bars > 0:
            final_idx = min(entry_idx + time_stop_bars, n_bars - 1)
            if final_idx > entry_idx:
                exit_indices[i] = final_idx
                exit_prices[i] = close_prices[final_idx]
                exit_reasons[i] = 4
    
    return exit_indices, exit_prices, exit_reasons

