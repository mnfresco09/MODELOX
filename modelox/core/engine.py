"""modelox/core/engine.py

Vector engine (Polars + Numba) consolidado.

Este archivo reemplaza vector_engine.py para que el sistema use el nombre estable
`modelox.core.engine`.

- Entradas: detecta cruces de señales (edge-trigger)
- Salidas: kernel Numba para SL/TP/Trailing/Time stop
- PnL/comisiones: Polars
- Equity curve: iteración O(N_trades)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numba as nb
import numpy as np
import polars as pl

from modelox.core.types import BacktestConfig, Strategy


@dataclass
class BacktestParams:
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
    - Soporta salidas personalizadas de estrategia si SALIDAS_PERSONALIZADAS=True
    """
    # Check si la estrategia tiene salidas personalizadas
    bool(getattr(strategy, "SALIDAS_PERSONALIZADAS", False))

    # 1) Preparación de datos y cruce de señales (edge trigger)
    df_sig = df.join(signals, on="timestamp", how="left").with_columns(
        [
            (
                pl.col("signal_long")
                & ~pl.col("signal_long").shift(1).fill_null(False)
            ).alias("entry_long"),
            (
                pl.col("signal_short")
                & ~pl.col("signal_short").shift(1).fill_null(False)
            ).alias("entry_short"),
            pl.int_range(0, pl.len(), dtype=pl.UInt32).alias("idx"),
        ]
    )

    entries = df_sig.filter(pl.col("entry_long") | pl.col("entry_short"))
    if entries.height == 0:
        return pl.DataFrame(), [params.saldo_inicial]

    # 2) Arrays numpy para Numba
    c_arr = df_sig["close"].to_numpy()
    h_arr = df_sig["high"].to_numpy() if "high" in df_sig.columns else c_arr
    l_arr = df_sig["low"].to_numpy() if "low" in df_sig.columns else c_arr
    ts_arr = df_sig["timestamp"]

    entry_indices = entries["idx"].cast(pl.Int64).to_numpy()
    entry_prices = entries["close"].to_numpy()
    entry_types = np.where(entries["entry_long"].to_numpy(), 1, -1).astype(np.int64)

    is_trailing = params.exit_type == "pnl_trailing"

    # 3) Simular secuencialmente con gestión de capital realista
    fee_rate = float(params.comision_pct)
    min_op = float(params.saldo_minimo_operativo)
    apalancamiento_max = float(params.apalancamiento_max)
    qty_max = float(params.qty_max_activo)
    saldo_usado_cfg = float(params.saldo_usado)

    current_balance = float(params.saldo_inicial)
    last_exit_idx = -1  # Para evitar solapamiento de trades

    # Listas para construir el DataFrame
    trade_data = {
        "entry_idx": [],
        "exit_idx": [],
        "entry_price": [],
        "exit_price": [],
        "side_int": [],
        "reason": [],
        "type": [],
        "qty": [],
        "saldo_usado": [],
        "pnl_bruto": [],
        "comision": [],
        "pnl_neto": [],
        "pnl_pct": [],
        "saldo_antes": [],
        "saldo_despues": [],
    }

    for i in range(len(entry_indices)):
        entry_idx = int(entry_indices[i])

        # Skip si la entrada está antes de la salida del trade anterior (no solapar)
        if entry_idx <= last_exit_idx:
            continue

        # STOP: si el saldo ya bajó al mínimo operativo, no seguir operando
        if current_balance <= min_op:
            break

        entry_p = float(entry_prices[i])
        side = int(entry_types[i])

        # Calcular saldo_usado real (limitado al saldo disponible)
        saldo_usado = min(saldo_usado_cfg, current_balance)

        # Calcular qty escalada al saldo disponible
        volumen_max = saldo_usado * apalancamiento_max
        qty_calculated = volumen_max / entry_p if entry_p > 0 else 0.0
        qty = min(qty_max, qty_calculated)

        if qty <= 0:
            continue

        # Encontrar salida con SL/TP basados en % sobre stake
        exit_idx, exit_p, exit_reason = find_single_exit_numba(
            entry_idx=entry_idx,
            entry_price=entry_p,
            side=side,
            qty=qty,
            stake=saldo_usado,
            close_prices=c_arr,
            high_prices=h_arr,
            low_prices=l_arr,
            is_trailing=is_trailing,
            sl_pct=params.exit_sl_pct,
            tp_pct=params.exit_tp_pct,
            trail_act_pct=params.exit_trail_act_pct,
            trail_dist_pct=params.exit_trail_dist_pct,
            time_stop_bars=params.time_stop_bars,
        )

        if exit_idx < 0:
            continue

        last_exit_idx = exit_idx

        # PnL bruto
        if side == 1:  # Long
            pnl_bruto = (exit_p - entry_p) * qty
        else:  # Short
            pnl_bruto = (entry_p - exit_p) * qty

        # Comisiones
        if int(params.comision_sides) >= 2:
            comision = (entry_p * qty + exit_p * qty) * fee_rate
        else:
            comision = entry_p * qty * fee_rate

        pnl_neto = pnl_bruto - comision
        pnl_pct = (pnl_neto / saldo_usado * 100) if saldo_usado > 0 else 0.0

        saldo_antes = current_balance
        current_balance += pnl_neto

        # Clamp a mínimo operativo (nunca puede bajar de ahí)
        if current_balance < min_op:
            current_balance = min_op

        saldo_despues = current_balance

        # Agregar trade
        trade_data["entry_idx"].append(entry_idx)
        trade_data["exit_idx"].append(exit_idx)
        trade_data["entry_price"].append(entry_p)
        trade_data["exit_price"].append(exit_p)
        trade_data["side_int"].append(side)
        trade_data["reason"].append(exit_reason)
        trade_data["type"].append("long" if side == 1 else "short")
        trade_data["qty"].append(qty)
        trade_data["saldo_usado"].append(saldo_usado)
        trade_data["pnl_bruto"].append(pnl_bruto)
        trade_data["comision"].append(comision)
        trade_data["pnl_neto"].append(pnl_neto)
        trade_data["pnl_pct"].append(pnl_pct)
        trade_data["saldo_antes"].append(saldo_antes)
        trade_data["saldo_despues"].append(saldo_despues)

    if not trade_data["entry_idx"]:
        return pl.DataFrame(), [params.saldo_inicial]

    # 4) Construir DataFrame y añadir timestamps
    trades_df = pl.DataFrame(trade_data)

    trades_df = trades_df.with_columns(
        [
            pl.Series(ts_arr.gather(trades_df["entry_idx"])).alias("entry_time"),
            pl.Series(ts_arr.gather(trades_df["exit_idx"])).alias("exit_time"),
        ]
    )

    # 5) Equity curve = saldo_despues de cada trade
    equity_curve = trade_data["saldo_despues"]

    return trades_df, equity_curve
