from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import polars as pl

from modelox.core.types import BacktestConfig, Strategy
from modelox.core.exits import (
    compute_atr_wilder,
    decide_exit_for_trade,
    exit_settings_from_params,
)


def generate_trades(
    df: pl.DataFrame,
    params: Dict[str, Any],
    *,
    saldo_apertura: float,
    strategy: Strategy,
) -> pd.DataFrame:
    """
    Genera un DataFrame base de trades usando arrays de Polars/Numpy.
    """
    close = df["close"].to_numpy()
    open_ = df["open"].to_numpy() if "open" in df.columns else None
    high = df["high"].to_numpy() if "high" in df.columns else None
    low = df["low"].to_numpy() if "low" in df.columns else None
    signal_long = (
        df["signal_long"].to_numpy()
        if "signal_long" in df.columns
        else np.zeros(len(df), dtype=bool)
    )
    signal_short = (
        df["signal_short"].to_numpy()
        if "signal_short" in df.columns
        else np.zeros(len(df), dtype=bool)
    )
    # Handle both 'timestamp' and 'datetime' column names
    if "timestamp" in df.columns:
        timestamps = df["timestamp"].to_numpy()
    elif "datetime" in df.columns:
        timestamps = df["datetime"].to_numpy()
    else:
        raise ValueError("DataFrame must have 'timestamp' or 'datetime' column")

    block_velas_after_exit = int(params.get("block_velas_after_exit", 0))

    # Exit settings (global) are centralized in modelox/core/exits.py
    exit_settings = exit_settings_from_params(params)

    if open_ is None or high is None or low is None:
        raise ValueError("Para SL/TP intra-vela por ATR se requieren columnas open/high/low")

    atr = compute_atr_wilder(high, low, close, exit_settings.atr_period)
    operaciones: List[Dict[str, Any]] = []
    last_exit_idx = -1

    s_long_idx = np.where(signal_long)[0]
    s_short_idx = np.where(signal_short)[0]
    all_indices = np.unique(np.concatenate([s_long_idx, s_short_idx]))
    all_indices.sort()

    idx_pos = 0
    while idx_pos < len(all_indices):
        i = all_indices[idx_pos]

        if block_velas_after_exit > 0 and last_exit_idx >= 0:
            if (i - last_exit_idx) <= block_velas_after_exit:
                idx_pos += 1
                continue

        if signal_long[i]:
            entry_idx = i
            entry_price = float(close[entry_idx])

            exit_result = decide_exit_for_trade(
                strategy=strategy,
                df=df,
                params=params,
                saldo_apertura=float(saldo_apertura),
                side="LONG",
                entry_idx=int(entry_idx),
                entry_price=float(entry_price),
                close=close,
                open_=open_,
                high=high,
                low=low,
                atr=atr,
                settings=exit_settings,
            )
            exit_idx = int(exit_result.exit_idx)
            exit_price = float(exit_result.exit_price)
            tipo_salida = str(exit_result.tipo_salida)

            operaciones.append(
                {
                    "type": "long",
                    "entry_time": timestamps[entry_idx],
                    "exit_time": timestamps[exit_idx],
                    "entry_price": entry_price,
                    "exit_price": float(exit_price),
                    "tipo_salida": tipo_salida,
                }
            )
            last_exit_idx = int(exit_idx)
            while idx_pos < len(all_indices) and all_indices[idx_pos] <= exit_idx:
                idx_pos += 1
            continue

        elif signal_short[i]:
            entry_idx = i
            entry_price = float(close[entry_idx])

            exit_result = decide_exit_for_trade(
                strategy=strategy,
                df=df,
                params=params,
                saldo_apertura=float(saldo_apertura),
                side="SHORT",
                entry_idx=int(entry_idx),
                entry_price=float(entry_price),
                close=close,
                open_=open_,
                high=high,
                low=low,
                atr=atr,
                settings=exit_settings,
            )
            exit_idx = int(exit_result.exit_idx)
            exit_price = float(exit_result.exit_price)
            tipo_salida = str(exit_result.tipo_salida)

            operaciones.append(
                {
                    "type": "short",
                    "entry_time": timestamps[entry_idx],
                    "exit_time": timestamps[exit_idx],
                    "entry_price": entry_price,
                    "exit_price": float(exit_price),
                    "tipo_salida": tipo_salida,
                }
            )
            last_exit_idx = int(exit_idx)
            while idx_pos < len(all_indices) and all_indices[idx_pos] <= exit_idx:
                idx_pos += 1
            continue
        idx_pos += 1

    trades = pd.DataFrame(operaciones)
    if not trades.empty:
        trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
        trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)
        trades["duracion_min"] = (
            trades["exit_time"] - trades["entry_time"]
        ).dt.total_seconds() / 60.0
    return trades


def simulate_trades(
    *,
    trades_base: pd.DataFrame,
    config: BacktestConfig,
) -> Tuple[pd.DataFrame, List[float]]:
    """
    Simulación financiera completa con EARLY EXIT para cuentas quebradas.
    
    Performance: Si saldo cae por debajo de saldo_minimo_operativo, 
    el bucle hace break instantáneo para ahorrar tiempo de procesamiento.
    """
    saldo = float(config.saldo_inicial)
    stake_max = float(config.saldo_operativo_max)
    apalancamiento = float(config.apalancamiento)
    comision_pct = float(config.comision_pct)
    comision_sides = int(config.comision_sides)
    saldo_min = float(config.saldo_minimo_operativo)
    qty_max_limit = float(config.qty_max_activo)
    
    # === EARLY EXIT: Cuenta ya quebrada desde el inicio ===
    if saldo <= saldo_min:
        return pd.DataFrame(), [saldo]

    if trades_base is None or trades_base.empty:
        return pd.DataFrame(), [saldo]

    n = len(trades_base)
    entry_p = trades_base["entry_price"].values
    exit_p = trades_base["exit_price"].values
    sides = trades_base["type"].values

    # Inicialización de arrays para métricas
    pnl_bruto = np.zeros(n)  # Esto será la columna 'pnl' esperada
    pnl_neto = np.zeros(n)
    pnl_pct = np.zeros(n)
    saldo_despues = np.zeros(n)
    saldo_antes = np.zeros(n)
    stake_aplicado = np.zeros(n)
    qty_aplicada = np.zeros(n)
    comisiones = np.zeros(n)

    equity_curve = [saldo]
    last_idx = 0

    # (diagnostic prints disabled)

    for i in range(n):
        # === EARLY EXIT: Cuenta quebrada - stop inmediato ===
        if saldo <= saldo_min:
            # No procesar más trades, la cuenta está en bancarrota
            break

        saldo_antes[i] = saldo
        # 1. Calcula el stake máximo permitido por saldo/configuración
        stake_max_posible = min(saldo, stake_max)
        # 2. ¿Cuánta cantidad se podría abrir con ese stake?
        qty_con_stake_max = (stake_max_posible * apalancamiento) / entry_p[i]
        initial_q = float(qty_con_stake_max)
        # 3. Si el stake máximo permite abrir más que qty_max_activo, ajusta el stake justo al necesario para abrir qty_max_activo
        if qty_con_stake_max > qty_max_limit:
            q = float(qty_max_limit)
            stk = (q * entry_p[i]) / apalancamiento
            q_clamped_by_qty = True
        else:
            stk = stake_max_posible
            q = (stk * apalancamiento) / entry_p[i]
            q_clamped_by_qty = False

        # Cálculo de PnL
        is_long = sides[i].lower() == "long"
        bruto = (
            (exit_p[i] - entry_p[i]) * q if is_long else (entry_p[i] - exit_p[i]) * q
        )

        c_ent = (q * entry_p[i] * comision_pct) if comision_sides >= 2 else 0.0
        c_ext = q * exit_p[i] * comision_pct
        c_tot = c_ent + c_ext

        neto = bruto - c_tot
        nuevo_saldo = saldo + neto

        # (diagnostic print removed)

        # === EARLY EXIT: Si el nuevo saldo cae por debajo o igual al mínimo ===
        if nuevo_saldo <= saldo_min:
            # Registrar este último trade antes de salir
            pnl_bruto[i] = bruto
            pnl_neto[i] = neto
            pnl_pct[i] = (neto / stk) * 100 if stk > 0 else 0
                        # (debug print removed)
            last_idx = i + 1
            break  # STOP INMEDIATO - Cuenta quebrada

        # Asignación de valores
        pnl_bruto[i] = bruto
        pnl_neto[i] = neto
        pnl_pct[i] = (neto / stk) * 100 if stk > 0 else 0
        comisiones[i] = c_tot
        stake_aplicado[i] = stk
        qty_aplicada[i] = q

        saldo = nuevo_saldo
        saldo_despues[i] = saldo
        equity_curve.append(saldo)
        last_idx = i + 1

    # Construcción del DataFrame final con mapeo exacto para metrics.py
    df_exec = trades_base.iloc[:last_idx].copy()
    df_exec["pnl"] = pnl_bruto[:last_idx]  # Requerido para 'saldo_sin_comisiones'
    df_exec["pnl_neto"] = pnl_neto[:last_idx]  # Requerido para winrate/expectativa
    df_exec["pnl_pct"] = pnl_pct[:last_idx]  # Requerido para Sharpe/Sortino
    df_exec["saldo_despues"] = saldo_despues[:last_idx]
    df_exec["saldo_antes"] = saldo_antes[:last_idx]
    df_exec["stake"] = stake_aplicado[:last_idx]
    df_exec["qty"] = qty_aplicada[:last_idx]
    df_exec["comision"] = comisiones[:last_idx]

    return df_exec, equity_curve
