"""
modelox/core/engine.py

Motor de Ejecución con Position Sizing Institucional (Fixed Fractional).

Cambios clave:
- Sin ATR: todas las distancias son porcentuales
- Sizing por riesgo fijo: qty = risk_amount / sl_distance
- sl_distance se calcula en generate_trades y se pasa a simulate_trades
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import polars as pl

from modelox.core.types import BacktestConfig, Strategy
from modelox.core.exits import (
    decide_exit_for_trade,
    exit_settings_from_params,
    trace_exit_pnl_trailing,
)


_TRAILING_TRACE_DUMPED = False


def generate_trades(
    df: pl.DataFrame,
    params: Dict[str, Any],
    *,
    saldo_apertura: float,
    strategy: Strategy,
) -> pd.DataFrame:
    """
    Genera un DataFrame base de trades usando arrays de Polars/Numpy.
    
    Ahora incluye `sl_distance` para cada trade (usado para sizing).
    """
    global _TRAILING_TRACE_DUMPED

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

    # Exit settings (sistema porcentual)
    exit_settings = exit_settings_from_params(params)

    # Sizing base para exits basados en STAKE
    STAKE_MIN = 60.0
    qty_target = float(params.get("__qty_max_activo", params.get("qty_max_activo", 0.0)))
    max_leverage = float(params.get("__apalancamiento", 1.0))
    if max_leverage <= 0:
        max_leverage = 1.0

    if open_ is None or high is None or low is None:
        raise ValueError("Se requieren columnas open/high/low para lógica intra-vela")

    operaciones: List[Dict[str, Any]] = []
    last_exit_idx = -1

    s_long_idx = np.where(signal_long)[0]
    s_short_idx = np.where(signal_short)[0]
    all_indices = np.unique(np.concatenate([s_long_idx, s_short_idx]))
    all_indices.sort()

    idx_pos = 0
    while idx_pos < len(all_indices):
        i = all_indices[idx_pos]

        # No abrir trades en la última vela
        if i >= (len(close) - 1):
            idx_pos += 1
            continue

        if block_velas_after_exit > 0 and last_exit_idx >= 0:
            if (i - last_exit_idx) <= block_velas_after_exit:
                idx_pos += 1
                continue

        if signal_long[i]:
            entry_idx = i
            entry_price = float(close[entry_idx])

            # stake/qty para este trade (se usan para SL/TP/Trailing sobre stake)
            q = float(qty_target) if qty_target > 0 else 0.0001
            notional = float(q * entry_price)
            lev_needed_for_stake_min = (notional / STAKE_MIN) if STAKE_MIN > 0 else max_leverage
            leverage_eff = min(max_leverage, max(1.0, lev_needed_for_stake_min))
            stake = (notional / leverage_eff) if leverage_eff > 0 else notional

            exit_result = decide_exit_for_trade(
                strategy=strategy,
                df=df,
                params=params,
                saldo_apertura=float(saldo_apertura),
                side="LONG",
                entry_idx=int(entry_idx),
                entry_price=float(entry_price),
                qty=float(q),
                stake=float(stake),
                close=close,
                open_=open_,
                high=high,
                low=low,
                settings=exit_settings,
            )
            exit_idx = int(exit_result.exit_idx)
            exit_price = float(exit_result.exit_price)
            tipo_salida = str(exit_result.tipo_salida)
            sl_distance = float(exit_result.sl_distance)

            # Optional: dump a bar-by-bar trace for the first TRAILING_STOP trade.
            # Enable with: MODELOX_TRACE_TRAILING=1
            if (
                (not _TRAILING_TRACE_DUMPED)
                and (os.environ.get("MODELOX_TRACE_TRAILING", "0") in {"1", "true", "True", "YES", "yes"})
                and (tipo_salida == "TRAILING_STOP")
                and (str(exit_settings.exit_type).strip().lower() == "pnl_trailing")
            ):
                try:
                    activo_name = str(params.get("__activo", "")) or "DEFAULT"
                    strat_name = getattr(strategy, "name", "strategy")
                    out_dir = os.path.join("resultados", "_debug")
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(
                        out_dir,
                        f"trailing_trace_{strat_name}_{activo_name}_entry{entry_idx}_exit{exit_idx}.csv".replace(" ", "_"),
                    )
                    rows = trace_exit_pnl_trailing(
                        side="LONG",
                        entry_idx=int(entry_idx),
                        entry_price=float(entry_price),
                        qty=float(q),
                        stake=float(stake),
                        timestamps=timestamps,
                        close=close,
                        open_=open_,
                        high=high,
                        low=low,
                        settings=exit_settings,
                    )
                    import pandas as _pd

                    _pd.DataFrame(rows).to_csv(out_path, index=False)
                    _TRAILING_TRACE_DUMPED = True
                except Exception:
                    # Never break backtests due to optional debug tracing.
                    _TRAILING_TRACE_DUMPED = True

            operaciones.append(
                {
                    "type": "long",
                    "entry_idx": int(entry_idx),
                    "exit_idx": int(exit_idx),
                    "entry_time": timestamps[entry_idx],
                    "exit_time": timestamps[exit_idx],
                    "entry_price": entry_price,
                    "exit_price": float(exit_price),
                    "tipo_salida": tipo_salida,
                    "sl_distance": sl_distance,
                    "qty": float(q),
                    "stake": float(stake),
                    "leverage_eff": float(leverage_eff),
                }
            )
            last_exit_idx = int(exit_idx)
            while idx_pos < len(all_indices) and all_indices[idx_pos] <= exit_idx:
                idx_pos += 1
            continue

        elif signal_short[i]:
            entry_idx = i
            entry_price = float(close[entry_idx])

            q = float(qty_target) if qty_target > 0 else 0.0001
            notional = float(q * entry_price)
            lev_needed_for_stake_min = (notional / STAKE_MIN) if STAKE_MIN > 0 else max_leverage
            leverage_eff = min(max_leverage, max(1.0, lev_needed_for_stake_min))
            stake = (notional / leverage_eff) if leverage_eff > 0 else notional

            exit_result = decide_exit_for_trade(
                strategy=strategy,
                df=df,
                params=params,
                saldo_apertura=float(saldo_apertura),
                side="SHORT",
                entry_idx=int(entry_idx),
                entry_price=float(entry_price),
                qty=float(q),
                stake=float(stake),
                close=close,
                open_=open_,
                high=high,
                low=low,
                settings=exit_settings,
            )
            exit_idx = int(exit_result.exit_idx)
            exit_price = float(exit_result.exit_price)
            tipo_salida = str(exit_result.tipo_salida)
            sl_distance = float(exit_result.sl_distance)

            if (
                (not _TRAILING_TRACE_DUMPED)
                and (os.environ.get("MODELOX_TRACE_TRAILING", "0") in {"1", "true", "True", "YES", "yes"})
                and (tipo_salida == "TRAILING_STOP")
                and (str(exit_settings.exit_type).strip().lower() == "pnl_trailing")
            ):
                try:
                    activo_name = str(params.get("__activo", "")) or "DEFAULT"
                    strat_name = getattr(strategy, "name", "strategy")
                    out_dir = os.path.join("resultados", "_debug")
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(
                        out_dir,
                        f"trailing_trace_{strat_name}_{activo_name}_entry{entry_idx}_exit{exit_idx}.csv".replace(" ", "_"),
                    )
                    rows = trace_exit_pnl_trailing(
                        side="SHORT",
                        entry_idx=int(entry_idx),
                        entry_price=float(entry_price),
                        qty=float(q),
                        stake=float(stake),
                        timestamps=timestamps,
                        close=close,
                        open_=open_,
                        high=high,
                        low=low,
                        settings=exit_settings,
                    )
                    import pandas as _pd

                    _pd.DataFrame(rows).to_csv(out_path, index=False)
                    _TRAILING_TRACE_DUMPED = True
                except Exception:
                    _TRAILING_TRACE_DUMPED = True

            operaciones.append(
                {
                    "type": "short",
                    "entry_idx": int(entry_idx),
                    "exit_idx": int(exit_idx),
                    "entry_time": timestamps[entry_idx],
                    "exit_time": timestamps[exit_idx],
                    "entry_price": entry_price,
                    "exit_price": float(exit_price),
                    "tipo_salida": tipo_salida,
                    "sl_distance": sl_distance,
                    "qty": float(q),
                    "stake": float(stake),
                    "leverage_eff": float(leverage_eff),
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
    """Simulación financiera.

    Sizing por trade (modo actual):
    - `qty` objetivo = `qty_max_activo` (fijo por trial/config)
    - Si no es posible por saldo/stake_max/apalancamiento, se reduce `qty` al máximo factible.
    - Si el stake resultante es < 60, se fuerza stake=60 reduciendo el apalancamiento efectivo
      (manteniendo la qty objetivo) siempre que el stake máximo posible lo permita.

    EARLY EXIT si el saldo cae por debajo del mínimo operativo.
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

    # qty/stake vienen de generate_trades (si no existen, fallback a qty_max_activo y stake mínimo)
    qty_arr = trades_base["qty"].values if "qty" in trades_base.columns else None
    stake_arr = trades_base["stake"].values if "stake" in trades_base.columns else None

    # Inicialización de arrays para métricas
    pnl_bruto = np.zeros(n)
    pnl_neto = np.zeros(n)
    pnl_pct = np.zeros(n)
    saldo_despues = np.zeros(n)
    saldo_antes = np.zeros(n)
    stake_aplicado = np.zeros(n)
    qty_aplicada = np.zeros(n)
    comisiones = np.zeros(n)

    equity_curve = [saldo]
    last_idx = 0

    for i in range(n):
        # === EARLY EXIT: Cuenta quebrada ===
        if saldo <= saldo_min:
            break

        saldo_antes[i] = saldo
        
        # =====================================================
        # QTY/STAKE: coherente con exits (basado en stake)
        # =====================================================
        q = float(qty_arr[i]) if qty_arr is not None else float(qty_max_limit)
        if q <= 0:
            q = 0.0001

        stk = float(stake_arr[i]) if stake_arr is not None else 60.0
        if stk <= 0:
            stk = 60.0

        # Nota: por diseño, el stake usado en exits es el stake del trade.
        # Si quieres impedir abrir trades cuando stk > saldo/stake_max, eso requiere
        # mover la lógica de apertura/validación a un motor secuencial (saldo-aware).

        # =====================================================
        # CÁLCULO DE PnL
        # =====================================================
        is_long = sides[i].lower() == "long"
        bruto = (
            (exit_p[i] - entry_p[i]) * q if is_long else (entry_p[i] - exit_p[i]) * q
        )

        c_ent = (q * entry_p[i] * comision_pct) if comision_sides >= 2 else 0.0
        c_ext = q * exit_p[i] * comision_pct
        c_tot = c_ent + c_ext

        neto = bruto - c_tot
        nuevo_saldo = saldo + neto
        
        # Liquidación implícita: saldo no puede ser negativo
        if nuevo_saldo < 0:
            nuevo_saldo = 0.0

        # === EARLY EXIT: Cuenta quebrada después del trade ===
        if nuevo_saldo <= saldo_min:
            pnl_bruto[i] = bruto
            pnl_neto[i] = neto
            pnl_pct[i] = (neto / stk) * 100 if stk > 0 else 0
            comisiones[i] = c_tot
            stake_aplicado[i] = stk
            qty_aplicada[i] = q
            saldo = float(nuevo_saldo)
            saldo_despues[i] = saldo
            equity_curve.append(saldo)
            last_idx = i + 1
            break

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

    # Construcción del DataFrame final
    df_exec = trades_base.iloc[:last_idx].copy()
    df_exec["pnl"] = pnl_bruto[:last_idx]
    df_exec["pnl_neto"] = pnl_neto[:last_idx]
    df_exec["pnl_pct"] = pnl_pct[:last_idx]
    df_exec["saldo_despues"] = saldo_despues[:last_idx]
    df_exec["saldo_antes"] = saldo_antes[:last_idx]
    df_exec["stake"] = stake_aplicado[:last_idx]
    df_exec["qty"] = qty_aplicada[:last_idx]
    df_exec["comision"] = comisiones[:last_idx]

    return df_exec, equity_curve
