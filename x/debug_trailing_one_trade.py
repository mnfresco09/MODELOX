"""Debug bar-a-bar de trailing (stake-based).

Uso:
  /Users/manuel/Desktop/MODELOX/.venv311/bin/python x/debug_trailing_one_trade.py

Genera un CSV en resultados/_debug/ con la traza vela-a-vela del primer trade
(encontrado) para el que se pueda calcular salida pnl_trailing.

Si no existe un trade que termine con TRAILING_STOP, igualmente genera la traza
para el primer trade disponible para inspeccionar cómo evoluciona el trailing.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

# Allow running this file directly from anywhere.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from general.configuracion import (
    ACTIVO_PRIMARIO,
    resolve_archivo_data_tf,
    resolve_qty_max_activo,
    CONFIG,
    FECHA_INICIO,
    FECHA_FIN,
    SALDO_INICIAL,
    APALANCAMIENTO,
    EXIT_SL_PCT,
    EXIT_TP_PCT,
    EXIT_TRAIL_ACT_PCT,
    EXIT_TRAIL_DIST_PCT,
)
from modelox.core.data import load_data
from modelox.core.engine import generate_trades
from modelox.core.exits import exit_settings_from_params, trace_exit_pnl_trailing
from modelox.core.types import filter_by_date
from modelox.strategies.registry import instantiate_strategies


def main() -> None:
    activo = str(ACTIVO_PRIMARIO).strip().upper() or "BTC"
    timeframe_base = CONFIG.get("TIMEFRAME", 15)

    data_path = resolve_archivo_data_tf(activo, timeframe_base, formato="parquet")
    df = load_data(data_path)
    df = filter_by_date(df, FECHA_INICIO, FECHA_FIN)

    # Estrategia por ID (usa la config actual; si no hay, cae a 3)
    strategies = instantiate_strategies(only_id=3)
    if not strategies:
        raise RuntimeError("No se pudo instanciar estrategia ID=3")

    strategy = strategies[0]

    # Params base (sin Optuna): tomamos defaults razonables.
    # Para StrategyCrossoverHLMA: hl_period + ma_type.
    params = {
        "hl_period": 50,
        "ma_type": "sma",
        "exit_type": "pnl_trailing",
        "exit_sl_pct": float(EXIT_SL_PCT),
        "exit_tp_pct": float(EXIT_TP_PCT),
        "exit_trail_act_pct": float(EXIT_TRAIL_ACT_PCT),
        "exit_trail_dist_pct": float(EXIT_TRAIL_DIST_PCT),
        "__qty_max_activo": float(resolve_qty_max_activo(activo)),
        "__apalancamiento": float(APALANCAMIENTO),
        "__activo": activo,
    }

    # Generar señales
    df_signals = strategy.generate_signals(df, params)

    # Forzar exit settings desde params
    exit_settings = exit_settings_from_params(params)

    trades = generate_trades(
        df_signals,
        params,
        saldo_apertura=float(SALDO_INICIAL),
        strategy=strategy,
    )

    if trades is None or trades.empty:
        raise RuntimeError("No se generaron trades con estos parámetros.")

    # Elegir trade: preferimos uno que salga por TRAILING_STOP
    target = None
    trailing_mask = trades["tipo_salida"].astype(str).str.upper().eq("TRAILING_STOP")
    if trailing_mask.any():
        t = trades[trailing_mask].copy()
        if "entry_idx" in t.columns and "exit_idx" in t.columns:
            t["_bars"] = (t["exit_idx"].astype(int) - t["entry_idx"].astype(int)).clip(lower=0)
            target = t.sort_values("_bars", ascending=False).iloc[0]
        else:
            target = t.iloc[0]
    else:
        target = trades.iloc[0]

    side = "LONG" if str(target["type"]).lower() == "long" else "SHORT"
    entry_idx = int(target.get("entry_idx"))
    exit_idx = int(target.get("exit_idx"))
    entry_price = float(target.get("entry_price"))
    qty = float(target.get("qty"))
    stake = float(target.get("stake"))

    # Extraer arrays
    close = df_signals["close"].to_numpy()
    open_ = df_signals["open"].to_numpy()
    high = df_signals["high"].to_numpy()
    low = df_signals["low"].to_numpy()
    if "timestamp" in df_signals.columns:
        timestamps = df_signals["timestamp"].to_numpy()
    else:
        timestamps = df_signals["datetime"].to_numpy()

    rows = trace_exit_pnl_trailing(
        side=side,
        entry_idx=entry_idx,
        entry_price=entry_price,
        qty=qty,
        stake=stake,
        timestamps=timestamps,
        close=close,
        open_=open_,
        high=high,
        low=low,
        settings=exit_settings,
    )

    out_dir = Path("resultados") / "_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"manual_trailing_trace_{strategy.name}_{activo}_entry{entry_idx}_exit{exit_idx}.csv"

    pd.DataFrame(rows).to_csv(out_path, index=False)

    print("Trade seleccionado:")
    print(
        {
            "strategy": strategy.name,
            "activo": activo,
            "side": side,
            "entry_idx": entry_idx,
            "exit_idx": exit_idx,
            "entry_price": entry_price,
            "exit_price": float(target.get("exit_price")),
            "tipo_salida": str(target.get("tipo_salida")),
            "qty": qty,
            "stake": stake,
            "trace_rows": len(rows),
            "trace_csv": str(out_path),
        }
    )


if __name__ == "__main__":
    # Ensure cwd is repo root when launched from elsewhere.
    os.chdir(_ROOT)
    main()
