from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Permite ejecutar el script desde cualquier working directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from general.configuracion import (
    SALDO_INICIAL,
    SALDO_OPERATIVO_MAX,
    COMISION_PCT,
    COMISION_SIDES,
    SALDO_MINIMO_OPERATIVO,
    SALDO_USADO,
    APALANCAMIENTO_MAX,
    EXIT_TYPE,
    EXIT_SL_PCT,
    EXIT_TP_PCT,
    EXIT_TRAIL_ACT_PCT,
    EXIT_TRAIL_DIST_PCT,
    resolve_archivo_data_tf,
    resolve_qty_max_activo,
)
from modelox.core.data import load_data
from modelox.core.engine import generate_trades, simulate_trades
from modelox.core.types import BacktestConfig, filter_by_date
from modelox.strategies.SUPERINDICADOR import EstrategiaResonanciaCinetica


def _make_params(*, activo: str) -> dict:
    # Midpoints (deterministas) dentro de los rangos de Optuna
    return {
        "len_trend": 30,
        "len_z": 100,
        "smooth": 5,
        "entry_threshold": 0.5,
        "exit_weakness_bars": 3,
        "emergency_sl_pct": 0.20,

        # Runtime params esperados por el engine
        "__activo": str(activo),
        "__saldo_usado": float(SALDO_USADO),
        "__apalancamiento_max": float(APALANCAMIENTO_MAX),
        "__qty_max_activo": float(resolve_qty_max_activo(activo)),

        # Exit settings globales (solo aplican si la estrategia no overridea)
        "__exit_type": str(EXIT_TYPE),
        "__exit_sl_pct": float(EXIT_SL_PCT),
        "__exit_tp_pct": float(EXIT_TP_PCT),
        "__exit_trail_act_pct": float(EXIT_TRAIL_ACT_PCT),
        "__exit_trail_dist_pct": float(EXIT_TRAIL_DIST_PCT),
    }


def _run_once(*, use_custom_exit: bool, activo: str, timeframe: int, start: str, end: str) -> pd.DataFrame:
    strategy = EstrategiaResonanciaCinetica()
    strategy.ACTIVAR_SALIDA_PERSONALIZADA = bool(use_custom_exit)

    path = resolve_archivo_data_tf(activo, timeframe, formato="feather")
    df = load_data(path)
    df = filter_by_date(df, start, end)

    params = _make_params(activo=activo)

    df_signals = strategy.generate_signals(df, params)

    trades_base = generate_trades(
        df_signals,
        params,
        saldo_apertura=float(SALDO_INICIAL),
        strategy=strategy,
    )

    cfg = BacktestConfig(
        saldo_inicial=float(SALDO_INICIAL),
        saldo_operativo_max=float(SALDO_OPERATIVO_MAX),
        comision_pct=float(COMISION_PCT),
        comision_sides=int(COMISION_SIDES),
        saldo_minimo_operativo=float(SALDO_MINIMO_OPERATIVO),
        qty_max_activo=float(resolve_qty_max_activo(activo)),
        saldo_usado=float(SALDO_USADO),
        apalancamiento_max=float(APALANCAMIENTO_MAX),
        exit_type=str(EXIT_TYPE),
        exit_sl_pct=float(EXIT_SL_PCT),
        exit_tp_pct=float(EXIT_TP_PCT),
        exit_trail_act_pct=float(EXIT_TRAIL_ACT_PCT),
        exit_trail_dist_pct=float(EXIT_TRAIL_DIST_PCT),
    )

    trades_exec, _equity = simulate_trades(trades_base=trades_base, config=cfg)
    return trades_exec


def main() -> None:
    p = argparse.ArgumentParser(description="Valida end-to-end exits custom vs global para SUPERINDICADOR.")
    p.add_argument("--activo", default="BTC")
    p.add_argument("--timeframe", type=int, default=15)
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="2021-12-31")
    args = p.parse_args()

    activo = str(args.activo).strip().upper()

    print("=== Config (resumen) ===")
    print(
        {
            "activo": activo,
            "timeframe": int(args.timeframe),
            "start": args.start,
            "end": args.end,
            "EXIT_TYPE": str(EXIT_TYPE),
            "EXIT_SL_PCT": float(EXIT_SL_PCT),
            "EXIT_TP_PCT": float(EXIT_TP_PCT),
            "EXIT_TRAIL_ACT_PCT": float(EXIT_TRAIL_ACT_PCT),
            "EXIT_TRAIL_DIST_PCT": float(EXIT_TRAIL_DIST_PCT),
        }
    )

    for flag in (True, False):
        print("\n=== RUN: ACTIVAR_SALIDA_PERSONALIZADA =", flag, "===")
        trades = _run_once(
            use_custom_exit=flag,
            activo=activo,
            timeframe=int(args.timeframe),
            start=str(args.start),
            end=str(args.end),
        )

        if trades is None or trades.empty:
            print("No hubo trades en este rango. Prueba ampliar fechas o bajar threshold.")
            continue

        vc = trades["tipo_salida"].astype(str).value_counts()
        print("Trades:", len(trades))
        print("tipo_salida value_counts:\n", vc.to_string())
        print("Sample (primeras 5 filas):")
        cols = [c for c in ["entry_time", "exit_time", "type", "entry_price", "exit_price", "tipo_salida"] if c in trades.columns]
        print(trades[cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
