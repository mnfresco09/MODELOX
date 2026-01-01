from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from modelox.core.types import ExitDecision


# =============================================================================
# PARÁMETROS GLOBALES DE SALIDA (ÚNICO LUGAR)
# =============================================================================
#
# Modelo global actual (confirmado en engine previo):
# - SL/TP calculados por ATR de Wilder en el momento de entrada (fijo al inicio)
# - Se ejecuta intra-vela usando open/high/low (conservador: SL antes que TP)
# - TIME EXIT por número máximo de velas
#
# Los valores pueden ser:
# - fijos por configuración
# - optimizados por Optuna (si optimize_exits/OPTIMIZAR_SALIDAS=True)
#
# En runtime, el runner inyecta las claves con prefijo `__exit_*` en `params`.
# =============================================================================

DEFAULT_EXIT_ATR_PERIOD = 14
DEFAULT_EXIT_SL_ATR = 1.0
DEFAULT_EXIT_TP_ATR = 1.0
DEFAULT_EXIT_TIME_STOP_BARS = 260

DEFAULT_OPTIMIZE_EXITS = True

DEFAULT_EXIT_ATR_PERIOD_RANGE = (7, 30, 1)
DEFAULT_EXIT_SL_ATR_RANGE = (0.5, 3.0, 0.1)
DEFAULT_EXIT_TP_ATR_RANGE = (1.0, 8.0, 0.1)
DEFAULT_EXIT_TIME_STOP_BARS_RANGE = (250, 800, 10)


def resolve_exit_settings_for_trial(*, trial: Any, config: Any) -> ExitSettings:
    """Resuelve los parámetros de salida para un trial.

    Fuente de verdad:
      - defaults/rangos definidos en este archivo
      - la configuración (BacktestConfig) puede sobreescribirlos
      - si `optimize_exits` está activo, Optuna sugiere dentro de los rangos
    """

    optimize_exits = bool(getattr(config, "optimize_exits", DEFAULT_OPTIMIZE_EXITS))

    exit_atr_period = int(getattr(config, "exit_atr_period", DEFAULT_EXIT_ATR_PERIOD))
    exit_sl_atr = float(getattr(config, "exit_sl_atr", DEFAULT_EXIT_SL_ATR))
    exit_tp_atr = float(getattr(config, "exit_tp_atr", DEFAULT_EXIT_TP_ATR))
    exit_time_stop_bars = int(getattr(config, "exit_time_stop_bars", DEFAULT_EXIT_TIME_STOP_BARS))

    if optimize_exits:
        p_rng = tuple(getattr(config, "exit_atr_period_range", DEFAULT_EXIT_ATR_PERIOD_RANGE))
        sl_rng = tuple(getattr(config, "exit_sl_atr_range", DEFAULT_EXIT_SL_ATR_RANGE))
        tp_rng = tuple(getattr(config, "exit_tp_atr_range", DEFAULT_EXIT_TP_ATR_RANGE))
        ts_rng = tuple(getattr(config, "exit_time_stop_bars_range", DEFAULT_EXIT_TIME_STOP_BARS_RANGE))

        p_min, p_max, p_step = (
            int(p_rng[0]),
            int(p_rng[1]),
            int(p_rng[2]) if len(p_rng) >= 3 else 1,
        )
        sl_min, sl_max, sl_step = (
            float(sl_rng[0]),
            float(sl_rng[1]),
            float(sl_rng[2]) if len(sl_rng) >= 3 else 0.1,
        )
        tp_min, tp_max, tp_step = (
            float(tp_rng[0]),
            float(tp_rng[1]),
            float(tp_rng[2]) if len(tp_rng) >= 3 else 0.1,
        )
        ts_min, ts_max, ts_step = (
            int(ts_rng[0]),
            int(ts_rng[1]),
            int(ts_rng[2]) if len(ts_rng) >= 3 else 1,
        )

        exit_atr_period = int(trial.suggest_int("exit_atr_period", p_min, p_max, step=max(1, p_step)))
        exit_sl_atr = float(trial.suggest_float("exit_sl_atr", sl_min, sl_max, step=sl_step))
        exit_tp_atr = float(trial.suggest_float("exit_tp_atr", tp_min, tp_max, step=tp_step))
        exit_time_stop_bars = int(
            trial.suggest_int("exit_time_stop_bars", ts_min, ts_max, step=max(1, ts_step))
        )

    # Normalización defensiva
    if exit_atr_period < 1:
        exit_atr_period = 1
    if exit_time_stop_bars < 1:
        exit_time_stop_bars = 1
    if exit_sl_atr < 0:
        exit_sl_atr = -exit_sl_atr
    if exit_tp_atr < 0:
        exit_tp_atr = -exit_tp_atr

    return ExitSettings(
        atr_period=int(exit_atr_period),
        sl_atr=float(exit_sl_atr),
        tp_atr=float(exit_tp_atr),
        time_stop_bars=int(exit_time_stop_bars),
    )


def compute_atr_wilder(
    high: "object",
    low: "object",
    close: "object",
    period: int,
) -> np.ndarray:
    """ATR de Wilder (vector) para uso en engine.

    - TR = max(H-L, |H-prevC|, |L-prevC|)
    - ATR seed: media simple de los primeros `period` TR
    - Wilder: ATR[i] = (ATR[i-1]*(period-1) + TR[i]) / period
    """
    p = int(period)
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if p <= 0 or n == 0 or n < p:
        return out

    tr = np.full(n, np.nan, dtype=np.float64)
    # i=0
    tr[0] = float(high[0]) - float(low[0])
    for i in range(1, n):
        h = float(high[i])
        l = float(low[i])
        pc = float(close[i - 1])
        a = h - l
        b = abs(h - pc)
        c = abs(l - pc)
        tr[i] = a if a >= b and a >= c else (b if b >= c else c)

    # seed SMA(TR)
    s = 0.0
    for i in range(p):
        s += tr[i]
    atr = s / p
    out[p - 1] = atr

    for i in range(p, n):
        atr = (atr * (p - 1) + tr[i]) / p
        out[i] = atr

    return out


@dataclass(frozen=True)
class IntrabarExit:
    triggered: bool
    reason: str = ""
    exit_price: float | None = None


def check_exit_sl_tp_intrabar(
    *,
    side: str,
    o: float,
    h: float,
    l: float,
    stop_loss: float | None,
    take_profit: float | None,
) -> IntrabarExit:
    """Chequea salida por SL/TP usando lógica intra-vela (sin sesgo de Close).

    Reglas (estrictas / conservadoras):
    - Orden de eventos: primero SL, luego TP.
    - LONG: SL si l <= SL; SHORT: SL si h >= SL.
    - Precio de ejecución SL:
        - LONG: si o <= SL (gap) => exit_price=o, si no => exit_price=SL
        - SHORT: si o >= SL (gap) => exit_price=o, si no => exit_price=SL
    - TP: se evalúa sólo si NO se tocó SL en esa vela.
        - Para no sobre-optimizar con gaps favorables, la ejecución de TP se fija en TP.
    """

    s = (side or "").upper()
    if s not in {"LONG", "SHORT"}:
        raise ValueError(f"side inválido: {side!r}")

    # STOP LOSS primero (peor caso)
    if stop_loss is not None:
        sl = float(stop_loss)
        if s == "LONG":
            if l <= sl:
                # Gap por debajo del SL
                exit_price = o if o <= sl else sl
                return IntrabarExit(True, "SL", exit_price)
        else:  # SHORT
            if h >= sl:
                # Gap por encima del SL
                exit_price = o if o >= sl else sl
                return IntrabarExit(True, "SL", exit_price)

    # TAKE PROFIT después
    if take_profit is not None:
        tp = float(take_profit)
        if s == "LONG":
            if h >= tp:
                return IntrabarExit(True, "TP", tp)
        else:  # SHORT
            if l <= tp:
                return IntrabarExit(True, "TP", tp)

    return IntrabarExit(False)


@dataclass(frozen=True)
class ExitSettings:
    atr_period: int = DEFAULT_EXIT_ATR_PERIOD
    sl_atr: float = DEFAULT_EXIT_SL_ATR
    tp_atr: float = DEFAULT_EXIT_TP_ATR
    time_stop_bars: int = DEFAULT_EXIT_TIME_STOP_BARS


@dataclass(frozen=True)
class ExitResult:
    exit_idx: int
    exit_price: float
    tipo_salida: str


def exit_settings_from_params(params: Dict[str, Any]) -> ExitSettings:
    """Lee settings de salida desde `params`.

    Prioridad:
      1) `__exit_*` (inyectado por runner por trial)
      2) `exit_*` (compat)
      3) defaults definidos en este archivo
    """
    atr_period = int(
        params.get("__exit_atr_period", params.get("exit_atr_period", DEFAULT_EXIT_ATR_PERIOD))
    )
    sl_atr = float(params.get("__exit_sl_atr", params.get("exit_sl_atr", DEFAULT_EXIT_SL_ATR)))
    tp_atr = float(params.get("__exit_tp_atr", params.get("exit_tp_atr", DEFAULT_EXIT_TP_ATR)))
    time_stop_bars = int(
        params.get(
            "__exit_time_stop_bars",
            params.get("exit_time_stop_bars", DEFAULT_EXIT_TIME_STOP_BARS),
        )
    )

    if atr_period < 1:
        atr_period = 1
    if time_stop_bars < 1:
        time_stop_bars = 1
    # negatives don't make sense; treat as 0
    if sl_atr < 0:
        sl_atr = -sl_atr
    if tp_atr < 0:
        tp_atr = -tp_atr

    return ExitSettings(
        atr_period=int(atr_period),
        sl_atr=float(sl_atr),
        tp_atr=float(tp_atr),
        time_stop_bars=int(time_stop_bars),
    )


def decide_exit_atr_fixed_intrabar(
    *,
    side: str,
    entry_idx: int,
    entry_price: float,
    close: "object",
    open_: "object",
    high: "object",
    low: "object",
    atr: np.ndarray,
    settings: ExitSettings,
) -> ExitResult:
    """Salida global: SL/TP por ATR fijo al inicio + TIME EXIT.

    - stop_loss y take_profit se fijan con ATR de la vela de entrada.
    - se escanea desde entry_idx+1 hasta entry_idx+time_stop_bars
    - SL tiene prioridad sobre TP en la misma vela (conservador).
    """

    s = (side or "").upper()
    if s not in {"LONG", "SHORT"}:
        raise ValueError(f"side inválido: {side!r}")

    n = len(close)
    e = int(entry_idx)
    end_idx = min(n - 1, e + int(settings.time_stop_bars))

    atr_entry = float(atr[e])
    if not np.isfinite(atr_entry) or atr_entry <= 0:
        atr_entry = max(float(high[e]) - float(low[e]), float(entry_price) * 0.001)

    sl_dist = atr_entry * max(float(settings.sl_atr), 0.0)
    tp_dist = atr_entry * max(float(settings.tp_atr), 0.0)

    if s == "LONG":
        stop_loss = float(entry_price) - sl_dist if sl_dist > 0 else None
        take_profit = float(entry_price) + tp_dist if tp_dist > 0 else None
    else:
        stop_loss = float(entry_price) + sl_dist if sl_dist > 0 else None
        take_profit = float(entry_price) - tp_dist if tp_dist > 0 else None

    # default: time exit at end_idx
    exit_idx = int(end_idx)
    exit_price = float(close[end_idx])
    tipo_salida = "TIME_EXIT"

    for j in range(e + 1, end_idx + 1):
        hit = check_exit_sl_tp_intrabar(
            side=s,
            o=float(open_[j]),
            h=float(high[j]),
            l=float(low[j]),
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        if hit.triggered:
            exit_idx = int(j)
            exit_price = float(hit.exit_price) if hit.exit_price is not None else float(close[j])
            tipo_salida = "SL_ATR" if hit.reason == "SL" else "TP_ATR"
            break

    return ExitResult(exit_idx=int(exit_idx), exit_price=float(exit_price), tipo_salida=str(tipo_salida))


def decide_exit_for_trade(
    *,
    strategy: Any,
    df: Any,
    params: Dict[str, Any],
    saldo_apertura: float,
    side: str,
    entry_idx: int,
    entry_price: float,
    close: "object",
    open_: "object",
    high: "object",
    low: "object",
    atr: np.ndarray,
    settings: ExitSettings,
) -> ExitResult:
    """Selecciona la lógica de salida:

    - Si la estrategia define `decide_exit(...)` y devuelve `ExitDecision`, se usa eso.
    - Si no, se usa la salida global (ATR fijo al inicio + SL/TP intra-vela + TIME EXIT).
    """

    decide_exit = getattr(strategy, "decide_exit", None)
    if callable(decide_exit):
        out: Optional[ExitDecision] = decide_exit(
            df,
            params,
            int(entry_idx),
            float(entry_price),
            str(side),
            saldo_apertura=float(saldo_apertura),
        )
        if isinstance(out, ExitDecision):
            idx = int(out.exit_idx)
            idx = max(0, min(idx, len(close) - 1))
            px = float(out.exit_price) if out.exit_price is not None else float(close[idx])
            tipo = str(out.reason or "STRATEGY_EXIT")
            return ExitResult(exit_idx=idx, exit_price=px, tipo_salida=tipo)

    return decide_exit_atr_fixed_intrabar(
        side=str(side),
        entry_idx=int(entry_idx),
        entry_price=float(entry_price),
        close=close,
        open_=open_,
        high=high,
        low=low,
        atr=atr,
        settings=settings,
    )

