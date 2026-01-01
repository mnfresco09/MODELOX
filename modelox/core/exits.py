from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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


@dataclass(frozen=True)
class StopTrailExit:
    triggered: bool
    exit_idx: int = -1
    reason: str = ""  # "STOP_LOSS_EMERGENCY" | "TRAILING_STOP"
    stop_level: float | None = None
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


def scan_emergency_sl_and_trailing_intrabar(
    *,
    entry_idx: int,
    entry_price: float,
    side: str,
    o: list[float] | tuple[float, ...] | "object",
    h: list[float] | tuple[float, ...] | "object",
    l: list[float] | tuple[float, ...] | "object",
    sl_pct: float | None,
    trailing_pct: float | None,
    end_idx: int,
) -> StopTrailExit:
    """Escanea desde entry_idx+1 hasta end_idx buscando SL emergencia (y opcionalmente trailing).

    - Usa High/Low para detectar toque intra-vela (sin Close).
    - Precio de ejecución: stop_level o Open si hay gap atravesándolo.
    - Conservador: aplica un único "effective_stop" (el más cercano al precio).
    """
    s = (side or "").upper()
    if s not in {"LONG", "SHORT"}:
        raise ValueError(f"side inválido: {side!r}")

    if sl_pct is None and trailing_pct is None:
        return StopTrailExit(False)

    sl_pct_v = float(sl_pct) if sl_pct is not None else 0.0
    tr_pct_v = float(trailing_pct) if trailing_pct is not None else 0.0

    if sl_pct_v <= 0 and tr_pct_v <= 0:
        return StopTrailExit(False)

    # extreme_price: máximo favorable (LONG) / mínimo favorable (SHORT)
    extreme = float(entry_price)
    sl_level = None
    if sl_pct_v > 0:
        sl_level = (
            float(entry_price) * (1.0 - sl_pct_v)
            if s == "LONG"
            else float(entry_price) * (1.0 + sl_pct_v)
        )

    start = int(entry_idx) + 1
    end = int(end_idx)
    if end < start:
        return StopTrailExit(False)

    for i in range(start, end + 1):
        oi = float(o[i])
        hi = float(h[i])
        li = float(l[i])

        if s == "LONG":
            if hi > extreme:
                extreme = hi

            trailing_level = extreme * (1.0 - tr_pct_v) if tr_pct_v > 0 else None

            if sl_level is None:
                effective = trailing_level
                reason = "TRAILING_STOP"
            elif trailing_level is None:
                effective = sl_level
                reason = "STOP_LOSS_EMERGENCY"
            else:
                # el más cercano al precio (mayor stop)
                if trailing_level > sl_level:
                    effective = trailing_level
                    reason = "TRAILING_STOP"
                else:
                    effective = sl_level
                    reason = "STOP_LOSS_EMERGENCY"

            if effective is not None and li <= effective:
                exit_price = oi if oi <= effective else float(effective)
                return StopTrailExit(
                    True,
                    exit_idx=i,
                    reason=reason,
                    stop_level=float(effective),
                    exit_price=float(exit_price),
                )

        else:  # SHORT
            if li < extreme:
                extreme = li

            trailing_level = extreme * (1.0 + tr_pct_v) if tr_pct_v > 0 else None

            if sl_level is None:
                effective = trailing_level
                reason = "TRAILING_STOP"
            elif trailing_level is None:
                effective = sl_level
                reason = "STOP_LOSS_EMERGENCY"
            else:
                # el más cercano al precio (menor stop)
                if trailing_level < sl_level:
                    effective = trailing_level
                    reason = "TRAILING_STOP"
                else:
                    effective = sl_level
                    reason = "STOP_LOSS_EMERGENCY"

            if effective is not None and hi >= effective:
                exit_price = oi if oi >= effective else float(effective)
                return StopTrailExit(
                    True,
                    exit_idx=i,
                    reason=reason,
                    stop_level=float(effective),
                    exit_price=float(exit_price),
                )

    return StopTrailExit(False)
