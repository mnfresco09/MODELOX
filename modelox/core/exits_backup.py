from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from modelox.core.types import ExitDecision


# =============================================================================
# PARÁMETROS GLOBALES DE SALIDA (ÚNICO LUGAR)
# =============================================================================
#
# TIPOS DE SALIDA DISPONIBLES:
# 1. "atr_fixed": SL/TP por ATR fijo al inicio (sin time exit)
# 2. "trailing": Trailing stop + SL emergencia (sin time exit)
#
# Los valores pueden ser:
# - fijos por configuración
# - optimizados por Optuna (si optimize_exits/OPTIMIZAR_SALIDAS=True)
#
# En runtime, el runner inyecta las claves con prefijo `__exit_*` en `params`.
#
# NOTA: TIME_EXIT eliminado - los trades se mantienen hasta SL/TP/trailing
# =============================================================================

# Tipo de salida: "atr_fixed" o "trailing" o "all"
DEFAULT_EXIT_TYPE = "all"

# Parámetros comunes
DEFAULT_EXIT_ATR_PERIOD = 14
DEFAULT_EXIT_TIME_STOP_BARS = 0  # Ya no se usa, mantener para compatibilidad
DEFAULT_OPTIMIZE_EXITS = True

# Parámetros para EXIT_TYPE = "atr_fixed"
DEFAULT_EXIT_SL_ATR = 1.0
DEFAULT_EXIT_TP_ATR = 1.0

# Parámetros para EXIT_TYPE = "trailing"
DEFAULT_EXIT_TRAILING_ATR_MULT = 2.0
DEFAULT_EXIT_EMERGENCY_SL_ATR_MULT = 4.0

# Rangos de optimización Optuna
DEFAULT_EXIT_ATR_PERIOD_RANGE = (7, 30, 1)
DEFAULT_EXIT_TIME_STOP_BARS_RANGE = (250, 800, 10)

# Rangos para "atr_fixed"
DEFAULT_EXIT_SL_ATR_RANGE = (0.5, 5.0, 0.1)
DEFAULT_EXIT_TP_ATR_RANGE = (1.0, 8.0, 0.1)

# Rangos para "trailing"
DEFAULT_EXIT_TRAILING_ATR_MULT_RANGE = (1.0, 4.0, 0.1)
DEFAULT_EXIT_EMERGENCY_SL_ATR_MULT_RANGE = (1.5, 6.0, 0.1)


def resolve_exit_settings_for_trial(*, trial: Any, config: Any) -> ExitSettings:
    """Resuelve los parámetros de salida para un trial.

    Fuente de verdad:
      - defaults/rangos definidos en este archivo
      - la configuración (BacktestConfig) puede sobreescribirlos
      - si `optimize_exits` está activo, Optuna sugiere dentro de los rangos
    """

    optimize_exits = bool(getattr(config, "optimize_exits", DEFAULT_OPTIMIZE_EXITS))
    # Normalizar defensivamente: evita bugs por espacios/uppercase (p.ej. "trailing ")
    exit_type = str(getattr(config, "exit_type", DEFAULT_EXIT_TYPE)).strip().lower()

    # Parámetros comunes
    exit_atr_period = int(getattr(config, "exit_atr_period", DEFAULT_EXIT_ATR_PERIOD))

    # Parámetros específicos según exit_type
    if exit_type == "atr_fixed":
        exit_sl_atr = float(getattr(config, "exit_sl_atr", DEFAULT_EXIT_SL_ATR))
        exit_tp_atr = float(getattr(config, "exit_tp_atr", DEFAULT_EXIT_TP_ATR))
        trailing_atr_mult = 0.0
        emergency_sl_atr_mult = 0.0
    else:  # trailing (default)
        exit_sl_atr = 0.0
        exit_tp_atr = 0.0
        trailing_atr_mult = float(getattr(config, "exit_trailing_atr_mult", DEFAULT_EXIT_TRAILING_ATR_MULT))
        emergency_sl_atr_mult = float(getattr(config, "exit_emergency_sl_atr_mult", DEFAULT_EXIT_EMERGENCY_SL_ATR_MULT))

    if optimize_exits:
        # Rangos comunes
        p_rng = tuple(getattr(config, "exit_atr_period_range", DEFAULT_EXIT_ATR_PERIOD_RANGE))

        p_min, p_max, p_step = (
            int(p_rng[0]),
            int(p_rng[1]),
            int(p_rng[2]) if len(p_rng) >= 3 else 1,
        )

        exit_atr_period = int(trial.suggest_int("exit_atr_period", p_min, p_max, step=max(1, p_step)))

        # Rangos específicos según exit_type
        if exit_type == "atr_fixed":
            sl_rng = tuple(getattr(config, "exit_sl_atr_range", DEFAULT_EXIT_SL_ATR_RANGE))
            tp_rng = tuple(getattr(config, "exit_tp_atr_range", DEFAULT_EXIT_TP_ATR_RANGE))

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

            exit_sl_atr = float(trial.suggest_float("exit_sl_atr", sl_min, sl_max, step=sl_step))
            exit_tp_atr = float(trial.suggest_float("exit_tp_atr", tp_min, tp_max, step=tp_step))
        else:  # trailing
            trail_rng = tuple(getattr(config, "exit_trailing_atr_mult_range", DEFAULT_EXIT_TRAILING_ATR_MULT_RANGE))
            emerg_rng = tuple(getattr(config, "exit_emergency_sl_atr_mult_range", DEFAULT_EXIT_EMERGENCY_SL_ATR_MULT_RANGE))

            trail_min, trail_max, trail_step = (
                float(trail_rng[0]),
                float(trail_rng[1]),
                float(trail_rng[2]) if len(trail_rng) >= 3 else 0.1,
            )
            emerg_min, emerg_max, emerg_step = (
                float(emerg_rng[0]),
                float(emerg_rng[1]),
                float(emerg_rng[2]) if len(emerg_rng) >= 3 else 0.1,
            )

            trailing_atr_mult = float(trial.suggest_float("exit_trailing_atr_mult", trail_min, trail_max, step=trail_step))
            emergency_sl_atr_mult = float(trial.suggest_float("exit_emergency_sl_atr_mult", emerg_min, emerg_max, step=emerg_step))

    # Normalización defensiva
    if exit_atr_period < 1:
        exit_atr_period = 1
    if exit_sl_atr < 0:
        exit_sl_atr = -exit_sl_atr
    if exit_tp_atr < 0:
        exit_tp_atr = -exit_tp_atr
    if trailing_atr_mult < 0:
        trailing_atr_mult = -trailing_atr_mult
    if emergency_sl_atr_mult < 0:
        emergency_sl_atr_mult = -emergency_sl_atr_mult

    return ExitSettings(
        exit_type=str(exit_type),
        atr_period=int(exit_atr_period),
        sl_atr=float(exit_sl_atr),
        tp_atr=float(exit_tp_atr),
        time_stop_bars=0,  # Ya no se usa, mantener por compatibilidad
        trailing_atr_mult=float(trailing_atr_mult),
        emergency_sl_atr_mult=float(emergency_sl_atr_mult),
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
    exit_type: str = DEFAULT_EXIT_TYPE
    atr_period: int = DEFAULT_EXIT_ATR_PERIOD
    sl_atr: float = DEFAULT_EXIT_SL_ATR
    tp_atr: float = DEFAULT_EXIT_TP_ATR
    time_stop_bars: int = DEFAULT_EXIT_TIME_STOP_BARS
    trailing_atr_mult: float = DEFAULT_EXIT_TRAILING_ATR_MULT
    emergency_sl_atr_mult: float = DEFAULT_EXIT_EMERGENCY_SL_ATR_MULT


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
    exit_type = str(params.get("__exit_type", params.get("exit_type", DEFAULT_EXIT_TYPE))).strip().lower()
    atr_period = int(
        params.get("__exit_atr_period", params.get("exit_atr_period", DEFAULT_EXIT_ATR_PERIOD))
    )
    sl_atr = float(params.get("__exit_sl_atr", params.get("exit_sl_atr", DEFAULT_EXIT_SL_ATR)))
    tp_atr = float(params.get("__exit_tp_atr", params.get("exit_tp_atr", DEFAULT_EXIT_TP_ATR)))
    trailing_atr_mult = float(
        params.get("__exit_trailing_atr_mult", params.get("exit_trailing_atr_mult", DEFAULT_EXIT_TRAILING_ATR_MULT))
    )
    emergency_sl_atr_mult = float(
        params.get("__exit_emergency_sl_atr_mult", params.get("exit_emergency_sl_atr_mult", DEFAULT_EXIT_EMERGENCY_SL_ATR_MULT))
    )

    if atr_period < 1:
        atr_period = 1
    # negatives don't make sense; treat as 0
    if sl_atr < 0:
        sl_atr = -sl_atr
    if tp_atr < 0:
        tp_atr = -tp_atr
    if trailing_atr_mult < 0:
        trailing_atr_mult = -trailing_atr_mult
    if emergency_sl_atr_mult < 0:
        emergency_sl_atr_mult = -emergency_sl_atr_mult

    return ExitSettings(
        exit_type=str(exit_type),
        atr_period=int(atr_period),
        sl_atr=float(sl_atr),
        tp_atr=float(tp_atr),
        time_stop_bars=0,  # Ya no se usa
        trailing_atr_mult=float(trailing_atr_mult),
        emergency_sl_atr_mult=float(emergency_sl_atr_mult),
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
    """Salida global: SL/TP por ATR fijo al inicio (SIN TIME EXIT).

    IMPORTANTE: SL/TP SON FIJOS
    - stop_loss y take_profit se calculan UNA SOLA VEZ con el ATR de la vela de entrada
    - Estos valores NO se modifican durante todo el trade
    - Se escanea desde entry_idx+1 hasta el final del DataFrame
    - SL tiene prioridad sobre TP en la misma vela (conservador)
    - El trade se mantiene hasta que se toque SL o TP
    
    Ejemplo LONG con entry_price=100, ATR[entry]=2.0, sl_atr=1.5, tp_atr=3.0:
      - SL fijo = 100 - (2.0 * 1.5) = 97.0  (nunca cambia)
      - TP fijo = 100 + (2.0 * 3.0) = 106.0 (nunca cambia)
    """

    s = (side or "").upper()
    if s not in {"LONG", "SHORT"}:
        raise ValueError(f"side inválido: {side!r}")

    n = len(close)
    e = int(entry_idx)
    
    # Escanear hasta el final del DataFrame (sin time exit)
    end_idx = n - 1

    # CALCULAR ATR DE LA VELA DE ENTRADA (una sola vez)
    atr_entry = float(atr[e])
    if not np.isfinite(atr_entry) or atr_entry <= 0:
        atr_entry = max(float(high[e]) - float(low[e]), float(entry_price) * 0.001)

    # CALCULAR SL/TP FIJOS (se establecen aquí y nunca se modifican)
    sl_dist = atr_entry * max(float(settings.sl_atr), 0.0)
    tp_dist = atr_entry * max(float(settings.tp_atr), 0.0)

    if s == "LONG":
        stop_loss = float(entry_price) - sl_dist if sl_dist > 0 else None
        take_profit = float(entry_price) + tp_dist if tp_dist > 0 else None
    else:
        stop_loss = float(entry_price) + sl_dist if sl_dist > 0 else None
        take_profit = float(entry_price) - tp_dist if tp_dist > 0 else None

    # Escanear velas buscando SL/TP
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
            tipo_salida = "SL" if hit.reason == "SL" else "TP"
            return ExitResult(exit_idx=int(exit_idx), exit_price=float(exit_price), tipo_salida=str(tipo_salida))

    # Si llegamos aquí, no se tocó ni SL ni TP (raro pero posible en datos cortos)
    # Salir en la última vela disponible
    exit_idx = int(end_idx)
    exit_price = float(close[end_idx])
    tipo_salida = "END_OF_DATA"
    
    return ExitResult(exit_idx=int(exit_idx), exit_price=float(exit_price), tipo_salida=str(tipo_salida))


def decide_exit_atr_trailing_with_emergency_sl(
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
    trailing_atr_mult: float = 2.0,
    emergency_sl_atr_mult: float = 4.0,
) -> ExitResult:
    """Salida con trailing stop TRADICIONAL + SL de emergencia (SIN TIME EXIT).
    
    Lógica TRADICIONAL (trailing SOLO se mueve en dirección favorable):
    
    - SL emergencia: FIJO desde entrada (entry_price +/- emergency_sl_atr_mult * ATR_ENTRY)
      * LONG: emergency_sl = entry_price - (emergency_sl_atr_mult * ATR_entry)
      * SHORT: emergency_sl = entry_price + (emergency_sl_atr_mult * ATR_entry)
      * Este valor NUNCA cambia durante el trade (protección catastrófica)
    
    - Trailing stop: Se actualiza SOLO cuando el precio se mueve A FAVOR
      * LONG: trailing_stop = max(trailing_stop_prev, close[i] - trailing_atr_mult * ATR[i])
        - SOLO SUBE cuando precio sube (nunca baja)
        - Protege ganancias progresivamente
      * SHORT: trailing_stop = min(trailing_stop_prev, close[i] + trailing_atr_mult * ATR[i])
        - SOLO BAJA cuando precio baja (nunca sube)
        - Protege ganancias progresivamente
    
    - Prioridad de salidas (intra-barra):
      1. SL emergencia (fijo - protección catastrófica)
      2. Trailing stop (dinámico - protección de ganancias)
    
    - El trade se mantiene hasta que se toque alguna salida
    
    VENTAJAS:
    - Protege ganancias sin retroceder
    - Se adapta a volatilidad cambiante (ATR dinámico)
    - SL emergencia evita pérdidas catastróficas
    """
    s = (side or "").upper()
    if s not in {"LONG", "SHORT"}:
        raise ValueError(f"side inválido: {side!r}")

    n = len(close)
    e = int(entry_idx)
    end_idx = n - 1  # Escanear hasta el final (sin time exit)

    # ATR de la vela de entrada
    atr_entry = float(atr[e])
    if not np.isfinite(atr_entry) or atr_entry <= 0:
        atr_entry = max(float(high[e]) - float(low[e]), float(entry_price) * 0.001)

    # SL de emergencia FIJO (basado en ATR de entrada)
    emergency_sl_dist = atr_entry * float(emergency_sl_atr_mult)
    if s == "LONG":
        emergency_sl = float(entry_price) - emergency_sl_dist
    else:
        emergency_sl = float(entry_price) + emergency_sl_dist

    # Trailing stop inicial: DEBE empezar MÁS ALEJADO que el emergency SL
    # para no ser tocado inmediatamente. El trailing se irá actualizando
    # conforme el precio se mueva favorablemente.
    trailing_atr_m = float(trailing_atr_mult)
    if s == "LONG":
        # Iniciar por debajo del emergency SL (más protección inicial)
        trailing_stop = emergency_sl - atr_entry
    else:
        # Iniciar por encima del emergency SL (más protección inicial)
        trailing_stop = emergency_sl + atr_entry

    # Escanear velas buscando salidas
    for j in range(e + 1, end_idx + 1):
        atr_j = float(atr[j])
        if not np.isfinite(atr_j) or atr_j <= 0:
            atr_j = max(float(high[j]) - float(low[j]), float(close[j]) * 0.001)

        # Actualizar trailing stop SOLO EN DIRECCIÓN FAVORABLE (tradicional)
        # LONG: trailing SOLO SUBE (nunca baja) - protege ganancias
        # SHORT: trailing SOLO BAJA (nunca sube) - protege ganancias
        if s == "LONG":
            new_trailing = float(close[j]) - trailing_atr_m * atr_j
            # Solo actualizar si el nuevo trailing es MÁS ALTO (protege más ganancia)
            if new_trailing > trailing_stop:
                trailing_stop = new_trailing
        else:  # SHORT
            new_trailing = float(close[j]) + trailing_atr_m * atr_j
            # Solo actualizar si el nuevo trailing es MÁS BAJO (protege más ganancia)
            if new_trailing < trailing_stop:
                trailing_stop = new_trailing

        # Chequear salidas intra-barra
        o_j = float(open_[j])
        h_j = float(high[j])
        l_j = float(low[j])

        # Prioridad: SL emergencia primero (peor caso)
        if s == "LONG":
            if l_j <= emergency_sl:
                # Gap por debajo del SL emergencia
                exit_price = o_j if o_j <= emergency_sl else emergency_sl
                exit_idx = int(j)
                tipo_salida = "SL_EMERGENCY"
                return ExitResult(exit_idx=exit_idx, exit_price=exit_price, tipo_salida=tipo_salida)
            
            # Trailing stop después
            if l_j <= trailing_stop:
                # Gap por debajo del trailing
                exit_price = o_j if o_j <= trailing_stop else trailing_stop
                exit_idx = int(j)
                tipo_salida = "TRAILING_STOP"
                return ExitResult(exit_idx=exit_idx, exit_price=exit_price, tipo_salida=tipo_salida)
        
        else:  # SHORT
            if h_j >= emergency_sl:
                # Gap por encima del SL emergencia
                exit_price = o_j if o_j >= emergency_sl else emergency_sl
                exit_idx = int(j)
                tipo_salida = "SL_EMERGENCY"
                return ExitResult(exit_idx=exit_idx, exit_price=exit_price, tipo_salida=tipo_salida)
            
            # Trailing stop después
            if h_j >= trailing_stop:
                # Gap por encima del trailing
                exit_price = o_j if o_j >= trailing_stop else trailing_stop
                exit_idx = int(j)
                tipo_salida = "TRAILING_STOP"
                return ExitResult(exit_idx=exit_idx, exit_price=exit_price, tipo_salida=tipo_salida)

    # Si llegamos aquí, no se tocó ninguna salida (raro pero posible en datos cortos)
    exit_idx = int(end_idx)
    exit_price = float(close[end_idx])
    tipo_salida = "END_OF_DATA"
    
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
    - Si no, se usa la salida global según exit_type:
        * "atr_fixed": SL/TP por ATR fijo (sin time exit)
        * "trailing": Trailing stop + SL emergencia (sin time exit)
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

    # Seleccionar según exit_type
    exit_type = str(settings.exit_type).strip().lower()
    
    if exit_type == "trailing":
        return decide_exit_atr_trailing_with_emergency_sl(
            side=str(side),
            entry_idx=int(entry_idx),
            entry_price=float(entry_price),
            close=close,
            open_=open_,
            high=high,
            low=low,
            atr=atr,
            settings=settings,
            trailing_atr_mult=float(settings.trailing_atr_mult),
            emergency_sl_atr_mult=float(settings.emergency_sl_atr_mult),
        )
    else:  # "atr_fixed" (default)
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

