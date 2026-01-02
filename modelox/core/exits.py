"""modelox/core/exits.py

Sistema de Salidas basado en PNL_PCT relativo al STAKE (ROI % sobre margen usado).

TIPOS DE SALIDA DISPONIBLES:
1. "pnl_fixed": SL/TP fijos por % sobre stake
2. "pnl_trailing": SL inicial + trailing activado por % sobre stake

Definición (por trade):
- pnl_eur = (exit_price - entry_price) * qty   [LONG]
- pnl_eur = (entry_price - exit_price) * qty  [SHORT]
- PNL_PCT_STAKE = (pnl_eur / stake) * 100

Por lo tanto, los parámetros `sl_pct`, `tp_pct`, `trail_act_pct`, `trail_dist_pct` son porcentajes
SIEMPRE respecto al stake aplicado en el trade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import numpy as np

from modelox.core.types import ExitDecision


# =============================================================================
# PARÁMETROS GLOBALES DE SALIDA (PNL_PCT)
# =============================================================================

# Tipo de salida: "pnl_fixed", "pnl_trailing", o "all"
DEFAULT_EXIT_TYPE = "pnl_fixed"

# Parámetros en términos de PNL_PCT (ROI % del trade)
DEFAULT_EXIT_SL_PCT = 10.0       # Salir si PNL_PCT <= -10% (pérdida)
DEFAULT_EXIT_TP_PCT = 10.0       # Salir si PNL_PCT >= +10% (ganancia)

# Parámetros exclusivos para pnl_trailing
DEFAULT_EXIT_TRAIL_ACT_PCT = 10.0   # Activar trailing cuando PNL_PCT >= +20%
DEFAULT_EXIT_TRAIL_DIST_PCT = 5.0  # Trailing retrocede 5% desde máximo PNL

# Optimización con Optuna
DEFAULT_OPTIMIZE_EXITS = True

# Rangos de optimización Optuna (min, max, step) - en PNL_PCT
DEFAULT_EXIT_SL_PCT_RANGE = (5.0, 40.0, 2.5)      # SL: 5% a 30%
DEFAULT_EXIT_TP_PCT_RANGE = (5.0, 50.0, 2.5)     # TP: 5% a 100%
DEFAULT_EXIT_TRAIL_ACT_PCT_RANGE = (5.0, 20.0, 5.0)   # Activación: 5% a 100%
DEFAULT_EXIT_TRAIL_DIST_PCT_RANGE = (2.5, 15.0, 2.5)  # Distancia: 2.5% a 50%

# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass(frozen=True)
class ExitSettings:
    """Configuración de salida basada en PNL_PCT."""
    exit_type: str = DEFAULT_EXIT_TYPE
    sl_pct: float = DEFAULT_EXIT_SL_PCT        # Máxima pérdida % permitida
    tp_pct: float = DEFAULT_EXIT_TP_PCT        # Ganancia % objetivo
    trail_act_pct: float = DEFAULT_EXIT_TRAIL_ACT_PCT   # PNL_PCT para activar trailing
    trail_dist_pct: float = DEFAULT_EXIT_TRAIL_DIST_PCT # Retroceso % desde máximo PNL
    time_stop_bars: int = 0  # Mantener para compatibilidad (no se usa actualmente)


@dataclass(frozen=True)
class ExitResult:
    """Resultado de una salida."""
    exit_idx: int
    exit_price: float
    tipo_salida: str
    sl_distance: float = 0.0  # Distancia del SL en precio (para sizing)


@dataclass(frozen=True)
class IntrabarExit:
    """Resultado de chequeo intra-barra."""
    triggered: bool
    reason: str = ""
    exit_price: float | None = None


# =============================================================================
# FUNCIONES DE RESOLUCIÓN DE PARÁMETROS
# =============================================================================

def resolve_exit_settings_for_trial(*, trial: Any, config: Any) -> ExitSettings:
    """Resuelve los parámetros de salida para un trial de Optuna.

    Fuente de verdad:
      - defaults definidos en este archivo
      - la configuración (BacktestConfig) puede sobreescribirlos
      - si `optimize_exits` está activo, Optuna sugiere dentro de los rangos
    """
    optimize_exits = bool(getattr(config, "optimize_exits", DEFAULT_OPTIMIZE_EXITS))
    exit_type = str(getattr(config, "exit_type", DEFAULT_EXIT_TYPE)).strip().lower()

    # Valores base desde config
    sl_pct = float(getattr(config, "exit_sl_pct", DEFAULT_EXIT_SL_PCT))
    tp_pct = float(getattr(config, "exit_tp_pct", DEFAULT_EXIT_TP_PCT))
    trail_act_pct = float(getattr(config, "exit_trail_act_pct", DEFAULT_EXIT_TRAIL_ACT_PCT))
    trail_dist_pct = float(getattr(config, "exit_trail_dist_pct", DEFAULT_EXIT_TRAIL_DIST_PCT))

    if optimize_exits:
        # Rangos desde config
        sl_rng = tuple(getattr(config, "exit_sl_pct_range", DEFAULT_EXIT_SL_PCT_RANGE))
        tp_rng = tuple(getattr(config, "exit_tp_pct_range", DEFAULT_EXIT_TP_PCT_RANGE))

        sl_min, sl_max, sl_step = float(sl_rng[0]), float(sl_rng[1]), float(sl_rng[2]) if len(sl_rng) >= 3 else 0.1
        tp_min, tp_max, tp_step = float(tp_rng[0]), float(tp_rng[1]), float(tp_rng[2]) if len(tp_rng) >= 3 else 0.1

        sl_pct = float(trial.suggest_float("exit_sl_pct", sl_min, sl_max, step=sl_step))
        tp_pct = float(trial.suggest_float("exit_tp_pct", tp_min, tp_max, step=tp_step))

        # Parámetros de trailing (solo si es trailing o all)
        if exit_type in {"pnl_trailing", "percent_trailing", "all"}:
            trail_act_rng = tuple(getattr(config, "exit_trail_act_pct_range", DEFAULT_EXIT_TRAIL_ACT_PCT_RANGE))
            trail_dist_rng = tuple(getattr(config, "exit_trail_dist_pct_range", DEFAULT_EXIT_TRAIL_DIST_PCT_RANGE))

            act_min, act_max, act_step = float(trail_act_rng[0]), float(trail_act_rng[1]), float(trail_act_rng[2]) if len(trail_act_rng) >= 3 else 0.1
            dist_min, dist_max, dist_step = float(trail_dist_rng[0]), float(trail_dist_rng[1]), float(trail_dist_rng[2]) if len(trail_dist_rng) >= 3 else 0.1

            trail_act_pct = float(trial.suggest_float("exit_trail_act_pct", act_min, act_max, step=act_step))
            trail_dist_pct = float(trial.suggest_float("exit_trail_dist_pct", dist_min, dist_max, step=dist_step))

    # Normalización defensiva
    sl_pct = abs(sl_pct) if sl_pct != 0 else 1.0
    tp_pct = abs(tp_pct) if tp_pct != 0 else 1.0
    trail_act_pct = abs(trail_act_pct) if trail_act_pct != 0 else 0.5
    trail_dist_pct = abs(trail_dist_pct) if trail_dist_pct != 0 else 0.25

    return ExitSettings(
        exit_type=str(exit_type),
        sl_pct=float(sl_pct),
        tp_pct=float(tp_pct),
        trail_act_pct=float(trail_act_pct),
        trail_dist_pct=float(trail_dist_pct),
    )


def exit_settings_from_params(params: Dict[str, Any]) -> ExitSettings:
    """Lee settings de salida desde `params`.

    Prioridad:
      1) `__exit_*` (inyectado por runner por trial)
      2) `exit_*` (compat)
      3) defaults definidos en este archivo
    """
    exit_type = str(params.get("__exit_type", params.get("exit_type", DEFAULT_EXIT_TYPE))).strip().lower()
    sl_pct = float(params.get("__exit_sl_pct", params.get("exit_sl_pct", DEFAULT_EXIT_SL_PCT)))
    tp_pct = float(params.get("__exit_tp_pct", params.get("exit_tp_pct", DEFAULT_EXIT_TP_PCT)))
    trail_act_pct = float(params.get("__exit_trail_act_pct", params.get("exit_trail_act_pct", DEFAULT_EXIT_TRAIL_ACT_PCT)))
    trail_dist_pct = float(params.get("__exit_trail_dist_pct", params.get("exit_trail_dist_pct", DEFAULT_EXIT_TRAIL_DIST_PCT)))

    # Normalización defensiva
    sl_pct = abs(sl_pct) if sl_pct != 0 else 1.0
    tp_pct = abs(tp_pct) if tp_pct != 0 else 1.0
    trail_act_pct = abs(trail_act_pct) if trail_act_pct != 0 else 0.5
    trail_dist_pct = abs(trail_dist_pct) if trail_dist_pct != 0 else 0.25

    return ExitSettings(
        exit_type=str(exit_type),
        sl_pct=float(sl_pct),
        tp_pct=float(tp_pct),
        trail_act_pct=float(trail_act_pct),
        trail_dist_pct=float(trail_dist_pct),
    )


# =============================================================================
# HELPER: Calcular PNL_PCT
# =============================================================================

def calc_pnl_pct(entry_price: float, current_price: float, side: str) -> float:
    """Calcula el PNL_PCT (ROI %) de un trade.
    
    LONG:  PNL_PCT = ((current - entry) / entry) * 100
    SHORT: PNL_PCT = ((entry - current) / entry) * 100
    
    Returns: float (puede ser negativo para pérdidas)
    """
    if entry_price <= 0:
        return 0.0
    
    s = (side or "").upper()
    if s == "LONG":
        return ((current_price - entry_price) / entry_price) * 100.0
    else:  # SHORT
        return ((entry_price - current_price) / entry_price) * 100.0


# =============================================================================
# LÓGICA INTRA-BARRA (SL/TP Check por PNL_PCT)
# =============================================================================

def check_exit_pnl_intrabar(
    *,
    side: str,
    entry_price: float,
    o: float,
    h: float,
    l: float,
    sl_pct: float,
    tp_pct: float,
) -> IntrabarExit:
    """Chequea salida por SL/TP usando PNL_PCT intra-vela.

    Reglas conservadoras:
    - Primero SL (peor caso), luego TP
    - LONG: SL si low produce PNL_PCT <= -sl_pct; TP si high produce PNL_PCT >= tp_pct
    - SHORT: SL si high produce PNL_PCT <= -sl_pct; TP si low produce PNL_PCT >= tp_pct
    """
    s = (side or "").upper()
    if s not in {"LONG", "SHORT"}:
        raise ValueError(f"side inválido: {side!r}")

    if entry_price <= 0:
        return IntrabarExit(False)

    # Calcular precios de SL/TP a partir de PNL_PCT objetivo
    if s == "LONG":
        # SL: queremos salir si PNL_PCT <= -sl_pct
        # -sl_pct = ((sl_price - entry) / entry) * 100
        # sl_price = entry * (1 - sl_pct/100)
        sl_price = entry_price * (1.0 - sl_pct / 100.0)
        # TP: queremos salir si PNL_PCT >= tp_pct
        # tp_pct = ((tp_price - entry) / entry) * 100
        # tp_price = entry * (1 + tp_pct/100)
        tp_price = entry_price * (1.0 + tp_pct / 100.0)
        
        # Check SL primero (peor caso)
        if l <= sl_price:
            exit_price = o if o <= sl_price else sl_price
            return IntrabarExit(True, "SL", exit_price)
        
        # Check TP
        if h >= tp_price:
            return IntrabarExit(True, "TP", tp_price)
    
    else:  # SHORT
        # SL: queremos salir si PNL_PCT <= -sl_pct
        # -sl_pct = ((entry - sl_price) / entry) * 100
        # sl_price = entry * (1 + sl_pct/100)
        sl_price = entry_price * (1.0 + sl_pct / 100.0)
        # TP: queremos salir si PNL_PCT >= tp_pct
        # tp_pct = ((entry - tp_price) / entry) * 100
        # tp_price = entry * (1 - tp_pct/100)
        tp_price = entry_price * (1.0 - tp_pct / 100.0)
        
        # Check SL primero (peor caso)
        if h >= sl_price:
            exit_price = o if o >= sl_price else sl_price
            return IntrabarExit(True, "SL", exit_price)
        
        # Check TP
        if l <= tp_price:
            return IntrabarExit(True, "TP", tp_price)

    return IntrabarExit(False)


def check_exit_stake_intrabar(
    *,
    side: str,
    entry_price: float,
    qty: float,
    stake: float,
    o: float,
    h: float,
    l: float,
    sl_pct: float,
    tp_pct: float,
) -> IntrabarExit:
    """Chequea salida por SL/TP usando % sobre STAKE intra-vela.

    Reglas conservadoras:
    - Primero SL (peor caso), luego TP.
    """
    s = (side or "").upper()
    if s not in {"LONG", "SHORT"}:
        raise ValueError(f"side inválido: {side!r}")

    if entry_price <= 0 or qty <= 0 or stake <= 0:
        return IntrabarExit(False)

    sl_pct = abs(float(sl_pct))
    tp_pct = abs(float(tp_pct))

    q = float(qty)
    stk = float(stake)
    if s == "LONG":
        sl_price = float(entry_price) - (stk * (sl_pct / 100.0)) / q
        tp_price = float(entry_price) + (stk * (tp_pct / 100.0)) / q

        if l <= sl_price:
            exit_price = o if o <= sl_price else sl_price
            return IntrabarExit(True, "SL", float(exit_price))
        if h >= tp_price:
            return IntrabarExit(True, "TP", float(tp_price))
    else:  # SHORT
        sl_price = float(entry_price) + (stk * (sl_pct / 100.0)) / q
        tp_price = float(entry_price) - (stk * (tp_pct / 100.0)) / q

        if h >= sl_price:
            exit_price = o if o >= sl_price else sl_price
            return IntrabarExit(True, "SL", float(exit_price))
        if l <= tp_price:
            return IntrabarExit(True, "TP", float(tp_price))

    return IntrabarExit(False)


# =============================================================================
# MODO A: PNL_FIXED (SL/TP Fijos por PNL_PCT)
# =============================================================================

def decide_exit_pnl_fixed(
    *,
    side: str,
    entry_idx: int,
    entry_price: float,
    qty: float,
    stake: float,
    close: "np.ndarray",
    open_: "np.ndarray",
    high: "np.ndarray",
    low: "np.ndarray",
    settings: ExitSettings,
) -> ExitResult:
    """Salida con SL/TP fijos basados en % sobre stake.

    Ejemplo LONG con sl_pct=2, tp_pct=5, entry=100:
    - SL cuando PNL_PCT <= -2% → precio <= 98
    - TP cuando PNL_PCT >= +5% → precio >= 105
    """
    s = (side or "").upper()
    if s not in {"LONG", "SHORT"}:
        raise ValueError(f"side inválido: {side!r}")

    n = len(close)
    e = int(entry_idx)
    end_idx = n - 1

    # Para sizing/auditoría: distancia física del SL en precio (derivada de stake)
    if float(qty) > 0 and float(stake) > 0:
        sl_distance = (float(stake) * (float(settings.sl_pct) / 100.0)) / float(qty)
    else:
        sl_distance = float(entry_price) * 0.01
    if sl_distance <= 0:
        sl_distance = float(entry_price) * 0.01

    # Escanear velas buscando SL/TP por PNL_PCT
    for j in range(e + 1, end_idx + 1):
        hit = check_exit_stake_intrabar(
            side=s,
            entry_price=float(entry_price),
            qty=float(qty),
            stake=float(stake),
            o=float(open_[j]),
            h=float(high[j]),
            l=float(low[j]),
            sl_pct=float(settings.sl_pct),
            tp_pct=float(settings.tp_pct),
        )
        if hit.triggered:
            exit_price = float(hit.exit_price) if hit.exit_price is not None else float(close[j])
            return ExitResult(
                exit_idx=int(j),
                exit_price=exit_price,
                tipo_salida=str(hit.reason),
                sl_distance=sl_distance,
            )

    # Sin salida - cerrar al final
    return ExitResult(
        exit_idx=int(end_idx),
        exit_price=float(close[end_idx]),
        tipo_salida="END_OF_DATA",
        sl_distance=sl_distance,
    )


# =============================================================================
# MODO B: PNL_TRAILING (Trailing Stop por PNL_PCT)
# =============================================================================

def decide_exit_pnl_trailing(
    *,
    side: str,
    entry_idx: int,
    entry_price: float,
    qty: float,
    stake: float,
    close: "np.ndarray",
    open_: "np.ndarray",
    high: "np.ndarray",
    low: "np.ndarray",
    settings: ExitSettings,
) -> ExitResult:
    """Salida con trailing stop activado por PNL_PCT.

    Lógica:
    1. SL inicial fijo: salir si PNL_PCT <= -sl_pct
    2. Cuando PNL_PCT >= trail_act_pct, activar trailing
    3. Una vez activo, el trailing protege desde el máximo PNL alcanzado:
       - Salir si PNL_PCT actual <= (max_pnl - trail_dist_pct)

    El trailing NUNCA retrocede el umbral - solo sube para proteger más.
    
    Ejemplo:
    - trail_act_pct = 1%, trail_dist_pct = 0.5%
    - Cuando PNL_PCT alcanza +1%, se activa el trailing
    - Si PNL_PCT sube a +3%, el nuevo umbral de salida es +2.5%
    - Si PNL_PCT baja a +2.5% o menos, salimos con TRAILING_STOP
    """
    s = (side or "").upper()
    if s not in {"LONG", "SHORT"}:
        raise ValueError(f"side inválido: {side!r}")

    n = len(close)
    e = int(entry_idx)
    end_idx = n - 1

    # Niveles basados en % sobre stake
    q = float(qty)
    stk = float(stake)
    if q <= 0 or stk <= 0 or float(entry_price) <= 0:
        return ExitResult(
            exit_idx=int(end_idx),
            exit_price=float(close[end_idx]),
            tipo_salida="END_OF_DATA",
            sl_distance=float(entry_price) * 0.01,
        )

    sl_eur = stk * (abs(float(settings.sl_pct)) / 100.0)
    tp_eur = stk * (abs(float(settings.tp_pct)) / 100.0)
    act_eur = stk * (abs(float(settings.trail_act_pct)) / 100.0)
    dist_eur = stk * (abs(float(settings.trail_dist_pct)) / 100.0)

    sl_distance = (sl_eur / q) if q > 0 else float(entry_price) * 0.01
    if sl_distance <= 0:
        sl_distance = float(entry_price) * 0.01

    if s == "LONG":
        sl_price = float(entry_price) - (sl_eur / q)
        tp_price = float(entry_price) + (tp_eur / q)
        trail_act_price = float(entry_price) + (act_eur / q)
        trail_dist_price = (dist_eur / q)
    else:  # SHORT
        sl_price = float(entry_price) + (sl_eur / q)
        tp_price = float(entry_price) - (tp_eur / q)
        trail_act_price = float(entry_price) - (act_eur / q)
        trail_dist_price = (dist_eur / q)

    # Estado del trailing
    is_trailing_active = False
    max_fav_price = float(entry_price)  # LONG: máximo high; SHORT: mínimo low

    # Escanear velas
    for j in range(e + 1, end_idx + 1):
        o_j = float(open_[j])
        h_j = float(high[j])
        l_j = float(low[j])
        c_j = float(close[j])

        if s == "LONG":
            # 1) SL conservador (peor caso)
            if l_j <= sl_price:
                exit_price = o_j if o_j <= sl_price else sl_price
                return ExitResult(int(j), float(exit_price), "SL", sl_distance=sl_distance)

            # 2) Actualizar favorable y activar trailing
            trailing_active_before = bool(is_trailing_active)
            if h_j > max_fav_price:
                max_fav_price = h_j
            if (not is_trailing_active) and (max_fav_price >= trail_act_price):
                is_trailing_active = True

            # 3) Trailing (distancia fija desde entry, aplicada sobre el máximo favorable)
            if is_trailing_active:
                trailing_price = float(max_fav_price) - float(trail_dist_price)
                if l_j <= trailing_price:
                    # Si el trailing YA estaba activo al abrir la vela, permitimos gap-fill al open.
                    # Si se activó dentro de esta vela, no podemos ejecutar antes de que exista,
                    # así que llenamos al nivel del stop (sin slippage).
                    if trailing_active_before:
                        exit_price = o_j if o_j <= trailing_price else trailing_price
                    else:
                        exit_price = trailing_price
                    return ExitResult(int(j), float(exit_price), "TRAILING_STOP", sl_distance=sl_distance)

            # 4) TP fijo (como SL)
            if h_j >= tp_price:
                return ExitResult(int(j), float(tp_price), "TP", sl_distance=sl_distance)

        else:  # SHORT
            # 1) SL conservador (peor caso)
            if h_j >= sl_price:
                exit_price = o_j if o_j >= sl_price else sl_price
                return ExitResult(int(j), float(exit_price), "SL", sl_distance=sl_distance)

            # 2) Actualizar favorable y activar trailing
            trailing_active_before = bool(is_trailing_active)
            if l_j < max_fav_price:
                max_fav_price = l_j
            if (not is_trailing_active) and (max_fav_price <= trail_act_price):
                is_trailing_active = True

            # 3) Trailing (distancia fija desde entry, aplicada sobre el mínimo favorable)
            if is_trailing_active:
                trailing_price = float(max_fav_price) + float(trail_dist_price)
                if h_j >= trailing_price:
                    if trailing_active_before:
                        exit_price = o_j if o_j >= trailing_price else trailing_price
                    else:
                        exit_price = trailing_price
                    return ExitResult(int(j), float(exit_price), "TRAILING_STOP", sl_distance=sl_distance)

            # 4) TP fijo (como SL)
            if l_j <= tp_price:
                return ExitResult(int(j), float(tp_price), "TP", sl_distance=sl_distance)

    # Sin salida - cerrar al final
    return ExitResult(
        exit_idx=int(end_idx),
        exit_price=float(close[end_idx]),
        tipo_salida="END_OF_DATA",
        sl_distance=sl_distance,
    )


def trace_exit_pnl_trailing(
    *,
    side: str,
    entry_idx: int,
    entry_price: float,
    qty: float,
    stake: float,
    timestamps: "np.ndarray",
    close: "np.ndarray",
    open_: "np.ndarray",
    high: "np.ndarray",
    low: "np.ndarray",
    settings: ExitSettings,
) -> List[Dict[str, Any]]:
    """Traza bar-a-bar del trailing (modo stake-based).

    Devuelve una lista de filas con el estado interno por vela desde entry+1 hasta exit.
    Está pensada para debugging (consistencia 1:1 con `decide_exit_pnl_trailing`).
    """
    s = (side or "").upper()
    if s not in {"LONG", "SHORT"}:
        raise ValueError(f"side inválido: {side!r}")

    n = len(close)
    e = int(entry_idx)
    end_idx = n - 1

    q = float(qty)
    stk = float(stake)
    ep = float(entry_price)
    if q <= 0 or stk <= 0 or ep <= 0:
        return []

    sl_eur = stk * (abs(float(settings.sl_pct)) / 100.0)
    tp_eur = stk * (abs(float(settings.tp_pct)) / 100.0)
    act_eur = stk * (abs(float(settings.trail_act_pct)) / 100.0)
    dist_eur = stk * (abs(float(settings.trail_dist_pct)) / 100.0)

    if s == "LONG":
        sl_price = ep - (sl_eur / q)
        tp_price = ep + (tp_eur / q)
        trail_act_price = ep + (act_eur / q)
        trail_dist_price = (dist_eur / q)
    else:  # SHORT
        sl_price = ep + (sl_eur / q)
        tp_price = ep - (tp_eur / q)
        trail_act_price = ep - (act_eur / q)
        trail_dist_price = (dist_eur / q)

    is_trailing_active = False
    max_fav_price = ep

    rows: List[Dict[str, Any]] = []

    for j in range(e + 1, end_idx + 1):
        o_j = float(open_[j])
        h_j = float(high[j])
        l_j = float(low[j])
        c_j = float(close[j])

        max_fav_before = float(max_fav_price)
        trailing_active_before = bool(is_trailing_active)
        trailing_price = None
        hit_reason = None
        hit_exit_price = None

        # --- 1) SL (peor caso), igual que decide_exit_pnl_trailing ---
        if s == "LONG":
            if l_j <= sl_price:
                hit_reason = "SL"
                hit_exit_price = o_j if o_j <= sl_price else float(sl_price)
        else:
            if h_j >= sl_price:
                hit_reason = "SL"
                hit_exit_price = o_j if o_j >= sl_price else float(sl_price)

        # Si SL se dispara, no actualizamos trailing (consistente con lógica conservadora)
        if hit_reason is None:
            # --- 2) Actualizar favorable y activar trailing ---
            trailing_active_before_eval = bool(is_trailing_active)
            if s == "LONG":
                if h_j > max_fav_price:
                    max_fav_price = h_j
                if (not is_trailing_active) and (max_fav_price >= trail_act_price):
                    is_trailing_active = True
            else:
                if l_j < max_fav_price:
                    max_fav_price = l_j
                if (not is_trailing_active) and (max_fav_price <= trail_act_price):
                    is_trailing_active = True

            # --- 3) Trailing ---
            if is_trailing_active:
                if s == "LONG":
                    trailing_price = float(max_fav_price) - float(trail_dist_price)
                    if l_j <= trailing_price:
                        hit_reason = "TRAILING_STOP"
                        if trailing_active_before_eval:
                            hit_exit_price = o_j if o_j <= trailing_price else float(trailing_price)
                        else:
                            hit_exit_price = float(trailing_price)
                else:
                    trailing_price = float(max_fav_price) + float(trail_dist_price)
                    if h_j >= trailing_price:
                        hit_reason = "TRAILING_STOP"
                        if trailing_active_before_eval:
                            hit_exit_price = o_j if o_j >= trailing_price else float(trailing_price)
                        else:
                            hit_exit_price = float(trailing_price)

            # --- 4) TP fijo ---
            if hit_reason is None:
                if s == "LONG":
                    if h_j >= tp_price:
                        hit_reason = "TP"
                        hit_exit_price = float(tp_price)
                else:
                    if l_j <= tp_price:
                        hit_reason = "TP"
                        hit_exit_price = float(tp_price)

        # PnL informativo a cierre (no afecta decisiones)
        if s == "LONG":
            pnl_eur_close = (c_j - ep) * q
        else:
            pnl_eur_close = (ep - c_j) * q
        pnl_pct_stake_close = (pnl_eur_close / stk) * 100.0 if stk > 0 else 0.0

        rows.append(
            {
                "bar_idx": int(j),
                "timestamp": timestamps[j] if timestamps is not None else None,
                "open": o_j,
                "high": h_j,
                "low": l_j,
                "close": c_j,
                "side": s,
                "entry_idx": int(e),
                "entry_price": ep,
                "qty": q,
                "stake": stk,
                "sl_pct": float(settings.sl_pct),
                "tp_pct": float(settings.tp_pct),
                "trail_act_pct": float(settings.trail_act_pct),
                "trail_dist_pct": float(settings.trail_dist_pct),
                "sl_price": float(sl_price),
                "tp_price": float(tp_price),
                "trail_act_price": float(trail_act_price),
                "trail_dist_price": float(trail_dist_price),
                "max_fav_before": max_fav_before,
                "max_fav_after": float(max_fav_price),
                "trailing_active_before": trailing_active_before,
                "trailing_active_after": bool(is_trailing_active),
                "trailing_price": trailing_price,
                "pnl_eur_close": float(pnl_eur_close),
                "pnl_pct_stake_close": float(pnl_pct_stake_close),
                "hit_reason": hit_reason,
                "hit_exit_price": hit_exit_price,
            }
        )

        if hit_reason is not None:
            break

    return rows


# =============================================================================
# DISPATCHER PRINCIPAL
# =============================================================================

def decide_exit_for_trade(
    *,
    strategy: Any,
    df: Any,
    params: Dict[str, Any],
    saldo_apertura: float,
    side: str,
    entry_idx: int,
    entry_price: float,
    qty: float,
    stake: float,
    close: "np.ndarray",
    open_: "np.ndarray",
    high: "np.ndarray",
    low: "np.ndarray",
    settings: ExitSettings,
    **kwargs: Any,
) -> ExitResult:
    """Selecciona la lógica de salida según exit_type.

    - Si la estrategia define `decide_exit(...)`, se usa eso.
    - Si no, se usa la salida global basada en PNL_PCT:
        * "pnl_fixed": SL/TP fijos por PNL_PCT
        * "pnl_trailing": Trailing activado por PNL_PCT
    """
    # Permitir override por estrategia
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
            if float(qty) > 0 and float(stake) > 0:
                sl_dist = (float(stake) * (float(settings.sl_pct) / 100.0)) / float(qty)
            else:
                sl_dist = float(entry_price) * (float(settings.sl_pct) / 100.0)
            return ExitResult(exit_idx=idx, exit_price=px, tipo_salida=tipo, sl_distance=sl_dist)

    # Seleccionar según exit_type
    exit_type = str(settings.exit_type).strip().lower()

    if exit_type == "pnl_trailing":
        return decide_exit_pnl_trailing(
            side=str(side),
            entry_idx=int(entry_idx),
            entry_price=float(entry_price),
            qty=float(qty),
            stake=float(stake),
            close=close,
            open_=open_,
            high=high,
            low=low,
            settings=settings,
        )
    else:  # "pnl_fixed" (default)
        return decide_exit_pnl_fixed(
            side=str(side),
            entry_idx=int(entry_idx),
            entry_price=float(entry_price),
            qty=float(qty),
            stake=float(stake),
            close=close,
            open_=open_,
            high=high,
            low=low,
            settings=settings,
        )

