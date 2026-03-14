"""modelox.core.exits — Fuente única de verdad para parámetros de salida (SL/TP/Trailing/ATR).

Los % son siempre sobre stake (colateral), no sobre precio.
Las salidas se evalúan en resolución de 1m independientemente del TF de entrada.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


# =============================================================================
# 1. DEFAULTS — FUENTE ÚNICA DE VERDAD
# =============================================================================

# ─── TIPO DE SALIDA ──────────────────────────────────────────────────────────
DEFAULT_EXIT_TYPE = "TRAILING"  # FIXED | TRAILING | BARS | ATR

# ─── PARÁMETROS DE SL/TP (% SOBRE STAKE) ────────────────────────────────────
DEFAULT_EXIT_SL_PCT = 30.0                    # Stop Loss
DEFAULT_EXIT_TP_PCT = 30.0                   # Take Profit

# ─── PARÁMETROS DE TRAILING (% SOBRE STAKE) ─────────────────────────────────
DEFAULT_EXIT_TRAIL_ACT_PCT = 30.0            # Activación del trailing
DEFAULT_EXIT_TRAIL_DIST_PCT = 6.0    

DEFAULT_EXIT_TIME_BARS = 20                   # 0 = desactivado

# ─── PARÁMETROS ATR ADAPTATIVO ───────────────────────────────────────────────
DEFAULT_EXIT_ATR_PERIOD   = 14    # Periodo Wilder ATR
DEFAULT_EXIT_ATR_MIN_PCT  = 15.0  # % stake mínimo (volatilidad baja)  — mismo formato que sl_pct
DEFAULT_EXIT_ATR_MAX_PCT  = 37.0  # % stake máximo (volatilidad alta)  — mismo formato que sl_pct
DEFAULT_EXIT_ATR_LOOKBACK = 100   # Ventana (barras) para normalizar volatilidad relativa


# =============================================================================
# 2. RANGOS DE OPTIMIZACIÓN (MIN, MAX, STEP)
# =============================================================================

DEFAULT_OPTIMIZE_EXITS = False

DEFAULT_EXIT_SL_PCT_RANGE = (10.0, 36.0, 2.0)
DEFAULT_EXIT_TP_PCT_RANGE = (10.0, 36.0, 2.0)
DEFAULT_EXIT_TRAIL_ACT_PCT_RANGE = (10.0, 36.0, 2.0)
DEFAULT_EXIT_TRAIL_DIST_PCT_RANGE = (2.0, 10.0, 1.0)

DEFAULT_EXIT_TIME_BARS_RANGE = (8, 8, 6)


# =============================================================================
# 3. DATACLASSES
# =============================================================================

@dataclass(frozen=True)
class ExitSettings:
    """
    CONFIGURACIÓN DE SALIDA PARA UN TRIAL.

    Generada por resolve_exit_settings_for_trial() o exit_settings_from_params().
    Consumida por engine.py para ejecutar la lógica de SL/TP/Trailing.
    """
    exit_type: str = DEFAULT_EXIT_TYPE
    sl_pct: float = DEFAULT_EXIT_SL_PCT
    tp_pct: float = DEFAULT_EXIT_TP_PCT
    trail_act_pct: float = DEFAULT_EXIT_TRAIL_ACT_PCT
    trail_dist_pct: float = DEFAULT_EXIT_TRAIL_DIST_PCT
    time_stop_bars: int = DEFAULT_EXIT_TIME_BARS  # 0 = desactivado
    # ── ATR adaptive — mismo formato % que sl_pct/tp_pct ──
    atr_period:   int   = DEFAULT_EXIT_ATR_PERIOD
    atr_min_pct:  float = DEFAULT_EXIT_ATR_MIN_PCT   # % stake mínimo (ej. 20.0)
    atr_max_pct:  float = DEFAULT_EXIT_ATR_MAX_PCT   # % stake máximo (ej. 40.0)
    atr_lookback: int   = DEFAULT_EXIT_ATR_LOOKBACK


@dataclass(frozen=True)
class ExitResult:
    """
    RESULTADO DE UNA SALIDA (PARA COMPATIBILIDAD CON ESTRATEGIAS PERSONALIZADAS).

    Campos:
        exit_idx:     Índice de la barra donde se ejecuta la salida.
        exit_price:   Precio de salida.
        tipo_salida:  Razón de salida ("sl", "tp", "trailing", etc.).
        sl_distance:  Distancia al SL en el momento de salida.
    """
    exit_idx: int
    exit_price: float
    tipo_salida: str
    sl_distance: float = 0.0


# =============================================================================
# 4. NORMALIZACIÓN
# =============================================================================

def _normalize_exit_values(
    exit_type: str,
    sl_pct: float,
    tp_pct: float,
    trail_act_pct: float,
    trail_dist_pct: float,
) -> tuple:
    """
    NORMALIZA VALORES DE EXIT SETTINGS.

    Reglas:
        - Todos los valores son positivos (abs).
        - En trailing, tp_pct puede ser 0 (no aplica).
        - trail_dist_pct <= trail_act_pct / 2 (el trailing no puede ser
          más ancho que la mitad de la activación).

    Returns:
        Tupla (sl_pct, tp_pct, trail_act_pct, trail_dist_pct) normalizados.
    """
    sl_pct = abs(sl_pct) if sl_pct != 0 else 1.0

    if exit_type == "TRAILING":
        tp_pct = abs(tp_pct) if tp_pct != 0 else 0.0
    else:
        tp_pct = abs(tp_pct) if tp_pct != 0 else 1.0

    trail_act_pct = abs(trail_act_pct) if trail_act_pct != 0 else 0.5
    trail_dist_pct = abs(trail_dist_pct) if trail_dist_pct != 0 else 0.25

    # LÓGICA DE SWAP ROBUSTA (Mismo concepto que ZLEMA Fast/Slow)
    # Si el optimizador saca una distancia mayor que la activación, se intercambian
    if trail_dist_pct >= trail_act_pct:
        temp = trail_act_pct
        trail_act_pct = trail_dist_pct
        trail_dist_pct = temp

    # Caso borde: si por algún motivo bizarro son idénticos, subimos un poco la activación
    if trail_act_pct == trail_dist_pct:
        trail_act_pct += 0.5

    return float(sl_pct), float(tp_pct), float(trail_act_pct), float(trail_dist_pct)


# =============================================================================
# 5. FUNCIONES DE RESOLUCIÓN
# =============================================================================

def resolve_exit_settings_for_trial(*, trial: Any, config: Any) -> ExitSettings:
    """
    RESUELVE PARÁMETROS DE SALIDA PARA UN TRIAL DE OPTUNA.

    Si optimize_exits == True:
        Usa trial.suggest_float() para optimizar SL/TP/Trailing.
    Si optimize_exits == False:
        Usa los valores fijos definidos en config.

    Args:
        trial:  Objeto Trial de Optuna.
        config: BacktestConfig con valores base y rangos.

    Returns:
        ExitSettings con valores resueltos (fijos u optimizados).
    """
    optimize = bool(getattr(config, "optimize_exits", DEFAULT_OPTIMIZE_EXITS))
    exit_type = str(getattr(config, "exit_type", DEFAULT_EXIT_TYPE)).strip().upper()

    # ─── VALORES BASE ────────────────────────────────────────────────────
    sl_pct = float(getattr(config, "exit_sl_pct", DEFAULT_EXIT_SL_PCT))
    tp_pct = float(getattr(config, "exit_tp_pct", DEFAULT_EXIT_TP_PCT))
    trail_act = float(getattr(config, "exit_trail_act_pct", DEFAULT_EXIT_TRAIL_ACT_PCT))
    trail_dist = float(getattr(config, "exit_trail_dist_pct", DEFAULT_EXIT_TRAIL_DIST_PCT))
    time_stop_bars = int(getattr(config, "exit_time_bars", DEFAULT_EXIT_TIME_BARS))

    # EN TRAILING, TP NO APLICA
    if exit_type == "TRAILING":
        tp_pct = 0.0

    # SI NO ES BARS (o ALL), TIME STOP NO APLICA
    if exit_type not in {"BARS", "ALL"}:
        time_stop_bars = 0

    # ─── OPTIMIZACIÓN TIME BARS ─────────────────────────────────────────
    time_bars_rng = tuple(getattr(config, "exit_time_bars_range", DEFAULT_EXIT_TIME_BARS_RANGE))

    # ─── OPTIMIZACIÓN CON OPTUNA ─────────────────────────────────────────
    if optimize:
        sl_rng = tuple(getattr(config, "exit_sl_pct_range", DEFAULT_EXIT_SL_PCT_RANGE))
        tp_rng = tuple(getattr(config, "exit_tp_pct_range", DEFAULT_EXIT_TP_PCT_RANGE))
        act_rng = tuple(getattr(config, "exit_trail_act_pct_range", DEFAULT_EXIT_TRAIL_ACT_PCT_RANGE))
        dist_rng = tuple(getattr(config, "exit_trail_dist_pct_range", DEFAULT_EXIT_TRAIL_DIST_PCT_RANGE))

        # SL SIEMPRE SE OPTIMIZA
        sl_pct = trial.suggest_float(
            "exit_sl_pct", sl_rng[0], sl_rng[1],
            step=sl_rng[2] if len(sl_rng) > 2 else 0.1,
        )

        # TP SOLO EN FIXED
        if exit_type in {"FIXED", "all"}:
            tp_pct = trial.suggest_float(
                "exit_tp_pct", tp_rng[0], tp_rng[1],
                step=tp_rng[2] if len(tp_rng) > 2 else 0.1,
            )
        else:
            tp_pct = 0.0

        # TRAILING PARAMS SOLO EN TRAILING
        if exit_type in {"TRAILING", "all"}:
            trail_act = trial.suggest_float(
                "exit_trail_act_pct", act_rng[0], act_rng[1],
                step=act_rng[2] if len(act_rng) > 2 else 0.1,
            )
            trail_dist = trial.suggest_float(
                "exit_trail_dist_pct", dist_rng[0], dist_rng[1],
                step=dist_rng[2] if len(dist_rng) > 2 else 0.1,
            )

        if exit_type == "BARS":
            time_stop_bars = trial.suggest_int(
                "exit_time_bars", int(time_bars_rng[0]), int(time_bars_rng[1]),
                step=int(time_bars_rng[2]) if len(time_bars_rng) > 2 else 1,
            )

    # ─── ATR ADAPTIVE (parámetros fijos, sin optimización por defecto) ───
    atr_period   = int(getattr(config, "exit_atr_period",   DEFAULT_EXIT_ATR_PERIOD))
    atr_min_pct  = float(getattr(config, "exit_atr_min_pct", DEFAULT_EXIT_ATR_MIN_PCT))
    atr_max_pct  = float(getattr(config, "exit_atr_max_pct", DEFAULT_EXIT_ATR_MAX_PCT))
    atr_lookback = int(getattr(config, "exit_atr_lookback", DEFAULT_EXIT_ATR_LOOKBACK))

    # ─── NORMALIZAR ──────────────────────────────────────────────────────
    sl_pct, tp_pct, trail_act, trail_dist = _normalize_exit_values(
        exit_type, sl_pct, tp_pct, trail_act, trail_dist
    )

    return ExitSettings(
        exit_type=exit_type,
        sl_pct=sl_pct,
        tp_pct=tp_pct,
        trail_act_pct=trail_act,
        trail_dist_pct=trail_dist,
        time_stop_bars=time_stop_bars,
        atr_period=atr_period,
        atr_min_pct=atr_min_pct,
        atr_max_pct=atr_max_pct,
        atr_lookback=atr_lookback,
    )


def exit_settings_from_params(params: Dict[str, Any]) -> ExitSettings:
    """
    LEE SETTINGS DESDE UN DICCIONARIO DE PARAMS (FUERA DE OPTIMIZACIÓN).

    Prioridad de lectura: __exit_* > exit_* > defaults.

    Args:
        params: Diccionario de parámetros del trial.

    Returns:
        ExitSettings con valores resueltos.
    """
    exit_type = str(
        params.get("__exit_type", params.get("exit_type", DEFAULT_EXIT_TYPE))
    ).strip().upper()
    sl_pct = float(
        params.get("__exit_sl_pct", params.get("exit_sl_pct", DEFAULT_EXIT_SL_PCT))
    )
    tp_pct = float(
        params.get("__exit_tp_pct", params.get("exit_tp_pct", DEFAULT_EXIT_TP_PCT))
    )
    trail_act = float(
        params.get("__exit_trail_act_pct", params.get("exit_trail_act_pct", DEFAULT_EXIT_TRAIL_ACT_PCT))
    )
    trail_dist = float(
        params.get("__exit_trail_dist_pct", params.get("exit_trail_dist_pct", DEFAULT_EXIT_TRAIL_DIST_PCT))
    )
    time_stop_bars = int(
        params.get("__exit_time_bars", params.get("exit_time_bars", DEFAULT_EXIT_TIME_BARS))
    )
    atr_period = int(
        params.get("__exit_atr_period", params.get("exit_atr_period", DEFAULT_EXIT_ATR_PERIOD))
    )
    atr_min_pct = float(
        params.get("__exit_atr_min_pct", params.get("exit_atr_min_pct", DEFAULT_EXIT_ATR_MIN_PCT))
    )
    atr_max_pct = float(
        params.get("__exit_atr_max_pct", params.get("exit_atr_max_pct", DEFAULT_EXIT_ATR_MAX_PCT))
    )
    atr_lookback = int(
        params.get("__exit_atr_lookback", params.get("exit_atr_lookback", DEFAULT_EXIT_ATR_LOOKBACK))
    )

    sl_pct, tp_pct, trail_act, trail_dist = _normalize_exit_values(
        exit_type, sl_pct, tp_pct, trail_act, trail_dist
    )

    if exit_type not in {"BARS", "ALL"}:
        time_stop_bars = 0

    return ExitSettings(
        exit_type=exit_type,
        sl_pct=sl_pct,
        tp_pct=tp_pct,
        trail_act_pct=trail_act,
        trail_dist_pct=trail_dist,
        time_stop_bars=time_stop_bars,
        atr_period=atr_period,
        atr_min_pct=atr_min_pct,
        atr_max_pct=atr_max_pct,
        atr_lookback=atr_lookback,
    )


# =============================================================================
 # 6. EXPORTACIONES
# =============================================================================

__all__ = [
    # ─── DEFAULTS ────────────────────────────────────────────────────────
    "DEFAULT_EXIT_TYPE",
    "DEFAULT_EXIT_SL_PCT",
    "DEFAULT_EXIT_TP_PCT",
    "DEFAULT_EXIT_TRAIL_ACT_PCT",
    "DEFAULT_EXIT_TRAIL_DIST_PCT",
    "DEFAULT_EXIT_TIME_BARS",
    "DEFAULT_OPTIMIZE_EXITS",
    "DEFAULT_EXIT_SL_PCT_RANGE",
    "DEFAULT_EXIT_TP_PCT_RANGE",
    "DEFAULT_EXIT_TRAIL_ACT_PCT_RANGE",
    "DEFAULT_EXIT_TRAIL_DIST_PCT_RANGE",
    "DEFAULT_EXIT_TIME_BARS_RANGE",
    "DEFAULT_EXIT_ATR_PERIOD",
    "DEFAULT_EXIT_ATR_MIN_PCT",
    "DEFAULT_EXIT_ATR_MAX_PCT",
    "DEFAULT_EXIT_ATR_LOOKBACK",
    # ─── DATACLASSES ─────────────────────────────────────────────────────
    "ExitSettings",
    "ExitResult",
    # ─── FUNCIONES ───────────────────────────────────────────────────────
    "resolve_exit_settings_for_trial",
    "exit_settings_from_params",
]
