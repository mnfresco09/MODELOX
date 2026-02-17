"""
================================================================================
MODELOX/CORE/EXITS.PY — CONFIGURACIÓN CENTRALIZADA DE SALIDAS
================================================================================

PROPÓSITO:
    FUENTE ÚNICA DE VERDAD para todos los parámetros de salida del sistema.
    Ningún otro archivo debe definir defaults de SL/TP/Trailing.

CONTENIDO:
    1. DEFAULTS               — Valores por defecto de SL/TP/Trailing
    2. RANGOS DE OPTIMIZACIÓN  — Rangos para Optuna (min, max, step)
    3. DATACLASSES             — ExitSettings, ExitResult
    4. NORMALIZACIÓN           — Validación y corrección de valores
    5. FUNCIONES DE RESOLUCIÓN — resolve_exit_settings_for_trial, exit_settings_from_params
    6. EXPORTACIONES           — __all__

ARQUITECTURA:
    ┌──────────────────────────────────────────────────────────────────────┐
    │  exits.py (ESTE ARCHIVO) → CONFIGURACIÓN                           │
    │    └── Defaults, rangos, resolve_exit_settings_for_trial()          │
    │                    │                                                │
    │                    ▼                                                │
    │  runner.py → Inyecta __exit_* en params por trial                   │
    │                    │                                                │
    │                    ▼                                                │
    │  engine.py → ÚNICA implementación de lógica (Numba optimizado)      │
    │    └── SL/TP/Trailing ejecutados INLINE en kernel                   │
    └──────────────────────────────────────────────────────────────────────┘

TIPOS DE SALIDA:
    - "pnl_fixed":    SL/TP fijos por % sobre stake.
    - "pnl_trailing": SL inicial + trailing activado por % sobre stake.

DEFINICIONES:
    - STAKE = SALDO_USADO = margen/colateral por trade.
    - Los % son SIEMPRE sobre stake, NO sobre precio.

EJEMPLO:
    - Stake = 100€, sl_pct = 8%  → Salir si pierdo 8€
    - Stake = 100€, tp_pct = 14% → Salir si gano 14€

IMPORTANTE: USO DE VELAS DE 1 MINUTO PARA SALIDAS

REGLA FUNDAMENTAL:
    Las SALIDAS (SL/TP/Trailing) deben SIEMPRE evaluarse usando velas de 1 minuto,
    independientemente del timeframe configurado para las ENTRADAS.

JUSTIFICACIÓN:
    - Si usas timeframe de 5m, 15m, 1h, etc. para señales de entrada, pero el
      precio toca tu SL durante esa vela, NO puedes esperar al cierre de la vela
      para salir. El SL se habría ejecutado en el momento exacto que se tocó.
    
    - Usando velas de 1m para salidas, obtienes precisión casi tick-a-tick
      sin perder rendimiento computacional.

IMPLEMENTACIÓN:

    1. BACKTESTING (engine.py):
       - Si el timeframe de entrada es > 1m, obtener TAMBIÉN datos de 1m
       - Evaluar salidas iterando vela por vela de 1m
       - Ajustar precio de salida al high/low REAL de la vela 1m donde se tocó
    
    2. TRADING EN VIVO (trader.py):
       - Obtener velas de 1m en paralelo al timeframe de señales
       - Verificar SL/TP/Trailing cada vela de 1m
       - Ejecutar orden de cierre cuando se detecte toque de nivel
    
    3. AJUSTE DE PRECIOS:
       - SL tocado en LONG → exit_price = sl_price (no el low de la vela)
       - TP tocado en LONG → exit_price = tp_price (no el high de la vela)
       - SL tocado en SHORT → exit_price = sl_price (no el high de la vela)
       - TP tocado en SHORT → exit_price = tp_price (no el low de la vela)
       
       Esto simula la ejecución REAL donde la orden se llena al nivel exacto,
       no al extremo de la vela (que sería demasiado optimista).
    
    4. TRAILING STOP:
       - La DISTANCIA del trailing se calcula al CIERRE de la vela PREVIA
       - Ejemplo LONG: 
         * Vela N cierra en 100€, trailing_distance = 2€
         * trailing_level = 98€
         * En vela N+1, si low toca 98€ → salir a 98€
       
       - El trailing_level se ACTUALIZA cada vela:
         * LONG: Si nuevo high > high previo → actualizar trailing hacia arriba
         * SHORT: Si nuevo low < low previo → actualizar trailing hacia abajo
    
    5. SLIPPAGE Y REALISMO:
       - Los precios calculados (sl_price, tp_price, trailing_level) son TEÓRICOS
       - En trading real, puede haber slippage (especialmente en mercado rápido)
       - BingX ejecuta órdenes SL/TP como STOP_MARKET, no LIMIT
       - El precio final puede diferir ligeramente del nivel configurado

EJEMPLO PRÁCTICO:

    Configuración:
    - Timeframe señales: 5m
    - Timeframe salidas: 1m (SIEMPRE)
    - Entry: LONG @ 50,000€
    - SL: 49,500€
    - TP: 51,000€
    
    Escenario:
    - Vela 5m en curso: open=50,000, high=50,200, low=49,400, close=50,100
    
    Sin velas 1m (INCORRECTO):
    - Se esperaría al cierre de la vela 5m para detectar que low=49,400 < SL
    - Exit price = 49,400 o 50,100 (close) → INCORRECTO
    
    Con velas 1m (CORRECTO):
    - Minuto 2 de la vela 5m: low=49,450 → SL aún no tocado
    - Minuto 3 de la vela 5m: low=49,480 → SL tocado!
    - Exit price = 49,500 (el SL exacto) → CORRECTO
    - No esperamos al cierre de la vela 5m

REFERENCIA DE IMPLEMENTACIÓN:
    Ver engine.py → find_exits_numba() para la lógica Numba optimizada
    Ver trader.py → check_and_update_positions() para trading en vivo

================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


# =============================================================================
# 1. DEFAULTS — FUENTE ÚNICA DE VERDAD
# =============================================================================

# ─── TIPO DE SALIDA ──────────────────────────────────────────────────────────
DEFAULT_EXIT_TYPE = "pnl_fixed"

# ─── PARÁMETROS DE SL/TP (% SOBRE STAKE) ────────────────────────────────────
DEFAULT_EXIT_SL_PCT = 8.0                    # Stop Loss
DEFAULT_EXIT_TP_PCT = 14.0                   # Take Profit

# ─── PARÁMETROS DE TRAILING (% SOBRE STAKE) ─────────────────────────────────
DEFAULT_EXIT_TRAIL_ACT_PCT = 15.0            # Activación del trailing
DEFAULT_EXIT_TRAIL_DIST_PCT = 3.0            # Distancia del trailing


# =============================================================================
# 2. RANGOS DE OPTIMIZACIÓN (MIN, MAX, STEP)
# =============================================================================

DEFAULT_OPTIMIZE_EXITS = True

DEFAULT_EXIT_SL_PCT_RANGE = (14.0, 30.0, 2.0)
DEFAULT_EXIT_TP_PCT_RANGE = (10.0, 40.0, 2.0)
DEFAULT_EXIT_TRAIL_ACT_PCT_RANGE = (10.0, 28.0, 1.0)
DEFAULT_EXIT_TRAIL_DIST_PCT_RANGE = (2.0, 8.0, 0.5)


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
    time_stop_bars: int = 0                  # 0 = desactivado


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

    if exit_type == "pnl_trailing":
        tp_pct = abs(tp_pct) if tp_pct != 0 else 0.0
    else:
        tp_pct = abs(tp_pct) if tp_pct != 0 else 1.0

    trail_act_pct = abs(trail_act_pct) if trail_act_pct != 0 else 0.5
    trail_dist_pct = abs(trail_dist_pct) if trail_dist_pct != 0 else 0.25

    # LIMITAR DISTANCIA DEL TRAILING
    max_dist = max(0.0, trail_act_pct / 2.0)
    if max_dist > 0:
        trail_dist_pct = min(trail_dist_pct, max_dist)

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
    exit_type = str(getattr(config, "exit_type", DEFAULT_EXIT_TYPE)).strip().lower()

    # ─── VALORES BASE ────────────────────────────────────────────────────
    sl_pct = float(getattr(config, "exit_sl_pct", DEFAULT_EXIT_SL_PCT))
    tp_pct = float(getattr(config, "exit_tp_pct", DEFAULT_EXIT_TP_PCT))
    trail_act = float(getattr(config, "exit_trail_act_pct", DEFAULT_EXIT_TRAIL_ACT_PCT))
    trail_dist = float(getattr(config, "exit_trail_dist_pct", DEFAULT_EXIT_TRAIL_DIST_PCT))

    # EN TRAILING, TP NO APLICA
    if exit_type == "pnl_trailing":
        tp_pct = 0.0

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
        if exit_type in {"pnl_fixed", "all"}:
            tp_pct = trial.suggest_float(
                "exit_tp_pct", tp_rng[0], tp_rng[1],
                step=tp_rng[2] if len(tp_rng) > 2 else 0.1,
            )
        else:
            tp_pct = 0.0

        # TRAILING PARAMS SOLO EN TRAILING
        if exit_type in {"pnl_trailing", "all"}:
            trail_act = trial.suggest_float(
                "exit_trail_act_pct", act_rng[0], act_rng[1],
                step=act_rng[2] if len(act_rng) > 2 else 0.1,
            )
            trail_dist = trial.suggest_float(
                "exit_trail_dist_pct", dist_rng[0], dist_rng[1],
                step=dist_rng[2] if len(dist_rng) > 2 else 0.1,
            )

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
    ).strip().lower()
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

    sl_pct, tp_pct, trail_act, trail_dist = _normalize_exit_values(
        exit_type, sl_pct, tp_pct, trail_act, trail_dist
    )

    return ExitSettings(
        exit_type=exit_type,
        sl_pct=sl_pct,
        tp_pct=tp_pct,
        trail_act_pct=trail_act,
        trail_dist_pct=trail_dist,
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
    "DEFAULT_OPTIMIZE_EXITS",
    "DEFAULT_EXIT_SL_PCT_RANGE",
    "DEFAULT_EXIT_TP_PCT_RANGE",
    "DEFAULT_EXIT_TRAIL_ACT_PCT_RANGE",
    "DEFAULT_EXIT_TRAIL_DIST_PCT_RANGE",
    # ─── DATACLASSES ─────────────────────────────────────────────────────
    "ExitSettings",
    "ExitResult",
    # ─── FUNCIONES ───────────────────────────────────────────────────────
    "resolve_exit_settings_for_trial",
    "exit_settings_from_params",
]
