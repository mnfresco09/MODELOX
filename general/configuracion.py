"""general/configuracion.py

CONFIGURACIÓN CENTRAL DE MODELOX.

REGLA DE ORO
  - TODO lo que vayas a ajustar tú está arriba en "AJUSTA AQUÍ".
  - El resto es "NO TOCAR" (helpers + compatibilidad).

ACTIVOS SOPORTADOS
  - BTC, GOLD, SP500, NASDAQ

TIMEFRAMES SOPORTADOS (minutos)
  - 5  -> sufijo "5m"
  - 15 -> sufijo "15m"
  - 60 -> sufijo "1h"
"""

from __future__ import annotations

from typing import Iterable

from modelox.core.exits import (
    DEFAULT_EXIT_ATR_PERIOD,
    DEFAULT_EXIT_ATR_PERIOD_RANGE,
    DEFAULT_EXIT_SL_ATR,
    DEFAULT_EXIT_SL_ATR_RANGE,
    DEFAULT_EXIT_TIME_STOP_BARS,
    DEFAULT_EXIT_TIME_STOP_BARS_RANGE,
    DEFAULT_EXIT_TP_ATR,
    DEFAULT_EXIT_TP_ATR_RANGE,
    DEFAULT_OPTIMIZE_EXITS,
)
from modelox.core.timeframes import normalize_timeframe_to_suffix


# =============================================================================
# AJUSTA AQUÍ (CONFIGURACIÓN DEL USUARIO)
# =============================================================================

# ----------------------------------------------------------------------------
# ACTIVOS A EJECUTAR
# ----------------------------------------------------------------------------
# Puedes pasar:
#   - "GOLD"                    (uno)
#   - "GOLD, BTC, SP500"        (varios, separado por coma)
#   - ["GOLD", "BTC"]           (lista)
ACTIVO = "GOLD, BTC, SP500"


# ----------------------------------------------------------------------------
# TIMEFRAME BASE (EJECUCIÓN)
# ----------------------------------------------------------------------------
# Timeframe base del backtest (minutos): 5 / 15 / 60
TIMEFRAME = 15


# ----------------------------------------------------------------------------
# FECHAS (BACKTEST Y PLOT)
# ----------------------------------------------------------------------------
FECHA_INICIO = "2019-01-10"
FECHA_FIN = "2025-10-15"

FECHA_INICIO_PLOT = "2019-01-10"
FECHA_FIN_PLOT = "2019-12-15"


# ----------------------------------------------------------------------------
# OPTUNA
# ----------------------------------------------------------------------------
N_TRIALS = 15
OPTUNA_N_JOBS = 1      # 1 recomendado en macOS (más estable)
OPTUNA_SEED = None     # None = seed aleatoria
OPTUNA_STORAGE = None  # None = in-memory (o ruta SQLite)


# ----------------------------------------------------------------------------
# EJECUCIÓN: QUÉ ESTRATEGIAS CORRER
# ----------------------------------------------------------------------------
# Single:    COMBINACION_A_EJECUTAR = 7
# Multiple:  COMBINACION_A_EJECUTAR = [3, 4, 7]
# All:       COMBINACION_A_EJECUTAR = "all"
COMBINACION_A_EJECUTAR = [2]


# ----------------------------------------------------------------------------
# CUENTA / COSTES
# ----------------------------------------------------------------------------
SALDO_INICIAL = 300
SALDO_OPERATIVO_MAX = 300
APALANCAMIENTO = 50
COMISION_PCT = 0.00043
COMISION_SIDES = 1
SALDO_MINIMO_OPERATIVO = 5


# ----------------------------------------------------------------------------
# LÍMITES DE POSICIÓN POR ACTIVO
# ----------------------------------------------------------------------------
# Límite duro por trade (aunque el apalancamiento permita más).
QTY_MAX_MAP = {
    "BTC": 0.025,
    "GOLD": 0.5,
    "SP500": 0.5,
    "NASDAQ": 0.15,
}

# Permitir que Optuna optimice qty_max_activo dentro de un rango por activo.
OPTIMIZAR_QTY_ACTIVO = True
QTY_MAX_RANGE_MAP = {
    "BTC": (0.01, 0.1, 0.01),
    "GOLD": (0.75, 3.5, 0.25),
    "SP500": (0.5, 4.0, 0.25),
    "NASDAQ": (0.025, 0.75, 0.01),
}


# ----------------------------------------------------------------------------
# SALIDAS (GLOBAL, ENGINE-OWNED)
# ----------------------------------------------------------------------------
EXIT_ATR_PERIOD = DEFAULT_EXIT_ATR_PERIOD
EXIT_SL_ATR = DEFAULT_EXIT_SL_ATR
EXIT_TP_ATR = DEFAULT_EXIT_TP_ATR
EXIT_TIME_STOP_BARS = DEFAULT_EXIT_TIME_STOP_BARS

OPTIMIZAR_SALIDAS = DEFAULT_OPTIMIZE_EXITS
EXIT_ATR_PERIOD_RANGE = DEFAULT_EXIT_ATR_PERIOD_RANGE
EXIT_SL_ATR_RANGE = DEFAULT_EXIT_SL_ATR_RANGE
EXIT_TP_ATR_RANGE = DEFAULT_EXIT_TP_ATR_RANGE
EXIT_TIME_STOP_BARS_RANGE = DEFAULT_EXIT_TIME_STOP_BARS_RANGE


# ----------------------------------------------------------------------------
# SALIDA DE RESULTADOS
# ----------------------------------------------------------------------------
MAX_ARCHIVOS_GUARDAR = 5
GENERAR_PLOTS = True
USAR_EXCEL = True


# ----------------------------------------------------------------------------
# MANTENIMIENTO (OPCIONAL)
# ----------------------------------------------------------------------------
# `ejecutar.py` puede usarlo para purgar __pycache__ al finalizar.
PURGE_PYCACHE_ON_EXIT = True


# =============================================================================
# NO TOCAR (HELPERS + COMPATIBILIDAD)
# =============================================================================

_ACTIVO_ALIASES = {
    "SP": "SP500",
    "NDX": "NASDAQ",
}


def _normalize_activos(v: object) -> list[str]:
    """Normaliza ACTIVO a una lista de símbolos (upper) sin vacíos."""
    if isinstance(v, (list, tuple)):
        raw: Iterable[str] = [str(x) for x in v]
    else:
        raw = str(v).split(",")

    out: list[str] = []
    for a in raw:
        a = str(a).strip().upper()
        if not a:
            continue
        a = _ACTIVO_ALIASES.get(a, a)
        out.append(a)

    return out or ["GOLD"]


# Multi-asset support: ejecutar.py iterará esta lista.
ACTIVOS = _normalize_activos(ACTIVO)
ACTIVO_PRIMARIO = ACTIVOS[0]


def resolve_archivo_data_tf(activo: str, timeframe: object = None, *, formato: str = "parquet") -> str:
    """Resuelve archivo de datos por activo + timeframe.

    Convención:
      data/ohlcv/<ACTIVO>_ohlcv_<SUFIJO>.<formato>
    donde SUFIJO ∈ {"5m", "15m", "1h"}
    """
    suf = normalize_timeframe_to_suffix(timeframe if timeframe is not None else TIMEFRAME)
    a = _ACTIVO_ALIASES.get(str(activo).upper().strip(), str(activo).upper().strip()) or "GOLD"
    ext = str(formato).lower().lstrip(".")
    return f"data/ohlcv/{a}_ohlcv_{suf}.{ext}"


def resolve_archivo_data(activo: str) -> str:
    """Compat: devuelve el path del timeframe 1h (histórico) en parquet."""
    return resolve_archivo_data_tf(activo, 60, formato="parquet")


# Compat (algunos outputs/prints lo usan)
ARCHIVO_DATA = resolve_archivo_data_tf(ACTIVO_PRIMARIO, TIMEFRAME, formato="parquet")


def resolve_qty_max_activo(activo: str) -> float:
    a = _ACTIVO_ALIASES.get(str(activo).upper().strip(), str(activo).upper().strip())
    return float(QTY_MAX_MAP.get(a, 3.0))


def resolve_qty_max_activo_range(activo: str) -> tuple[float, float, float]:
    a = _ACTIVO_ALIASES.get(str(activo).upper().strip(), str(activo).upper().strip())
    return tuple(QTY_MAX_RANGE_MAP.get(a, (0.01, 5.0, 0.01)))


# Compat: límites “por defecto” apuntan al activo primario
QTY_MAX_ACTIVO = resolve_qty_max_activo(ACTIVO_PRIMARIO)


# Dict unificado (compatibilidad con módulos legacy)
CONFIG = {
    "ACTIVO": ACTIVO_PRIMARIO,
    "ACTIVOS": ACTIVOS,
    "TIMEFRAME": TIMEFRAME,

    "SALDO_INICIAL": SALDO_INICIAL,
    "SALDO_OPERATIVO_MAX": SALDO_OPERATIVO_MAX,
    "APALANCAMIENTO": APALANCAMIENTO,
    "COMISION_PCT": COMISION_PCT,
    "COMISION_SIDES": COMISION_SIDES,
    "SALDO_MINIMO_OPERATIVO": SALDO_MINIMO_OPERATIVO,

    "QTY_MAX_ACTIVO": QTY_MAX_ACTIVO,
    "OPTIMIZAR_QTY_ACTIVO": OPTIMIZAR_QTY_ACTIVO,
    "QTY_MAX_MAP": QTY_MAX_MAP,
    "QTY_MAX_RANGE_MAP": QTY_MAX_RANGE_MAP,

    "N_TRIALS": N_TRIALS,
    "OPTUNA_N_JOBS": OPTUNA_N_JOBS,
    "OPTUNA_SEED": OPTUNA_SEED,
    "OPTUNA_STORAGE": OPTUNA_STORAGE,

    "COMBINACION_A_EJECUTAR": COMBINACION_A_EJECUTAR,

    "EXIT_ATR_PERIOD": EXIT_ATR_PERIOD,
    "EXIT_SL_ATR": EXIT_SL_ATR,
    "EXIT_TP_ATR": EXIT_TP_ATR,
    "EXIT_TIME_STOP_BARS": EXIT_TIME_STOP_BARS,
    "OPTIMIZAR_SALIDAS": OPTIMIZAR_SALIDAS,
    "EXIT_ATR_PERIOD_RANGE": EXIT_ATR_PERIOD_RANGE,
    "EXIT_SL_ATR_RANGE": EXIT_SL_ATR_RANGE,
    "EXIT_TP_ATR_RANGE": EXIT_TP_ATR_RANGE,
    "EXIT_TIME_STOP_BARS_RANGE": EXIT_TIME_STOP_BARS_RANGE,

    "MAX_ARCHIVOS_GUARDAR": MAX_ARCHIVOS_GUARDAR,
    "GENERAR_PLOTS": GENERAR_PLOTS,
    "USAR_EXCEL": USAR_EXCEL,

    "PURGE_PYCACHE_ON_EXIT": PURGE_PYCACHE_ON_EXIT,
}
