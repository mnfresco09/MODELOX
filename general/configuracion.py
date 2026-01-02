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
    DEFAULT_EXIT_TYPE,
    DEFAULT_EXIT_SL_PCT,
    DEFAULT_EXIT_TP_PCT,
    DEFAULT_EXIT_TRAIL_ACT_PCT,
    DEFAULT_EXIT_TRAIL_DIST_PCT,
    DEFAULT_OPTIMIZE_EXITS,
    DEFAULT_EXIT_SL_PCT_RANGE,
    DEFAULT_EXIT_TP_PCT_RANGE,
    DEFAULT_EXIT_TRAIL_ACT_PCT_RANGE,
    DEFAULT_EXIT_TRAIL_DIST_PCT_RANGE,
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
ACTIVO = " BTC"


# ----------------------------------------------------------------------------
# TIMEFRAME BASE (EJECUCIÓN)
# ----------------------------------------------------------------------------
# Timeframe base del backtest (minutos): 5 / 15 / 60
TIMEFRAME = 15


# ----------------------------------------------------------------------------
# FECHAS (BACKTEST Y PLOT)
# ----------------------------------------------------------------------------
FECHA_INICIO = "2022-01-10"
FECHA_FIN = "2024-08-15"

FECHA_INICIO_PLOT = "2022-01-10"
FECHA_FIN_PLOT = "2022-08-15"

# ----------------------------------------------------------------------------
# OPTUNA
# ----------------------------------------------------------------------------
N_TRIALS = 1500
OPTUNA_N_JOBS = 1      # 1 recomendado en macOS (más estable)
OPTUNA_SEED = None     # None = seed aleatoria
OPTUNA_STORAGE = None  # None = in-memory (o ruta SQLite)


# ----------------------------------------------------------------------------
# EJECUCIÓN: QUÉ ESTRATEGIAS CORRER
# ----------------------------------------------------------------------------
# Single:    COMBINACION_A_EJECUTAR = 7
# Multiple:  COMBINACION_A_EJECUTAR = [3, 4, 7]
# All:       COMBINACION_A_EJECUTAR = "all"
COMBINACION_A_EJECUTAR = [11,10]


# ----------------------------------------------------------------------------
# CUENTA / COSTES
# ----------------------------------------------------------------------------
SALDO_INICIAL = 300
SALDO_OPERATIVO_MAX = 300
COMISION_PCT = 0.00043
COMISION_SIDES = 1
SALDO_MINIMO_OPERATIVO = 15


# ----------------------------------------------------------------------------
# POSITION SIZING
# ----------------------------------------------------------------------------
# Sistema simplificado:
#   - SALDO_USADO: Fijo (margen/colateral por trade)
#   - QTY: Fija (del QTY_MAX_MAP, opcionalmente optimizable)
#   - APALANCAMIENTO: Variable, calculado dinámicamente, nunca supera el máximo
#
# Fórmula:
#   volumen = qty × precio
#   apalancamiento_necesario = volumen / saldo_usado
#
# Si apalancamiento_necesario > APALANCAMIENTO_MAX:
#   - Se usa APALANCAMIENTO_MAX
#   - Se reduce qty para respetar el límite:
#     volumen_max = saldo_usado × APALANCAMIENTO_MAX
#     qty = volumen_max / precio
#
SALDO_USADO = 75.0           # Margen/colateral fijo por trade
APALANCAMIENTO_MAX = 60      # Límite máximo de apalancamiento (nunca se supera)


# ----------------------------------------------------------------------------
# LÍMITES DE POSICIÓN POR ACTIVO (QTY)
# ----------------------------------------------------------------------------
# Cantidad objetivo por trade (se reduce si el apalancamiento excede el máximo).
QTY_MAX_MAP = {
    "BTC": 0.04,
    "GOLD": 1.25,
    "SP500": 1.0,
    "NASDAQ": 0.25,
}

# Permitir que Optuna optimice qty_max_activo dentro de un rango por activo.
OPTIMIZAR_QTY_ACTIVO = False
QTY_MAX_RANGE_MAP = {
    "BTC": (0.005, 0.03, 0.005),
    "GOLD": (0.25, 2.0, 0.25),
    "SP500": (0.25, 2.5, 0.25),
    "NASDAQ": (0.025, 0.5, 0.01),
}


# ----------------------------------------------------------------------------
# SALIDAS (GLOBAL, ENGINE-OWNED) - SISTEMA PNL_PCT
# ----------------------------------------------------------------------------
# Tipo de salida: "pnl_fixed", "pnl_trailing" o "all"
#
# Los parámetros son DIRECTAMENTE el PNL_PCT (ROI %) objetivo:
# - SL_PCT: Salir si PNL_PCT <= -sl_pct (pérdida máxima)
# - TP_PCT: Salir si PNL_PCT >= +tp_pct (ganancia objetivo)
# - TRAIL_ACT_PCT: Activar trailing cuando PNL_PCT >= trail_act_pct
# - TRAIL_DIST_PCT: Trailing retrocede trail_dist_pct desde máximo PNL
#
# OPCIONES:
#   1. "pnl_fixed": SL/TP fijos por PNL_PCT
#      - Ejemplo: sl_pct=2, tp_pct=5 → salir si pierde 2% o gana 5%
#
#   2. "pnl_trailing": SL inicial + trailing por PNL_PCT
#      - SL fijo hasta que PNL_PCT >= trail_act_pct
#      - Luego trailing protege (max_pnl - trail_dist_pct)
#
#   3. "all": Ejecuta AMBOS tipos secuencialmente
#
EXIT_TYPE = DEFAULT_EXIT_TYPE  # "pnl_trailing" por defecto

# ----------------------------------------------------------------------------
# POSITION SIZING: Fixed Fractional (Riesgo % por Trade)
# ----------------------------------------------------------------------------
# Fórmula: qty = (saldo * riesgo_pct) / sl_distance
# Esto asegura que cada trade arriesga exactamente el % configurado del saldo.
RIESGO_POR_TRADE_PCT = 0.10  # 10% del saldo por trade

# ----------------------------------------------------------------------------
# Parámetros de Salida PNL_PCT (ROI % del Trade)
# ----------------------------------------------------------------------------
# Stop Loss: Salir si PNL_PCT <= -sl_pct
EXIT_SL_PCT = DEFAULT_EXIT_SL_PCT  # 2.0% (pérdida máxima)

# Take Profit: Salir si PNL_PCT >= +tp_pct
EXIT_TP_PCT = DEFAULT_EXIT_TP_PCT  # 5.0% (ganancia objetivo)

# Trailing: Activar cuando PNL_PCT >= trail_act_pct
EXIT_TRAIL_ACT_PCT = DEFAULT_EXIT_TRAIL_ACT_PCT  # 1.0%

# Trailing: Retroceso desde máximo PNL alcanzado
EXIT_TRAIL_DIST_PCT = DEFAULT_EXIT_TRAIL_DIST_PCT  # 0.5%

# Optimización con Optuna
OPTIMIZAR_SALIDAS = DEFAULT_OPTIMIZE_EXITS

# Rangos para Optuna (min, max, step)
EXIT_SL_PCT_RANGE = DEFAULT_EXIT_SL_PCT_RANGE      # (0.5, 5.0, 0.1)
EXIT_TP_PCT_RANGE = DEFAULT_EXIT_TP_PCT_RANGE      # (1.0, 10.0, 0.1)
EXIT_TRAIL_ACT_PCT_RANGE = DEFAULT_EXIT_TRAIL_ACT_PCT_RANGE  # (0.5, 3.0, 0.1)
EXIT_TRAIL_DIST_PCT_RANGE = DEFAULT_EXIT_TRAIL_DIST_PCT_RANGE  # (0.2, 2.0, 0.1)


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
PURGE_PYCACHE_ON_EXIT = False


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
    "SALDO_USADO": SALDO_USADO,
    "APALANCAMIENTO_MAX": APALANCAMIENTO_MAX,
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

    # Sistema de Salidas Porcentual
    "EXIT_TYPE": EXIT_TYPE,
    "RIESGO_POR_TRADE_PCT": RIESGO_POR_TRADE_PCT,
    "EXIT_SL_PCT": EXIT_SL_PCT,
    "EXIT_TP_PCT": EXIT_TP_PCT,
    "EXIT_TRAIL_ACT_PCT": EXIT_TRAIL_ACT_PCT,
    "EXIT_TRAIL_DIST_PCT": EXIT_TRAIL_DIST_PCT,
    "OPTIMIZAR_SALIDAS": OPTIMIZAR_SALIDAS,
    "EXIT_SL_PCT_RANGE": EXIT_SL_PCT_RANGE,
    "EXIT_TP_PCT_RANGE": EXIT_TP_PCT_RANGE,
    "EXIT_TRAIL_ACT_PCT_RANGE": EXIT_TRAIL_ACT_PCT_RANGE,
    "EXIT_TRAIL_DIST_PCT_RANGE": EXIT_TRAIL_DIST_PCT_RANGE,

    "MAX_ARCHIVOS_GUARDAR": MAX_ARCHIVOS_GUARDAR,
    "GENERAR_PLOTS": GENERAR_PLOTS,
    "USAR_EXCEL": USAR_EXCEL,

    "PURGE_PYCACHE_ON_EXIT": PURGE_PYCACHE_ON_EXIT,
}
