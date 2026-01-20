"""general/configuracion.py

CONFIGURACIÓN MAESTRA DEL SISTEMA MODELOX (VERSIÓN CLOUD/VM).
DISEÑADA PARA MÁXIMA ESTABILIDAD Y RENDIMIENTO.
"""

from __future__ import annotations

# IMPORTAMOS CONSTANTES DE MODELOX (VERIFICAR QUE ESTOS ARCHIVOS EXISTAN)
from modelox.core.exits import (
    DEFAULT_EXIT_TYPE, DEFAULT_EXIT_SL_PCT, DEFAULT_EXIT_TP_PCT,
    DEFAULT_EXIT_TRAIL_ACT_PCT, DEFAULT_EXIT_TRAIL_DIST_PCT, DEFAULT_OPTIMIZE_EXITS,
    DEFAULT_EXIT_SL_PCT_RANGE, DEFAULT_EXIT_TP_PCT_RANGE,
    DEFAULT_EXIT_TRAIL_ACT_PCT_RANGE, DEFAULT_EXIT_TRAIL_DIST_PCT_RANGE,
)
from modelox.core.types import normalize_timeframe_to_suffix

# =============================================================================
# [SECCIÓN 1] CONFIGURACIÓN DE USUARIO - PANEL DE CONTROL
# =============================================================================

# ----------------------------------------------------------------------------
# 1.1 SELECCIÓN DE ACTIVOS
# ----------------------------------------------------------------------------
# ELIGE EL ACTIVO A OPERAR. EJEMPLOS: "BTC", "GOLD", "SP500", "NASDAQ".
ACTIVO = "BTC"

# ----------------------------------------------------------------------------
# 1.2 TIMEFRAME BASE (RESOLUCIÓN DE DATOS)
# ----------------------------------------------------------------------------
# EL SISTEMA CARGA LA MÁXIMA RESOLUCIÓN DISPONIBLE (EL "ÁTOMO").
# NORMALMENTE "1m" (1 MINUTO). LAS ESTRATEGIAS PUEDEN PEDIR MÁS (1H, 4H).
TIMEFRAME_BASE = "1m"
TIMEFRAMES = [1]  # COMPATIBILIDAD (NO MODIFICAR)

# ----------------------------------------------------------------------------
# 1.3 RANGOS DE FECHAS (DIVISIÓN ESTRATÉGICA DEL HISTÓRICO)
# ----------------------------------------------------------------------------
# EL PERIODO 2020-2024 SE DIVIDE EN 3 FASES DE MERCADO DISTINTAS.
# SELECCIONA EL NÚMERO CORRESPONDIENTE PARA ENFOCAR EL BACKTEST.
#
#   [1] : 2020-01-01 -> 2021-07-28 (PANDEMIA + INICIO BULL RUN)
#   [2] : 2021-07-29 -> 2023-02-23 (ATH 69K + BEAR MARKET CRASH)
#   [3] : 2023-02-24 -> 2024-09-18 (RECUPERACIÓN + INSTITUCIONAL/ETF)
# ["ALL"]: 2020-01-01 -> 2024-09-18 (CICLO COMPLETO)
#
SELECCION_RANGO = 1  # <--- ¡MODIFICA AQUÍ! (1, 2, 3 o "ALL")

_RANGOS_FECHAS = {
    1:     ("2020-01-01", "2021-07-28"),
    2:     ("2021-07-29", "2023-02-23"),
    3:     ("2023-02-24", "2024-09-18"),
    "ALL": ("2020-01-01", "2024-09-18"),
}

# VALIDACIÓN DE SEGURIDAD PARA EL RANGO
if SELECCION_RANGO not in _RANGOS_FECHAS:
    print(f"⚠️ [CONFIG] RANGO '{SELECCION_RANGO}' NO RECONOCIDO. USANDO OPCIÓN 3 POR DEFECTO.")
    SELECCION_RANGO = 3

FECHA_INICIO, FECHA_FIN = _RANGOS_FECHAS[SELECCION_RANGO]

# FECHAS PARA GENERACIÓN DE GRÁFICOS (VISUALIZACIÓN DETALLADA)
FECHA_INICIO_PLOT = "2021-01-01"
FECHA_FIN_PLOT = "2021-03-15"

# ----------------------------------------------------------------------------
# 1.4 MOTOR DE OPTIMIZACIÓN (OPTUNA - MÁXIMA POTENCIA VM)
# ----------------------------------------------------------------------------
N_TRIALS = 5       # NÚMERO DE PRUEBAS (AUMENTAR SI TIENES BUENA CPU)
OPTUNA_N_JOBS = 1      # -1 = USAR TODOS LOS NÚCLEOS DISPONIBLES
OPTUNA_SEED = None      # SEMILLA ALEATORIA (NONE PARA VARIEDAD)
OPTUNA_STORAGE = None   # NONE = EJECUCIÓN EN RAM (MÁS RÁPIDO)

# ----------------------------------------------------------------------------
# 1.5 ESTRATEGIAS A EJECUTAR
# ----------------------------------------------------------------------------
# LISTA DE IDs DE LAS ESTRATEGIAS QUE SE VAN A PROBAR.
COMBINACION_A_EJECUTAR = [17]

# ----------------------------------------------------------------------------
# 1.6 GESTIÓN DE CAPITAL Y COSTES
# ----------------------------------------------------------------------------
SALDO_INICIAL = 1000.0
SALDO_OPERATIVO_MAX = 1000.0
COMISION_PCT = 0.0005        # 0.05% TAKER FEE (BINANCE/BYBIT)
COMISION_SIDES = 1           # 1 = SOLO AL ABRIR? RECOMENDADO 2 (ABRIR Y CERRAR)
SALDO_MINIMO_OPERATIVO = 300.0

# ----------------------------------------------------------------------------
# 1.7 TAMAÑO DE POSICIÓN (POSITION SIZING)
# ----------------------------------------------------------------------------
SALDO_USADO = 100.0           # COLATERAL FIJO POR OPERACIÓN
APALANCAMIENTO_MAX = 60       # TECHO DE APALANCAMIENTO

# LÍMITES DE CANTIDAD (LOT SIZE) POR ACTIVO
QTY_MAX_MAP = {
    "BTC": 0.045,
    "GOLD": 1.25,
    "SP500": 1.0,
    "NASDAQ": 0.25,
}

# RANGOS PARA QUE LA IA OPTIMICE EL TAMAÑO (SI SE ACTIVA)
OPTIMIZAR_QTY_ACTIVO = True
QTY_MAX_RANGE_MAP = {
    "BTC": (0.01, 0.08, 0.005),
    "GOLD": (0.5, 2.5, 0.5),
    "SP500": (0.5, 2.0, 0.5),
    "NASDAQ": (0.05, 0.5, 0.05),
}

# ----------------------------------------------------------------------------
# 1.8 SISTEMA DE SALIDAS (STOP LOSS Y TAKE PROFIT)
# ----------------------------------------------------------------------------
EXIT_TYPE = DEFAULT_EXIT_TYPE  # EJ: "pnl_trailing"
RIESGO_POR_TRADE_PCT = 0.10    # 10% DE RIESGO SOBRE EL CAPITAL

# PARÁMETROS BASE DE SALIDA
EXIT_SL_PCT = DEFAULT_EXIT_SL_PCT
EXIT_TP_PCT = DEFAULT_EXIT_TP_PCT
EXIT_TRAIL_ACT_PCT = DEFAULT_EXIT_TRAIL_ACT_PCT
EXIT_TRAIL_DIST_PCT = DEFAULT_EXIT_TRAIL_DIST_PCT

# CONFIGURACIÓN DE OPTIMIZACIÓN DE SALIDAS
OPTIMIZAR_SALIDAS = DEFAULT_OPTIMIZE_EXITS
EXIT_SL_PCT_RANGE = DEFAULT_EXIT_SL_PCT_RANGE
EXIT_TP_PCT_RANGE = DEFAULT_EXIT_TP_PCT_RANGE
EXIT_TRAIL_ACT_PCT_RANGE = DEFAULT_EXIT_TRAIL_ACT_PCT_RANGE
EXIT_TRAIL_DIST_PCT_RANGE = DEFAULT_EXIT_TRAIL_DIST_PCT_RANGE

# ----------------------------------------------------------------------------
# 1.9 RESULTADOS Y LIMPIEZA
# ----------------------------------------------------------------------------
MAX_ARCHIVOS_GUARDAR = 5
GENERAR_PLOTS = True
USAR_EXCEL = True
PURGE_PYCACHE_ON_EXIT = True


# =============================================================================
# [SECCIÓN 2] FUNCIONES INTERNAS (NO MODIFICAR)
# =============================================================================

_ACTIVO_ALIASES = {"SP": "SP500", "NDX": "NASDAQ"}

def _normalize_activos(v):
    """Convierte la entrada de activos en una lista limpia y en mayúsculas."""
    if isinstance(v, (list, tuple)): raw = [str(x) for x in v]
    else: raw = str(v).split(",")
    out = []
    for a in raw:
        a = str(a).strip().upper()
        if a: out.append(_ACTIVO_ALIASES.get(a, a))
    return out or ["GOLD"]

ACTIVOS = _normalize_activos(ACTIVO)
ACTIVO_PRIMARIO = ACTIVOS[0]

def _normalize_timeframes(v):
    """
    Convierte cualquier formato de timeframe (str, int, lista) a minutos (int).
    Soporta: '1h' -> 60, '4h' -> 240, '15m' -> 15.
    """
    def _parse_single(x):
        s = str(x).strip().lower()
        if not s: return None
        # Caso Horas (h)
        if s.endswith("h"):
            try:
                return int(float(s[:-1]) * 60)
            except ValueError:
                return None
        # Caso Minutos (m) o número puro
        s = s.replace("m", "")
        try:
            return int(float(s))
        except ValueError:
            return None

    if isinstance(v, (list, tuple)):
        raw = v
    else:
        raw = str(v).split(",")
    
    out = []
    for item in raw:
        m = _parse_single(item)
        if m is not None and m > 0:
            out.append(m)
    
    return sorted(list(set(out))) or [15]

TIMEFRAMES_NORM = _normalize_timeframes(TIMEFRAMES)
# Usamos el primer timeframe normalizado como referencia por defecto
TIMEFRAME = TIMEFRAMES_NORM[0] if TIMEFRAMES_NORM else 15

def resolve_archivo_data_tf(activo, timeframe=None, *, formato="parquet"):
    """Genera la ruta correcta del archivo de datos."""
    tf_val = timeframe if timeframe is not None else TIMEFRAME
    suf = normalize_timeframe_to_suffix(tf_val)
    a = _ACTIVO_ALIASES.get(str(activo).upper().strip(), str(activo).upper().strip()) or "GOLD"
    ext = formato.lower().lstrip('.')
    return f"data/ohlcv/{a}_ohlcv_{suf}.{ext}"

def resolve_archivo_data(activo):
    return resolve_archivo_data_tf(activo, 60, formato="parquet")

ARCHIVO_DATA = resolve_archivo_data_tf(ACTIVO_PRIMARIO, TIMEFRAME, formato="parquet")

def resolve_qty_max_activo(activo):
    a = _ACTIVO_ALIASES.get(str(activo).upper().strip(), str(activo).upper().strip())
    return float(QTY_MAX_MAP.get(a, 3.0))

def resolve_qty_max_activo_range(activo):
    a = _ACTIVO_ALIASES.get(str(activo).upper().strip(), str(activo).upper().strip())
    return tuple(QTY_MAX_RANGE_MAP.get(a, (0.01, 5.0, 0.01)))

QTY_MAX_ACTIVO = resolve_qty_max_activo(ACTIVO_PRIMARIO)

# DICCIONARIO UNIFICADO EXPORTABLE
CONFIG = {
    "ACTIVO": ACTIVO_PRIMARIO, "ACTIVOS": ACTIVOS,
    "TIMEFRAME": TIMEFRAME, "TIMEFRAMES": TIMEFRAMES_NORM,
    "SALDO_INICIAL": SALDO_INICIAL, "SALDO_OPERATIVO_MAX": SALDO_OPERATIVO_MAX,
    "SALDO_USADO": SALDO_USADO, "APALANCAMIENTO_MAX": APALANCAMIENTO_MAX,
    "COMISION_PCT": COMISION_PCT, "COMISION_SIDES": COMISION_SIDES,
    "SALDO_MINIMO_OPERATIVO": SALDO_MINIMO_OPERATIVO,
    "QTY_MAX_ACTIVO": QTY_MAX_ACTIVO, "OPTIMIZAR_QTY_ACTIVO": OPTIMIZAR_QTY_ACTIVO,
    "QTY_MAX_MAP": QTY_MAX_MAP, "QTY_MAX_RANGE_MAP": QTY_MAX_RANGE_MAP,
    "N_TRIALS": N_TRIALS, "OPTUNA_N_JOBS": OPTUNA_N_JOBS,
    "OPTUNA_SEED": OPTUNA_SEED, "OPTUNA_STORAGE": OPTUNA_STORAGE,
    "COMBINACION_A_EJECUTAR": COMBINACION_A_EJECUTAR,
    "EXIT_TYPE": EXIT_TYPE, "RIESGO_POR_TRADE_PCT": RIESGO_POR_TRADE_PCT,
    "EXIT_SL_PCT": EXIT_SL_PCT, "EXIT_TP_PCT": EXIT_TP_PCT,
    "EXIT_TRAIL_ACT_PCT": EXIT_TRAIL_ACT_PCT, "EXIT_TRAIL_DIST_PCT": EXIT_TRAIL_DIST_PCT,
    "OPTIMIZAR_SALIDAS": OPTIMIZAR_SALIDAS,
    "EXIT_SL_PCT_RANGE": EXIT_SL_PCT_RANGE, "EXIT_TP_PCT_RANGE": EXIT_TP_PCT_RANGE,
    "EXIT_TRAIL_ACT_PCT_RANGE": EXIT_TRAIL_ACT_PCT_RANGE, "EXIT_TRAIL_DIST_PCT_RANGE": EXIT_TRAIL_DIST_PCT_RANGE,
    "MAX_ARCHIVOS_GUARDAR": MAX_ARCHIVOS_GUARDAR, "GENERAR_PLOTS": GENERAR_PLOTS,
    "USAR_EXCEL": USAR_EXCEL, "PURGE_PYCACHE_ON_EXIT": PURGE_PYCACHE_ON_EXIT,
}