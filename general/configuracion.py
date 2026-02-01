"""
# =============================================================================
#
#      ██████╗ ██████╗ ███╗   ██╗███████╗██╗ ██████╗
#     ██╔════╝██╔═══██╗████╗  ██║██╔════╝██║██╔════╝
#     ██║     ██║   ██║██╔██╗ ██║█████╗  ██║██║  ███╗
#     ██║     ██║   ██║██║╚██╗██║██╔══╝  ██║██║   ██║
#     ╚██████╗╚██████╔╝██║ ╚████║██║     ██║╚██████╔╝
#      ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝     ╚═╝ ╚═════╝
#
#     CONFIGURACION.PY - PANEL DE CONTROL MAESTRO
#
# =============================================================================
#
#     SECCIONES:
#     1. Activo y Timeframe
#     2. Rangos de Fechas
#     3. Capital y Gestión de Riesgo
#     4. Salidas (SL/TP/Trailing)
#     5. Optimización
#
# =============================================================================
"""

from __future__ import annotations

from modelox.core.exits import (
    DEFAULT_EXIT_TYPE, DEFAULT_EXIT_SL_PCT, DEFAULT_EXIT_TP_PCT,
    DEFAULT_EXIT_TRAIL_ACT_PCT, DEFAULT_EXIT_TRAIL_DIST_PCT, DEFAULT_OPTIMIZE_EXITS,
    DEFAULT_EXIT_SL_PCT_RANGE, DEFAULT_EXIT_TP_PCT_RANGE,
    DEFAULT_EXIT_TRAIL_ACT_PCT_RANGE, DEFAULT_EXIT_TRAIL_DIST_PCT_RANGE,
)
from modelox.core.types import normalize_timeframe_to_suffix


# =============================================================================
# 1. ACTIVO Y TIMEFRAME
# =============================================================================

ACTIVO = "BTC"
TIMEFRAME_BASE = "1m"
TIMEFRAMES = [1]


# =============================================================================
# 2. RANGOS DE FECHAS
# =============================================================================
#
#   "1"   : 2020-01-01 -> 2021-07-28 (PANDEMIA + BULL RUN)
#   "2"   : 2021-07-29 -> 2023-02-23 (ATH 69K + BEAR MARKET)
#   "3"   : 2023-02-24 -> 2024-09-18 (RECUPERACIÓN + ETF)
#   "all" : 2020-01-01 -> 2024-09-18 (CICLO COMPLETO)
#   "1,2" : Combina rangos 1 y 2
#   "2,3" : Combina rangos 2 y 3
#
# =============================================================================

SELECCION_RANGO = "2,3"

_RANGOS_FECHAS = {
    1:     ("2020-01-01", "2021-07-28"),
    2:     ("2021-07-29", "2023-02-23"),
    3:     ("2023-02-24", "2024-09-18"),
    "all": ("2020-01-01", "2024-09-18"),
}


def _normalizar_rango(seleccion):
    """NORMALIZA SELECCION_RANGO A FORMATO VÁLIDO."""
    if isinstance(seleccion, str):
        seleccion = seleccion.strip().lower()
        
        if seleccion == "all":
            return "all"
        
        if "," in seleccion:
            partes = [p.strip() for p in seleccion.split(",")]
            try:
                nums = sorted([int(p) for p in partes])
                if all(n in _RANGOS_FECHAS for n in nums):
                    fecha_inicio = _RANGOS_FECHAS[nums[0]][0]
                    fecha_fin = _RANGOS_FECHAS[nums[-1]][1]
                    return ("_combo", fecha_inicio, fecha_fin)
            except ValueError:
                pass
            return None
        
        try:
            return int(seleccion)
        except ValueError:
            return None
    
    if isinstance(seleccion, int):
        return seleccion
    
    return None


_rango_normalizado = _normalizar_rango(SELECCION_RANGO)

if _rango_normalizado is None:
    print(f"⚠️ RANGO '{SELECCION_RANGO}' NO RECONOCIDO. USANDO 3.")
    FECHA_INICIO, FECHA_FIN = _RANGOS_FECHAS[3]
elif isinstance(_rango_normalizado, tuple) and _rango_normalizado[0] == "_combo":
    FECHA_INICIO, FECHA_FIN = _rango_normalizado[1], _rango_normalizado[2]
elif _rango_normalizado in _RANGOS_FECHAS:
    FECHA_INICIO, FECHA_FIN = _RANGOS_FECHAS[_rango_normalizado]
else:
    print(f"⚠️ RANGO '{SELECCION_RANGO}' NO RECONOCIDO. USANDO 3.")
    FECHA_INICIO, FECHA_FIN = _RANGOS_FECHAS[3]

FECHA_INICIO_PLOT = "2021-01-01"
FECHA_FIN_PLOT = "2021-03-15"


# =============================================================================
# 3. CONFIGURACIÓN DE OPTIMIZACIÓN
# =============================================================================

N_TRIALS = 75000
OPTUNA_N_JOBS = -1
OPTUNA_SEED = None
OPTUNA_STORAGE = None

# SAMPLER: "PLATEAU" (RECOMENDADO), "CYCLIC", "CMA", "TPE", "GP" o "BOTORCH"
# - PLATEAU: Optimización en 3 fases (exploración + clustering + refinamiento)
# - CYCLIC: Descenso de Coordenadas Cíclico - Optimiza un parámetro a la vez
#           Repite ciclos hasta convergencia. Ideal para encontrar interacciones.
# - CMA: CMA-ES - Estrategia evolutiva adaptativa
# - TPE: Tree-Parzen Estimator - Bayesiano con árboles
# - GP: Gaussian Process (Optuna v4.0+) - Bayesiano clásico con procesos gaussianos
#       Equilibra exploración/explotación vía incertidumbre. Ideal para <20 dimensiones.
# - BOTORCH: GP avanzado con BoTorch (requiere: pip install botorch optuna-integration)
#       Soporta restricciones complejas y optimización multiobjetivo
OPTUNA_SAMPLER = "CYCLIC"

# TOPÓGRAFO DE MESETAS - FASE 1
PLATEAU_EXPLORATION_RATIO = 0.67
PLATEAU_EXPLORATION_SAMPLER = "qmc"

# TOPÓGRAFO DE MESETAS - FASE 2 (CLUSTERING)
PLATEAU_MIN_CLUSTER_SIZE = 10
PLATEAU_MIN_SAMPLES = 5
PLATEAU_MIN_TRIALS_FOR_MESETA = 2
PLATEAU_DBSCAN_EPS = 0.35

# TOPÓGRAFO DE MESETAS - FASE 3 (REFINAMIENTO)
PLATEAU_MAX_MESETAS = 0
PLATEAU_MIN_TRIALS_POR_MESETA = 50
PLATEAU_CENTROID_SELECTION = "centroid"
PLATEAU_AUTO_EPS = True

# =============================================================================
# DESCENSO DE COORDENADAS CÍCLICO (CYCLIC COORDINATE DESCENT)
# =============================================================================
# Optimiza un parámetro a la vez mientras fija los demás.
# Repite ciclos hasta que los parámetros converjan (no cambien).
#
# DOS MODOS DE OPERACIÓN:
# - CYCLIC_USE_N_TRIALS = True  → Usa N_TRIALS: hace ciclos hasta consumir todos los trials
# - CYCLIC_USE_N_TRIALS = False → Usa convergencia: para cuando no hay variación
#
# ANALOGÍA: Como afinar instrumentos en una banda:
# - Ajustas la Batería (RSI) primero
# - Luego la Guitarra (EMA) con la batería ya fijada
# - Vuelves a la Batería porque ahora la Guitarra cambió
# - Repites hasta que todo suene perfecto
# =============================================================================

# MODO DE OPERACIÓN
CYCLIC_USE_N_TRIALS = True         # True = usa N_TRIALS, False = usa convergencia

# PARÁMETROS DE CICLO
CYCLIC_MAX_CYCLES = 50              # Máximo de ciclos (seguridad, aplica en ambos modos)
CYCLIC_MIN_CYCLES = 3               # MÍNIMO 3 vueltas garantizadas
CYCLIC_CONVERGENCE_THRESHOLD = 0.02 # Umbral de convergencia entre ciclos (2%) - solo modo convergencia

# Convergencia ADAPTATIVA por parámetro (sin número fijo de trials) - solo modo convergencia
CYCLIC_PARAM_MIN_TRIALS = 20        # Mínimo trials antes de evaluar convergencia
CYCLIC_PARAM_MAX_TRIALS = 200       # Máximo trials por parámetro (seguridad)
CYCLIC_PARAM_PATIENCE = 15          # Trials sin mejora para converger parámetro
CYCLIC_PARAM_MIN_IMPROVEMENT = 0.001 # Mejora mínima para considerar progreso (0.1%)

# Trials por parámetro en MODO N_TRIALS (CYCLIC_USE_N_TRIALS = True)
# Se calcula automáticamente: N_TRIALS / (num_params * ciclos_estimados)
# O se puede forzar un valor fijo:
CYCLIC_TRIALS_PER_PARAM_FIXED = None  # None = auto, o poner número (ej: 100)

# MESETAS: Usar centroide de meseta en lugar del mejor valor exacto
# Más robusto contra overfitting - las mesetas se recalculan cada ciclo
CYCLIC_USE_PLATEAU = True           # True = fija con centroide de meseta, False = valor exacto
CYCLIC_PLATEAU_TOLERANCE = 0.02     # 2% tolerancia para definir meseta (scores similares)
CYCLIC_PLATEAU_MIN_POINTS = 5       # Mínimo puntos para considerar meseta válida

# AGRUPAR EXITS: Optimizar SL/TP (o SL/Trail) juntos como un bloque
# Tiene sentido porque SL y TP están relacionados (ratio riesgo/beneficio)
CYCLIC_GROUP_EXITS = True           # True = exits juntos, False = uno a uno

CYCLIC_PARAM_SAMPLER = "tpe"        # Sampler interno: "tpe" o "random"
CYCLIC_VERBOSE = True               # Mostrar progreso detallado
CYCLIC_INCLUDE_EXITS = True         # Incluir SL/TP/Trailing en optimización cíclica

# LIMPIEZA DE MEMORIA
CLEANUP_INTERVAL = 100

# PERTURBACIÓN (ANTI-OVERFITTING)
PERTURBACION_ACTIVAR = False
PERTURBACION_METHOD = "returns_perturbation"
                                               # "stationary_bootstrap" = Politis&Romano 1994
                                               # "returns_shuffle" = Shuffle simple de retornos
PERTURBACION_NOISE_SCALE = 0.5   # Escala del ruido (0.5 = 50% de la volatilidad)
PERTURBACION_BLOCK_SIZE = 360    # Tamaño de bloque para bootstrap
PERTURBACION_SEED = 42           # Semilla base para reproducibilidad
PERTURBACION_VERIFY = True       # Verificar coherencia OHLCV después de perturbar

# ----------------------------------------------------------------------------
# 1.5 ESTRATEGIAS A EJECUTAR
# ----------------------------------------------------------------------------
# LISTA DE IDs DE LAS ESTRATEGIAS QUE SE VAN A PROBAR.
COMBINACION_A_EJECUTAR = [4]

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
OPTIMIZAR_QTY_ACTIVO = False
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
GENERAR_PLOTS = True       # Generar gráficos HTML interactivos
USAR_EXCEL = True          # Generar archivos Excel con resumen y trades
PURGE_PYCACHE_ON_EXIT = True

# ----------------------------------------------------------------------------
# 1.10 SISTEMA DE SCORING UNIFICADO v8.0 (CALIDAD PURA)
# ----------------------------------------------------------------------------
# Sistema de evaluación basado en métricas de calidad.
# La robustez se valida en fases posteriores (Fase 2 y 3 del Topógrafo).
#
# COMPONENTE DE CALIDAD (0-600 puntos):
# - Normaliza Sharpe, SQN, ROI, Drawdown usando tanh
# - Factor de actividad (trades/día) se aplica como multiplicador


# =============================================================================
# [SECCIÓN 2] FUNCIONES INTERNAS (NO MODIFICAR)
# =============================================================================

_ACTIVO_ALIASES = {"SP": "SP500", "NDX": "NASDAQ"}

def _normalize_activos(v):
    """Convierte la entrada de activos en una lista limpia y en mayúsculas."""
    if isinstance(v, (list, tuple)):
        raw = [str(x) for x in v]
    else:
        raw = str(v).split(",")
    out = []
    for a in raw:
        a = str(a).strip().upper()
        if a:
            out.append(_ACTIVO_ALIASES.get(a, a))
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
        if not s:
            return None
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
    "OPTUNA_SAMPLER": OPTUNA_SAMPLER,  # PLATEAU, CMA, TPE, GP o BOTORCH
    "COMBINACION_A_EJECUTAR": COMBINACION_A_EJECUTAR,
    "EXIT_TYPE": EXIT_TYPE, "RIESGO_POR_TRADE_PCT": RIESGO_POR_TRADE_PCT,
    "EXIT_SL_PCT": EXIT_SL_PCT, "EXIT_TP_PCT": EXIT_TP_PCT,
    "EXIT_TRAIL_ACT_PCT": EXIT_TRAIL_ACT_PCT, "EXIT_TRAIL_DIST_PCT": EXIT_TRAIL_DIST_PCT,
    "OPTIMIZAR_SALIDAS": OPTIMIZAR_SALIDAS,
    "EXIT_SL_PCT_RANGE": EXIT_SL_PCT_RANGE, "EXIT_TP_PCT_RANGE": EXIT_TP_PCT_RANGE,
    "EXIT_TRAIL_ACT_PCT_RANGE": EXIT_TRAIL_ACT_PCT_RANGE, "EXIT_TRAIL_DIST_PCT_RANGE": EXIT_TRAIL_DIST_PCT_RANGE,
    "MAX_ARCHIVOS_GUARDAR": MAX_ARCHIVOS_GUARDAR, "GENERAR_PLOTS": GENERAR_PLOTS,
    "USAR_EXCEL": USAR_EXCEL, "PURGE_PYCACHE_ON_EXIT": PURGE_PYCACHE_ON_EXIT,
    # CONFIGURACIÓN DE PERTURBACIÓN
    "PERTURBACION_ACTIVAR": PERTURBACION_ACTIVAR,
    "PERTURBACION_METHOD": PERTURBACION_METHOD,
    "PERTURBACION_NOISE_SCALE": PERTURBACION_NOISE_SCALE,
    "PERTURBACION_BLOCK_SIZE": PERTURBACION_BLOCK_SIZE,
    "PERTURBACION_SEED": PERTURBACION_SEED,
    "PERTURBACION_VERIFY": PERTURBACION_VERIFY,
    # CONFIGURACIÓN DEL TOPÓGRAFO DE MESETAS
    "PLATEAU_EXPLORATION_RATIO": PLATEAU_EXPLORATION_RATIO,
    "PLATEAU_EXPLORATION_SAMPLER": PLATEAU_EXPLORATION_SAMPLER,
    "PLATEAU_MIN_CLUSTER_SIZE": PLATEAU_MIN_CLUSTER_SIZE,
    "PLATEAU_MIN_SAMPLES": PLATEAU_MIN_SAMPLES,
    "PLATEAU_DBSCAN_EPS": PLATEAU_DBSCAN_EPS,
    "PLATEAU_MIN_TRIALS_FOR_MESETA": PLATEAU_MIN_TRIALS_FOR_MESETA,
    "PLATEAU_MAX_MESETAS": PLATEAU_MAX_MESETAS,
    "PLATEAU_MIN_TRIALS_POR_MESETA": PLATEAU_MIN_TRIALS_POR_MESETA,
    "PLATEAU_CENTROID_SELECTION": PLATEAU_CENTROID_SELECTION,
    "PLATEAU_AUTO_EPS": PLATEAU_AUTO_EPS,
}
