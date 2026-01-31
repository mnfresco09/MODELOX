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
#   "1"       : 2020-01-01 -> 2021-07-28 (PANDEMIA + INICIO BULL RUN)
#   "2"       : 2021-07-29 -> 2023-02-23 (ATH 69K + BEAR MARKET CRASH)
#   "3"       : 2023-02-24 -> 2024-09-18 (RECUPERACIÓN + INSTITUCIONAL/ETF)
#   "all"     : 2020-01-01 -> 2024-09-18 (CICLO COMPLETO)
#   "1,2"     : Combina rangos 1 y 2 (2020-01-01 -> 2023-02-23)
#   "2,3"     : Combina rangos 2 y 3 (2021-07-29 -> 2024-09-18)
#
SELECCION_RANGO = "2,3"  # <--- ¡MODIFICA AQUÍ! ("1", "2", "3", "all", "1,2", "2,3")

_RANGOS_FECHAS = {
    1:     ("2020-01-01", "2021-07-28"),
    2:     ("2021-07-29", "2023-02-23"),
    3:     ("2023-02-24", "2024-09-18"),
    "all": ("2020-01-01", "2024-09-18"),
}

# NORMALIZACIÓN Y VALIDACIÓN DEL RANGO
def _normalizar_rango(seleccion):
    """Normaliza SELECCION_RANGO a formato válido."""
    # Si es string, convertir a minúsculas y limpiar
    if isinstance(seleccion, str):
        seleccion = seleccion.strip().lower()
        
        # Caso "all"
        if seleccion == "all":
            return "all"
        
        # Caso combinación "1,2" o "2,3"
        if "," in seleccion:
            partes = [p.strip() for p in seleccion.split(",")]
            try:
                nums = sorted([int(p) for p in partes])
                if all(n in _RANGOS_FECHAS for n in nums):
                    # Combinar: fecha inicio del primero, fecha fin del último
                    fecha_inicio = _RANGOS_FECHAS[nums[0]][0]
                    fecha_fin = _RANGOS_FECHAS[nums[-1]][1]
                    return ("_combo", fecha_inicio, fecha_fin)
            except ValueError:
                pass
            return None
        
        # Caso número como string "1", "2", "3"
        try:
            return int(seleccion)
        except ValueError:
            return None
    
    # Si es int, devolverlo directamente (compatibilidad)
    if isinstance(seleccion, int):
        return seleccion
    
    return None

_rango_normalizado = _normalizar_rango(SELECCION_RANGO)

# Validación
if _rango_normalizado is None:
    print(f"⚠️ [CONFIG] RANGO '{SELECCION_RANGO}' NO RECONOCIDO. USANDO OPCIÓN 3 POR DEFECTO.")
    FECHA_INICIO, FECHA_FIN = _RANGOS_FECHAS[3]
elif isinstance(_rango_normalizado, tuple) and _rango_normalizado[0] == "_combo":
    # Rango combinado
    FECHA_INICIO, FECHA_FIN = _rango_normalizado[1], _rango_normalizado[2]
elif _rango_normalizado in _RANGOS_FECHAS:
    FECHA_INICIO, FECHA_FIN = _RANGOS_FECHAS[_rango_normalizado]
else:
    print(f"⚠️ [CONFIG] RANGO '{SELECCION_RANGO}' NO RECONOCIDO. USANDO OPCIÓN 3 POR DEFECTO.")
    FECHA_INICIO, FECHA_FIN = _RANGOS_FECHAS[3]

# FECHAS PARA GENERACIÓN DE GRÁFICOS (VISUALIZACIÓN DETALLADA)
FECHA_INICIO_PLOT = "2021-01-01"
FECHA_FIN_PLOT = "2021-03-15"

# ----------------------------------------------------------------------------
# 1.4 MOTOR DE OPTIMIZACIÓN (OPTUNA - MÁXIMA POTENCIA VM)
# ----------------------------------------------------------------------------
N_TRIALS = 4000   # NÚMERO DE PRUEBAS (AUMENTAR SI TIENES BUENA CPU)
OPTUNA_N_JOBS = -1      # -1 = USAR TODOS LOS NÚCLEOS DISPONIBLES
OPTUNA_SEED = None      # SEMILLA ALEATORIA (NONE PARA VARIEDAD)
OPTUNA_STORAGE = None   # NONE = EJECUCIÓN EN RAM (MÁS RÁPIDO)

# ----------------------------------------------------------------------------
# 1.4.A ALGORITMO DE OPTIMIZACIÓN (SAMPLER)
# ----------------------------------------------------------------------------
# SELECCIONA EL ALGORITMO DE BÚSQUEDA:
#
#   "PLATEAU" = TOPÓGRAFO DE MESETAS (NUEVO - RECOMENDADO)
#               - Fase 1: Exploración masiva con RandomSampler
#               - Fase 2: Detección de mesetas con DBSCAN
#               - Fase 3: Refinamiento CMA-ES en cada meseta
#               - MÁXIMA ROBUSTEZ, evita overfitting
#
#   "CMA"  = CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
#            - Algoritmo clásico de optimización evolutiva
#            - Adapta la matriz de covarianza según los scores
#            - Puede caer en picos locales (overfitting)
#
#   "TPE"  = Tree-structured Parzen Estimator
#            - Algoritmo clásico de Optuna
#            - Bueno para espacios mixtos (continuos + categóricos)
#            - Más rápido en las primeras iteraciones
#
OPTUNA_SAMPLER = "PLATEAU"  # <--- MODIFICA AQUÍ ("PLATEAU", "CMA" o "TPE")

# ----------------------------------------------------------------------------
# 1.4.B CONFIGURACIÓN DEL TOPÓGRAFO DE MESETAS
# ----------------------------------------------------------------------------
# Solo aplica si OPTUNA_SAMPLER = "PLATEAU"
#
# FASE 1: EXPLORACIÓN
# El sistema "llena" el espacio de parámetros para ver el terreno completo.
PLATEAU_EXPLORATION_RATIO = 0.50  # 50% de trials para exploración (el otro 50% para refinamiento)
PLATEAU_EXPLORATION_SAMPLER = "qmc"  # "qmc" (RECOMENDADO), "random" o "tpe"
#
#   "qmc"    = Quasi-Monte Carlo (Secuencia Sobol) [RECOMENDADO]
#              - Cobertura SISTEMÁTICA y UNIFORME de todo el espacio
#              - NO es aleatorio, pero NO se centra en ninguna zona
#              - Más trials = mayor resolución (como aumentar pixeles)
#              - 100 trials = malla gruesa, 1000 trials = malla fina
#              - GARANTIZA explorar TODO el rango de cada parámetro
#
#   "random" = Aleatorio puro
#              - Distribución uniforme pero con "huecos" aleatorios
#              - Puede dejar zonas sin explorar por mala suerte
#
#   "tpe"    = Tree-structured Parzen Estimator
#              - APRENDE de trials anteriores
#              - Tiende a CONCENTRARSE en zonas prometedoras (greedy)
#              - Puede perderse mesetas si encuentra un pico primero

# FASE 2: DETECCIÓN DE MESETAS (HDBSCAN/DBSCAN)
# HDBSCAN agrupa puntos cercanos en el espacio de parámetros.
#
# PARÁMETROS DE CLUSTERING:
PLATEAU_MIN_CLUSTER_SIZE = 10  # Tamaño mínimo de una meseta (cluster)
                               # - Valor BAJO (5-10): Detecta mesetas pequeñas, más sensible
                               # - Valor ALTO (20-50): Solo mesetas grandes, más conservador
                               # RECOMENDADO: 10-30 para 1000+ trials

PLATEAU_MIN_SAMPLES = 5        # Mínimo de vecinos cercanos para ser "núcleo" de cluster
                               # - Valor BAJO (3-5): Acepta zonas poco densas
                               # - Valor ALTO (10-15): Solo zonas muy densas
                               # RECOMENDADO: 5-10

PLATEAU_MIN_TRIALS_FOR_MESETA = 2  # Mínimo trials para considerar meseta válida (post-filtro)

# LEGACY (solo si use_hdbscan=False)
PLATEAU_DBSCAN_EPS = 0.35  # Radio de vecindad para DBSCAN clásico (0.1-0.5)

# FILTRADO ANTES DEL CLUSTERING (AUTOMÁTICO - NO CONFIGURABLE)
# Se aplican DOS filtros secuenciales:
#
#   FILTRO 1: ROI >= 0%
#             Descarta TODOS los trials con ROI negativo (perdedores)
#
#   FILTRO 2: Score >= Media (μ)
#             Descarta TODOS los trials con score < media
#             Elimina automáticamente la mitad inferior
#
# Ejemplo: 2500 trials → Filtro ROI → 1800 pasan → Filtro Media → ~900 para clustering

# FASE 3: REFINAMIENTO CMA-ES
# Los trials de refinamiento (50%) se distribuyen proporcionalmente entre mesetas.
# Ejemplo: 4000 trials → 2000 exploración, 2000 refinamiento
#          Si DBSCAN encuentra 5 mesetas → 400 trials/meseta
#          Si DBSCAN encuentra 10 mesetas → 200 trials/meseta
PLATEAU_MAX_MESETAS = 0  # 0 = sin límite (refinar TODAS las mesetas encontradas)
PLATEAU_MIN_TRIALS_POR_MESETA = 50  # Mínimo de trials por meseta (si hay muchas)

# SELECCIÓN DEL REPRESENTANTE
# "centroid" = Trial más cercano al centro de la meseta (RECOMENDADO)
# "best" = Trial con mejor score en la meseta
# "median" = Trial con score mediano
PLATEAU_CENTROID_SELECTION = "centroid"

# AJUSTE AUTOMÁTICO
PLATEAU_AUTO_EPS = True  # Ajustar eps automáticamente según los datos

# ----------------------------------------------------------------------------
# 1.4.0 LIMPIEZA PERIÓDICA DE MEMORIA (ANTI-SLOWDOWN)
# ----------------------------------------------------------------------------
# Cada N trials se limpia la memoria para mantener velocidad constante.
# Si notas que el sistema se ralentiza con muchos trials, reduce este valor.
CLEANUP_INTERVAL = 100  # Limpiar memoria cada 100 trials (50-200 recomendado)

# ----------------------------------------------------------------------------
# 1.4.1 PERTURBACIÓN DE DATOS (ANTI-OVERFITTING)
# ----------------------------------------------------------------------------
# ACTIVA LA PERTURBACIÓN DE DATOS DURANTE LA OPTIMIZACIÓN:
#
#   False = OPTIMIZACIÓN NORMAL (sin perturbación)
#           - Datos históricos reales
#           - Busca mejores parámetros directamente
#           - Más rápido, resultados directos
#
#   True  = OPTIMIZACIÓN CON PERTURBACIÓN
#           - Cada trial usa datos ligeramente diferentes
#           - Valida robustez (detecta overfitting)
#           - Los mejores params funcionan EN PROMEDIO
#
PERTURBACION_ACTIVAR = True  # <--- ¡MODIFICA AQUÍ! (True o False)

# CONFIGURACIÓN DE PERTURBACIÓN (solo aplica si PERTURBACION_ACTIVAR = True)
# La perturbación añade ruido calibrado a los datos para detectar overfitting
PERTURBACION_METHOD = "returns_perturbation"  # MÉTODOS DISPONIBLES:
                                               # "returns_perturbation" = Ruido gaussiano calibrado (RECOMENDADO)
                                               # "block_bootstrap" = Block bootstrap sobre retornos
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
    "OPTUNA_SAMPLER": OPTUNA_SAMPLER,  # PLATEAU, CMA o TPE
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
