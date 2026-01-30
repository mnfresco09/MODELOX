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
N_TRIALS = 50000   # NÚMERO DE PRUEBAS (AUMENTAR SI TIENES BUENA CPU)
OPTUNA_N_JOBS = -1      # -1 = USAR TODOS LOS NÚCLEOS DISPONIBLES
OPTUNA_SEED = None      # SEMILLA ALEATORIA (NONE PARA VARIEDAD)
OPTUNA_STORAGE = None   # NONE = EJECUCIÓN EN RAM (MÁS RÁPIDO)

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
# 1.10 SISTEMA DE SCORING UNIFICADO v7.0 (NEIGHBORHOOD ROBUSTNESS)
# ----------------------------------------------------------------------------
# Sistema de evaluación que combina CALIDAD × ACTIVIDAD × ROBUSTEZ
#
# FÓRMULA: Score = Calidad_Raw × Factor_Actividad × Factor_Robustez
#
# COMPONENTE DE CALIDAD (Calidad_Raw):
# - Rango: [0, 1000]
# - Normaliza Sharpe, SQN, ROI, Drawdown usando funciones tanh (sigmoides)
# - Rendimientos decrecientes: mejorar Sharpe de 2→3 da más que 5→6
#
# COMPONENTE DE ACTIVIDAD (Factor_Actividad):
# - Sigmoide logística centrada en 0.25 trades/día
# - Si trades/día < 0.25 → factor cae hacia 0
# - Si trades/día >= 0.5 → factor ~1.0
#
# COMPONENTE DE ROBUSTEZ (Factor_Robustez) - "TECHO DE CRISTAL":
# - SIN TEST: Factor = 0.30 → Score máximo = 300 puntos
#   Para superar 300, el optimizador DEBE buscar configs que activen el test
#
# - CON TEST: Factor = e^(-3.0 × Incertidumbre)
#   - Incertidumbre = dispersión agregada (ROI, Sharpe, Drawdown)
#   - Dispersión 0% → Factor = 1.0 (se liberan los 1000 puntos)
#   - Dispersión ~23% → Factor = 0.5 (~500 puntos)
#   - Dispersión ~40% → Factor = 0.30 (igual que sin test)
#   - Dispersión >40% → Factor < 0.30 (PEOR que sin test)

VECINDARIO_ACTIVAR = True             # ✓ ACTIVADO - Sistema de Neighborhood Fitness

# ═══════════════════════════════════════════════════════════════════════════════
# PARÁMETROS PRINCIPALES DE VECINDARIO (AJUSTAR AQUÍ)
# ═══════════════════════════════════════════════════════════════════════════════
VECINDARIO_N_NEIGHBORS = 10            # Número de vecinos a generar (K) → K+1 backtests por trial
VECINDARIO_MAX_DISPERSION = 0.3     # Dispersión máxima permitida (CV) para aprobar robustez
                                      # v7.0: Límite suavizado a 40% (antes 15%)
                                      # El decaimiento exponencial ya penaliza la alta dispersión
                                      # Más bajo = más estricto (ej: 0.25 = 25%)
                                      # Más alto = más permisivo (ej: 0.50 = 50%)
# ═══════════════════════════════════════════════════════════════════════════════

VECINDARIO_PERTURBATION_STD = 0.05    # Desviación estándar del ruido gaussiano (5%)
VECINDARIO_LAMBDA_PENALTY = 1.5       # Factor de penalización por varianza (Score = μ - λ·σ)
VECINDARIO_SEED = 42                  # Semilla base para reproducibilidad
VECINDARIO_EXCEL = True               # Generar Excel con detalle de vecinos
VECINDARIO_GUARDAR_MEJORES = 5        # Guardar los N mejores trials en Excel

# ═══════════════════════════════════════════════════════════════════════════════
# CRITERIOS PARA HACER TEST DE VECINDARIO (v7.1)
# Un trial SOLO recibe test de robustez si cumple TODOS estos criterios:
# ═══════════════════════════════════════════════════════════════════════════════
VECINDARIO_MIN_TRADES_DIA = 0.25      # Trades/día > 0.25
VECINDARIO_MIN_PROFIT_FACTOR = 1.1    # Profit Factor > 1.1
VECINDARIO_MIN_SHARPE = 1.25          # Sharpe > 1.25


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
    # Vecindario (Neighborhood Fitness Aggregation)
    "VECINDARIO_ACTIVAR": VECINDARIO_ACTIVAR,
    "VECINDARIO_N_NEIGHBORS": VECINDARIO_N_NEIGHBORS,
    "VECINDARIO_MAX_DISPERSION": VECINDARIO_MAX_DISPERSION,
    "VECINDARIO_PERTURBATION_STD": VECINDARIO_PERTURBATION_STD,
    "VECINDARIO_LAMBDA_PENALTY": VECINDARIO_LAMBDA_PENALTY,
    "VECINDARIO_SEED": VECINDARIO_SEED,
    "VECINDARIO_EXCEL": VECINDARIO_EXCEL,
    "VECINDARIO_GUARDAR_MEJORES": VECINDARIO_GUARDAR_MEJORES,
}
