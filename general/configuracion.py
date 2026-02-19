"""
================================================================================
GENERAL/CONFIGURACION.PY — CONFIGURACIÓN MAESTRA DEL SISTEMA
================================================================================

PROPÓSITO:
    Archivo ÚNICO para configurar backtests, optimizaciones y trading real.
    Centraliza parámetros de activo, fechas, capital y estrategias.

CONTENIDO:
     1. EJECUCIÓN                 — Estrategias a correr (IDs)
     2. DATOS                     — Activo, timeframe base, rango fechas
     3. OPTIMIZACIÓN (OPTUNA)     — Trials, sampler, perturbación
     4. CAPITAL & RIESGO          — Saldo inicial, comisiones, apalancamiento
     5. SALIDAS & RESULTADOS      — Plots, Excel, limpieza

USO:
    1. Ajustar parámetros en este archivo.
    2. Ejecutar `python ejecutar.py` para backtest/optimización.
    3. Resultados en carpeta `resultados/`.

NOTAS:
    - Los datos se resamplean automáticamente desde 1m.
    - Optuna busca combinaciones óptimas si N_TRIALS > 1.

================================================================================
"""

from __future__ import annotations

# IMPORTAMOS CONSTANTES DE MODELOX
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
# 1.0 ESTRATEGIAS A EJECUTAR
# ----------------------------------------------------------------------------
# Los IDs corresponden a los archivos en  modelox/strategies/
COMBINACION_A_EJECUTAR = [7]
# ----------------------------------------------------------------------------
# 1.1 SELECCIÓN DE ACTIVOS Y DATOS
# ----------------------------------------------------------------------------
# ¿Con qué activo quieres hacer el backtest?
#
# Activos disponibles:
#   "BTC"     → Bitcoin (Binance, desde 2017)
#   "ETH"     → Ethereum (Binance, desde 2017)
#   "GOLD"    → Oro XAU/USD (desde 2009)
#   "SILVER"  → Plata XAG/USD (desde 2009)
#   "SP500"   → S&P 500 (desde 2010)
#   "NASDAQ"  → Nasdaq 100 (desde 2010)
#   "BIST100" → Borsa Istanbul (desde 2024, solo 1h)
#
# También acepta varios:  "BTC,GOLD"  →  ejecuta ambos en secuencia
ACTIVO = "BTC"

# ¿Dónde están los archivos de datos?
# El sistema busca:  <CARPETA_DATOS>/<ACTIVO>_ohlcv_1m.<FORMATO_DATOS>
# Ejemplo:           nuevos_datos/BTC_ohlcv_1m.feather
#
# Carpetas disponibles en tu proyecto:
#   "nuevos_datos"  → Datos frescos descargados (TODOS los activos, 1m)
#   "data/ohlcv"    → Datos antiguos (solo BTC, GOLD, SP500, NASDAQ)
CARPETA_DATOS = "datos"

# Formato de archivo:
#   "feather"  → Rápido en lectura, archivos más grandes (~120 MB)
#   "parquet"  → Más comprimido (~50 MB), ligeramente más lento
FORMATO_DATOS = "feather"

# ----------------------------------------------------------------------------
# 1.2 MOTOR DE OPTIMIZACIÓN (OPTUNA)
# ----------------------------------------------------------------------------
# Optuna busca los mejores parámetros para tu estrategia.
# Cada "trial" es un backtest con parámetros diferentes.
#
# N_TRIALS = cuántas combinaciones probar.
#   - 3-10     → prueba rápida (minutos)
#   - 50-100   → exploración decente (horas)
#   - 500-1000 → búsqueda exhaustiva (muchas horas)
N_TRIALS = 50

# OPTUNA_N_JOBS = cuántos cores de CPU usar en paralelo.
#   -1 = todos los disponibles (recomendado)
#    1 = secuencial (más lento, menos RAM)
OPTUNA_N_JOBS = -1

OPTUNA_SEED = None      # None = resultados diferentes cada vez
                        # 42 (o cualquier int) = reproducible
OPTUNA_STORAGE = None   # None = todo en RAM (más rápido)

# ALGORITMO DE BÚSQUEDA (SAMPLER)
#   "CMA"  → CMA-ES: Bueno para parámetros continuos (precios, %).
#            Converge rápido pero necesita ≥10 trials.
#   "TPE"  → Tree Parzen: Más versátil, bueno con pocos trials.
#            Mejor si mezclas parámetros categóricos y numéricos.
#   "GT"   → GT-Score: Anti-overfitting con topología k-NN.
#   "ML"   → ML-Forest: Analista Senior con Machine Learning.
#            Fase 1: Explora 500 trials. Fase 2: Random Forest guía.
#            Score 0-100. Dos cerebros: Codicioso + Paranoico.
#   "QMC"  → Quasi-Monte Carlo: Secuencias Sobol de baja discrepancia.
#            Cobertura EQUIDISTANTE del espacio de parámetros.
#            NO usa el score para guiar — pura matemática.
#            Ideal para exploración exhaustiva sin sesgo ni overfitting.
OPTUNA_SAMPLER = "QMC"

# ----------------------------------------------------------------------------
# 1.3 GESTIÓN DE CAPITAL Y COSTES
# ----------------------------------------------------------------------------
# Simula tu cuenta de trading real.
#
SALDO_INICIAL = 1000.0          # Con cuánto $ arrancas
SALDO_OPERATIVO_MAX = 1000.0    # Tope máximo de saldo (no reinviertes ganancias)
SALDO_MINIMO_OPERATIVO = 300.0  # Si el saldo baja de aquí, deja de operar

# Comisiones (ajusta según tu exchange):
#   Binance Spot:  0.001  (0.1%)
#   Binance Futuros Taker: 0.0005 (0.05%)
#   Bybit Taker:   0.0006 (0.06%)
COMISION_PCT = 0.0003

# ¿Cobra comisión al abrir, al cerrar, o ambas?
#   1 = solo al abrir la posición
#   2 = al abrir Y al cerrar (más realista)
COMISION_SIDES = 2

# ----------------------------------------------------------------------------
# 1.4 TAMAÑO DE POSICIÓN (POSITION SIZING)
# ----------------------------------------------------------------------------
# Cuánto arriesgas por operación.
#
# La cantidad (qty) se calcula DINÁMICAMENTE en cada trade:
#   qty = (SALDO_USADO × APALANCAMIENTO_MAX) / precio_entrada
#
# Ejemplo con BTC a 100.000$:
#   SALDO_USADO=100, APALANCAMIENTO_MAX=60
#   → Volumen: 100 × 60 = 6.000$
#   → qty = 6.000 / 100.000 = 0.06 BTC
#
# Si BTC baja a 50.000$:
#   → qty = 6.000 / 50.000 = 0.12 BTC (más unidades, mismo volumen)
#
SALDO_USADO = 60.0           # Colateral fijo por operación ($)
APALANCAMIENTO_MAX = 35      # Apalancamiento máximo permitido

# ----------------------------------------------------------------------------
# 2.1 TIMEFRAME BASE (RESOLUCIÓN DE OPERACIÓN)
# ----------------------------------------------------------------------------
# ¿En qué temporalidad opera tu estrategia?
#
# Los datos originales están en 1m. El sistema los agrupa (resamplea)
# automáticamente al timeframe que elijas aquí.
#
#   "1m"  → Cada vela = 1 minuto  (máxima resolución, más lento)
#   "5m"  → Cada vela = 5 minutos (buen equilibrio velocidad/detalle)
#   "15m" → Cada vela = 15 minutos
#   "30m" → Cada vela = 30 minutos
#   "1h"  → Cada vela = 1 hora    (rápido, menos detalle)
#   "4h"  → Cada vela = 4 horas   (swing trading)
#   "12h" → Cada vela = 12 horas  (position trading)
#   "1d"  → Cada vela = 1 día     (daily)
#
# ⚡ Si usas "12h" o "1d", las gráficas se generan automáticamente
#    con el rango COMPLETO de datos (ej: 2017 → 2025).
#
# También acepta enteros (minutos): 1, 5, 15, 30, 60, 120, 240, 720, 1440
TIMEFRAME_BASE = "5"
TIMEFRAMES = [TIMEFRAME_BASE]  # No tocar — se deriva de TIMEFRAME_BASE

# ----------------------------------------------------------------------------
# 2.2 RANGO DE FECHAS
# ----------------------------------------------------------------------------
# ¿Sobre qué periodo histórico quieres hacer el backtest?
#
# Formato: "AAAA-MM-DD"
# El sistema filtra los datos entre estas dos fechas.
# Si la fecha excede los datos disponibles, usa hasta donde lleguen.
#
# Rangos disponibles en tus datos (nuevos_datos/*.feather):
#   BTC/ETH  : 2017-08-17 → 2026-02-08
#   GOLD     : 2009-03-15 → 2025-12-31
#   SILVER   : 2009-05-03 → 2025-12-31
#   SP500    : 2010-11-14 → 2025-12-31
#   NASDAQ   : 2010-11-14 → 2025-12-31
#   BIST100  : 2024-02-12 → 2026-02-09 (solo 1h)
#
FECHA_INICIO = "2021-09-01"
FECHA_FIN    = "2025-08-01"

# ── MODO DE USO DEL RANGO ──
#
# USAR_RANGOS_POR_TRIAL controla CÓMO se usan las fechas de arriba:
#
#   False (por defecto):
#     → Cada trial de Optuna usa TODO el rango FECHA_INICIO → FECHA_FIN.
#     → Ideal para optimizar sobre un periodo concreto que tú eliges.
#     → Ejemplo: pones 2020-01-01 a 2024-01-01 y TODOS los trials
#       hacen backtest sobre esos 4 años completos.
#
#   True (anti-overfitting):
#     → Cada trial usa un trozo ALEATORIO de MESES_POR_TRIAL meses
#       dentro de FECHA_INICIO → FECHA_FIN.
#     → Evita que Optuna memorice un periodo concreto.
#     → Ejemplo: con MESES_POR_TRIAL=6, el trial 1 puede probar
#       Mar-2021 a Sep-2021, el trial 2 puede probar Nov-2023 a May-2024, etc.
#
USAR_RANGOS_POR_TRIAL = False
MESES_POR_TRIAL = 6  # Solo aplica si USAR_RANGOS_POR_TRIAL = True

# Fechas para el gráfico detallado de trades (zoom visual).
# Esto NO afecta al backtest, solo al gráfico HTML que se genera.
FECHA_INICIO_PLOT = "2025-01-01"
FECHA_FIN_PLOT = "2025-02-10"

# ── PERTURBACIÓN DE DATOS (anti-overfitting) ──
#
# Añade ruido microscópico a los precios para que Optuna no memorice
# patrones exactos del histórico. La estrategia resultante es más robusta.
#
#   True  = Activada (recomendado con muchos trials)
#   False = Datos originales sin modificar
PERTURBACION_ACTIVAR = False

# Intensidad del ruido: 0.1 = suave, 0.5 = moderado, 1.0 = agresivo.
# 0.5 significa que el ruido es ~50% de la volatilidad real del activo.
PERTURBACION_NOISE_SCALE = 0.65

PERTURBACION_SEED = 42               # Semilla (int = reproducible, None = aleatorio)
PERTURBACION_VERIFY = True           # Verificar que OHLCV sigue siendo coherente

# ----------------------------------------------------------------------------
# 2.5 SALIDAS Y RESULTADOS
# ----------------------------------------------------------------------------
# ¿Qué archivos generar al terminar?
#
GENERAR_PLOTS = True       # Gráfico HTML interactivo con velas + trades
USAR_EXCEL = True          # Excel con resumen de métricas y lista de trades
MAX_ARCHIVOS_GUARDAR = 5   # Guarda solo los N mejores resultados (borra los peores)

# Los resultados se guardan en:
#   resultados/<ESTRATEGIA>/FIXED/<ACTIVO>/EXCEL/
#   resultados/<ESTRATEGIA>/FIXED/<ACTIVO>/GRAFICA/

TELEGRAM = True            # Enviar reportes a Telegram (True) o no (False)

# ── Gráfico detallado ──
# El gráfico HTML muestra un subrango del backtest para ver trades de cerca.
PLOT_MESES_DURACION = 2         # Cuántos meses mostrar en el gráfico
PLOT_UBICACION_ALEATORIA = False # False = desde el inicio, True = zona aleatoria

# ── Mantenimiento ──
CLEANUP_INTERVAL = 200          # Liberar RAM cada N trials (50-200)
PURGE_PYCACHE_ON_EXIT = True    # Limpiar cachés de Python al terminar


# =============================================================================
# [SECCIÓN 3] FUNCIONES INTERNAS (NO MODIFICAR)
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
    """Convierte cualquier formato de timeframe a minutos (int)."""
    def _parse_single(x):
        s = str(x).strip().lower()
        if not s:
            return None
        if s.endswith("d"):
            try:
                return int(float(s[:-1]) * 1440)
            except ValueError:
                return None
        if s.endswith("h"):
            try:
                return int(float(s[:-1]) * 60)
            except ValueError:
                return None
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

    return sorted(list(set(out))) or [1]  # Default: 1m

TIMEFRAMES_NORM = _normalize_timeframes(TIMEFRAMES)
TIMEFRAME = TIMEFRAMES_NORM[0] if TIMEFRAMES_NORM else 1  # Default: 1m

def resolve_archivo_data_tf(activo, timeframe=None, *, formato=None, **_kwargs):
    """Genera la ruta correcta del archivo de datos.
    
    Usa CARPETA_DATOS y FORMATO_DATOS de la configuración.
    SIEMPRE apunta al archivo de 1m (el resampleo se hace después).
    
    Resultado:  <CARPETA_DATOS>/<ACTIVO>_ohlcv_1m.<FORMATO>
    Ejemplo:    nuevos_datos/BTC_ohlcv_1m.feather
    """
    a = _ACTIVO_ALIASES.get(str(activo).upper().strip(), str(activo).upper().strip()) or "GOLD"
    ext = (formato or FORMATO_DATOS).lower().lstrip('.')
    return f"{CARPETA_DATOS}/{a}_ohlcv_1m.{ext}"

def resolve_archivo_data(activo):
    return resolve_archivo_data_tf(activo)

ARCHIVO_DATA = resolve_archivo_data_tf(ACTIVO_PRIMARIO)


# =============================================================================
# [SECCIÓN 4] DICCIONARIO UNIFICADO EXPORTABLE
# =============================================================================

CONFIG = {
    # Activos
    "ACTIVO": ACTIVO_PRIMARIO,
    "ACTIVOS": ACTIVOS,
    "CARPETA_DATOS": CARPETA_DATOS,
    "FORMATO_DATOS": FORMATO_DATOS,
    # Rangos de fechas
    "FECHA_INICIO": FECHA_INICIO,
    "FECHA_FIN": FECHA_FIN,
    # Rangos por trial
    "USAR_RANGOS_POR_TRIAL": USAR_RANGOS_POR_TRIAL,
    "MESES_POR_TRIAL": MESES_POR_TRIAL,
    # Timeframes
    "TIMEFRAME": TIMEFRAME,
    "TIMEFRAMES": TIMEFRAMES_NORM,
    # Capital
    "SALDO_INICIAL": SALDO_INICIAL,
    "SALDO_OPERATIVO_MAX": SALDO_OPERATIVO_MAX,
    "SALDO_USADO": SALDO_USADO,
    "APALANCAMIENTO_MAX": APALANCAMIENTO_MAX,
    "COMISION_PCT": COMISION_PCT,
    "COMISION_SIDES": COMISION_SIDES,
    "SALDO_MINIMO_OPERATIVO": SALDO_MINIMO_OPERATIVO,
    "RIESGO_POR_TRADE_PCT": 0.10,  # Legacy (no usado con SALDO_USADO fijo)
    # Optuna
    "N_TRIALS": N_TRIALS,
    "OPTUNA_N_JOBS": OPTUNA_N_JOBS,
    "OPTUNA_SEED": OPTUNA_SEED,
    "OPTUNA_STORAGE": OPTUNA_STORAGE,
    "OPTUNA_SAMPLER": OPTUNA_SAMPLER,
    "COMBINACION_A_EJECUTAR": COMBINACION_A_EJECUTAR,
    # Exits (valores de modelox/core/exits.py)
    "EXIT_TYPE": DEFAULT_EXIT_TYPE,
    "EXIT_SL_PCT": DEFAULT_EXIT_SL_PCT,
    "EXIT_TP_PCT": DEFAULT_EXIT_TP_PCT,
    "EXIT_TRAIL_ACT_PCT": DEFAULT_EXIT_TRAIL_ACT_PCT,
    "EXIT_TRAIL_DIST_PCT": DEFAULT_EXIT_TRAIL_DIST_PCT,
    "OPTIMIZAR_SALIDAS": DEFAULT_OPTIMIZE_EXITS,
    "EXIT_SL_PCT_RANGE": DEFAULT_EXIT_SL_PCT_RANGE,
    "EXIT_TP_PCT_RANGE": DEFAULT_EXIT_TP_PCT_RANGE,
    "EXIT_TRAIL_ACT_PCT_RANGE": DEFAULT_EXIT_TRAIL_ACT_PCT_RANGE,
    "EXIT_TRAIL_DIST_PCT_RANGE": DEFAULT_EXIT_TRAIL_DIST_PCT_RANGE,
    # Resultados
    "MAX_ARCHIVOS_GUARDAR": MAX_ARCHIVOS_GUARDAR,
    "GENERAR_PLOTS": GENERAR_PLOTS,
    "USAR_EXCEL": USAR_EXCEL,
    "PURGE_PYCACHE_ON_EXIT": PURGE_PYCACHE_ON_EXIT,
    # Gráficos (subrango)
    "PLOT_MESES_DURACION": PLOT_MESES_DURACION,
    "PLOT_UBICACION_ALEATORIA": PLOT_UBICACION_ALEATORIA,
    # Perturbación
    "PERTURBACION_ACTIVAR": PERTURBACION_ACTIVAR,
    "PERTURBACION_NOISE_SCALE": PERTURBACION_NOISE_SCALE,
    "PERTURBACION_SEED": PERTURBACION_SEED,
    "PERTURBACION_VERIFY": PERTURBACION_VERIFY,
    "TELEGRAM": TELEGRAM,
}
