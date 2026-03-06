from __future__ import annotations
from modelox.core.exits import (
    DEFAULT_EXIT_TYPE, DEFAULT_EXIT_SL_PCT, DEFAULT_EXIT_TP_PCT,
    DEFAULT_EXIT_TRAIL_ACT_PCT, DEFAULT_EXIT_TRAIL_DIST_PCT, DEFAULT_OPTIMIZE_EXITS,
    DEFAULT_EXIT_SL_PCT_RANGE, DEFAULT_EXIT_TP_PCT_RANGE,
    DEFAULT_EXIT_TRAIL_ACT_PCT_RANGE, DEFAULT_EXIT_TRAIL_DIST_PCT_RANGE,
    DEFAULT_EXIT_TIME_BARS, DEFAULT_EXIT_TIME_BARS_RANGE,
)
from modelox.core.types import normalize_timeframe_to_suffix

# =============================================================================
# CAPITAL (colateral & apalancamiento)
# =============================================================================
SALDO_USADO        = 500    # Colateral por operación ($)
APALANCAMIENTO_MAX = 20    # Apalancamiento máximo

# =============================================================================
# COMBINACIÓN & ESTRATEGIA
# =============================================================================
COMBINACION_A_EJECUTAR = [2]   # IDs de estrategias en modelox/strategies/

# =============================================================================
# OPTIMIZACIÓN (OPTUNA)
# =============================================================================
N_TRIALS                 = 32     # Trials (potencias de 2 para QMC: 32,64,128,256,512,1024,2048…)
OPTUNA_SAMPLER           = "HYBRID" # HYBRID | QMC | TPE | CMA | GT | ML

# =============================================================================
# ACTIVO
# =============================================================================
ACTIVO = "BTC"  # BTC | ETH | GOLD | SILVER | SP500 | NASDAQ | BIST100

# =============================================================================
# RANGO DE FECHAS
# =============================================================================
FECHA_INICIO = "2018-01-01"
FECHA_FIN    = "2025-06-01"

USAR_RANGOS_POR_TRIAL = False  # False = rango fijo | True = sub-ventana aleatoria por trial
MESES_POR_TRIAL       = 7     # Duración de la ventana (solo si USAR_RANGOS_POR_TRIAL=True)

# =============================================================================
# TIMEFRAME
# =============================================================================
TIMEFRAME_BASE = "60"           # 1|5|15|30|60|240|720|1440 (min) o "1m","5m","1h","4h","1d"
TIMEFRAMES     = [TIMEFRAME_BASE]

# =============================================================================
# RÉGIMEN DE MERCADO (EMA 21/200 en 1D)
# =============================================================================
REGIMEN_ACTIVO = True          # True = filtrar por régimen | False = operar siempre
REGIMEN_TIPO   = "BAJISTA"      # ALCISTA (EMA21>EMA200) | BAJISTA (EMA21<EMA200)

# =============================================================================
# PERTURBACIONES (anti-overfitting — Leashed Brownian Motion)
# =============================================================================
ENTRENAMIENTO_ROBUSTO_ACTIVAR = False    # Mutar OHLCV por trial
MAX_DRIFT_PCT                 = 0.15    # Desviación máxima del precio real (±%)
DRIFT_STEP_VOLATILITY         = 0.0050  # Volatilidad browniana por vela
RUIDO_MECHAS_PCT              = 0.10    # Ruido en mechas
RUIDO_VOLUMEN_PCT             = 0.08    # Ruido en volumen


# =============================================================================
# GRÁFICA
# =============================================================================
GRAFICA_RANGO_PERSONALIZADO = False         # False = últimos 2 meses auto | True = fechas manuales
GRAFICA_FECHA_INICIO        = "2022-01-01"  # Solo si GRAFICA_RANGO_PERSONALIZADO=True
GRAFICA_FECHA_FIN           = "2025-06-10"

# =============================================================================
# COMPARATIVA POR PERÍODOS (gráfica en Excel Trial)
# =============================================================================
COMPARATIVA_PERIODOS_ACTIVAR = True    # True = añadir gráfica periódica | False = desactivar
COMPARATIVA_PERIODO          = "1mes"  # Ejemplos: "1mes" | "3 semanas" | "2 meses" | "1 año"
                                       # Unidades: dia/dias | semana/semanas | mes/meses | año/años

# =============================================================================
# SALIDAS & RESULTADOS
# =============================================================================
GENERAR_PLOTS         = True      # Gráfico HTML interactivo
USAR_EXCEL            = True      # Excel con métricas y trades
MAX_ARCHIVOS_GUARDAR  = 5         # Conservar solo los N mejores
TELEGRAM              = True      # Enviar reporte a Telegram
CLEANUP_INTERVAL      = 200       # Liberar RAM cada N trials
PURGE_PYCACHE_ON_EXIT = True      # Limpiar __pycache__ al terminar
CARPETA_DATOS         = "datos"   # Carpeta con <ACTIVO>_ohlcv_1m.<ext>
FORMATO_DATOS         = "feather" # feather | parquet

# =============================================================================
# CAPITAL (resto)
# =============================================================================
SALDO_INICIAL          = 10000.0  # Saldo de inicio ($)
SALDO_OPERATIVO_MAX    = 10000.0  # Tope máximo (sin reinversión)
SALDO_MINIMO_OPERATIVO = 300.0   # Parar si cae por debajo
COMISION_PCT           = 0.0003  # Comisión por operación (0.03%)
COMISION_SIDES         = 2       # 1=solo apertura | 2=apertura+cierre

# =============================================================================
# OPTIMIZACIÓN 2.0
# =============================================================================

OPTUNA_N_JOBS            = 1        # Cores paralelos (-1 = todos)
OPTUNA_SEED              = None     # None = aleatorio | int = reproducible
OPTUNA_CREATE_DATABASE   = True     # Crear IDx.db por estrategia
OPTUNA_RESET_DB_ON_START = True     # Borrar DB al arrancar
GUARDAR_EQUITY_EN_DB     = True     # Guardar curva de equity por trial

# =============================================================================
# INTERNOS — NO MODIFICAR
# =============================================================================
_ACTIVO_ALIASES = {"SP": "SP500", "NDX": "NASDAQ"}

def _normalize_activos(v):
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

ACTIVOS         = _normalize_activos(ACTIVO)
ACTIVO_PRIMARIO = ACTIVOS[0]

def _normalize_timeframes(v):
    def _parse_single(x):
        s = str(x).strip().lower()
        if not s: return None
        if s.endswith("d"):
            try: return int(float(s[:-1]) * 1440)
            except ValueError: return None
        if s.endswith("h"):
            try: return int(float(s[:-1]) * 60)
            except ValueError: return None
        s = s.replace("m", "")
        try: return int(float(s))
        except ValueError: return None

    raw = v if isinstance(v, (list, tuple)) else str(v).split(",")
    out = [m for item in raw if (m := _parse_single(item)) is not None and m > 0]
    return sorted(list(set(out))) or [1]

TIMEFRAMES_NORM = _normalize_timeframes(TIMEFRAMES)
TIMEFRAME       = TIMEFRAMES_NORM[0] if TIMEFRAMES_NORM else 1

def resolve_archivo_data_tf(activo, timeframe=None, *, formato=None, **_kwargs):
    a   = _ACTIVO_ALIASES.get(str(activo).upper().strip(), str(activo).upper().strip()) or "GOLD"
    ext = (formato or FORMATO_DATOS).lower().lstrip('.')
    return f"{CARPETA_DATOS}/{a}_ohlcv_1m.{ext}"

def resolve_archivo_data(activo):
    return resolve_archivo_data_tf(activo)

ARCHIVO_DATA = resolve_archivo_data_tf(ACTIVO_PRIMARIO)


# =============================================================================
# CONFIG — DICCIONARIO EXPORTABLE
# =============================================================================
CONFIG = {
    "ACTIVO": ACTIVO_PRIMARIO, "ACTIVOS": ACTIVOS,
    "CARPETA_DATOS": CARPETA_DATOS, "FORMATO_DATOS": FORMATO_DATOS,
    "FECHA_INICIO": FECHA_INICIO, "FECHA_FIN": FECHA_FIN,
    "USAR_RANGOS_POR_TRIAL": USAR_RANGOS_POR_TRIAL, "MESES_POR_TRIAL": MESES_POR_TRIAL,
    "TIMEFRAME": TIMEFRAME, "TIMEFRAMES": TIMEFRAMES_NORM,
    "SALDO_INICIAL": SALDO_INICIAL, "SALDO_OPERATIVO_MAX": SALDO_OPERATIVO_MAX,
    "SALDO_USADO": SALDO_USADO, "APALANCAMIENTO_MAX": APALANCAMIENTO_MAX,
    "COMISION_PCT": COMISION_PCT, "COMISION_SIDES": COMISION_SIDES,
    "SALDO_MINIMO_OPERATIVO": SALDO_MINIMO_OPERATIVO,
    "N_TRIALS": N_TRIALS, "OPTUNA_N_JOBS": OPTUNA_N_JOBS, "OPTUNA_SEED": OPTUNA_SEED,
    "OPTUNA_CREATE_DATABASE": OPTUNA_CREATE_DATABASE,
    "OPTUNA_RESET_DB_ON_START": OPTUNA_RESET_DB_ON_START,
    "GUARDAR_EQUITY_EN_DB": GUARDAR_EQUITY_EN_DB, "OPTUNA_SAMPLER": OPTUNA_SAMPLER,
    "COMBINACION_A_EJECUTAR": COMBINACION_A_EJECUTAR,
    "EXIT_TYPE": DEFAULT_EXIT_TYPE, "EXIT_SL_PCT": DEFAULT_EXIT_SL_PCT,
    "EXIT_TP_PCT": DEFAULT_EXIT_TP_PCT, "EXIT_TRAIL_ACT_PCT": DEFAULT_EXIT_TRAIL_ACT_PCT,
    "EXIT_TRAIL_DIST_PCT": DEFAULT_EXIT_TRAIL_DIST_PCT, "EXIT_TIME_BARS": DEFAULT_EXIT_TIME_BARS,
    "OPTIMIZAR_SALIDAS": DEFAULT_OPTIMIZE_EXITS,
    "EXIT_SL_PCT_RANGE": DEFAULT_EXIT_SL_PCT_RANGE, "EXIT_TP_PCT_RANGE": DEFAULT_EXIT_TP_PCT_RANGE,
    "EXIT_TRAIL_ACT_PCT_RANGE": DEFAULT_EXIT_TRAIL_ACT_PCT_RANGE,
    "EXIT_TRAIL_DIST_PCT_RANGE": DEFAULT_EXIT_TRAIL_DIST_PCT_RANGE,
    "EXIT_TIME_BARS_RANGE": DEFAULT_EXIT_TIME_BARS_RANGE,
    "MAX_ARCHIVOS_GUARDAR": MAX_ARCHIVOS_GUARDAR, "GENERAR_PLOTS": GENERAR_PLOTS,
    "USAR_EXCEL": USAR_EXCEL, "PURGE_PYCACHE_ON_EXIT": PURGE_PYCACHE_ON_EXIT,
    "GRAFICA_RANGO_PERSONALIZADO": GRAFICA_RANGO_PERSONALIZADO,
    "GRAFICA_FECHA_INICIO": GRAFICA_FECHA_INICIO, "GRAFICA_FECHA_FIN": GRAFICA_FECHA_FIN,
    "COMPARATIVA_PERIODOS_ACTIVAR": COMPARATIVA_PERIODOS_ACTIVAR,
    "COMPARATIVA_PERIODO": COMPARATIVA_PERIODO,
    "TELEGRAM": TELEGRAM,
    "ENTRENAMIENTO_ROBUSTO_ACTIVAR": ENTRENAMIENTO_ROBUSTO_ACTIVAR,
    "MAX_DRIFT_PCT": MAX_DRIFT_PCT, "DRIFT_STEP_VOLATILITY": DRIFT_STEP_VOLATILITY,
    "RUIDO_MECHAS_PCT": RUIDO_MECHAS_PCT, "RUIDO_VOLUMEN_PCT": RUIDO_VOLUMEN_PCT,
    "REGIMEN_ACTIVO": REGIMEN_ACTIVO, "REGIMEN_TIPO": REGIMEN_TIPO,
}
