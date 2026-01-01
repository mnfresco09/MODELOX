# general/configuracion.py
"""
MODELOX Configuration - Multi-Asset Timeframe System

Supported Assets:
  - BTC: Bitcoin (crypto)
  - GOLD: Gold (commodity)
  - SP500: S&P 500 Index (equities)
  - NASDAQ: NASDAQ 100 Index (equities)

Nota:
    - Los archivos de datos OHLCV usan el sufijo "_1h".
"""

# ============================================================================
# ASSET SELECTION
# ============================================================================
# Options: 'BTC', 'GOLD', 'SP500', 'NASDAQ'
ACTIVO = "GOLD, BTC, SP500"


def _normalize_activos(v):
    """Normaliza ACTIVO a una lista de strings (upper) sin vacíos.

    Permite:
    - "GOLD" (single)
    - "GOLD,BTC" (comma-separated)
    - ["GOLD", "BTC"] (list/tuple)
    """
    if isinstance(v, (list, tuple)):
        raw = [str(x) for x in v]
    else:
        raw = str(v).split(",")
    out = []
    for a in raw:
        a = a.strip()
        if not a:
            continue
        out.append(a.upper())
    return out or ["GOLD"]


# Multi-asset support: ejecutar.py can iterate ACTIVOS sequentially.
ACTIVOS = _normalize_activos(ACTIVO)
ACTIVO_PRIMARIO = ACTIVOS[0]

# ============================================================================
# DATA PATHS (1-Hour OHLCV Parquet)
# ============================================================================
ARCHIVO_DATA_BTC = "data/ohlcv/BTC_ohlcv_1h.parquet"
ARCHIVO_DATA_GOLD = "data/ohlcv/GOLD_ohlcv_1h.parquet"
ARCHIVO_DATA_SP500 = "data/ohlcv/SP500_ohlcv_1h.parquet"
ARCHIVO_DATA_NASDAQ = "data/ohlcv/NASDAQ_ohlcv_1h.parquet"

# Auto-select data file based on ACTIVO
_DATA_MAP = {
    "BTC": ARCHIVO_DATA_BTC,
    "GOLD": ARCHIVO_DATA_GOLD,
    "SP500": ARCHIVO_DATA_SP500,
    "SP": ARCHIVO_DATA_SP500,  # Alias
    "NASDAQ": ARCHIVO_DATA_NASDAQ,
    "NDX": ARCHIVO_DATA_NASDAQ,  # Alias
}


def resolve_archivo_data(activo: str) -> str:
    return _DATA_MAP.get(str(activo).upper(), ARCHIVO_DATA_GOLD)


# Backward compatibility: single path points to the primary asset.
ARCHIVO_DATA = resolve_archivo_data(ACTIVO_PRIMARIO)

# ============================================================================
# CAPITAL & EXECUTION SETTINGS
# ============================================================================
SALDO_INICIAL = 300
SALDO_OPERATIVO_MAX = 300
APALANCAMIENTO = 50
COMISION_PCT = 0.00043
COMISION_SIDES = 1  # 1 = one-way, 2 = round-trip
SALDO_MINIMO_OPERATIVO = 5  # Hard stop when balance <= this value

# ============================================================================
# POSITION SIZING LIMITS (Per Asset)
# ============================================================================
# QTY_MAX_ACTIVO: Maximum quantity per trade for the selected asset.
# This is the MAXIMUM AUTHORITY - even if leverage allows more, 
# the bot will never exceed this limit.
_QTY_MAX_MAP = {
    "BTC": 0.025,       # BTC max per trade
    "GOLD": 0.5,       # oz Gold max per trade
    "SP500": 0.5,      # contracts SP500 max
    "NASDAQ": 0.15,     # contracts NASDAQ max
}


def resolve_qty_max_activo(activo: str) -> float:
    return float(_QTY_MAX_MAP.get(str(activo).upper(), 3.0))


# Optuna ranges for qty_max_activo per asset (min, max, step)
OPTIMIZAR_QTY_ACTIVO = True
_QTY_MAX_RANGE_MAP = {
    # Ajusta a tu mercado/broker
    "BTC": (0.01, 0.1, 0.01),
    "GOLD": (0.75, 3.5, 0.25),
    "SP500": (0.5, 4.0, 0.25),
    "NASDAQ": (0.025, 0.75, 0.01),
}


def resolve_qty_max_activo_range(activo: str) -> tuple[float, float, float]:
    return tuple(_QTY_MAX_RANGE_MAP.get(str(activo).upper(), (0.01, 5.0, 0.01)))


# Backward compatibility: single qty limit points to the primary asset.
QTY_MAX_ACTIVO = resolve_qty_max_activo(ACTIVO_PRIMARIO)

# ============================================================================
# DATE RANGES (Backtest & Plot)
# ============================================================================
FECHA_INICIO = "2024-01-21"
FECHA_FIN = "2024-11-15"
FECHA_INICIO_PLOT = "2024-01-22"
FECHA_FIN_PLOT = "2024-10-15"

# ============================================================================
# OPTUNA OPTIMIZATION
# ============================================================================
N_TRIALS = 300
    
# ==========================================================================
# EXIT SETTINGS (GLOBAL, ENGINE-OWNED)
# ==========================================================================
# Salidas fijas (no móviles) por ATR calculado en el momento de entrada.
# Se ejecutan intra-vela al nivel exacto si el precio lo toca.
EXIT_ATR_PERIOD = 14
EXIT_SL_ATR = 1.0
EXIT_TP_ATR = 1.0
EXIT_TIME_STOP_BARS = 260

# Optuna: permitir optimizar también la salida global desde configuración
OPTIMIZAR_SALIDAS = True

# Rangos para Optuna (min, max, step)
EXIT_ATR_PERIOD_RANGE = (7, 30, 1)
EXIT_SL_ATR_RANGE = (0.5, 3.0, 0.1)
EXIT_TP_ATR_RANGE = (1.0, 8.0, 0.1)
EXIT_TIME_STOP_BARS_RANGE = (250, 800, 10)
OPTUNA_N_JOBS = 1      # Parallel jobs (1 = sequential)
OPTUNA_SEED = None     # None = random seed each run
OPTUNA_STORAGE = None  # None = in-memory, or SQLite path

# ============================================================================
# STRATEGY SELECTION
# ============================================================================
# Available strategies (see modelox/strategies/):
#   1 = MFI Persistence
#   2 = Kalman Sniper
#   3 = Nadaraya-Watson
#   4 = SSL Hull Hybrid
#   5 = TTM Squeeze
#   6 = Connors RSI(2)
#   7 = Laguerre RSI
#   9 = KAMA Trend
#  10 = Elder Ray
#  13 = Lorentzian ML (Machine Learning)
#
# Single strategy:  COMBINACION_A_EJECUTAR = 7
# Multiple:         COMBINACION_A_EJECUTAR = [3, 4, 7, 9, 10]
# All available:    COMBINACION_A_EJECUTAR = "all"
#
COMBINACION_A_EJECUTAR = [6]

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================
MAX_ARCHIVOS_GUARDAR = 5  # Keep only top N results
GENERAR_PLOTS = True      # Generate interactive HTML charts
USAR_EXCEL = True         # Generate Excel reports

# ============================================================================
# UNIFIED CONFIG DICT (For Backward Compatibility)
# ============================================================================
CONFIG = {
    # Backward compatibility: expose one ACTIVO as the primary one.
    "ACTIVO": ACTIVO_PRIMARIO,
    # New: list of assets to run sequentially.
    "ACTIVOS": ACTIVOS,
    "TIMEFRAME": "1h",
    "SALDO_INICIAL": SALDO_INICIAL,
    "SALDO_OPERATIVO_MAX": SALDO_OPERATIVO_MAX,
    "APALANCAMIENTO": APALANCAMIENTO,
    "COMISION_PCT": COMISION_PCT,
    "COMISION_SIDES": COMISION_SIDES,
    "SALDO_MINIMO_OPERATIVO": SALDO_MINIMO_OPERATIVO,
    "QTY_MAX_ACTIVO": QTY_MAX_ACTIVO,
    "OPTIMIZAR_QTY_ACTIVO": OPTIMIZAR_QTY_ACTIVO,
    # Per-asset caps and Optuna ranges for qty_max_activo
    "QTY_MAX_MAP": _QTY_MAX_MAP,
    "QTY_MAX_RANGE_MAP": _QTY_MAX_RANGE_MAP,
    "GENERAR_PLOTS": GENERAR_PLOTS,
    "USAR_EXCEL": USAR_EXCEL,
    "USAR_DATA_LIMPIA": False,
    "OPTUNA_N_JOBS": OPTUNA_N_JOBS,
    "OPTUNA_SEED": OPTUNA_SEED,
    "OPTUNA_STORAGE": OPTUNA_STORAGE,

    # Global exits (engine)
    "EXIT_ATR_PERIOD": EXIT_ATR_PERIOD,
    "EXIT_SL_ATR": EXIT_SL_ATR,
    "EXIT_TP_ATR": EXIT_TP_ATR,
    "EXIT_TIME_STOP_BARS": EXIT_TIME_STOP_BARS,

    # Optuna exits
    "OPTIMIZAR_SALIDAS": OPTIMIZAR_SALIDAS,
    "EXIT_ATR_PERIOD_RANGE": EXIT_ATR_PERIOD_RANGE,
    "EXIT_SL_ATR_RANGE": EXIT_SL_ATR_RANGE,
    "EXIT_TP_ATR_RANGE": EXIT_TP_ATR_RANGE,
    "EXIT_TIME_STOP_BARS_RANGE": EXIT_TIME_STOP_BARS_RANGE,
}
