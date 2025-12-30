# general/configuracion.py
"""
MODELOX Configuration - Multi-Asset 5-Minute Timeframe System

Supported Assets:
  - BTC: Bitcoin (crypto)
  - GOLD: Gold (commodity)
  - SP500: S&P 500 Index (equities)
  - NASDAQ: NASDAQ 100 Index (equities)

All data files are in 5-minute OHLCV format.
"""

# ============================================================================
# ASSET SELECTION
# ============================================================================
# Options: 'BTC', 'GOLD', 'SP500', 'NASDAQ'
ACTIVO = "GOLD"

# ============================================================================
# DATA PATHS (5-Minute OHLCV Parquet)
# ============================================================================
ARCHIVO_DATA_BTC = "data/ohlcv/BTC_ohlcv_5m.parquet"
ARCHIVO_DATA_GOLD = "data/ohlcv/GOLD_ohlcv_5m.parquet"
ARCHIVO_DATA_SP500 = "data/ohlcv/SP500_ohlcv_5m.parquet"
ARCHIVO_DATA_NASDAQ = "data/ohlcv/NASDAQ_ohlcv_5m.parquet"

# Auto-select data file based on ACTIVO
_DATA_MAP = {
    "BTC": ARCHIVO_DATA_BTC,
    "GOLD": ARCHIVO_DATA_GOLD,
    "SP500": ARCHIVO_DATA_SP500,
    "SP": ARCHIVO_DATA_SP500,  # Alias
    "NASDAQ": ARCHIVO_DATA_NASDAQ,
    "NDX": ARCHIVO_DATA_NASDAQ,  # Alias
}
ARCHIVO_DATA = _DATA_MAP.get(ACTIVO.upper(), ARCHIVO_DATA_GOLD)

# ============================================================================
# CAPITAL & EXECUTION SETTINGS
# ============================================================================
SALDO_INICIAL = 300
SALDO_OPERATIVO_MAX = 300
APALANCAMIENTO = 50
COMISION_PCT = 0.0005
COMISION_SIDES = 1  # 1 = one-way, 2 = round-trip
SALDO_MINIMO_OPERATIVO = 5  # Hard stop when balance <= this value

# ============================================================================
# POSITION SIZING LIMITS (Per Asset)
# ============================================================================
# QTY_MAX_ACTIVO: Maximum quantity per trade for the selected asset.
# This is the MAXIMUM AUTHORITY - even if leverage allows more, 
# the bot will never exceed this limit.
_QTY_MAX_MAP = {
    "BTC": 0.07,       # 0.07 BTC max per trade
    "GOLD": 2.5,       # 2.5 oz Gold max per trade
    "SP500": 2.5,      # 3 contracts SP500 max
    "NASDAQ": 0.2,     # 3 contracts NASDAQ max
}
QTY_MAX_ACTIVO = _QTY_MAX_MAP.get(ACTIVO.upper(), 3.0)

# ============================================================================
# DATE RANGES (Backtest & Plot)
# ============================================================================
FECHA_INICIO = "2024-11-21"
FECHA_FIN = "2025-11-15"
FECHA_INICIO_PLOT = "2024-11-21"
FECHA_FIN_PLOT = "2024-12-15"

# ============================================================================
# OPTUNA OPTIMIZATION
# ============================================================================
N_TRIALS = 15
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
COMBINACION_A_EJECUTAR = [1]

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
    "ACTIVO": ACTIVO,
    "TIMEFRAME": "5m",
    "SALDO_INICIAL": SALDO_INICIAL,
    "SALDO_OPERATIVO_MAX": SALDO_OPERATIVO_MAX,
    "APALANCAMIENTO": APALANCAMIENTO,
    "COMISION_PCT": COMISION_PCT,
    "COMISION_SIDES": COMISION_SIDES,
    "SALDO_MINIMO_OPERATIVO": SALDO_MINIMO_OPERATIVO,
    "QTY_MAX_ACTIVO": QTY_MAX_ACTIVO,
    "GENERAR_PLOTS": GENERAR_PLOTS,
    "USAR_EXCEL": USAR_EXCEL,
    "USAR_DATA_LIMPIA": False,
    "OPTUNA_N_JOBS": OPTUNA_N_JOBS,
    "OPTUNA_SEED": OPTUNA_SEED,
    "OPTUNA_STORAGE": OPTUNA_STORAGE,
}
