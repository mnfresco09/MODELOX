"""
================================================================================
IA/CONFIG.PY — CONFIGURACIÓN CENTRALIZADA DEL PIPELINE IA BTC
================================================================================
Modifica aquí todos los parámetros del sistema.
Ejecutar: python IA/main.py
================================================================================
"""

from pathlib import Path

# ─── Rutas ────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.parent.resolve()
DATA_PATH  = ROOT_DIR / "datos" / "BTC_ohlcv_1m.feather"
IA_DIR     = Path(__file__).parent.resolve()
MODELS_DIR = IA_DIR / "models"
RESULTS_DIR= IA_DIR / "resultados"

MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ─── Datos y Preprocesamiento ─────────────────────────────────────────────────
TIMEFRAME          = "1m"
ZSCORE_WINDOW_FAST = 20      # Ventana rápida rolling z-score
ZSCORE_WINDOW_SLOW = 50      # Ventana lenta rolling z-score
LOOKBACK           = 24      # Timesteps de lookback para el modelo (24 candles 1m)
STRIDE             = 10      # Usar cada N candle para generar secuencias (eficiencia)
N_FEATURES         = 6       # Número de features por timestep

# ─── Etiquetado (TP/SL en USD) ────────────────────────────────────────────────
TP_USD             = 500.0   # Take Profit: precio_entrada + 500 USD (LONG)
SL_USD             = 500.0   # Stop Loss:   precio_entrada - 500 USD (LONG)
MAX_FORWARD_CANDLES= 480     # Máximo de candles adelante para buscar TP/SL (8h a 1m)
#  label=1.0 → LONG (TP long hit first)  |  label=0.0 → SHORT (SL long hit first)
#  label=-1.0 → sin etiqueta (excluido del entrenamiento)

# ─── Arquitectura del Modelo GRU ─────────────────────────────────────────────
GRU_UNITS    = 64    # Unidades por capa GRU
N_GRU_LAYERS = 2     # Número de capas GRU apiladas
DROPOUT      = 0.4   # Dropout intradiario (0.4 recomendado para 1m)
FC_UNITS     = 32    # Unidades de la capa densa intermedia

# ─── Función de Pérdida Asimétrica ────────────────────────────────────────────
ALPHA_LOSS = 2.0     # Factor de penalización: α>1 penaliza errores de dirección

# ─── Entrenamiento ────────────────────────────────────────────────────────────
LEARNING_RATE          = 1e-4   # Adam LR inicial
BATCH_SIZE             = 32     # Batch size
MAX_EPOCHS             = 200    # Épocas máximas
PATIENCE               = 15     # Early stopping patience
CLASS_WEIGHT_MINORITY  = 3.0    # Peso extra para la clase minoritaria

# ─── Señal de Entrada ─────────────────────────────────────────────────────────
PROB_THRESHOLD     = 0.70   # P > umbral → LONG
SHORT_THRESHOLD    = 0.30   # P < umbral → SHORT  (P(SHORT) > 0.70)
ENTROPY_THRESHOLD  = 0.85   # Rechazar si entropía binaria normalizada > este valor
ANOMALY_PERCENTILE = 95     # Percentil para detector de anomalías (precio inusual)
ANOMALY_WINDOW     = 1440   # Ventana histórica para anomalías (1440 min = 1 día)

# ─── Validación Walk-Forward ──────────────────────────────────────────────────
TRAIN_YEARS  = 1.5   # Años de entrenamiento por fold
EMBARGO_DAYS = 60    # Días de embargo entre train y val (elimina autocorrelación)
VAL_YEARS    = 1.0   # Años de validación por fold
STEP_MONTHS  = 12    # Desplazamiento entre folds (meses)
MIN_FOLDS    = 3     # Mínimo de folds a evaluar

# ─── Optimización Bayesiana (Optuna) ─────────────────────────────────────────
N_OPTUNA_TRIALS      = 25    # Número de trials
OPTUNA_TIMEOUT       = 3600  # Timeout (segundos)
OPTUNA_GRU_UNITS     = [32, 64, 128, 256]
OPTUNA_BATCH_SIZES   = [16, 32, 64, 128]
OPTUNA_LR_RANGE      = (5e-5, 5e-4)
OPTUNA_DROPOUT_RANGE = (0.1, 0.5)
OPTUNA_LOOKBACK      = [12, 24, 48, 72]

# ─── Capital y Riesgo ─────────────────────────────────────────────────────────
SALDO_INICIAL  = 1000.0  # Capital inicial USD
APALANCAMIENTO = 25      # Apalancamiento
SALDO_USADO    = 100.0   # Colateral por operación USD → posición = $2,500
COMISION_PCT   = 0.0003  # Comisión taker (0.03%)
COMISION_SIDES = 2       # Cobro apertura + cierre

# ─── Modo Rápido (Para pruebas) ───────────────────────────────────────────────
QUICK_MODE       = False         # True = menos datos, menos épocas
QUICK_DATE_START = "2022-01-01"
QUICK_DATE_END   = "2025-01-01"
QUICK_MAX_EPOCHS = 30
QUICK_STRIDE     = 25
QUICK_TRIALS     = 5
