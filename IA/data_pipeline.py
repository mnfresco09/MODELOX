"""
================================================================================
IA/DATA_PIPELINE.PY — PREPROCESAMIENTO Y ETIQUETADO DE DATOS
================================================================================
Pipeline completo:
  1. Carga datos BTC 1m (feather)
  2. Log-returns para open/close/high/low
  3. Normalización volumen (log + rolling z-score)
  4. Rolling Z-Score (ventana 20 y 50) → evita data leakage
  5. Forward-fill + eliminación de NaN tras warmup
  6. Etiquetado TP/SL con numba (CPU-vectorizado)
  7. Creación de secuencias (X, y) con lookback window
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd

# Numba disponible en el proyecto
try:
    from numba import njit
    _NUMBA_OK = True
except ImportError:
    _NUMBA_OK = False

sys.path.insert(0, str(Path(__file__).parent.parent))
from IA.config import (
    DATA_PATH, ZSCORE_WINDOW_FAST, ZSCORE_WINDOW_SLOW,
    LOOKBACK, STRIDE, N_FEATURES,
    TP_USD, SL_USD, MAX_FORWARD_CANDLES,
    QUICK_MODE, QUICK_DATE_START, QUICK_DATE_END,
    QUICK_STRIDE,
)


# =============================================================================
# 1. CARGA DE DATOS
# =============================================================================

def load_btc_data(
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Carga datos BTC 1m desde feather.
    Devuelve DataFrame con índice datetime UTC.
    """
    path = DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"No se encontró: {path}")

    df = pd.read_feather(str(path))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.set_index("timestamp")

    # Filtro de fechas
    if QUICK_MODE and date_start is None:
        date_start = QUICK_DATE_START
        date_end   = QUICK_DATE_END

    if date_start:
        df = df[df.index >= pd.Timestamp(date_start, tz="UTC")]
    if date_end:
        df = df[df.index <= pd.Timestamp(date_end,   tz="UTC")]

    # Rellenar huecos con forward-fill
    df = df.ffill()
    df = df.dropna()

    if verbose:
        print(f"  Datos cargados: {len(df):,} velas | {df.index[0]} → {df.index[-1]}")

    return df


# =============================================================================
# 2. ROLLING Z-SCORE (SIN DATA LEAKAGE)
# =============================================================================

def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Z-score sobre ventana deslizante hacia atrás.
    Solo usa datos pasados → NO hay data leakage.

    z_t = (x_t - mean_{t-window:t-1}) / std_{t-window:t-1}
    """
    roll  = series.rolling(window=window, min_periods=window)
    mu    = roll.mean()
    sigma = roll.std(ddof=1)
    return (series - mu) / (sigma + 1e-10)


# =============================================================================
# 3. INGENIERÍA DE FEATURES
# =============================================================================

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula 6 features por vela, todas normalizadas con rolling z-score:

      Feature 1: close_ret_z  — log-return de close/prev_close (z-score w=20)
      Feature 2: open_ret_z   — log-return de open/prev_close  (z-score w=20)
      Feature 3: high_ret_z   — log(high/close)                (z-score w=20)
      Feature 4: low_ret_z    — log(close/low) → siempre ≥ 0   (z-score w=20)
      Feature 5: vol_z        — log(volume) rolling z-score     (w=20)
      Feature 6: range_z      — log(high/low) = rango rel.      (z-score w=20)

    No se agregan indicadores técnicos: el modelo los aprende solo.
    """
    out = pd.DataFrame(index=df.index)

    # ── Log-returns ─────────────────────────────────────────────────
    close    = df["close"]
    prev_cls = close.shift(1)

    close_ret = np.log(close / prev_cls)
    open_ret  = np.log(df["open"] / prev_cls)
    high_ret  = np.log(df["high"] / close)
    low_ret   = np.log(close / df["low"])          # positivo (close ≥ low)
    range_ret = np.log(df["high"] / df["low"])     # amplitud relativa

    # ── Volumen: log-transform ───────────────────────────────────────
    log_vol   = np.log(df["volume"] + 1.0)

    # ── Rolling Z-Score (ventana 20) ────────────────────────────────
    w = ZSCORE_WINDOW_FAST
    out["close_ret_z"] = rolling_zscore(close_ret, w)
    out["open_ret_z"]  = rolling_zscore(open_ret,  w)
    out["high_ret_z"]  = rolling_zscore(high_ret,  w)
    out["low_ret_z"]   = rolling_zscore(low_ret,   w)
    out["vol_z"]       = rolling_zscore(log_vol,   w)
    out["range_z"]     = rolling_zscore(range_ret, w)

    # ── Guardar precios crudos para el backtest ─────────────────────
    out["close_raw"] = df["close"].values
    out["high_raw"]  = df["high"].values
    out["low_raw"]   = df["low"].values
    out["open_raw"]  = df["open"].values

    # ── Eliminar NaN del período de warmup ──────────────────────────
    warmup = max(ZSCORE_WINDOW_SLOW, ZSCORE_WINDOW_FAST) + 1
    out = out.iloc[warmup:]
    out = out.dropna()

    return out


# =============================================================================
# 4. ETIQUETADO TP/SL CON NUMBA (VECTORIZADO)
# =============================================================================

if _NUMBA_OK:
    @njit(cache=True, fastmath=True)
    def _label_tpsl_numba(
        high:        np.ndarray,
        low:         np.ndarray,
        close:       np.ndarray,
        tp_usd:      float,
        sl_usd:      float,
        max_forward: int,
    ) -> np.ndarray:
        """
        Para cada vela i (LONG desde close[i]):
          TP_long = close[i] + tp_usd   → si high[j] >= TP_long  → label = 1.0
          SL_long = close[i] - sl_usd   → si low[j]  <= SL_long  → label = 0.0
          Ambas en misma vela → usa dirección de la vela previa
          Sin hit en max_forward velas   → label = -1.0 (excluir)
        """
        n      = len(close)
        labels = np.full(n, -1.0)

        for i in range(n - max_forward - 1):
            entry   = close[i]
            tp_long = entry + tp_usd
            sl_long = entry - sl_usd

            for j in range(1, max_forward + 1):
                idx = i + j
                h   = high[idx]
                lo  = low[idx]

                tp_hit = h >= tp_long
                sl_hit = lo <= sl_long

                if tp_hit and sl_hit:
                    # Ambas: usar vela previa para desambiguar
                    prev_close = close[idx - 1]
                    labels[i]  = 1.0 if prev_close >= entry else 0.0
                    break
                elif tp_hit:
                    labels[i] = 1.0
                    break
                elif sl_hit:
                    labels[i] = 0.0
                    break

        return labels

else:
    def _label_tpsl_numba(high, low, close, tp_usd, sl_usd, max_forward):
        """Versión Python pura como fallback (más lenta)."""
        n      = len(close)
        labels = np.full(n, -1.0)
        for i in range(n - max_forward - 1):
            entry   = close[i]
            tp_long = entry + tp_usd
            sl_long = entry - sl_usd
            for j in range(1, max_forward + 1):
                idx = i + j
                tp_hit = high[idx] >= tp_long
                sl_hit = low[idx]  <= sl_long
                if tp_hit and sl_hit:
                    labels[i] = 1.0 if close[idx - 1] >= entry else 0.0
                    break
                elif tp_hit:
                    labels[i] = 1.0
                    break
                elif sl_hit:
                    labels[i] = 0.0
                    break
        return labels


def compute_labels(feat_df: pd.DataFrame) -> pd.Series:
    """
    Calcula etiquetas binarias para toda la serie de features:
      1.0 = LONG favorable (TP +$500 alcanzado antes que SL -$500)
      0.0 = SHORT favorable (SL -$500 alcanzado antes que TP +$500)
     -1.0 = sin etiqueta (ninguno en ventana MAX_FORWARD_CANDLES)
    """
    high  = feat_df["high_raw"].values.astype(np.float64)
    low   = feat_df["low_raw"].values.astype(np.float64)
    close = feat_df["close_raw"].values.astype(np.float64)

    labels_arr = _label_tpsl_numba(
        high, low, close,
        float(TP_USD), float(SL_USD), int(MAX_FORWARD_CANDLES)
    )
    return pd.Series(labels_arr, index=feat_df.index, name="label")


# =============================================================================
# 5. CREACIÓN DE SECUENCIAS (X, y)
# =============================================================================

FEATURE_COLS = ["close_ret_z", "open_ret_z", "high_ret_z", "low_ret_z", "vol_z", "range_z"]


def build_sequences(
    feat_df:   pd.DataFrame,
    labels:    pd.Series,
    lookback:  int  = LOOKBACK,
    stride:    int  = STRIDE,
    min_label: float = 0.0,  # excluir labels < 0 (sin etiqueta)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Crea arrays de secuencias para entrenamiento:
      X : (N, lookback, n_features)  — features normalizadas
      y : (N,)                       — etiquetas binarias {0, 1}
      idx: (N,) índices del DataFrame de la última vela de cada secuencia

    Solo incluye muestras con label válida (≥ 0).
    Usa stride para reducir autocorrelación y agilizar entrenamiento.
    """
    feature_arr = feat_df[FEATURE_COLS].values.astype(np.float32)
    label_arr   = labels.values.astype(np.float32)
    n           = len(feature_arr)

    X_list, y_list, idx_list = [], [], []

    for i in range(lookback, n - MAX_FORWARD_CANDLES, stride):
        lbl = label_arr[i]
        if lbl < min_label:          # excluir -1.0
            continue
        window = feature_arr[i - lookback:i]   # (lookback, n_features)
        X_list.append(window)
        y_list.append(lbl)
        idx_list.append(i)

    if len(X_list) == 0:
        return (
            np.empty((0, lookback, N_FEATURES), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    X   = np.stack(X_list, axis=0)                    # (N, lookback, n_features)
    y   = np.array(y_list, dtype=np.float32)           # (N,)
    idx = np.array(idx_list, dtype=np.int64)           # (N,)

    return X, y, idx


# =============================================================================
# 6. FUNCIÓN PÚBLICA PRINCIPAL
# =============================================================================

def prepare_data(
    date_start: Optional[str] = None,
    date_end:   Optional[str] = None,
    verbose:    bool           = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Pipeline completo de datos:
      1. Carga BTC 1m
      2. Calcula features (log-returns + z-score)
      3. Calcula etiquetas (TP/SL $500)

    Returns:
      feat_df  — DataFrame con features + raw prices
      labels   — Series con etiquetas {-1, 0, 1}
      raw_df   — DataFrame con OHLCV original (para backtest)
    """
    raw_df  = load_btc_data(date_start, date_end, verbose=verbose)
    feat_df = compute_features(raw_df)
    # Alinear raw_df con feat_df (el warmup recorta filas iniciales)
    raw_df  = raw_df.loc[feat_df.index]
    labels  = compute_labels(feat_df)
    return feat_df, labels, raw_df


def get_class_weights(y: np.ndarray) -> Tuple[float, float]:
    """
    Calcula pesos de clase para manejar desbalance:
      w_0 = N / (2 * n_0)
      w_1 = N / (2 * n_1)
    """
    n   = len(y)
    n_0 = float((y == 0).sum())
    n_1 = float((y == 1).sum())
    w_0 = n / (2.0 * n_0) if n_0 > 0 else 1.0
    w_1 = n / (2.0 * n_1) if n_1 > 0 else 1.0
    return w_0, w_1


def data_summary(feat_df: pd.DataFrame, labels: pd.Series) -> dict:
    """Estadísticas del dataset para mostrar en consola."""
    valid_mask = labels >= 0
    valid_lbls = labels[valid_mask]
    n_total    = len(labels)
    n_valid    = valid_mask.sum()
    n_long     = (valid_lbls == 1).sum()
    n_short    = (valid_lbls == 0).sum()
    n_skip     = n_total - n_valid

    return {
        "n_total"    : n_total,
        "n_valid"    : n_valid,
        "n_long"     : n_long,
        "n_short"    : n_short,
        "n_skip"     : n_skip,
        "pct_long"   : 100.0 * n_long  / n_valid if n_valid > 0 else 0.0,
        "pct_short"  : 100.0 * n_short / n_valid if n_valid > 0 else 0.0,
        "date_start" : feat_df.index[0].strftime("%Y-%m-%d"),
        "date_end"   : feat_df.index[-1].strftime("%Y-%m-%d"),
        "n_features" : N_FEATURES,
        "lookback"   : LOOKBACK,
    }
