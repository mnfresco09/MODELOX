"""
================================================================================
IA/WALK_FORWARD.PY — VALIDACIÓN WALK-FORWARD
================================================================================
Genera folds de entrenamiento/validación:
  • Train window : 1.5 años
  • Embargo      : 60 días (elimina autocorrelación)
  • Val window   : 1 año
  • Step size    : 12 meses
  • Mínimo       : 3 folds

Diagrama temporal:
  [====TRAIN 1.5yr====] [--60d--] [==VAL 1yr==]
                                               ←12m→
                        [====TRAIN 1.5yr====] [--60d--] [==VAL 1yr==]
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from IA.config import (
    TRAIN_YEARS, EMBARGO_DAYS, VAL_YEARS, STEP_MONTHS, MIN_FOLDS,
    LOOKBACK, STRIDE, MAX_FORWARD_CANDLES,
    QUICK_MODE,
)
from IA.data_pipeline import build_sequences, FEATURE_COLS


# =============================================================================
# FOLD DATA CLASS
# =============================================================================

@dataclass
class WalkForwardFold:
    fold_n:         int
    train_start:    pd.Timestamp
    train_end:      pd.Timestamp
    embargo_start:  pd.Timestamp
    embargo_end:    pd.Timestamp
    val_start:      pd.Timestamp
    val_end:        pd.Timestamp
    # Arrays de secuencias
    X_train:        Optional[np.ndarray] = None
    y_train:        Optional[np.ndarray] = None
    X_val:          Optional[np.ndarray] = None
    y_val:          Optional[np.ndarray] = None
    train_indices:  Optional[np.ndarray] = None
    val_indices:    Optional[np.ndarray] = None


# =============================================================================
# GENERADOR DE FOLDS
# =============================================================================

def generate_folds(
    feat_df:  pd.DataFrame,
    labels:   pd.Series,
    train_years:  float = TRAIN_YEARS,
    embargo_days: int   = EMBARGO_DAYS,
    val_years:    float = VAL_YEARS,
    step_months:  int   = STEP_MONTHS,
    max_folds:    int   = 20,
) -> List[WalkForwardFold]:
    """
    Genera folds walk-forward sin data leakage.
    El embargo garantiza que no haya autocorrelación entre train y val.

    Args:
      feat_df      : DataFrame de features con índice datetime
      labels       : Series de etiquetas alineada con feat_df
      train_years  : Tamaño ventana de entrenamiento en años
      embargo_days : Gap entre fin de train y inicio de val
      val_years    : Tamaño ventana de validación en años
      step_months  : Meses de avance entre folds consecutivos
      max_folds    : Máximo de folds a generar

    Returns:
      Lista de WalkForwardFold con arrays X/y listos para entrenar
    """
    idx       = feat_df.index
    start_all = idx[0]
    end_all   = idx[-1]

    # ── Duraciones ───────────────────────────────────────────────────
    train_td   = timedelta(days=int(train_years * 365.25))
    embargo_td = timedelta(days=embargo_days)
    val_td     = timedelta(days=int(val_years * 365.25))
    step_td    = timedelta(days=int(step_months * 30.44))

    folds: List[WalkForwardFold] = []
    fold_n = 1
    cursor = start_all

    while fold_n <= max_folds:
        train_start   = cursor
        train_end     = train_start + train_td
        embargo_start = train_end
        embargo_end   = embargo_start + embargo_td
        val_start     = embargo_end
        val_end       = val_start + val_td

        # ── Verificar que hay datos suficientes ──────────────────────
        if val_end > end_all:
            break

        # ── Extraer subsets ──────────────────────────────────────────
        train_mask = (idx >= train_start) & (idx < train_end)
        val_mask   = (idx >= val_start)   & (idx < val_end)

        train_feat = feat_df[train_mask]
        train_lbls = labels[train_mask]
        val_feat   = feat_df[val_mask]
        val_lbls   = labels[val_mask]

        if len(train_feat) < LOOKBACK * 10 or len(val_feat) < LOOKBACK * 10:
            cursor += step_td
            continue

        # ── Crear secuencias ──────────────────────────────────────────
        stride_use = STRIDE
        X_tr, y_tr, idx_tr = build_sequences(train_feat, train_lbls, stride=stride_use)
        X_vl, y_vl, idx_vl = build_sequences(val_feat,   val_lbls,   stride=max(1, stride_use // 2))

        if len(X_tr) < 100 or len(X_vl) < 50:
            cursor += step_td
            continue

        fold = WalkForwardFold(
            fold_n        = fold_n,
            train_start   = train_start,
            train_end     = train_end,
            embargo_start = embargo_start,
            embargo_end   = embargo_end,
            val_start     = val_start,
            val_end       = val_end,
            X_train       = X_tr,
            y_train       = y_tr,
            X_val         = X_vl,
            y_val         = y_vl,
            train_indices = idx_tr,
            val_indices   = idx_vl,
        )
        folds.append(fold)
        fold_n += 1
        cursor += step_td

    return folds


def fold_summary(fold: WalkForwardFold) -> dict:
    """Estadísticas del fold para mostrar en consola."""
    n_tr  = len(fold.X_train) if fold.X_train is not None else 0
    n_vl  = len(fold.X_val)   if fold.X_val   is not None else 0
    y_tr  = fold.y_train if fold.y_train is not None else np.array([])
    y_vl  = fold.y_val   if fold.y_val   is not None else np.array([])

    n_tr_long  = int((y_tr == 1).sum()) if len(y_tr) > 0 else 0
    n_tr_short = int((y_tr == 0).sum()) if len(y_tr) > 0 else 0
    n_vl_long  = int((y_vl == 1).sum()) if len(y_vl) > 0 else 0
    n_vl_short = int((y_vl == 0).sum()) if len(y_vl) > 0 else 0

    return {
        "fold_n"      : fold.fold_n,
        "train_start" : fold.train_start.strftime("%Y-%m-%d"),
        "train_end"   : fold.train_end.strftime("%Y-%m-%d"),
        "val_start"   : fold.val_start.strftime("%Y-%m-%d"),
        "val_end"     : fold.val_end.strftime("%Y-%m-%d"),
        "n_train"     : n_tr,
        "n_val"       : n_vl,
        "tr_long_pct" : 100.0 * n_tr_long  / n_tr if n_tr > 0 else 0.0,
        "tr_short_pct": 100.0 * n_tr_short / n_tr if n_tr > 0 else 0.0,
        "vl_long_pct" : 100.0 * n_vl_long  / n_vl if n_vl > 0 else 0.0,
        "vl_short_pct": 100.0 * n_vl_short / n_vl if n_vl > 0 else 0.0,
    }
