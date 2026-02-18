"""
================================================================================
IA/SIGNAL.PY — LÓGICA DE SEÑAL Y FILTROS
================================================================================
  1. Threshold filter:  P > 0.70 → LONG  |  P < 0.30 → SHORT
  2. Entropy filter:    rechaza si incertidumbre binaria es alta
  3. Anomaly detector:  avisa si precio en zona extrema (percentil 95 z-score)
================================================================================
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Tuple

from IA.config import (
    PROB_THRESHOLD, SHORT_THRESHOLD,
    ENTROPY_THRESHOLD, ANOMALY_PERCENTILE, ANOMALY_WINDOW,
)
from IA.model import GRUTradingModel, DEVICE


# =============================================================================
# 1. GENERACIÓN DE PROBABILIDADES
# =============================================================================

@torch.no_grad()
def predict_batch(
    model:   GRUTradingModel,
    X:       np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    """
    Genera predicciones en batches para no saturar memoria.
    Devuelve array de probabilidades P(LONG) en [0, 1].
    """
    model.eval()
    probs = []
    n     = len(X)

    for i in range(0, n, batch_size):
        batch = torch.from_numpy(X[i:i + batch_size]).float().to(DEVICE)
        p     = model(batch).cpu().numpy()
        probs.append(p)

    return np.concatenate(probs, axis=0) if probs else np.array([])


# =============================================================================
# 2. FILTRO DE ENTROPÍA
# =============================================================================

def binary_entropy(p: np.ndarray) -> np.ndarray:
    """
    Entropía binaria normalizada: H(p) / log(2) ∈ [0, 1]
    H = 1.0 → máxima incertidumbre (p ≈ 0.5)
    H = 0.0 → máxima certeza (p ≈ 0 ó 1)
    """
    eps = 1e-10
    H   = -(p * np.log(p + eps) + (1 - p) * np.log(1 - p + eps))
    return H / np.log(2.0)    # normalizar a [0, 1]


def entropy_filter(probs: np.ndarray, threshold: float = ENTROPY_THRESHOLD) -> np.ndarray:
    """
    Devuelve máscara booleana: True = señal válida (baja incertidumbre).
    Rechaza si entropía normalizada > threshold.
    """
    H = binary_entropy(probs)
    return H < threshold


# =============================================================================
# 3. DETECTOR DE ANOMALÍAS
# =============================================================================

def anomaly_score(
    close_prices: np.ndarray,
    current_idx:  int,
    window:       int   = ANOMALY_WINDOW,
) -> float:
    """
    Z-score absoluto del precio actual respecto a ventana histórica.
    |z| > z_percentil_95 → precio en zona extrema (anomalía).
    """
    if current_idx < window:
        return 0.0
    hist  = close_prices[max(0, current_idx - window): current_idx]
    mu    = float(np.mean(hist))
    sigma = float(np.std(hist))
    if sigma < 1e-10:
        return 0.0
    z = abs(close_prices[current_idx] - mu) / sigma
    return float(z)


def is_anomaly(
    close_prices: np.ndarray,
    current_idx:  int,
    window:       int   = ANOMALY_WINDOW,
    percentile:   float = ANOMALY_PERCENTILE,
) -> bool:
    """
    Devuelve True si el precio está en zona de anomalía extrema.
    Se calcula el z-score histórico y se compara con el percentil 95.
    """
    if current_idx < window:
        return False

    hist   = close_prices[max(0, current_idx - window): current_idx]
    mu     = float(np.mean(hist))
    sigma  = float(np.std(hist))
    if sigma < 1e-10:
        return False

    z_current  = abs(close_prices[current_idx] - mu) / sigma

    # Z-score histórico de los últimos `window` precios
    z_hist = np.abs((hist - np.mean(hist)) / (np.std(hist) + 1e-10))
    z_thresh = float(np.percentile(z_hist, percentile))

    return z_current > z_thresh


# =============================================================================
# 4. GENERADOR DE SEÑALES
# =============================================================================

def generate_signals(
    probs:           np.ndarray,
    close_prices:    np.ndarray,
    indices:         np.ndarray,
    prob_threshold:  float = PROB_THRESHOLD,
    short_threshold: float = SHORT_THRESHOLD,
    use_entropy:     bool  = True,
    use_anomaly:     bool  = True,
) -> np.ndarray:
    """
    Genera señales de trading para cada posición de predicción:
      +1 → LONG  (P > prob_threshold,  baja entropía)
      -1 → SHORT (P < short_threshold, baja entropía)
       0 → SIN SEÑAL

    Args:
      probs        : Array (N,) de probabilidades P(LONG)
      close_prices : Array de todos los precios de cierre (para anomalías)
      indices      : Índices absolutos en close_prices para cada predicción
      prob_threshold  : Umbral para LONG (default 0.70)
      short_threshold : Umbral para SHORT (default 0.30)
      use_entropy     : Aplicar filtro de entropía
      use_anomaly     : Aplicar detector de anomalías
    """
    n       = len(probs)
    signals = np.zeros(n, dtype=np.int8)

    # ── Filtro de entropía ────────────────────────────────────────────
    if use_entropy:
        valid_entropy = entropy_filter(probs)
    else:
        valid_entropy = np.ones(n, dtype=bool)

    for i in range(n):
        if not valid_entropy[i]:
            continue

        p   = probs[i]
        idx = int(indices[i])

        # ── Filtro de anomalías ───────────────────────────────────────
        if use_anomaly and idx < len(close_prices):
            if is_anomaly(close_prices, idx):
                # En zona de anomalía: invertir señal (mean-reversion)
                if p > prob_threshold:
                    signals[i] = -1   # precio extremo alto → SHORT
                elif p < short_threshold:
                    signals[i] = +1   # precio extremo bajo → LONG
                continue

        # ── Señal normal ──────────────────────────────────────────────
        if p > prob_threshold:
            signals[i] = +1   # LONG
        elif p < short_threshold:
            signals[i] = -1   # SHORT

    return signals


def signals_summary(signals: np.ndarray) -> dict:
    """Estadísticas básicas de las señales generadas."""
    n_long    = int((signals == +1).sum())
    n_short   = int((signals == -1).sum())
    n_none    = int((signals ==  0).sum())
    n_total   = len(signals)
    return {
        "n_long"   : n_long,
        "n_short"  : n_short,
        "n_none"   : n_none,
        "pct_long" : 100.0 * n_long  / n_total if n_total > 0 else 0.0,
        "pct_short": 100.0 * n_short / n_total if n_total > 0 else 0.0,
        "pct_none" : 100.0 * n_none  / n_total if n_total > 0 else 0.0,
    }
