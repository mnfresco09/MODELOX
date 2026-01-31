"""modelox/core/scoring.py

═══════════════════════════════════════════════════════════════════════════════
SISTEMA DE SCORING UNIFICADO v9.0 - CALIDAD PURA
═══════════════════════════════════════════════════════════════════════════════

FILOSOFÍA:
El Score se basa únicamente en métricas de CALIDAD.
La robustez se valida en Fase 2 y Fase 3 del Topógrafo de Mesetas.

Score = Puntos_Calidad × Factor_Actividad

Máximo: 600 puntos

COMPONENTES:
1. CALIDAD (0-600 puntos):
   - Sharpe, SQN, ROI, Drawdown normalizados con tanh
   - Rendimientos decrecientes

2. ACTIVIDAD (factor 0-1):
   - Penaliza estrategias con pocos trades/día
   - Centro en 0.25 trades/día

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np


# =============================================================================
# PARÁMETROS DE SCORING v9.0
# =============================================================================

# Máximo de puntos (solo calidad, sin robustez)
MAX_PUNTOS_CALIDAD = 1000.0
MAX_SCORE = 1000.0

# Pesos de métricas de calidad (suman 1.0)
PESO_SHARPE = 0.30
PESO_SQN = 0.28
PESO_DRAWDOWN = 0.18
PESO_ROI = 0.24

# Rangos progresivos para cada métrica [mínimo_para_0, máximo_para_techo]
# Sharpe (×100): -200 → 0 puntos, 400 → máximo
SHARPE_MIN = -20.0
SHARPE_MAX = 20.0

# SQN: -2 → 0 puntos, 5 → máximo
SQN_MIN = -3.0
SQN_MAX = 4.5

# Drawdown: 60% → 0 puntos, 10% → máximo (invertido: menor es mejor)
DD_MIN = 60.0   # Peor caso (0 puntos)
DD_MAX = 10.0   # Mejor caso (máximo puntos)

# ROI: -50% → 0 puntos, 350% → máximo
ROI_MIN = -60.0
ROI_MAX = 350.0

# Factor de actividad
ACTIVIDAD_CENTRO = 0.25
ACTIVIDAD_PENDIENTE = 12.0

# Umbrales de suspenso
REF_ROI_SUSPENSO = 100.0
REF_TRADES_DIA_MIN = 0.15


# =============================================================================
# UTILIDADES
# =============================================================================

def _get(metrics: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    """Extrae un valor de métricas de forma segura."""
    val = metrics.get(key, default)
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _linear_normalize(value: float, min_val: float, max_val: float) -> float:
    """
    Normaliza un valor al rango [0, 1] de forma lineal progresiva.
    
    - valor <= min_val → 0.0
    - valor >= max_val → 1.0
    - valor intermedio → proporción lineal
    """
    if value <= min_val:
        return 0.0
    if value >= max_val:
        return 1.0
    return (value - min_val) / (max_val - min_val)


def _linear_normalize_inverted(value: float, min_val: float, max_val: float) -> float:
    """
    Normaliza un valor al rango [0, 1] de forma INVERTIDA (menor es mejor).
    
    - valor >= min_val (peor) → 0.0
    - valor <= max_val (mejor) → 1.0
    """
    if value >= min_val:
        return 0.0
    if value <= max_val:
        return 1.0
    return (min_val - value) / (min_val - max_val)


# =============================================================================
# CÁLCULO DE COMPONENTES
# =============================================================================

def calculate_calidad_raw(
    sharpe: float,
    sqn: float,
    drawdown: float,
    roi: float,
) -> float:
    """
    Calcula puntos de calidad (0-600) usando normalización lineal progresiva.
    
    Rangos:
    - Sharpe: -2 (0 pts) → 4 (máx)
    - SQN: -2 (0 pts) → 5 (máx)
    - Drawdown: 60% (0 pts) → 10% (máx) [invertido]
    - ROI: -50% (0 pts) → 350% (máx)
    """
    # Normalizar cada métrica (0-1) con rangos progresivos
    sharpe_norm = _linear_normalize(sharpe, SHARPE_MIN, SHARPE_MAX)
    sqn_norm = _linear_normalize(sqn, SQN_MIN, SQN_MAX)
    roi_norm = _linear_normalize(roi, ROI_MIN, ROI_MAX)
    
    # Drawdown: invertido (menor es mejor)
    dd_norm = _linear_normalize_inverted(drawdown, DD_MIN, DD_MAX)
    
    # Suma ponderada
    calidad_norm = (
        PESO_SHARPE * sharpe_norm +
        PESO_SQN * sqn_norm +
        PESO_ROI * roi_norm +
        PESO_DRAWDOWN * dd_norm
    )
    
    return calidad_norm * MAX_PUNTOS_CALIDAD


def calculate_factor_actividad(trades_dia: float) -> float:
    """
    Factor de actividad: penaliza si hay muy pocos trades/día.
    
    Sigmoide logística centrada en ACTIVIDAD_CENTRO:
    - trades/día < centro → factor baja hacia 0
    - trades/día = centro → factor = 0.5
    - trades/día > centro → factor sube hacia 1.0
    """
    if trades_dia <= 0:
        return 0.0
    
    x = (trades_dia - ACTIVIDAD_CENTRO) * ACTIVIDAD_PENDIENTE
    return 1.0 / (1.0 + math.exp(-x))


# =============================================================================
# FUNCIÓN PRINCIPAL: SCORE
# =============================================================================

def score_unified(
    metrics: Mapping[str, Any],
    neighborhood_result: Optional[Mapping[str, Any]] = None,  # Ignorado
    trial_number: int = 0,
    equity_curve: Optional[List[float]] = None,
) -> float:
    """
    SCORE v9.0 - Solo calidad.
    
    Fórmula: Score = Calidad_Raw × Factor_Actividad
    
    RANGO: [0, 600]
    
    Args:
        metrics: Métricas del trial
        neighborhood_result: IGNORADO (compatibilidad)
        trial_number: Número del trial
        equity_curve: Curva de equity (opcional)
    
    Returns:
        Score final [0, 600]
    """
    # Extraer trades/día
    trades_dia = _get(metrics, "trades_por_dia", 0.0)
    if trades_dia == 0:
        trades_dia = _get(metrics, "trades_dia", 0.0)
    if trades_dia == 0:
        trades_dia = _get(metrics, "trades_per_day", 0.0)
    if trades_dia == 0:
        trades_dia = _get(metrics, "tpd", 0.0)
    
    # Suspenso si trades/día muy bajo
    if trades_dia < REF_TRADES_DIA_MIN:
        return 0.001
    
    # Extraer métricas
    sqn = _get(metrics, "sqn", 0.0)
    
    sharpe = _get(metrics, "sharpe", 0.0)
    if sharpe == 0:
        sharpe = _get(metrics, "sharpe_ratio", 0.0)
    
    drawdown = _get(metrics, "drawdown", 50.0)
    if drawdown == 0:
        drawdown = _get(metrics, "max_drawdown", 50.0)
    
    roi = _get(metrics, "roi", 0.0)
    if roi == 0:
        roi = _get(metrics, "roi_pct", 0.0)
    
    # Calcular calidad
    calidad_raw = calculate_calidad_raw(
        sharpe=sharpe,
        sqn=sqn,
        drawdown=drawdown,
        roi=roi,
    )
    
    # Aplicar factor de actividad
    factor_actividad = calculate_factor_actividad(trades_dia)
    score = calidad_raw * factor_actividad
    
    return float(max(0.001, min(MAX_SCORE, score)))


def format_score(score: float) -> str:
    """Formatea el score: X.XX si >=1, 0.XXX si <1."""
    if score >= 1.0:
        return f"{score:.2f}"
    else:
        return f"{score:.3f}"


# =============================================================================
# FUNCIONES LEGACY PARA COMPATIBILIDAD
# =============================================================================

def score_quality_only(metrics: Mapping[str, Any]) -> float:
    """Score de calidad."""
    return score_unified(metrics)


def score_optuna(metrics: Mapping[str, Any]) -> float:
    """Alias para compatibilidad."""
    return score_unified(metrics)


def nsga2_objectives(metrics: Mapping[str, Any]) -> Tuple[float, float]:
    """Objetivos para NSGA-II (quality, drawdown)."""
    quality = score_unified(metrics)
    drawdown = _get(metrics, "drawdown", 50.0)
    if drawdown == 0:
        drawdown = _get(metrics, "max_drawdown", 50.0)
    return (max(0.001, quality), max(0.0, min(100.0, drawdown)))


def calculate_deflated_sharpe_score(
    sharpe: float,
    trial_number: int,
    n_trades: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """DSR Score [0, 100] para compatibilidad."""
    if n_trades < 5 or sharpe <= 0:
        return max(5.0, 15.0 + sharpe * 10.0) if sharpe > -1 else 5.0
    
    sr_var = max(0.5, 1.0 + 0.5 * sharpe**2)
    sr_std = math.sqrt(sr_var / max(n_trades, 10))
    trial_factor = math.sqrt(2 * math.log(max(2, trial_number + 1)))
    expected_sr = sr_std * trial_factor
    
    if sr_std > 0.001:
        z = (sharpe - expected_sr) / sr_std
        prob = 0.5 * (1.0 + math.erf(z / 1.414))
    else:
        prob = 0.9 if sharpe > 0.5 else 0.5
    
    return min(100.0, max(0.1, prob * 100.0))


def deflated_sharpe_ratio(sharpe_obs: float, n_trials: int, n_trades: int, **kw) -> float:
    """Legacy: Retorna probabilidad [0, 1]."""
    return calculate_deflated_sharpe_score(sharpe_obs, n_trials, n_trades) / 100.0


def score_robust(metrics: Mapping[str, Any], **kw) -> float:
    """Legacy."""
    return score_unified(metrics)


@dataclass
class ScoringConfig:
    """Legacy: Configuración del sistema de scoring."""
    pass


def set_study_for_scorer(study: Any) -> None:
    """Legacy: Ya no se necesita."""
    pass


def cleanup_scoring_resources() -> None:
    """Legacy: Ya no hay recursos que limpiar."""
    pass
