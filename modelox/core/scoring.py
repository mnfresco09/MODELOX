"""
# =============================================================================
#
#     ███████╗ ██████╗ ██████╗ ██████╗ ██╗███╗   ██╗ ██████╗
#     ██╔════╝██╔════╝██╔═══██╗██╔══██╗██║████╗  ██║██╔════╝
#     ███████╗██║     ██║   ██║██████╔╝██║██╔██╗ ██║██║  ███╗
#     ╚════██║██║     ██║   ██║██╔══██╗██║██║╚██╗██║██║   ██║
#     ███████║╚██████╗╚██████╔╝██║  ██║██║██║ ╚████║╚██████╔╝
#     ╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝
#
#     SCORING.PY - SISTEMA DE PUNTUACIÓN v9.0
#
# =============================================================================
#
#     FILOSOFÍA:
#     El Score mide CALIDAD pura de la estrategia.
#     La robustez se valida aparte en el Topógrafo de Mesetas.
#
#     FÓRMULA:
#     Score = Calidad_Raw × Factor_Actividad
#
#     COMPONENTES:
#     - CALIDAD (0-1000 pts): Sharpe + SQN + ROI + Drawdown
#     - ACTIVIDAD (0-1): Penaliza pocas operaciones/día
#
# =============================================================================
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Tuple


# =============================================================================
# 1. CONSTANTES DE CONFIGURACIÓN
# =============================================================================

# PUNTUACIÓN MÁXIMA
MAX_PUNTOS_CALIDAD: float = 1000.0
MAX_SCORE: float = 1000.0

# PESOS DE MÉTRICAS (SUMAN 1.0)
PESO_SHARPE: float = 0.30
PESO_SQN: float = 0.28
PESO_DRAWDOWN: float = 0.18
PESO_ROI: float = 0.24

# RANGOS DE NORMALIZACIÓN
# Sharpe estándar sin escalar: valores típicos -2 a 3
SHARPE_MIN: float = -2.0
SHARPE_MAX: float = 3.0
SQN_MIN: float = -3.0
SQN_MAX: float = 4.5
DD_MIN: float = 60.0   # Peor (0 pts)
DD_MAX: float = 10.0   # Mejor (máx pts)
ROI_MIN: float = -60.0
ROI_MAX: float = 350.0

# FACTOR DE ACTIVIDAD
ACTIVIDAD_CENTRO: float = 0.25
ACTIVIDAD_PENDIENTE: float = 12.0

# UMBRALES MÍNIMOS
REF_TRADES_DIA_MIN: float = 0.15


# =============================================================================
# 2. UTILIDADES
# =============================================================================

def _get(metrics: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    """EXTRAE VALOR DE MÉTRICAS DE FORMA SEGURA."""
    val = metrics.get(key, default)
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _linear_normalize(value: float, min_val: float, max_val: float) -> float:
    """NORMALIZA VALOR A [0,1] - MÁS ES MEJOR."""
    if value <= min_val:
        return 0.0
    if value >= max_val:
        return 1.0
    return (value - min_val) / (max_val - min_val)


def _linear_normalize_inverted(value: float, min_val: float, max_val: float) -> float:
    """NORMALIZA VALOR A [0,1] - MENOS ES MEJOR (para drawdown)."""
    if value >= min_val:
        return 0.0
    if value <= max_val:
        return 1.0
    return (min_val - value) / (min_val - max_val)


# =============================================================================
# 3. CÁLCULO DE COMPONENTES
# =============================================================================

def calculate_calidad_raw(
    sharpe: float,
    sqn: float,
    drawdown: float,
    roi: float,
) -> float:
    """
    CALCULA PUNTOS DE CALIDAD (0-1000)
    
    Normaliza cada métrica a [0,1] y aplica pesos.
    Drawdown usa normalización invertida (menor = mejor).
    """
    sharpe_norm = _linear_normalize(sharpe, SHARPE_MIN, SHARPE_MAX)
    sqn_norm = _linear_normalize(sqn, SQN_MIN, SQN_MAX)
    roi_norm = _linear_normalize(roi, ROI_MIN, ROI_MAX)
    dd_norm = _linear_normalize_inverted(drawdown, DD_MIN, DD_MAX)
    
    calidad_norm = (
        PESO_SHARPE * sharpe_norm +
        PESO_SQN * sqn_norm +
        PESO_ROI * roi_norm +
        PESO_DRAWDOWN * dd_norm
    )
    
    return calidad_norm * MAX_PUNTOS_CALIDAD


def calculate_factor_actividad(trades_dia: float) -> float:
    """
    FACTOR DE ACTIVIDAD (0-1)
    
    Sigmoide que penaliza pocas operaciones:
    - trades/día < 0.25 → factor baja hacia 0
    - trades/día = 0.25 → factor = 0.5
    - trades/día > 0.25 → factor sube hacia 1.0
    """
    if trades_dia <= 0:
        return 0.0
    x = (trades_dia - ACTIVIDAD_CENTRO) * ACTIVIDAD_PENDIENTE
    return 1.0 / (1.0 + math.exp(-x))


# =============================================================================
# 4. FUNCIÓN PRINCIPAL
# =============================================================================

def score_unified(
    metrics: Mapping[str, Any],
    neighborhood_result: Optional[Mapping[str, Any]] = None,
    trial_number: int = 0,
    equity_curve: Optional[List[float]] = None,
) -> float:
    """
    SCORE UNIFICADO v9.0
    
    Fórmula: Score = Calidad × Actividad
    Rango: [0, 1000]
    
    Args:
        metrics: Diccionario con métricas del backtest
        neighborhood_result: Ignorado (compatibilidad)
        trial_number: Número del trial
        equity_curve: Curva de equity (opcional)
    
    Returns:
        Score final entre 0 y 1000
    """
    # EXTRAER TRADES/DÍA (BUSCAR EN VARIOS NOMBRES)
    trades_dia = _get(metrics, "trades_por_dia", 0.0)
    if trades_dia == 0:
        trades_dia = _get(metrics, "trades_dia", 0.0)
    if trades_dia == 0:
        trades_dia = _get(metrics, "trades_per_day", 0.0)
    
    # SUSPENSO SI MUY POCOS TRADES
    if trades_dia < REF_TRADES_DIA_MIN:
        return 0.001
    
    # EXTRAER MÉTRICAS
    sqn = _get(metrics, "sqn", 0.0)
    sharpe = _get(metrics, "sharpe", 0.0) or _get(metrics, "sharpe_ratio", 0.0)
    drawdown = _get(metrics, "drawdown", 50.0) or _get(metrics, "max_drawdown", 50.0)
    roi = _get(metrics, "roi", 0.0) or _get(metrics, "roi_pct", 0.0)
    
    # CALCULAR SCORE
    calidad_raw = calculate_calidad_raw(sharpe, sqn, drawdown, roi)
    factor_actividad = calculate_factor_actividad(trades_dia)
    score = calidad_raw * factor_actividad
    
    return float(max(0.001, min(MAX_SCORE, score)))


def format_score(score: float) -> str:
    """FORMATEA SCORE PARA MOSTRAR."""
    return f"{score:.2f}" if score >= 1.0 else f"{score:.3f}"


# =============================================================================
# 5. FUNCIONES DE COMPATIBILIDAD
# =============================================================================

def score_quality_only(metrics: Mapping[str, Any]) -> float:
    """ALIAS: Score de calidad."""
    return score_unified(metrics)


def score_optuna(metrics: Mapping[str, Any]) -> float:
    """ALIAS: Para uso con Optuna."""
    return score_unified(metrics)


def nsga2_objectives(metrics: Mapping[str, Any]) -> Tuple[float, float]:
    """OBJETIVOS PARA NSGA-II: (calidad, drawdown)."""
    quality = score_unified(metrics)
    drawdown = _get(metrics, "drawdown", 50.0) or _get(metrics, "max_drawdown", 50.0)
    return (max(0.001, quality), max(0.0, min(100.0, drawdown)))


def calculate_deflated_sharpe_score(
    sharpe: float,
    trial_number: int,
    n_trades: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """DEFLATED SHARPE RATIO SCORE [0, 100]."""
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
    """LEGACY: Retorna probabilidad [0, 1]."""
    return calculate_deflated_sharpe_score(sharpe_obs, n_trials, n_trades) / 100.0


def score_robust(metrics: Mapping[str, Any], **kw) -> float:
    """LEGACY: Alias de score_unified."""
    return score_unified(metrics)


@dataclass
class ScoringConfig:
    """LEGACY: Configuración vacía para compatibilidad."""
    pass


def set_study_for_scorer(study: Any) -> None:
    """LEGACY: No-op."""
    pass


def cleanup_scoring_resources() -> None:
    """LEGACY: No-op."""
    pass
