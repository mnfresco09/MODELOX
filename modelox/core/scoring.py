from __future__ import annotations
from typing import Any, Mapping
import math


def _get(metrics: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    """Helper seguro para extraer valores numéricos de las métricas."""
    try:
        val = metrics.get(key, default)
        if val is None:
            return default
        f_val = float(val)
        if math.isnan(f_val) or math.isinf(f_val):
            return default
        return f_val
    except Exception:
        return default


def score_quality_only(metrics: Mapping[str, Any]) -> float:
    """
    ═══════════════════════════════════════════════════════════════════════════
    SCORE DE CALIDAD PURA (Para optimización Multi-Objetivo)
    ═══════════════════════════════════════════════════════════════════════════
    
    OBJETIVO:
    Medir exclusivamente la "Fuerza" y "Robustez" de la estrategia para ganar dinero.
    
    FILTROS HARD-CUT:
    1. Frecuencia Mínima: Si trades_por_dia < 0.25 -> Score 0.1 (Descarte inmediato)
    
    IMPORTANTE:
    NO penalizamos Drawdown aquí. El Drawdown se minimiza como un 
    OBJETIVO INDEPENDIENTE en el optimizador NSGA-II.
    ═══════════════════════════════════════════════════════════════════════════
    """
    
    # 1. Extracción de Datos Básicos
    n_trades = _get(metrics, "n_trades", 0.0)
    if n_trades == 0:
        n_trades = _get(metrics, "total_trades", 0.0)
    
    trades_dia = _get(metrics, "trades_por_dia", 0.0)
    if trades_dia == 0:
        trades_dia = _get(metrics, "trades_dia", 0.0)
    
    # ═════════════════════════════════════════════════════════════════════════
    # HARD CUT: FILTRO DE ACTIVIDAD MÍNIMA
    # ═════════════════════════════════════════════════════════════════════════
    if trades_dia < 0.25:
        return 0.1
    # ═════════════════════════════════════════════════════════════════════════

    winrate_raw = _get(metrics, "winrate", 0.0)
    # Normalizar winrate: si viene como porcentaje (>1) dividir por 100
    if winrate_raw > 1.0:
        winrate = winrate_raw / 100.0
    else:
        winrate = winrate_raw
    
    payoff_ratio = _get(metrics, "payoff_ratio", 0.0)
    profit_factor = _get(metrics, "profit_factor", 0.0)
    sqn_val = _get(metrics, "sqn", 0.0)
    
    max_ganancia = _get(metrics, "max_ganancia", 0.0)
    pnl_neto = _get(metrics, "pnl_neto", 0.0)
    if pnl_neto == 0:
        pnl_neto = _get(metrics, "net_pnl", 0.0)
    
    roi = _get(metrics, "roi", 0.0)
    if roi == 0:
        roi = _get(metrics, "roi_pct", 0.0)

    # 2. Cálculo de Ventaja Estadística (Edge)
    # Edge = Win% * Payoff - (1 - Win%)
    if payoff_ratio > 0:
        edge = (winrate * payoff_ratio) - (1.0 - winrate)
    else:
        edge = -1.0

    # 3. Factores base
    evidence_factor = n_trades / (n_trades + 50.0)
    sqn_score = math.log(1.0 + max(0.0, sqn_val) * 2.0) if sqn_val > 0 else 0.0
    pf_bonus = min(max(profit_factor, 0.1), 3.0)

    # 4. Si la estrategia tiene edge negativo, usar PF como base
    # Esto permite diferenciar entre estrategias "malas" y "muy malas"
    if edge <= 0:
        # Score basado en profit_factor cuando edge es negativo
        # PF=0.5 -> score ~5, PF=0.8 -> score ~8, PF=1.0 -> score ~10
        base_score = profit_factor * 10.0 * evidence_factor
        # Añadir pequeño bonus por ROI si es positivo
        if roi > 0:
            base_score += min(roi / 10.0, 5.0)
        return max(0.1, min(base_score, 15.0))  # Cap en 15 para estrategias perdedoras

    # 5. Penalización de Concentración (anti-overfitting)
    concentration = 0.0
    if pnl_neto > 0 and max_ganancia > 0:
        concentration = max_ganancia / pnl_neto
    
    conc_penalty = 1.0
    if concentration > 0.30:
        conc_penalty = math.exp(-3.0 * (concentration - 0.30))
    
    # 6. Cálculo Final para estrategias con edge positivo
    raw_score = (edge * 100.0) * pf_bonus * (1.0 + sqn_score) * evidence_factor * conc_penalty
    
    return max(0.1, float(raw_score))


def score_optuna(metrics: Mapping[str, Any]) -> float:
    """
    LEGACY: Mantenido por compatibilidad si se usa TPE (Single Objective).
    Usa una versión simplificada pero robusta.
    """
    quality = score_quality_only(metrics)
    
    # Si quality devolvió 0.1 (Hard Cut), devolvemos eso directamente
    if quality <= 0.101:
        return quality
        
    drawdown = _get(metrics, "drawdown", 100.0)
    
    # Penalización suave de drawdown para single-objective
    dd_factor = 1.0 / (1.0 + (drawdown / 20.0)) 
    
    return quality * dd_factor


def nsga2_objectives(metrics: Mapping[str, Any]) -> tuple[float, float]:
    """
    ═══════════════════════════════════════════════════════════════════════════
    OBJETIVOS PARA NSGA-II (Multi-Objetivo)
    ═══════════════════════════════════════════════════════════════════════════
    
    Retorna una tupla de dos valores:
    1. quality (MAXIMIZAR): Score de calidad pura de la estrategia
    2. drawdown (MINIMIZAR): Drawdown máximo en porcentaje
    
    NSGA-II optimiza ambos simultáneamente, buscando el Frente de Pareto.
    ═══════════════════════════════════════════════════════════════════════════
    """
    quality = score_quality_only(metrics)
    drawdown = _get(metrics, "drawdown", 100.0)
    
    # Asegurar valores válidos
    quality = max(0.1, float(quality))
    drawdown = max(0.0, min(100.0, float(drawdown)))
    
    return (quality, drawdown)