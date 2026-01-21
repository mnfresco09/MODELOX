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
    trades_dia = _get(metrics, "trades_por_dia", 0.0)
    
    # ═════════════════════════════════════════════════════════════════════════
    # HARD CUT: FILTRO DE ACTIVIDAD MÍNIMA
    # ═════════════════════════════════════════════════════════════════════════
    # Si la estrategia opera menos de 0.25 veces al día (1 trade cada 4 días),
    # se considera inactiva o "Zombie". Se descarta inmediatamente.
    if trades_dia < 0.25:
        return 0.1
    # ═════════════════════════════════════════════════════════════════════════

    winrate = _get(metrics, "winrate", 0.0) / 100.0  # 0.0 a 1.0
    payoff_ratio = _get(metrics, "payoff_ratio", 0.0)
    profit_factor = _get(metrics, "profit_factor", 0.0)
    sqn_val = _get(metrics, "sqn", 0.0)
    
    max_ganancia = _get(metrics, "max_ganancia", 0.0)
    pnl_neto = _get(metrics, "pnl_neto", 0.0)
    if pnl_neto == 0:
        pnl_neto = _get(metrics, "net_pnl", 0.0)

    # 2. Cálculo de Ventaja Estadística (Edge)
    # Fórmula: (Win% * AvgWin) - (Loss% * AvgLoss) -> Normalizado por AvgLoss
    # Edge = Win% * Payoff - (1 - Win%)
    if payoff_ratio > 0:
        edge = (winrate * payoff_ratio) - (1.0 - winrate)
    else:
        edge = -1.0 # Estrategia perdedora

    # Si la ventaja es negativa o nula, score mínimo inmediato
    if edge <= 0:
        return 0.1

    # 3. Factores Multiplicadores
    
    # A) Evidencia (Pocos trades = poca confianza)
    # Con 50 trades ya tienes el 50% del factor, con 100 el 66%, con 300 el 85%
    evidence_factor = n_trades / (n_trades + 50.0)
    
    # B) Consistencia (SQN)
    # SQN suele ir de 1.0 (pobre) a 5.0 (santo grial). Usamos log para suavizar extremos.
    sqn_score = math.log(1.0 + max(0.0, sqn_val) * 2.0)
    
    # C) Profit Factor Sano
    # Buscamos PF > 1.2. Más de 3.0 suele ser curve-fitting, pero no lo penalizamos aquí,
    # solo limitamos el beneficio del bonus.
    pf_bonus = min(profit_factor, 3.0)

    # 4. Penalización de Anti-Overfitting (Concentración)
    concentration = 0.0
    if pnl_neto > 0 and max_ganancia > 0:
        concentration = max_ganancia / pnl_neto
    
    # Si un solo trade hace más del 30% del beneficio anual, penalizamos.
    conc_penalty = 1.0
    if concentration > 0.30:
        # Penalización exponencial: si es 40% -> factor 0.8, si es 80% -> factor 0.3
        conc_penalty = math.exp(-3.0 * (concentration - 0.30))
    
    # 5. Cálculo Final
    # Base 100 * Edge * Factores
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