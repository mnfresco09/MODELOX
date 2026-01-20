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


def score_optuna(metrics: Mapping[str, Any]) -> float:
    """
    ═══════════════════════════════════════════════════════════════════════════
    SCORE ROBUSTO ANTI-OVERFITTING v2.1 (Winrate Penalization)
    ═══════════════════════════════════════════════════════════════════════════
    
    FILOSOFÍA:
    ───────────────────────────────────────────────────────────────────────────
    1. SUPERVIVENCIA PRIMERO: No queremos estrategias que quiebren.
    
    2. VENTAJA ESTADÍSTICA: Necesitamos evidencia de que funciona.
    
    3. ANTI-OVERFITTING: Castigar señales de curve-fitting.
        - NUEVO: Penalización severa y progresiva para Win Rates < 30%.
          Previene estrategias "lottery ticket" que son overfitting puro.
    
    4. SCORE SIEMPRE POSITIVO: Nunca 0 ni negativo.
    
    MÉTRICAS CLAVE USADAS:
    ───────────────────────────────────────────────────────────────────────────
    - winrate: % de trades ganadores (CRÍTICO para nueva penalización)
    - n_trades: Número de trades ejecutados
    - trades_por_dia: Frecuencia de operativa
    - drawdown: Max Drawdown %
    - saldo_actual / saldo_inicial: Supervivencia
    - payoff_ratio: Ganancia media / Pérdida media
    - sqn: System Quality Number (consistencia)
    - max_ganancia / pnl_neto: Concentración de ganancias
    - roi: Retorno sobre la inversión
    
    RETORNO:
    ───────────────────────────────────────────────────────────────────────────
    float > 0 (típicamente entre 0.01 y 100+)
    
    ═══════════════════════════════════════════════════════════════════════════
    """
    
    # =========================================================================
    # 1. EXTRACCIÓN DE MÉTRICAS
    # =========================================================================
    n_trades = _get(metrics, "n_trades", 0.0)
    trades_dia = _get(metrics, "trades_por_dia", 0.0)
    drawdown_pct = _get(metrics, "drawdown", 100.0)
    winrate = _get(metrics, "winrate", 0.0) / 100.0  # Convertir a decimal (0.0 a 1.0)
    
    saldo_actual = _get(metrics, "saldo_actual", 0.0)
    saldo_inicial = _get(metrics, "saldo_mean", 300.0)
    if saldo_inicial == 0:
        saldo_inicial = 300.0
    
    profit_factor = _get(metrics, "profit_factor", 0.0)
    payoff_ratio = _get(metrics, "payoff_ratio", 0.0)
    sqn_val = _get(metrics, "sqn", 0.0)
    
    max_ganancia = _get(metrics, "max_ganancia", 0.0)
    pnl_neto = _get(metrics, "pnl_neto", 0.0)
    if pnl_neto == 0:
        pnl_neto = _get(metrics, "net_pnl", 0.0)
    
    roi = _get(metrics, "roi", 0.0)
    
    # =========================================================================
    # 2. PENALIZACIÓN WINRATE BAJO (NUEVO)
    # =========================================================================
    # Objetivo: Penalizar severamente winrates por debajo del umbral de robustez (30%).
    # Esto evita que Optuna "aprenda" a depender de un único trade afortunado.
    # La penalización es CÚBICA, haciéndola muy agresiva en los extremos.
    
    winrate_threshold = 0.30  # 30%
    winrate_penalty = 1.0
    
    if winrate < winrate_threshold:
        # Calcular qué tan por debajo estamos del umbral (rango 0 a 1)
        # ej. WR=0.15 -> (0.30-0.15)/0.30 = 0.5 (estamos a mitad de camino hacia cero)
        gap = (winrate_threshold - winrate) / winrate_threshold
        
        # El factor de penalización es (1 - gap)^3.
        # - WR = 30% -> gap=0.0 -> penalty = (1-0)^3 = 1.0 (sin penalización)
        # - WR = 25% -> gap=0.16 -> penalty = (0.84)^3 = 0.59 (penalización del 41%)
        # - WR = 20% -> gap=0.33 -> penalty = (0.67)^3 = 0.30 (penalización del 70%)
        # - WR = 10% -> gap=0.67 -> penalty = (0.33)^3 = 0.03 (penalización del 97%)
        # - WR = 0%  -> gap=1.0 -> penalty = (0.0)^3 = 0.0  (penalización total)
        
        winrate_penalty = (1.0 - gap)**3
        
        # Asegurar un piso mínimo para no devolver exactamente cero
        winrate_penalty = max(0.01, winrate_penalty)

    # =========================================================================
    # 3. COMPONENTE: SUPERVIVENCIA
    # =========================================================================
    dd_factor = 1.0 / (1.0 + math.exp((drawdown_pct - 35.0) / 8.0))
    dd_factor = max(0.01, dd_factor)
    
    survival_ratio = saldo_actual / saldo_inicial if saldo_inicial > 0 else 0.0
    saldo_factor = math.sqrt(max(0.0, min(1.0, survival_ratio)))
    saldo_factor = max(0.01, saldo_factor)
    
    supervivencia = math.sqrt(dd_factor * saldo_factor)
    
    # =========================================================================
    # 4. COMPONENTE: FRECUENCIA
    # =========================================================================
    freq_factor = trades_dia / (trades_dia + 0.5)
    freq_factor = max(0.01, freq_factor)
    
    evidence_factor = n_trades / (n_trades + 100.0)
    evidence_factor = max(0.01, evidence_factor)
    
    # =========================================================================
    # 5. COMPONENTE: VENTAJA ESTADÍSTICA
    # =========================================================================
    if payoff_ratio > 0:
        kelly_edge = winrate * payoff_ratio - (1.0 - winrate)
    else:
        kelly_edge = winrate - 0.5
        
    edge_factor = math.log(1.0 + math.exp(kelly_edge * 2.0))
    
    sqn_bonus = 1.0 + 0.2 * max(0.0, sqn_val)
    sqn_bonus = min(2.0, sqn_bonus)
    
    if profit_factor > 1.0:
        pf_bonus = 1.0 + 0.1 * (profit_factor - 1.0)
        pf_bonus = min(1.5, pf_bonus)
    else:
        pf_bonus = max(0.5, profit_factor) if profit_factor > 0 else 0.5
    
    ventaja = edge_factor * sqn_bonus * pf_bonus
    ventaja = max(0.01, ventaja)
    
    # =========================================================================
    # 6. COMPONENTE: ANTI-OVERFITTING
    # =========================================================================
    concentration = 0.0
    if pnl_neto > 0 and max_ganancia > 0:
        concentration = max_ganancia / pnl_neto
    
    conc_penalty = math.exp(-2.0 * (concentration - 0.3)) if concentration > 0.3 else 1.0
    conc_penalty = max(0.3, conc_penalty)
    
    uncertainty_penalty = 1.0 / (1.0 + 2.0 / math.sqrt(n_trades + 1.0))
    uncertainty_penalty = max(0.3, uncertainty_penalty)
    
    anti_overfit = conc_penalty * uncertainty_penalty
    
    # =========================================================================
    # 7. CÁLCULO FINAL
    # =========================================================================
    # Producto de todos los componentes, AHORA INCLUYENDO LA PENALIZACIÓN DE WINRATE
    
    raw_score = (
        supervivencia * 
        freq_factor * 
        evidence_factor * 
        ventaja * 
        anti_overfit *
        winrate_penalty  # <-- Penalización aplicada aquí
    )
    
    final_score = raw_score * 10.0
    
    if roi > 20.0:
        roi_bonus = 1.0 + 0.01 * (roi - 20.0)
        roi_bonus = min(3.0, roi_bonus)
        final_score *= roi_bonus
    
    # =========================================================================
    # 8. GARANTÍA: SCORE SIEMPRE > 0
    # =========================================================================
    return max(0.001, float(final_score))
