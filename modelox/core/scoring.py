from __future__ import annotations
from typing import Any, Mapping
import math

def _f(metrics: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        v = float(metrics.get(key, default))
        if math.isnan(v) or math.isinf(v):
            return float(default)
        return v
    except Exception:
        return float(default)

def score_optuna(metrics: Mapping[str, Any]) -> float:
    """
    Score objetivo para Optuna modificado.
    
    Lógica de Regímenes:
    1. Si Trades/Día < 0.25:
       - Score = (Trades/Día) / 0.25
       - Rango resultante: [0.0, 1.0)
       - Objetivo: Penalizar la inactividad sin dar scores negativos, 
         pero haciendo imposible que compita con estrategias activas y rentables.
    
    2. Si Trades/Día >= 0.25:
       - Score = Suma ponderada de métricas de calidad.
       - Base mínima teórica: 1.0 (para estar siempre por encima del régimen inactivo).
       - Componentes:
         a) SQN: Escalado (aprox SQN*20). SQN 5.0 => 100 pts.
         b) Win Rate: Directo (0-100).
         c) Drawdown: Inverso (100 - DD). 0% DD => 100 pts.
         d) Sharpe: Escalado (aprox Sharpe*20). Sharpe 5.0 => 100 pts.
    """
    
    # --- Extracción de métricas ---
    trades_por_dia = _f(metrics, "trades_por_dia", 0.0)
    
    # --- RÉGIMEN 1: Baja Frecuencia (< 0.25) ---
    # Puntuamos solo la cercanía al umbral de activación.
    threshold = 0.25
    if trades_por_dia < threshold:
        # Ejemplo: 0.0 tpd -> 0.0
        # Ejemplo: 0.125 tpd -> 0.5
        # Ejemplo: 0.24 tpd -> 0.96
        return max(0.0, trades_por_dia) / threshold

    # --- RÉGIMEN 2: Alta Frecuencia (>= 0.25) ---
    # La estrategia es válida, evaluamos la calidad.
    
    sqn = _f(metrics, "sqn", 0.0)
    win_rate = _f(metrics, "winrate", _f(metrics, "porc_ganadoras", 0.0)) # Acepta ambas keys
    drawdown = _f(metrics, "drawdown", 100.0)
    sharpe = _f(metrics, "sharpe", 0.0)

    # 1. Score SQN (1 a 100 idealmente)
    # Asumimos que un SQN de 5.0 es excelente (100 puntos).
    # SQN negativos suman 0.
    score_sqn = max(0.0, sqn) * 20.0  
    
    # 2. Score Win Rate (0 a 100)
    score_wr = max(0.0, min(100.0, win_rate))
    
    # 3. Score Drawdown (0 a 100)
    # Menos drawdown es mejor. DD=0 -> 100, DD=100 -> 0.
    score_dd = max(0.0, 100.0 - drawdown)
    
    # 4. Score Sharpe (0 a 100 idealmente)
    # Similar al SQN, un Sharpe de 5.0 es estelar (100 puntos).
    score_sharpe = max(0.0, sharpe) * 20.0
    
    # Suma Total
    # Añadimos +1.0 para asegurar que cualquier estrategia "neutra" en este régimen
    # supere al mejor score del régimen inactivo (que topa en 0.999...).
    total_score = 1.0 + score_sqn + score_wr + score_dd + score_sharpe
    
    return float(total_score)