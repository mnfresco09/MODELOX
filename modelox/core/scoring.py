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
    Score Objetivo: ROBUSTEZ Y CRECIMIENTO REAL (Anti-Overfitting).
    
    Filosofía:
    - Ignoramos el Win Rate puro (evita estrategias de scalping suicida con SL infinito).
    - Priorizamos la relación Riesgo/Beneficio y la estabilidad (SQN).
    
    RÉGIMEN 1: Filtro de Actividad (< 0.25 trades/día)
      - Penalización lineal.
      
    RÉGIMEN 2: Calidad de Estrategia (>= 0.25 trades/día)
      - Componentes:
        1. SQN: Estabilidad estadística.
        2. Profit Factor: Eficiencia del dinero (Capped).
        3. Sharpe/Sortino: Retorno ajustado al riesgo.
        4. Penalización Exponencial de Drawdown: Odiamos las grandes caídas.
    """
    
    # --- Extracción de métricas ---
    trades_por_dia = _f(metrics, "trades_por_dia", 0.0)
    
    # --- RÉGIMEN 1: Baja Frecuencia (< 0.25) ---
    threshold = 0.25
    if trades_por_dia < threshold:
        # Score de 0.0 a 1.0 basado puramente en actividad
        return max(0.0, trades_por_dia) / threshold

    # --- RÉGIMEN 2: Alta Frecuencia (>= 0.25) ---
    # Aquí empieza la búsqueda de calidad real.
    
    sqn = _f(metrics, "sqn", 0.0)
    sharpe = _f(metrics, "sharpe", 0.0)
    sortino = _f(metrics, "sortino", sharpe) # Usamos Sortino si existe, si no Sharpe
    drawdown = _f(metrics, "drawdown", 100.0)
    profit_factor = _f(metrics, "profit_factor", 1.0)
    
    # 1. Score SQN (Peso Alto)
    # SQN mide la facilidad del sistema para hacer dinero.
    # SQN 2.0 = Decente, SQN 5.0 = Excelente.
    # Escalamos: SQN * 15. (Ej: 3.0 * 15 = 45 pts)
    score_sqn = max(0.0, sqn) * 15.0
    
    # 2. Score Profit Factor (Peso Medio - Eficiencia)
    # Un PF infinito (sin pérdidas) rompe el optimizador. Lo "capamos" a 4.0.
    # Un PF de 1.5 ya es rentable. Un PF de 3.0 es una máquina.
    # Escalamos: PF * 20 (Capped en 80 pts).
    pf_capped = min(4.0, max(1.0, profit_factor))
    score_pf = pf_capped * 20.0
    
    # 3. Score Sharpe/Sortino (Peso Medio - Volatilidad)
    # Premia curvas de capital suaves hacia arriba.
    # Escalamos: Ratio * 10.
    score_risk_adj = max(0.0, sortino) * 10.0
    
    # 4. Penalización de Drawdown (No Lineal)
    # Queremos castigar severamente los DD profundos.
    # DD 10% -> Penalización baja.
    # DD 50% -> Penalización masiva.
    # Fórmula: (100 - DD) ^ 1.2 (La potencia hace que caer duela más)
    safe_dd = max(0.0, min(100.0, drawdown))
    score_dd = pow(100.0 - safe_dd, 1.2) * 0.5  # Factor de ajuste
    
    # --- Suma Total ---
    # Base 1.0 para superar siempre al régimen inactivo
    total_score = 1.0 + score_sqn + score_pf + score_risk_adj + score_dd
    
    return float(total_score)