from __future__ import annotations
from typing import Any, Dict
import math

"""
MODELOX - Progressive Scoring Engine (v2.5)
------------------------------------------
Arquitectura de puntuación institucional que equilibra:
1. Calidad (Sortino)
2. Potencia (Expectativa)
3. Consistencia (Estabilidad)
4. Seguridad (Drawdown)
5. Frecuencia (Trades/Día)
6. Confianza (Volumen Total)
"""

def _score_progresivo(valor: float, v_min: float, v_max: float, exponente: float = 1.0) -> float:
    """
    Normalización con curva de potencia. 
    Si exponente > 1, castiga más los valores mediocres y premia la excelencia.
    """
    if valor <= v_min:
        return 0.0
    if valor >= v_max:
        return 1.0
    
    # Normalización base 0-1
    base = (valor - v_min) / (v_max - v_min)
    
    # Aplicar curva de potencia para progresividad
    return math.pow(base, exponente)

def score_maestro_modelox(metricas: Dict[str, Any]) -> float:
    """
    Cálculo del Score Maestro con penalización por baja frecuencia y volumen.
    """
    
    # --- 1. EXTRACCIÓN Y LIMPIEZA DE DATOS ---
    n_trades = float(metricas.get("total_trades", 0))
    tpd = float(metricas.get("trades_por_dia", 0.0))
    
    # Filtro de seguridad absoluta (Menos de 10 trades en total es ruido)
    if n_trades < 10:
        return 0.0

    # --- 2. FACTOR DE FRECUENCIA (TRADES POR DÍA) ---
    # Buscamos el "Sweet Spot" para 5 minutos: entre 0.5 y 3.0 trades/día
    if tpd < 0.2:
        s_frecuencia = 0.4  # Muy lenta: penalización fuerte
    elif 0.2 <= tpd < 0.5:
        s_frecuencia = 0.7  # Aceptable pero lenta
    elif 0.5 <= tpd <= 3.5:
        s_frecuencia = 1.0  # Frecuencia óptima para Scalping/Intradía
    elif 3.5 < tpd <= 6.0:
        s_frecuencia = 0.8  # Empieza a ser Overtrading (comisiones)
    else:
        s_frecuencia = 0.5  # Demasiados trades, probable ruido/errores

    # --- 3. CÁLCULO DE COMPONENTES (PROGRESIVOS) ---
    
    # Sortino: Calidad del retorno. Ideal > 1.2. Exponente 1.5 para premiar lo mejor.
    s_sortino = _score_progresivo(float(metricas.get("sortino", 0.0)), 0.0, 1.5, exponente=1.5)
    
    # Expectativa: Valor esperado en $ por trade. 
    s_expectativa = _score_progresivo(float(metricas.get("expectativa", 0.0)), 0.0, 50.0)
    
    # Estabilidad: Suavidad de la equity (0.0 a 1.0).
    s_estabilidad = float(metricas.get("estabilidad", 0.0))
    
    # Drawdown: Seguridad. 5% es perfecto, 20% es el límite aceptable.
    s_dd = 1.0 - _score_progresivo(float(metricas.get("drawdown", 100.0)), 5.0, 20.0, exponente=1.3)
    s_dd = max(0.0, s_dd)

    # --- 4. DISTRIBUCIÓN DE PESOS (TOTAL 100) ---
    # 30% Calidad, 25% Potencia, 25% Curva, 20% Riesgo
    total_ponderado = (
        30.0 * s_sortino +
        25.0 * s_expectativa +
        25.0 * s_estabilidad +
        20.0 * s_dd
    )

    # --- 5. MULTIPLICADORES FINALES (EL FILTRO DE VERDAD) ---
    
    # Confianza Estadística: Crece con el número de trades (Raíz cuadrada)
    # 50 trades o más = 1.0. 25 trades = 0.7. 10 trades = 0.44.
    confianza = min(1.0, math.sqrt(n_trades / 50.0))
    
    # Score Final: Puntos * Volumen * Frecuencia
    score_final = total_ponderado * confianza * s_frecuencia

    return float(round(score_final, 2))

# Alias para compatibilidad con el resto del sistema
def score_optuna(metricas: Dict[str, Any]) -> float:
    return score_maestro_modelox(metricas)