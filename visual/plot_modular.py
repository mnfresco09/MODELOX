"""
Detector simple de indicadores - solo eso.
La visualización real está en plot.py
"""

from typing import Dict, Any
import pandas as pd


def detect_indicators(df: pd.DataFrame) -> Dict[str, str]:
    """
    Detecta qué indicadores están presentes en el DataFrame.
    Retorna {nombre: columna}
    """
    indicators = {}
    known = [
        'zscore', 'z_score', 'rsi', 'macd', 'macd_signal', 'macd_hist',
        'mfi', 'ema', 'sma', 'wma', 'dpo', 'adx', 'atr', 'roc',
        'stoch_k', 'stoch_d', 'cci',
    ]
    
    for col in df.columns:
        col_lower = col.lower()
        for indicator in known:
            if col_lower == indicator:
                indicators[indicator] = col
                break
    
    return indicators
