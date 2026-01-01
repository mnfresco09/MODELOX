"""
MODELOX package - High-Performance Backtesting & Optimization Engine.

Ultra-modular architecture:
- core: Engine, data, types
- strategies: Registry of trading strategies
- reporting: CSV, Excel, Rich console, Plotly charts

Adding new indicators:
1. Implement in logic/indicators.py
2. Use the cfg_* helpers (also in logic/indicators.py) inside the strategy
3. visual/grafico.py will plot what the trial used via params["__indicators_used"]

This repo started as a script-based project. The `modelox/` package provides a
clean, modular core so adding new strategies ("combinaciones") is plug & play.
"""

__version__ = "7.0.0"
__author__ = "MODELOX Quant Team"

__all__ = []


