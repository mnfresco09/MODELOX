"""
MODELOX package - High-Performance Backtesting & Optimization Engine.

Ultra-modular architecture:
- indicators_metadata: Centralized indicator metadata (colors, ranges, types)
- core: Engine, data, types
- strategies: Registry of trading strategies
- reporting: CSV, Excel, Rich console, Plotly charts

Adding new indicators:
1. Implement in logic/indicators.py
2. Add spec in modelox/strategies/indicator_specs.py
3. Register metadata in modelox/indicators_metadata.py
4. plot.py adapts automatically

This repo started as a script-based project. The `modelox/` package provides a
clean, modular core so adding new strategies ("combinaciones") is plug & play.
"""

__version__ = "7.0.0"
__author__ = "MODELOX Quant Team"

# Core exports
from modelox.indicators_metadata import (
    IndicatorRegistry,
    IndicatorMetadata,
    IndicatorRange,
    create_indicator_metadata_from_params,
)

__all__ = [
    "IndicatorRegistry",
    "IndicatorMetadata",
    "IndicatorRange",
    "create_indicator_metadata_from_params",
]


