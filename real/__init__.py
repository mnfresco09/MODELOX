"""
MODELOX REAL TRADING MODULE
============================
Módulo para trading en vivo con exchanges reales.
Soporta BingX con futuros perpetuos y estándar.
"""

from .bingx_client import BingXClient
from .trader import RealTrader

__all__ = ["BingXClient", "RealTrader"]
