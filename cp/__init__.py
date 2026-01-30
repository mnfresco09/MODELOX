"""
MODELOX C Extensions - Módulo de Aceleración Nuclear
=====================================================

Este módulo contiene extensiones en C/Cython para acelerar
las operaciones críticas del backtesting:

- simulate_trades_c: Simulación secuencial de trades (kernel principal)
- find_exits_c: Búsqueda de salidas SL/TP/Trailing
- compute_metrics_c: Cálculo de métricas estadísticas
- perturb_returns_c: Perturbación de retornos para validación
- compute_cvar_95_c: CVaR 95% para análisis de riesgo
- compute_equity_r2_c: R² de equity para estabilidad
- aggregate_neighbor_metrics_c: Agregación de métricas vecinales

USO:
    from cp import (
        simulate_trades_c,
        find_exits_c,
        compute_metrics_c,
        perturb_returns_c,
        C_AVAILABLE
    )
    
    if C_AVAILABLE:
        # Usar versiones C (más rápido)
        results = simulate_trades_c(...)
    else:
        # Fallback a Numba/Python
        results = simulate_trades_numba(...)

COMPILACIÓN:
    cd cp
    python setup.py build_ext --inplace
    
    O usando el Makefile:
    make build-c
"""

from __future__ import annotations

# Intentar importar extensiones C compiladas
C_AVAILABLE = False
_C_VERSION = "3.0.0"

try:
    from .nuclear_engine import (
        simulate_trades_c,
        find_exits_c,
        compute_metrics_c,
        compute_drawdown_c,
        compute_sharpe_c,
        compute_sqn_c,
        # Nuevas funciones v3
        perturb_returns_c,
        compute_cvar_95_c,
        compute_equity_r2_c,
        aggregate_neighbor_metrics_c,
    )
    C_AVAILABLE = True
except ImportError:
    # Fallback: funciones dummy que indican que C no está disponible
    def simulate_trades_c(*args, **kwargs):
        raise NotImplementedError("Extensión C no compilada. Ejecuta: cd cp && python setup.py build_ext --inplace")
    
    def find_exits_c(*args, **kwargs):
        raise NotImplementedError("Extensión C no compilada. Ejecuta: cd cp && python setup.py build_ext --inplace")
    
    def compute_metrics_c(*args, **kwargs):
        raise NotImplementedError("Extensión C no compilada. Ejecuta: cd cp && python setup.py build_ext --inplace")
    
    def compute_drawdown_c(*args, **kwargs):
        raise NotImplementedError("Extensión C no compilada. Ejecuta: cd cp && python setup.py build_ext --inplace")
    
    def compute_sharpe_c(*args, **kwargs):
        raise NotImplementedError("Extensión C no compilada. Ejecuta: cd cp && python setup.py build_ext --inplace")
    
    def compute_sqn_c(*args, **kwargs):
        raise NotImplementedError("Extensión C no compilada. Ejecuta: cd cp && python setup.py build_ext --inplace")
    
    def perturb_returns_c(*args, **kwargs):
        raise NotImplementedError("Extensión C no compilada. Ejecuta: cd cp && python setup.py build_ext --inplace")
    
    def compute_cvar_95_c(*args, **kwargs):
        raise NotImplementedError("Extensión C no compilada. Ejecuta: cd cp && python setup.py build_ext --inplace")
    
    def compute_equity_r2_c(*args, **kwargs):
        raise NotImplementedError("Extensión C no compilada. Ejecuta: cd cp && python setup.py build_ext --inplace")
    
    def aggregate_neighbor_metrics_c(*args, **kwargs):
        raise NotImplementedError("Extensión C no compilada. Ejecuta: cd cp && python setup.py build_ext --inplace")


def get_version() -> str:
    """Retorna la versión del módulo C."""
    return _C_VERSION


def is_available() -> bool:
    """Verifica si las extensiones C están disponibles."""
    return C_AVAILABLE


__all__ = [
    "C_AVAILABLE",
    "simulate_trades_c",
    "find_exits_c", 
    "compute_metrics_c",
    "compute_drawdown_c",
    "compute_sharpe_c",
    "compute_sqn_c",
    "get_version",
    "is_available",
]
