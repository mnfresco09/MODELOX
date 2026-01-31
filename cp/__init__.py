"""
# =============================================================================
#
#      ██████╗    ███████╗██╗  ██╗████████╗███████╗███╗   ██╗███████╗
#     ██╔════╝    ██╔════╝╚██╗██╔╝╚══██╔══╝██╔════╝████╗  ██║██╔════╝
#     ██║         █████╗   ╚███╔╝    ██║   █████╗  ██╔██╗ ██║███████╗
#     ██║         ██╔══╝   ██╔██╗    ██║   ██╔══╝  ██║╚██╗██║╚════██║
#     ╚██████╗    ███████╗██╔╝ ██╗   ██║   ███████╗██║ ╚████║███████║
#      ╚═════╝    ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═══╝╚══════╝
#
#     CP - EXTENSIONES C PARA ACELERACIÓN
#
# =============================================================================
#
#     FUNCIONES C/CYTHON:
#     - simulate_trades_c: Simulación de trades
#     - find_exits_c: Búsqueda de salidas SL/TP/Trailing
#     - compute_metrics_c: Métricas estadísticas
#
#     COMPILACIÓN:
#         cd cp && python setup.py build_ext --inplace
#
# =============================================================================
"""

from __future__ import annotations

C_AVAILABLE: bool = False
_C_VERSION: str = "3.0.0"

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
