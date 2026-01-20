"""
modelox/core/__init__.py

Core package.

Nota importante:
- Este __init__ debe ser liviano y no importar dependencias pesadas (optuna/numba)
    ni módulos que puedan cambiar con refactors (runner/engine).
- Evita side-effects al hacer `import modelox.core` (por ejemplo, cuando
    `general/configuracion.py` importa `modelox.core.exits`).
"""

from .types import BacktestConfig, Reporter, Strategy, TrialArtifacts

__all__ = ["BacktestConfig", "Reporter", "Strategy", "TrialArtifacts"]