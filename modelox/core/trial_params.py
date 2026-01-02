"""
Trial Parameters Management

Centraliza la gestión de parámetros de trials de Optuna, separando:
- Parámetros de estrategia (sugeridos por strategy.suggest_params)
- Parámetros de salida (exits)
- Parámetros de runtime (__warmup_bars, __indicators_used, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class TrialParameters:
    """
    Contenedor unificado para todos los parámetros de un trial.
    
    Responsabilidades:
    1. Separar parámetros de estrategia, exits y runtime
    2. Filtrar parámetros para reporting (sin prefijos __)
    3. Facilitar serialización y acceso estructurado
    """
    
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    exit_params: Dict[str, Any] = field(default_factory=dict)
    runtime_params: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_merged(cls, params: Dict[str, Any]) -> TrialParameters:
        """
        Crea TrialParameters desde un diccionario unificado.
        Separa automáticamente por prefijos.
        """
        strategy_params = {}
        exit_params = {}
        runtime_params = {}
        
        for key, value in params.items():
            if key.startswith("__"):
                # Runtime params: __warmup_bars, __indicators_used, etc.
                runtime_params[key] = value
            elif key.startswith("exit_"):
                # Exit params: exit_sl_atr, exit_tp_atr, etc.
                exit_params[key] = value
            else:
                # Strategy params: hl_period, ma_type, etc.
                strategy_params[key] = value
        
        return cls(
            strategy_params=strategy_params,
            exit_params=exit_params,
            runtime_params=runtime_params,
        )
    
    def to_merged(self) -> Dict[str, Any]:
        """Combina todos los parámetros en un diccionario unificado."""
        merged = {}
        merged.update(self.strategy_params)
        merged.update(self.exit_params)
        merged.update(self.runtime_params)
        return merged
    
    def to_reporting(self) -> Dict[str, Any]:
        """
        Retorna parámetros para reporting (sin __ prefijos).
        Útil para mostrar en consola, Excel, etc.
        """
        reporting = {}
        reporting.update(self.strategy_params)
        reporting.update(self.exit_params)
        
        # Incluir runtime params sin el prefijo __
        for key, value in self.runtime_params.items():
            if key.startswith("__"):
                clean_key = key[2:]  # Remove __
                reporting[clean_key] = value
        
        return reporting
    
    def get_warmup_bars(self) -> int:
        """Helper para obtener warmup_bars (valor común)."""
        return int(self.runtime_params.get("__warmup_bars", 0))
    
    def get_indicators_used(self) -> list[str]:
        """Helper para obtener indicadores usados."""
        return list(self.runtime_params.get("__indicators_used", []))
    
    def set_warmup_bars(self, bars: int) -> None:
        """Helper para establecer warmup_bars."""
        self.runtime_params["__warmup_bars"] = bars
    
    def set_indicators_used(self, indicators: list[str]) -> None:
        """Helper para establecer indicadores usados."""
        self.runtime_params["__indicators_used"] = indicators
