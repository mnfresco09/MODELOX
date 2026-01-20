"""
Base Reporter Interface

Define el protocolo explícito para reporters con métodos de introspección.
"""

from __future__ import annotations

from typing import Any, Protocol

from modelox.core.types import TrialArtifacts


class BaseReporter(Protocol):
    """
    Protocolo base para todos los reporters.
    
    Todos los reporters deben implementar:
    - needs_dataframe(): Indica si necesita df_signals en Pandas
    - on_trial_end(): Procesa resultados de un trial
    - on_strategy_end(): Procesa resultados finales de estrategia
    """
    
    def needs_dataframe(self, score: float) -> bool:
        """
        Indica si este reporter necesita df_signals convertido a Pandas.
        
        Args:
            score: Score del trial actual
        
        Returns:
            True si necesita df_signals, False si puede trabajar sin él
        
        Nota:
            Este método permite optimizar la conversión Polars→Pandas,
            que es costosa para datasets grandes (50k+ velas).
        """
        ...
    
    def on_trial_end(self, artifacts: TrialArtifacts) -> None:
        """
        Procesa resultados de un trial completado.
        
        Args:
            artifacts: Artefactos del trial (params, metrics, trades, etc.)
        """
        ...
    
    def on_strategy_end(self, strategy_name: str, study: Any) -> None:
        """
        Procesa resultados finales de una estrategia.
        
        Args:
            strategy_name: Nombre de la estrategia
            study: Estudio de Optuna completado
        """
        ...
