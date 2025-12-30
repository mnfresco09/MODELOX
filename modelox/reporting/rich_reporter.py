"""
Elegant Rich Console Reporter - Bloomberg/TradingView Style.
High-end financial terminal interface for trial results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List

from modelox.core.types import Reporter, TrialArtifacts
from visual.rich import mostrar_panel_elegante, mostrar_top_trials


@dataclass
class ElegantRichReporter(Reporter):
    """
    Bloomberg/TradingView-style Rich console reporter.
    
    Displays a professional 3-column panel after each trial:
    - RENDIMIENTO: Sharpe, Sortino, Winrate, Profit Factor, Max Drawdown
    - ESTADO FINANCIERO: Initial/Final Balance, Commissions, PnL, Gross
    - CONFIGURACIÓN: Trial params (clean) + active indicators
    
    Features:
    - Sobrio color palette (cyan, magenta, green, yellow)
    - Conditional coloring based on metric quality
    - Clean parameter display (no __ prefixes)
    - Best-so-far tracking
    """
    
    saldo_inicial: float = 300.0
    activo: str = ""
    _best_score: float = field(default=float("-inf"), init=False, repr=False)
    
    def on_trial_end(self, artifacts: TrialArtifacts) -> None:
        """Display elegant panel for completed trial."""
        
        # Track best score
        current_score = artifacts.score or 0
        best_so_far = None
        
        if current_score > self._best_score:
            self._best_score = current_score
        best_so_far = self._best_score if self._best_score > float("-inf") else None
        
        # Get indicators used
        indicadores = list(getattr(artifacts, "indicators_used", []))
        if not indicadores and artifacts.params:
            # Try to get from params
            raw = artifacts.params.get("__indicators_used", [])
            if isinstance(raw, (list, tuple)):
                indicadores = [str(x) for x in raw if x]
        
        # Display panel
        mostrar_panel_elegante(
            metrics=artifacts.metrics or {},
            params=artifacts.params or {},
            score=artifacts.score or 0,
            trial_num=artifacts.trial_number,
            saldo_inicial=self.saldo_inicial,
            indicadores_activos=indicadores,
            combo_str=artifacts.strategy_name or "",
            activo=self.activo,
            best_so_far=best_so_far,
        )
    
    def on_strategy_end(self, strategy_name: str, study) -> None:
        """Display top trials summary at strategy end."""
        if study and hasattr(study, 'trials') and study.trials:
            mostrar_top_trials(study, n=5)


# Legacy alias for backward compatibility
RichReporter = ElegantRichReporter
