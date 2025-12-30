from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional

from modelox.core.types import Reporter, TrialArtifacts
from visual.plot import plot_trades


@dataclass
class PlotReporter(Reporter):
    """Lightweight Charts (TradingView) HTML exporter.

    Features:
    - TradingView-style interactive charts with hollow/solid candles
    - Intelligent LOD downsampling for 50k+ candles
    - Synchronized subplots: Volume, RSI, MACD, Equity
    - Trade markers with PnL tooltips
    - Direct Polars/Pandas support

    Nota:
    - `TrialArtifacts.params` contiene los params "puros" del trial (los sugeridos por Optuna).
    - `TrialArtifacts.params_reporting` contiene params internos para reporting (p.ej. __indicators_used).
    Para el plot debemos usar `params_reporting` para poder filtrar indicadores y mostrar best-so-far.
    """

    plot_base: str = "resultados/plots"
    fecha_inicio_plot: str = (
        "2025-01-01"  # Valores por defecto (se sobrescriben desde ejecutar.py)
    )
    fecha_fin_plot: str = (
        "2025-01-20"  # Valores por defecto (se sobrescriben desde ejecutar.py)
    )
    max_archivos: int = 5  # Número máximo de plots a mantener según score
    saldo_inicial: float = 300.0  # Saldo inicial para calcular equity curve
    activo: Optional[str] = None  # Activo (BTC, GOLD, SP) para mostrar en el plot

    def _get_existing_scores(self) -> List[float]:
        """Obtiene los scores de los plots existentes."""
        if not os.path.exists(self.plot_base):
            return []

        try:
            # New format: TRIAL-{n}_SC-{score}_{combo}.html
            all_files = [
                f
                for f in os.listdir(self.plot_base)
                if f.endswith(".html") and f.startswith("TRIAL-")
            ]
            scores = []
            for f in all_files:
                match = re.search(r"TRIAL-\d+_SC-([\d.]+)_.*\.html", f)
                if match:
                    try:
                        score_val = float(match.group(1))
                        scores.append(score_val)
                    except ValueError:
                        continue
            return sorted(scores, reverse=True)  # Mejores primero
        except Exception:
            return []

    def _should_generate_plot(self, score: float) -> bool:
        """Determina si se debe generar el plot basado en el score."""
        if score is None:
            return False

        existing_scores = self._get_existing_scores()

        # Si hay menos archivos que el máximo, siempre generar
        if len(existing_scores) < self.max_archivos:
            return True

        # Si el score es mejor que el peor de los N mejores, generar
        worst_of_best = (
            existing_scores[self.max_archivos - 1]
            if len(existing_scores) >= self.max_archivos
            else float("-inf")
        )
        return score > worst_of_best

    def on_trial_end(self, artifacts: TrialArtifacts) -> None:
        # Verificar si debemos generar el plot
        if not self._should_generate_plot(artifacts.score):
            return  # No generar si el score no es mejor que los guardados

        # Ensure directories exist (resultados/, resultados/plots/)
        os.makedirs(self.plot_base, exist_ok=True)

        # Usar params_reporting para que el plot pueda:
        # - filtrar indicadores por combinación (__indicators_used)
        # - mostrar best-so-far (__best_score_so_far) si se utiliza en títulos/labels
        params_for_plot = (
            getattr(artifacts, "params_reporting", None) or artifacts.params
        )

        plot_trades(
            df=artifacts.df_signals,
            df_trades=artifacts.trades,
            plot_base=self.plot_base,
            fecha_inicio_plot=self.fecha_inicio_plot,
            fecha_fin_plot=self.fecha_fin_plot,
            trial_number=artifacts.trial_number,
            params=params_for_plot,
            score=artifacts.score,
            combo=artifacts.strategy_name,
            metrics=artifacts.metrics,
            equity_curve=artifacts.equity_curve,
            saldo_inicial=self.saldo_inicial,
            max_archivos=self.max_archivos,
            activo=self.activo,
        )

    def on_strategy_end(self, strategy_name: str, study) -> None:
        return
