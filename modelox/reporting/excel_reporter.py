from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List

from modelox.core.types import Reporter, TrialArtifacts
from visual.excel import exportar_trades_excel


@dataclass
class ExcelReporter(Reporter):
    """
    Excel exporter wrapper.

    We inject `NOMBRE_COMBO` into params so the exporter can create per-strategy files.
    Optimizado para solo guardar cuando el score es mejor que los guardados.
    """

    resumen_path: str = "resultados/excel/resumen_trials.xlsx"
    trades_base_dir: str = "resultados/excel"
    max_archivos: int = 5  # Número máximo de Excel a mantener según score

    def _get_existing_scores(self, strategy_name: str) -> List[float]:
        """Obtiene los scores de los Excel existentes para una estrategia."""
        if not os.path.exists(self.trades_base_dir):
            return []

        try:
            base_name = (
                f"trades_trial_{strategy_name.replace('+', '_').replace(' ', '_')}"
            )
            existing = [
                f
                for f in os.listdir(self.trades_base_dir)
                if f.startswith(base_name) and f.endswith(".xlsx")
            ]

            scores = []
            for f in existing:
                # Buscar score en el nombre del archivo: trades_trial_..._score_{score}.xlsx
                # El formato es: score_12_34 (donde 12.34 es el score con punto reemplazado por guion bajo)
                score_match = re.search(r"score_([\d_]+)\.xlsx", f)
                if score_match:
                    try:
                        score_str = score_match.group(1).replace("_", ".")
                        score_from_file = float(score_str)
                        scores.append(score_from_file)
                    except (ValueError, TypeError):
                        continue
            return sorted(scores, reverse=True)  # Mejores primero
        except Exception:
            return []

    def _should_save_trades(self, strategy_name: str, score: float) -> bool:
        """Determina si se debe guardar el Excel de trades basado en el score."""
        if score is None:
            return False

        existing_scores = self._get_existing_scores(strategy_name)

        # Si hay menos archivos que el máximo, siempre guardar
        if len(existing_scores) < self.max_archivos:
            return True

        # Si el score es mejor que el peor de los N mejores, guardar
        worst_of_best = (
            existing_scores[self.max_archivos - 1]
            if len(existing_scores) >= self.max_archivos
            else float("-inf")
        )
        return score > worst_of_best

    def on_trial_end(self, artifacts: TrialArtifacts) -> None:
        # Ensure directories exist (resultados/, resultados/excel/)
        resumen_dir = os.path.dirname(self.resumen_path)
        if resumen_dir:
            os.makedirs(resumen_dir, exist_ok=True)
        os.makedirs(self.trades_base_dir, exist_ok=True)

        # Para Excel/exports usamos params_reporting (incluye __indicators_used, etc.)
        # para que el resumen/trades incluya correctamente la info de indicadores.
        params_src = getattr(artifacts, "params_reporting", None) or artifacts.params
        params = dict(params_src)
        params["NOMBRE_COMBO"] = artifacts.strategy_name

        base = f"{self.trades_base_dir}/trades_trial_{artifacts.strategy_name.replace('+', '_').replace(' ', '_')}"

        # Guardar resumen SIEMPRE y trades SOLO si el score es mejor
        should_save_trades = self._should_save_trades(
            artifacts.strategy_name, artifacts.score
        )

        exportar_trades_excel(
            df_trades=artifacts.trades,
            resumen_path=self.resumen_path,
            metrics=artifacts.metrics,
            params=params,
            trial_number=artifacts.trial_number,
            trades_actual_base=base,
            score=artifacts.score,
            max_archivos=self.max_archivos,
            skip_trades_file=not should_save_trades,
        )

    def on_strategy_end(self, strategy_name: str, study) -> None:
        return
