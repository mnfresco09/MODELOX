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

    resumen_path: str = "resultados/excel/resumen.xlsx"
    trades_base_dir: str = "resultados/excel"
    max_archivos: int = 5  # Número máximo de Excel a mantener según score

    @staticmethod
    def _safe_activo_name(activo: str) -> str:
        return str(activo).strip().replace(" ", "_").upper() if activo else "DEFAULT"

    def _excel_dir_for(self, activo: str) -> str:
        """Ruta final por activo.

        `trades_base_dir` se asume como raíz por estrategia, p.ej.:
          resultados/<ESTRATEGIA>/excel

        y aquí añadimos la carpeta del activo:
          resultados/<ESTRATEGIA>/excel/<ACTIVO>
        """
        return os.path.join(self.trades_base_dir, self._safe_activo_name(activo))

    def _get_existing_scores(self, base_dir: str) -> List[float]:
        """Obtiene los scores de los Excel existentes dentro del directorio."""
        if not os.path.exists(base_dir):
            return []

        try:
            existing = [
                f
                for f in os.listdir(base_dir)
                if f.endswith(".xlsx") and f.startswith("TRIAL-")
            ]

            scores = []
            for f in existing:
                # Buscar score en el nombre del archivo: TRIAL-{n}_SCORE-{score}.xlsx
                score_match = re.search(r"TRIAL-\d+_SCORE-([\d.]+)\.xlsx", f)
                if score_match:
                    try:
                        score_from_file = float(score_match.group(1))
                        scores.append(score_from_file)
                    except (ValueError, TypeError):
                        continue
            return sorted(scores, reverse=True)  # Mejores primero
        except Exception:
            return []

    def _should_save_trades(self, base_dir: str, score: float) -> bool:
        """Determina si se debe guardar el Excel de trades basado en el score."""
        if score is None:
            return False

        existing_scores = self._get_existing_scores(base_dir)

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
        # Resolve activo from params so we can route outputs as:
        # resultados/<ESTRATEGIA>/excel/<ACTIVO>/...
        params_src = getattr(artifacts, "params_reporting", None) or artifacts.params
        activo = None
        if isinstance(params_src, dict):
            activo = params_src.get("__activo") or params_src.get("ACTIVO") or params_src.get("activo")
        base_dir = self._excel_dir_for(str(activo) if activo is not None else "DEFAULT")
        os.makedirs(base_dir, exist_ok=True)

        resumen_path = os.path.join(base_dir, "resumen.xlsx")

        # Para Excel/exports usamos params_reporting (incluye __indicators_used, etc.)
        # para que el resumen/trades incluya correctamente la info de indicadores.
        params = dict(params_src)
        params["NOMBRE_COMBO"] = artifacts.strategy_name

        base = os.path.join(
            base_dir,
            f"trades_trial_{artifacts.strategy_name.replace('+', '_').replace(' ', '_')}",
        )

        # Guardar resumen SIEMPRE y trades SOLO si el score es mejor
        should_save_trades = self._should_save_trades(base_dir, artifacts.score)

        exportar_trades_excel(
            df_trades=artifacts.trades,
            resumen_path=resumen_path,
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
