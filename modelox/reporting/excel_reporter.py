from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from modelox.core.types import TrialArtifacts
from modelox.reporting.base import BaseReporter
# Importamos solo lo que existe en visual/excel.py v3.0
from visual.excel import exportar_trades_excel_rapido, convertir_resumen_csv_a_excel


@dataclass
class ExcelReporter(BaseReporter):
    """
    Excel exporter wrapper - OPTIMIZADO (v3.0 Compatible).

    Mejoras de velocidad:
    - Usa SIEMPRE CSV append durante trials (100x más rápido que Excel)
    - Convierte CSV→Excel Dashboard PRO solo al final de la estrategia
    """

    resumen_path: str = "resultados/excel/resumen.xlsx"  # Legacy
    trades_base_dir: str = "resultados/excel"
    max_archivos: int = 5  # Número máximo de Excel a mantener según score
    use_fast_mode: bool = True  # Deprecated: Ahora siempre es True implícitamente
    _csv_resumen_path: Optional[str] = field(default=None, init=False, repr=False)
    
    def needs_dataframe(self, score: float) -> bool:
        """ExcelReporter no necesita df_signals (solo usa trades)."""
        return False

    @staticmethod
    def _safe_activo_name(activo: str) -> str:
        return str(activo).strip().replace(" ", "_").upper() if activo else "DEFAULT"

    def _excel_dir_for(self, activo: str) -> str:
        return os.path.join(self.trades_base_dir, self._safe_activo_name(activo))

    def _get_existing_scores(self, base_dir: str) -> List[float]:
        """Obtiene los scores de los Excel existentes dentro del directorio."""
        if not os.path.exists(base_dir):
            return []

        try:
            existing = [
                f for f in os.listdir(base_dir)
                if f.endswith(".xlsx") and f.startswith("TRADES_TRIAL") # v3.0 prefix
            ]
            
            # Fallback legacy prefix
            if not existing:
                existing = [
                    f for f in os.listdir(base_dir)
                    if f.endswith(".xlsx") and f.startswith("TRIAL-")
                ]

            scores = []
            for f in existing:
                # Regex v3.0: TRADES_TRIAL{n}_SCORE{score}.xlsx
                match = re.search(r"SCORE(-?[\d.]+)\\.xlsx", f)
                if match:
                    try:
                        scores.append(float(match.group(1)))
                    except ValueError:
                        continue
            return sorted(scores, reverse=True)
        except Exception:
            return []

    def _should_save_trades(self, base_dir: str, score: float) -> bool:
        """Determina si se debe guardar el Excel de trades basado en el score."""
        if score is None:
            return False

        existing_scores = self._get_existing_scores(base_dir)

        if len(existing_scores) < self.max_archivos:
            return True

        worst_of_best = existing_scores[self.max_archivos - 1] if len(existing_scores) >= self.max_archivos else float("-inf")
        return score > worst_of_best

    def on_trial_end(self, artifacts: TrialArtifacts) -> None:
        params_src = getattr(artifacts, "params_reporting", None) or artifacts.params
        activo = None
        if isinstance(params_src, dict):
            activo = params_src.get("__activo") or params_src.get("ACTIVO") or params_src.get("activo")
        
        base_dir = self._excel_dir_for(str(activo) if activo is not None else "DEFAULT")
        os.makedirs(base_dir, exist_ok=True)

        # Configurar paths
        activo_safe = self._safe_activo_name(str(activo) if activo is not None else "DEFAULT")
        resumen_xlsx = os.path.join(base_dir, f"RESUMEN_{activo_safe}.xlsx")
        self._csv_resumen_path = resumen_xlsx.replace(".xlsx", ".csv")

        params = dict(params_src)
        params["NOMBRE_COMBO"] = artifacts.strategy_name

        # Determinar si guardar trades individuales
        should_save_trades = self._should_save_trades(base_dir, artifacts.score)

        # SIEMPRE usar modo rápido (v3.0 visual/excel solo soporta fast mode eficientemente)
        exportar_trades_excel_rapido(
            df_trades=artifacts.trades,
            resumen_csv_path=self._csv_resumen_path,
            metrics=artifacts.metrics,
            params=params,
            trial_number=artifacts.trial_number,
            trades_actual_base=os.path.join(base_dir, "trades"), # Base path
            score=artifacts.score,
            max_archivos=self.max_archivos,
            perturbado=artifacts.perturbado,
            perturb_seed=artifacts.perturb_seed,
            skip_trades_file=not should_save_trades,
        )

    def on_strategy_end(self, strategy_name: str, study) -> None:
        """Convierte CSV temporal a Excel PRO al final."""
        if self._csv_resumen_path and os.path.exists(self._csv_resumen_path):
            try:
                # Inferir activo del path o usar default
                # El path es .../excel/ACTIVO/RESUMEN_ACTIVO.csv
                parent_dir = os.path.dirname(self._csv_resumen_path)
                activo_name = os.path.basename(parent_dir)
                
                convertir_resumen_csv_a_excel(
                    csv_path=self._csv_resumen_path,
                    strategy_name=strategy_name,
                    activo=activo_name,
                    output_dir=parent_dir
                )
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Error generando Dashboard Excel: {e}")
