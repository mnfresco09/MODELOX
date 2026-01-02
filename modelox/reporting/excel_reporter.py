from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Any, Optional

from modelox.core.types import TrialArtifacts
from modelox.reporting.base import BaseReporter
from visual.excel import exportar_trades_excel, exportar_trades_excel_rapido


@dataclass
class ExcelReporter(BaseReporter):
    """
    Excel exporter wrapper - OPTIMIZADO.

    Mejoras de velocidad:
    - Usa CSV append durante trials (100x más rápido que Excel)
    - Convierte CSV→Excel solo al final de la estrategia
    - Nombre del resumen: RESUMEN_{ACTIVO}.xlsx
    
    Optimizado para solo guardar trades cuando el score es mejor que los guardados.
    """

    resumen_path: str = "resultados/excel/resumen.xlsx"  # Legacy, se sobrescribe
    trades_base_dir: str = "resultados/excel"
    max_archivos: int = 5  # Número máximo de Excel a mantener según score
    use_fast_mode: bool = True  # Si True, usa CSV append (mucho más rápido)
    _csv_resumen_path: Optional[str] = field(default=None, init=False, repr=False)
    
    def needs_dataframe(self, score: float) -> bool:
        """ExcelReporter no necesita df_signals (solo usa trades)."""
        return False

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

        # NUEVO: Nombre de archivo incluye activo
        activo_safe = self._safe_activo_name(str(activo) if activo is not None else "DEFAULT")
        resumen_path = os.path.join(base_dir, f"RESUMEN_{activo_safe}.xlsx")

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

        # OPTIMIZACIÓN: Usar CSV append durante trials (100x más rápido)
        if self.use_fast_mode:
            # Guardar CSV temporal para append rápido
            csv_path = resumen_path.replace(".xlsx", ".csv")
            self._csv_resumen_path = csv_path
            
            exportar_trades_excel_rapido(
                df_trades=artifacts.trades,
                resumen_csv_path=csv_path,
                metrics=artifacts.metrics,
                params=params,
                trial_number=artifacts.trial_number,
                trades_actual_base=base,
                score=artifacts.score,
                max_archivos=self.max_archivos,
                skip_trades_file=not should_save_trades,
            )
        else:
            # Modo legacy: escribir Excel directamente (más lento)
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
        """Convierte CSV temporal a Excel con formato al final de la estrategia."""
        if not self.use_fast_mode:
            return
        
        # Convertir CSV→Excel con formato profesional
        if self._csv_resumen_path and os.path.exists(self._csv_resumen_path):
            try:
                from visual.excel import convertir_resumen_csv_a_excel
                
                excel_path = self._csv_resumen_path.replace(".csv", ".xlsx")
                convertir_resumen_csv_a_excel(
                    csv_path=self._csv_resumen_path,
                    excel_path=excel_path,
                    strategy_name=strategy_name,
                )
                
                # Opcional: eliminar CSV temporal
                # os.remove(self._csv_resumen_path)
                
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Error convirtiendo CSV a Excel: {e}")
        
        return
