from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd

from modelox.core.types import Reporter, TrialArtifacts
from visual.excel import convertir_resumen_csv_a_excel


@dataclass
class CSVReporter(Reporter):
    """
    CSV reporter - MUCHO más rápido que Excel, más simple que SQLite.

    Objetivos:
    - Resumen por estrategia (1 CSV por combinación)
    - Escritura verdaderamente incremental (append) para máxima velocidad
    - Trades individuales top-K por score (por estrategia), como ya haces con Excel/plots
    """

    trades_base_dir: str = "resultados/csv"
    resumen_base_dir: str = "resultados/csv/resumen"
    # Carpeta destino para los Excels finales (uno por estrategia)
    resumen_excel_dir: str = "resultados/excel"
    max_archivos: int = 5  # Top-K trades CSV por estrategia (según score)

    def __post_init__(self):
        os.makedirs(self.trades_base_dir, exist_ok=True)
        os.makedirs(self.resumen_base_dir, exist_ok=True)
        os.makedirs(self.resumen_excel_dir, exist_ok=True)

    @staticmethod
    def _safe_strategy_name(strategy_name: str) -> str:
        return str(strategy_name).replace("+", "_").replace(" ", "_").upper()

    def _resumen_path_for(self, strategy_name: str) -> str:
        safe = self._safe_strategy_name(strategy_name)
        return os.path.join(self.resumen_base_dir, f"resumen_trials_{safe}.csv")

    def _get_existing_scores(self, strategy_name: str) -> List[float]:
        """Obtiene los scores de los trades CSV existentes para una estrategia."""
        if not os.path.exists(self.trades_base_dir):
            return []

        try:
            base_name = (
                f"trades_trial_{strategy_name.replace('+', '_').replace(' ', '_')}"
            )
            existing = [
                f
                for f in os.listdir(self.trades_base_dir)
                if f.startswith(base_name) and f.endswith(".csv")
            ]

            scores: List[float] = []
            for f in existing:
                score_match = re.search(r"score_([\d_]+)\.csv", f)
                if score_match:
                    try:
                        score_from_file = float(score_match.group(1).replace("_", "."))
                        scores.append(score_from_file)
                    except (ValueError, TypeError):
                        continue
            return sorted(scores, reverse=True)  # Mejores primero
        except Exception:
            return []

    def _should_save_trades(self, strategy_name: str, score: float) -> bool:
        """Determina si se debe guardar el CSV de trades basado en el score (top-K)."""
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

    def _flatten_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aplana params para CSV, excluyendo claves internas/reservadas (p.ej. __indicators_used).
        """
        out: Dict[str, Any] = {}
        for k, v in (params or {}).items():
            if str(k).startswith("__"):
                continue
            # Evitar volcar listas grandes (p.ej. indicadores_used) incluso si alguien cambia el prefijo
            if isinstance(v, (list, tuple, set, dict)):
                out[f"param_{k}"] = str(v)
            else:
                out[f"param_{k}"] = v
        return out

    def on_trial_end(self, artifacts: TrialArtifacts) -> None:
        """
        Guardar trial con append real:
        - NO leer CSV previo
        - escribir header solo si el archivo no existe
        """
        resumen_path = self._resumen_path_for(artifacts.strategy_name)
        write_header = not os.path.exists(resumen_path)

        # Normalizar columnas (mayúsculas como en Excel) + columna explícita de estrategia
        metrics = {str(k).upper(): v for k, v in (artifacts.metrics or {}).items()}
        row: Dict[str, Any] = {
            "ESTRATEGIA": self._safe_strategy_name(artifacts.strategy_name),
            "TRIAL": artifacts.trial_number,
            "SCORE": artifacts.score,
            **metrics,
            **self._flatten_params(dict(artifacts.params)),
        }

        pd.DataFrame([row]).to_csv(
            resumen_path,
            index=False,
            mode="a",
            header=write_header,
        )

        # Trades individuales SOLO si entra en top-K (por estrategia)
        if self._should_save_trades(artifacts.strategy_name, artifacts.score):
            score_str = (
                f"{artifacts.score:.2f}".replace(".", "_")
                if artifacts.score is not None
                else "unknown"
            )
            safe = artifacts.strategy_name.replace("+", "_").replace(" ", "_")
            trades_filename = (
                f"trades_trial_{safe}_{artifacts.trial_number}_score_{score_str}.csv"
            )
            trades_path = os.path.join(self.trades_base_dir, trades_filename)

            df_trades = artifacts.trades.copy()
            df_trades["TRIAL"] = artifacts.trial_number
            df_trades["SCORE"] = artifacts.score
            df_trades["ESTRATEGIA"] = self._safe_strategy_name(artifacts.strategy_name)

            df_trades.to_csv(trades_path, index=False, mode="w")

            if artifacts.score is not None and self.max_archivos > 0:
                self._cleanup_old_files(artifacts.strategy_name)

    def _cleanup_old_files(self, strategy_name: str) -> None:
        """Elimina trades CSV antiguos manteniendo solo los mejores scores (top-K) por estrategia."""
        try:
            base_name = (
                f"trades_trial_{strategy_name.replace('+', '_').replace(' ', '_')}"
            )
            existing = [
                f
                for f in os.listdir(self.trades_base_dir)
                if f.startswith(base_name) and f.endswith(".csv")
            ]

            files_with_scores = []
            for f in existing:
                score_match = re.search(r"score_([\d_]+)\.csv", f)
                if score_match:
                    try:
                        score_from_file = float(score_match.group(1).replace("_", "."))
                        files_with_scores.append((score_from_file, f))
                    except (ValueError, TypeError):
                        files_with_scores.append((float("-inf"), f))
                else:
                    files_with_scores.append((float("-inf"), f))

            files_with_scores.sort(key=lambda x: x[0], reverse=True)

            if len(files_with_scores) > self.max_archivos:
                for _, fname in files_with_scores[self.max_archivos :]:
                    file_path = os.path.join(self.trades_base_dir, fname)
                    if os.path.exists(file_path):
                        os.remove(file_path)
        except Exception:
            pass

    def on_strategy_end(self, strategy_name: str, study) -> None:
        # Al terminar la estrategia, convertir su CSV de resumen (rápido) al Excel final con formato pro.
        # Puede ocurrir que no exista CSV (p.ej. si nunca se llamó on_trial_end, o si un flujo externo
        # abortó antes de escribir). En ese caso, no debemos crashear el proceso completo.
        csv_path = self._resumen_path_for(strategy_name)
        if not os.path.exists(csv_path):
            return

        safe = self._safe_strategy_name(strategy_name)
        excel_path = os.path.join(self.resumen_excel_dir, f"resumen_trials_{safe}.xlsx")
        convertir_resumen_csv_a_excel(
            csv_path=csv_path,
            excel_path=excel_path,
            strategy_name=strategy_name,
        )
        return
