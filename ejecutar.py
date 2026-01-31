"""
# =============================================================================
#
#     ███████╗     ██╗███████╗ ██████╗██╗   ██╗████████╗ █████╗ ██████╗
#     ██╔════╝     ██║██╔════╝██╔════╝██║   ██║╚══██╔══╝██╔══██╗██╔══██╗
#     █████╗       ██║█████╗  ██║     ██║   ██║   ██║   ███████║██████╔╝
#     ██╔══╝  ██   ██║██╔══╝  ██║     ██║   ██║   ██║   ██╔══██║██╔══██╗
#     ███████╗╚█████╔╝███████╗╚██████╗╚██████╔╝   ██║   ██║  ██║██║  ██║
#     ╚══════╝ ╚════╝ ╚══════╝ ╚═════╝ ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝
#
#     EJECUTAR.PY - PUNTO DE ENTRADA PRINCIPAL
#
# =============================================================================
#
#     FLUJO:
#     1. Configuración de hilos para VM
#     2. Carga de datos (feather/csv)
#     3. Optimización bayesiana (CMA-ES / TPE / PLATEAU)
#     4. Generación de reportes (Excel, gráficos)
#
# =============================================================================
"""

from __future__ import annotations


# =============================================================================
# 1. CONFIGURACIÓN DE ALTO RENDIMIENTO (ANTES DE IMPORTS)
# =============================================================================

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"

os.environ.setdefault("MODELOX_TIMINGS", "1")
os.environ.setdefault("MODELOX_TIMINGS_PRINT_EVERY", "10")


# =============================================================================
# 2. IMPORTS DEL SISTEMA
# =============================================================================

import csv
import warnings
import logging
import atexit
import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol
import re

try:
    _PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    _PROJECT_ROOT = Path.cwd()

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# 3. CONFIGURACIÓN DE LOGGING
# =============================================================================

def setup_logging(level: int = logging.INFO) -> None:
    """CONFIGURA EL SISTEMA DE LOGGING."""
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        logging.getLogger("optuna").setLevel(logging.WARNING)
    except ImportError:
        pass


_pending_log_level = None
logger = logging.getLogger(__name__)


# =============================================================================
# 4. IMPORTS DEL PROYECTO
# =============================================================================

from general.configuracion import (
    ACTIVOS, ACTIVO_PRIMARIO, resolve_archivo_data, resolve_archivo_data_tf,
    resolve_qty_max_activo, resolve_qty_max_activo_range,
    COMBINACION_A_EJECUTAR, CONFIG,
    FECHA_FIN, FECHA_INICIO, N_TRIALS, FECHA_FIN_PLOT, FECHA_INICIO_PLOT,
    GENERAR_PLOTS, MAX_ARCHIVOS_GUARDAR, USAR_EXCEL,
    PERTURBACION_ACTIVAR,
    OPTUNA_SAMPLER,
    CLEANUP_INTERVAL,
    PLATEAU_EXPLORATION_RATIO,
    PLATEAU_EXPLORATION_SAMPLER,
    PLATEAU_MIN_CLUSTER_SIZE,
    PLATEAU_MIN_SAMPLES,
    PLATEAU_DBSCAN_EPS,
    PLATEAU_MIN_TRIALS_FOR_MESETA,
    PLATEAU_MAX_MESETAS,
    PLATEAU_MIN_TRIALS_POR_MESETA,
    PLATEAU_CENTROID_SELECTION,
    PLATEAU_AUTO_EPS,
)
from modelox.core.data import load_data
from modelox.core.runner import OptimizationRunner, OptunaConfig, PerturbationConfig

_log_level_map = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}
_configured_log_level = _log_level_map.get(str(CONFIG.get("LOG_LEVEL", "WARNING")).upper(), logging.WARNING)
setup_logging(level=_configured_log_level)
from modelox.core.plateau_optimizer import (
    PlateauOptimizerConfig,
    run_plateau_optimization,
)
from modelox.core.topology import PlateauConfig
from modelox.core.types import normalize_timeframe_to_suffix, BacktestConfig, filter_by_date, nuclear_cleanup, full_system_cleanup, TrialArtifacts
from modelox.strategies.registry import instantiate_strategies
from visual.excel import convertir_resumen_csv_a_excel
from visual.grafico import plot_trades
from visual.rich import (
    mostrar_cabecera_inicio, mostrar_fin_optimizacion,
    mostrar_panel_elegante, mostrar_top_trials, actualizar_estadisticas,
    mostrar_evolucion_metricas,
    resetear_estadisticas,
)

# Configurar intervalo de limpieza desde configuracion.py
os.environ["MODELOX_CLEANUP_INTERVAL"] = str(CLEANUP_INTERVAL)

# ============================================================================
# REPORTERS INTEGRADOS (fusionados de modelox/reporting/)
# ============================================================================

class BaseReporter(Protocol):
    """
    Protocolo base para todos los reporters.
    """

    def needs_dataframe(self, score: float) -> bool:
        """Indica si este reporter necesita df_signals convertido a Pandas."""
        ...

    def on_trial_end(self, artifacts: TrialArtifacts) -> None:
        """Procesa resultados de un trial completado."""
        ...

    def on_strategy_end(self, strategy_name: str, study: Any) -> None:
        """Procesa resultados finales de una estrategia."""
        ...


@dataclass
class ExcelReporter(BaseReporter):
    """
    Excel exporter wrapper - ULTRA OPTIMIZADO (v3.1).
    """

    resumen_path: str = "resultados/excel/resumen.xlsx"
    trades_base_dir: str = "resultados/excel"
    max_archivos: int = 5
    use_fast_mode: bool = True
    
    _csv_resumen_path: Optional[str] = field(default=None, init=False, repr=False)
    _resumen_rows: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _trade_candidates: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _min_candidate_score: float = field(default=float("-inf"), init=False, repr=False)
    _activo: Optional[str] = field(default=None, init=False, repr=False)

    def needs_dataframe(self, score: float) -> bool:
        return False

    @staticmethod
    def _safe_activo_name(activo: str) -> str:
        return str(activo).strip().replace(" ", "_").upper() if activo else "DEFAULT"

    def _excel_dir_for(self, activo: str) -> str:
        # Ya no creamos subcarpeta por activo, usamos trades_base_dir directamente
        return self.trades_base_dir

    def _update_min_score(self):
        if self._trade_candidates:
            self._min_candidate_score = min(c["score"] for c in self._trade_candidates)
        else:
            self._min_candidate_score = float("-inf")

    def on_trial_end(self, artifacts: TrialArtifacts) -> None:
        params_src = getattr(artifacts, "params_reporting", None) or artifacts.params
        activo = None
        if isinstance(params_src, dict):
            activo = params_src.get("__activo") or params_src.get("ACTIVO") or params_src.get("activo")
        
        self._activo = activo
        score = artifacts.score if artifacts.score is not None else 0.0
        
        params = dict(params_src)
        params["NOMBRE_COMBO"] = artifacts.strategy_name
        
        resumen_row = {
            "trial_number": artifacts.trial_number,
            "score": score,
            "metrics": deepcopy(artifacts.metrics) if artifacts.metrics else {},
            "params": {k: v for k, v in params.items() if not str(k).startswith("__")},
            "perturbado": artifacts.perturbado,
            "perturb_seed": artifacts.perturb_seed,
            "strategy_name": artifacts.strategy_name,
        }
        self._resumen_rows.append(resumen_row)

        try:
            base_dir = self.trades_base_dir
            os.makedirs(base_dir, exist_ok=True)
            if not self._csv_resumen_path:
                self._csv_resumen_path = os.path.join(base_dir, "RESUMEN.csv")
            self._write_resumen_csv(self._csv_resumen_path)
        except Exception:
            pass
        
        is_candidate = (
            len(self._trade_candidates) < self.max_archivos or 
            score > self._min_candidate_score
        )
        
        if is_candidate and artifacts.trades is not None:
            candidate = {
                "score": score,
                "trial_number": artifacts.trial_number,
                "trades": artifacts.trades,
                "params": params,
                "metrics": artifacts.metrics,
                "perturbado": artifacts.perturbado,
                "perturb_seed": artifacts.perturb_seed,
            }
            self._trade_candidates.append(candidate)
            
            if len(self._trade_candidates) > self.max_archivos:
                self._trade_candidates.sort(key=lambda x: x["score"], reverse=True)
                removed = self._trade_candidates.pop()
                del removed
            
            self._update_min_score()

    def on_strategy_end(self, strategy_name: str, study) -> None:
        if not self._resumen_rows:
            return
        
        activo = self._activo
        base_dir = self.trades_base_dir
        os.makedirs(base_dir, exist_ok=True)
        
        activo_safe = self._safe_activo_name(str(activo) if activo else "DEFAULT")
        csv_path = os.path.join(base_dir, "RESUMEN.csv")
        
        self._write_resumen_csv(csv_path)
        
        self._trade_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Guardar trades directamente en base_dir, sin subcarpeta trades
        for candidate in self._trade_candidates[:self.max_archivos]:
            try:
                self._write_trades_excel(base_dir, candidate)
            except Exception as e:
                logger.warning(f"Error guardando trades trial {candidate['trial_number']}: {e}")
        
        try:
            convertir_resumen_csv_a_excel(
                csv_path=csv_path,
                strategy_name=strategy_name,
                activo=activo_safe,
                output_dir=base_dir
            )
        except Exception as e:
            logger.warning(f"Error generando Dashboard Excel: {e}")
        
        self._resumen_rows = []
        self._trade_candidates = []
        self._min_candidate_score = float("-inf")

    def _write_resumen_csv(self, csv_path: str):
        if not self._resumen_rows:
            return
        
        all_keys = set()
        for row in self._resumen_rows:
            all_keys.add("trial")
            all_keys.add("score")
            all_keys.add("strategy")
            if row.get("metrics"):
                all_keys.update(row["metrics"].keys())
            if row.get("params"):
                all_keys.update(f"param_{k}" for k in row["params"].keys())
        
        columns = ["trial", "score", "strategy"]
        metric_cols = sorted([k for k in all_keys if k not in columns and not k.startswith("param_")])
        param_cols = sorted([k for k in all_keys if k.startswith("param_")])
        columns.extend(metric_cols)
        columns.extend(param_cols)
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            
            for row in self._resumen_rows:
                csv_row = {
                    "trial": row["trial_number"],
                    "score": row["score"],
                    "strategy": row["strategy_name"],
                }
                if row.get("metrics"):
                    csv_row.update(row["metrics"])
                if row.get("params"):
                    csv_row.update({f"param_{k}": v for k, v in row["params"].items()})
                
                writer.writerow(csv_row)

    def _write_trades_excel(self, trades_dir: str, candidate: Dict[str, Any]):
        import pandas as pd
        
        trades = candidate["trades"]
        if trades is None or (hasattr(trades, "empty") and trades.empty):
            return
        
        if hasattr(trades, "to_pandas"):
            df_trades = trades.to_pandas()
        else:
            df_trades = trades

        try:
            df_trades = df_trades.copy()
            if isinstance(df_trades.index, pd.DatetimeIndex) and df_trades.index.tz is not None:
                df_trades.index = df_trades.index.tz_localize(None)
            for col in df_trades.columns:
                try:
                    if isinstance(df_trades[col].dtype, pd.DatetimeTZDtype):
                        df_trades[col] = df_trades[col].dt.tz_localize(None)
                except Exception:
                    continue
            for col in df_trades.columns:
                if df_trades[col].dtype != object:
                    continue
                s = df_trades[col]
                try:
                    converted = pd.to_datetime(s, errors="ignore", utc=True)
                    if isinstance(converted.dtype, pd.DatetimeTZDtype):
                        df_trades[col] = converted.dt.tz_localize(None)
                        continue
                except Exception:
                    pass
                try:
                    sample = next((v for v in s.head(50).tolist() if v is not None), None)
                    if sample is None:
                        continue
                    tzinfo = getattr(sample, "tzinfo", None)
                    if tzinfo is None:
                        continue
                    df_trades[col] = s.apply(
                        lambda v: v.replace(tzinfo=None)
                        if hasattr(v, "tzinfo") and v.tzinfo is not None else v
                    )
                except Exception:
                    continue
        except Exception:
            pass
        
        score_str = f"{candidate['score']:.2f}".replace(".", "_")
        filename = f"TRADES_TRIAL{candidate['trial_number']}_SCORE{score_str}.xlsx"
        filepath = os.path.join(trades_dir, filename)
        
        df_trades.to_excel(filepath, index=False, sheet_name="Trades")


@dataclass
class PlotReporter(BaseReporter):
    """Lightweight Charts (TradingView) HTML exporter - OPTIMIZADO."""

    plot_base: str = "resultados/graficos"
    fecha_inicio_plot: str = "2025-01-01"
    fecha_fin_plot: str = "2025-01-20"
    max_archivos: int = 5
    saldo_inicial: float = 300.0
    activo: Optional[str] = None
    
    _candidates: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _min_candidate_score: float = field(default=float("-inf"), init=False, repr=False)

    def needs_dataframe(self, score: float) -> bool:
        if score is None:
            return False
        if len(self._candidates) < self.max_archivos:
            return True
        return score > self._min_candidate_score

    def _update_min_score(self):
        if self._candidates:
            self._min_candidate_score = min(c["score"] for c in self._candidates)
        else:
            self._min_candidate_score = float("-inf")

    def on_trial_end(self, artifacts: TrialArtifacts) -> None:
        if artifacts.trial_number == 0 and os.path.exists(self.plot_base):
            try:
                for f in os.listdir(self.plot_base):
                    if f.startswith("TRIAL-") and f.endswith(".html"):
                        os.remove(os.path.join(self.plot_base, f))
            except Exception:
                pass
            self._candidates = []
            self._min_candidate_score = float("-inf")

        score = artifacts.score
        if score is None:
            return

        is_candidate = (
            len(self._candidates) < self.max_archivos or 
            score > self._min_candidate_score
        )
        
        if not is_candidate:
            return
        
        if getattr(artifacts, "df_signals", None) is None:
            return

        params_for_plot = getattr(artifacts, "params_reporting", None) or artifacts.params
        
        candidate = {
            "score": score,
            "trial_number": artifacts.trial_number,
            "strategy_name": artifacts.strategy_name,
            "params": deepcopy(params_for_plot),
            "metrics": deepcopy(artifacts.metrics) if artifacts.metrics else {},
            "equity_curve": list(artifacts.equity_curve) if artifacts.equity_curve else [],
            "df_signals": artifacts.df_signals,
            "trades": artifacts.trades,
        }
        
        self._candidates.append(candidate)
        
        if len(self._candidates) > self.max_archivos:
            self._candidates.sort(key=lambda x: x["score"], reverse=True)
            removed = self._candidates.pop()
            del removed
        
        self._update_min_score()

    def on_strategy_end(self, strategy_name: str, study) -> None:
        if not self._candidates:
            return
        
        os.makedirs(self.plot_base, exist_ok=True)
        
        self._candidates.sort(key=lambda x: x["score"], reverse=True)
        
        for candidate in self._candidates[:self.max_archivos]:
            try:
                plot_trades(
                    df=candidate["df_signals"],
                    df_trades=candidate["trades"],
                    plot_base=self.plot_base,
                    fecha_inicio_plot=self.fecha_inicio_plot,
                    fecha_fin_plot=self.fecha_fin_plot,
                    trial_number=candidate["trial_number"],
                    params=candidate["params"],
                    score=candidate["score"],
                    combo=candidate["strategy_name"],
                    metrics=candidate["metrics"],
                    equity_curve=candidate["equity_curve"],
                    saldo_inicial=self.saldo_inicial,
                    max_archivos=self.max_archivos,
                    activo=self.activo,
                )
            except Exception as e:
                logger.warning(f"Error generando plot para trial {candidate['trial_number']}: {e}")
        
        self._candidates = []
        self._min_candidate_score = float("-inf")


@dataclass
class ElegantRichReporter(BaseReporter):
    """Bloomberg/TradingView-style Rich console reporter."""

    saldo_inicial: float = 300.0
    activo: str = ""
    mostrar_evolucion_cada: int = 50
    mostrar_evolucion_inline: bool = False  # Disabled - averages now shown in panel
    _best_score: float = field(default=float("-inf"), init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)
    
    # Información de fase (para Topógrafo de Mesetas)
    phase_name: str = ""
    phase_total_trials: int = 0
    _phase_trial_count: int = field(default=0, init=False, repr=False)
    
    def set_phase(self, name: str, total_trials: int = 0) -> None:
        """Establece la fase actual de optimización."""
        self.phase_name = name
        self.phase_total_trials = total_trials
        self._phase_trial_count = 0

    def needs_dataframe(self, score: float) -> bool:
        return False

    def on_trial_end(self, artifacts: TrialArtifacts) -> None:
        if not self._initialized:
            resetear_estadisticas()
            self._initialized = True

        current_score = artifacts.score or 0
        best_so_far = None

        if current_score > self._best_score:
            self._best_score = current_score
        best_so_far = self._best_score if self._best_score > float("-inf") else None

        indicadores = list(getattr(artifacts, "indicators_used", []))
        if not indicadores and artifacts.params:
            raw = artifacts.params.get("__indicators_used", [])
            if isinstance(raw, (list, tuple)):
                indicadores = [str(x) for x in raw if x]

        metrics = dict(artifacts.metrics or {})
        try:
            eq = getattr(artifacts, "equity_curve", None)
            if isinstance(eq, (list, tuple)) and len(eq) > 0:
                if "saldo_max" not in metrics or metrics.get("saldo_max") in (None, 0, 0.0):
                    metrics["saldo_max"] = float(max(eq))
        except Exception:
            pass

        actualizar_estadisticas(metrics, current_score, artifacts.trial_number)

        tf_entry = ""
        tf_exit = ""
        if artifacts.params:
            tf_entry = str(artifacts.params.get("__timeframe_entry", ""))
            tf_exit = str(artifacts.params.get("__timeframe_exit", ""))

        params_for_reporting = artifacts.params.copy() if artifacts.params else {}
        if '__qty_max_activo' in params_for_reporting:
            params_for_reporting['cantidad'] = params_for_reporting['__qty_max_activo']

        # Calcular progreso de fase
        self._phase_trial_count += 1
        phase_progress = ""
        if self.phase_total_trials > 0:
            phase_progress = f"{self._phase_trial_count}/{self.phase_total_trials}"

        mostrar_panel_elegante(
            metrics=metrics,
            params=params_for_reporting,
            score=artifacts.score or 0,
            trial_num=artifacts.trial_number,
            saldo_inicial=self.saldo_inicial,
            indicadores_activos=indicadores,
            combo_str=artifacts.strategy_name or "",
            activo=self.activo,
            best_so_far=best_so_far,
            timeframe_entry=tf_entry,
            timeframe_exit=tf_exit,
            neighborhood_result=artifacts.neighborhood_result,
            phase_name=self.phase_name,
            phase_progress=phase_progress,
        )

    def on_strategy_end(self, strategy_name: str, study) -> None:
        self._initialized = False
        if study and hasattr(study, 'trials') and study.trials:
            mostrar_evolucion_metricas(forzar=True)  # SUMMARY solo al final
            mostrar_top_trials(study, n=5)


# nuclear_cleanup ya importado desde types arriba

# ============================================================================
# LÓGICA DE EJECUCIÓN (SINGLE EXIT TYPE)
# ============================================================================
def run_single_exit_type(
    exit_type: str,
    strategy: object,
    strategy_name: str,
    strategy_safe: str,
    activo: str,
    df_filtrado: object,
    tf_cache: dict[str, object],
    timeframe_base: str,
    cfg: BacktestConfig,
    tf_display: str,
    archivo_data: str,
    periodo_datos: str,
    resolve_archivo_data_tf_func: object = None,
    fecha_inicio: str | None = None,
    fecha_fin: str | None = None,
) -> None:

    # 1. CONFIGURACIÓN
    cfg_dict = cfg.__dict__.copy()
    cfg_dict["exit_type"] = str(exit_type)
    cfg_updated = BacktestConfig(**cfg_dict)

    # 2. DETECCIÓN DE CAPACIDADES
    try:
        indicadores = list(getattr(strategy, "parametros_optuna", {}).keys())
    except Exception:
        indicadores = []

    try:
        strategy_exit_enabled = bool(
            callable(getattr(strategy, "decide_exit", None))
            and bool(getattr(strategy, "ACTIVAR_SALIDA_PERSONALIZADA", False))
        )
    except Exception:
        strategy_exit_enabled = False

    # 3. MOSTRAR HEADER
    mostrar_cabecera_inicio(
        activo=activo,
        combo_nombre=strategy_name,
        indicadores=indicadores,
        n_trials=int(N_TRIALS),
        archivo_data=archivo_data,
        timeframe=tf_display,
        periodo=periodo_datos,
        exit_type=exit_type,
        strategy_exit_enabled=strategy_exit_enabled,
        perturbacion_activar=PERTURBACION_ACTIVAR,
        sampler_type=OPTUNA_SAMPLER,
    )

    # 4. RUTAS DE SALIDA
    # Estructura: {RESULTADOS_BASE_DIR}/ESTRATEGIA/TRAILING|FIXED/ACTIVO/GRAFICA|EXCEL|ROBUSTEZ
    # RESULTADOS_BASE_DIR se importa de CONFIG (configurable en configuracion.py)
    from general.configuracion import CONFIG
    resultados_base = CONFIG.get("RESULTADOS_BASE_DIR", "resultados")
    activo_safe = str(activo).upper()
    exit_type_folder = "TRAILING" if "trail" in str(exit_type).lower() else "FIXED"
    strategy_root_dir = os.path.join(
        resultados_base,
        strategy_safe,
        exit_type_folder,
        activo_safe,
    )
    excel_dir = os.path.join(strategy_root_dir, "EXCEL")
    graficos_dir = os.path.join(strategy_root_dir, "GRAFICA")
    os.makedirs(excel_dir, exist_ok=True)
    os.makedirs(graficos_dir, exist_ok=True)

    # 5. REPORTEROS
    reporters = [ElegantRichReporter(saldo_inicial=cfg_updated.saldo_inicial, activo=activo)]

    if USAR_EXCEL:
        reporters.append(ExcelReporter(
            resumen_path=f"{excel_dir}/RESUMEN.xlsx",
            trades_base_dir=excel_dir,
            max_archivos=int(MAX_ARCHIVOS_GUARDAR)
        ))

    if GENERAR_PLOTS:
        reporters.append(PlotReporter(
            plot_base=graficos_dir,
            fecha_inicio_plot=FECHA_INICIO_PLOT,
            fecha_fin_plot=FECHA_FIN_PLOT,
            max_archivos=int(MAX_ARCHIVOS_GUARDAR),
            saldo_inicial=cfg_updated.saldo_inicial,
            activo=activo,
        ))

    # 6. RUNNER - Seleccionar modo de optimización
    use_plateau_mode = str(OPTUNA_SAMPLER).upper() == "PLATEAU"
    
    if use_plateau_mode:
        # =====================================================================
        # MODO TOPÓGRAFO DE MESETAS (3 FASES)
        # =====================================================================
        # Fase 1: Exploración masiva con RandomSampler
        # Fase 2: Detección de mesetas con DBSCAN
        # Fase 3: Refinamiento CMA-ES en cada meseta
        
        # Configurar el sistema de mesetas
        plateau_cfg = PlateauConfig(
            min_cluster_size=int(PLATEAU_MIN_CLUSTER_SIZE),
            min_samples=int(PLATEAU_MIN_SAMPLES),
            eps=float(PLATEAU_DBSCAN_EPS),
            min_trials_for_plateau=int(PLATEAU_MIN_TRIALS_FOR_MESETA),
            centroid_selection=str(PLATEAU_CENTROID_SELECTION),
        )
        
        optimizer_cfg = PlateauOptimizerConfig(
            exploration_ratio=float(PLATEAU_EXPLORATION_RATIO),
            exploration_sampler=str(PLATEAU_EXPLORATION_SAMPLER),
            plateau_config=plateau_cfg,
            auto_tune_dbscan=bool(PLATEAU_AUTO_EPS),
            min_trials_per_plateau=int(PLATEAU_MIN_TRIALS_POR_MESETA),
            max_plateaus_to_refine=int(PLATEAU_MAX_MESETAS),
            verbose=True,
        )
        
        # Configurar perturbación si está habilitada
        perturbation_config = None
        if PERTURBACION_ACTIVAR:
            perturbation_config = PerturbationConfig(
                enabled=True,
                method=CONFIG.get("PERTURBACION_METHOD", "returns_perturbation"),
                noise_factor=float(CONFIG.get("PERTURBACION_NOISE_SCALE", 0.5)),
                block_size=int(CONFIG.get("PERTURBACION_BLOCK_SIZE", 360)),
                seed=CONFIG.get("PERTURBACION_SEED", 42),
            )
        
        # Carga diferida de timeframes extra
        entry_tf = getattr(strategy, "timeframe_entry", None) or timeframe_base
        exit_tf = getattr(strategy, "timeframe_exit", None) or timeframe_base
        needed_tfs = [timeframe_base, entry_tf, exit_tf]
        
        for tf in needed_tfs:
            tf_suf = normalize_timeframe_to_suffix(tf)
            if tf_suf in tf_cache:
                continue
            
            if resolve_archivo_data_tf_func:
                try:
                    path_tf = resolve_archivo_data_tf_func(activo, tf, formato="parquet")
                    df_tf = load_data(path_tf)
                    if fecha_inicio and fecha_fin:
                        df_tf = filter_by_date(df_tf, fecha_inicio, fecha_fin)
                    tf_cache[tf_suf] = df_tf
                except Exception as e:
                    logger.warning(f"No se pudo cargar TF extra {tf}: {e}")
        
        try:
            # Ejecutar optimización por mesetas
            plateau_result = run_plateau_optimization(
                df=df_filtrado,
                strategy=strategy,
                backtest_config=cfg_updated,
                n_trials=int(N_TRIALS),
                reporters=reporters,
                plateau_config=optimizer_cfg,
                df_by_timeframe=tf_cache,
                base_timeframe=timeframe_base,
                perturbation_config=perturbation_config,
                activo=activo,
                seed=None,
            )
            
            # Mostrar resultado final
            if plateau_result.best_plateau:
                mostrar_fin_optimizacion(
                    total_trials=plateau_result.total_trials,
                    best_score=plateau_result.best_refined_score,
                    best_trial=plateau_result.best_refined_trial,
                    estrategia=strategy_name,
                )
            else:
                # Si no se encontraron mesetas, usar resultado de exploración
                mostrar_fin_optimizacion(
                    total_trials=plateau_result.total_trials,
                    best_score=plateau_result.best_exploration_score,
                    best_trial=plateau_result.phase1_exploration.best_trial_number,
                    estrategia=strategy_name,
                )
            
            # Notificar a reporters
            for reporter in reporters:
                if hasattr(reporter, "on_strategy_end"):
                    try:
                        # Usar el estudio de exploración para compatibilidad
                        reporter.on_strategy_end(strategy_name, plateau_result.exploration_study)
                    except Exception:
                        pass
                        
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(f"Error en optimización por mesetas {strategy_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            del reporters
            nuclear_cleanup()
    
    else:
        # =====================================================================
        # MODO CLÁSICO (CMA-ES o TPE)
        # =====================================================================
        
        # [FIX v2.1] Configurar perturbación si está habilitada (antes se ignoraba en este modo)
        perturbation_config = PerturbationConfig(enabled=False)  # Default desactivado
        if PERTURBACION_ACTIVAR:
            perturbation_config = PerturbationConfig(
                enabled=True,
                method=CONFIG.get("PERTURBACION_METHOD", "returns_perturbation"),
                noise_factor=float(CONFIG.get("PERTURBACION_NOISE_SCALE", 0.5)),
                block_size=int(CONFIG.get("PERTURBACION_BLOCK_SIZE", 360)),
                seed=CONFIG.get("PERTURBACION_SEED", 42),
            )
        
        runner = OptimizationRunner(
            config=cfg_updated, 
            n_trials=int(N_TRIALS), 
            reporters=reporters,
            perturbation_config=perturbation_config,  # Ahora se pasa la configuración
        )

        # [CAMBIO v2.0] Usar sampler configurado (CMA-ES por defecto para scoring institucional)
        # CMA-ES aprende de los scores y favorece regiones estables (mesetas de parámetros)
        runner.optuna = OptunaConfig(
            seed=None, 
            n_jobs=1, 
            storage=None,
            sampler=OPTUNA_SAMPLER,  # "CMA" o "TPE" desde configuracion.py
        )

        runner.activo = activo

        try:
            # Carga diferida de timeframes extra
            entry_tf = getattr(strategy, "timeframe_entry", None) or timeframe_base
            exit_tf = getattr(strategy, "timeframe_exit", None) or timeframe_base
            needed_tfs = [timeframe_base, entry_tf, exit_tf]

            for tf in needed_tfs:
                tf_suf = normalize_timeframe_to_suffix(tf)
                if tf_suf in tf_cache:
                    continue

                if resolve_archivo_data_tf_func:
                    try:
                        path_tf = resolve_archivo_data_tf_func(activo, tf, formato="parquet")
                        df_tf = load_data(path_tf)
                        if fecha_inicio and fecha_fin:
                            df_tf = filter_by_date(df_tf, fecha_inicio, fecha_fin)
                        tf_cache[tf_suf] = df_tf
                    except Exception as e:
                        logger.warning(f"No se pudo cargar TF extra {tf}: {e}")

            runner.optimize_strategies(
                df=df_filtrado,
                strategies=[strategy],
                df_by_timeframe=tf_cache,
                base_timeframe=timeframe_base,
            )

            # -------------------------------------------------------------------------
            # [CORRECCIÓN] VISUALIZACIÓN DE RESULTADOS COMPATIBLE CON NSGA-II (MULTI-OBJETIVO)
            # -------------------------------------------------------------------------
            if hasattr(runner, "_last_study") and runner._last_study:
                study = runner._last_study

                # Detectar si hay más de 1 objetivo
                is_multiobj = len(study.directions) > 1

                try:
                    best_trial = None
                    best_val = 0.0

                    if is_multiobj:
                        # NSGA-II: Obtenemos la Frontera de Pareto
                        # Ordenamos por Calidad (values[0]) descendente para coger el "mejor"
                        # Nota: values[0] = Calidad, values[1] = Riesgo (Drawdown)
                        pareto_front = sorted(study.best_trials, key=lambda t: t.values[0], reverse=True)
                        if pareto_front:
                            best_trial = pareto_front[0]
                            best_val = best_trial.values[0]
                    else:
                        # TPE Clásico: Existe un único best_trial
                        if study.best_trial:
                            best_trial = study.best_trial
                            best_val = study.best_value

                    # Mostrar resultado si se encontró un trial válido
                    if best_trial:
                        mostrar_fin_optimizacion(
                            total_trials=len(study.trials),
                            best_score=best_val,
                            best_trial=best_trial.number,
                            estrategia=strategy_name,
                        )
                except Exception as e:
                    # Fallback seguro para no detener la ejecución si Optuna cambia
                    logger.warning(f"No se pudo extraer el mejor trial para el reporte final: {e}")
            # -------------------------------------------------------------------------

        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(f"Error en {strategy_name}: {e}")
        finally:
            del runner
            del reporters
            # Limpieza nuclear para asegurar que la RAM se devuelve al SO
            nuclear_cleanup()

# ============================================================================
# LIMPIEZA Y CIERRE SEGURO
# ============================================================================
def _purge_pycache(*, root: Path, exclude: set[str]) -> None:
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            p = Path(dirpath)
            if any(part in exclude for part in p.parts):
                dirnames[:] = []
                continue
            if p.name == "__pycache__":
                shutil.rmtree(p, ignore_errors=True)
                dirnames[:] = []
                continue
            for f in filenames:
                if f.endswith(".pyc"):
                    try:
                        (p / f).unlink(missing_ok=True)
                    except OSError:
                        pass
    except OSError:
        pass

class HealthGuard:
    @staticmethod
    def final_cleanup():
        nuclear_cleanup()
        # Usamos la variable global _PROJECT_ROOT capturada al inicio
        if CONFIG.get("PURGE_PYCACHE_ON_EXIT"):
            _purge_pycache(root=_PROJECT_ROOT, exclude={".git", ".venv", "data"})

def main() -> None:
    atexit.register(HealthGuard.final_cleanup)

    # 1. PARSEAR ESTRATEGIAS
    if COMBINACION_A_EJECUTAR == "all":
        ids = None
    elif isinstance(COMBINACION_A_EJECUTAR, (list, tuple)):
        ids = list(COMBINACION_A_EJECUTAR)
    else:
        ids = [int(COMBINACION_A_EJECUTAR)]

    # 2. ACTIVOS Y TIMEFRAMES
    activos = list(ACTIVOS) if ACTIVOS else [ACTIVO_PRIMARIO]

    raw_tfs = CONFIG.get("TIMEFRAMES", None)
    fallback_tf = CONFIG.get("TIMEFRAME", 15)

    def _ensure_int_list(raw, fallback):
        res = []
        candidatos = raw if isinstance(raw, (list, tuple)) else str(raw).split(",")
        for c in candidatos:
            try:
                clean = str(c).lower().replace("m", "")
                val = int(float(clean))
                if val > 0:
                    res.append(val)
            except (ValueError, TypeError):
                pass
        return res if res else [int(fallback)]

    tfs_run = _ensure_int_list(raw_tfs, fallback_tf)

    # 3. EJECUCIÓN
    try:
        for activo in activos:
            for tf_base in tfs_run:

                # Carga de datos
                try:
                    archivo = resolve_archivo_data_tf(activo, tf_base, formato="parquet")
                    if not os.path.exists(archivo):
                        archivo = resolve_archivo_data(activo)
                except Exception:
                    archivo = resolve_archivo_data(activo)

                if not os.path.exists(archivo):
                    print(f"❌ DATA NOT FOUND: {archivo}")
                    continue

                df = load_data(archivo)
                df_filtrado = filter_by_date(df, FECHA_INICIO, FECHA_FIN)

                periodo_str = ""
                try:
                    ts = df_filtrado["timestamp"] if "timestamp" in df_filtrado.columns else df_filtrado["datetime"]
                    periodo_str = f"{ts.min():%Y-%m-%d} -> {ts.max():%Y-%m-%d}"
                except (KeyError, AttributeError):
                    pass

                tf_cache = {normalize_timeframe_to_suffix(tf_base): df_filtrado}

                cfg = BacktestConfig(
                    saldo_inicial=float(CONFIG["SALDO_INICIAL"]),
                    saldo_operativo_max=float(CONFIG["SALDO_OPERATIVO_MAX"]),
                    comision_pct=float(CONFIG["COMISION_PCT"]),
                    comision_sides=int(CONFIG["COMISION_SIDES"]),
                    saldo_minimo_operativo=float(CONFIG["SALDO_MINIMO_OPERATIVO"]),
                    qty_max_activo=float(resolve_qty_max_activo(activo)),
                    saldo_usado=float(CONFIG["SALDO_USADO"]),
                    apalancamiento_max=float(CONFIG["APALANCAMIENTO_MAX"]),
                    riesgo_por_trade_pct=float(CONFIG["RIESGO_POR_TRADE_PCT"]),
                    optimize_qty_max_activo=bool(CONFIG["OPTIMIZAR_QTY_ACTIVO"]),
                    qty_max_activo_range=tuple(resolve_qty_max_activo_range(activo)),
                    exit_sl_pct=float(CONFIG["EXIT_SL_PCT"]),
                    exit_tp_pct=float(CONFIG["EXIT_TP_PCT"]),
                    exit_trail_act_pct=float(CONFIG["EXIT_TRAIL_ACT_PCT"]),
                    exit_trail_dist_pct=float(CONFIG["EXIT_TRAIL_DIST_PCT"]),
                    optimize_exits=bool(CONFIG["OPTIMIZAR_SALIDAS"]),
                    exit_sl_pct_range=tuple(CONFIG["EXIT_SL_PCT_RANGE"]),
                    exit_tp_pct_range=tuple(CONFIG["EXIT_TP_PCT_RANGE"]),
                    exit_trail_act_pct_range=tuple(CONFIG["EXIT_TRAIL_ACT_PCT_RANGE"]),
                    exit_trail_dist_pct_range=tuple(CONFIG["EXIT_TRAIL_DIST_PCT_RANGE"]),
                )

                loop_ids = ids if ids else [None]
                for sid in loop_ids:
                    strategies = instantiate_strategies(only_id=sid if sid else None)
                    if not strategies:
                        continue

                    for strat in strategies:
                        s_name = getattr(strat, 'name', "STRAT")
                        s_safe = re.sub(r"[^A-Z0-9_]+", "_", s_name.upper())

                        base_suf = normalize_timeframe_to_suffix(tf_base)
                        # Fix: use base_suf when strategy timeframe is None
                        entry_raw = getattr(strat, "timeframe_entry", None)
                        exit_raw = getattr(strat, "timeframe_exit", None)
                        entry_suf = normalize_timeframe_to_suffix(entry_raw) if entry_raw else base_suf
                        exit_suf = normalize_timeframe_to_suffix(exit_raw) if exit_raw else base_suf
                        # Simplificar display: solo mostrar timeframe base
                        tf_display = base_suf.upper()

                        e_type = str(CONFIG["EXIT_TYPE"]).lower()
                        types_run = ["pnl_fixed", "pnl_trailing"] if e_type == "all" else [CONFIG["EXIT_TYPE"]]

                        for et in types_run:
                            run_single_exit_type(
                                exit_type=et,
                                strategy=strat,
                                strategy_name=s_name,
                                strategy_safe=s_safe,
                                activo=activo,
                                df_filtrado=df_filtrado,
                                tf_cache=tf_cache,
                                timeframe_base=tf_base,
                                cfg=cfg,
                                tf_display=tf_display,
                                archivo_data=archivo,
                                periodo_datos=periodo_str,
                                resolve_archivo_data_tf_func=resolve_archivo_data_tf,
                                fecha_inicio=FECHA_INICIO,
                                fecha_fin=FECHA_FIN
                            )

                # LIMPIEZA NUCLEAR AL CAMBIAR DE ACTIVO/TF PARA LIBERAR DATOS VIEJOS
                del df, df_filtrado
                nuclear_cleanup()

    except KeyboardInterrupt:
        pass
    finally:
        HealthGuard.final_cleanup()
        # LIMPIEZA TOTAL AL FINALIZAR
        full_system_cleanup()
        print("\n✅ [CLEANUP] Memoria y recursos liberados correctamente.")

if __name__ == "__main__":
    main()
