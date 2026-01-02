"""
Execution Helpers

Funciones auxiliares para ejecutar.py que simplifican el flujo principal.
"""

from __future__ import annotations

import logging
import os
from typing import List, Dict, Any

import polars as pl

from general.configuracion import (
    N_TRIALS,
    USAR_EXCEL,
    GENERAR_PLOTS,
    MAX_ARCHIVOS_GUARDAR,
    FECHA_INICIO_PLOT,
    FECHA_FIN_PLOT,
)
from modelox.core.runner import OptimizationRunner
from modelox.core.optuna_config import OptunaConfig
from modelox.core.types import BacktestConfig, Strategy, filter_by_date
from modelox.core.health_monitor import get_health_monitor
from modelox.core.timeframes import normalize_timeframe_to_suffix
from modelox.reporting.rich_reporter import ElegantRichReporter
from modelox.reporting.excel_reporter import ExcelReporter
from modelox.reporting.plot_reporter import PlotReporter
from visual.rich import mostrar_cabecera_inicio, mostrar_fin_optimizacion

logger = logging.getLogger(__name__)


def run_single_exit_type(
    exit_type: str,
    strategy: Strategy,
    strategy_name: str,
    strategy_safe: str,
    activo: str,
    df_filtrado: pl.DataFrame,
    tf_cache: Dict[str, pl.DataFrame],
    timeframe_base: str,
    cfg: BacktestConfig,
    tf_display: str,
    archivo_data: str,
    periodo_datos: str,
    is_all_mode: bool,
    resolve_archivo_data_tf_func: Any = None,
    fecha_inicio: str = None,
    fecha_fin: str = None,
) -> None:
    """
    Ejecuta optimización para un tipo de salida específico.
    
    Args:
        exit_type: Tipo de salida ("pnl_fixed" o "pnl_trailing")
        strategy: Instancia de la estrategia
        strategy_name: Nombre de la estrategia
        strategy_safe: Nombre seguro para carpetas
        activo: Activo a testear
        df_filtrado: DataFrame filtrado por fechas
        tf_cache: Cache de timeframes
        timeframe_base: Timeframe base
        cfg: Configuración de backtest
        tf_display: String de display de timeframes
        archivo_data: Path al archivo de datos
        periodo_datos: String de display del período
        is_all_mode: Si True, añade sufijo al nombre de carpeta
    """
    
    # 1. Actualizar config con exit_type actual
    cfg_dict = cfg.__dict__.copy()
    cfg_dict["exit_type"] = exit_type
    cfg_updated = BacktestConfig(**cfg_dict)
    
    # 2. Mostrar header
    mostrar_cabecera_inicio(
        activo=activo,
        combo_nombre=strategy_name,
        indicadores=list(strategy.parametros_optuna.keys()),
        n_trials=int(N_TRIALS),
        archivo_data=archivo_data,
        timeframe=tf_display,
        periodo=periodo_datos,
        exit_type=exit_type,
    )
    
    # 3. Setup de carpetas y reporters
    activo_safe = str(activo).upper()
    
    # Añadir sufijo de exit_type si estamos en modo "all"
    if is_all_mode:
        strategy_root_dir = os.path.join("resultados", f"{strategy_safe}_{exit_type.upper()}")
    else:
        strategy_root_dir = os.path.join("resultados", strategy_safe)
    
    excel_dir = os.path.join(strategy_root_dir, "excel")
    graficos_dir = os.path.join(strategy_root_dir, "graficos", activo_safe)
    
    # Crear carpetas
    os.makedirs(excel_dir, exist_ok=True)
    os.makedirs(os.path.dirname(graficos_dir), exist_ok=True)
    
    # 4. Crear reporters
    reporters: List[Any] = [
        ElegantRichReporter(saldo_inicial=cfg_updated.saldo_inicial, activo=activo),
    ]
    
    if USAR_EXCEL:
        reporters.append(
            ExcelReporter(
                resumen_path=f"{excel_dir}/resumen.xlsx",
                trades_base_dir=excel_dir,
                max_archivos=int(MAX_ARCHIVOS_GUARDAR),
            )
        )
    
    if GENERAR_PLOTS:
        reporters.append(
            PlotReporter(
                plot_base=graficos_dir,
                fecha_inicio_plot=FECHA_INICIO_PLOT,
                fecha_fin_plot=FECHA_FIN_PLOT,
                max_archivos=int(MAX_ARCHIVOS_GUARDAR),
                saldo_inicial=cfg_updated.saldo_inicial,
                activo=activo,
            )
        )
    
    # 5. Crear runner
    runner = OptimizationRunner(
        config=cfg_updated,
        n_trials=int(N_TRIALS),
        reporters=reporters,
    )
    
    # Mac-optimized: single-core
    runner.optuna = OptunaConfig(
        seed=None,
        n_jobs=1,
        storage=None,
    )
    runner.activo = activo
    
    # 6. Ejecutar optimización
    try:
        # Multi-timeframe: cargar timeframes adicionales si la estrategia los necesita
        entry_tf = getattr(strategy, "timeframe_entry", None) or timeframe_base
        exit_tf = getattr(strategy, "timeframe_exit", None) or timeframe_base
        
        needed_tfs = [timeframe_base, entry_tf, exit_tf]
        
        for tf in needed_tfs:
            tf_suffix = normalize_timeframe_to_suffix(tf)
            if tf_suffix in tf_cache:
                continue
            try:
                if resolve_archivo_data_tf_func is not None:
                    from modelox.core.data import load_data
                    path_tf = resolve_archivo_data_tf_func(activo, tf, formato="parquet")
                    df_tf = load_data(path_tf)
                    if fecha_inicio and fecha_fin:
                        df_tf = filter_by_date(df_tf, fecha_inicio, fecha_fin)
                    tf_cache[tf_suffix] = df_tf
            except Exception as e:
                logger.warning(f"No se pudo cargar timeframe {tf} para {activo}: {e}")
                continue
        
        runner.optimize_strategies(
            df=df_filtrado,
            strategies=[strategy],
            df_by_timeframe=tf_cache,
            base_timeframe=timeframe_base,
        )
        
        # Mostrar resultado
        if hasattr(runner, '_last_study') and runner._last_study.best_trial:
            study = runner._last_study
            mostrar_fin_optimizacion(
                total_trials=len(study.trials),
                best_score=study.best_value,
                best_trial=study.best_trial.number,
                estrategia=strategy_name,
            )
    
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logger.error(f"Error optimizando estrategia {strategy_name} con exit_type={exit_type}: {e}")
    finally:
        # Cleanup
        del runner
        del reporters
        monitor = get_health_monitor()
        monitor.force_cleanup(deep=True)
