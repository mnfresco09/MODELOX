from __future__ import annotations

# ============================================================================
# [MAC-OPTIMIZED] CAPA 1: LIMITACIÓN DE HILOS (CRÍTICO: Poner ANTES de imports)
# ============================================================================
import os
# Esto obliga a Numpy/Pandas a no secuestrar todos los núcleos
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import warnings
import logging
import gc
import psutil  # Necesario: pip install psutil

"""
MODELOX - Production Entry Point (Smart Mac Optimization).
"""

# ============================================================================
# SILENCE ENGINE LOGS
# ============================================================================
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("optuna").setLevel(logging.WARNING)

# Silenciar Reporters
logging.getLogger("modelox.reporting.excel_reporter").setLevel(logging.WARNING)
logging.getLogger("modelox.reporting.plot_reporter").setLevel(logging.WARNING)
logging.getLogger("modelox.reporting.csv_reporter").setLevel(logging.WARNING)

logging.basicConfig(level=logging.WARNING)

# ============================================================================
# IMPORTS
# ============================================================================
from general.configuracion import (
    ACTIVOS, ACTIVO_PRIMARIO, resolve_archivo_data, resolve_qty_max_activo,
    COMBINACION_A_EJECUTAR, CONFIG,
    FECHA_FIN, FECHA_INICIO, N_TRIALS, FECHA_FIN_PLOT, FECHA_INICIO_PLOT,
    GENERAR_PLOTS, MAX_ARCHIVOS_GUARDAR, USAR_EXCEL,
)
from modelox.core.data import load_data
from modelox.core.optuna_config import OptunaConfig
from modelox.core.runner import OptimizationRunner
from modelox.core.types import BacktestConfig, filter_by_date
from modelox.reporting.csv_reporter import CSVReporter
from modelox.reporting.excel_reporter import ExcelReporter
from modelox.reporting.plot_reporter import PlotReporter
from modelox.reporting.rich_reporter import ElegantRichReporter
from modelox.strategies.registry import instantiate_strategies
from visual.rich import mostrar_cabecera_inicio, mostrar_fin_optimizacion

# ============================================================================
# [MAC-OPTIMIZED] CAPA 2: MONITOR DE SALUD DEL SISTEMA
# ============================================================================
class HealthGuard:
    @staticmethod
    def check_system_health(ram_threshold=85.0, cpu_cooldown_threshold=90.0):
        """
        Si la RAM está llena o la CPU ardiendo, pausa el script para dejar respirar al Mac.
        """
        # 1. Chequeo de RAM
        mem = psutil.virtual_memory()
        if mem.percent > ram_threshold:
            print(f"\n[ALERTA SISTEMA] RAM crítica ({mem.percent}%). Pausando 10s para drenaje...")
            gc.collect()
            time.sleep(10)
        
        # 2. Chequeo de Carga CPU (Load Average)
        # load_avg 1 min / nº cpus. Si es > 1.5, hay cola de procesos.
        load1, _, _ = psutil.getloadavg()
        cpu_count = psutil.cpu_count() or 1
        
        if (load1 / cpu_count) > 1.5:
             print(f"\n[ALERTA SISTEMA] CPU saturada. Enfriando 5s...")
             time.sleep(5)

    @staticmethod
    def force_cleanup():
        """Limpieza profunda de memoria"""
        gc.collect()
        # En Mac, a veces malloc no libera memoria al SO inmediatamente, pero gc ayuda.

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main() -> None:
    # 0. LIMPIEZA INICIAL
    HealthGuard.force_cleanup()

    # 2. PARSE STRATEGY IDS
    if COMBINACION_A_EJECUTAR == "all":
        strategy_ids = None 
    elif isinstance(COMBINACION_A_EJECUTAR, (list, tuple)):
        strategy_ids = list(COMBINACION_A_EJECUTAR)
    else:
        strategy_ids = [int(COMBINACION_A_EJECUTAR)]
    
    # 3. Resolve assets to run sequentially
    activos = list(ACTIVOS) if ACTIVOS else [ACTIVO_PRIMARIO]
    if not activos:
        activos = [ACTIVO_PRIMARIO]
    
    # 4. ITERATE OVER STRATEGIES
    # Convertimos a lista segura para evitar errores de iterador
    lista_ids = strategy_ids if strategy_ids is not None else [None]

    for activo in activos:
        # --- HEALTH CHECK ANTES DE CADA ACTIVO ---
        HealthGuard.check_system_health()
        HealthGuard.force_cleanup()

        archivo_data = resolve_archivo_data(activo)
        qty_max_activo = resolve_qty_max_activo(activo)

        # 1. DATA PIPELINE (por activo)
        df = load_data(archivo_data)
        df_filtrado = filter_by_date(df, FECHA_INICIO, FECHA_FIN)
        if df is not df_filtrado:
            del df
            gc.collect()

        # 2. BACKTEST CONFIGURATION (por activo)
        cfg = BacktestConfig(
            saldo_inicial=float(CONFIG.get("SALDO_INICIAL", 300)),
            saldo_operativo_max=float(CONFIG.get("SALDO_OPERATIVO_MAX", 100000)),
            apalancamiento=float(CONFIG.get("APALANCAMIENTO", 1)),
            comision_pct=float(CONFIG.get("COMISION_PCT", 0.0001)),
            comision_sides=int(CONFIG.get("COMISION_SIDES", 2)),
            saldo_minimo_operativo=float(CONFIG.get("SALDO_MINIMO_OPERATIVO", 5.0)),
            qty_max_activo=float(qty_max_activo),
        )
        print(f"[CONFIG] ACTIVO={activo} QTY_MAX_ACTIVO={cfg.qty_max_activo} DATA={archivo_data}")

        for strat_id in lista_ids:
            # --- HEALTH CHECK ANTES DE CADA ESTRATEGIA ---
            HealthGuard.check_system_health()
            HealthGuard.force_cleanup()

            if strat_id is None:
                strategies = instantiate_strategies(only_id=None)
            else:
                strategies = instantiate_strategies(only_id=int(strat_id))

            if not strategies:
                continue

            for strategy in strategies:
                strategy_name = strategy.name if hasattr(strategy, 'name') else "MODELOX_STRAT"

                mostrar_cabecera_inicio(
                    activo=activo,
                    combo_nombre=strategy_name,
                    indicadores=list(strategy.parametros_optuna.keys()),
                    n_trials=int(N_TRIALS),
                    archivo_data=archivo_data,
                )

                # REPORTERS SETUP
                activo_safe = str(activo).upper()
                csv_dir = f"resultados/csv/{activo_safe}"
                csv_resumen_dir = f"{csv_dir}/resumen"
                excel_dir = f"resultados/excel/{activo_safe}"
                plots_dir = f"resultados/plots/{activo_safe}"
                reporters = [
                    ElegantRichReporter(saldo_inicial=cfg.saldo_inicial, activo=activo),
                    CSVReporter(
                        trades_base_dir=csv_dir,
                        resumen_base_dir=csv_resumen_dir,
                        resumen_excel_dir=excel_dir,
                        max_archivos=int(MAX_ARCHIVOS_GUARDAR),
                    )
                ]

                if USAR_EXCEL:
                    reporters.append(
                        ExcelReporter(
                            resumen_path=f"{excel_dir}/resumen_trials.xlsx",
                            trades_base_dir=excel_dir,
                            max_archivos=int(MAX_ARCHIVOS_GUARDAR),
                        )
                    )

                if GENERAR_PLOTS:
                    reporters.append(PlotReporter(plot_base=plots_dir, fecha_inicio_plot=FECHA_INICIO_PLOT,
                                                 fecha_fin_plot=FECHA_FIN_PLOT, max_archivos=int(MAX_ARCHIVOS_GUARDAR),
                                                 saldo_inicial=cfg.saldo_inicial, activo=activo))

                # RUNNER EXECUTION
                runner = OptimizationRunner(config=cfg, n_trials=int(N_TRIALS), reporters=reporters)

                # [MAC-OPTIMIZED] Configuración vital para single-core performance estable
                runner.optuna = OptunaConfig(
                    seed=None,
                    n_jobs=1,  # ESTRICTAMENTE 1. Optuna con n_jobs > 1 en Mac + Python es inestable.
                    storage=None
                )
                runner.activo = activo

                try:
                    runner.optimize_strategies(df=df_filtrado, strategies=[strategy])

                    if hasattr(runner, '_last_study') and runner._last_study.best_trial:
                        study = runner._last_study
                        mostrar_fin_optimizacion(
                            total_trials=len(study.trials),
                            best_score=study.best_value,
                            best_trial=study.best_trial.number,
                            estrategia=strategy_name
                        )
                except Exception as e:
                    logging.error(f"Error optimizando estrategia {strategy_name}: {e}")
                finally:
                    del runner
                    del reporters
                    HealthGuard.force_cleanup()

        # Limpieza por activo
        del df_filtrado
        HealthGuard.force_cleanup()

    print("\n[FIN] Ejecución completada y memoria liberada.")

if __name__ == "__main__":
    main()