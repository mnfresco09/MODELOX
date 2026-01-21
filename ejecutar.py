from __future__ import annotations

# ============================================================================
# [CRÍTICO] CONFIGURACIÓN DE ALTO RENDIMIENTO PARA VM (CLOUD)
# ============================================================================
import os
import sys

# CONFIGURACIÓN DE HILOS: FORZAMOS A 1 HILO POR PROCESO.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"

# VARIABLES DE DEPURACIÓN
os.environ.setdefault("MODELOX_TIMINGS", "1")
os.environ.setdefault("MODELOX_TIMINGS_PRINT_EVERY", "10")

# ============================================================================
# IMPORTS DEL SISTEMA
# ============================================================================
import warnings
import logging
import gc
import atexit
import shutil
from pathlib import Path
import re

# CAPTURA SEGURA DE LA RUTA RAÍZ AL INICIO (EVITA ERROR EN ATEXIT)
try:
    _PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    # Fallback si __file__ no está definido (ej: shell interactiva)
    _PROJECT_ROOT = Path.cwd()

# SILENCIAR ADVERTENCIAS
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# LOGGING SETUP
# ============================================================================
def setup_logging(level: int = logging.INFO) -> None:
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
    logging.getLogger("modelox.reporting").setLevel(logging.WARNING)

setup_logging(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ============================================================================
# IMPORTS DEL PROYECTO
# ============================================================================
from general.configuracion import (
    ACTIVOS, ACTIVO_PRIMARIO, resolve_archivo_data, resolve_archivo_data_tf, 
    resolve_qty_max_activo, resolve_qty_max_activo_range,
    COMBINACION_A_EJECUTAR, CONFIG,
    FECHA_FIN, FECHA_INICIO, N_TRIALS, FECHA_FIN_PLOT, FECHA_INICIO_PLOT,
    GENERAR_PLOTS, MAX_ARCHIVOS_GUARDAR, USAR_EXCEL,
)
from modelox.core.data import load_data
from modelox.core.runner import OptimizationRunner, OptunaConfig
from modelox.core.types import normalize_timeframe_to_suffix, BacktestConfig, filter_by_date
from modelox.reporting.excel_reporter import ExcelReporter
from modelox.reporting.plot_reporter import PlotReporter
from modelox.reporting.rich_reporter import ElegantRichReporter
from modelox.strategies.registry import instantiate_strategies
from visual.rich import mostrar_cabecera_inicio, mostrar_fin_optimizacion

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
    )

    # 4. RUTAS DE SALIDA
    activo_safe = str(activo).upper()
    tf_suffix = normalize_timeframe_to_suffix(timeframe_base)
    strategy_root_dir = os.path.join(
        "resultados",
        f"{strategy_safe}_{str(exit_type).upper()}",
        str(tf_suffix),
    )
    excel_dir = os.path.join(strategy_root_dir, "excel")
    graficos_dir = os.path.join(strategy_root_dir, "graficos", activo_safe)
    os.makedirs(excel_dir, exist_ok=True)
    os.makedirs(os.path.dirname(graficos_dir), exist_ok=True)

    # 5. REPORTEROS
    reporters = [ElegantRichReporter(saldo_inicial=cfg_updated.saldo_inicial, activo=activo)]
    
    if USAR_EXCEL:
        reporters.append(ExcelReporter(
            resumen_path=f"{excel_dir}/resumen.xlsx",
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

    # 6. RUNNER
    runner = OptimizationRunner(config=cfg_updated, n_trials=int(N_TRIALS), reporters=reporters)
    runner.optuna = OptunaConfig(seed=None, n_jobs=-1, storage=None)
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

        if hasattr(runner, "_last_study") and runner._last_study and runner._last_study.best_trial:
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
        logger.error(f"Error en {strategy_name}: {e}")
    finally:
        del runner
        del reporters
        gc.collect()

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
        gc.collect()
        # Usamos la variable global _PROJECT_ROOT capturada al inicio
        if CONFIG.get("PURGE_PYCACHE_ON_EXIT"):
            _purge_pycache(root=_PROJECT_ROOT, exclude={".git", ".venv", "data"})


def run_montecarlo_mode_single_exit(
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
    """
    Ejecuta Monte Carlo Optimization con NSGA-II (Multi-Objetivo).
    
    CONCEPTO: Cada trial usa un mercado sintético DIFERENTE.
    NSGA-II optimiza DOS objetivos:
      1. MAXIMIZAR: Calidad/Rentabilidad
      2. MINIMIZAR: Drawdown
    """
    from modelox.core.runner_montecarlo import MonteCarloRunner, MonteCarloConfig
    
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
    
    # 3. PARÁMETROS MC
    mc_n_trials = int(N_TRIALS)
    mc_noise = float(CONFIG.get("MC_NOISE_PCT", 0.5))
    mc_noise_range = float(CONFIG.get("MC_NOISE_RANGE", 100.0))
    mc_block = int(CONFIG.get("MC_BLOCK_SIZE", 1440))
    mc_method = str(CONFIG.get("MC_METHOD", "monetary"))
    mc_seed = int(CONFIG.get("MC_SEED", 42))
    use_nsga2 = bool(CONFIG.get("MC_USE_NSGA2", True))  # Por defecto usa NSGA-II
    
    # 4. MOSTRAR HEADER
    sampler_name = "NSGA-II" if use_nsga2 else "TPE"
    mostrar_cabecera_inicio(
        activo=activo,
        combo_nombre=f"{strategy_name} [MONTE CARLO {sampler_name}]",
        indicadores=indicadores,
        n_trials=mc_n_trials,
        archivo_data=archivo_data,
        timeframe=tf_display,
        periodo=periodo_datos,
        exit_type=exit_type,
        strategy_exit_enabled=strategy_exit_enabled,
    )
    
    # Info MC
    print(f"  🎲 Monte Carlo: {mc_n_trials} mercados sintéticos únicos")
    print(f"  🧬 Sampler: {sampler_name} {'(Multi-Objetivo: Quality↑ + DD↓)' if use_nsga2 else '(Single-Objetivo)'}")
    print(f"  📊 Método: {mc_method} | Ruido: {mc_noise}% | Block: {mc_block}")
    print(f"  💡 Cada trial evalúa parámetros en un mercado diferente")
    print()
    
    # 5. RUTAS DE SALIDA
    activo_safe = str(activo).upper()
    tf_suffix = normalize_timeframe_to_suffix(timeframe_base)
    strategy_root_dir = os.path.join(
        "resultados",
        f"{strategy_safe}_{str(exit_type).upper()}_MC",
        str(tf_suffix),
    )
    excel_dir = os.path.join(strategy_root_dir, "excel")
    graficos_dir = os.path.join(strategy_root_dir, "graficos", activo_safe)
    os.makedirs(excel_dir, exist_ok=True)
    os.makedirs(graficos_dir, exist_ok=True)
    
    # 6. REPORTEROS
    reporters = [ElegantRichReporter(saldo_inicial=cfg_updated.saldo_inicial, activo=activo)]
    
    if USAR_EXCEL:
        reporters.append(ExcelReporter(
            resumen_path=f"{excel_dir}/resumen_mc.xlsx",
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
    
    # 7. CONFIGURACIÓN MC con NSGA-II
    mc_config = MonteCarloConfig(
        n_trials=mc_n_trials,
        noise_pct=mc_noise,
        noise_range=mc_noise_range,
        block_size=mc_block,
        method=mc_method,
        seed=mc_seed,
        use_nsga2=use_nsga2,
    )
    
    # 8. RUNNER MC
    runner = MonteCarloRunner(
        strategy=strategy,
        config=cfg_updated,
        mc_config=mc_config,
        reporters=reporters,
        df=df_filtrado,
    )
    runner.activo = activo
    
    try:
        results = runner.run()
        
        if results:
            robustness = results.get("robustness_pct", 0)
            best_trial = results.get("best_trial")
            best_score = best_trial.score if best_trial else 0
            pareto_count = len(results.get("pareto_front", []))
            
            suffix = f"[MC {sampler_name}: {robustness:.1f}% robustez"
            if use_nsga2 and pareto_count > 0:
                suffix += f", {pareto_count} Pareto"
            suffix += "]"
            
            mostrar_fin_optimizacion(
                total_trials=mc_n_trials,
                best_score=best_score,
                best_trial=best_trial.trial_number if best_trial else 0,
                estrategia=f"{strategy_name} {suffix}",
            )
    
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logger.error(f"Error en MC {strategy_name}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        del runner
        del reporters
        gc.collect()


def main() -> None:
    atexit.register(HealthGuard.final_cleanup)
    
    # DETECTAR MODO DE OPTIMIZACIÓN
    modo = CONFIG.get("MODO_OPTIMIZACION", "NORMAL").upper()
    
    # HEADER según modo
    if modo == "MONTECARLO":
        print("\n" + "="*70)
        print("  🎲 MODO: MONTE CARLO + OPTIMIZATION")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("  ⚡ MODO: OPTIMIZACIÓN NORMAL")
        print("="*70 + "\n")
    
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
                        entry_suf = normalize_timeframe_to_suffix(getattr(strat, "timeframe_entry", base_suf))
                        exit_suf = normalize_timeframe_to_suffix(getattr(strat, "timeframe_exit", base_suf))
                        tf_display = f"BASE:{base_suf} IN:{entry_suf} OUT:{exit_suf}"

                        e_type = str(CONFIG["EXIT_TYPE"]).lower()
                        types_run = ["pnl_fixed", "pnl_trailing"] if e_type == "all" else [CONFIG["EXIT_TYPE"]]

                        for et in types_run:
                            if modo == "MONTECARLO":
                                run_montecarlo_mode_single_exit(
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
                            else:
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
                del df, df_filtrado
                gc.collect()

    except KeyboardInterrupt:
        pass
    finally:
        HealthGuard.final_cleanup()

if __name__ == "__main__":
    main()