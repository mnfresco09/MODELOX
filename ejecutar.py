from __future__ import annotations


# =============================================================================
# [SECCIÓN 1] CONFIGURACIÓN CRÍTICA DE RENDIMIENTO
# =============================================================================

import os
import sys
import multiprocessing as mp

# ── LÍMITE GLOBAL: máximo N_JOBS, dejando 2 cores libres ──
N_JOBS = max(1, mp.cpu_count() - 2)

# Forzar 1 hilo por proceso para evitar contención en optimización
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"

# Propagar límite global para subprocesos que lean esta variable
os.environ["MODELOX_MAX_NJOBS"] = str(N_JOBS)

# VARIABLES DE DEPURACIÓN (opcionales)
os.environ.setdefault("MODELOX_TIMINGS", "1")
os.environ.setdefault("MODELOX_TIMINGS_PRINT_EVERY", "10")

# ── PRIORIDAD DEL PROCESO: ejecutar.py tiene máxima prioridad ──
try:
    os.nice(-5)  # Mayor prioridad (requiere permisos en algunos OS)
except (OSError, PermissionError):
    try:
        os.nice(0)  # Prioridad normal si no hay permisos
    except (OSError, PermissionError):
        pass


# =============================================================================
# [SECCIÓN 2] IMPORTS
# =============================================================================

import atexit
import logging
import re
import warnings
from pathlib import Path

# Captura segura de la ruta raíz (antes de cualquier cambio de directorio)
try:
    _PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    _PROJECT_ROOT = Path.cwd()

# Silenciar advertencias
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# [SECCIÓN 3] LOGGING
# =============================================================================

def _setup_logging(level: int = logging.WARNING) -> None:
    """Configura el sistema de logging."""
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    # Silenciar loggers ruidosos
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        logging.getLogger("optuna").setLevel(logging.WARNING)
    except ImportError:
        pass

_setup_logging(level=logging.WARNING)
logger = logging.getLogger(__name__)


# =============================================================================
# [SECCIÓN 4] IMPORTS DEL PROYECTO
# =============================================================================

from general.configuracion import (
    # Activos y datos
    ACTIVOS, ACTIVO_PRIMARIO, 
    resolve_archivo_data, resolve_archivo_data_tf,
    # Configuración general
    CONFIG, COMBINACION_A_EJECUTAR,
    # Timeframe
    TIMEFRAME_BASE,
    # Fechas
    FECHA_FIN, FECHA_INICIO,
    # Gráficos (rango)
    GRAFICA_RANGO_PERSONALIZADO, GRAFICA_FECHA_INICIO, GRAFICA_FECHA_FIN,
    # Optimización
    N_TRIALS, OPTUNA_SAMPLER,
    # Rangos por trial
    USAR_RANGOS_POR_TRIAL, MESES_POR_TRIAL,
    # Salidas
    GENERAR_PLOTS, MAX_ARCHIVOS_GUARDAR, USAR_EXCEL,
    # Limpieza
    CLEANUP_INTERVAL,
)

from modelox.core.data import load_data, resample_to_base_timeframe, candles_per_month_for_tf
from modelox.core.runner import (
    OptimizationRunner, OptunaConfig,
    run_single_exit_type,
)
from modelox.core.types import (
    BacktestConfig,
    filter_by_date,
    normalize_timeframe_to_suffix,
    nuclear_cleanup, full_system_cleanup,
    HealthGuard,
)
from modelox.strategies.registry import instantiate_strategies
from visual.rich import mostrar_cabecera_inicio, mostrar_fin_optimizacion

# Configurar intervalo de limpieza
os.environ["MODELOX_CLEANUP_INTERVAL"] = str(CLEANUP_INTERVAL)


# =============================================================================
# [SECCIÓN 5] FUNCIÓN PRINCIPAL
# =============================================================================

def main() -> None:
    """
    Función principal de ejecución.
    
    Flujo:
    1. Registrar limpieza al cierre
    2. Parsear estrategias a ejecutar
    3. Iterar por activos y timeframes
    4. Ejecutar optimización por tipo de salida
    5. Limpiar recursos
    """
    # -------------------------------------------------------------------------
    # 5.1 CONFIGURAR LIMPIEZA AL CIERRE
    # -------------------------------------------------------------------------
    HealthGuard.configure(
        project_root=_PROJECT_ROOT,
        purge_pycache=True  # Siempre limpiar __pycache__ al terminar
    )
    atexit.register(HealthGuard.final_cleanup)

    # -------------------------------------------------------------------------
    # 5.2 PARSEAR ESTRATEGIAS
    # -------------------------------------------------------------------------
    if COMBINACION_A_EJECUTAR == "all":
        strategy_ids = None
    elif isinstance(COMBINACION_A_EJECUTAR, (list, tuple)):
        strategy_ids = list(COMBINACION_A_EJECUTAR)
    else:
        strategy_ids = [int(COMBINACION_A_EJECUTAR)]

    # -------------------------------------------------------------------------
    # 5.3 ACTIVOS Y TIMEFRAMES
    # -------------------------------------------------------------------------
    activos = list(ACTIVOS) if ACTIVOS else [ACTIVO_PRIMARIO]

    raw_tfs = CONFIG.get("TIMEFRAMES", None)
    fallback_tf = CONFIG.get("TIMEFRAME", 1)  # Default: 1m

    def _ensure_int_list(raw, fallback):
        """Convierte timeframes a lista de enteros (minutos)."""
        res = []
        candidatos = raw if isinstance(raw, (list, tuple)) else str(raw).split(",")
        for c in candidatos:
            try:
                s = str(c).strip().lower()
                if s.endswith("d"):
                    val = int(float(s[:-1]) * 1440)
                elif s.endswith("h"):
                    val = int(float(s[:-1]) * 60)
                else:
                    val = int(float(s.replace("m", "")))
                if val > 0:
                    res.append(val)
            except (ValueError, TypeError):
                pass
        return res if res else [int(fallback)]

    timeframes = _ensure_int_list(raw_tfs, fallback_tf)

    # -------------------------------------------------------------------------
    # 5.4 BUCLE PRINCIPAL DE EJECUCIÓN
    # -------------------------------------------------------------------------
    try:
        for activo in activos:
            for tf_base in timeframes:
                # ---------------------------------------------------------
                # CARGAR DATOS
                # ---------------------------------------------------------
                data_provider = None
                using_trial_ranges = False

                try:
                    archivo = resolve_archivo_data_tf(activo, tf_base)
                    if not os.path.exists(archivo):
                        archivo = resolve_archivo_data(activo)
                except Exception:
                    archivo = resolve_archivo_data(activo)

                if not os.path.exists(archivo):
                    print(f"❌ DATA NOT FOUND: {archivo}")
                    continue

                df = load_data(archivo)

                # Resamplear de 1m al timeframe base elegido en config
                tf_base_suffix = normalize_timeframe_to_suffix(tf_base)
                
                if tf_base_suffix == "1m":
                    print(f"   ⏱️ Timeframe base: {tf_base_suffix.upper()} (data original 1m)")
                else:
                    df = resample_to_base_timeframe(df, tf_base_suffix)
                    print(f"   ⏱️ Timeframe base: {tf_base_suffix.upper()} (resampleado desde 1m)")

                # Filtrar por FECHA_INICIO → FECHA_FIN
                df_filtrado = filter_by_date(df, FECHA_INICIO, FECHA_FIN)

                # Velas por mes según timeframe base
                candles_per_month = candles_per_month_for_tf(tf_base_suffix)

                # Auto-desactivar rangos por trial en TF grandes (12h, 1d)
                # Con TF ≥ 12h hay muy pocas velas por ventana y las estrategias
                # necesitan warmup largo → se usa el rango completo
                tf_minutes = int(tf_base) if isinstance(tf_base, (int, float)) else 1
                usar_rangos_efectivo = USAR_RANGOS_POR_TRIAL
                if tf_minutes >= 720 and USAR_RANGOS_POR_TRIAL:
                    usar_rangos_efectivo = False
                    print(f"   ⚠️  TF ≥ 12h ({tf_base_suffix.upper()}) → Rangos por trial DESACTIVADOS automáticamente")
                    print(f"       (con {candles_per_month} velas/mes × {MESES_POR_TRIAL} meses = {MESES_POR_TRIAL * candles_per_month} velas por trial, insuficiente para warmup)")
                    print(f"       → Se usa rango completo: {len(df_filtrado):,} velas")

                # ---------------------------------------------------------
                # MODO RANGOS ALEATORIOS POR TRIAL (anti-overfitting)
                # ---------------------------------------------------------
                if usar_rangos_efectivo:
                    print("\n" + "=" * 60)
                    print("🎲 MODO RANGOS ALEATORIOS POR TRIAL")
                    print("=" * 60)

                    # Guardar periodo COMPLETO para display
                    try:
                        ts_col = "timestamp" if "timestamp" in df_filtrado.columns else "datetime"
                        periodo_str_completo = f"{df_filtrado[ts_col].min():%Y-%m-%d} -> {df_filtrado[ts_col].max():%Y-%m-%d}"
                    except Exception:
                        periodo_str_completo = ""

                    total_candles = len(df_filtrado)
                    candles_per_trial = MESES_POR_TRIAL * candles_per_month
                    # Puntos de inicio posibles = cualquier índice donde quepa una ventana completa
                    max_start_positions = max(1, total_candles - candles_per_trial + 1)

                    print(f"   📊 Total velas: {total_candles:,}")
                    print(f"   🎯 Meses por trial: {MESES_POR_TRIAL}")
                    print(f"   🔄 Ventanas posibles: {max_start_positions:,} (inicio aleatorio por trial)")

                    class TrialDataProvider:
                        """Provider con selección ALEATORIA de rangos por trial."""
                        def __init__(self, df, candles_per_trial, candles_per_month, seed=None):
                            self.df = df
                            self.candles_per_trial = candles_per_trial
                            self.candles_per_month = candles_per_month
                            self.total = len(df)
                            self.max_start = max(0, self.total - candles_per_trial)
                            self.rng = __import__('random').Random(seed)
                            self.last_range_months = None

                        def get_trial_data(self, trial_number, verbose=False):
                            trial_rng = __import__('random').Random(self.rng.randint(0, 2**31) + trial_number)
                            start_idx = trial_rng.randint(0, self.max_start)
                            mes_inicio = start_idx // self.candles_per_month + 1
                            mes_fin = (start_idx + self.candles_per_trial - 1) // self.candles_per_month + 1
                            self.last_range_months = (mes_inicio, mes_fin)
                            return self.df.slice(start_idx, self.candles_per_trial)

                        def get_trial_data_multiframe(self, trial_number, timeframes, verbose=False):
                            from modelox.core.types import normalize_timeframe_to_suffix
                            df_trial = self.get_trial_data(trial_number, verbose)
                            base_tf = normalize_timeframe_to_suffix(tf_base)
                            return {base_tf: df_trial}

                    data_provider = TrialDataProvider(df_filtrado, candles_per_trial, candles_per_month)
                    using_trial_ranges = True

                    print("=" * 60 + "\n")

                # Determinar periodo para display
                periodo_str = ""
                if using_trial_ranges:
                    periodo_str = periodo_str_completo
                else:
                    try:
                        ts = df_filtrado["timestamp"] if "timestamp" in df_filtrado.columns else df_filtrado["datetime"]
                        periodo_str = f"{ts.min():%Y-%m-%d} -> {ts.max():%Y-%m-%d}"
                    except (KeyError, AttributeError):
                        pass

                # Cache de timeframes
                tf_cache = {normalize_timeframe_to_suffix(tf_base): df_filtrado}

                # -------------------------------------------------------------
                # CONFIGURACIÓN DE BACKTEST
                # -------------------------------------------------------------
                cfg = BacktestConfig(
                    saldo_inicial=float(CONFIG["SALDO_INICIAL"]),
                    saldo_operativo_max=float(CONFIG["SALDO_OPERATIVO_MAX"]),
                    comision_pct=float(CONFIG["COMISION_PCT"]),
                    comision_sides=int(CONFIG["COMISION_SIDES"]),
                    saldo_minimo_operativo=float(CONFIG["SALDO_MINIMO_OPERATIVO"]),
                    saldo_usado=float(CONFIG["SALDO_USADO"]),
                    apalancamiento_max=float(CONFIG["APALANCAMIENTO_MAX"]),
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

                # -------------------------------------------------------------
                # ITERAR POR ESTRATEGIAS
                # -------------------------------------------------------------
                loop_ids = strategy_ids if strategy_ids else [None]
                
                for sid in loop_ids:
                    strategies = instantiate_strategies(only_id=sid if sid else None)
                    if not strategies:
                        continue

                    for strat in strategies:
                        s_name = getattr(strat, 'name', "STRAT")
                        s_safe = re.sub(r"[^A-Z0-9_]+", "_", s_name.upper())

                        # Timeframe display
                        base_suf = normalize_timeframe_to_suffix(tf_base)
                        tf_display = base_suf.upper()

                        # Tipos de salida a ejecutar
                        e_type = str(CONFIG["EXIT_TYPE"]).lower()
                        exit_types = ["pnl_fixed", "pnl_trailing"] if e_type == "all" else [CONFIG["EXIT_TYPE"]]

                        # ---------------------------------------------------------
                        # EJECUTAR POR TIPO DE SALIDA
                        # ---------------------------------------------------------
                        for et in exit_types:
                            # Rutas de salida
                            activo_safe = str(activo).upper()
                            exit_folder = "TRAILING" if "trail" in str(et).lower() else "FIXED"
                            strategy_root = os.path.join("resultados", s_safe, exit_folder, activo_safe)
                            excel_dir = os.path.join(strategy_root, "EXCEL")
                            graficos_dir = os.path.join(strategy_root, "GRAFICA")
                            os.makedirs(excel_dir, exist_ok=True)
                            os.makedirs(graficos_dir, exist_ok=True)


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
                                # Configuración
                                n_trials=int(N_TRIALS),
                                optuna_sampler=OPTUNA_SAMPLER,
                                # Datos con rangos por trial
                                synthetic_mode=False,
                                synthetic_years=MESES_POR_TRIAL // 12 if USAR_RANGOS_POR_TRIAL else 0,
                                futuro_data_provider=data_provider if using_trial_ranges else None,
                                # Rutas
                                excel_dir=excel_dir,
                                graficos_dir=graficos_dir,
                                # Opciones
                                usar_excel=USAR_EXCEL,
                                generar_plots=GENERAR_PLOTS,
                                max_archivos=int(MAX_ARCHIVOS_GUARDAR),
                                # Gráficos (rango binario)
                                grafica_rango_personalizado=GRAFICA_RANGO_PERSONALIZADO,
                                grafica_fecha_inicio=GRAFICA_FECHA_INICIO,
                                grafica_fecha_fin=GRAFICA_FECHA_FIN,
                                # Funciones
                                resolve_archivo_data_tf_func=resolve_archivo_data_tf,
                                fecha_inicio=FECHA_INICIO,
                                fecha_fin=FECHA_FIN,
                                mostrar_cabecera_func=mostrar_cabecera_inicio,
                                mostrar_fin_func=mostrar_fin_optimizacion,
                                logger=logger,
                            )

                # Limpieza al cambiar de activo/timeframe
                del df, df_filtrado
                nuclear_cleanup()

    except KeyboardInterrupt:
        print("\n⚠️ Ejecución interrumpida por el usuario.")
    finally:
        # ---------------------------------------------------------------------
        # 5.5 LIMPIEZA FINAL
        # ---------------------------------------------------------------------
        HealthGuard.final_cleanup()
        full_system_cleanup()
        print("\n✅ [CLEANUP] Memoria y recursos liberados correctamente.")


# =============================================================================
# [SECCIÓN 6] PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    main()
