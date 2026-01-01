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
import atexit
import signal
import sys
import shutil
from pathlib import Path
import psutil  # Necesario: pip install psutil
import re

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

logging.basicConfig(level=logging.WARNING)

# ============================================================================
# IMPORTS
# ============================================================================
from general.configuracion import (
    ACTIVOS, ACTIVO_PRIMARIO, resolve_archivo_data, resolve_archivo_data_tf, resolve_qty_max_activo, resolve_qty_max_activo_range,
    COMBINACION_A_EJECUTAR, CONFIG,
    FECHA_FIN, FECHA_INICIO, N_TRIALS, FECHA_FIN_PLOT, FECHA_INICIO_PLOT,
    GENERAR_PLOTS, MAX_ARCHIVOS_GUARDAR, USAR_EXCEL,
)
from modelox.core.data import load_data
from modelox.core.optuna_config import OptunaConfig
from modelox.core.runner import OptimizationRunner
from modelox.core.timeframes import normalize_timeframe_to_suffix
from modelox.core.types import BacktestConfig, filter_by_date
from modelox.reporting.excel_reporter import ExcelReporter
from modelox.reporting.plot_reporter import PlotReporter
from modelox.reporting.rich_reporter import ElegantRichReporter
from modelox.strategies.registry import instantiate_strategies
from visual.rich import mostrar_cabecera_inicio, mostrar_fin_optimizacion


def _purge_pycache(*, root: Path, exclude_dirnames: set[str]) -> None:
    """Elimina __pycache__ y .pyc dentro del proyecto.

    - Evita tocar entornos virtuales o carpetas grandes (según exclude_dirnames).
    - Seguro: Python los regenera automáticamente cuando haga falta.
    """

    def _is_excluded(p: Path) -> bool:
        # Si cualquier parte del path coincide con un excluded, saltamos.
        return any(part in exclude_dirnames for part in p.parts)

    try:
        for dirpath, dirnames, filenames in os.walk(root):
            p = Path(dirpath)
            if _is_excluded(p):
                dirnames[:] = []
                continue

            # borrar __pycache__
            if p.name == "__pycache__":
                shutil.rmtree(p, ignore_errors=True)
                dirnames[:] = []
                continue

            # borrar .pyc sueltos
            for fname in filenames:
                if fname.endswith(".pyc"):
                    try:
                        (p / fname).unlink(missing_ok=True)
                    except Exception:
                        pass
    except Exception:
        # Nunca romper el shutdown por esto.
        return

# ============================================================================
# [MAC-OPTIMIZED] CAPA 2: MONITOR DE SALUD DEL SISTEMA
# ============================================================================
class HealthGuard:
    _last_check_ts: float = 0.0
    _last_cooldown_ts: float = 0.0
    _cleanup_ran: bool = False

    @staticmethod
    def check_system_health(
        *,
        ram_threshold: float = 80.0,
        load_per_core_threshold: float = 1.75,
        min_check_interval_s: float = 2.0,
        cpu_cooldown_s: float = 0.35,
        cooldown_min_interval_s: float = 8.0,
    ) -> None:
        """
        Mantenimiento ligero y rate-limited.

        - Evita sleeps largos (no frena el arranque).
        - Solo aplica cooldown si el sistema está realmente saturado.
        """
        now = time.monotonic()
        if (now - HealthGuard._last_check_ts) < min_check_interval_s:
            return
        HealthGuard._last_check_ts = now

        # 1. Chequeo de RAM
        mem = psutil.virtual_memory()
        if mem.percent > ram_threshold:
            # Limpieza más agresiva sin sleeps largos.
            HealthGuard.force_cleanup(deep=True)
            # Ceder un poco de CPU para que el SO recupere.
            time.sleep(0.15)
        
        # 2. Chequeo de carga CPU (Load Average / core)
        load1, _, _ = psutil.getloadavg()
        cpu_count = psutil.cpu_count() or 1

        load_per_core = load1 / cpu_count
        if load_per_core > load_per_core_threshold:
            # Cooldown pequeño y con rate-limit para no duplicarlo.
            if (now - HealthGuard._last_cooldown_ts) >= cooldown_min_interval_s:
                HealthGuard._last_cooldown_ts = now
                time.sleep(cpu_cooldown_s)

    @staticmethod
    def force_cleanup(*, deep: bool = False) -> None:
        """Limpieza de memoria.

        Nota: macOS no garantiza que el allocator devuelva RAM al SO inmediatamente,
        pero `gc.collect()` reduce presión y libera objetos Python.
        """
        if deep:
            # 3 pasadas suele ser suficiente para ciclos complejos.
            gc.collect(2)
            gc.collect(1)
            gc.collect(0)
        else:
            gc.collect()

    @staticmethod
    def final_cleanup(*, reason: str) -> None:
        if HealthGuard._cleanup_ran:
            return
        HealthGuard._cleanup_ran = True

        try:
            HealthGuard.force_cleanup(deep=True)
        except Exception:
            # Nunca fallar durante shutdown.
            pass

        # Limpieza de cachés Python (opcional)
        if _PURGE_PYCACHE_ON_EXIT:
            _purge_pycache(root=_PROJECT_ROOT, exclude_dirnames=_PYCACHE_EXCLUDES)


def _install_shutdown_handlers() -> None:
    def _on_exit() -> None:
        HealthGuard.final_cleanup(reason="atexit")

    atexit.register(_on_exit)

    def _signal_handler(signum: int, frame: object | None) -> None:
        HealthGuard.final_cleanup(reason=f"signal:{signum}")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


# ----------------------------------------------------------------------------
# Configuración de limpieza de caches (opcional)
# ----------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent

# Excluir venv y carpetas grandes/irrelevantes para bytecode.
_PYCACHE_EXCLUDES: set[str] = {
    ".git",
    ".venv",
    ".venv311",
    "node_modules",
    "data",
    "resultados",
}

# Opción: borra __pycache__ y *.pyc al finalizar (incluye Ctrl+C).
# Se puede activar:
#   - en configuracion: CONFIG["PURGE_PYCACHE_ON_EXIT"] = True
#   - o por env var: MODELOX_PURGE_PYCACHE=1
_PURGE_PYCACHE_ON_EXIT = bool(CONFIG.get("PURGE_PYCACHE_ON_EXIT", False)) or (
    os.environ.get("MODELOX_PURGE_PYCACHE", "0") in {"1", "true", "True", "YES", "yes"}
)

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main() -> None:
    _install_shutdown_handlers()

    # 0. LIMPIEZA INICIAL
    HealthGuard.force_cleanup(deep=True)

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

    try:
        for activo in activos:
            # --- MANTENIMIENTO ANTES DE CADA ACTIVO ---
            HealthGuard.check_system_health()

            # Timeframe base configurable (minutos): CONFIG["TIMEFRAME"]
            timeframe_base = CONFIG.get("TIMEFRAME", None)
            try:
                archivo_data = resolve_archivo_data_tf(activo, timeframe_base, formato="parquet")
            except Exception:
                # Fallback legacy
                archivo_data = resolve_archivo_data(activo)
            qty_max_activo = resolve_qty_max_activo(activo)
            qty_max_range = resolve_qty_max_activo_range(activo)

        # 1. DATA PIPELINE (por activo)
            df = load_data(archivo_data)
            df_filtrado = filter_by_date(df, FECHA_INICIO, FECHA_FIN)
            if df is not df_filtrado:
                del df
                HealthGuard.force_cleanup()

            # Cache por activo: evita re-leer parquet de timeframes en cada estrategia.
            base_tf_suffix = normalize_timeframe_to_suffix(timeframe_base)
            tf_cache: dict[str, object] = {base_tf_suffix: df_filtrado}

        # 2. BACKTEST CONFIGURATION (por activo)
            cfg = BacktestConfig(
                saldo_inicial=float(CONFIG.get("SALDO_INICIAL", 300)),
                saldo_operativo_max=float(CONFIG.get("SALDO_OPERATIVO_MAX", 100000)),
                apalancamiento=float(CONFIG.get("APALANCAMIENTO", 1)),
                comision_pct=float(CONFIG.get("COMISION_PCT", 0.0001)),
                comision_sides=int(CONFIG.get("COMISION_SIDES", 2)),
                saldo_minimo_operativo=float(CONFIG.get("SALDO_MINIMO_OPERATIVO", 5.0)),
                qty_max_activo=float(qty_max_activo),

                # Optuna qty cap (per asset)
                optimize_qty_max_activo=bool(CONFIG.get("OPTIMIZAR_QTY_ACTIVO", False)),
                qty_max_activo_range=tuple(qty_max_range),

                # Global exits (engine-owned)
                exit_atr_period=int(CONFIG.get("EXIT_ATR_PERIOD", 14)),
                exit_sl_atr=float(CONFIG.get("EXIT_SL_ATR", 1.0)),
                exit_tp_atr=float(CONFIG.get("EXIT_TP_ATR", 1.0)),
                exit_time_stop_bars=int(CONFIG.get("EXIT_TIME_STOP_BARS", 260)),

                # Optuna exits
                optimize_exits=bool(CONFIG.get("OPTIMIZAR_SALIDAS", False)),
                exit_atr_period_range=tuple(CONFIG.get("EXIT_ATR_PERIOD_RANGE", (7, 30, 1))),
                exit_sl_atr_range=tuple(CONFIG.get("EXIT_SL_ATR_RANGE", (0.5, 5.0, 0.1))),
                exit_tp_atr_range=tuple(CONFIG.get("EXIT_TP_ATR_RANGE", (0.5, 8.0, 0.1))),
                exit_time_stop_bars_range=tuple(CONFIG.get("EXIT_TIME_STOP_BARS_RANGE", (50, 800, 10))),
            )

            for strat_id in lista_ids:
                # --- MANTENIMIENTO ANTES DE CADA ESTRATEGIA ---
                HealthGuard.check_system_health()

                if strat_id is None:
                    strategies = instantiate_strategies(only_id=None)
                else:
                    strategies = instantiate_strategies(only_id=int(strat_id))

                if not strategies:
                    continue

                for strategy in strategies:
                    strategy_name = strategy.name if hasattr(strategy, 'name') else "MODELOX_STRAT"

                    # Normaliza nombre para carpeta (evita espacios y caracteres raros)
                    strategy_safe = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(strategy_name).strip()).upper() or "MODELOX_STRAT"

                    mostrar_cabecera_inicio(
                        activo=activo,
                        combo_nombre=strategy_name,
                        indicadores=list(strategy.parametros_optuna.keys()),
                        n_trials=int(N_TRIALS),
                        archivo_data=archivo_data,
                    )

                    # REPORTERS SETUP
                    activo_safe = str(activo).upper()
                    # Estructura requerida:
                    # resultados/<ESTRATEGIA>/excel/<ACTIVO>/...
                    # resultados/<ESTRATEGIA>/graficos/<ACTIVO>/...
                    strategy_root_dir = os.path.join("resultados", strategy_safe)
                    excel_dir = os.path.join(strategy_root_dir, "excel")
                    graficos_dir = os.path.join(strategy_root_dir, "graficos", activo_safe)

                    # Crear carpeta base de estrategia (y excel) por adelantado
                    os.makedirs(excel_dir, exist_ok=True)
                    os.makedirs(os.path.dirname(graficos_dir), exist_ok=True)

                    reporters = [
                        ElegantRichReporter(saldo_inicial=cfg.saldo_inicial, activo=activo),
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
                                saldo_inicial=cfg.saldo_inicial,
                                activo=activo,
                            )
                        )

                    # RUNNER EXECUTION
                    runner = OptimizationRunner(config=cfg, n_trials=int(N_TRIALS), reporters=reporters)

                    # [MAC-OPTIMIZED] Configuración vital para single-core performance estable
                    runner.optuna = OptunaConfig(
                        seed=None,
                        n_jobs=1,  # ESTRICTAMENTE 1. Optuna con n_jobs > 1 en Mac + Python es inestable.
                        storage=None,
                    )
                    runner.activo = activo

                    try:
                        # Multi-timeframe: si la estrategia define `timeframe_entry/timeframe_exit`,
                        # se carga ese TF; si no, se usa el TF base de CONFIG.
                        entry_tf = getattr(strategy, "timeframe_entry", None) or timeframe_base
                        exit_tf = getattr(strategy, "timeframe_exit", None) or timeframe_base

                        needed_tfs = [timeframe_base, entry_tf, exit_tf]

                        for tf in needed_tfs:
                            tf_suffix = normalize_timeframe_to_suffix(tf)
                            if tf_suffix in tf_cache:
                                continue
                            try:
                                path_tf = resolve_archivo_data_tf(activo, tf, formato="parquet")
                                df_tf = load_data(path_tf)
                                df_tf = filter_by_date(df_tf, FECHA_INICIO, FECHA_FIN)
                                tf_cache[tf_suffix] = df_tf
                            except Exception:
                                # Best-effort: si falta el archivo, se cae al base TF
                                continue

                        runner.optimize_strategies(
                            df=df_filtrado,
                            strategies=[strategy],
                            df_by_timeframe=tf_cache,  # cache contiene solo lo necesario (base + overrides usados)
                            base_timeframe=timeframe_base,
                        )

                        if hasattr(runner, '_last_study') and runner._last_study.best_trial:
                            study = runner._last_study
                            mostrar_fin_optimizacion(
                                total_trials=len(study.trials),
                                best_score=study.best_value,
                                best_trial=study.best_trial.number,
                                estrategia=strategy_name,
                            )
                    except KeyboardInterrupt:
                        # Salida limpia: sin traceback, con cleanup final garantizado.
                        raise
                    except Exception as e:
                        logging.error(f"Error optimizando estrategia {strategy_name}: {e}")
                    finally:
                        del runner
                        del reporters
                        HealthGuard.force_cleanup(deep=True)

            # Limpieza por activo
            del df_filtrado
            HealthGuard.force_cleanup(deep=True)

    except KeyboardInterrupt:
        # No spamear tracebacks al usuario; el atexit/signal ya dispara cleanup.
        return
    finally:
        # Limpieza final SIEMPRE.
        HealthGuard.final_cleanup(reason="main:finally")

if __name__ == "__main__":
    main()