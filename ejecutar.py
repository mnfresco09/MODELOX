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

from modelox.core.logging_config import setup_logging

# Configurar logging centralizado
setup_logging(level=logging.WARNING, enable_optuna=False)

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
from modelox.core.health_monitor import HealthGuard, get_health_monitor
from modelox.core.timeframe_cache import get_timeframe_cache
from modelox.core.execution_helpers import run_single_exit_type
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

    # Cadencia de limpieza durante Optuna (por defecto: cada 200 trials)
    # Esto lo consume `modelox/core/runner.py` vía env var.
    try:
        cleanup_every = int(CONFIG.get("CLEANUP_EVERY_N_TRIALS", 200))
    except Exception:
        cleanup_every = 200
    if cleanup_every and cleanup_every > 0:
        os.environ["MODELOX_CLEANUP_EVERY_N_TRIALS"] = str(int(cleanup_every))

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

    # 3b. Resolve base timeframes to run sequentially
    # - New: CONFIG["TIMEFRAMES"] can be a list like [5, 15, 60]
    # - Fallback: CONFIG["TIMEFRAME"]
    raw_timeframes = CONFIG.get("TIMEFRAMES", None)
    if isinstance(raw_timeframes, (list, tuple)) and raw_timeframes:
        timeframes_to_run = list(raw_timeframes)
    else:
        timeframes_to_run = [CONFIG.get("TIMEFRAME", None)]
    
    # 4. ITERATE OVER STRATEGIES
    # Convertimos a lista segura para evitar errores de iterador
    lista_ids = strategy_ids if strategy_ids is not None else [None]

    try:
        for activo in activos:
            for timeframe_base in timeframes_to_run:
                # --- MANTENIMIENTO ANTES DE CADA ACTIVO/TIMEFRAME ---
                HealthGuard.check_system_health()

                # Timeframe base configurable (minutos)
                try:
                    archivo_data = resolve_archivo_data_tf(activo, timeframe_base, formato="parquet")
                except Exception:
                    # Fallback legacy
                    archivo_data = resolve_archivo_data(activo)

                qty_max_activo = resolve_qty_max_activo(activo)
                qty_max_range = resolve_qty_max_activo_range(activo)

                # 1. DATA PIPELINE (por activo/timeframe)
                df = load_data(archivo_data)
                df_filtrado = filter_by_date(df, FECHA_INICIO, FECHA_FIN)
                if df is not df_filtrado:
                    del df
                    HealthGuard.force_cleanup()

                # Periodo real de datos usado (para mostrar en Rich)
                periodo_datos = ""
                try:
                    import polars as pl

                    ts_col = "timestamp" if "timestamp" in df_filtrado.columns else (
                        "datetime" if "datetime" in df_filtrado.columns else None
                    )
                    if ts_col:
                        ts_min = df_filtrado.select(pl.col(ts_col).min()).item()
                        ts_max = df_filtrado.select(pl.col(ts_col).max()).item()
                        if ts_min is not None and ts_max is not None:
                            # dt puede venir como datetime python (tz-aware) -> strftime OK
                            periodo_datos = f"{ts_min:%Y-%m-%d %H:%M} → {ts_max:%Y-%m-%d %H:%M} UTC"
                except Exception:
                    periodo_datos = ""

                # Cache por activo/timeframe: evita re-leer parquet de timeframes en cada estrategia.
                base_tf_suffix = normalize_timeframe_to_suffix(timeframe_base)
                tf_cache: dict[str, object] = {base_tf_suffix: df_filtrado}

                # 2. BACKTEST CONFIGURATION (por activo/timeframe)
                cfg = BacktestConfig(
                    saldo_inicial=float(CONFIG.get("SALDO_INICIAL", 300)),
                    saldo_operativo_max=float(CONFIG.get("SALDO_OPERATIVO_MAX", 100000)),
                    comision_pct=float(CONFIG.get("COMISION_PCT", 0.0001)),
                    comision_sides=int(CONFIG.get("COMISION_SIDES", 2)),
                    saldo_minimo_operativo=float(CONFIG.get("SALDO_MINIMO_OPERATIVO", 5.0)),
                    qty_max_activo=float(qty_max_activo),

                    # Position Sizing: SALDO_USADO fijo, QTY fija, APALANCAMIENTO variable
                    saldo_usado=float(CONFIG.get("SALDO_USADO", 75.0)),
                    apalancamiento_max=float(CONFIG.get("APALANCAMIENTO_MAX", 60.0)),

                    # Fixed Fractional Sizing (legacy)
                    riesgo_por_trade_pct=float(CONFIG.get("RIESGO_POR_TRADE_PCT", 0.10)),

                    # Optuna qty cap (per asset)
                    optimize_qty_max_activo=bool(CONFIG.get("OPTIMIZAR_QTY_ACTIVO", False)),
                    qty_max_activo_range=tuple(qty_max_range),

                    # Sistema de Salidas Porcentual
                    exit_sl_pct=float(CONFIG.get("EXIT_SL_PCT", 1.0)),
                    exit_tp_pct=float(CONFIG.get("EXIT_TP_PCT", 3.0)),
                    exit_trail_act_pct=float(CONFIG.get("EXIT_TRAIL_ACT_PCT", 1.0)),
                    exit_trail_dist_pct=float(CONFIG.get("EXIT_TRAIL_DIST_PCT", 0.5)),

                    # Optuna exits
                    optimize_exits=bool(CONFIG.get("OPTIMIZAR_SALIDAS", True)),
                    exit_sl_pct_range=tuple(CONFIG.get("EXIT_SL_PCT_RANGE", (0.5, 5.0, 0.1))),
                    exit_tp_pct_range=tuple(CONFIG.get("EXIT_TP_PCT_RANGE", (1.0, 10.0, 0.1))),
                    exit_trail_act_pct_range=tuple(CONFIG.get("EXIT_TRAIL_ACT_PCT_RANGE", (0.5, 3.0, 0.1))),
                    exit_trail_dist_pct_range=tuple(CONFIG.get("EXIT_TRAIL_DIST_PCT_RANGE", (0.2, 2.0, 0.1))),
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

                        # Timeframes usados (base/entry/exit) en formato legible
                        base_tf = normalize_timeframe_to_suffix(timeframe_base)
                        entry_tf = normalize_timeframe_to_suffix(getattr(strategy, "timeframe_entry", None) or base_tf)
                        exit_tf = normalize_timeframe_to_suffix(getattr(strategy, "timeframe_exit", None) or base_tf)
                        if entry_tf == base_tf and exit_tf == base_tf:
                            tf_display = str(base_tf)
                        else:
                            tf_display = f"BASE {base_tf} · ENTRY {entry_tf} · EXIT {exit_tf}"

                        # Get exit type from config
                        exit_type = str(CONFIG.get("EXIT_TYPE", "pnl_trailing"))
                        
                        # Determine if we need to run multiple exit types
                        exit_types_to_run = []
                        if exit_type.lower() == "all":
                            exit_types_to_run = ["pnl_fixed", "pnl_trailing"]
                        else:
                            exit_types_to_run = [exit_type]

                        # Iterate over exit types (one or multiple)
                        for current_exit_type in exit_types_to_run:
                            run_single_exit_type(
                            exit_type=current_exit_type,
                            strategy=strategy,
                            strategy_name=strategy_name,
                            strategy_safe=strategy_safe,
                            activo=activo,
                            df_filtrado=df_filtrado,
                            tf_cache=tf_cache,
                            timeframe_base=timeframe_base,
                            cfg=cfg,
                            tf_display=tf_display,
                            archivo_data=archivo_data,
                            periodo_datos=periodo_datos,
                            is_all_mode=(exit_type.lower() == "all"),
                            resolve_archivo_data_tf_func=resolve_archivo_data_tf,
                            fecha_inicio=FECHA_INICIO,
                            fecha_fin=FECHA_FIN,
                        )

                # Limpieza por activo/timeframe
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