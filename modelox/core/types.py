"""
================================================================================
MODELOX/CORE/TYPES.PY — TIPOS, PROTOCOLOS Y UTILIDADES BASE
================================================================================

PROPÓSITO:
    Define TODOS los tipos, protocolos y dataclasses que el sistema necesita.
    Es la BASE de la que dependen todos los demás módulos del core.

CONTENIDO:
    1. TIPOS UNIFICADOS           — TradesDF (Polars | Pandas)
    2. IMPORTACIÓN DE DEFAULTS    — Acceso a exits.py sin circular imports
    3. CONFIGURACIÓN DE BACKTEST  — BacktestConfig (frozen dataclass)
    4. ARTEFACTOS DE TRIAL        — ExitDecision, TrialArtifacts
    5. PROTOCOLOS                 — Reporter, Strategy (interfaces)
    6. UTILIDADES DE FECHAS       — filter_by_date, ensure_utc_index
    7. UTILIDADES DE TIMEFRAME    — normalize_timeframe_to_suffix, suffix_to_minutes
    8. ALINEACIÓN DE SEÑALES      — align_signals_to_base (join_asof anti-lookahead)
    9. GESTIÓN DE MEMORIA         — nuclear_cleanup, full_system_cleanup
   10. MANTENIMIENTO              — purge_pycache, HealthGuard

DEPENDENCIAS:
    → Este módulo NO importa de ningún otro módulo del core (excepto exits.py
      de forma diferida, para evitar ciclos).
    ← Todos los demás módulos del core importan de aquí.

================================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Union

import pandas as pd
import polars as pl


# =============================================================================
# 1. TIPOS UNIFICADOS
# =============================================================================
# TradesDF acepta tanto Polars como Pandas durante la transición.
# El objetivo final es usar solo Polars.

TradesDF = Union[pd.DataFrame, pl.DataFrame]


# =============================================================================
# 2. IMPORTACIÓN DIFERIDA DE DEFAULTS DE SALIDA
# =============================================================================
# Los valores por defecto de salidas se definen ÚNICAMENTE en exits.py.
# Esta función permite acceder a ellos sin crear import circular.

def _get_exit_defaults() -> dict:
    """
    OBTIENE LOS DEFAULTS DE SALIDA DESDE EXITS.PY (FUENTE ÚNICA DE VERDAD).

    Returns:
        Diccionario con todos los valores por defecto de salida.
    """
    from modelox.core.exits import (
        DEFAULT_EXIT_TYPE, DEFAULT_EXIT_SL_PCT, DEFAULT_EXIT_TP_PCT,
        DEFAULT_EXIT_TRAIL_ACT_PCT, DEFAULT_EXIT_TRAIL_DIST_PCT,
        DEFAULT_OPTIMIZE_EXITS,
        DEFAULT_EXIT_SL_PCT_RANGE, DEFAULT_EXIT_TP_PCT_RANGE,
        DEFAULT_EXIT_TRAIL_ACT_PCT_RANGE, DEFAULT_EXIT_TRAIL_DIST_PCT_RANGE,
    )
    return {
        "exit_type": DEFAULT_EXIT_TYPE,
        "exit_sl_pct": DEFAULT_EXIT_SL_PCT,
        "exit_tp_pct": DEFAULT_EXIT_TP_PCT,
        "exit_trail_act_pct": DEFAULT_EXIT_TRAIL_ACT_PCT,
        "exit_trail_dist_pct": DEFAULT_EXIT_TRAIL_DIST_PCT,
        "optimize_exits": DEFAULT_OPTIMIZE_EXITS,
        "exit_sl_pct_range": DEFAULT_EXIT_SL_PCT_RANGE,
        "exit_tp_pct_range": DEFAULT_EXIT_TP_PCT_RANGE,
        "exit_trail_act_pct_range": DEFAULT_EXIT_TRAIL_ACT_PCT_RANGE,
        "exit_trail_dist_pct_range": DEFAULT_EXIT_TRAIL_DIST_PCT_RANGE,
    }


# =============================================================================
# 3. CONFIGURACIÓN DE BACKTEST
# =============================================================================

@dataclass(frozen=True)
class BacktestConfig:
    """
    CONFIGURACIÓN COMPLETA PARA UN BACKTEST.

    NOTA IMPORTANTE SOBRE SALIDAS:
        Los parámetros de salida (exit_*) tienen valores por defecto que se
        sincronizan automáticamente desde modelox/core/exits.py.

        La fuente ÚNICA de verdad para defaults de salida es exits.py.
        Cualquier modificación de valores por defecto debe hacerse allí.

        Para usar valores personalizados en tiempo de ejecución:
        1. Modificar configuracion.py (que importa de exits.py).
        2. Los valores se propagan automáticamente vía runner.py.
    """

    # ─── CAPITAL ─────────────────────────────────────────────────────────
    saldo_inicial: float                         # Saldo de partida ($)
    saldo_operativo_max: float                   # Saldo máximo operativo ($)
    comision_pct: float                          # Comisión por lado (%)
    comision_sides: int = 2                      # Lados con comisión (1 o 2)
    saldo_minimo_operativo: float = 1.0          # Mínimo para seguir operando ($)
    qty_max_activo: float = float("inf")         # Límite de cantidad máxima

    # ─── POSITION SIZING ─────────────────────────────────────────────────
    saldo_usado: float = 75.0                    # Margen fijo por trade ($)
    apalancamiento_max: float = 60.0             # Apalancamiento máximo (x)
    riesgo_por_trade_pct: float = 0.10           # Riesgo % por trade (legacy)

    # ─── SISTEMA DE SALIDAS (SINCRONIZADO DESDE exits.py) ────────────────
    # IMPORTANTE: No modificar estos defaults aquí. Cambiarlos en exits.py.
    # Los valores aquí son solo fallbacks de emergencia.
    # El runner SIEMPRE usa resolve_exit_settings_for_trial() de exits.py.
    exit_type: str = "pnl_fixed"                 # "pnl_fixed" o "pnl_trailing"
    exit_sl_pct: float = 8.0                     # Stop Loss % sobre stake
    exit_tp_pct: float = 14.0                    # Take Profit % sobre stake
    exit_trail_act_pct: float = 15.0             # Activación trailing % sobre stake
    exit_trail_dist_pct: float = 3.0             # Distancia trailing % sobre stake

    # ─── OPTIMIZACIÓN DE SALIDAS (OPTUNA) ────────────────────────────────
    optimize_exits: bool = True
    exit_sl_pct_range: tuple[float, float, float] = (7.0, 27.0, 1.0)
    exit_tp_pct_range: tuple[float, float, float] = (20.0, 40.0, 1.0)
    exit_trail_act_pct_range: tuple[float, float, float] = (10.0, 28.0, 1.0)
    exit_trail_dist_pct_range: tuple[float, float, float] = (2.0, 8.0, 0.5)

    # ─── OPTIMIZACIÓN DE CANTIDAD POR ACTIVO ─────────────────────────────
    optimize_qty_max_activo: bool = False
    qty_max_activo_range: tuple[float, float, float] = (0.01, 5.0, 0.01)


# =============================================================================
# 4. ARTEFACTOS DE TRIAL
# =============================================================================

@dataclass(frozen=True)
class ExitDecision:
    """
    DECISIÓN DE SALIDA GENERADA POR UNA ESTRATEGIA PERSONALIZADA.

    Solo se usa cuando la estrategia tiene SALIDAS_PERSONALIZADAS = True.
    """
    exit_idx: int                                # Índice de la barra de salida
    reason: str = ""                             # Razón de la salida ("sl", "tp", etc.)
    exit_price: float | None = None              # Precio de salida (None = close)


@dataclass(frozen=True)
class TrialArtifacts:
    """
    ARTEFACTOS GENERADOS POR CADA TRIAL DE OPTIMIZACIÓN.

    Contiene todo lo necesario para reporting, plots y análisis post-trial.
    """
    # ─── IDENTIFICACIÓN ──────────────────────────────────────────────────
    strategy_name: str                           # Nombre de la estrategia
    trial_number: int                            # Número de trial en Optuna

    # ─── PARÁMETROS ──────────────────────────────────────────────────────
    params: Dict[str, Any]                       # Params completos (incluye __)
    params_reporting: Dict[str, Any]             # Params limpios para display

    # ─── RESULTADOS ──────────────────────────────────────────────────────
    score: float                                 # Score del optimizador
    metrics: Dict[str, Any]                      # Diccionario de métricas
    df_signals: Optional[Union[pd.DataFrame, pl.DataFrame]]  # OHLCV + indicadores
    trades: TradesDF                             # DataFrame de trades
    equity_curve: List[float]                    # Curva de equity
    indicators_used: List[str]                   # Indicadores para plots

    # ─── PERTURBACIÓN ────────────────────────────────────────────────────
    perturbado: bool = False                     # ¿Datos perturbados?
    perturb_seed: Optional[int] = None           # Seed de perturbación

    # ─── METADATA EXTRA ──────────────────────────────────────────────────
    trial_date_range: Optional[tuple] = None     # (fecha_inicio, fecha_fin)
    neighborhood_result: Optional[Dict[str, Any]] = None  # Análisis de vecindario


# =============================================================================
# 5. PROTOCOLOS (INTERFACES)
# =============================================================================

class Reporter(Protocol):
    """
    PROTOCOLO PARA REPORTERS (EXCEL, RICH, TELEGRAM, ETC.).

    Todo reporter debe implementar estos dos métodos.
    """
    def on_trial_end(self, artifacts: TrialArtifacts) -> None: ...
    def on_strategy_end(self, strategy_name: str, study: Any) -> None: ...


class Strategy(Protocol):
    """
    PROTOCOLO PARA ESTRATEGIAS DE TRADING.

    Toda estrategia debe tener:
        - combinacion_id: Identificador único (corresponde al nombre del archivo)
        - name: Nombre descriptivo
        - suggest_params(): Define el espacio de búsqueda de Optuna
        - generate_signals(): Genera señales long/short sobre un DataFrame
    """
    combinacion_id: int
    name: str

    def suggest_params(self, trial: Any) -> Dict[str, Any]: ...
    def generate_signals(
        self, df: pl.DataFrame, params: Dict[str, Any]
    ) -> pl.DataFrame: ...


# =============================================================================
# 6. UTILIDADES DE FECHAS
# =============================================================================

def filter_by_date(df: pl.DataFrame, start: str, end: str) -> pl.DataFrame:
    """
    FILTRO DE FECHAS INTELIGENTE CON ALINEACIÓN DE PRECISIÓN Y ZONA HORARIA.

    Resuelve el error de supertipo al comparar datetime[ns, UTC] con datetime[us].
    La columna 'timestamp' ya está normalizada a UTC en data.py.

    Args:
        df:     DataFrame Polars con columna 'timestamp'.
        start:  Fecha de inicio como string ("YYYY-MM-DD").
        end:    Fecha de fin como string ("YYYY-MM-DD").

    Returns:
        DataFrame filtrado por rango de fechas [start, end].
    """
    start_expr = (
        pl.lit(start)
        .str.to_datetime()
        .dt.cast_time_unit("us")
        .dt.replace_time_zone("UTC")
    )
    end_expr = (
        pl.lit(end)
        .str.to_datetime()
        .dt.cast_time_unit("us")
        .dt.replace_time_zone("UTC")
    )
    return df.filter(pl.col("timestamp").is_between(start_expr, end_expr))


def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    CONVIERTE UN DATAFRAME PANDAS A ÍNDICE DATETIME UTC.

    Mantenido para compatibilidad con procesos que aún usen Pandas.

    Args:
        df: DataFrame Pandas con columna de timestamp.

    Returns:
        DataFrame con índice DatetimeIndex en UTC, ordenado.
    """
    df = df.copy()
    col_time = next(
        (c for c in ["timestamp", "date", "time", "datetime"] if c in df.columns), None
    )
    if col_time is not None:
        df[col_time] = pd.to_datetime(df[col_time], utc=True, errors="coerce")
        df = df.set_index(col_time)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df.sort_index()
    return df


# =============================================================================
# 7. UTILIDADES DE TIMEFRAME
# =============================================================================

def normalize_timeframe_to_suffix(timeframe: Any) -> str:
    """
    NORMALIZA CUALQUIER FORMATO DE TIMEFRAME A SUFIJO ESTÁNDAR.

    Acepta:
        - int/float (minutos): 1, 5, 15, 60, 120, 720, 1440...
        - str: "1", "1m", "5m", "15m", "1h", "2h", "12h", "1d"...

    Returns:
        Sufijo normalizado: "1m", "5m", "15m", "1h", "2h", "12h", "1d".
        Default: "1m" si no se puede parsear.
    """
    if timeframe is None:
        return "1m"

    # ─── ENTRADA NUMÉRICA (MINUTOS) ──────────────────────────────────────
    if isinstance(timeframe, (int, float)):
        m = int(timeframe)
        if m <= 0:
            return "1m"
        if m % 1440 == 0:
            d = int(m // 1440)
            return "1d" if d == 1 else f"{d}d"
        if m % 60 == 0:
            h = int(m // 60)
            return "1h" if h == 1 else f"{h}h"
        return f"{m}m"

    # ─── ENTRADA STRING ──────────────────────────────────────────────────
    s = str(timeframe).strip().lower()
    if not s:
        return "1m"

    if s.endswith("d"):
        try:
            d = int(float(s[:-1]))
        except Exception:
            return "1d"
        return "1d" if d == 1 else f"{d}d"

    if s.endswith("h"):
        try:
            h = int(float(s[:-1]))
        except Exception:
            return "1m"
        return "1h" if h == 1 else f"{h}h"

    if s.endswith("m"):
        s = s[:-1]

    try:
        m = int(float(s))
    except Exception:
        return "1m"

    if m <= 0:
        return "1m"
    if m % 1440 == 0:
        d = int(m // 1440)
        return "1d" if d == 1 else f"{d}d"
    if m % 60 == 0:
        h = int(m // 60)
        return "1h" if h == 1 else f"{h}h"
    return f"{m}m"


def suffix_to_minutes(suffix: str) -> int:
    """
    CONVIERTE UN SUFIJO DE TIMEFRAME A MINUTOS.

    Args:
        suffix: Sufijo como "1m", "5m", "1h", "1d", etc.

    Returns:
        Número entero de minutos.
    """
    s = str(suffix).strip().lower()
    if s.endswith("d"):
        try:
            d = float(s[:-1])
            return int(round(d * 1440.0))
        except Exception:
            return 1440
    if s.endswith("h"):
        try:
            h = float(s[:-1])
            return int(round(h * 60.0))
        except Exception:
            return 60
    if s.endswith("m"):
        s = s[:-1]
    return int(float(s))


def convert_warmup_bars_to_base(*, warmup_bars: int, from_tf: str, to_tf: str) -> int:
    """
    CONVIERTE WARMUP EXPRESADO EN BARRAS DE UN TF A BARRAS DE OTRO TF.

    Ejemplo:
        warmup=50 en 15m → base=5m → 150 barras.

    Args:
        warmup_bars: Número de barras de warmup en el TF origen.
        from_tf:     Timeframe origen (ej: "15m").
        to_tf:       Timeframe destino (ej: "5m").

    Returns:
        Número de barras equivalentes en el TF destino.
    """
    wb = int(warmup_bars)
    if wb <= 0:
        return 0
    f = suffix_to_minutes(from_tf)
    t = suffix_to_minutes(to_tf)
    if t <= 0:
        return wb
    mins = wb * f
    return int(math.ceil(mins / t))


# =============================================================================
# 8. ALINEACIÓN DE SEÑALES (ANTI-LOOKAHEAD)
# =============================================================================

def align_signals_to_base(
    *,
    df_base: pl.DataFrame,
    df_signals: pl.DataFrame,
) -> pl.DataFrame:
    """
    ALINEA SEÑALES/INDICADORES AL DATAFRAME BASE SIN LOOKAHEAD.

    Usa join_asof con estrategia "backward": cada fila del base recibe
    el último valor de señal cuyo timestamp <= timestamp del base.

    Reglas:
        - Conserva OHLCV del base timeframe (no se sobreescriben).
        - Copia columnas no-OHLCV desde df_signals.
        - signal_long / signal_short se rellenan con False si son null.

    Args:
        df_base:    DataFrame con datos OHLCV del TF de operación.
        df_signals: DataFrame con señales e indicadores.

    Returns:
        DataFrame base enriquecido con señales alineadas.
    """
    if "timestamp" not in df_base.columns:
        raise ValueError("df_base must contain 'timestamp'")
    if "timestamp" not in df_signals.columns:
        raise ValueError("df_signals must contain 'timestamp'")

    base = df_base.sort("timestamp")
    sig = df_signals.sort("timestamp")

    skip = {"open", "high", "low", "close", "volume"}
    extra_cols = [c for c in sig.columns if c not in skip and c != "timestamp"]
    sig_small = sig.select(["timestamp", *extra_cols])

    out = base.join_asof(sig_small, on="timestamp", strategy="backward")

    for col in ("signal_long", "signal_short"):
        if col in out.columns:
            out = out.with_columns(pl.col(col).fill_null(False).cast(pl.Boolean))

    return out


# =============================================================================
# 9. GESTIÓN DE MEMORIA
# =============================================================================
import gc
import platform
import ctypes
import sys


def nuclear_cleanup() -> None:
    """
    LIMPIEZA AGRESIVA DE MEMORIA (COMPATIBLE macOS / Linux).

    Pasos:
        1. Fuerza la recolección de basura de Python (Generaciones 0, 1 y 2).
        2. En Linux, fuerza a la librería C (malloc) a devolver RAM al SO.
        3. En macOS, libera caché de tipos internos de Python.
    """
    # Triple GC para asegurar limpieza completa de referencias cíclicas
    for _ in range(3):
        gc.collect()

    system = platform.system()

    if system == "Linux":
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass
    elif system == "Darwin":  # macOS
        try:
            if hasattr(sys, '_clear_type_cache'):
                sys._clear_type_cache()
        except Exception:
            pass


def full_system_cleanup() -> None:
    """
    LIMPIEZA COMPLETA DEL SISTEMA AL FINALIZAR EJECUCIÓN.

    Libera RAM, cierra figuras matplotlib, limpia cachés de módulos.
    """
    # ─── 1. LIMPIAR CACHÉS DE MÓDULOS NUMÉRICOS ─────────────────────────
    modules_to_clear = ['numpy', 'pandas', 'polars', 'torch', 'numba']

    for mod_name in modules_to_clear:
        if mod_name in sys.modules:
            mod = sys.modules[mod_name]
            if mod_name == 'numpy' and hasattr(mod, 'finfo'):
                try:
                    mod.finfo.cache.clear() if hasattr(mod.finfo, 'cache') else None
                except Exception:
                    pass
            if mod_name == 'torch':
                try:
                    if hasattr(mod, 'cuda') and mod.cuda.is_available():
                        mod.cuda.empty_cache()
                    if hasattr(mod, 'mps') and hasattr(mod.mps, 'empty_cache'):
                        mod.mps.empty_cache()
                except Exception:
                    pass
            if mod_name == 'numba':
                try:
                    from numba.core import caching
                    if hasattr(caching, '_CacheImpl'):
                        caching._CacheImpl._cache.clear() if hasattr(caching._CacheImpl, '_cache') else None
                except Exception:
                    pass

    # ─── 2. CERRAR FIGURAS MATPLOTLIB ────────────────────────────────────
    if 'matplotlib.pyplot' in sys.modules:
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except Exception:
            pass

    # ─── 3. GC AGRESIVO ──────────────────────────────────────────────────
    gc.collect()
    gc.collect()
    gc.collect()

    # ─── 4. LIBERAR MEMORIA DEL SISTEMA ──────────────────────────────────
    nuclear_cleanup()

    # ─── 5. macOS: SUGERIR LIBERACIÓN DE MEMORIA COMPRIMIDA ──────────────
    if platform.system() == "Darwin":
        try:
            import subprocess
            subprocess.run(['purge'], capture_output=True, timeout=5)
        except Exception:
            pass


def clean_trial_variables(*vars_to_delete) -> None:
    """
    BORRA VARIABLES ESPECÍFICAS Y EJECUTA LIMPIEZA.

    Uso:
        clean_trial_variables(df, signals, trades)
    """
    for v in vars_to_delete:
        try:
            del v
        except (UnboundLocalError, NameError):
            pass
    nuclear_cleanup()


# =============================================================================
# 10. MANTENIMIENTO — PYCACHE Y HEALTHGUARD
# =============================================================================
import shutil
import os as _os_module


def purge_pycache(*, root, exclude: set = None) -> None:
    """
    ELIMINA CARPETAS __pycache__ Y ARCHIVOS .pyc RECURSIVAMENTE.

    Args:
        root:    Path raíz desde donde buscar.
        exclude: Set de nombres de carpetas a excluir (ej: {".git", ".venv"}).
    """
    from pathlib import Path
    exclude = exclude or set()

    try:
        for dirpath, dirnames, filenames in _os_module.walk(root):
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
    """
    GESTOR DE LIMPIEZA AL CIERRE DEL SISTEMA.

    Uso:
        import atexit
        from modelox.core.types import HealthGuard

        HealthGuard.configure(project_root=Path(__file__).parent, purge_pycache=True)
        atexit.register(HealthGuard.final_cleanup)
    """
    _project_root = None
    _purge_pycache_on_exit = False

    @classmethod
    def configure(cls, *, project_root, purge_pycache: bool = False):
        """CONFIGURA EL HEALTHGUARD ANTES DE REGISTRARLO."""
        cls._project_root = project_root
        cls._purge_pycache_on_exit = purge_pycache

    @staticmethod
    def final_cleanup():
        """LIMPIEZA FINAL AL CERRAR EL PROGRAMA."""
        nuclear_cleanup()
        if HealthGuard._purge_pycache_on_exit and HealthGuard._project_root:
            purge_pycache(
                root=HealthGuard._project_root,
                exclude={".git", ".venv", "data"},
            )
