"""modelox.core.types — Tipos, protocolos y utilidades base del sistema.

Base del core: todos los demás módulos del core importan de aquí.
Solo importa de exits.py de forma diferida para evitar ciclos.
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
    """Obtiene defaults de salida desde exits.py (evita import circular)."""
    from modelox.core.exits import (
        DEFAULT_EXIT_TYPE, DEFAULT_EXIT_SL_PCT, DEFAULT_EXIT_TP_PCT,
        DEFAULT_EXIT_TRAIL_ACT_PCT, DEFAULT_EXIT_TRAIL_DIST_PCT,
        DEFAULT_EXIT_TIME_BARS,
        DEFAULT_OPTIMIZE_EXITS,
        DEFAULT_EXIT_SL_PCT_RANGE, DEFAULT_EXIT_TP_PCT_RANGE,
        DEFAULT_EXIT_TRAIL_ACT_PCT_RANGE, DEFAULT_EXIT_TRAIL_DIST_PCT_RANGE,
        DEFAULT_EXIT_TIME_BARS_RANGE,
    )
    return {
        "exit_type": DEFAULT_EXIT_TYPE,
        "exit_sl_pct": DEFAULT_EXIT_SL_PCT,
        "exit_tp_pct": DEFAULT_EXIT_TP_PCT,
        "exit_trail_act_pct": DEFAULT_EXIT_TRAIL_ACT_PCT,
        "exit_trail_dist_pct": DEFAULT_EXIT_TRAIL_DIST_PCT,
        "exit_time_bars": DEFAULT_EXIT_TIME_BARS,
        "optimize_exits": DEFAULT_OPTIMIZE_EXITS,
        "exit_sl_pct_range": DEFAULT_EXIT_SL_PCT_RANGE,
        "exit_tp_pct_range": DEFAULT_EXIT_TP_PCT_RANGE,
        "exit_trail_act_pct_range": DEFAULT_EXIT_TRAIL_ACT_PCT_RANGE,
        "exit_trail_dist_pct_range": DEFAULT_EXIT_TRAIL_DIST_PCT_RANGE,
        "exit_time_bars_range": DEFAULT_EXIT_TIME_BARS_RANGE,
    }


# =============================================================================
# 3. CONFIGURACIÓN DE BACKTEST
# =============================================================================

_EXIT_DEFAULTS = _get_exit_defaults()


@dataclass(frozen=True)
class BacktestConfig:
    """Configuración completa de un backtest. Defaults de exit sincronizados desde exits.py."""

    # ─── CAPITAL ─────────────────────────────────────────────────────────
    saldo_inicial: float                         # Saldo de partida ($)
    saldo_operativo_max: float                   # Saldo máximo operativo ($)
    comision_pct: float                          # Comisión por lado (%)
    comision_sides: int = 2                      # Lados con comisión (1 o 2)
    saldo_minimo_operativo: float = 1.0          # Mínimo para seguir operando ($)

    # ─── POSITION SIZING ─────────────────────────────────────────────────
    # qty = (saldo_usado × apalancamiento_max) / precio_entrada
    saldo_usado: float = 75.0                    # Margen fijo por trade ($)
    apalancamiento_max: float = 60.0             # Apalancamiento máximo (x)

    # ─── SISTEMA DE SALIDAS (SINCRONIZADO DESDE exits.py) ────────────────
    # IMPORTANTE: No modificar estos defaults aquí. Cambiarlos en exits.py.
    # Los valores aquí son solo fallbacks de emergencia.
    # El runner SIEMPRE usa resolve_exit_settings_for_trial() de exits.py.
    exit_type: str = _EXIT_DEFAULTS["exit_type"]                 # "FIXED" | "TRAILING" | "BARS" | "ATR"
    exit_sl_pct: float = _EXIT_DEFAULTS["exit_sl_pct"]           # Stop Loss % sobre stake
    exit_tp_pct: float = _EXIT_DEFAULTS["exit_tp_pct"]           # Take Profit % sobre stake
    exit_trail_act_pct: float = _EXIT_DEFAULTS["exit_trail_act_pct"]   # Activación trailing % sobre stake
    exit_trail_dist_pct: float = _EXIT_DEFAULTS["exit_trail_dist_pct"] # Distancia trailing % sobre stake
    exit_time_bars: int = _EXIT_DEFAULTS["exit_time_bars"]       # Barras para time_bars (0=off)

    # ─── OPTIMIZACIÓN DE SALIDAS (OPTUNA) ────────────────────────────────
    optimize_exits: bool = True
    exit_sl_pct_range: tuple[float, float, float] = _EXIT_DEFAULTS["exit_sl_pct_range"]
    exit_tp_pct_range: tuple[float, float, float] = _EXIT_DEFAULTS["exit_tp_pct_range"]
    exit_trail_act_pct_range: tuple[float, float, float] = _EXIT_DEFAULTS["exit_trail_act_pct_range"]
    exit_trail_dist_pct_range: tuple[float, float, float] = _EXIT_DEFAULTS["exit_trail_dist_pct_range"]
    exit_time_bars_range: tuple[int, int, int] = _EXIT_DEFAULTS["exit_time_bars_range"]


# =============================================================================
# 4. ARTEFACTOS DE TRIAL
# =============================================================================

@dataclass(frozen=True)
class ExitDecision:
    """Decisión de salida de una estrategia personalizada (SALIDAS_PERSONALIZADAS=True)."""
    exit_idx: int                                # Índice de la barra de salida
    reason: str = ""                             # Razón de la salida ("sl", "tp", etc.)
    exit_price: float | None = None              # Precio de salida (None = close)


@dataclass(frozen=True)
class TrialArtifacts:
    """Artefactos generados por cada trial: métricas, trades, equity, parámetros."""
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


    # ─── METADATA EXTRA ──────────────────────────────────────────────────
    trial_date_range: Optional[tuple] = None     # (fecha_inicio, fecha_fin)
    neighborhood_result: Optional[Dict[str, Any]] = None  # Análisis de vecindario


# =============================================================================
# 5. PROTOCOLOS (INTERFACES)
# =============================================================================

class Reporter(Protocol):
    """Protocolo para reporters (Excel, Rich, Telegram, etc.)."""
    def on_trial_end(self, artifacts: TrialArtifacts) -> None: ...
    def on_strategy_end(self, strategy_name: str, study: Any) -> None: ...


class Strategy(Protocol):
    """Protocolo para estrategias: combinacion_id, name, suggest_params(), generate_signals()."""
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
    """Filtra df por rango de fechas [start, end] con alineación UTC."""
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


# =============================================================================
# 7. UTILIDADES DE TIMEFRAME
# =============================================================================

def normalize_timeframe_to_suffix(timeframe: Any) -> str:
    """Normaliza int/str de timeframe a sufijo estándar ("1m", "5m", "1h", "1d"). Default: "1m"."""
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
    """Convierte sufijo de TF a minutos ("1h" → 60, "1d" → 1440)."""
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
    """Convierte warmup en barras de from_tf a barras equivalentes en to_tf."""
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
    """Une señales al base vía join_asof backward (anti-lookahead)."""
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
    """GC triple + malloc_trim (Linux) o _clear_type_cache (macOS)."""
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
    """Limpieza completa al finalizar: cachés, matplotlib, GC agresivo."""
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


# =============================================================================
# 10. MANTENIMIENTO — PYCACHE Y HEALTHGUARD
# =============================================================================
import shutil
import os


def purge_pycache(*, root, exclude: set = None) -> None:
    """Elimina __pycache__ y .pyc recursivamente desde root."""
    from pathlib import Path
    exclude = exclude or set()

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
    """Gestor de limpieza al cierre. Configurar + atexit.register(HealthGuard.final_cleanup)."""
    _project_root = None
    _purge_pycache_on_exit = False

    @classmethod
    def configure(cls, *, project_root, purge_pycache: bool = False):
        cls._project_root = project_root
        cls._purge_pycache_on_exit = purge_pycache

    @staticmethod
    def final_cleanup():
        nuclear_cleanup()
        if HealthGuard._purge_pycache_on_exit and HealthGuard._project_root:
            purge_pycache(
                root=HealthGuard._project_root,
                exclude={".git", ".venv", "data"},
            )
