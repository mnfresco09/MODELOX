"""
# =============================================================================
#
#     ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗      ██████╗ ██╗  ██╗
#     ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║     ██╔═══██╗╚██╗██╔╝
#     ██╔████╔██║██║   ██║██║  ██║█████╗  ██║     ██║   ██║ ╚███╔╝
#     ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║     ██║   ██║ ██╔██╗
#     ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗╚██████╔╝██╔╝ ██╗
#     ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝
#
#     TYPES.PY - TIPOS Y ESTRUCTURAS DE DATOS FUNDAMENTALES
#
# =============================================================================
#
#     PROPÓSITO:
#     Define todas las estructuras de datos básicas del sistema:
#     - Configuración de backtesting
#     - Artefactos de cada trial
#     - Protocolos para estrategias y reporters
#     - Utilidades de fechas y timeframes
#     - Gestión de memoria
#
# =============================================================================
"""
from __future__ import annotations

import ctypes
import gc
import math
import platform
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Union

import pandas as pd
import polars as pl


# =============================================================================
# 1. TIPOS GLOBALES
# =============================================================================

# Tipo unificado para DataFrames (soporta Polars y Pandas)
TradesDF = Union[pd.DataFrame, pl.DataFrame]


# =============================================================================
# 2. CONFIGURACIÓN DE BACKTEST
# =============================================================================

@dataclass(frozen=True)
class BacktestConfig:
    """
    CONFIGURACIÓN PRINCIPAL DEL BACKTEST
    
    Contiene todos los parámetros necesarios para ejecutar un backtest:
    - Capital inicial y límites operativos
    - Comisiones
    - Sistema de salidas (SL/TP/Trailing)
    - Rangos de optimización para Optuna
    """
    
    # -------------------------------------------------------------------------
    # CAPITAL Y LÍMITES
    # -------------------------------------------------------------------------
    saldo_inicial: float
    saldo_operativo_max: float
    comision_pct: float
    comision_sides: int = 2
    saldo_minimo_operativo: float = 1.0
    qty_max_activo: float = float("inf")
    
    # -------------------------------------------------------------------------
    # POSITION SIZING
    # -------------------------------------------------------------------------
    saldo_usado: float = 75.0
    apalancamiento_max: float = 60.0
    riesgo_por_trade_pct: float = 0.10
    
    # -------------------------------------------------------------------------
    # SISTEMA DE SALIDAS (FUENTE: modelox/core/exits.py)
    # -------------------------------------------------------------------------
    exit_type: str = "pnl_trailing"
    exit_sl_pct: float = 8.0
    exit_tp_pct: float = 14.0
    exit_trail_act_pct: float = 15.0
    exit_trail_dist_pct: float = 3.0
    allow_custom_exits: bool = False
    
    # -------------------------------------------------------------------------
    # OPTIMIZACIÓN OPTUNA
    # -------------------------------------------------------------------------
    optimize_exits: bool = True
    exit_sl_pct_range: tuple[float, float, float] = (1.0, 50.0, 1.0)
    exit_tp_pct_range: tuple[float, float, float] = (20.0, 40.0, 1.0)
    exit_trail_act_pct_range: tuple[float, float, float] = (1.0, 50.0, 1.0)
    exit_trail_dist_pct_range: tuple[float, float, float] = (0.5, 20.0, 0.5)
    optimize_qty_max_activo: bool = False
    qty_max_activo_range: tuple[float, float, float] = (0.01, 5.0, 0.01)


# =============================================================================
# 3. ESTRUCTURAS DE RESULTADOS
# =============================================================================

@dataclass(frozen=True)
class ExitDecision:
    """DECISIÓN DE SALIDA: barra, motivo y precio."""
    exit_idx: int
    reason: str = ""
    exit_price: float | None = None


@dataclass(frozen=True)
class TrialArtifacts:
    """ARTEFACTOS DE UN TRIAL: params, métricas, trades, equity."""
    strategy_name: str
    trial_number: int
    params: Dict[str, Any]
    params_reporting: Dict[str, Any]
    score: float
    metrics: Dict[str, Any]
    df_signals: Optional[Union[pd.DataFrame, pl.DataFrame]]
    trades: TradesDF
    equity_curve: List[float]
    indicators_used: List[str]
    perturbado: bool = False
    perturb_seed: Optional[int] = None
    neighborhood_result: Optional[Dict[str, Any]] = None


# =============================================================================
# 4. PROTOCOLOS (INTERFACES)
# =============================================================================

class Reporter(Protocol):
    """INTERFAZ PARA REPORTERS DE EVENTOS."""
    
    def on_trial_end(self, artifacts: TrialArtifacts) -> None: ...
    def on_strategy_end(self, strategy_name: str, study: Any) -> None: ...


class Strategy(Protocol):
    """INTERFAZ PARA ESTRATEGIAS DE TRADING."""
    combinacion_id: int
    name: str
    
    def suggest_params(self, trial: Any) -> Dict[str, Any]: ...
    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame: ...


# =============================================================================
# 5. UTILIDADES DE FECHAS
# =============================================================================

def filter_by_date(df: pl.DataFrame, start: str, end: str) -> pl.DataFrame:
    """FILTRA DATAFRAME POR RANGO DE FECHAS (UTC)."""
    start_expr = pl.lit(start).str.to_datetime().dt.cast_time_unit("us").dt.replace_time_zone("UTC")
    end_expr = pl.lit(end).str.to_datetime().dt.cast_time_unit("us").dt.replace_time_zone("UTC")
    return df.filter(pl.col("timestamp").is_between(start_expr, end_expr))


def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """NORMALIZA ÍNDICE TEMPORAL A UTC (PANDAS)."""
    df = df.copy()
    col_time = next((c for c in ["timestamp", "date", "time", "datetime"] if c in df.columns), None)
    if col_time is not None:
        df[col_time] = pd.to_datetime(df[col_time], utc=True, errors="coerce")
        df = df.set_index(col_time)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


# =============================================================================
# 6. UTILIDADES DE TIMEFRAME
# =============================================================================

def normalize_timeframe_to_suffix(timeframe: Any) -> str:
    """NORMALIZA TIMEFRAME A SUFIJO: 5 -> "5m", 60 -> "1h", "15m" -> "15m"."""
    if timeframe is None:
        return "1h"
    
    if isinstance(timeframe, (int, float)):
        m = int(timeframe)
        if m <= 0:
            return "1h"
        if m % 60 == 0:
            h = int(m // 60)
            return "1h" if h == 1 else f"{h}h"
        return f"{m}m"
    
    s = str(timeframe).strip().lower()
    if not s:
        return "1h"
    
    if s.endswith("h"):
        try:
            h = int(float(s[:-1]))
            return "1h" if h == 1 else f"{h}h"
        except Exception:
            return "1h"
    
    if s.endswith("m"):
        s = s[:-1]
    
    try:
        m = int(float(s))
        if m <= 0:
            return "1h"
        if m % 60 == 0:
            h = int(m // 60)
            return "1h" if h == 1 else f"{h}h"
        return f"{m}m"
    except Exception:
        return "1h"


def suffix_to_minutes(suffix: str) -> int:
    """CONVIERTE SUFIJO A MINUTOS: "5m" -> 5, "1h" -> 60."""
    s = str(suffix).strip().lower()
    if s.endswith("h"):
        try:
            return int(round(float(s[:-1]) * 60.0))
        except Exception:
            return 60
    if s.endswith("m"):
        s = s[:-1]
    return int(float(s))


def convert_warmup_bars_to_base(*, warmup_bars: int, from_tf: str, to_tf: str) -> int:
    """CONVIERTE BARRAS DE WARMUP ENTRE TIMEFRAMES."""
    wb = int(warmup_bars)
    if wb <= 0:
        return 0
    f = suffix_to_minutes(from_tf)
    t = suffix_to_minutes(to_tf)
    if t <= 0:
        return wb
    return int(math.ceil(wb * f / t))


def align_signals_to_base(*, df_base: pl.DataFrame, df_signals: pl.DataFrame) -> pl.DataFrame:
    """ALINEA SEÑALES AL TIMEFRAME BASE SIN LOOKAHEAD."""
    if "timestamp" not in df_base.columns or "timestamp" not in df_signals.columns:
        raise ValueError("Ambos DataFrames deben contener columna 'timestamp'")
    
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
# 7. GESTIÓN DE MEMORIA
# =============================================================================

def nuclear_cleanup() -> None:
    """LIMPIEZA AGRESIVA DE MEMORIA (TRIPLE GC + LIBERACIÓN OS)."""
    for _ in range(3):
        gc.collect()
    
    system = platform.system()
    if system == "Linux":
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass
    elif system == "Darwin":
        try:
            if hasattr(sys, '_clear_type_cache'):
                sys._clear_type_cache()
        except Exception:
            pass


def full_system_cleanup() -> None:
    """LIMPIEZA COMPLETA AL FINALIZAR OPTIMIZACIÓN."""
    modules_to_clear = ['numpy', 'pandas', 'polars', 'torch', 'numba']
    
    for mod_name in modules_to_clear:
        if mod_name in sys.modules:
            mod = sys.modules[mod_name]
            if mod_name == 'torch':
                try:
                    if hasattr(mod, 'cuda') and mod.cuda.is_available():
                        mod.cuda.empty_cache()
                    if hasattr(mod, 'mps') and hasattr(mod.mps, 'empty_cache'):
                        mod.mps.empty_cache()
                except Exception:
                    pass
    
    if 'matplotlib.pyplot' in sys.modules:
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except Exception:
            pass
    
    gc.collect()
    gc.collect()
    gc.collect()
    nuclear_cleanup()
    
    if platform.system() == "Darwin":
        try:
            import subprocess
            subprocess.run(['purge'], capture_output=True, timeout=5)
        except Exception:
            pass


def clean_trial_variables(*vars_to_delete: Any) -> None:
    """LIMPIA VARIABLES DE UN TRIAL Y EJECUTA GC."""
    for v in vars_to_delete:
        try:
            del v
        except (UnboundLocalError, NameError):
            pass
    nuclear_cleanup()
