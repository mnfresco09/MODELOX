from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

import pandas as pd
import polars as pl


@dataclass(frozen=True)
class BacktestConfig:
    saldo_inicial: float
    saldo_operativo_max: float
    apalancamiento: float
    comision_pct: float
    comision_sides: int = 2
    saldo_minimo_operativo: float = 1.0
    qty_max_activo: float = float("inf")

    # Global exit settings (engine-owned)
    exit_type: str = "atr_fixed"  # "atr_fixed", "trailing", or "all"
    exit_atr_period: int = 14
    exit_sl_atr: float = 1.0
    exit_tp_atr: float = 1.0
    exit_time_stop_bars: int = 260
    exit_trailing_atr_mult: float = 2.0
    exit_emergency_sl_atr_mult: float = 4.0

    # Optuna: allow optimizing global exits from configuration
    optimize_exits: bool = True
    # Ranges are inclusive; tuples support optional step: (min, max) or (min, max, step)
    exit_atr_period_range: tuple[int, int, int] = (7, 30, 1)
    exit_sl_atr_range: tuple[float, float, float] = (0.5, 3.0, 0.1)
    exit_tp_atr_range: tuple[float, float, float] = (1.0, 8.0, 0.1)
    exit_time_stop_bars_range: tuple[int, int, int] = (250, 800, 10)
    exit_trailing_atr_mult_range: tuple[float, float, float] = (1.0, 4.0, 0.1)
    exit_emergency_sl_atr_mult_range: tuple[float, float, float] = (2.0, 6.0, 0.1)

    # Optuna: allow optimizing qty cap per asset
    optimize_qty_max_activo: bool = False
    qty_max_activo_range: tuple[float, float, float] = (0.01, 5.0, 0.01)


@dataclass(frozen=True)
class ExitDecision:
    exit_idx: int
    reason: str = ""
    exit_price: float | None = None


@dataclass(frozen=True)
class TrialArtifacts:
    strategy_name: str
    trial_number: int
    params: Dict[str, Any]
    params_reporting: Dict[str, Any]
    score: float
    metrics: Dict[str, Any]
    # NOTE: df_signals (OHLCV + indicadores) puede ser costoso de construir.
    # Se permite None para optimizar trials donde no se generan plots.
    df_signals: Optional[pd.DataFrame]
    trades: pd.DataFrame
    equity_curve: List[float]
    indicators_used: List[str]


class Reporter(Protocol):
    def on_trial_end(self, artifacts: TrialArtifacts) -> None: ...
    def on_strategy_end(self, strategy_name: str, study: Any) -> None: ...


class Strategy(Protocol):
    combinacion_id: int
    name: str

    def suggest_params(self, trial: Any) -> Dict[str, Any]: ...
    def generate_signals(
        self, df: pl.DataFrame, params: Dict[str, Any]
    ) -> pl.DataFrame: ...


# NOTE:
# La lógica de salida vive en el engine; las estrategias sólo generan entradas.


def filter_by_date(df: pl.DataFrame, start: str, end: str) -> pl.DataFrame:
    """
    Filtro de fechas inteligente que alinea precisión y zona horaria.
    Resuelve el error de supertipo al comparar datetime[ns, UTC] con datetime[us].
    """
    # Preparamos los límites con precisión de microsegundos ('us') y en UTC.
    # La columna 'timestamp' ya está normalizada a UTC en data.py, pero aquí
    # hacemos el filtro robusto sin depender de atributos del dtype.
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
    """Mantenido para procesos secundarios que aún usen Pandas."""
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
