"""modelox.core — API pública del núcleo."""

from .types import BacktestConfig, Reporter, Strategy, TrialArtifacts
from .data import (
    load_data,
    prepare_multitimeframe_data,
    resample_ohlcv,
    resample_to_base_timeframe,
    candles_per_month_for_tf,
    get_available_timeframes,
)
from .metrics import (
    resumen_metricas,
    sharpe, sortino, sqn, max_drawdown,
    profit_factor, payoff_ratio, expectativa,
    winrate_pct, roi_pct, calmar, trades_por_dia,
)
from .exits import (
    ExitSettings,
    ExitResult,
    resolve_exit_settings_for_trial,
    exit_settings_from_params,
    DEFAULT_EXIT_TYPE,
    DEFAULT_EXIT_SL_PCT,
    DEFAULT_EXIT_TP_PCT,
)

__all__ = [
    # types
    "BacktestConfig", "Reporter", "Strategy", "TrialArtifacts",
    # data
    "load_data", "prepare_multitimeframe_data", "resample_ohlcv",
    "resample_to_base_timeframe", "candles_per_month_for_tf", "get_available_timeframes",
    # metrics
    "resumen_metricas", "sharpe", "sortino", "sqn", "max_drawdown",
    "profit_factor", "payoff_ratio", "expectativa", "winrate_pct",
    "roi_pct", "calmar", "trades_por_dia",
    # exits
    "ExitSettings", "ExitResult", "resolve_exit_settings_for_trial",
    "exit_settings_from_params", "DEFAULT_EXIT_TYPE",
    "DEFAULT_EXIT_SL_PCT", "DEFAULT_EXIT_TP_PCT",
]
