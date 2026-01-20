from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl


# =============================================================================
# CONFIGURACIÓN DE MÉTRICAS NUMBA
# =============================================================================
USE_NUMBA_METRICS = True  # Cambiar a False para usar métricas Python puras

try:
    from numba import njit

    NUMBA_METRICS_AVAILABLE = True
except Exception:
    NUMBA_METRICS_AVAILABLE = False
    USE_NUMBA_METRICS = False


def _empty_metrics_fast(saldo_inicial: float) -> Dict[str, Any]:
    return {
        "roi": 0.0,
        "winrate": 0.0,
        "drawdown": 0.0,
        "expectativa": 0.0,
        "retorno_promedio": 0.0,
        "sqn": 0.0,
        "estabilidad": 0.0,
        "racha_ganadora": 0,
        "racha_perdedora": 0,
        "porc_ganadoras": 0.0,
        "porc_perdedoras": 0.0,
        "trades_por_dia": 0.0,
        "pnl_neto_por_dia_operado": 0.0,
        "n_trades": 0,
        "total_trades": 0,
        "num_trades": 0,
        "n_trades_long": 0,
        "count_longs": 0,
        "num_longs": 0,
        "n_trades_short": 0,
        "count_shorts": 0,
        "num_shorts": 0,
        "riesgo_beneficio": float("nan"),
        "sharpe": 0.0,
        "sortino": 0.0,
        "profit_factor": float("nan"),
        "payoff_ratio": float("nan"),
        "calmar": 0.0,
        "saldo_actual": float(saldo_inicial),
        "saldo_min": float(saldo_inicial),
        "saldo_max": float(saldo_inicial),
        "saldo_mean": float(saldo_inicial),
        "max_ganancia": 0.0,
        "max_perdida": 0.0,
        "duration_mean_min": 0.0,
        "comisiones_total": 0.0,
        "saldo_sin_comisiones": float(saldo_inicial),
        "pnl_neto": 0.0,
        "net_pnl": 0.0,
    }


if NUMBA_METRICS_AVAILABLE:

    @njit(cache=True, fastmath=True)
    def _max_drawdown_numba(equity_curve: np.ndarray) -> Tuple[float, float]:
        n = len(equity_curve)
        if n < 2:
            return 0.0, 0.0

        max_dd_abs = 0.0
        max_dd_pct = 0.0
        peak = equity_curve[0]

        for i in range(1, n):
            val = equity_curve[i]
            if val > peak:
                peak = val

            dd_abs = peak - val
            if dd_abs > max_dd_abs:
                max_dd_abs = dd_abs

            if peak > 0:
                dd_pct = 100.0 * dd_abs / peak
                if dd_pct > max_dd_pct:
                    max_dd_pct = dd_pct

        return max_dd_abs, max_dd_pct


    @njit(cache=True, fastmath=True)
    def _compute_all_metrics_numba(
        pnl_neto: np.ndarray,
        pnl_pct: np.ndarray,
        saldo_despues: np.ndarray,
        saldo_antes: np.ndarray,
        equity_curve: np.ndarray,
        saldo_inicial: float,
    ) -> Tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        int,
        int,
        float,
        float,
        float,
        float,
        float,
        int,
        int,
    ]:
        n = len(pnl_neto)
        if n == 0:
            return (
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                np.nan,
                np.nan,
                0.0,
                0.0,
                0,
                0,
                saldo_inicial,
                saldo_inicial,
                saldo_inicial,
                0.0,
                0.0,
                0,
                0,
            )

        sum_pnl = 0.0
        sum_pnl_sq = 0.0
        sum_wins = 0.0
        sum_losses = 0.0
        n_wins = 0
        n_losses = 0
        max_ganancia = pnl_neto[0]
        max_perdida = pnl_neto[0]

        sum_returns = 0.0
        sum_returns_sq = 0.0
        sum_neg_returns_sq = 0.0
        n_neg_returns = 0

        saldo_min = saldo_despues[0]
        saldo_max = saldo_despues[0]
        saldo_sum = 0.0

        curr_win_streak = 0
        curr_loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0

        for i in range(n):
            pnl = pnl_neto[i]
            ret = pnl_pct[i] / 100.0
            saldo = saldo_despues[i]

            sum_pnl += pnl
            sum_pnl_sq += pnl * pnl

            if pnl > 0:
                sum_wins += pnl
                n_wins += 1
                curr_win_streak += 1
                curr_loss_streak = 0
                if curr_win_streak > max_win_streak:
                    max_win_streak = curr_win_streak
            else:
                sum_losses += abs(pnl)
                if pnl < 0:
                    n_losses += 1
                curr_loss_streak += 1
                curr_win_streak = 0
                if curr_loss_streak > max_loss_streak:
                    max_loss_streak = curr_loss_streak

            if pnl > max_ganancia:
                max_ganancia = pnl
            if pnl < max_perdida:
                max_perdida = pnl

            sum_returns += ret
            sum_returns_sq += ret * ret
            if ret < 0:
                sum_neg_returns_sq += ret * ret
                n_neg_returns += 1

            saldo_sum += saldo
            if saldo < saldo_min:
                saldo_min = saldo
            if saldo > saldo_max:
                saldo_max = saldo

        saldo_final = saldo_despues[n - 1]
        roi = 100.0 * (saldo_final - saldo_inicial) / saldo_inicial if saldo_inicial != 0 else 0.0
        winrate = 100.0 * n_wins / n

        mean_pnl = sum_pnl / n
        var_pnl = (sum_pnl_sq / n) - (mean_pnl * mean_pnl)
        if var_pnl < 0:
            var_pnl = 0.0
        std_pnl = np.sqrt(var_pnl * n / (n - 1)) if n > 1 else 0.0
        sqn = np.sqrt(float(n)) * (mean_pnl / std_pnl) if std_pnl > 0 else 0.0

        mean_ret = sum_returns / n
        var_ret = (sum_returns_sq / n) - (mean_ret * mean_ret)
        if var_ret < 0:
            var_ret = 0.0
        std_ret = np.sqrt(var_ret * n / (n - 1)) if n > 1 else 0.0
        sharpe = mean_ret / std_ret if std_ret > 0 else 0.0

        neg_var = sum_neg_returns_sq / n_neg_returns if n_neg_returns > 1 else 0.0
        neg_std = (
            np.sqrt(neg_var * n_neg_returns / (n_neg_returns - 1))
            if n_neg_returns > 1
            else 0.0
        )
        sortino = mean_ret / neg_std if neg_std > 0 else 0.0

        profit_factor = sum_wins / sum_losses if sum_losses > 0 else np.nan
        avg_win = sum_wins / n_wins if n_wins > 0 else 0.0
        avg_loss = sum_losses / n_losses if n_losses > 0 else 0.0
        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else np.nan

        p_win = n_wins / n
        expectativa = p_win * avg_win + (1.0 - p_win) * (-avg_loss)
        retorno_promedio = mean_pnl
        saldo_mean = saldo_sum / n

        _, max_dd_pct = _max_drawdown_numba(equity_curve)

        return (
            roi,
            winrate,
            max_dd_pct,
            sqn,
            sharpe,
            sortino,
            profit_factor,
            payoff_ratio,
            expectativa,
            retorno_promedio,
            max_win_streak,
            max_loss_streak,
            saldo_min,
            saldo_max,
            saldo_mean,
            max_ganancia,
            max_perdida,
            n_wins,
            n_losses,
        )


    def resumen_metricas_fast(
        pnl_neto: np.ndarray,
        pnl_pct: np.ndarray,
        saldo_despues: np.ndarray,
        saldo_antes: np.ndarray,
        equity_curve: np.ndarray,
        saldo_inicial: float,
        n_trades_long: int = 0,
        n_trades_short: int = 0,
    ) -> Dict[str, Any]:
        n = len(pnl_neto)
        if n == 0:
            return _empty_metrics_fast(saldo_inicial)

        pnl_neto = np.ascontiguousarray(pnl_neto, dtype=np.float64)
        pnl_pct = np.ascontiguousarray(pnl_pct, dtype=np.float64)
        saldo_despues = np.ascontiguousarray(saldo_despues, dtype=np.float64)
        saldo_antes = np.ascontiguousarray(saldo_antes, dtype=np.float64)
        equity_curve = np.ascontiguousarray(equity_curve, dtype=np.float64)

        (
            roi,
            winrate,
            max_dd_pct,
            sqn,
            sharpe,
            sortino,
            profit_factor,
            payoff_ratio,
            expectativa,
            retorno_promedio,
            max_win_streak,
            max_loss_streak,
            saldo_min,
            saldo_max,
            saldo_mean,
            max_ganancia,
            max_perdida,
            n_wins,
            n_losses,
        ) = _compute_all_metrics_numba(
            pnl_neto, pnl_pct, saldo_despues, saldo_antes, equity_curve, saldo_inicial
        )

        return {
            "roi": float(roi),
            "winrate": float(winrate),
            "drawdown": float(max_dd_pct),
            "expectativa": float(expectativa),
            "retorno_promedio": float(retorno_promedio),
            "sqn": float(sqn),
            "estabilidad": 0.0,
            "racha_ganadora": int(max_win_streak),
            "racha_perdedora": int(max_loss_streak),
            "porc_ganadoras": 100.0 * n_wins / n if n > 0 else 0.0,
            "porc_perdedoras": 100.0 * n_losses / n if n > 0 else 0.0,
            "trades_por_dia": 0.0,
            "pnl_neto_por_dia_operado": 0.0,
            "n_trades": int(n),
            "total_trades": int(n),
            "num_trades": int(n),
            "n_trades_long": int(n_trades_long),
            "count_longs": int(n_trades_long),
            "num_longs": int(n_trades_long),
            "n_trades_short": int(n_trades_short),
            "count_shorts": int(n_trades_short),
            "num_shorts": int(n_trades_short),
            "riesgo_beneficio": float(payoff_ratio),
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "profit_factor": float(profit_factor),
            "payoff_ratio": float(payoff_ratio),
            "calmar": 0.0,
            "saldo_actual": float(saldo_despues[-1]) if n > 0 else float(saldo_inicial),
            "saldo_min": float(saldo_min),
            "saldo_max": float(saldo_max),
            "saldo_mean": float(saldo_mean),
            "max_ganancia": float(max_ganancia),
            "max_perdida": float(max_perdida),
            "duration_mean_min": 0.0,
            "comisiones_total": 0.0,
            "saldo_sin_comisiones": float(saldo_inicial),
            "pnl_neto": float(np.sum(pnl_neto)),
            "net_pnl": float(np.sum(pnl_neto)),
        }


else:

    def resumen_metricas_fast(
        pnl_neto: np.ndarray,
        pnl_pct: np.ndarray,
        saldo_despues: np.ndarray,
        saldo_antes: np.ndarray,
        equity_curve: np.ndarray,
        saldo_inicial: float,
        n_trades_long: int = 0,
        n_trades_short: int = 0,
    ) -> Dict[str, Any]:
        return _empty_metrics_fast(saldo_inicial)


# Tipo unificado para trades
TradesDF = Union[pd.DataFrame, pl.DataFrame]


def _is_empty(trades: Optional[TradesDF]) -> bool:
    """Verifica si trades está vacío (compatible Polars/Pandas)."""
    if trades is None:
        return True
    if isinstance(trades, pl.DataFrame):
        return trades.is_empty()
    return trades.empty


def _to_numpy(trades: TradesDF, col: str) -> np.ndarray:
    """Extrae columna como numpy array (zero-copy cuando es posible)."""
    if isinstance(trades, pl.DataFrame):
        return trades[col].to_numpy()
    return trades[col].to_numpy(dtype=np.float64, copy=False)


def _get_column(trades: TradesDF, col: str, default: Any = None) -> Any:
    """Obtiene una columna de forma segura."""
    if isinstance(trades, pl.DataFrame):
        if col in trades.columns:
            return trades[col]
        return default
    if col in trades.columns:
        return trades[col]
    return default


def _empty(trades: Optional[TradesDF]) -> bool:
    return _is_empty(trades)


def roi_pct(trades: TradesDF, saldo_inicial: float) -> float:
    if _empty(trades) or saldo_inicial == 0:
        return 0.0
    if isinstance(trades, pl.DataFrame):
        saldo_final = float(trades["saldo_despues"][-1])
    else:
        saldo_final = float(trades["saldo_despues"].iloc[-1])
    return 100.0 * (saldo_final - saldo_inicial) / saldo_inicial


def winrate_pct(trades: TradesDF) -> float:
    if _empty(trades):
        return 0.0
    pnl = _to_numpy(trades, "pnl_neto")
    return 100.0 * float((pnl > 0).sum()) / float(len(pnl))


def max_drawdown(equity_curve: List[float]) -> Tuple[float, float]:
    """
    Max drawdown in absolute value and percent.

    Equity curve is expected to be a list of saldo values after each trade.
    """

    if not equity_curve or len(equity_curve) < 2:
        return 0.0, 0.0
    arr = np.asarray(equity_curve, dtype=np.float64)
    peaks = np.maximum.accumulate(arr)
    drawdowns = peaks - arr
    drawdowns_pct = np.where(peaks != 0, 100.0 * drawdowns / peaks, 0.0)
    return float(np.max(drawdowns)), float(np.max(drawdowns_pct))


def expectativa(trades: TradesDF) -> float:
    """Expected value per trade in $."""

    if _empty(trades):
        return 0.0
    pnl = _to_numpy(trades, "pnl_neto")
    p_win = float((pnl > 0).mean())
    avg_win = float(pnl[pnl > 0].mean()) if (pnl > 0).any() else 0.0
    avg_loss = float(pnl[pnl <= 0].mean()) if (pnl <= 0).any() else 0.0
    return p_win * avg_win + (1.0 - p_win) * avg_loss


def retorno_promedio(trades: TradesDF) -> float:
    """Mean net pnl per trade in $."""

    if _empty(trades):
        return 0.0
    pnl = _to_numpy(trades, "pnl_neto")
    return float(np.mean(pnl))


def sqn(trades: TradesDF) -> float:
    """System Quality Number (SQN).

    Fórmula:
        $SQN = \sqrt{N} \times (\bar{R} / \sigma_R)$

    Donde R es el resultado por trade. Aquí usamos `pnl_neto` (PnL neto por trade)
    porque ya incluye comisiones y es consistente con el resto de métricas.
    """

    if _empty(trades):
        return 0.0

    r = _to_numpy(trades, "pnl_neto")
    r = r[np.isfinite(r)]
    n = int(r.size)
    if n < 2:
        return 0.0

    mean = float(np.mean(r))
    std = float(np.std(r, ddof=1))
    if std == 0.0 or not np.isfinite(std):
        return 0.0

    val = float(np.sqrt(float(n)) * (mean / std))
    return val if np.isfinite(val) else 0.0


def estabilidad_equity(equity_curve: List[float]) -> float:
    """
    Simple smoothness measure (1 - std(delta)/mean(equity)).

    This is not a standard metric; it is kept because the project already uses it.
    """

    if not equity_curve or len(equity_curve) < 2:
        return 0.0
    arr = np.asarray(equity_curve, dtype=np.float64)
    cambios = np.diff(arr)
    mean_eq = float(np.mean(arr))
    return float(1.0 - (np.std(cambios) / mean_eq)) if mean_eq != 0 else 0.0


def racha_maxima(trades: TradesDF) -> Tuple[int, int]:
    """Max winning and losing streaks."""

    if _empty(trades):
        return 0, 0
    
    pnl = _to_numpy(trades, "pnl_neto")
    
    def max_streak_np(arr: np.ndarray) -> int:
        """Calcula racha máxima usando numpy puro."""
        if len(arr) == 0:
            return 0
        # Encontrar cambios
        changes = np.diff(arr, prepend=0)
        # Grupos de 1's consecutivos
        groups = np.cumsum(changes != 0)
        # Contar dentro de cada grupo
        counts = np.zeros_like(arr)
        for i, (g, v) in enumerate(zip(groups, arr)):
            if v:
                if i == 0 or groups[i-1] != g:
                    counts[i] = 1
                else:
                    counts[i] = counts[i-1] + 1
        return int(counts.max()) if len(counts) > 0 else 0
    
    gan = (pnl > 0).astype(int)
    per = (pnl < 0).astype(int)
    
    return max_streak_np(gan), max_streak_np(per)


def porcentaje_ganadoras_perdedoras(trades: TradesDF) -> Tuple[float, float]:
    if _empty(trades):
        return 0.0, 0.0
    pnl = _to_numpy(trades, "pnl_neto")
    n = float(len(pnl))
    n_win = float((pnl > 0).sum())
    n_loss = float((pnl < 0).sum())
    return 100.0 * n_win / n, 100.0 * n_loss / n


def _extract_times_polars(trades: TradesDF) -> Tuple[pl.Series, pl.Series]:
    """Extrae entry_time y exit_time como pl.Series datetime (Polars puro, sin pandas)."""
    if isinstance(trades, pl.DataFrame):
        entry_times = trades["entry_time"]
        exit_times = trades["exit_time"]
        
        # Asegurar tipo Datetime
        if entry_times.dtype != pl.Datetime:
            entry_times = entry_times.cast(pl.Datetime("us"))
        if exit_times.dtype != pl.Datetime:
            exit_times = exit_times.cast(pl.Datetime("us"))
    else:
        # Convertir pandas a polars si es necesario
        entry_times = pl.Series(trades["entry_time"].values).cast(pl.Datetime("us"))
        exit_times = pl.Series(trades["exit_time"].values).cast(pl.Datetime("us"))
    return entry_times, exit_times


def trades_por_dia(
    trades: TradesDF,
    *,
    period_start: Optional[Any] = None,
    period_end: Optional[Any] = None,
) -> float:
    """Trades por día (Polars puro, sin pandas)."""

    if _empty(trades):
        return 0.0

    entry_times, exit_times = _extract_times_polars(trades)

    # Combinar y filtrar nulls usando Polars
    all_times = pl.concat([entry_times.drop_nulls(), exit_times.drop_nulls()])
    if all_times.is_empty():
        return 0.0

    min_ts = all_times.min()
    max_ts = all_times.max()

    if period_start is None or period_end is None:
        start = min_ts
        end = max_ts
    else:
        # Convertir a datetime si es string
        start = period_start
        end = period_end

    # Calcular días usando Polars temporal
    if start is None or end is None:
        return 0.0
    
    # Extraer fecha (día) y calcular diferencia usando el DataFrame
    dates_df = pl.DataFrame({"ts": [min_ts, max_ts]})
    dates_result = dates_df.select(pl.col("ts").dt.date())
    start_date = dates_result["ts"][0]
    end_date = dates_result["ts"][1]
    days = (end_date - start_date).days + 1
    
    n_trades = trades.height if isinstance(trades, pl.DataFrame) else len(trades)
    return float(n_trades) / float(days) if days > 0 else 0.0


def _to_utc_polars(ts: Any) -> Any:
    """Convert timestamp to UTC-aware (Polars compatible)."""
    if ts is None:
        return None
    # Si ya es datetime de Polars, devolverlo tal cual
    if hasattr(ts, 'dt'):
        return ts
    # Si es string, convertir
    if isinstance(ts, str):
        return pl.lit(ts).str.to_datetime().dt.replace_time_zone("UTC")
    return ts


def pnl_neto_por_dia_operado(
    trades: TradesDF,
    *,
    period_start: Optional[Any] = None,
    period_end: Optional[Any] = None,
) -> float:
    """PnL neto por día operado (Polars puro, sin pandas)."""

    if _empty(trades):
        return 0.0

    entry_times, exit_times = _extract_times_polars(trades)
    all_times = pl.concat([entry_times.drop_nulls(), exit_times.drop_nulls()])
    if all_times.is_empty():
        return 0.0

    min_ts = all_times.min()
    max_ts = all_times.max()

    if period_start is None or period_end is None:
        start = min_ts
        end = max_ts
    else:
        start = _to_utc_polars(period_start)
        end_cfg = _to_utc_polars(period_end)
        end = min(end_cfg, max_ts) if end_cfg is not None else max_ts

    # Filtrar eventos en rango y contar días únicos
    events_df = pl.DataFrame({"ts": all_times})
    if start is not None and end is not None:
        events_df = events_df.filter(
            (pl.col("ts") >= start) & (pl.col("ts") <= end)
        )
    
    if events_df.is_empty():
        return 0.0

    # Contar días únicos usando Polars
    dias_operados = events_df.select(
        pl.col("ts").dt.date().n_unique()
    ).item()
    
    if dias_operados <= 0:
        return 0.0

    pnl_neto_total = float(_to_numpy(trades, "pnl_neto").sum())
    return pnl_neto_total / float(dias_operados)


def riesgo_beneficio(trades: TradesDF) -> float:
    """Average win / average loss (absolute)."""

    if _empty(trades):
        return float("nan")
    pnl = _to_numpy(trades, "pnl_neto")
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    return abs(avg_win / avg_loss) if avg_loss != 0 else float("nan")


def _returns_series(trades: TradesDF) -> np.ndarray:
    """
    Per-trade returns in decimals (e.g. +0.01 == +1%).

    Prefer `pnl_pct` if present; otherwise derive from pnl/stake.
    """
    cols = trades.columns if isinstance(trades, pl.DataFrame) else list(trades.columns)
    
    if "pnl_pct" in cols:
        r = _to_numpy(trades, "pnl_pct") / 100.0
    elif "stake" in cols:
        stake = _to_numpy(trades, "stake")
        pnl = _to_numpy(trades, "pnl_neto")
        r = np.where(stake != 0, pnl / stake, 0.0)
    else:
        # Fallback: usar saldo_usado como stake
        if "saldo_usado" in cols:
            stake = _to_numpy(trades, "saldo_usado")
            pnl = _to_numpy(trades, "pnl_neto")
            r = np.where(stake != 0, pnl / stake, 0.0)
        else:
            r = np.zeros(len(trades))
    r = r[np.isfinite(r)]
    return r


def sharpe(trades: TradesDF, *, annualize: bool = False) -> float:
    """
    Sharpe-like ratio using per-trade returns (NOT dollar pnl).
    """

    if _empty(trades):
        return 0.0
    r = _returns_series(trades)
    if r.size == 0:
        return 0.0
    mean = float(np.mean(r))
    std = float(np.std(r, ddof=1)) if r.size > 1 else 0.0
    if std == 0:
        return 0.0
    ratio = mean / std
    if annualize:
        tpd = trades_por_dia(trades)
        ratio *= float(np.sqrt(max(tpd * 365.25, 0.0)))
    return float(ratio)


def sortino(trades: TradesDF, *, annualize: bool = False) -> float:
    """Sortino-like ratio using downside deviation of per-trade returns."""

    if _empty(trades):
        return 0.0
    r = _returns_series(trades)
    if r.size == 0:
        return 0.0
    downside = r[r < 0]
    neg_std = float(np.std(downside, ddof=1)) if downside.size > 1 else 0.0
    if neg_std == 0:
        return 0.0
    ratio = float(np.mean(r) / neg_std)
    if annualize:
        tpd = trades_por_dia(trades)
        ratio *= float(np.sqrt(max(tpd * 365.25, 0.0)))
    return ratio


def profit_factor(trades: TradesDF) -> float:
    if _empty(trades):
        return float("nan")
    pnl = _to_numpy(trades, "pnl_neto")
    wins = float(pnl[pnl > 0].sum())
    losses = float(abs(pnl[pnl < 0].sum()))
    return wins / losses if losses != 0 else float("nan")


def payoff_ratio(trades: TradesDF) -> float:
    if _empty(trades):
        return float("nan")
    pnl = _to_numpy(trades, "pnl_neto")
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 0.0
    return avg_win / avg_loss if avg_loss != 0 else float("nan")


def calmar(trades: TradesDF, equity_curve: List[float]) -> float:
    """
    Calmar ratio (annualized return / max drawdown).
    """

    if _empty(trades) or not equity_curve:
        return 0.0

    entry_times, exit_times = _extract_times_polars(trades)
    
    # Calcular días usando Polars temporal
    start = entry_times.min()
    end = exit_times.max()
    if start is None or end is None:
        return 0.0
    
    # Calcular diferencia en días
    diff_df = pl.DataFrame({"start": [start], "end": [end]})
    diff_result = diff_df.select(
        ((pl.col("end") - pl.col("start")).dt.total_seconds() / 86400.0).alias("days")
    )
    days = float(diff_result["days"][0]) if not diff_result.is_empty() else 0.0
    years = days / 365.25 if days > 0 else 0.0

    initial = float(equity_curve[0])
    final = float(equity_curve[-1])
    if initial <= 0 or years <= 0:
        cagr = 0.0
    elif final <= 0:
        cagr = float("nan")
    else:
        ratio = final / initial
        if ratio <= 0:
            cagr = float("nan")
        else:
            try:
                if ratio > 1e6:
                    cagr = float("inf")
                elif years < 0.001:
                    cagr = 0.0
                else:
                    cagr_val = ratio ** (1.0 / years) - 1.0
                    if isinstance(cagr_val, complex):
                        cagr = float("nan")
                    else:
                        cagr = float(cagr_val)
                        if cagr > 1e10:
                            cagr = float("inf")
            except (OverflowError, ValueError):
                cagr = float("inf")

    _, max_dd_pct = max_drawdown(equity_curve)
    if max_dd_pct == 0 or np.isnan(cagr) or np.isinf(cagr):
        return float("nan")
    result = cagr / (max_dd_pct / 100.0)
    if isinstance(result, complex) or np.isinf(result):
        return float("nan")
    return float(result)


def resumen_metricas(
    trades: TradesDF,
    *,
    saldo_inicial: float,
    equity_curve: Optional[List[float]] = None,
    period_start: Optional[pd.Timestamp] = None,
    period_end: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    """
    Main metrics dictionary used by scoring and reporting.
    
    Acepta tanto Polars como Pandas DataFrame.
    """
    if _empty(trades):
        return _empty_metrics_dict(saldo_inicial)
    
    # =========================================================================
    # RUTA RÁPIDA: Usar Numba si está disponible
    # =========================================================================
    if USE_NUMBA_METRICS and NUMBA_METRICS_AVAILABLE:
        try:
            return _resumen_metricas_numba_wrapper(
                trades, saldo_inicial, equity_curve, period_start, period_end
            )
        except Exception:
            pass
    
    # =========================================================================
    # RUTA ESTÁNDAR: Cálculo Python tradicional
    # =========================================================================
    return _resumen_metricas_python(trades, saldo_inicial, equity_curve, period_start, period_end)


def _empty_metrics_dict(saldo_inicial: float) -> Dict[str, Any]:
    """Diccionario de métricas vacío."""
    return {
        "roi": 0.0,
        "winrate": 0.0,
        "drawdown": 0.0,
        "expectativa": 0.0,
        "retorno_promedio": 0.0,
        "sqn": 0.0,
        "estabilidad": 0.0,
        "racha_ganadora": 0,
        "racha_perdedora": 0,
        "porc_ganadoras": 0.0,
        "porc_perdedoras": 0.0,
        "trades_por_dia": 0.0,
        "n_trades": 0,
        "n_trades_long": 0,
        "n_trades_short": 0,
        "riesgo_beneficio": float("nan"),
        "sharpe": 0.0,
        "sortino": 0.0,
        "profit_factor": float("nan"),
        "payoff_ratio": float("nan"),
        "calmar": 0.0,
        "saldo_actual": float(saldo_inicial),
        "saldo_min": float(saldo_inicial),
        "saldo_max": float(saldo_inicial),
        "saldo_mean": float(saldo_inicial),
        "max_ganancia": 0.0,
        "max_perdida": 0.0,
        "duration_mean_min": 0.0,
        "comisiones_total": 0.0,
        "saldo_sin_comisiones": float(saldo_inicial),
    }


def _resumen_metricas_numba_wrapper(
    trades: TradesDF,
    saldo_inicial: float,
    equity_curve: Optional[List[float]],
    period_start: Optional[pd.Timestamp],
    period_end: Optional[pd.Timestamp],
) -> Dict[str, Any]:
    """Wrapper que prepara datos para la versión Numba y completa métricas faltantes."""
    
    # Extraer arrays numpy (compatible Polars/Pandas)
    pnl_neto = _to_numpy(trades, "pnl_neto")
    cols = trades.columns if isinstance(trades, pl.DataFrame) else list(trades.columns)
    pnl_pct = _to_numpy(trades, "pnl_pct") if "pnl_pct" in cols else np.zeros_like(pnl_neto)
    saldo_despues = _to_numpy(trades, "saldo_despues")
    saldo_antes = _to_numpy(trades, "saldo_antes") if "saldo_antes" in cols else np.zeros_like(saldo_despues)
    
    eq_arr = np.asarray(equity_curve if equity_curve else list(saldo_despues), dtype=np.float64)
    
    # Contar trades por tipo
    if isinstance(trades, pl.DataFrame):
        type_arr = trades["type"].to_list() if "type" in trades.columns else []
        type_upper = [str(t).upper() for t in type_arr]
        n_trades_long = sum(1 for t in type_upper if t == "LONG")
        n_trades_short = sum(1 for t in type_upper if t == "SHORT")
    else:
        type_col = trades.get("type", pd.Series(dtype=str))
        if len(type_col) > 0:
            type_upper = type_col.astype(str).str.upper()
            n_trades_long = int((type_upper == "LONG").sum())
            n_trades_short = int((type_upper == "SHORT").sum())
        else:
            n_trades_long = 0
            n_trades_short = 0
    
    # Llamar a Numba
    metrics = resumen_metricas_fast(
        pnl_neto, pnl_pct, saldo_despues, saldo_antes, eq_arr, saldo_inicial,
        n_trades_long, n_trades_short
    )
    
    # Completar métricas que requieren timestamps (Python)
    metrics["trades_por_dia"] = trades_por_dia(trades, period_start=period_start, period_end=period_end)
    metrics["pnl_neto_por_dia_operado"] = pnl_neto_por_dia_operado(trades, period_start=period_start, period_end=period_end)
    metrics["calmar"] = calmar(trades, list(eq_arr))
    
    # Duración y comisiones (compatible Polars/Pandas)
    if "duracion_min" in cols:
        metrics["duration_mean_min"] = float(np.mean(_to_numpy(trades, "duracion_min")))
    else:
        metrics["duration_mean_min"] = 0.0
        
    if "comision" in cols:
        metrics["comisiones_total"] = float(np.sum(_to_numpy(trades, "comision")))
    else:
        metrics["comisiones_total"] = 0.0
    
    # Estabilidad (simple, no crítica)
    if len(eq_arr) >= 2:
        cambios = np.diff(eq_arr)
        mean_eq = float(np.mean(eq_arr))
        metrics["estabilidad"] = float(1.0 - (np.std(cambios) / mean_eq)) if mean_eq != 0 else 0.0
    
    # Saldo sin comisiones
    if "pnl" in cols:
        pnl_bruto = _to_numpy(trades, "pnl")
        metrics["saldo_sin_comisiones"] = float(saldo_antes[0]) + float(pnl_bruto.sum())
    
    return metrics


def _resumen_metricas_python(
    trades: TradesDF,
    saldo_inicial: float,
    equity_curve: Optional[List[float]],
    period_start: Optional[pd.Timestamp],
    period_end: Optional[pd.Timestamp],
) -> Dict[str, Any]:
    """Versión Python de resumen_metricas (compatible Polars/Pandas)."""
    
    if _empty(trades):
        return _empty_metrics_dict(saldo_inicial)

    # Extraer arrays
    saldo_despues = _to_numpy(trades, "saldo_despues")
    pnl_neto = _to_numpy(trades, "pnl_neto")
    cols = trades.columns if isinstance(trades, pl.DataFrame) else list(trades.columns)
    
    equity_curve = list(saldo_despues) if equity_curve is None else equity_curve
    _, max_dd_pct = max_drawdown(equity_curve)
    racha_g, racha_p = racha_maxima(trades)
    porc_gan, porc_perd = porcentaje_ganadoras_perdedoras(trades)

    # "Saldo sin comisiones"
    saldo_antes = _to_numpy(trades, "saldo_antes") if "saldo_antes" in cols else np.array([saldo_inicial])
    pnl_bruto = _to_numpy(trades, "pnl") if "pnl" in cols else pnl_neto
    saldo_sin_comisiones = float(saldo_antes[0]) + float(pnl_bruto.sum())

    # Contar trades long y short
    if isinstance(trades, pl.DataFrame):
        type_arr = trades["type"].to_list() if "type" in trades.columns else []
        type_upper = [str(t).upper() for t in type_arr]
        n_trades_long = sum(1 for t in type_upper if t == "LONG")
        n_trades_short = sum(1 for t in type_upper if t == "SHORT")
    else:
        type_col = trades.get("type", pd.Series(dtype=str))
        if len(type_col) > 0:
            type_upper = type_col.astype(str).str.upper()
            n_trades_long = int((type_upper == "LONG").sum())
            n_trades_short = int((type_upper == "SHORT").sum())
        else:
            n_trades_long = 0
            n_trades_short = 0

    # Métricas de saldo
    saldo_actual = float(saldo_despues[-1])
    saldo_min = float(np.min(saldo_despues))
    saldo_max = float(np.max(saldo_despues))
    saldo_mean = float(np.mean(saldo_despues))
    
    # Duración y comisiones
    duracion_min = _to_numpy(trades, "duracion_min") if "duracion_min" in cols else np.array([0.0])
    comision = _to_numpy(trades, "comision") if "comision" in cols else np.array([0.0])

    return {
        "roi": roi_pct(trades, saldo_inicial),
        "winrate": winrate_pct(trades),
        "drawdown": max_dd_pct,
        "expectativa": expectativa(trades),
        "retorno_promedio": retorno_promedio(trades),
        "sqn": sqn(trades),
        "estabilidad": estabilidad_equity(equity_curve),
        "racha_ganadora": racha_g,
        "racha_perdedora": racha_p,
        "porc_ganadoras": porc_gan,
        "porc_perdedoras": porc_perd,
        "trades_por_dia": trades_por_dia(trades, period_start=period_start, period_end=period_end),
        "pnl_neto_por_dia_operado": pnl_neto_por_dia_operado(
            trades, period_start=period_start, period_end=period_end
        ),
        # Trade counts
        "n_trades": int(len(trades)),
        "total_trades": int(len(trades)),
        "num_trades": int(len(trades)),
        "n_trades_long": n_trades_long,
        "count_longs": n_trades_long,
        "num_longs": n_trades_long,
        "n_trades_short": n_trades_short,
        "count_shorts": n_trades_short,
        "num_shorts": n_trades_short,
        "riesgo_beneficio": riesgo_beneficio(trades),
        "sharpe": sharpe(trades, annualize=False),
        "sortino": sortino(trades, annualize=False),
        "profit_factor": profit_factor(trades),
        "payoff_ratio": payoff_ratio(trades),
        "calmar": calmar(trades, equity_curve),
        "saldo_actual": saldo_actual,
        "saldo_min": saldo_min,
        "saldo_max": saldo_max,
        "saldo_mean": saldo_mean,
        "max_ganancia": float(np.max(pnl_neto)),
        "max_perdida": float(np.min(pnl_neto)),
        "duration_mean_min": float(np.mean(duracion_min)),
        "comisiones_total": float(np.sum(comision)),
        "saldo_sin_comisiones": saldo_sin_comisiones,
        # PnL aliases
        "pnl_neto": float(np.sum(pnl_neto)),
        "net_pnl": float(np.sum(pnl_neto)),
    }
