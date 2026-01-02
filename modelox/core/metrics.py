from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _empty(trades: Optional[pd.DataFrame]) -> bool:
    return trades is None or trades.empty


def roi_pct(trades: pd.DataFrame, saldo_inicial: float) -> float:
    if _empty(trades) or saldo_inicial == 0:
        return 0.0
    return 100.0 * (float(trades["saldo_despues"].iloc[-1]) - saldo_inicial) / saldo_inicial


def winrate_pct(trades: pd.DataFrame) -> float:
    if _empty(trades):
        return 0.0
    return 100.0 * float((trades["pnl_neto"] > 0).sum()) / float(len(trades))


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


def expectativa(trades: pd.DataFrame) -> float:
    """Expected value per trade in $."""

    if _empty(trades):
        return 0.0
    pnl = trades["pnl_neto"].to_numpy(dtype=np.float64, copy=False)
    p_win = float((pnl > 0).mean())
    avg_win = float(pnl[pnl > 0].mean()) if (pnl > 0).any() else 0.0
    avg_loss = float(pnl[pnl <= 0].mean()) if (pnl <= 0).any() else 0.0
    return p_win * avg_win + (1.0 - p_win) * avg_loss


def retorno_promedio(trades: pd.DataFrame) -> float:
    """Mean net pnl per trade in $."""

    if _empty(trades):
        return 0.0
    return float(trades["pnl_neto"].mean())


def sqn(trades: pd.DataFrame) -> float:
    """System Quality Number (SQN).

    Fórmula:
        $SQN = \sqrt{N} \times (\bar{R} / \sigma_R)$

    Donde R es el resultado por trade. Aquí usamos `pnl_neto` (PnL neto por trade)
    porque ya incluye comisiones y es consistente con el resto de métricas.
    """

    if _empty(trades):
        return 0.0

    r = trades["pnl_neto"].to_numpy(dtype=np.float64, copy=False)
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


def racha_maxima(trades: pd.DataFrame) -> Tuple[int, int]:
    """Max winning and losing streaks."""

    if _empty(trades):
        return 0, 0
    gan = (trades["pnl_neto"] > 0).astype(int)
    per = (trades["pnl_neto"] < 0).astype(int)

    def max_streak(arr: pd.Series) -> int:
        # Runs of 1's; convert change points into groups.
        return int((arr.groupby((arr != arr.shift()).cumsum()).cumsum() * arr).max())

    return max_streak(gan), max_streak(per)


def porcentaje_ganadoras_perdedoras(trades: pd.DataFrame) -> Tuple[float, float]:
    if _empty(trades):
        return 0.0, 0.0
    n = float(len(trades))
    n_win = float((trades["pnl_neto"] > 0).sum())
    n_loss = float((trades["pnl_neto"] < 0).sum())
    return 100.0 * n_win / n, 100.0 * n_loss / n


def trades_por_dia(
    trades: pd.DataFrame,
    *,
    period_start: Optional[pd.Timestamp] = None,
    period_end: Optional[pd.Timestamp] = None,
) -> float:
    """Trades por día.

    Regla solicitada:
    - Inicio: el día de inicio configurado (o, en práctica, el inicio del dataset ya filtrado).
    - Fin: el día de fin configurado *o* el día en el que el sistema deja de operar
      (último trade ejecutado), lo que ocurra primero.

    Esto evita dividir por días donde ya no hay operativa tras llegar al
    `saldo_minimo_operativo` (early stop).
    """

    if _empty(trades):
        return 0.0

    entry_times = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
    exit_times = pd.to_datetime(trades["exit_time"], utc=True, errors="coerce")

    # Último instante con operativa (entry o exit). Si no hay tiempos válidos, 0.
    last_event = pd.concat([entry_times.dropna(), exit_times.dropna()], ignore_index=True)
    if last_event.empty:
        return 0.0
    last_trade_ts = pd.Timestamp(last_event.max()).tz_convert("UTC")

    # Si no se pasa periodo explícito, usar la ventana real de trades.
    if period_start is None or period_end is None:
        start = pd.Timestamp(last_event.min()).tz_convert("UTC")
        end = pd.Timestamp(last_event.max()).tz_convert("UTC")
    else:
        start = pd.Timestamp(period_start).tz_convert("UTC")
        end_cfg = pd.Timestamp(period_end).tz_convert("UTC")
        end = min(end_cfg, last_trade_ts)

    start_day = start.normalize()
    end_day = end.normalize()
    days = int((end_day - start_day).days) + 1
    return float(len(trades)) / float(days) if days > 0 else 0.0


def _to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    """Convert timestamp-like to tz-aware UTC Timestamp."""

    t = pd.Timestamp(ts)
    if t.tz is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def pnl_neto_por_dia_operado(
    trades: pd.DataFrame,
    *,
    period_start: Optional[pd.Timestamp] = None,
    period_end: Optional[pd.Timestamp] = None,
) -> float:
    """PnL neto por día operado.

    Definición solicitada: PnL neto dividido entre los *días en los que se opera*
    (días que tienen al menos un evento de trade: entry o exit), dentro de la
    ventana del backtest (respetando `period_start/period_end`) y con corte en el
    último trade real (early stop).
    """

    if _empty(trades):
        return 0.0

    entry_times = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
    exit_times = pd.to_datetime(trades["exit_time"], utc=True, errors="coerce")
    events = pd.concat([entry_times.dropna(), exit_times.dropna()], ignore_index=True)
    if events.empty:
        return 0.0

    last_trade_ts = _to_utc(events.max())

    if period_start is None or period_end is None:
        start = _to_utc(events.min())
        end = _to_utc(events.max())
    else:
        start = _to_utc(period_start)
        end_cfg = _to_utc(period_end)
        end = min(end_cfg, last_trade_ts)

    # Filtrar eventos dentro de la ventana efectiva.
    events_in = events[(events >= start) & (events <= end)]
    if events_in.empty:
        return 0.0

    dias_operados = int(pd.Series(events_in).dt.normalize().nunique())
    if dias_operados <= 0:
        return 0.0

    pnl_neto_total = float(trades["pnl_neto"].sum())
    return pnl_neto_total / float(dias_operados)


def riesgo_beneficio(trades: pd.DataFrame) -> float:
    """Average win / average loss (absolute)."""

    if _empty(trades):
        return float("nan")
    wins = trades.loc[trades["pnl_neto"] > 0, "pnl_neto"]
    losses = trades.loc[trades["pnl_neto"] < 0, "pnl_neto"]
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    return abs(avg_win / avg_loss) if avg_loss != 0 else float("nan")


def _returns_series(trades: pd.DataFrame) -> np.ndarray:
    """
    Per-trade returns in decimals (e.g. +0.01 == +1%).

    Prefer `pnl_pct` if present; otherwise derive from pnl/stake.
    """

    if "pnl_pct" in trades.columns:
        r = trades["pnl_pct"].to_numpy(dtype=np.float64, copy=False) / 100.0
    else:
        stake = trades["stake"].to_numpy(dtype=np.float64, copy=False)
        pnl = trades["pnl_neto"].to_numpy(dtype=np.float64, copy=False)
        r = np.where(stake != 0, pnl / stake, 0.0)
    r = r[np.isfinite(r)]
    return r


def sharpe(trades: pd.DataFrame, *, annualize: bool = False) -> float:
    """
    Sharpe-like ratio using per-trade returns (NOT dollar pnl).

    If `annualize=True`, we scale by sqrt(trades_per_year) using observed trades/day.
    This is an approximation, but it's far more comparable than using $ PnL.
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


def sortino(trades: pd.DataFrame, *, annualize: bool = False) -> float:
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


def profit_factor(trades: pd.DataFrame) -> float:
    if _empty(trades):
        return float("nan")
    wins = float(trades.loc[trades["pnl_neto"] > 0, "pnl_neto"].sum())
    losses = float(abs(trades.loc[trades["pnl_neto"] < 0, "pnl_neto"].sum()))
    return wins / losses if losses != 0 else float("nan")


def payoff_ratio(trades: pd.DataFrame) -> float:
    if _empty(trades):
        return float("nan")
    wins = trades.loc[trades["pnl_neto"] > 0, "pnl_neto"]
    losses = trades.loc[trades["pnl_neto"] < 0, "pnl_neto"]
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(abs(losses.mean())) if not losses.empty else 0.0
    return avg_win / avg_loss if avg_loss != 0 else float("nan")


def calmar(trades: pd.DataFrame, equity_curve: List[float]) -> float:
    """
    Calmar ratio (annualized return / max drawdown).

    We annualize using the actual time span of the trade series.
    """

    if _empty(trades) or not equity_curve:
        return 0.0

    start = pd.to_datetime(trades["entry_time"], utc=True).min()
    end = pd.to_datetime(trades["exit_time"], utc=True).max()
    days = float((end - start).total_seconds() / 86400.0) if pd.notna(start) and pd.notna(end) else 0.0
    years = days / 365.25 if days > 0 else 0.0

    initial = float(equity_curve[0])
    final = float(equity_curve[-1])
    if initial <= 0 or years <= 0:
        cagr = 0.0
    elif final <= 0:
        # Si el saldo final es <= 0, el CAGR no tiene sentido matemático (sería complejo).
        # Retornamos NaN para indicar que no se puede calcular.
        cagr = float("nan")
    else:
        ratio = final / initial
        # Proteger contra números complejos: si ratio es negativo, retornar NaN.
        if ratio <= 0:
            cagr = float("nan")
        else:
            try:
                # Protect against overflow when ratio is huge and years is small
                # Cap ratio to prevent overflow (e.g., 1000x gain = 100000%)
                if ratio > 1e6:  # More than 1 million x gain
                    cagr = float("inf")  # Just return infinity
                elif years < 0.001:  # Less than ~8 hours
                    cagr = 0.0  # Too short to calculate meaningful CAGR
                else:
                    cagr_val = ratio ** (1.0 / years) - 1.0
                    # Verificar que no sea complejo antes de convertir a float.
                    if isinstance(cagr_val, complex):
                        cagr = float("nan")
                    else:
                        cagr = float(cagr_val)
                        # Cap extreme CAGR values
                        if cagr > 1e10:
                            cagr = float("inf")
            except (OverflowError, ValueError):
                cagr = float("inf")  # Overflow means extremely high CAGR

    _, max_dd_pct = max_drawdown(equity_curve)
    if max_dd_pct == 0 or pd.isna(cagr) or np.isinf(cagr):
        return float("nan")
    result = cagr / (max_dd_pct / 100.0)
    # Proteger contra números complejos en el resultado final.
    if isinstance(result, complex) or np.isinf(result):
        return float("nan")
    return float(result)


def resumen_metricas(
    trades: pd.DataFrame,
    *,
    saldo_inicial: float,
    equity_curve: Optional[List[float]] = None,
    period_start: Optional[pd.Timestamp] = None,
    period_end: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    """
    Main metrics dictionary used by scoring and reporting.

    Important: risk metrics (sharpe/sortino) use per-trade returns for comparability.
    """

    if _empty(trades):
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

    equity_curve = list(trades["saldo_despues"]) if equity_curve is None else equity_curve
    _, max_dd_pct = max_drawdown(equity_curve)
    racha_g, racha_p = racha_maxima(trades)
    porc_gan, porc_perd = porcentaje_ganadoras_perdedoras(trades)

    # "Saldo sin comisiones" = initial + gross pnl (ignoring commissions).
    saldo_sin_comisiones = float(trades["saldo_antes"].iloc[0]) + float(trades["pnl"].sum())

    # Contar trades long y short
    type_col = trades.get("type", pd.Series(dtype=str))
    if len(type_col) > 0:
        type_upper = type_col.astype(str).str.upper()
        n_trades_long = int((type_upper == "LONG").sum())
        n_trades_short = int((type_upper == "SHORT").sum())
    else:
        n_trades_long = 0
        n_trades_short = 0

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
        # Trade counts - multiple keys for compatibility
        "n_trades": int(len(trades)),
        "total_trades": int(len(trades)),  # Alias for rich reporter
        "num_trades": int(len(trades)),    # Alias for plot
        "n_trades_long": n_trades_long,
        "count_longs": n_trades_long,      # Alias for rich reporter
        "num_longs": n_trades_long,        # Alias
        "n_trades_short": n_trades_short,
        "count_shorts": n_trades_short,    # Alias for rich reporter  
        "num_shorts": n_trades_short,      # Alias
        "riesgo_beneficio": riesgo_beneficio(trades),
        "sharpe": sharpe(trades, annualize=False),
        "sortino": sortino(trades, annualize=False),
        "profit_factor": profit_factor(trades),
        "payoff_ratio": payoff_ratio(trades),
        "calmar": calmar(trades, equity_curve),
        "saldo_actual": float(trades["saldo_despues"].iloc[-1]),
        "saldo_min": float(trades["saldo_despues"].min()),
        "saldo_max": float(trades["saldo_despues"].max()),
        "saldo_mean": float(trades["saldo_despues"].mean()),
        "max_ganancia": float(trades["pnl_neto"].max()),
        "max_perdida": float(trades["pnl_neto"].min()),
        "duration_mean_min": float(trades["duracion_min"].mean()) if "duracion_min" in trades.columns else 0.0,
        "comisiones_total": float(trades["comision"].sum()) if "comision" in trades.columns else 0.0,
        "saldo_sin_comisiones": saldo_sin_comisiones,
        # PnL aliases for reporters
        "pnl_neto": float(trades["pnl_neto"].sum()),
        "net_pnl": float(trades["pnl_neto"].sum()),
    }


