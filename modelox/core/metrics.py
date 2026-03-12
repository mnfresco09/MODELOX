"""
================================================================================
MODELOX/CORE/METRICS.PY — SISTEMA CENTRALIZADO DE MÉTRICAS
================================================================================

PROPÓSITO:
    FUENTE ÚNICA DE VERDAD para el cálculo de TODAS las métricas de rendimiento.
    Ningún otro módulo debe calcular métricas directamente.

MÉTRICAS CANÓNICAS (6 métricas oficiales):
    1. PROFIT FACTOR   PF  = Σ ganancias / |Σ pérdidas|
    2. WIN RATE        WR  = nº trades ganadores / nº total de trades
    3. EXPECTANCY      E   = (WR × ganancia_media) − ((1−WR) × pérdida_media_abs)
    4. MAX DRAWDOWN    DD  = máx((peak − equity) / peak) × 100
    5. ROI             ROI = (saldo_final − saldo_inicial) / saldo_inicial × 100
    6. SHARPE RATIO    SR  = media(retornos) / desv_std(retornos)  [per-trade]

REGLAS:
    NUNCA calcular métricas fuera de este archivo.
    SIEMPRE importar las funciones de aquí.

USO:
    from modelox.core.metrics import resumen_metricas
    metrics = resumen_metricas(trades_df, saldo_inicial=1000, equity_curve=eq)

================================================================================
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl


# =============================================================================
# 1. CONFIGURACIÓN DE MÉTRICAS NUMBA
# =============================================================================
# Si Numba está disponible, las métricas se calculan con kernel JIT (10-50x faster).
# Si no, se usa la ruta Python pura como fallback.

USE_NUMBA_METRICS = True

try:
    from numba import njit
    NUMBA_METRICS_AVAILABLE = True
except Exception:
    NUMBA_METRICS_AVAILABLE = False
    USE_NUMBA_METRICS = False


# =============================================================================
# 2. DICCIONARIO DE MÉTRICAS VACÍAS (FAST)
# =============================================================================

def _empty_metrics_fast(saldo_inicial: float) -> Dict[str, Any]:
    """DICCIONARIO DE MÉTRICAS VACÍAS PARA TRIALS SIN TRADES (VERSIÓN RÁPIDA)."""
    return {
        # ── 6 métricas canónicas ──────────────────────────────────────────
        "roi": 0.0,
        "winrate": 0.0,
        "drawdown": 0.0,
        "expectativa": 0.0,
        "profit_factor": float("nan"),
        "sharpe": 0.0,
        # ── operacional ──────────────────────────────────────────────────
        "trades_por_dia": 0.0,
        "n_trades": 0,
        "total_trades": 0,
        "num_trades": 0,
        "n_trades_long": 0,
        "count_longs": 0,
        "num_longs": 0,
        "n_trades_short": 0,
        "count_shorts": 0,
        "num_shorts": 0,
        "saldo_actual": float(saldo_inicial),
        "saldo_min": float(saldo_inicial),
        "saldo_max": float(saldo_inicial),
        "saldo_mean": float(saldo_inicial),
        "duration_mean_min": 0.0,
        "comisiones_total": 0.0,
        "saldo_sin_comisiones": 0.0,
        "pnl_neto": 0.0,
        "net_pnl": 0.0,
    }


# =============================================================================
# 3. KERNEL NUMBA — CÁLCULO VECTORIZADO DE MÉTRICAS
# =============================================================================

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

        if n_wins == 0:
            sqn = -10.0
        elif n_wins == n:
            sqn = 10.0
        else:
            sqn = np.sqrt(float(n)) * (mean_pnl / std_pnl) if std_pnl > 1e-12 else 0.0
            
            # Clamp SQN to reasonable range
            if sqn > 20.0:
                sqn = 20.0
            elif sqn < -20.0:
                sqn = -20.0

        mean_ret = sum_returns / n
        var_ret = (sum_returns_sq / n) - (mean_ret * mean_ret)
        if var_ret < 0:
            var_ret = 0.0
        std_ret = np.sqrt(var_ret * n / (n - 1)) if n > 1 else 0.0
        
        if n_wins == 0:
            sharpe = -10.0
        elif n_wins == n:
            sharpe = 10.0
        else:
            sharpe = mean_ret / std_ret if std_ret > 1e-8 else 0.0
            if sharpe > 20.0:
                sharpe = 20.0
            elif sharpe < -20.0:
                sharpe = -20.0

        # Sortino: usa downside deviation (desviación de retornos negativos respecto a 0)
        # La fórmula correcta es: sqrt(mean(min(r, 0)^2)) - esto da la "semi-desviación"
        # Pero aquí usamos solo los retornos negativos: sqrt(sum(r_neg^2) / n_total)
        # para ser consistentes con la práctica común en trading
        downside_var = sum_neg_returns_sq / n if n > 0 else 0.0
        downside_std = np.sqrt(downside_var) if downside_var > 0 else 0.0
        
        if n_wins == 0:
            sortino = -10.0
        elif n_wins == n:
            sortino = 10.0
        else:
            sortino = mean_ret / downside_std if downside_std > 1e-8 else 0.0
            if sortino > 20.0:
                sortino = 20.0
            elif sortino < -20.0:
                sortino = -20.0

        profit_factor = sum_wins / sum_losses if sum_losses > 0 else np.nan
        avg_win = sum_wins / n_wins if n_wins > 0 else 0.0
        avg_loss_abs = sum_losses / n_losses if n_losses > 0 else 0.0

        # EXPECTANCY: E = WR × ganancia_media − (1−WR) × pérdida_media_abs
        wr_float = float(n_wins) / float(n)
        expectativa = wr_float * avg_win - (1.0 - wr_float) * avg_loss_abs

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


    # =================================================================
    # 4. WRAPPER NUMBA — RESUMEN_METRICAS_FAST
    # =================================================================

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
        """CALCULA MÉTRICAS USANDO KERNEL NUMBA Y DEVUELVE DICCIONARIO COMPLETO."""
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
            _sqn,           # ignorado
            sharpe,
            _sortino,       # ignorado
            profit_factor,
            _payoff,        # ignorado
            expectativa,
            _ret_prom,      # ignorado
            _win_streak,    # ignorado
            _loss_streak,   # ignorado
            saldo_min,
            saldo_max,
            saldo_mean,
            _max_gan,       # ignorado
            _max_perd,      # ignorado
            n_wins,
            n_losses,
        ) = _compute_all_metrics_numba(
            pnl_neto, pnl_pct, saldo_despues, saldo_antes, equity_curve, saldo_inicial
        )

        return {
            # ── 6 métricas canónicas ─────────────────────────────────────
            "roi": float(roi),
            "winrate": float(winrate),
            "drawdown": float(max_dd_pct),
            "expectativa": float(expectativa),
            "profit_factor": float(profit_factor),
            "sharpe": float(sharpe),   # se recalcula en wrapper con _returns_series
            # ── operacional ─────────────────────────────────────────────
            "trades_por_dia": 0.0,     # completado en wrapper
            "n_trades": int(n),
            "total_trades": int(n),
            "num_trades": int(n),
            "n_trades_long": int(n_trades_long),
            "count_longs": int(n_trades_long),
            "num_longs": int(n_trades_long),
            "n_trades_short": int(n_trades_short),
            "count_shorts": int(n_trades_short),
            "num_shorts": int(n_trades_short),
            "saldo_actual": float(saldo_despues[-1]) if n > 0 else float(saldo_inicial),
            "saldo_min": float(saldo_min),
            "saldo_max": float(saldo_max),
            "saldo_mean": float(saldo_mean),
            "duration_mean_min": 0.0,
            "comisiones_total": 0.0,
            "saldo_sin_comisiones": float(np.sum(pnl_neto)),
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


# =============================================================================
# 5. UTILIDADES DE DATAFRAME (COMPATIBILIDAD POLARS / PANDAS)
# =============================================================================

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


# =============================================================================
# 6. MÉTRICAS INDIVIDUALES
# =============================================================================

def roi_pct(trades: TradesDF, saldo_inicial: float) -> float:
    """ROI EN PORCENTAJE: (saldo_final - saldo_inicial) / saldo_inicial * 100."""
    if _empty(trades) or saldo_inicial == 0:
        return 0.0
    if isinstance(trades, pl.DataFrame):
        saldo_final = float(trades["saldo_despues"][-1])
    else:
        saldo_final = float(trades["saldo_despues"].iloc[-1])
    return 100.0 * (saldo_final - saldo_inicial) / saldo_inicial


def winrate_pct(trades: TradesDF) -> float:
    """TASA DE ACIERTOS EN PORCENTAJE: trades ganadores / total * 100."""
    if _empty(trades):
        return 0.0
    pnl = _to_numpy(trades, "pnl_neto")
    return 100.0 * float((pnl > 0).sum()) / float(len(pnl))


def max_drawdown(equity_curve: List[float]) -> Tuple[float, float]:
    """
    MAX DRAWDOWN EN VALOR ABSOLUTO Y PORCENTAJE.

    El equity curve es una lista de valores de saldo después de cada trade.
    Retorna (max_dd_abs, max_dd_pct).
    """

    if not equity_curve or len(equity_curve) < 2:
        return 0.0, 0.0
    arr = np.asarray(equity_curve, dtype=np.float64)
    peaks = np.maximum.accumulate(arr)
    drawdowns = peaks - arr
    drawdowns_pct = np.where(peaks != 0, 100.0 * drawdowns / peaks, 0.0)
    return float(np.max(drawdowns)), float(np.max(drawdowns_pct))


def expectativa(trades: TradesDF) -> float:
    """EXPECTANCY: E = WR × ganancia_media − (1−WR) × pérdida_media_abs.

    Fórmula canónica: cuánto espera ganar o perder por trade en promedio.
    """
    if _empty(trades):
        return 0.0
    pnl = _to_numpy(trades, "pnl_neto")
    n = len(pnl)
    if n == 0:
        return 0.0
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    wr = len(wins) / n
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss_abs = float(abs(losses.mean())) if len(losses) > 0 else 0.0
    return float(wr * avg_win - (1.0 - wr) * avg_loss_abs)


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


def _estimate_duration_mean_min(trades: TradesDF) -> float:
    """Estima duración media en minutos usando timestamp de entrada/salida."""
    if _empty(trades):
        return 0.0

    cols = trades.columns if isinstance(trades, pl.DataFrame) else list(trades.columns)

    # Prioridad 1: columna precomputada
    if "duracion_min" in cols:
        vals = _to_numpy(trades, "duracion_min")
        vals = vals[np.isfinite(vals)]
        # Evitar 0MIN por granularidad de timestamp: mínimo 1 minuto por trade cerrado.
        vals = np.where(vals < 1.0, 1.0, vals)
        return float(np.mean(vals)) if vals.size > 0 else 0.0

    # Prioridad 2: derivar de entry_time/exit_time
    if "entry_time" in cols and "exit_time" in cols:
        if isinstance(trades, pl.DataFrame):
            try:
                d = trades.select(
                    ((pl.col("exit_time") - pl.col("entry_time")).dt.total_seconds() / 60.0)
                    .alias("duracion_min")
                )["duracion_min"].to_numpy()
                d = d[np.isfinite(d)]
                d = d[d >= 0]
                d = np.where(d < 1.0, 1.0, d)
                return float(np.mean(d)) if d.size > 0 else 0.0
            except Exception:
                return 0.0
        else:
            try:
                entry = pd.to_datetime(trades["entry_time"], errors="coerce")
                exit_ = pd.to_datetime(trades["exit_time"], errors="coerce")
                d = (exit_ - entry).dt.total_seconds().to_numpy(dtype=np.float64) / 60.0
                d = d[np.isfinite(d)]
                d = d[d >= 0]
                d = np.where(d < 1.0, 1.0, d)
                return float(np.mean(d)) if d.size > 0 else 0.0
            except Exception:
                return 0.0

    return 0.0


# =============================================================================
# 7. MÉTRICAS DE ACTIVIDAD
# =============================================================================

def trades_por_dia(
    trades: TradesDF,
    *,
    period_start: Optional[Any] = None,
    period_end: Optional[Any] = None,
) -> float:
    """Trades por día (Polars puro, sin pandas).
    
    Usa el rango completo del periodo (period_start → period_end) para calcular
    los días, NO solo el rango entre el primer y último trade.
    Esto evita inflar trades/day cuando hay pocos trades concentrados.
    """

    if _empty(trades):
        return 0.0

    entry_times, exit_times = _extract_times_polars(trades)

    # Combinar y filtrar nulls usando Polars
    all_times = pl.concat([entry_times.drop_nulls(), exit_times.drop_nulls()])
    if all_times.is_empty():
        return 0.0

    min_ts = all_times.min()
    max_ts = all_times.max()

    # Determinar rango para calcular días:
    # Si se pasan period_start/period_end (rango del trial/backtest), usarlos.
    # Si no, usar el rango de los trades como fallback.
    if period_start is not None and period_end is not None:
        # Convertir a date para calcular días
        import datetime as _dt
        if isinstance(period_start, str):
            start_date = _dt.date.fromisoformat(period_start[:10])
        elif hasattr(period_start, 'date'):
            start_date = period_start.date() if callable(getattr(period_start, 'date')) else period_start.date
        else:
            start_date = _dt.date.fromisoformat(str(period_start)[:10])
        
        if isinstance(period_end, str):
            end_date = _dt.date.fromisoformat(period_end[:10])
        elif hasattr(period_end, 'date'):
            end_date = period_end.date() if callable(getattr(period_end, 'date')) else period_end.date
        else:
            end_date = _dt.date.fromisoformat(str(period_end)[:10])
        
        days = (end_date - start_date).days + 1
    else:
        # Fallback: rango entre primer y último trade
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


# =============================================================================
# 8. MÉTRICAS DE CALIDAD
# =============================================================================

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


def sharpe(trades: TradesDF, *, annualize: bool = False, timeframe: Optional[str] = None) -> float:
    """SHARPE RATIO PER-TRADE: media(retornos) / desv_std(retornos).

    Mide la calidad promedio de cada trade individual.
    Fórmula: Sharpe = media(retornos_por_trade) / std(retornos_por_trade)
    """
    if _empty(trades):
        return 0.0
    r = _returns_series(trades)
    n_trades = r.size
    if n_trades == 0:
        return 0.0
    mean_r = float(np.mean(r))
    std_r = float(np.std(r, ddof=1)) if n_trades > 1 else 0.0
    if std_r < 1e-8:
        return 0.0
    ratio = mean_r / std_r
    return float(max(-20.0, min(20.0, ratio)))


def profit_factor(trades: TradesDF) -> float:
    """PROFIT FACTOR: Σ ganancias / |Σ pérdidas|."""
    if _empty(trades):
        return float("nan")
    pnl = _to_numpy(trades, "pnl_neto")
    wins = float(pnl[pnl > 0].sum())
    losses = float(abs(pnl[pnl < 0].sum()))
    return wins / losses if losses != 0 else float("nan")


# =============================================================================
# 9. FUNCIÓN PRINCIPAL — RESUMEN_METRICAS
# =============================================================================

def resumen_metricas(
    trades: TradesDF,
    *,
    saldo_inicial: float,
    equity_curve: Optional[List[float]] = None,
    period_start: Optional[pd.Timestamp] = None,
    period_end: Optional[pd.Timestamp] = None,
    timeframe: Optional[str] = None,
) -> Dict[str, Any]:
    """
    FUNCIÓN PRINCIPAL: CALCULA TODAS LAS MÉTRICAS Y DEVUELVE DICCIONARIO COMPLETO.

    Acepta tanto Polars como Pandas DataFrame.
    Detecta automáticamente si usar Numba (rápido) o Python (fallback).

    Args:
        trades:       DataFrame con los trades ejecutados.
        saldo_inicial: Saldo inicial de la cuenta ($).
        equity_curve:  Curva de equity opcional.
        period_start:  Inicio del periodo (para trades_por_dia preciso).
        period_end:    Fin del periodo.
        timeframe:     Timeframe de operación.

    Returns:
        Diccionario con TODAS las métricas del sistema.
    """
    if _empty(trades):
        return _empty_metrics_dict(saldo_inicial)

    # =========================================================================
    # RUTA RÁPIDA: Usar Numba si está disponible
    # =========================================================================
    if USE_NUMBA_METRICS and NUMBA_METRICS_AVAILABLE:
        try:
            return _resumen_metricas_numba_wrapper(
                trades, saldo_inicial, equity_curve, period_start, period_end, timeframe
            )
        except Exception:
            pass

    # =========================================================================
    # RUTA ESTÁNDAR: Cálculo Python tradicional
    # =========================================================================
    return _resumen_metricas_python(trades, saldo_inicial, equity_curve, period_start, period_end, timeframe)


# =============================================================================
# 10. IMPLEMENTACIONES INTERNAS
# =============================================================================

def _empty_metrics_dict(saldo_inicial: float) -> Dict[str, Any]:
    """DICCIONARIO DE MÉTRICAS VACÍO (VERSIÓN COMPLETA PARA RUTA PYTHON)."""
    return {
        # ── 6 métricas canónicas ──────────────────────────────────────────
        "roi": 0.0,
        "winrate": 0.0,
        "drawdown": 0.0,
        "expectativa": 0.0,
        "profit_factor": float("nan"),
        "sharpe": 0.0,
        # ── operacional ──────────────────────────────────────────────────
        "trades_por_dia": 0.0,
        "n_trades": 0,
        "total_trades": 0,
        "num_trades": 0,
        "n_trades_long": 0,
        "count_longs": 0,
        "num_longs": 0,
        "n_trades_short": 0,
        "count_shorts": 0,
        "num_shorts": 0,
        "saldo_actual": float(saldo_inicial),
        "saldo_min": float(saldo_inicial),
        "saldo_max": float(saldo_inicial),
        "saldo_mean": float(saldo_inicial),
        "duration_mean_min": 0.0,
        "comisiones_total": 0.0,
        "saldo_sin_comisiones": 0.0,
        "pnl_neto": 0.0,
        "net_pnl": 0.0,
    }


def _resumen_metricas_numba_wrapper(
    trades: TradesDF,
    saldo_inicial: float,
    equity_curve: Optional[List[float]],
    period_start: Optional[pd.Timestamp],
    period_end: Optional[pd.Timestamp],
    timeframe: Optional[str] = None,
) -> Dict[str, Any]:
    """
    WRAPPER NUMBA: PREPARA DATOS, LLAMA AL KERNEL Y COMPLETA MÉTRICAS.

    El kernel Numba calcula las métricas puramente numéricas.
    Este wrapper completa las que requieren timestamps (trades_por_dia, calmar, etc.).
    """

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
    metrics["trades_por_dia"] = trades_por_dia(
        trades, period_start=period_start, period_end=period_end
    )

    # Duración y comisiones (compatible Polars/Pandas)
    metrics["duration_mean_min"] = _estimate_duration_mean_min(trades)

    if "comision" in cols:
        metrics["comisiones_total"] = float(np.sum(_to_numpy(trades, "comision")))
    else:
        metrics["comisiones_total"] = 0.0

    # Sharpe per-trade recalculado en Python (más preciso que el kernel Numba)
    metrics["sharpe"] = sharpe(trades)

    # Beneficio bruto (sin comisiones) = PNL neto + comisiones pagadas
    pnl_neto_total = metrics.get("pnl_neto", 0.0) or metrics.get("net_pnl", 0.0)
    metrics["saldo_sin_comisiones"] = pnl_neto_total + metrics["comisiones_total"]
    if "pnl" in cols:
        pnl_bruto = _to_numpy(trades, "pnl")
        metrics["saldo_sin_comisiones"] = float(pnl_bruto.sum())

    return metrics


def _resumen_metricas_python(
    trades: TradesDF,
    saldo_inicial: float,
    equity_curve: Optional[List[float]],
    period_start: Optional[pd.Timestamp],
    period_end: Optional[pd.Timestamp],
    timeframe: Optional[str] = None,
) -> Dict[str, Any]:
    """
    VERSIÓN PYTHON PURA DE RESUMEN_METRICAS.

    Compatible con Polars y Pandas. Se usa como fallback cuando Numba no
    está disponible o cuando el kernel Numba falla.
    """

    if _empty(trades):
        return _empty_metrics_dict(saldo_inicial)

    # Extraer arrays
    saldo_despues = _to_numpy(trades, "saldo_despues")
    pnl_neto = _to_numpy(trades, "pnl_neto")
    cols = trades.columns if isinstance(trades, pl.DataFrame) else list(trades.columns)

    equity_curve = list(saldo_despues) if equity_curve is None else equity_curve
    _, max_dd_pct = max_drawdown(equity_curve)

    # Beneficio bruto (sin comisiones)
    pnl_bruto = _to_numpy(trades, "pnl") if "pnl" in cols else pnl_neto
    beneficio_bruto = float(pnl_bruto.sum())

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
    duration_mean_min = _estimate_duration_mean_min(trades)
    comision = _to_numpy(trades, "comision") if "comision" in cols else np.array([0.0])

    return {
        # ── 6 métricas canónicas ─────────────────────────────────────────
        "roi": roi_pct(trades, saldo_inicial),
        "winrate": winrate_pct(trades),
        "drawdown": max_dd_pct,
        "expectativa": expectativa(trades),
        "profit_factor": profit_factor(trades),
        "sharpe": sharpe(trades),
        # ── operacional ──────────────────────────────────────────────────
        "trades_por_dia": trades_por_dia(trades, period_start=period_start, period_end=period_end),
        "n_trades": int(len(trades)),
        "total_trades": int(len(trades)),
        "num_trades": int(len(trades)),
        "n_trades_long": n_trades_long,
        "count_longs": n_trades_long,
        "num_longs": n_trades_long,
        "n_trades_short": n_trades_short,
        "count_shorts": n_trades_short,
        "num_shorts": n_trades_short,
        "saldo_actual": saldo_actual,
        "saldo_min": saldo_min,
        "saldo_max": saldo_max,
        "saldo_mean": saldo_mean,
        "duration_mean_min": duration_mean_min,
        "comisiones_total": float(np.sum(comision)),
        "saldo_sin_comisiones": beneficio_bruto,
        "pnl_neto": float(np.sum(pnl_neto)),
        "net_pnl": float(np.sum(pnl_neto)),
    }

