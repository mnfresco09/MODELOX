"""
================================================================================
IA/BACKTEST.PY — MOTOR DE BACKTEST CON TP/SL EN USD
================================================================================
Reglas:
  • TP = precio_entrada + $500 (LONG)  /  precio_entrada - $500 (SHORT)
  • SL = precio_entrada - $500 (LONG)  /  precio_entrada + $500 (SHORT)
  • Se detecta TP/SL usando high/low de velas 1m (no solo close)
  • Posición = SALDO_USADO * APALANCAMIENTO / precio_entrada
  • Comisión = pos_valor * COMISION_PCT * COMISION_SIDES (apertura + cierre)
  • Una sola posición abierta a la vez
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from numba import njit
    _NUMBA_OK = True
except ImportError:
    _NUMBA_OK = False

from IA.config import (
    TP_USD, SL_USD, MAX_FORWARD_CANDLES,
    SALDO_INICIAL, APALANCAMIENTO, SALDO_USADO,
    COMISION_PCT, COMISION_SIDES,
)


# =============================================================================
# ESTRUCTURA DE TRADE
# =============================================================================

@dataclass
class Trade:
    tipo:          str      # "LONG" o "SHORT"
    entry_idx:     int      # índice en raw_df
    entry_time:    pd.Timestamp
    entry_price:   float
    tp_price:      float
    sl_price:      float
    exit_price:    float    = 0.0
    exit_time:     Optional[pd.Timestamp] = None
    exit_idx:      int      = 0
    exit_reason:   str      = ""    # "TP" | "SL" | "TIMEOUT"
    qty:           float    = 0.0
    pos_value:     float    = 0.0
    pnl_bruto:     float    = 0.0
    comision:      float    = 0.0
    pnl_neto:      float    = 0.0
    saldo_antes:   float    = 0.0
    saldo_despues: float    = 0.0
    duracion_velas:int      = 0


# =============================================================================
# KERNEL NUMBA: BÚSQUEDA DE TP/SL
# =============================================================================

if _NUMBA_OK:
    @njit(cache=True, fastmath=True)
    def _find_exit_numba(
        high:        np.ndarray,
        low:         np.ndarray,
        close:       np.ndarray,
        start_idx:   int,
        tp_price:    float,
        sl_price:    float,
        is_long:     bool,
        max_forward: int,
    ) -> Tuple[float, int, str]:
        """
        Busca la vela de salida para un trade.
        Devuelve: (exit_price, exit_offset, reason)
        """
        n = len(close)
        for j in range(1, min(max_forward + 1, n - start_idx)):
            idx = start_idx + j
            h   = high[idx]
            lo  = low[idx]

            if is_long:
                tp_hit = h  >= tp_price
                sl_hit = lo <= sl_price
            else:
                tp_hit = lo <= tp_price
                sl_hit = h  >= sl_price

            if tp_hit and sl_hit:
                # Ambas: usar vela previa para desambiguar dirección
                prev_c = close[idx - 1]
                entry  = close[start_idx]
                if is_long:
                    if prev_c >= entry:
                        return tp_price, j, "TP"
                    else:
                        return sl_price, j, "SL"
                else:
                    if prev_c <= entry:
                        return tp_price, j, "TP"
                    else:
                        return sl_price, j, "SL"
            elif tp_hit:
                return tp_price, j, "TP"
            elif sl_hit:
                return sl_price, j, "SL"

        # Timeout: cerrar al close de la última vela
        last = min(start_idx + max_forward, n - 1)
        return close[last], max_forward, "TIMEOUT"

else:
    def _find_exit_numba(high, low, close, start_idx, tp_price, sl_price, is_long, max_forward):
        """Fallback Python puro."""
        n = len(close)
        for j in range(1, min(max_forward + 1, n - start_idx)):
            idx    = start_idx + j
            h, lo  = high[idx], low[idx]
            if is_long:
                tp_hit = h  >= tp_price
                sl_hit = lo <= sl_price
            else:
                tp_hit = lo <= tp_price
                sl_hit = h  >= sl_price
            if tp_hit and sl_hit:
                prev_c = close[idx - 1]
                entry  = close[start_idx]
                if is_long:
                    return (tp_price, j, "TP") if prev_c >= entry else (sl_price, j, "SL")
                else:
                    return (tp_price, j, "TP") if prev_c <= entry else (sl_price, j, "SL")
            elif tp_hit:
                return tp_price, j, "TP"
            elif sl_hit:
                return sl_price, j, "SL"
        last = min(start_idx + max_forward, n - 1)
        return close[last], max_forward, "TIMEOUT"


# =============================================================================
# MOTOR DE BACKTEST
# =============================================================================

def run_backtest(
    raw_df:         pd.DataFrame,
    feat_df:        pd.DataFrame,
    signals:        np.ndarray,
    signal_indices: np.ndarray,
    saldo_inicial:  float = SALDO_INICIAL,
    apalancamiento: float = APALANCAMIENTO,
    saldo_usado:    float = SALDO_USADO,
    comision_pct:   float = COMISION_PCT,
    comision_sides: int   = COMISION_SIDES,
    tp_usd:         float = TP_USD,
    sl_usd:         float = SL_USD,
    max_forward:    int   = MAX_FORWARD_CANDLES,
) -> Tuple[List[Trade], np.ndarray]:
    """
    Ejecuta el backtest sobre los datos reales alineados con feat_df.

    Args:
      raw_df         : OHLCV completo (índice datetime, columnas open/high/low/close/volume)
      feat_df        : Features DataFrame (alineado con raw_df)
      signals        : Array (N,) de señales {-1, 0, +1}
      signal_indices : Índices en feat_df de cada señal
      saldo_inicial  : Capital inicial
      ...

    Returns:
      trades        : Lista de Trade ejecutados
      equity_curve  : Curva de equity (saldo después de cada trade)
    """
    # ── Arrays numpy del DataFrame completo ─────────────────────────
    all_times  = raw_df.index
    high_arr   = raw_df["high"].values.astype(np.float64)
    low_arr    = raw_df["low"].values.astype(np.float64)
    close_arr  = raw_df["close"].values.astype(np.float64)

    # ── Mapa: timestamp → índice numérico en raw_df ──────────────────
    time_to_idx = {t: i for i, t in enumerate(all_times)}

    trades:       List[Trade] = []
    equity_curve: List[float] = [saldo_inicial]
    saldo         = saldo_inicial
    last_exit_idx = -1    # índice de la última salida (no solapar trades)

    for si, sig in enumerate(signals):
        if sig == 0:
            continue

        feat_row_idx   = int(signal_indices[si])
        if feat_row_idx >= len(feat_df):
            continue

        entry_timestamp = feat_df.index[feat_row_idx]
        raw_entry_idx   = time_to_idx.get(entry_timestamp, None)
        if raw_entry_idx is None:
            continue

        # ── No solapar trades ─────────────────────────────────────────
        if raw_entry_idx <= last_exit_idx:
            continue

        # ── Precio de entrada = close de la vela de señal ────────────
        entry_price = float(close_arr[raw_entry_idx])
        if entry_price <= 0:
            continue

        is_long   = (sig == +1)
        tipo      = "LONG" if is_long else "SHORT"

        if is_long:
            tp_price = entry_price + tp_usd
            sl_price = entry_price - sl_usd
        else:
            tp_price = entry_price - tp_usd
            sl_price = entry_price + sl_usd

        # ── Buscar salida (TP o SL) ───────────────────────────────────
        exit_price, exit_offset, exit_reason = _find_exit_numba(
            high_arr, low_arr, close_arr,
            raw_entry_idx,
            tp_price, sl_price,
            is_long, max_forward,
        )

        exit_raw_idx   = min(raw_entry_idx + exit_offset, len(close_arr) - 1)
        exit_timestamp = all_times[exit_raw_idx]
        last_exit_idx  = exit_raw_idx

        # ── PnL ───────────────────────────────────────────────────────
        pos_value  = saldo_usado * float(apalancamiento)
        qty        = pos_value / entry_price
        comision   = pos_value * comision_pct * comision_sides

        if is_long:
            pnl_bruto = qty * (float(exit_price) - entry_price)
        else:
            pnl_bruto = qty * (entry_price - float(exit_price))

        pnl_neto      = pnl_bruto - comision
        saldo_antes   = saldo
        saldo        += pnl_neto
        saldo_despues = saldo

        equity_curve.append(saldo)

        trade = Trade(
            tipo           = tipo,
            entry_idx      = raw_entry_idx,
            entry_time     = entry_timestamp,
            entry_price    = entry_price,
            tp_price       = tp_price,
            sl_price       = sl_price,
            exit_price     = float(exit_price),
            exit_time      = exit_timestamp,
            exit_idx       = exit_raw_idx,
            exit_reason    = exit_reason,
            qty            = qty,
            pos_value      = pos_value,
            pnl_bruto      = pnl_bruto,
            comision       = comision,
            pnl_neto       = pnl_neto,
            saldo_antes    = saldo_antes,
            saldo_despues  = saldo_despues,
            duracion_velas = exit_offset,
        )
        trades.append(trade)

    return trades, np.array(equity_curve, dtype=np.float64)


# =============================================================================
# MÉTRICAS FINANCIERAS
# =============================================================================

def compute_backtest_metrics(
    trades:        List[Trade],
    equity_curve:  np.ndarray,
    saldo_inicial: float = SALDO_INICIAL,
) -> dict:
    """
    Calcula métricas financieras completas:
      SQN, ROI, Max Drawdown, WR, Longs/Shorts, Profit Factor,
      Expectancy, Sharpe, Best/Worst trade, Streaks, etc.
    """
    if not trades:
        return _empty_metrics(saldo_inicial)

    pnls     = np.array([t.pnl_neto      for t in trades], dtype=np.float64)
    is_long  = np.array([t.tipo == "LONG" for t in trades], dtype=bool)
    is_short = ~is_long
    wins     = pnls > 0

    n        = len(trades)
    n_long   = int(is_long.sum())
    n_short  = int(is_short.sum())
    n_wins   = int(wins.sum())
    n_losses = n - n_wins

    # ── SQN ──────────────────────────────────────────────────────────
    mean_pnl = float(np.mean(pnls))
    std_pnl  = float(np.std(pnls, ddof=1)) if n > 1 else 0.0
    sqn      = float(np.sqrt(n) * mean_pnl / std_pnl) if std_pnl > 1e-10 else 0.0
    sqn      = max(-100.0, min(100.0, sqn))

    # ── ROI ───────────────────────────────────────────────────────────
    roi = 100.0 * (equity_curve[-1] - saldo_inicial) / saldo_inicial

    # ── Max Drawdown ──────────────────────────────────────────────────
    peaks = np.maximum.accumulate(equity_curve)
    dds   = 100.0 * (peaks - equity_curve) / np.where(peaks > 0, peaks, 1.0)
    max_dd = float(np.max(dds))

    # ── Win Rate ──────────────────────────────────────────────────────
    wr = 100.0 * n_wins / n

    # ── Win Rate Longs / Shorts ───────────────────────────────────────
    pnls_long  = pnls[is_long]
    pnls_short = pnls[is_short]
    wr_long    = 100.0 * float((pnls_long  > 0).sum()) / n_long  if n_long  > 0 else 0.0
    wr_short   = 100.0 * float((pnls_short > 0).sum()) / n_short if n_short > 0 else 0.0

    # ── Profit Factor ─────────────────────────────────────────────────
    sum_wins   = float(pnls[pnls > 0].sum()) if (pnls > 0).any() else 0.0
    sum_losses = float(abs(pnls[pnls < 0].sum())) if (pnls < 0).any() else 0.0
    pf         = sum_wins / sum_losses if sum_losses > 0 else float("nan")

    # ── Payoff Ratio ──────────────────────────────────────────────────
    avg_win  = float(pnls[pnls > 0].mean()) if (pnls > 0).any() else 0.0
    avg_loss = float(abs(pnls[pnls < 0].mean())) if (pnls < 0).any() else 0.0
    pr       = avg_win / avg_loss if avg_loss > 0 else float("nan")

    # ── Expectancy ────────────────────────────────────────────────────
    p_win    = n_wins / n
    exp_val  = p_win * avg_win - (1 - p_win) * avg_loss

    # ── Sharpe per-trade ──────────────────────────────────────────────
    ret_pct  = pnls / SALDO_USADO
    sharpe   = float(np.mean(ret_pct) / (np.std(ret_pct, ddof=1) + 1e-10)) if n > 1 else 0.0
    sharpe   = max(-100.0, min(100.0, sharpe))

    # ── Rachas ────────────────────────────────────────────────────────
    max_win_streak  = _max_streak(wins)
    max_loss_streak = _max_streak(~wins)

    # ── Duración media ────────────────────────────────────────────────
    dur_mean = float(np.mean([t.duracion_velas for t in trades]))

    # ── Exit reasons ─────────────────────────────────────────────────
    n_tp      = sum(1 for t in trades if t.exit_reason == "TP")
    n_sl      = sum(1 for t in trades if t.exit_reason == "SL")
    n_timeout = sum(1 for t in trades if t.exit_reason == "TIMEOUT")

    return {
        # Principales
        "sqn"            : round(sqn, 3),
        "roi"            : round(roi, 2),
        "max_drawdown"   : round(max_dd, 2),
        "winrate"        : round(wr, 2),
        # Trades
        "n_trades"       : n,
        "n_long"         : n_long,
        "n_short"        : n_short,
        "n_wins"         : n_wins,
        "n_losses"       : n_losses,
        # WR por tipo
        "wr_long"        : round(wr_long,  2),
        "wr_short"       : round(wr_short, 2),
        # Calidad
        "profit_factor"  : round(pf,       3) if not np.isnan(pf)  else float("nan"),
        "payoff_ratio"   : round(pr,       3) if not np.isnan(pr)  else float("nan"),
        "expectancy"     : round(exp_val,  3),
        "sharpe"         : round(sharpe,   3),
        # PnL
        "pnl_total"      : round(float(pnls.sum()),   2),
        "pnl_mean"       : round(mean_pnl,            2),
        "best_trade"     : round(float(pnls.max()),   2),
        "worst_trade"    : round(float(pnls.min()),   2),
        "avg_win"        : round(avg_win,             2),
        "avg_loss"       : round(-avg_loss,           2),
        # Rachas
        "max_win_streak" : max_win_streak,
        "max_loss_streak": max_loss_streak,
        # Duración / salidas
        "dur_mean_velas" : round(dur_mean,  1),
        "n_tp"           : n_tp,
        "n_sl"           : n_sl,
        "n_timeout"      : n_timeout,
        # Equity
        "saldo_inicial"  : saldo_inicial,
        "saldo_final"    : round(equity_curve[-1], 2),
        "saldo_max"      : round(float(equity_curve.max()), 2),
        "saldo_min"      : round(float(equity_curve.min()), 2),
    }


def _max_streak(condition: np.ndarray) -> int:
    """Calcula racha máxima de True consecutivos."""
    if len(condition) == 0:
        return 0
    max_s = cur_s = 0
    for v in condition:
        if v:
            cur_s += 1
            max_s  = max(max_s, cur_s)
        else:
            cur_s  = 0
    return max_s


def _empty_metrics(saldo_inicial: float) -> dict:
    return {
        "sqn": 0.0, "roi": 0.0, "max_drawdown": 0.0, "winrate": 0.0,
        "n_trades": 0, "n_long": 0, "n_short": 0, "n_wins": 0, "n_losses": 0,
        "wr_long": 0.0, "wr_short": 0.0, "profit_factor": float("nan"),
        "payoff_ratio": float("nan"), "expectancy": 0.0, "sharpe": 0.0,
        "pnl_total": 0.0, "pnl_mean": 0.0, "best_trade": 0.0, "worst_trade": 0.0,
        "avg_win": 0.0, "avg_loss": 0.0, "max_win_streak": 0, "max_loss_streak": 0,
        "dur_mean_velas": 0.0, "n_tp": 0, "n_sl": 0, "n_timeout": 0,
        "saldo_inicial": saldo_inicial, "saldo_final": saldo_inicial,
        "saldo_max": saldo_inicial, "saldo_min": saldo_inicial,
    }


def trades_to_dataframe(trades: List[Trade]) -> pd.DataFrame:
    """Convierte lista de Trade a DataFrame para análisis."""
    if not trades:
        return pd.DataFrame()
    rows = []
    for t in trades:
        rows.append({
            "tipo"          : t.tipo,
            "entry_time"    : t.entry_time,
            "exit_time"     : t.exit_time,
            "entry_price"   : t.entry_price,
            "exit_price"    : t.exit_price,
            "tp_price"      : t.tp_price,
            "sl_price"      : t.sl_price,
            "exit_reason"   : t.exit_reason,
            "qty"           : round(t.qty, 6),
            "pos_value"     : round(t.pos_value, 2),
            "pnl_bruto"     : round(t.pnl_bruto,  2),
            "comision"      : round(t.comision,    2),
            "pnl_neto"      : round(t.pnl_neto,    2),
            "saldo_antes"   : round(t.saldo_antes,  2),
            "saldo_despues" : round(t.saldo_despues, 2),
            "duracion_velas": t.duracion_velas,
        })
    return pd.DataFrame(rows)
