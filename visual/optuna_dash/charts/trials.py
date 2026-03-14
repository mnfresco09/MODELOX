"""
MODELOX · charts/trials.py
Single-trial detail charts: equity curve, waterfall, PnL distribution,
exit breakdown, rolling Sharpe.
"""
from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from visual.optuna_dash.theme import (
    C, PALETTE, METRIC_LABEL,
    get_m, _lay, _empty, _fix_sub, _MONO,
)


def _pnl(equity: list, initial: float = 1000.0) -> list:
    """Convert equity curve to per-trade PnL list."""
    if not equity:
        return []
    return [equity[0] - initial] + [equity[i] - equity[i - 1] for i in range(1, len(equity))]


def make_equity_curve(trial: dict) -> go.Figure:
    """
    Equity curve with drawdown shading.
    Upper panel: equity line + running max shading (drawdown fill).
    Lower panel: drawdown % bar.
    """
    eq = trial.get("equity", [])
    m = trial.get("met", {})
    if not eq:
        return _empty("NO EQUITY CURVE")

    n = len(eq)
    idx = list(range(n))
    peak = [eq[0]]
    for v in eq[1:]:
        peak.append(max(peak[-1], v))
    dd = [-(peak[i] - eq[i]) / peak[i] * 100 if peak[i] > 0 else 0.0 for i in range(n)]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.70, 0.30],
        vertical_spacing=0.04,
        subplot_titles=["EQUITY CURVE", "DRAWDOWN (%)"],
    )

    # Equity fill to running-max (drawdown shading)
    fig.add_trace(go.Scatter(
        x=idx + idx[::-1],
        y=peak + eq[::-1],
        fill="toself",
        fillcolor="rgba(255,61,113,0.08)",
        line_color="rgba(0,0,0,0)",
        hoverinfo="skip",
        showlegend=False,
    ), row=1, col=1)

    # Equity line
    fig.add_trace(go.Scatter(
        x=idx, y=eq, mode="lines",
        name="Balance",
        line=dict(color=C["accent"], width=2),
        hovertemplate="Trade %{x} — $%{y:,.2f}<extra></extra>",
    ), row=1, col=1)

    # Peak/initial baseline
    fig.add_hline(y=1000, line=dict(color=C["border2"], width=1, dash="dot"), row=1, col=1)

    # Peak annotation
    pv = max(eq); pi = eq.index(pv)
    fig.add_annotation(
        x=pi, y=pv, text=f"PEAK ${pv:,.0f}", showarrow=True, arrowhead=2,
        arrowcolor=C["green"], arrowwidth=1,
        font=dict(color=C["green"], size=9, family=_MONO),
        xref="x1", yref="y1",
    )

    # Final value
    fig.add_annotation(
        x=idx[-1], y=eq[-1], text=f"${eq[-1]:,.0f}", showarrow=False,
        font=dict(color=C["text"], size=9, family=_MONO),
        xref="x1", yref="y1", xanchor="right",
    )

    # Drawdown bars
    fig.add_trace(go.Scatter(
        x=idx, y=dd, mode="lines",
        name="DD%",
        line=dict(color=C["red"], width=1.5),
        fill="tozeroy",
        fillcolor="rgba(255,61,113,0.10)",
        hovertemplate="Trade %{x} — DD %{y:.2f}%<extra></extra>",
    ), row=2, col=1)

    # Stats annotation
    fig.add_annotation(
        text=(f"ROI {m.get('roi', 0):.1f}%  ·  MaxDD {m.get('drawdown', 0):.2f}%"
              f"  ·  Sharpe {m.get('sharpe', 0):.3f}"),
        xref="paper", yref="paper", x=0.01, y=0.99, showarrow=False,
        bgcolor="rgba(7,11,20,0.70)",
        bordercolor=C["border"], borderwidth=1,
        font=dict(color=C["text2"], size=9, family=_MONO),
    )

    from visual.optuna_dash.theme import PL
    d = dict(PL)
    d.update(dict(
        height=400,
        title=f"TRIAL #{trial['number']} · EQUITY & DRAWDOWN",
        margin=dict(l=50, r=14, t=38, b=30),
    ))
    fig.update_layout(**d)
    _fix_sub(fig)
    return fig


def make_waterfall(trial: dict) -> go.Figure:
    """
    Trade-by-trade P&L bar chart with cumulative line.
    Green wins, red losses.
    """
    eq = trial.get("equity", [])
    if not eq:
        return _empty("NO EQUITY DATA")
    p = _pnl(eq)
    cum = [sum(p[: i + 1]) for i in range(len(p))]
    colors = [C["green"] if v >= 0 else C["red"] for v in p]

    # Streak stats
    ws = ls = cw = cl = 0
    for v in p:
        if v > 0:
            cw += 1; cl = 0; ws = max(ws, cw)
        elif v < 0:
            cl += 1; cw = 0; ls = max(ls, cl)
    wins = sum(1 for v in p if v > 0)
    tot = len(p)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.45],
        vertical_spacing=0.04,
        subplot_titles=["P&L PER TRADE ($)", "CUMULATIVE ($)"],
    )
    fig.add_trace(go.Bar(
        x=list(range(len(p))), y=p,
        marker=dict(color=colors, line_width=0),
        name="P&L",
        hovertemplate="Trade %{x} — $%{y:+.2f}<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=0, line=dict(color=C["border2"], width=1, dash="dot"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=list(range(len(cum))), y=cum, mode="lines",
        name="Cumulative",
        line=dict(color=C["accent"], width=1.8),
        fill="tozeroy",
        fillcolor="rgba(0,200,255,0.07)",
        hovertemplate="Trade %{x} — $%{y:+.2f}<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=0, line=dict(color=C["border2"], width=1, dash="dot"), row=2, col=1)
    fig.add_annotation(
        text=f"WR {100*wins/tot:.1f}%  ·  Max win streak {ws}  ·  Max loss streak {ls}",
        xref="paper", yref="paper", x=0.01, y=0.99, showarrow=False,
        bgcolor="rgba(7,11,20,0.70)",
        bordercolor=C["border"], borderwidth=1,
        font=dict(color=C["text2"], size=9, family=_MONO),
    )

    from visual.optuna_dash.theme import PL
    d = dict(PL)
    d.update(dict(
        height=380,
        title=f"TRIAL #{trial['number']} · TRADE ANALYSIS",
        bargap=0.06,
        margin=dict(l=50, r=14, t=38, b=30),
    ))
    fig.update_layout(**d)
    _fix_sub(fig)
    return fig


def make_pnl_distribution(trial: dict) -> go.Figure:
    """Histogram of per-trade PnL, green for wins, red for losses."""
    eq = trial.get("equity", [])
    m = trial.get("met", {})
    if not eq:
        return _empty("NO EQUITY DATA")
    p = _pnl(eq)
    wins = [v for v in p if v > 0]
    losses = [v for v in p if v < 0]
    aw = sum(wins) / len(wins) if wins else 0
    al = sum(losses) / len(losses) if losses else 0

    fig = go.Figure()
    if wins:
        fig.add_trace(go.Histogram(
            x=wins, nbinsx=25, name="Win",
            marker_color=C["green"], opacity=0.70,
        ))
        fig.add_vline(
            x=aw, line=dict(color=C["green"], width=1.4, dash="dash"),
            annotation_text=f"+${aw:.2f}",
            annotation_font_color=C["green"], annotation_font_size=8,
            annotation_position="top right",
        )
    if losses:
        fig.add_trace(go.Histogram(
            x=losses, nbinsx=25, name="Loss",
            marker_color=C["red"], opacity=0.70,
        ))
        fig.add_vline(
            x=al, line=dict(color=C["red"], width=1.4, dash="dash"),
            annotation_text=f"-${abs(al):.2f}",
            annotation_font_color=C["red"], annotation_font_size=8,
            annotation_position="top left",
        )
    fig.add_vline(x=0, line=dict(color=C["border2"], width=1))
    exp = float(m.get("expectativa", 0))
    rr = abs(aw / al) if al != 0 else 0.0
    return _lay(
        fig,
        title=f"P&L DISTRIBUTION  ·  Expectancy ${exp:.2f}  ·  R:R {rr:.2f}",
        xaxis_title="P&L ($)",
        yaxis_title="Trades",
        barmode="overlay",
        bargap=0.04,
        height=280,
    )


def make_exit_breakdown(trial: dict) -> go.Figure:
    """
    Donut chart of exit reasons.
    Exit codes: 0=EndData, 1=SL, 2=TP, 3=Trail, 4=Time, 5=Signal
    Derived from equity curve length since exit codes aren't stored per-trade;
    use metrics to derive approximate breakdown, or just display metric stats.
    """
    m = trial.get("met", {})
    eq = trial.get("equity", [])
    if not m and not eq:
        return _empty("NO METRICS")

    # Build approximate breakdown from available metrics
    n_trades = int(m.get("n_trades", len(eq)) or len(eq))
    if n_trades == 0:
        return _empty("0 TRADES")

    wr = float(m.get("winrate", 0)) / 100
    wins = int(n_trades * wr)
    losses = n_trades - wins

    labels = ["WIN", "LOSS"]
    values = [wins, losses]
    colors = [C["green"], C["red"]]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color=C["bg"], width=2)),
        textfont=dict(color=C["bg"], size=10, family=_MONO),
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    ))
    fig.add_annotation(
        text=f"{wr*100:.1f}%<br>WR",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(color=C["text"], size=13, family=_MONO),
        xanchor="center",
    )

    return _lay(
        fig,
        title=f"TRIAL #{trial['number']} · WIN/LOSS BREAKDOWN",
        height=280,
        showlegend=True,
    )


def make_rolling_sharpe(trial: dict) -> go.Figure:
    """Rolling 30-trade Sharpe ratio, win rate, and profit factor evolution."""
    eq = trial.get("equity", [])
    if len(eq) < 5:
        return _empty("TOO FEW TRADES")
    p = _pnl(eq)
    wr_vals, pf_vals, sharpe_vals = [], [], []
    W = 30
    for i in range(len(p)):
        window = p[max(0, i - W + 1) : i + 1]
        w_wins = [v for v in window if v > 0]
        w_loss = [v for v in window if v < 0]
        tot_w = len(window)
        wc = len(w_wins)
        wr_vals.append(100 * wc / tot_w if tot_w else 50)
        w_sum_w = sum(w_wins); w_sum_l = abs(sum(w_loss))
        pf_vals.append(w_sum_w / w_sum_l if w_sum_l > 0 else float("nan"))
        if len(window) >= 2:
            mu = sum(window) / len(window)
            std = (sum((v - mu) ** 2 for v in window) / len(window)) ** 0.5
            sharpe_vals.append(mu / std if std > 1e-9 else 0.0)
        else:
            sharpe_vals.append(float("nan"))

    idx = list(range(len(p)))
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.34, 0.33, 0.33],
        vertical_spacing=0.04,
        subplot_titles=["ROLLING WIN RATE (%)", "ROLLING PROFIT FACTOR", "ROLLING SHARPE (30-trade)"],
    )
    for row, vals, color, ref in [
        (1, wr_vals, C["accent"], 50),
        (2, pf_vals, C["green"], 1),
        (3, sharpe_vals, C["purple"], 0),
    ]:
        fig.add_trace(go.Scatter(
            x=idx, y=[v if v == v else None for v in vals],
            mode="lines", line=dict(color=color, width=1.5),
            showlegend=False,
        ), row=row, col=1)
        fig.add_hline(y=ref, line=dict(color=C["border2"], width=1, dash="dot"), row=row, col=1)

    from visual.optuna_dash.theme import PL
    d = dict(PL)
    d.update(dict(
        height=420,
        title=f"TRIAL #{trial['number']} · ROLLING STATISTICS (W={W})",
        margin=dict(l=50, r=14, t=38, b=30),
    ))
    fig.update_layout(**d)
    _fix_sub(fig)
    return fig
