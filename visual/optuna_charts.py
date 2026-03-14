"""visual/optuna_charts.py

Genera y guarda en la DB de Optuna un conjunto completo y profesional de
gráficas para cada estudio, visibles en optuna-dashboard → "Custom Plots".

Secciones por estudio:
  A) Nativas Optuna   — 8 gráficas estándar (optimization history, etc.)
  B) Visión global    — Score landscape, distribución, métricas, correlaciones
  C) Mejor trial      — Equity curve, P&L por operación, drawdown, distribución
  D) Solo Global      — Comparativas por activo (box plots, radar)

Uso:
    from visual.optuna_charts import generate_all_charts_for_strategy
    generate_all_charts_for_strategy(strategy_id=1)
"""
from __future__ import annotations

import json
import warnings
from typing import Optional

import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots


# ─────────────────────────────────────────────────────────────────────────────
# TEMA PROFESIONAL
# ─────────────────────────────────────────────────────────────────────────────
_BG       = "#0d1117"       # fondo panel
_BG_PLOT  = "#161b22"       # fondo área de plot
_GRID     = "#21262d"       # líneas de cuadrícula
_TEXT     = "#e6edf3"       # texto principal
_TEXT_MUT = "#8b949e"       # texto secundario
_ACCENT   = "#58a6ff"       # azul acento
_GREEN    = "#3fb950"       # profit / positivo
_RED      = "#f85149"       # loss / negativo
_YELLOW   = "#d29922"       # warning / neutro
_PURPLE   = "#bc8cff"       # info adicional
_ORANGE   = "#ff7b72"       # alternativo negativo

_ASSET_COLORS = ["#58a6ff", "#3fb950", "#d29922", "#bc8cff", "#ff7b72",
                 "#79c0ff", "#56d364", "#e3b341", "#d2a8ff", "#ffa198"]

def _layout(**extra) -> dict:
    """Base de layout profesional oscuro."""
    base = dict(
        paper_bgcolor = _BG,
        plot_bgcolor  = _BG_PLOT,
        font          = dict(color=_TEXT, family="'JetBrains Mono', 'Fira Code', monospace, sans-serif", size=11),
        title_font    = dict(color=_TEXT, size=14, family="'JetBrains Mono', sans-serif"),
        legend        = dict(bgcolor="rgba(0,0,0,0)", bordercolor=_GRID, borderwidth=1),
        margin        = dict(l=60, r=30, t=50, b=50),
        hoverlabel    = dict(bgcolor=_BG, bordercolor=_ACCENT, font_color=_TEXT),
        xaxis         = dict(gridcolor=_GRID, linecolor=_GRID, zerolinecolor=_GRID, color=_TEXT_MUT),
        yaxis         = dict(gridcolor=_GRID, linecolor=_GRID, zerolinecolor=_GRID, color=_TEXT_MUT),
    )
    base.update(extra)
    return base


def _annotate(fig, text: str, x=0.02, y=0.97) -> None:
    """Añade un texto de anotación flotante al gráfico."""
    fig.add_annotation(
        text=text, xref="paper", yref="paper",
        x=x, y=y, showarrow=False, align="left",
        bgcolor="rgba(0,0,0,0.5)", bordercolor=_GRID, borderwidth=1,
        font=dict(color=_TEXT_MUT, size=10),
    )


def _save(study, fig: go.Figure, gid: str) -> None:
    from optuna_dashboard import save_plotly_graph_object
    save_plotly_graph_object(study, fig, graph_object_id=gid)


def _safe_id(text: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9_-]", "_", text)[:64].strip("_-") or "chart"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

def _get_scores(study) -> list[float]:
    return [t.value for t in study.trials if t.value is not None]


def _get_metricas(trial) -> dict:
    raw = trial.user_attrs.get("metricas", None)
    if raw is None:
        return {}
    try:
        return raw if isinstance(raw, dict) else json.loads(raw)
    except Exception:
        return {}


def _get_equity(trial) -> list[float]:
    raw = trial.user_attrs.get("equity_curve", None)
    if raw is None:
        return []
    try:
        return raw if isinstance(raw, list) else json.loads(raw)
    except Exception:
        return []


def _best_trial(study):
    """Devuelve el trial con mayor value."""
    completed = [t for t in study.trials if t.value is not None]
    if not completed:
        return None
    return max(completed, key=lambda t: t.value)


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN A — NATIVAS OPTUNA (con layout oscuro aplicado)
# ─────────────────────────────────────────────────────────────────────────────
_NATIVE_CHARTS = [
    ("plot_optimization_history", "nat_opt_history",   "Optimization History"),
    ("plot_param_importances",    "nat_param_imp",     "Parameter Importances"),
    ("plot_parallel_coordinate",  "nat_parallel",      "Parallel Coordinate"),
    ("plot_slice",                "nat_slice",         "Slice Plot"),
    ("plot_contour",              "nat_contour",       "Contour Plot"),
    ("plot_edf",                  "nat_edf",           "Empirical Distribution (EDF)"),
    ("plot_rank",                 "nat_rank",          "Rank Plot"),
    ("plot_timeline",             "nat_timeline",      "Trial Timeline"),
]

def _section_a(study) -> int:
    import optuna.visualization as viz
    saved = 0
    for fn_name, gid, _title in _NATIVE_CHARTS:
        fn = getattr(viz, fn_name, None)
        if fn is None:
            continue
        try:
            fig = fn(study)
            fig.update_layout(**_layout())
            _save(study, fig, gid)
            saved += 1
        except Exception:
            pass
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN B — VISIÓN GLOBAL DE TODOS LOS TRIALS
# ─────────────────────────────────────────────────────────────────────────────

def _chart_score_distribution(study) -> Optional[go.Figure]:
    """Histograma + KDE de scores con percentiles marcados."""
    scores = _get_scores(study)
    if len(scores) < 5:
        return None

    import numpy as np
    scores_arr = sorted(scores)
    p25, p50, p75, p90 = (float(np.percentile(scores_arr, p)) for p in [25, 50, 75, 90])

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores, nbinsx=50, name="Score",
        marker_color=_ACCENT, opacity=0.75,
        hovertemplate="Score: %{x:.1f}<br>Count: %{y}<extra></extra>",
    ))
    for val, label, color in [(p50, "P50", _TEXT_MUT), (p75, "P75", _YELLOW), (p90, "P90", _GREEN)]:
        fig.add_vline(x=val, line_dash="dash", line_color=color, line_width=1.5,
                      annotation_text=f"<b>{label}</b> {val:.0f}",
                      annotation_font_color=color, annotation_position="top right")

    fig.update_layout(**_layout(
        title="Score Distribution — All Trials",
        xaxis_title="Final Score",
        yaxis_title="Count",
        bargap=0.05,
    ))
    _annotate(fig, f"n={len(scores):,}  best={max(scores):.0f}  mean={sum(scores)/len(scores):.0f}")
    return fig


def _chart_score_landscape(study) -> Optional[go.Figure]:
    """Scatter de todos los trials coloreados por score (top params)."""
    trials = [t for t in study.trials if t.value is not None and t.params]
    if len(trials) < 10:
        return None

    param_names = list(trials[0].params.keys())
    if len(param_names) < 2:
        return None

    px, py = param_names[0], param_names[1]
    xs = [t.params[px] for t in trials]
    ys = [t.params[py] for t in trials]
    scores = [t.value for t in trials]
    trial_nums = [t.number for t in trials]

    fig = go.Figure(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(
            color=scores, colorscale="Viridis", size=5, opacity=0.8,
            colorbar=dict(title="Score", thickness=12, tickfont=dict(color=_TEXT_MUT)),
            line=dict(width=0),
        ),
        text=[f"Trial {n}<br>Score: {s:.1f}" for n, s in zip(trial_nums, scores)],
        hovertemplate="%{text}<extra></extra>",
        name="Trials",
    ))
    fig.update_layout(**_layout(
        title=f"Score Landscape — {px} vs {py}",
        xaxis_title=px, yaxis_title=py,
    ))
    return fig


def _chart_metrics_scatter_matrix(study) -> Optional[go.Figure]:
    """Scatter de métricas clave (sharpe, roi, drawdown, pf) por trial."""
    import numpy as np

    data: dict = {"trial": [], "sharpe": [], "roi": [], "drawdown": [], "profit_factor": []}
    for t in study.trials:
        m = _get_metricas(t)
        if not m:
            continue
        data["trial"].append(t.number)
        data["sharpe"].append(float(m.get("sharpe", float("nan"))))
        data["roi"].append(float(m.get("roi", float("nan"))))
        data["drawdown"].append(float(m.get("drawdown", float("nan"))))
        data["profit_factor"].append(float(m.get("profit_factor", float("nan"))))

    if not data["trial"]:
        return None

    scores = [t.value for t in study.trials if t.value is not None]
    score_map = {t.number: t.value for t in study.trials if t.value is not None}
    colors = [score_map.get(n, 0) for n in data["trial"]]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Sharpe Ratio", "ROI (%)", "Drawdown (%)", "Profit Factor"],
        horizontal_spacing=0.12, vertical_spacing=0.18,
    )

    def _scatter(row, col, key, color_key=None):
        vals = data[key]
        c = colors if color_key is None else colors
        fig.add_trace(go.Scatter(
            x=data["trial"], y=vals, mode="markers",
            marker=dict(color=c, colorscale="Viridis", size=4, opacity=0.7, line_width=0),
            hovertemplate=f"Trial %{{x}}<br>{key}: %{{y:.3f}}<extra></extra>",
            showlegend=False,
        ), row=row, col=col)

    _scatter(1, 1, "sharpe")
    _scatter(1, 2, "roi")
    _scatter(2, 1, "drawdown")
    _scatter(2, 2, "profit_factor")

    fig.update_layout(**_layout(title="Key Metrics — All Trials", height=520))
    for ax in fig.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].update(gridcolor=_GRID, linecolor=_GRID, color=_TEXT_MUT)
    for ann in fig.layout.annotations:
        ann.font.color = _TEXT_MUT
    return fig


def _chart_trial_correlation(study) -> Optional[go.Figure]:
    """Heatmap de correlación entre métricas numéricas de todos los trials."""
    import numpy as np

    keys = ["sharpe", "roi", "drawdown", "profit_factor", "winrate", "trades_por_dia"]
    rows = []
    for t in study.trials:
        m = _get_metricas(t)
        if not m:
            continue
        row = [float(m.get(k, float("nan"))) for k in keys]
        if any(v != v for v in row):  # skip rows with NaN
            continue
        rows.append(row)

    if len(rows) < 10:
        return None

    arr = [rows[i] for i in range(len(rows))]
    n = len(keys)
    corr = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            xi = [r[i] for r in arr]
            xj = [r[j] for r in arr]
            xi_m = sum(xi)/len(xi)
            xj_m = sum(xj)/len(xj)
            num = sum((a-xi_m)*(b-xj_m) for a, b in zip(xi, xj))
            den = (sum((a-xi_m)**2 for a in xi) * sum((b-xj_m)**2 for b in xj)) ** 0.5
            corr[i][j] = round(num/den, 3) if den > 1e-9 else 0.0

    labels = ["Sharpe", "ROI", "Drawdown", "Prof.Factor", "Winrate", "Trades/Day"]
    fig = go.Figure(go.Heatmap(
        z=corr, x=labels, y=labels,
        colorscale=[
            [0.0, _RED], [0.5, _BG_PLOT], [1.0, _GREEN],
        ],
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr],
        texttemplate="%{text}",
        hovertemplate="<b>%{y} × %{x}</b><br>r = %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(**_layout(title="Metrics Correlation Matrix", height=420))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN C — ANÁLISIS DEL MEJOR TRIAL
# ─────────────────────────────────────────────────────────────────────────────

def _chart_equity_curve(study) -> Optional[go.Figure]:
    """Equity curve del mejor trial con drawdown, anotaciones y estadísticas."""
    bt = _best_trial(study)
    if bt is None:
        return None
    eq = _get_equity(bt)
    if len(eq) < 3:
        return None
    m = _get_metricas(bt)

    trades_x = list(range(len(eq)))
    initial  = 1000.0  # saldo de partida

    # Calcular drawdown running
    peak    = [eq[0]]
    for v in eq[1:]:
        peak.append(max(peak[-1], v))
    dd_pct = [-(peak[i] - eq[i]) / peak[i] * 100 if peak[i] > 0 else 0 for i in range(len(eq))]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.04,
        subplot_titles=["Equity Curve", "Drawdown (%)"],
    )

    # Área de equity
    fig.add_trace(go.Scatter(
        x=trades_x, y=eq, mode="lines",
        line=dict(color=_ACCENT, width=1.8),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.08)",
        name="Balance",
        hovertemplate="Trade %{x}<br>Balance: $%{y:,.2f}<extra></extra>",
    ), row=1, col=1)

    # Línea inicial
    fig.add_hline(y=initial, line_dash="dot", line_color=_TEXT_MUT, line_width=1, row=1, col=1)

    # High watermark
    peak_idx = eq.index(max(eq))
    fig.add_annotation(
        x=peak_idx, y=max(eq), text=f"Peak<br>${max(eq):,.0f}",
        showarrow=True, arrowhead=2, arrowcolor=_GREEN,
        font=dict(color=_GREEN, size=9), row=1, col=1,
        xref="x1", yref="y1",
    )

    # Drawdown area
    fig.add_trace(go.Scatter(
        x=trades_x, y=dd_pct, mode="lines",
        line=dict(color=_RED, width=1.2),
        fill="tozeroy", fillcolor="rgba(248,81,73,0.15)",
        name="Drawdown %",
        hovertemplate="Trade %{x}<br>DD: %{y:.2f}%<extra></extra>",
    ), row=2, col=1)

    # Métricas en anotación
    roi_str   = f"{m.get('roi', 0):.1f}%"
    dd_str    = f"{m.get('drawdown', 0):.2f}%"
    sh_str    = f"{m.get('sharpe', 0):.3f}"
    pf_str    = f"{m.get('profit_factor', 0):.2f}"
    wr_str    = f"{m.get('winrate', 0):.1f}%"
    nt_str    = str(int(m.get('n_trades', len(eq))))
    score_str = f"{bt.value:.1f}"
    info = (f"Score {score_str} │ ROI {roi_str} │ DD {dd_str}<br>"
            f"Sharpe {sh_str} │ PF {pf_str} │ WR {wr_str} │ Trades {nt_str}")

    fig.update_layout(**_layout(
        title=f"Best Trial #{bt.number} — Full Performance",
        height=460,
    ))
    fig.update_xaxes(gridcolor=_GRID, linecolor=_GRID, color=_TEXT_MUT)
    fig.update_yaxes(gridcolor=_GRID, linecolor=_GRID, color=_TEXT_MUT)
    for ann in fig.layout.annotations:
        ann.font.color = _TEXT_MUT
    _annotate(fig, info)
    return fig


def _chart_pnl_per_trade(study) -> Optional[go.Figure]:
    """Barras P&L por operación (verde=ganancia, rojo=pérdida) del mejor trial."""
    bt = _best_trial(study)
    if bt is None:
        return None
    eq = _get_equity(bt)
    if len(eq) < 3:
        return None

    initial = 1000.0
    pnl = [eq[0] - initial] + [eq[i] - eq[i-1] for i in range(1, len(eq))]
    colors  = [_GREEN if v >= 0 else _RED for v in pnl]
    running = list(eq)  # ya es acumulado

    m = _get_metricas(bt)
    wins   = sum(1 for v in pnl if v > 0)
    losses = sum(1 for v in pnl if v < 0)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35], vertical_spacing=0.04,
        subplot_titles=["P&L per Trade", "Cumulative P&L"],
    )

    fig.add_trace(go.Bar(
        x=list(range(len(pnl))), y=pnl,
        marker_color=colors,
        name="P&L",
        hovertemplate="Trade %{x}<br>P&L: $%{y:+.2f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=list(range(len(running))), y=[v - initial for v in running],
        mode="lines", line=dict(color=_ACCENT, width=1.5),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.1)",
        name="Cumulative P&L",
        hovertemplate="Trade %{x}<br>Cum. P&L: $%{y:+.2f}<extra></extra>",
    ), row=2, col=1)

    fig.add_hline(y=0, line_dash="dot", line_color=_TEXT_MUT, line_width=1, row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color=_TEXT_MUT, line_width=1, row=2, col=1)

    fig.update_layout(**_layout(
        title=f"Best Trial #{bt.number} — Trade-by-Trade Analysis",
        height=440, bargap=0.05,
    ))
    fig.update_xaxes(gridcolor=_GRID, linecolor=_GRID, color=_TEXT_MUT, title_text="Trade #", row=2, col=1)
    fig.update_yaxes(gridcolor=_GRID, linecolor=_GRID, color=_TEXT_MUT)
    for ann in fig.layout.annotations:
        ann.font.color = _TEXT_MUT
    _annotate(fig, f"Wins: {wins} ({100*wins/len(pnl):.1f}%)  Losses: {losses} ({100*losses/len(pnl):.1f}%)")
    return fig


def _chart_pnl_distribution(study) -> Optional[go.Figure]:
    """Histograma + estadísticas de la distribución de P&L por operación."""
    bt = _best_trial(study)
    if bt is None:
        return None
    eq = _get_equity(bt)
    if len(eq) < 5:
        return None

    initial = 1000.0
    pnl = [eq[0] - initial] + [eq[i] - eq[i-1] for i in range(1, len(eq))]
    wins   = [v for v in pnl if v > 0]
    losses = [v for v in pnl if v < 0]
    avg_win  = sum(wins)  / len(wins)   if wins   else 0
    avg_loss = sum(losses)/ len(losses) if losses else 0

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=wins, nbinsx=25, name="Winning Trades",
        marker_color=_GREEN, opacity=0.75,
        hovertemplate="P&L: $%{x:.2f}<br>Count: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Histogram(
        x=losses, nbinsx=25, name="Losing Trades",
        marker_color=_RED, opacity=0.75,
        hovertemplate="P&L: $%{x:.2f}<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(x=avg_win,  line_dash="dash", line_color=_GREEN, line_width=1.5,
                  annotation_text=f"Avg win ${avg_win:.2f}", annotation_font_color=_GREEN)
    fig.add_vline(x=avg_loss, line_dash="dash", line_color=_RED, line_width=1.5,
                  annotation_text=f"Avg loss ${avg_loss:.2f}", annotation_font_color=_RED)
    fig.add_vline(x=0, line_color=_TEXT_MUT, line_width=1)

    expectancy = (len(wins)/len(pnl)) * avg_win + (len(losses)/len(pnl)) * avg_win if wins else 0
    m = _get_metricas(bt)
    expectancy = float(m.get("expectativa", 0))

    fig.update_layout(**_layout(
        title=f"Best Trial #{bt.number} — P&L Distribution",
        xaxis_title="P&L per Trade ($)", yaxis_title="Count",
        barmode="overlay", bargap=0.05,
    ))
    _annotate(fig, f"Expectancy: ${expectancy:.2f}  |  Avg win: ${avg_win:.2f}  |  Avg loss: ${avg_loss:.2f}")
    return fig


def _chart_drawdown_periods(study) -> Optional[go.Figure]:
    """Visualización del drawdown underwater con profundidades marcadas."""
    bt = _best_trial(study)
    if bt is None:
        return None
    eq = _get_equity(bt)
    if len(eq) < 5:
        return None

    peak = [eq[0]]
    for v in eq[1:]:
        peak.append(max(peak[-1], v))
    dd_abs = [eq[i] - peak[i] for i in range(len(eq))]
    dd_pct = [dd_abs[i] / peak[i] * 100 if peak[i] > 0 else 0 for i in range(len(eq))]
    max_dd = min(dd_pct)
    max_dd_idx = dd_pct.index(max_dd)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(dd_pct))), y=dd_pct,
        mode="lines", line=dict(color=_RED, width=1.5),
        fill="tozeroy", fillcolor="rgba(248,81,73,0.2)",
        name="Drawdown %",
        hovertemplate="Trade %{x}<br>DD: %{y:.2f}%<extra></extra>",
    ))
    fig.add_annotation(
        x=max_dd_idx, y=max_dd,
        text=f"Max DD<br>{max_dd:.2f}%",
        showarrow=True, arrowhead=2, arrowcolor=_RED,
        font=dict(color=_RED, size=9),
    )
    fig.add_hline(y=-5,  line_dash="dot", line_color=_YELLOW, line_width=1,
                  annotation_text="-5%", annotation_font_color=_YELLOW)
    fig.add_hline(y=-10, line_dash="dot", line_color=_ORANGE, line_width=1,
                  annotation_text="-10%", annotation_font_color=_ORANGE)
    fig.add_hline(y=-20, line_dash="dot", line_color=_RED, line_width=1,
                  annotation_text="-20%", annotation_font_color=_RED)

    m = _get_metricas(bt)
    fig.update_layout(**_layout(
        title=f"Best Trial #{bt.number} — Drawdown Periods",
        xaxis_title="Trade #", yaxis_title="Drawdown (%)",
    ))
    _annotate(fig, f"Max Drawdown: {max_dd:.2f}%  |  Final Balance: ${eq[-1]:,.0f}")
    return fig


def _chart_running_metrics(study) -> Optional[go.Figure]:
    """Winrate, profit factor y ratio acumulados a lo largo de las operaciones."""
    bt = _best_trial(study)
    if bt is None:
        return None
    eq = _get_equity(bt)
    if len(eq) < 10:
        return None

    initial = 1000.0
    pnl = [eq[0] - initial] + [eq[i] - eq[i-1] for i in range(1, len(eq))]
    n = len(pnl)

    run_wr  = []
    run_pf  = []
    run_exp = []
    wins_cum = 0
    losses_cum = 0
    win_sum = 0.0
    loss_sum = 0.0

    for i, p in enumerate(pnl):
        if p > 0:
            wins_cum += 1
            win_sum  += p
        elif p < 0:
            losses_cum += 1
            loss_sum   += abs(p)
        total = wins_cum + losses_cum
        run_wr.append(100 * wins_cum / total if total > 0 else 50)
        run_pf.append(win_sum / loss_sum if loss_sum > 0 else float("nan"))
        avg_w = win_sum / wins_cum if wins_cum > 0 else 0
        avg_l = loss_sum / losses_cum if losses_cum > 0 else 0
        wr = wins_cum / total if total > 0 else 0.5
        run_exp.append(wr * avg_w - (1 - wr) * avg_l)

    idx = list(range(n))
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.34, 0.33, 0.33], vertical_spacing=0.04,
        subplot_titles=["Running Win Rate (%)", "Running Profit Factor", "Running Expectancy ($)"],
    )
    fig.add_trace(go.Scatter(x=idx, y=run_wr, mode="lines",
                             line=dict(color=_ACCENT, width=1.5), showlegend=False,
                             hovertemplate="Trade %{x}<br>WR: %{y:.1f}%<extra></extra>"), row=1, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color=_TEXT_MUT, line_width=1, row=1, col=1)

    pf_vals = [v if v == v else None for v in run_pf]
    fig.add_trace(go.Scatter(x=idx, y=pf_vals, mode="lines",
                             line=dict(color=_GREEN, width=1.5), showlegend=False,
                             hovertemplate="Trade %{x}<br>PF: %{y:.2f}<extra></extra>"), row=2, col=1)
    fig.add_hline(y=1, line_dash="dot", line_color=_TEXT_MUT, line_width=1, row=2, col=1)

    fig.add_trace(go.Scatter(x=idx, y=run_exp, mode="lines",
                             line=dict(color=_YELLOW, width=1.5), showlegend=False,
                             hovertemplate="Trade %{x}<br>Exp: $%{y:.2f}<extra></extra>"), row=3, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color=_TEXT_MUT, line_width=1, row=3, col=1)

    fig.update_layout(**_layout(title=f"Best Trial #{bt.number} — Running Statistics", height=520))
    fig.update_xaxes(gridcolor=_GRID, linecolor=_GRID, color=_TEXT_MUT)
    fig.update_yaxes(gridcolor=_GRID, linecolor=_GRID, color=_TEXT_MUT)
    for ann in fig.layout.annotations:
        ann.font.color = _TEXT_MUT
    return fig


def _chart_top_trials_comparison(study, top_n: int = 10) -> Optional[go.Figure]:
    """Equity curves de los TOP N trials superpuestas."""
    completed = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value, reverse=True,
    )[:top_n]
    if len(completed) < 2:
        return None

    fig = go.Figure()
    for i, trial in enumerate(completed):
        eq = _get_equity(trial)
        if not eq:
            continue
        is_best = (i == 0)
        color = _ACCENT if is_best else f"rgba(88,166,255,{max(0.1, 0.6 - i*0.05):.2f})"
        width = 2.0 if is_best else 1.0
        fig.add_trace(go.Scatter(
            x=list(range(len(eq))), y=eq, mode="lines",
            line=dict(color=color, width=width),
            name=f"#{trial.number} ({trial.value:.0f})" if is_best else f"#{trial.number}",
            opacity=1.0 if is_best else 0.6,
            hovertemplate=f"Trial {trial.number}<br>Score {trial.value:.1f}<br>Balance: $%{{y:,.2f}}<extra></extra>",
        ))

    fig.add_hline(y=1000, line_dash="dot", line_color=_TEXT_MUT, line_width=1)
    fig.update_layout(**_layout(
        title=f"Top {top_n} Trials — Equity Curve Comparison",
        xaxis_title="Trade #", yaxis_title="Balance ($)",
    ))
    _annotate(fig, f"Best trial #{completed[0].number}  Score {completed[0].value:.1f}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN D — COMPARATIVAS POR ACTIVO (solo estudio global)
# ─────────────────────────────────────────────────────────────────────────────

_METRICS_GLOBAL = [
    ("final_score",    "Final Score",       False),
    ("sharpe",         "Sharpe Ratio",      False),
    ("roi",            "ROI (%)",           False),
    ("drawdown",       "Drawdown (%)",      True),   # True = lower is better
    ("profit_factor",  "Profit Factor",     False),
    ("winrate",        "Win Rate (%)",      False),
    ("trades_por_dia", "Trades per Day",    False),
]

def _collect_by_activo(study, metric_key: str) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for t in study.trials:
        activo = t.user_attrs.get("__activo", "")
        if not activo:
            continue
        # final_score is the objective value
        if metric_key == "final_score":
            val = t.value
        else:
            m = _get_metricas(t)
            val = m.get(metric_key, None)
        if val is None:
            continue
        try:
            v = float(val)
            if v == v:
                result.setdefault(activo.upper(), []).append(v)
        except Exception:
            pass
    return result


def _chart_metric_box(study, metric_key: str, label: str, lower_better: bool) -> Optional[go.Figure]:
    """Box plot de una métrica por activo."""
    by_asset = _collect_by_activo(study, metric_key)
    if len(by_asset) < 2:
        return None

    fig = go.Figure()
    for i, (asset, vals) in enumerate(sorted(by_asset.items())):
        color = _ASSET_COLORS[i % len(_ASSET_COLORS)]
        fig.add_trace(go.Box(
            y=vals, name=asset, boxmean="sd",
            marker_color=color, line_color=color,
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.3)",
            hovertemplate=f"<b>{asset}</b><br>{label}: %{{y:.3f}}<extra></extra>",
        ))

    direction = "↓ lower is better" if lower_better else "↑ higher is better"
    fig.update_layout(**_layout(
        title=f"{label} by Asset  ({direction})",
        yaxis_title=label,
        xaxis_title="Asset",
        showlegend=False,
    ))
    return fig


def _chart_radar_by_activo(study) -> Optional[go.Figure]:
    """Radar comparativo de métricas normalizadas por activo."""
    radar_metrics = ["sharpe", "roi", "winrate", "profit_factor"]
    radar_labels  = ["Sharpe", "ROI", "Win Rate", "Profit Factor"]

    # Calcular mediana por activo
    asset_medians: dict[str, list[float]] = {}
    for t in study.trials:
        activo = t.user_attrs.get("__activo", "")
        if not activo:
            continue
        m = _get_metricas(t)
        vals = [float(m.get(k, 0)) for k in radar_metrics]
        asset_medians.setdefault(activo.upper(), []).append(vals)

    if len(asset_medians) < 2:
        return None

    # Mediana para cada activo
    medians = {}
    for asset, rows in asset_medians.items():
        n = len(rows[0])
        med = []
        for col in range(n):
            col_vals = sorted([r[col] for r in rows])
            mid = len(col_vals) // 2
            med.append(col_vals[mid] if len(col_vals) % 2 else (col_vals[mid-1]+col_vals[mid])/2)
        medians[asset] = med

    # Normalizar 0-1 por columna
    n_cols = len(radar_metrics)
    col_min = [min(medians[a][c] for a in medians) for c in range(n_cols)]
    col_max = [max(medians[a][c] for a in medians) for c in range(n_cols)]
    col_rng = [col_max[c] - col_min[c] for c in range(n_cols)]

    def _norm(val, c):
        return (val - col_min[c]) / col_rng[c] if col_rng[c] > 1e-9 else 0.5

    categories = radar_labels + [radar_labels[0]]  # cerrar el polígono

    fig = go.Figure()
    for i, (asset, vals) in enumerate(sorted(medians.items())):
        color = _ASSET_COLORS[i % len(_ASSET_COLORS)]
        normed = [_norm(vals[c], c) for c in range(n_cols)]
        normed_closed = normed + [normed[0]]
        fig.add_trace(go.Scatterpolar(
            r=normed_closed, theta=categories,
            fill="toself",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)",
            line_color=color, name=asset,
            hovertemplate="<b>" + asset + "</b><br>%{theta}: %{r:.2f}<extra></extra>",
        ))

    fig.update_layout(**_layout(
        title="Asset Comparison — Normalized Metrics Radar",
        polar=dict(
            bgcolor=_BG_PLOT,
            radialaxis=dict(visible=True, range=[0, 1], color=_TEXT_MUT, gridcolor=_GRID),
            angularaxis=dict(color=_TEXT_MUT, gridcolor=_GRID),
        ),
    ))
    return fig


def _chart_score_heatmap_assets(study) -> Optional[go.Figure]:
    """Heatmap de score medio por activo (para global study)."""
    by_asset = _collect_by_activo(study, "final_score")
    if len(by_asset) < 2:
        return None

    assets  = sorted(by_asset.keys())
    metrics = ["final_score", "sharpe", "roi", "drawdown", "winrate", "profit_factor"]
    labels  = ["Score", "Sharpe", "ROI%", "Drawdown%", "WinRate%", "PF"]

    # Mediana de cada métrica por activo
    z = []
    for mk in metrics:
        row = []
        d = _collect_by_activo(study, mk)
        for asset in assets:
            vals = sorted(d.get(asset, [0]))
            mid = len(vals) // 2
            med = vals[mid] if vals else 0
            row.append(med)
        z.append(row)

    # Normalizar cada fila a 0-1
    z_norm = []
    for row in z:
        mn, mx = min(row), max(row)
        rng = mx - mn
        z_norm.append([(v - mn) / rng if rng > 1e-9 else 0.5 for v in row])

    text_vals = [[f"{z[ri][ci]:.2f}" for ci in range(len(assets))] for ri in range(len(metrics))]

    fig = go.Figure(go.Heatmap(
        z=z_norm, x=assets, y=labels,
        colorscale=[[0, _RED], [0.5, _BG_PLOT], [1.0, _GREEN]],
        text=text_vals, texttemplate="%{text}",
        hovertemplate="<b>%{y} — %{x}</b><br>Value: %{text}<extra></extra>",
        showscale=False,
    ))
    fig.update_layout(**_layout(title="Cross-Asset Metrics Heatmap (normalized)", height=350))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def generate_charts_for_study(study, *, is_global: bool = False) -> int:
    """Genera y guarda TODAS las gráficas para un estudio Optuna.

    Retorna el número de gráficas guardadas correctamente.
    """
    from optuna.exceptions import ExperimentalWarning
    warnings.filterwarnings("ignore", category=ExperimentalWarning)

    n_complete = len([t for t in study.trials if t.value is not None])
    if n_complete == 0:
        return 0

    saved = 0

    def _try_save(fn, gid):
        nonlocal saved
        try:
            fig = fn()
            if fig is not None:
                _save(study, fig, gid)
                saved += 1
        except Exception:
            pass

    # ── A. Nativas Optuna ────────────────────────────────────────────────────
    saved += _section_a(study)

    # ── B. Visión global de todos los trials ─────────────────────────────────
    _try_save(lambda: _chart_score_distribution(study),   "score_distribution")
    _try_save(lambda: _chart_score_landscape(study),      "score_landscape")
    _try_save(lambda: _chart_metrics_scatter_matrix(study),"metrics_scatter")
    _try_save(lambda: _chart_trial_correlation(study),    "metrics_correlation")

    # ── C. Análisis del mejor trial ──────────────────────────────────────────
    _try_save(lambda: _chart_equity_curve(study),         "best_equity_curve")
    _try_save(lambda: _chart_pnl_per_trade(study),        "best_pnl_trades")
    _try_save(lambda: _chart_pnl_distribution(study),     "best_pnl_dist")
    _try_save(lambda: _chart_drawdown_periods(study),     "best_drawdown")
    _try_save(lambda: _chart_running_metrics(study),      "best_running_metrics")
    _try_save(lambda: _chart_top_trials_comparison(study),"top_trials_equity")

    # ── D. Comparativas por activo (solo estudio global) ─────────────────────
    if is_global:
        for mk, label, lower in _METRICS_GLOBAL:
            _try_save(
                lambda mk=mk, label=label, lower=lower: _chart_metric_box(study, mk, label, lower),
                _safe_id(f"global_box_{mk}"),
            )
        _try_save(lambda: _chart_radar_by_activo(study),      "global_radar")
        _try_save(lambda: _chart_score_heatmap_assets(study), "global_heatmap")

    return saved


def generate_all_charts_for_strategy(
    strategy_id: Optional[int] = None,
    *,
    base_dir: Optional[str] = None,
) -> int:
    """Genera gráficas para todos los estudios del DB de una estrategia.

    Retorna el número total de gráficas guardadas.
    """
    try:
        import optuna
    except ImportError:
        return 0

    from modelox.optimizers.storage import get_database_file_path
    import os

    try:
        sid = int(strategy_id) if strategy_id is not None else None
    except Exception:
        sid = None

    db_name = f"ID{sid}.db" if sid and sid > 0 else "optuna.db"
    db_path = get_database_file_path(db_name, base_dir=base_dir, create_dir=False)
    if not os.path.exists(db_path):
        return 0

    storage_url = f"sqlite:///{db_path}"

    try:
        all_names = optuna.get_all_study_names(storage=storage_url)
    except Exception:
        return 0

    total = 0
    for name in all_names:
        try:
            study     = optuna.load_study(study_name=name, storage=storage_url)
            is_global = name.endswith("-global") or name.endswith("_global")
            total    += generate_charts_for_study(study, is_global=is_global)
        except Exception:
            pass

    return total
