# ─────────────────────────────────────────────
#  Optuna Dashboard Pro — components/rank.py
#  Rank scatter: dots colored by percentile rank
#  blue = worst rank, red = best rank
# ─────────────────────────────────────────────
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from assets.theme import (BG, SURFACE, SURFACE_2, BORDER, TEXT, TEXT_DIM, ACCENT, LAYOUT, pick_gradient)
from data.loader import compute_contour_data, _resolve_col

# Rank colorscale: blue (rank 0 = worst) → red (rank 1 = best)
RANK_COLORSCALE = [
    [0.00, "#1a3a6b"],
    [0.15, "#2166ac"],
    [0.30, "#4dac26"],
    [0.50, "#b8e186"],
    [0.65, "#f1b6da"],
    [0.80, "#e7298a"],
    [1.00, "#c0392b"],
]


def build_rank_figure(
    df: pd.DataFrame,
    x_param: str,
    y_param: str,
    metric: str,
    direction: str = "minimize",
    aggregate: bool = True,
) -> go.Figure:

    if df is None or df.empty:
        return _empty("No complete trials available")

    # ── Get data ──────────────────────────────────────────────────────────────
    if aggregate:
        grp = compute_contour_data(df, x_param, y_param, metric)
        if grp.empty:
            return _empty(f"No data for  {x_param}  ×  {y_param}")
        xs     = grp["x"].values.astype(float)
        ys     = grp["y"].values.astype(float)
        zs     = grp["z"].values.astype(float)
        counts = grp["count"].values
        labels = [
            f"<b>{x_param}</b>: {grp.iloc[i]['x']}<br>"
            f"<b>{y_param}</b>: {grp.iloc[i]['y']}<br>"
            f"<b>{metric}</b>: {zs[i]:.5f}<br>"
            f"<b>Trials</b>: {int(counts[i])}"
            + ("<br><i>mean of duplicates</i>" if counts[i] > 1 else "")
            for i in range(len(grp))
        ]
    else:
        x_col = _resolve_col(df, x_param)
        y_col = _resolve_col(df, y_param)
        z_col = _resolve_col(df, metric)
        sub   = df[[x_col, y_col, z_col, "trial_number"]].dropna()
        if sub.empty:
            return _empty(f"No data for  {x_param}  ×  {y_param}")
        xs = sub[x_col].values.astype(float)
        ys = sub[y_col].values.astype(float)
        zs = sub[z_col].values.astype(float)
        labels = [
            f"<b>Trial #{int(sub.iloc[i]['trial_number'])}</b><br>"
            f"<b>{x_param}</b>: {xs[i]}<br>"
            f"<b>{y_param}</b>: {ys[i]}<br>"
            f"<b>{metric}</b>: {zs[i]:.5f}"
            for i in range(len(xs))
        ]

    # ── Compute rank (0=worst, 1=best) ─────────────────────────────────────────
    n = len(zs)
    order = np.argsort(zs)
    ranks = np.empty(n, dtype=float)
    if direction == "maximize":
        # Highest z → rank 1
        for pos, idx in enumerate(order):
            ranks[idx] = pos / (n - 1) if n > 1 else 0.5
    else:
        # Lowest z → rank 1
        for pos, idx in enumerate(order[::-1]):
            ranks[idx] = pos / (n - 1) if n > 1 else 0.5

    # ── Dot sizes: slightly bigger for high rank ────────────────────────────────
    dot_sizes = 7 + ranks * 6   # 7–13 px

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers",
        marker=dict(
            color=ranks,
            colorscale=RANK_COLORSCALE,
            cmin=0, cmax=1,
            size=dot_sizes.tolist(),
            showscale=True,
            colorbar=dict(
                title=dict(text="Rank", font=dict(color=TEXT_DIM, size=11)),
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=["0.0 (worst)", "0.25", "0.5", "0.75", "1.0 (best)"],
                tickfont=dict(color=TEXT_DIM, size=10),
                bgcolor=SURFACE_2, bordercolor=BORDER, borderwidth=1,
                outlinewidth=0, thickness=14, len=0.85,
            ),
            line=dict(width=0.8, color="rgba(255,255,255,0.20)"),
            opacity=0.92,
        ),
        text=labels,
        hovertemplate="%{text}<br><b>Rank</b>: %{marker.color:.3f}<extra></extra>",
        showlegend=False,
    ))

    # Best star
    best_i = int(np.argmax(ranks))
    fig.add_trace(go.Scatter(
        x=[xs[best_i]], y=[ys[best_i]], mode="markers",
        marker=dict(symbol="star", size=22, color=ACCENT,
                    line=dict(width=1.5, color="white"), opacity=1.0),
        hovertemplate=(
            f"<b>★ Best rank</b><br>{x_param}: {xs[best_i]}<br>"
            f"{y_param}: {ys[best_i]}<br>{metric}: {zs[best_i]:.5f}<br>"
            f"Rank: {ranks[best_i]:.3f}<extra></extra>"
        ),
        showlegend=False,
    ))

    agg_label = "mean-aggregated" if aggregate else "individual trials"
    fig.update_layout(
        paper_bgcolor=LAYOUT["paper_bgcolor"],
        plot_bgcolor=BG,
        font=LAYOUT["font"],
        margin=dict(l=60, r=30, t=52, b=56),
        hoverlabel=LAYOUT["hoverlabel"],
        title=dict(
            text=(
                f"<b>Rank</b>  ·  {x_param}  ×  {y_param}  —  {metric}"
                f"  <span style='color:{TEXT_DIM};font-size:11px;'>({agg_label})</span>"
            ),
            font=dict(size=14, color=TEXT), x=0.01, xanchor="left",
        ),
        xaxis=dict(
            showgrid=True, gridcolor="rgba(36,48,80,0.25)",
            linecolor=BORDER, zerolinecolor="rgba(0,0,0,0)",
            tickfont=dict(color=TEXT_DIM),
            title=dict(text=x_param, font=dict(color=TEXT_DIM, size=12)),
        ),
        yaxis=dict(
            showgrid=True, gridcolor="rgba(36,48,80,0.25)",
            linecolor=BORDER, zerolinecolor="rgba(0,0,0,0)",
            tickfont=dict(color=TEXT_DIM),
            title=dict(text=y_param, font=dict(color=TEXT_DIM, size=12)),
        ),
        height=560, hovermode="closest", showlegend=False,
    )
    return fig


def _empty(msg: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(color=TEXT_DIM, size=13, family="DM Mono, monospace"))
    fig.update_layout(
        paper_bgcolor=SURFACE, plot_bgcolor=BG, height=560,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=20, b=20))
    return fig

empty_figure = _empty