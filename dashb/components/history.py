# ─────────────────────────────────────────────
#  Optuna Dashboard Pro — components/history.py
# ─────────────────────────────────────────────
from __future__ import annotations

import sys, os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from assets.theme import BG, SURFACE, BORDER, TEXT, TEXT_DIM, ACCENT, SUCCESS, LAYOUT


def build_history_figure(df: pd.DataFrame, direction: str = "minimize") -> go.Figure:
    if df is None or df.empty:
        return _empty("No trials to display")

    df = df.sort_values("trial_number").copy()

    if "value" not in df.columns or df["value"].isna().all():
        return _empty("No metric values found")

    vals       = df["value"].values.astype(float)
    trial_nums = df["trial_number"].values

    if direction == "minimize":
        running_best = np.minimum.accumulate(vals)
        best_color   = SUCCESS
    else:
        running_best = np.maximum.accumulate(vals)
        best_color   = ACCENT

    fig = go.Figure()

    # All trials scatter
    fig.add_trace(go.Scatter(
        x=trial_nums, y=vals,
        mode="markers",
        name="Trial value",
        marker=dict(color="#1e2740", size=6,
                    line=dict(width=1, color=BORDER), opacity=0.75),
        hovertemplate="Trial #%{x}<br>Value: %{y:.5f}<extra></extra>",
    ))

    # Running best line
    fig.add_trace(go.Scatter(
        x=trial_nums, y=running_best,
        mode="lines",
        name="Best so far",
        line=dict(color=best_color, width=2.5, shape="hv"),
        hovertemplate="Trial #%{x}<br>Best: %{y:.5f}<extra></extra>",
    ))

    # Best star
    best_idx = int(np.argmin(running_best) if direction == "minimize" else np.argmax(running_best))
    fig.add_trace(go.Scatter(
        x=[trial_nums[best_idx]], y=[running_best[best_idx]],
        mode="markers",
        name="Best",
        marker=dict(symbol="star", size=16, color=best_color,
                    line=dict(width=1.5, color="white")),
        hovertemplate=f"★ Best trial #%{{x}}<br>Value: %{{y:.5f}}<extra></extra>",
    ))

    fig.add_annotation(
        x=trial_nums[best_idx], y=running_best[best_idx],
        text=f"  {running_best[best_idx]:.4f}",
        showarrow=False,
        font=dict(color=best_color, size=11, family="DM Mono, monospace"),
        xanchor="left",
    )

    fig.update_layout(
        paper_bgcolor=LAYOUT["paper_bgcolor"],
        plot_bgcolor=LAYOUT["plot_bgcolor"],
        font=LAYOUT["font"],
        margin=LAYOUT["margin"],
        hoverlabel=LAYOUT["hoverlabel"],
        legend=dict(
            **LAYOUT["legend"],
            orientation="h",
            x=0.01, y=1.08, xanchor="left",
        ),
        title=dict(
            text="<b>Optimization History</b>",
            font=dict(size=14, color=TEXT),
            x=0.02, xanchor="left",
        ),
        xaxis=dict(
            gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER,
            tickfont=dict(color=TEXT_DIM),
            title=dict(text="Trial Number", font=dict(color=TEXT_DIM, size=12)),
        ),
        yaxis=dict(
            gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER,
            tickfont=dict(color=TEXT_DIM),
            title=dict(text="Objective Value", font=dict(color=TEXT_DIM, size=12)),
        ),
        height=440,
        hovermode="x unified",
    )

    return fig


def _empty(msg: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(color=TEXT_DIM, size=13, family="DM Mono, monospace"),
    )
    fig.update_layout(
        paper_bgcolor=SURFACE, plot_bgcolor=BG, height=440,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


empty_figure = _empty