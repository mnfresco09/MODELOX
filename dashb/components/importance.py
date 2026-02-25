# ─────────────────────────────────────────────
#  Optuna Dashboard Pro — components/importance.py
# ─────────────────────────────────────────────
from __future__ import annotations

import sys, os
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from assets.theme import BG, SURFACE, BORDER, TEXT, TEXT_DIM, LAYOUT, GRADIENT


def build_importance_figure(importance: dict[str, float]) -> go.Figure:
    if not importance:
        return _empty("Not enough completed trials to compute importance")

    items  = sorted(importance.items(), key=lambda x: x[1])
    params = [k for k, _ in items]
    values = [v for _, v in items]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values, y=params,
        orientation="h",
        marker=dict(
            color=values,
            colorscale=GRADIENT,
            cmin=0, cmax=max(values) if values else 1,
            line=dict(width=0),
        ),
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        textfont=dict(color=TEXT_DIM, size=11, family="DM Mono, monospace"),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor=LAYOUT["paper_bgcolor"],
        plot_bgcolor=LAYOUT["plot_bgcolor"],
        font=LAYOUT["font"],
        margin=LAYOUT["margin"],
        hoverlabel=LAYOUT["hoverlabel"],
        title=dict(
            text="<b>Parameter Importance</b>",
            font=dict(size=14, color=TEXT),
            x=0.02, xanchor="left",
        ),
        xaxis=dict(
            gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER,
            tickfont=dict(color=TEXT_DIM),
            title=dict(text="Importance Score", font=dict(color=TEXT_DIM, size=12)),
            range=[0, max(values) * 1.18] if values else [0, 1],
        ),
        yaxis=dict(
            gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER,
            tickfont=dict(color=TEXT, size=12, family="DM Mono, monospace"),
        ),
        height=max(300, 60 + len(params) * 42),
        bargap=0.35,
        showlegend=False,
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
        paper_bgcolor=SURFACE, plot_bgcolor=BG, height=300,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


empty_figure = _empty