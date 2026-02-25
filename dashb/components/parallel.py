# ─────────────────────────────────────────────
#  Optuna Dashboard Pro — components/parallel.py
# ─────────────────────────────────────────────
from __future__ import annotations

import sys, os
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from assets.theme import (BG, SURFACE, SURFACE_2, BORDER, TEXT, TEXT_DIM,
                           LAYOUT, pick_gradient)


def build_parallel_figure(
    df: pd.DataFrame,
    selected_params: list[str],
    metric: str = "value",
    direction: str = "minimize",
) -> go.Figure:

    if df is None or df.empty or not selected_params:
        return _empty("Select parameters above")

    colorscale = pick_gradient(direction)
    metric_col = metric if metric in df.columns else "value"

    dimensions = []

    # ── Metric axis first ──────────────────────────────────────────────────────
    if metric_col in df.columns:
        vals = df[metric_col].dropna()
        dimensions.append(go.parcoords.Dimension(
            label=f"◈ {metric_col}",
            values=df[metric_col],
            range=[float(vals.min()), float(vals.max())],
        ))

    # ── Parameter axes ─────────────────────────────────────────────────────────
    for param in selected_params:
        col = f"param_{param}" if f"param_{param}" in df.columns else param
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue

        if pd.api.types.is_numeric_dtype(series):
            dimensions.append(go.parcoords.Dimension(
                label=param,
                values=df[col],
                range=[float(series.min()), float(series.max())],
            ))
        else:
            # Categorical → encode as integers
            categories = sorted(series.unique().tolist())
            mapping = {v: i for i, v in enumerate(categories)}
            encoded  = df[col].map(mapping)
            dimensions.append(go.parcoords.Dimension(
                label=param,
                values=encoded,
                tickvals=list(mapping.values()),
                ticktext=list(mapping.keys()),
                range=[0, len(categories) - 1],
            ))

    if not dimensions:
        return _empty("No valid parameter columns found")

    line_color = df[metric_col] if metric_col in df.columns else None

    fig = go.Figure(go.Parcoords(
        line=dict(
            color=line_color,
            colorscale=colorscale,
            showscale=True,
            cmin=float(df[metric_col].min()) if metric_col in df.columns else 0,
            cmax=float(df[metric_col].max()) if metric_col in df.columns else 1,
            colorbar=dict(
                title=dict(text=metric_col, font=dict(color=TEXT_DIM, size=11)),
                tickfont=dict(color=TEXT_DIM, size=10),
                bgcolor=SURFACE_2,
                bordercolor=BORDER,
                borderwidth=1,
                outlinewidth=0,
                thickness=14,
            ),
        ),
        dimensions=dimensions,
        labelfont=dict(color=TEXT_DIM, size=11, family="DM Mono, monospace"),
        tickfont=dict(color=TEXT_DIM, size=9, family="DM Mono, monospace"),
        rangefont=dict(color=TEXT_DIM, size=9, family="DM Mono, monospace"),
    ))

    fig.update_layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=BG,
        font=dict(color=TEXT, family="DM Mono, monospace", size=11),
        title=dict(
            text="<b>Parallel Coordinates</b>  <span style='color:#6878a8;font-size:11px;'>"
                 "Drag axes to reorder · Drag on axis to filter</span>",
            font=dict(size=14, color=TEXT),
            x=0.02, xanchor="left",
        ),
        height=500,
        margin=dict(l=60, r=60, t=60, b=40),
        hoverlabel=LAYOUT["hoverlabel"],
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
        paper_bgcolor=SURFACE, plot_bgcolor=BG, height=500,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


empty_figure = _empty