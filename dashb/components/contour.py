# ─────────────────────────────────────────────
#  Optuna Dashboard Pro — components/contour.py
# ─────────────────────────────────────────────
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import griddata

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from assets.theme import (BG, SURFACE, SURFACE_2, BORDER,
                           TEXT, TEXT_DIM, ACCENT, LAYOUT, pick_gradient)
from data.loader import compute_contour_data, _resolve_col


def _gaussian_rbf_surface(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    nx: int = 200,
    ny: int = 200,
    bandwidth_factor: float = 0.12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mantengo esta función por compatibilidad. 
    Es requerida por robustness.py y otros componentes.
    """
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_range = x_max - x_min if x_max != x_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0

    xi = np.linspace(x_min, x_max, nx)
    yi = np.linspace(y_min, y_max, ny)
    Xi, Yi = np.meshgrid(xi, yi)

    Xi_n = (Xi - x_min) / x_range
    Yi_n = (Yi - y_min) / y_range
    xs_n = (xs - x_min) / x_range
    ys_n = (ys - y_min) / y_range
    bw   = bandwidth_factor

    dx = Xi_n[:, :, np.newaxis] - xs_n[np.newaxis, np.newaxis, :]
    dy = Yi_n[:, :, np.newaxis] - ys_n[np.newaxis, np.newaxis, :]
    W  = np.exp(-0.5 * ((dx / bw) ** 2 + (dy / bw) ** 2))

    W_sum = W.sum(axis=2)
    Zi    = np.einsum("ijk,k->ij", W, zs) / np.where(W_sum > 1e-10, W_sum, np.nan)
    return xi, yi, Zi


def build_contour_figure(
    df: pd.DataFrame,
    x_param: str,
    y_param: str,
    metric: str,
    direction: str = "minimize",
    aggregate: bool = True,        # True = mean per cell, False = all individual points
) -> go.Figure:

    if df is None or df.empty:
        return _empty("No complete trials available")

    colorscale = pick_gradient(direction)

    # ── Get data ──────────────────────────────────────────────────────────────
    if aggregate:
        grp = compute_contour_data(df, x_param, y_param, metric)
        if grp.empty:
            return _empty(f"No data for  {x_param}  ×  {y_param}")
        xs = grp["x"].values.astype(float)
        ys = grp["y"].values.astype(float)
        zs = grp["z"].values.astype(float)
        counts = grp["count"].values
    else:
        # Raw individual trials
        x_col = _resolve_col(df, x_param)
        y_col = _resolve_col(df, y_param)
        z_col = _resolve_col(df, metric)
        sub   = df[[x_col, y_col, z_col, "trial_number"]].dropna()
        if sub.empty:
            return _empty(f"No data for  {x_param}  ×  {y_param}")
        xs     = sub[x_col].values.astype(float)
        ys     = sub[y_col].values.astype(float)
        zs     = sub[z_col].values.astype(float)
        counts = np.ones(len(xs), dtype=int)

    z_min = float(zs.min())
    z_max = float(zs.max())

    fig = go.Figure()

    # ── 1. Scipy Griddata Interpolation (Optuna Style) ────────────────────────
    # Creamos una malla uniforme 100x100
    grid_size = 100
    xi = np.linspace(xs.min(), xs.max(), grid_size)
    yi = np.linspace(ys.min(), ys.max(), grid_size)
    Xi, Yi = np.meshgrid(xi, yi)

    try:
        # Interpolación lineal (la estándar de Optuna)
        Zi = griddata((xs, ys), zs, (Xi, Yi), method='linear')
        
        # Rellenamos los posibles huecos NaN (bordes) con el valor más cercano 
        # para que el contorno llegue hasta los bordes del gráfico
        if np.isnan(Zi).any():
            Zi_nearest = griddata((xs, ys), zs, (Xi, Yi), method='nearest')
            Zi[np.isnan(Zi)] = Zi_nearest[np.isnan(Zi)]
            
    except Exception as exc:
        print(f"[contour interpolation] {exc}")
        return _empty("Interpolation failed for contour plot")

    # ── 2. Contour Map ────────────────────────────────────────────────────────
    fig.add_trace(go.Contour(
        x=xi, y=yi, z=Zi,
        colorscale=colorscale,
        zmin=z_min, zmax=z_max,
        showscale=True,
        contours=dict(
            coloring="heatmap", # Mismo estilo de coloreado suave que usa Optuna
            showlines=True,     # Muestra las líneas topográficas
        ),
        line=dict(width=0.5, color="rgba(0,0,0,0.25)"), # Líneas sutiles y elegantes
        colorbar=dict(
            title=dict(text=metric, font=dict(color=TEXT_DIM, size=11)),
            tickfont=dict(color=TEXT_DIM, size=10),
            bgcolor=SURFACE_2, bordercolor=BORDER, borderwidth=1,
            outlinewidth=0, thickness=14, len=0.85,
        ),
        hoverinfo="skip",
    ))

    # ── 3. Data points (Scatter dots) ─────────────────────────────────────────
    if aggregate:
        hover_texts = [
            (
                f"<b>{x_param}</b>: {grp.iloc[i]['x']}<br>"
                f"<b>{y_param}</b>: {grp.iloc[i]['y']}<br>"
                f"<b>{metric}</b>: {grp.iloc[i]['z']:.5f}<br>"
                f"<b>Trials</b>: {int(grp.iloc[i]['count'])}"
                + ("<br><i>mean of duplicates</i>" if grp.iloc[i]['count'] > 1 else "")
            )
            for i in range(len(grp))
        ]
    else:
        hover_texts = [
            f"<b>Trial #{int(sub.iloc[i]['trial_number'])}</b><br>"
            f"<b>{x_param}</b>: {xs[i]}<br>"
            f"<b>{y_param}</b>: {ys[i]}<br>"
            f"<b>{metric}</b>: {zs[i]:.5f}"
            for i in range(len(xs))
        ]

    # Puntos profesionales: semi-transparentes oscuros con borde fino claro
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(
            color="rgba(30, 30, 30, 0.6)", 
            size=6, 
            line=dict(width=0.5, color="rgba(255, 255, 255, 0.7)")
        ),
        text=hover_texts,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))

    # ── 4. Layout configuration ───────────────────────────────────────────────
    agg_label = "mean-aggregated" if aggregate else "individual trials"
    title_dir = "▼ minimize" if direction == "minimize" else "▲ maximize"
    fig.update_layout(
        paper_bgcolor=LAYOUT["paper_bgcolor"],
        plot_bgcolor=BG,
        font=LAYOUT["font"],
        margin=dict(l=60, r=30, t=52, b=56),
        hoverlabel=LAYOUT["hoverlabel"],
        title=dict(
            text=(
                f"<b>{x_param}</b>  ×  <b>{y_param}</b>  —  {metric}"
                f"  <span style='color:{TEXT_DIM};font-size:11px;'>({title_dir} · {agg_label})</span>"
            ),
            font=dict(size=14, color=TEXT), x=0.01, xanchor="left",
        ),
        xaxis=dict(
            showgrid=False, linecolor=BORDER, zerolinecolor="rgba(0,0,0,0)",
            tickfont=dict(color=TEXT_DIM),
            title=dict(text=x_param, font=dict(color=TEXT_DIM, size=12)),
        ),
        yaxis=dict(
            showgrid=False, linecolor=BORDER, zerolinecolor="rgba(0,0,0,0)",
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