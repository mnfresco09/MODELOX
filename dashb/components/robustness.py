# ─────────────────────────────────────────────
#  Optuna Dashboard Pro — components/robustness.py
#
#  Three robustness analyses:
#  1. Std Map       — volatility of metric at each (x,y)
#  2. Neighborhood  — k-nearest-neighbor mean score per point
#  3. Plateau Index — high score + low std = robust plateau
# ─────────────────────────────────────────────
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import griddata

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from assets.theme import (BG, SURFACE, SURFACE_2, BORDER,
                           TEXT, TEXT_DIM, ACCENT, SUCCESS, WARNING, DANGER, LAYOUT, pick_gradient)
from data.loader import _resolve_col, compute_contour_data


# ─────────────────────────────────────────────────────────────────────────────
#  1. Std Map
# ─────────────────────────────────────────────────────────────────────────────

def build_std_map(
    df: pd.DataFrame,
    x_param: str,
    y_param: str,
    metric: str,
) -> go.Figure:
    """
    Heatmap of metric standard deviation at each (x, y).
    Low std = consistent/robust region, high std = volatile.
    """
    x_col = _resolve_col(df, x_param)
    y_col = _resolve_col(df, y_param)
    z_col = _resolve_col(df, metric)

    sub = df[[x_col, y_col, z_col]].dropna()
    if sub.empty or len(sub) < 4:
        return _empty("Not enough data for std map (need ≥4 trials)")

    grp = sub.groupby([x_col, y_col])[z_col].agg(
        mean_val="mean", std_val="std", count="count"
    ).reset_index()

    # Only keep cells with ≥2 trials for std to be meaningful;
    # single-trial cells get std=0 (stable by assumption)
    grp["std_val"] = grp["std_val"].fillna(0.0)

    xs = grp[x_col].values.astype(float)
    ys = grp[y_col].values.astype(float)
    zs = grp["std_val"].values.astype(float)

    # Low std = good → invert for visual (we want green=low, red=high)
    STD_CS = [
        [0.00, "#00e5a0"],   # teal  — very stable
        [0.25, "#27ae60"],   # green
        [0.50, "#f1c40f"],   # yellow
        [0.75, "#e67e22"],   # orange
        [1.00, "#c0392b"],   # red   — very volatile
    ]

    fig = go.Figure()

    grid_size = 100
    xi = np.linspace(xs.min(), xs.max(), grid_size)
    yi = np.linspace(ys.min(), ys.max(), grid_size)
    Xi, Yi = np.meshgrid(xi, yi)

    try:
        Zi = griddata((xs, ys), zs, (Xi, Yi), method='linear')
        if np.isnan(Zi).any():
            Zi_nearest = griddata((xs, ys), zs, (Xi, Yi), method='nearest')
            Zi[np.isnan(Zi)] = Zi_nearest[np.isnan(Zi)]
            
        fig.add_trace(go.Contour(
            x=xi, y=yi, z=Zi,
            colorscale=STD_CS,
            zmin=zs.min(), zmax=zs.max(),
            showscale=True,
            contours=dict(coloring="heatmap", showlines=True),
            line=dict(width=0.5, color="rgba(0,0,0,0.25)"),
            colorbar=dict(
                title=dict(text=f"Std({metric})", font=dict(color=TEXT_DIM, size=11)),
                tickfont=dict(color=TEXT_DIM, size=10),
                bgcolor=SURFACE_2, bordercolor=BORDER, borderwidth=1,
                outlinewidth=0, thickness=14, len=0.85,
            ),
            hoverinfo="skip",
        ))
    except Exception as exc:
        print(f"[std map interpolation] {exc}")
        return _empty("Interpolation failed for std map")

    # Data points
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(
            color="rgba(30, 30, 30, 0.6)", 
            size=6, 
            line=dict(width=0.5, color="rgba(255, 255, 255, 0.7)")
        ),
        text=[f"<b>{x_param}</b>: {grp.iloc[i][x_col]}<br>"
              f"<b>{y_param}</b>: {grp.iloc[i][y_col]}<br>"
              f"<b>Std</b>: {zs[i]:.5f}<br><b>Mean</b>: {grp.iloc[i]['mean_val']:.5f}<br>"
              f"<b>Trials</b>: {int(grp.iloc[i]['count'])}"
              for i in range(len(grp))],
        hovertemplate="%{text}<extra></extra>", showlegend=False,
    ))

    fig.update_layout(
        paper_bgcolor=LAYOUT["paper_bgcolor"], plot_bgcolor=BG,
        font=LAYOUT["font"], margin=dict(l=60, r=30, t=52, b=56),
        hoverlabel=LAYOUT["hoverlabel"],
        title=dict(
            text=f"<b>Volatility Map</b>  ·  {x_param} × {y_param}  —  Std({metric})",
            font=dict(size=14, color=TEXT), x=0.01, xanchor="left",
        ),
        xaxis=dict(showgrid=False, linecolor=BORDER, zerolinecolor="rgba(0,0,0,0)",
                   tickfont=dict(color=TEXT_DIM),
                   title=dict(text=x_param, font=dict(color=TEXT_DIM, size=12))),
        yaxis=dict(showgrid=False, linecolor=BORDER, zerolinecolor="rgba(0,0,0,0)",
                   tickfont=dict(color=TEXT_DIM),
                   title=dict(text=y_param, font=dict(color=TEXT_DIM, size=12))),
        height=520, hovermode="closest", showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  2. Neighborhood Score
# ─────────────────────────────────────────────────────────────────────────────

def build_neighborhood_figure(
    df: pd.DataFrame,
    x_param: str,
    y_param: str,
    metric: str,
    direction: str = "minimize",
    k: int = 8,
) -> go.Figure:
    """
    For each data point, compute the mean metric of its k nearest neighbors
    (in normalised parameter space). High neighborhood score = robust plateau.
    """
    x_col = _resolve_col(df, x_param)
    y_col = _resolve_col(df, y_param)
    z_col = _resolve_col(df, metric)

    grp = compute_contour_data(df, x_param, y_param, metric)
    if grp.empty or len(grp) < 3:
        return _empty("Not enough data for neighborhood analysis (need ≥3 distinct cells)")

    xs = grp["x"].values.astype(float)
    ys = grp["y"].values.astype(float)
    zs = grp["z"].values.astype(float)

    x_range = xs.ptp() if xs.ptp() > 0 else 1.0
    y_range = ys.ptp() if ys.ptp() > 0 else 1.0
    xs_n = (xs - xs.min()) / x_range
    ys_n = (ys - ys.min()) / y_range

    k_eff = min(k, len(grp) - 1)
    neigh_scores = np.empty(len(grp))

    for i in range(len(grp)):
        dists = np.sqrt((xs_n - xs_n[i]) ** 2 + (ys_n - ys_n[i]) ** 2)
        dists[i] = np.inf
        nn_idx = np.argpartition(dists, k_eff)[:k_eff]
        neigh_scores[i] = zs[nn_idx].mean()

    colorscale = pick_gradient(direction)
    ns_min, ns_max = float(neigh_scores.min()), float(neigh_scores.max())

    fig = go.Figure()

    grid_size = 100
    xi = np.linspace(xs.min(), xs.max(), grid_size)
    yi = np.linspace(ys.min(), ys.max(), grid_size)
    Xi, Yi = np.meshgrid(xi, yi)

    try:
        Zi = griddata((xs, ys), neigh_scores, (Xi, Yi), method='linear')
        if np.isnan(Zi).any():
            Zi_nearest = griddata((xs, ys), neigh_scores, (Xi, Yi), method='nearest')
            Zi[np.isnan(Zi)] = Zi_nearest[np.isnan(Zi)]

        fig.add_trace(go.Contour(
            x=xi, y=yi, z=Zi,
            colorscale=colorscale,
            zmin=ns_min, zmax=ns_max,
            showscale=True,
            contours=dict(coloring="heatmap", showlines=True),
            line=dict(width=0.5, color="rgba(0,0,0,0.25)"),
            colorbar=dict(
                title=dict(text=f"Neighbor mean", font=dict(color=TEXT_DIM, size=11)),
                tickfont=dict(color=TEXT_DIM, size=10),
                bgcolor=SURFACE_2, bordercolor=BORDER, borderwidth=1,
                outlinewidth=0, thickness=14, len=0.85,
            ),
            hoverinfo="skip",
        ))
    except Exception as exc:
        print(f"[neighborhood interpolation] {exc}")
        return _empty("Interpolation failed for neighborhood map")

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(
            color="rgba(30, 30, 30, 0.6)", 
            size=6, 
            line=dict(width=0.5, color="rgba(255, 255, 255, 0.7)")
        ),
        text=[f"<b>{x_param}</b>: {grp.iloc[i]['x']}<br>"
              f"<b>{y_param}</b>: {grp.iloc[i]['y']}<br>"
              f"<b>{metric}</b>: {zs[i]:.5f}<br>"
              f"<b>Neighbor mean</b>: {neigh_scores[i]:.5f}"
              for i in range(len(grp))],
        hovertemplate="%{text}<extra></extra>", showlegend=False,
    ))

    fig.update_layout(
        paper_bgcolor=LAYOUT["paper_bgcolor"], plot_bgcolor=BG,
        font=LAYOUT["font"], margin=dict(l=60, r=30, t=52, b=56),
        hoverlabel=LAYOUT["hoverlabel"],
        title=dict(
            text=f"<b>Neighborhood Score</b>  ·  {x_param} × {y_param}  —  k={k_eff} neighbors",
            font=dict(size=14, color=TEXT), x=0.01, xanchor="left",
        ),
        xaxis=dict(showgrid=False, linecolor=BORDER, zerolinecolor="rgba(0,0,0,0)",
                   tickfont=dict(color=TEXT_DIM),
                   title=dict(text=x_param, font=dict(color=TEXT_DIM, size=12))),
        yaxis=dict(showgrid=False, linecolor=BORDER, zerolinecolor="rgba(0,0,0,0)",
                   tickfont=dict(color=TEXT_DIM),
                   title=dict(text=y_param, font=dict(color=TEXT_DIM, size=12))),
        height=520, hovermode="closest", showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  3. Plateau Index  =  high score  +  low volatility  +  good neighborhood
# ─────────────────────────────────────────────────────────────────────────────

def build_plateau_figure(
    df: pd.DataFrame,
    x_param: str,
    y_param: str,
    metric: str,
    direction: str = "minimize",
    k: int = 8,
) -> go.Figure:
    """
    Composite robustness index per (x, y) cell:
        plateau = rank(mean) * (1 - rank(std)) * rank(neighbor_mean)
    Score is in [0,1]; higher = more robust plateau.
    """
    x_col = _resolve_col(df, x_param)
    y_col = _resolve_col(df, y_param)
    z_col = _resolve_col(df, metric)

    sub = df[[x_col, y_col, z_col]].dropna()
    if sub.empty or len(sub) < 4:
        return _empty("Not enough data for plateau analysis (need ≥4 trials)")

    grp_agg = sub.groupby([x_col, y_col])[z_col].agg(
        mean_val="mean", std_val="std", count="count"
    ).reset_index()
    grp_agg["std_val"] = grp_agg["std_val"].fillna(0.0)

    xs   = grp_agg[x_col].values.astype(float)
    ys   = grp_agg[y_col].values.astype(float)
    means = grp_agg["mean_val"].values.astype(float)
    stds  = grp_agg["std_val"].values.astype(float)

    # Neighborhood mean
    n = len(grp_agg)
    x_range = xs.ptp() if xs.ptp() > 0 else 1.0
    y_range = ys.ptp() if ys.ptp() > 0 else 1.0
    xs_n = (xs - xs.min()) / x_range
    ys_n = (ys - ys.min()) / y_range
    k_eff = min(k, n - 1)
    neigh = np.empty(n)
    for i in range(n):
        dists = np.sqrt((xs_n - xs_n[i]) ** 2 + (ys_n - ys_n[i]) ** 2)
        dists[i] = np.inf
        nn_idx = np.argpartition(dists, k_eff)[:k_eff]
        neigh[i] = means[nn_idx].mean()

    def _rank01(arr, ascending=True):
        order = np.argsort(arr)
        ranks = np.empty(len(arr))
        for pos, idx in enumerate(order):
            ranks[idx] = pos / max(len(arr) - 1, 1)
        return ranks if ascending else 1.0 - ranks

    if direction == "maximize":
        r_mean  = _rank01(means, ascending=True)
        r_neigh = _rank01(neigh, ascending=True)
    else:
        r_mean  = _rank01(means, ascending=False)
        r_neigh = _rank01(neigh, ascending=False)

    r_std    = _rank01(stds, ascending=False)   # low std = good
    plateau  = r_mean * r_std * r_neigh
    plateau  = plateau / max(plateau.max(), 1e-9)   # normalise to [0,1]

    # Colorscale for plateau (red=bad, blue=best)
    PLATEAU_CS = [
        [0.00, "#c0392b"],
        [0.20, "#e67e22"],
        [0.40, "#f1c40f"],
        [0.60, "#27ae60"],
        [0.80, "#1abc9c"],
        [1.00, "#2980b9"],
    ]

    fig = go.Figure()

    grid_size = 100
    xi = np.linspace(xs.min(), xs.max(), grid_size)
    yi = np.linspace(ys.min(), ys.max(), grid_size)
    Xi, Yi = np.meshgrid(xi, yi)

    try:
        Zi = griddata((xs, ys), plateau, (Xi, Yi), method='linear')
        if np.isnan(Zi).any():
            Zi_nearest = griddata((xs, ys), plateau, (Xi, Yi), method='nearest')
            Zi[np.isnan(Zi)] = Zi_nearest[np.isnan(Zi)]

        fig.add_trace(go.Contour(
            x=xi, y=yi, z=Zi,
            colorscale=PLATEAU_CS,
            zmin=0, zmax=1,
            showscale=True,
            contours=dict(coloring="heatmap", showlines=True),
            line=dict(width=0.5, color="rgba(0,0,0,0.25)"),
            colorbar=dict(
                title=dict(text="Plateau Index", font=dict(color=TEXT_DIM, size=11)),
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=["0 (poor)", "0.25", "0.5", "0.75", "1 (best)"],
                tickfont=dict(color=TEXT_DIM, size=10),
                bgcolor=SURFACE_2, bordercolor=BORDER, borderwidth=1,
                outlinewidth=0, thickness=14, len=0.85,
            ),
            hoverinfo="skip",
        ))
    except Exception as exc:
        print(f"[plateau interpolation] {exc}")
        return _empty("Interpolation failed for plateau map")

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(
            color="rgba(30, 30, 30, 0.6)", 
            size=6, 
            line=dict(width=0.5, color="rgba(255, 255, 255, 0.7)")
        ),
        text=[f"<b>{x_param}</b>: {grp_agg.iloc[i][x_col]}<br>"
              f"<b>{y_param}</b>: {grp_agg.iloc[i][y_col]}<br>"
              f"<b>Plateau Index</b>: {plateau[i]:.4f}<br>"
              f"<b>Mean {metric}</b>: {means[i]:.5f}<br>"
              f"<b>Std</b>: {stds[i]:.5f}<br>"
              f"<b>Neighbor mean</b>: {neigh[i]:.5f}"
              for i in range(n)],
        hovertemplate="%{text}<extra></extra>", showlegend=False,
    ))

    fig.update_layout(
        paper_bgcolor=LAYOUT["paper_bgcolor"], plot_bgcolor=BG,
        font=LAYOUT["font"], margin=dict(l=60, r=30, t=52, b=56),
        hoverlabel=LAYOUT["hoverlabel"],
        title=dict(
            text="<b>Plateau Robustness Index</b>"
                 f"  ·  {x_param} × {y_param}  —  {metric}"
                 f"  <span style='color:{TEXT_DIM};font-size:11px;'>high score · low std · good neighborhood</span>",
            font=dict(size=14, color=TEXT), x=0.01, xanchor="left",
        ),
        xaxis=dict(showgrid=False, linecolor=BORDER, zerolinecolor="rgba(0,0,0,0)",
                   tickfont=dict(color=TEXT_DIM),
                   title=dict(text=x_param, font=dict(color=TEXT_DIM, size=12))),
        yaxis=dict(showgrid=False, linecolor=BORDER, zerolinecolor="rgba(0,0,0,0)",
                   tickfont=dict(color=TEXT_DIM),
                   title=dict(text=y_param, font=dict(color=TEXT_DIM, size=12))),
        height=520, hovermode="closest", showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────

def _empty(msg: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(color=TEXT_DIM, size=13, family="DM Mono, monospace"))
    fig.update_layout(
        paper_bgcolor=SURFACE, plot_bgcolor=BG, height=520,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=20, b=20))
    return fig

empty_figure = _empty