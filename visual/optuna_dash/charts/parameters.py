"""
MODELOX · charts/parameters.py
Parameter-space analysis: sweep, heatmap, contour, surface-3D, importance,
box distributions, range analysis, parallel coordinates, interaction matrix,
stability heatmap, robustness score.

Design:
  → Semáforo colorscale (rojo→naranja→amarillo→verde)
  → Minimalist chrome — every element earns its pixel
  → Scientific typography: monospace for data, sans for labels
  → Dark plot_bgcolor on surface charts
  → Single warm accent for optimal-point markers
"""
from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from visual.optuna_dash.theme import (
    C, PALETTE, PALETTE_MUTED, METRIC_LABEL, METRIC_INVERT, SCALE_DIV,
    get_m, _lay, _empty, _fix_sub, _colorscale, _MONO, _SANS,
)
from visual.optuna_dash.loader import _build_dense_smooth_grid, _calc_importance


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  INTERNAL LAYOUT HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_BG      = "#FAFBFC"
_SURFACE = "#FFFFFF"
_GRID    = "#F0F1F3"
_BORDER  = "#E2E4E8"
_TEXT    = "#1A1D23"
_TEXT2   = "#5A5F6B"
_DIM     = "#8C919C"
_ACCENT  = "#E85D3A"
_ACCENT2 = "#2563EB"
_FONT    = "'Inter','Helvetica Neue',sans-serif"
_DARK_BG = "#1A1D23"


def _base(**kw) -> dict:
    defaults = dict(
        paper_bgcolor=_BG,
        plot_bgcolor=_SURFACE,
        font=dict(family=_FONT, color=_TEXT, size=11),
        title=dict(
            font=dict(family=_MONO, size=11, color=_TEXT2),
            x=0.01, xanchor="left", y=0.98, yanchor="top",
            pad=dict(t=4, b=0),
        ),
        margin=dict(l=52, r=16, t=42, b=44),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor=_BORDER,
            font=dict(family=_MONO, size=10, color=_TEXT),
        ),
        showlegend=False,
    )
    defaults.update(kw)
    return defaults


def _ax(title: str = "", **kw) -> dict:
    base = dict(
        title=dict(text=title, font=dict(family=_FONT, size=10, color=_TEXT2), standoff=8),
        gridcolor=_GRID, gridwidth=0.5,
        linecolor=_BORDER, linewidth=0.8,
        zeroline=False,
        tickfont=dict(family=_MONO, size=9, color=_DIM),
        ticks="outside", ticklen=3, tickwidth=0.6, tickcolor=_BORDER,
    )
    base.update(kw)
    return base


def _cbar(metric_key: str) -> dict:
    return dict(
        title=dict(
            text=METRIC_LABEL.get(metric_key, metric_key),
            font=dict(family=_FONT, size=9, color=_TEXT2), side="right",
        ),
        thickness=7, len=0.75, x=1.02,
        tickfont=dict(family=_MONO, size=8, color=_DIM),
        tickformat=".2f", outlinewidth=0, bgcolor="rgba(0,0,0,0)",
    )


def _apply(fig: go.Figure, title: str, height: int = 440, **kw) -> go.Figure:
    layout = _base(title=dict(text=title), height=height, **kw)
    fig.update_layout(**layout)
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DATA EXTRACTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _extract_2d(trials, px, py, metric_key):
    """Extract (x, y, z, trial_num, trades_per_day) tuples."""
    pts = []
    for t in trials:
        x = t["params"].get(px); y = t["params"].get(py)
        if x is not None and y is not None:
            try:
                tpd = float(t.get("met", {}).get("trades_por_dia", 0) or 0)
                pts.append((float(x), float(y), get_m(t, metric_key), t["number"], tpd))
            except Exception:
                pass
    return pts


def _extract_1d(trials, param, metric_key):
    pts = []
    for t in trials:
        v = t["params"].get(param)
        if v is not None:
            try:
                pts.append((float(v), get_m(t, metric_key), t["number"]))
            except Exception:
                pass
    return pts


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SWEEP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_sweep(trials: list[dict], param: str, metric_key: str = "score") -> go.Figure:
    if not trials or not param:
        return _empty()
    pts = _extract_1d(trials, param, metric_key)
    if not pts:
        return _empty()
    xs, ys, nums = zip(*pts)

    n = len(xs); sx, sy = sum(xs), sum(ys)
    sxy = sum(xs[i] * ys[i] for i in range(n))
    sx2 = sum(v ** 2 for v in xs)
    denom = n * sx2 - sx ** 2
    has_trend = abs(denom) > 1e-9
    if has_trend:
        b1 = (n * sxy - sx * sy) / denom
        b0 = (sy - b1 * sx) / n
        tx = [min(xs), max(xs)]
        ty = [b0 + b1 * v for v in tx]
    else:
        tx, ty = [], []

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(color=ys, colorscale=_colorscale(metric_key),
                    size=5, opacity=0.75, line=dict(width=0),
                    colorbar=_cbar(metric_key)),
        text=[f"#{n}  {param}={x:.3g}  {METRIC_LABEL.get(metric_key,metric_key)}={y:.3f}"
              for x, y, n in zip(xs, ys, nums)],
        hovertemplate="%{text}<extra></extra>",
    ))
    if has_trend and tx:
        fig.add_trace(go.Scatter(
            x=tx, y=ty, mode="lines",
            line=dict(color=_DIM, width=1.2, dash="dot"),
            hoverinfo="skip",
        ))

    return _apply(fig, title=f"sweep · {param}",
                  xaxis=_ax(param), yaxis=_ax(METRIC_LABEL.get(metric_key, metric_key)))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HEATMAP 2D
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_heatmap2d(
    trials: list[dict], px: str, py: str,
    metric_key: str = "score", bins: int = 40,
) -> go.Figure:
    if not trials or not px or not py:
        return _empty()
    pts = _extract_2d(trials, px, py, metric_key)
    if len(pts) < 8:
        return _empty("insufficient data")
    xs, ys, zs, _, _ = zip(*pts)

    grid = _build_dense_smooth_grid(xs, ys, zs, bins=bins, smooth_passes=7)
    if grid is None:
        return _empty()
    xc, yc, z_grid, *_ = grid

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=xc, y=yc, z=z_grid,
        colorscale=_colorscale(metric_key),
        zsmooth="best", hoverongaps=False,
        colorbar=_cbar(metric_key),
        hovertemplate=(f"{px}: %{{x:.4g}}<br>{py}: %{{y:.4g}}<br>"
                       f"{METRIC_LABEL.get(metric_key,metric_key)}: %{{z:.3f}}<extra></extra>"),
    ))

    return _apply(fig, title=f"heatmap · {px} × {py}",
                  xaxis=_ax(px, showgrid=False), yaxis=_ax(py, showgrid=False),
                  plot_bgcolor=_DARK_BG)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONTOUR — Response surface
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_contour_gravity(
    trials: list[dict], px: str, py: str,
    metric_key: str = "score", bins: int = 36,
) -> go.Figure:
    """
    Clean filled contour with thin dark iso-lines.
    No markers, no crosses — let the topology speak.
    """
    if not trials or not px or not py:
        return _empty()
    pts = _extract_2d(trials, px, py, metric_key)
    if len(pts) < 10:
        return _empty("insufficient data for contour")
    xs, ys, zs, nums, tpds = zip(*pts)

    grid = _build_dense_smooth_grid(xs, ys, zs, bins=bins, smooth_passes=5)
    if grid is None:
        return _empty("insufficient range")
    xc, yc, z_grid, *_ = grid

    z_flat = [v for row in z_grid for v in row]
    z_lo, z_hi = min(z_flat), max(z_flat)
    z_range = z_hi - z_lo
    if z_range < 1e-9:
        return _empty("flat surface — no variation")

    # ~15 levels: wide enough bands to read color, not too many lines
    n_levels = 15
    z_step = z_range / n_levels

    fig = go.Figure()

    # Single contour trace: filled + thin dark lines + sparse labels
    fig.add_trace(go.Contour(
        x=xc, y=yc, z=z_grid,
        colorscale=_colorscale(metric_key),
        connectgaps=True,
        contours=dict(
            coloring="heatmap",
            start=z_lo,
            end=z_hi,
            size=z_step,
            showlabels=True,
            labelfont=dict(family=_MONO, size=9, color="rgba(24,24,27,0.65)"),
        ),
        line=dict(width=0.6, color="rgba(24,24,27,0.22)"),
        showscale=True,
        colorbar=_cbar(metric_key),
        hovertemplate=(
            f"{px}: %{{x:.4g}}<br>"
            f"{py}: %{{y:.4g}}<br>"
            f"{METRIC_LABEL.get(metric_key, metric_key)}: %{{z:.3f}}"
            "<extra></extra>"
        ),
    ))

    # Trial scatter — tiny dots, hover shows trial # + metric + T/day
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=2, color="rgba(24,24,27,0.15)", line=dict(width=0)),
        text=[
            f"#{nums[i]}  {METRIC_LABEL.get(metric_key, metric_key)}={zs[i]:.3f}  T/day={tpds[i]:.2f}"
            for i in range(len(xs))
        ],
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))

    return _apply(
        fig,
        title=f"response surface · contour · {px} × {py}",
        height=480,
        xaxis=_ax(px, showgrid=False, linecolor=_BORDER),
        yaxis=_ax(py, showgrid=False, linecolor=_BORDER),
        plot_bgcolor="#FFFFFF",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SURFACE 3D
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_surface3d(
    trials: list[dict], px: str, py: str,
    metric_key: str = "score", bins: int = 40,
) -> go.Figure:
    """
    Clean 3D surface — no contour lines drawn on it, no floating markers.
    The color gradient alone communicates the topology.
    """
    if not trials or not px or not py:
        return _empty()
    pts = _extract_2d(trials, px, py, metric_key)
    if len(pts) < 20:
        return _empty("insufficient data for 3D surface")
    xs, ys, zs, nums, tpds = zip(*pts)

    grid = _build_dense_smooth_grid(xs, ys, zs, bins=bins, smooth_passes=6)
    if grid is None:
        return _empty("insufficient range")
    xc, yc, z_grid, *_ = grid

    z_flat = [v for row in z_grid for v in row]
    z_min, z_max = min(z_flat), max(z_flat)

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=xc, y=yc, z=z_grid,
        colorscale=_colorscale(metric_key),
        cmin=z_min, cmax=z_max,
        opacity=0.96,
        lighting=dict(
            ambient=0.70,
            diffuse=0.55,
            roughness=0.85,
            specular=0.05,
            fresnel=0.03,
        ),
        lightposition=dict(x=100, y=100, z=800),
        contours=dict(
            x=dict(show=False),
            y=dict(show=False),
            z=dict(
                show=True,
                usecolormap=False,
                color="rgba(20,20,20,0.55)",
                width=1.2,
                highlightcolor="rgba(20,20,20,0.55)",
                project=dict(z=False),
            ),
        ),
        colorbar={**_cbar(metric_key), "len": 0.70},
        hovertemplate=(
            f"{px}: %{{x:.4g}}<br>"
            f"{py}: %{{y:.4g}}<br>"
            f"{METRIC_LABEL.get(metric_key, metric_key)}: %{{z:.3f}}"
            "<extra></extra>"
        ),
        showlegend=False,
    ))

    _ax3d = dict(
        gridcolor="#EBEBEF",
        linecolor=_BORDER,
        zerolinecolor=_BORDER,
        backgroundcolor="#FAFBFC",
        tickfont=dict(family=_MONO, size=8, color=_DIM),
        title_font=dict(family=_FONT, size=9, color=_TEXT2),
        showspikes=False,
        showline=True,
        linewidth=0.6,
        ticks="outside",
        ticklen=2,
    )
    fig.update_layout(**_base(
        title=dict(text=f"response surface · 3D · {px} × {py}"),
        height=500,
        margin=dict(l=0, r=16, t=42, b=8),
        scene=dict(
            xaxis=dict(**_ax3d, title=px),
            yaxis=dict(**_ax3d, title=py),
            zaxis=dict(**_ax3d, title=METRIC_LABEL.get(metric_key, metric_key)),
            bgcolor="#FAFBFC",
            aspectratio=dict(x=1.4, y=1.2, z=0.70),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=-0.08),
                eye=dict(x=1.50, y=1.50, z=0.70),
            ),
        ),
    ))
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  IMPORTANCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_importance_chart(trials: list[dict], metric_key: str = "score") -> go.Figure:
    imps = _calc_importance(trials, metric_key)
    if not imps:
        return _empty()

    srt = sorted(imps.items(), key=lambda x: x[1])
    names = [k for k, _ in srt]
    vals = [v for _, v in srt]
    max_v = max(vals) if vals else 1

    colors = []
    for v in vals:
        ratio = v / max_v if max_v > 0 else 0
        if ratio > 0.5:   colors.append(_ACCENT)
        elif ratio > 0.2: colors.append(_ACCENT2)
        else:              colors.append("#C8CCD4")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=vals, y=names, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f" |r| = {v:.3f}" for v in vals],
        textposition="inside",
        textfont=dict(family=_MONO, size=8, color="#FFFFFF"),
        insidetextanchor="start",
        hovertemplate="%{y}   |r| = %{x:.4f}<extra></extra>",
    ))
    fig.add_vline(x=0.15, line=dict(color=_ACCENT, width=0.6, dash="dot"))
    fig.add_vline(x=0.06, line=dict(color=_ACCENT2, width=0.6, dash="dot"))

    return _apply(fig,
                  title=f"parameter importance · |r| vs {METRIC_LABEL.get(metric_key,metric_key)}",
                  height=max(300, 70 + 26 * len(names)),
                  xaxis=_ax("|Pearson r|"),
                  yaxis=dict(tickfont=dict(family=_MONO, size=9, color=_TEXT2), showgrid=False))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  BOX / VIOLIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_box_distributions(trials: list[dict]) -> go.Figure:
    if not trials or not trials[0]["params"]:
        return _empty()
    pnames = list(trials[0]["params"].keys())
    fig = go.Figure()
    for i, pn in enumerate(pnames):
        vals = []
        for t in trials:
            v = t["params"].get(pn)
            if v is not None:
                try: vals.append(float(v))
                except Exception: pass
        if not vals:
            continue
        mn, mx = min(vals), max(vals)
        nv = [((v - mn) / (mx - mn) if mx > mn else 0.5) for v in vals]
        col = PALETTE_MUTED[i % len(PALETTE_MUTED)]
        r, g, b = int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16)
        fig.add_trace(go.Violin(
            y=nv, name=pn, box_visible=True, meanline_visible=True, points=False,
            line_color=col, fillcolor=f"rgba({r},{g},{b},0.06)",
            hovertemplate=f"{pn}: %{{y:.3f}}<extra></extra>",
        ))
    return _apply(fig, title="parameter distributions · normalized [0 – 1]", height=420,
                  xaxis=_ax("parameter"), yaxis=_ax("normalized value"),
                  violinmode="overlay")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RANGE ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_range_analysis(
    trials: list[dict], metric_key: str = "score", top_pct: int = 25,
) -> go.Figure:
    if not trials:
        return _empty()
    pnames = list(trials[0]["params"].keys()) if trials[0].get("params") else []
    if not pnames:
        return _empty()

    ms = sorted(get_m(t, metric_key) for t in trials)
    n = len(ms)
    invert = metric_key in METRIC_INVERT
    if invert:
        threshold = ms[int((top_pct / 100) * (n - 1))]
        top_set = {i for i, t in enumerate(trials) if get_m(t, metric_key) <= threshold}
    else:
        threshold = ms[int((1 - top_pct / 100) * (n - 1))]
        top_set = {i for i, t in enumerate(trials) if get_m(t, metric_key) >= threshold}

    n_cols = min(3, len(pnames))
    n_rows = math.ceil(len(pnames) / n_cols)
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=pnames,
                        vertical_spacing=0.08, horizontal_spacing=0.06)

    for idx, pn in enumerate(pnames):
        row, col = idx // n_cols + 1, idx % n_cols + 1
        all_v, top_v = [], []
        for i, t in enumerate(trials):
            v = t["params"].get(pn)
            if v is not None:
                try:
                    fv = float(v); all_v.append(fv)
                    if i in top_set: top_v.append(fv)
                except Exception: pass
        if not all_v:
            continue
        show = idx == 0
        fig.add_trace(go.Violin(
            y=all_v, name="all trials", side="negative",
            line_color=_DIM, fillcolor="rgba(140,145,156,0.08)",
            box_visible=True, meanline_visible=True, points=False,
            legendgroup="all", showlegend=show,
            hovertemplate=f"{pn}: %{{y:.4g}}<extra>all</extra>",
        ), row=row, col=col)
        if top_v:
            fig.add_trace(go.Violin(
                y=top_v, name=f"top {top_pct}%", side="positive",
                line_color=_ACCENT, fillcolor="rgba(232,93,58,0.10)",
                box_visible=True, meanline_visible=True, points=False,
                legendgroup=f"top{top_pct}", showlegend=show,
                hovertemplate=f"{pn}: %{{y:.4g}}<extra>top {top_pct}%</extra>",
            ), row=row, col=col)

    from visual.optuna_dash.theme import PL
    d = dict(PL)
    d.update(dict(
        title=f"RANGE ANALYSIS · Top {top_pct}% vs Universe · {METRIC_LABEL.get(metric_key, metric_key)}",
        height=max(360, 240 * n_rows),
        violinmode="overlay", showlegend=True,
        margin=dict(l=40, r=14, t=50, b=30),
    ))
    fig.update_layout(**d)
    _fix_sub(fig)
    return fig

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PARAMETER INTERACTION MATRIX
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_interaction_matrix(
    trials: list[dict], metric_key: str = "score", top_pct: int = 25,
) -> go.Figure:
    """
    N×N heatmap of pairwise Pearson correlation between parameters,
    computed ONLY on the top-performing trials.

    Reveals hidden dependencies: if two parameters are strongly correlated
    in top trials but not in the full universe, the optimizer found a
    co-dependent optimum that may be fragile.
    """
    if not trials:
        return _empty()
    pnames = list(trials[0]["params"].keys()) if trials[0].get("params") else []
    if len(pnames) < 2:
        return _empty("need ≥ 2 parameters")

    # Select top N% trials
    ms = sorted(get_m(t, metric_key) for t in trials)
    n = len(ms)
    invert = metric_key in METRIC_INVERT
    if invert:
        threshold = ms[int((top_pct / 100) * (n - 1))]
        top_trials = [t for t in trials if get_m(t, metric_key) <= threshold]
    else:
        threshold = ms[int((1 - top_pct / 100) * (n - 1))]
        top_trials = [t for t in trials if get_m(t, metric_key) >= threshold]

    if len(top_trials) < 5:
        return _empty("insufficient top trials")

    # Extract parameter vectors
    param_vals: dict[str, list[float]] = {}
    for pn in pnames:
        vals = []
        for t in top_trials:
            v = t["params"].get(pn)
            try:
                vals.append(float(v) if v is not None else float("nan"))
            except Exception:
                vals.append(float("nan"))
        param_vals[pn] = vals

    # Compute correlation matrix
    np_k = len(pnames)
    corr = [[0.0] * np_k for _ in range(np_k)]
    for i in range(np_k):
        for j in range(np_k):
            if i == j:
                corr[i][j] = 1.0
                continue
            a = param_vals[pnames[i]]
            b = param_vals[pnames[j]]
            # Filter NaN pairs
            pairs = [(av, bv) for av, bv in zip(a, b)
                     if not (math.isnan(av) or math.isnan(bv))]
            if len(pairs) < 3:
                corr[i][j] = 0.0
                continue
            va, vb = zip(*pairs)
            n_p = len(va)
            ma = sum(va) / n_p
            mb = sum(vb) / n_p
            cov = sum((va[k] - ma) * (vb[k] - mb) for k in range(n_p))
            var_a = sum((v - ma) ** 2 for v in va)
            var_b = sum((v - mb) ** 2 for v in vb)
            if var_a > 1e-9 and var_b > 1e-9:
                corr[i][j] = round(cov / (var_a ** 0.5 * var_b ** 0.5), 3)
            else:
                corr[i][j] = 0.0

    txt = [[f"{corr[i][j]:.2f}" for j in range(np_k)] for i in range(np_k)]

    fig = go.Figure(go.Heatmap(
        x=pnames, y=pnames, z=corr,
        colorscale=SCALE_DIV,
        zmid=0, zmin=-1, zmax=1,
        text=txt, texttemplate="%{text}",
        textfont=dict(size=9, color=_TEXT, family=_MONO),
        colorbar=dict(
            title="r", thickness=7,
            tickfont=dict(color=_DIM, size=8, family=_MONO),
            outlinewidth=0, bgcolor="rgba(0,0,0,0)",
        ),
        hoverongaps=False,
        hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
    ))

    return _apply(
        fig,
        title=f"parameter interaction · top {top_pct}% · {METRIC_LABEL.get(metric_key, metric_key)}",
        height=max(380, 50 + 32 * len(pnames)),
        margin=dict(l=80, r=16, t=42, b=80),
        xaxis=dict(tickfont=dict(family=_MONO, size=9, color=_TEXT2), tickangle=-45),
        yaxis=dict(tickfont=dict(family=_MONO, size=9, color=_TEXT2), autorange="reversed"),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STABILITY HEATMAP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_stability_heatmap(
    trials: list[dict], px: str, py: str,
    metric_key: str = "score", bins: int = 20,
) -> go.Figure:
    """
    Local variance heatmap: for each cell in the parameter grid, compute
    the standard deviation of the metric among trials that fall in that cell
    and its immediate neighbors.

    Low std (green) = stable/robust zone.
    High std (red) = unstable zone — overfitting risk.

    This is the visual complement to the response surface: the surface shows
    WHERE the optimum is, this shows HOW MUCH you can trust it.
    """
    if not trials or not px or not py:
        return _empty()

    pts = _extract_2d(trials, px, py, metric_key)
    if len(pts) < 15:
        return _empty("insufficient data for stability analysis")

    xs, ys, zs, _, _ = zip(*pts)
    x_arr = np.array(xs, dtype=np.float64)
    y_arr = np.array(ys, dtype=np.float64)
    z_arr = np.array(zs, dtype=np.float64)

    x_min, x_max = float(x_arr.min()), float(x_arr.max())
    y_min, y_max = float(y_arr.min()), float(y_arr.max())
    if x_max == x_min or y_max == y_min:
        return _empty("insufficient parameter range")

    x_step = (x_max - x_min) / bins
    y_step = (y_max - y_min) / bins

    # Bin trials into grid cells
    xi = np.clip(((x_arr - x_min) / x_step).astype(int), 0, bins - 1)
    yi = np.clip(((y_arr - y_min) / y_step).astype(int), 0, bins - 1)

    # Collect z values per cell
    cells: dict[tuple[int, int], list[float]] = {}
    for i in range(len(xs)):
        key = (int(yi[i]), int(xi[i]))
        cells.setdefault(key, []).append(zs[i])

    # Compute local std: cell + 8 neighbors
    std_grid = np.full((bins, bins), np.nan, dtype=np.float64)
    for r in range(bins):
        for c in range(bins):
            local_vals = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < bins and 0 <= nc < bins:
                        local_vals.extend(cells.get((nr, nc), []))
            if len(local_vals) >= 2:
                mu = sum(local_vals) / len(local_vals)
                var = sum((v - mu) ** 2 for v in local_vals) / len(local_vals)
                std_grid[r, c] = var ** 0.5

    # Fill NaN with neighborhood averaging (same as _build_dense_smooth_grid)
    for _ in range(bins):
        nan_mask = np.isnan(std_grid)
        if not np.any(nan_mask):
            break
        zp = np.pad(std_grid, 1, mode="constant", constant_values=np.nan)
        s_neigh = np.zeros_like(std_grid)
        c_neigh = np.zeros_like(std_grid, dtype=int)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                w = zp[1 + dy:1 + dy + bins, 1 + dx:1 + dx + bins]
                valid = ~np.isnan(w)
                s_neigh += np.where(valid, w, 0.0)
                c_neigh += valid.astype(int)
        fill = nan_mask & (c_neigh >= 2)
        if not np.any(fill):
            break
        std_grid[fill] = s_neigh[fill] / c_neigh[fill]

    # Remaining NaN → global mean
    if np.isnan(std_grid).any():
        gmean = np.nanmean(std_grid)
        std_grid = np.where(np.isnan(std_grid), gmean if np.isfinite(gmean) else 0, std_grid)

    # Light smoothing
    for _ in range(2):
        zp = np.pad(std_grid, 1, mode="edge")
        std_grid = (
            (zp[:-2, :-2] + 2 * zp[:-2, 1:-1] + zp[:-2, 2:]) +
            (2 * zp[1:-1, :-2] + 6 * zp[1:-1, 1:-1] + 2 * zp[1:-1, 2:]) +
            (zp[2:, :-2] + 2 * zp[2:, 1:-1] + zp[2:, 2:])
        ) / 18.0

    xc = [x_min + (i + 0.5) * x_step for i in range(bins)]
    yc = [y_min + (i + 0.5) * y_step for i in range(bins)]

    # Stability scale: Green (low std = stable) → Red (high std = unstable)
    # Inverted semáforo — low = good here
    stability_cs = [
        [0.000, "#15803D"],   # deep green — very stable
        [0.250, "#22C55E"],   # green
        [0.500, "#FBBF24"],   # amber — moderate variance
        [0.750, "#EA580C"],   # orange — concerning
        [1.000, "#991B1B"],   # deep red — unstable
    ]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=xc, y=yc, z=std_grid.tolist(),
        colorscale=stability_cs,
        zsmooth="best",
        hoverongaps=False,
        colorbar=dict(
            title=dict(text="Local σ", font=dict(family=_FONT, size=9, color=_TEXT2), side="right"),
            thickness=7, len=0.75, x=1.02,
            tickfont=dict(family=_MONO, size=8, color=_DIM),
            tickformat=".2f", outlinewidth=0, bgcolor="rgba(0,0,0,0)",
        ),
        hovertemplate=(
            f"{px}: %{{x:.4g}}<br>"
            f"{py}: %{{y:.4g}}<br>"
            f"Local σ: %{{z:.3f}}<extra></extra>"
        ),
    ))

    return _apply(
        fig,
        title=f"stability · local σ({METRIC_LABEL.get(metric_key, metric_key)}) · {px} × {py}",
        xaxis=_ax(px, showgrid=False),
        yaxis=_ax(py, showgrid=False),
        plot_bgcolor=_SURFACE,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PARALLEL COORDINATES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_parallel_coordinates(
    trials: list[dict], metric_key: str = "score",
) -> go.Figure:
    """
    Parallel coordinates: one axis per parameter + the metric.
    Lines colored by metric value (semáforo).

    Key diagnostic: if top trials converge to narrow bands on most axes,
    the optimum is robust. If they span the full range, the metric landscape
    is flat or noisy — parameters don't matter much.
    """
    if not trials:
        return _empty()

    pnames = list(trials[0]["params"].keys()) if trials[0].get("params") else []
    if not pnames:
        return _empty("no parameters")

    # Build dimensions
    dimensions = []
    for pn in pnames:
        vals = []
        for t in trials:
            v = t["params"].get(pn)
            try:
                vals.append(float(v) if v is not None else float("nan"))
            except Exception:
                vals.append(float("nan"))
        if all(math.isnan(v) for v in vals):
            continue
        clean = [v for v in vals if not math.isnan(v)]
        mn, mx = min(clean), max(clean)
        if mn == mx:
            mx = mn + 1
        dimensions.append(dict(
            label=pn,
            values=vals,
            range=[mn, mx],
        ))

    if not dimensions:
        return _empty("no numeric parameters")

    # Add metric as final dimension
    metric_vals = [get_m(t, metric_key) for t in trials]
    m_clean = [v for v in metric_vals if not math.isnan(v)]
    m_min, m_max = min(m_clean), max(m_clean)
    if m_min == m_max:
        m_max = m_min + 1
    dimensions.append(dict(
        label=METRIC_LABEL.get(metric_key, metric_key),
        values=metric_vals,
        range=[m_min, m_max],
    ))

    fig = go.Figure(go.Parcoords(
        line=dict(
            color=metric_vals,
            colorscale=_colorscale(metric_key),
            cmin=m_min, cmax=m_max,
            showscale=True,
            colorbar=dict(
                title=dict(
                    text=METRIC_LABEL.get(metric_key, metric_key),
                    font=dict(family=_FONT, size=9, color=_TEXT2),
                ),
                thickness=7, len=0.75,
                tickfont=dict(family=_MONO, size=8, color=_DIM),
                tickformat=".2f", outlinewidth=0,
            ),
        ),
        dimensions=dimensions,
        labelfont=dict(family=_MONO, size=9, color=_TEXT2),
        rangefont=dict(family=_MONO, size=8, color=_DIM),
        tickfont=dict(family=_MONO, size=8, color=_DIM),
    ))

    return _apply(
        fig,
        title=f"parallel coordinates · colored by {METRIC_LABEL.get(metric_key, metric_key)}",
        height=420,
        margin=dict(l=60, r=60, t=42, b=30),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ROBUSTNESS SCORE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_robustness_scatter(
    trials: list[dict], metric_key: str = "score", k_neighbors: int = 10,
) -> go.Figure:
    """
    Robustness diagnostic: for each trial, compare its metric to the average
    of its K nearest neighbors in normalized parameter space.

    X-axis = trial metric value.
    Y-axis = mean neighbor metric value.
    Diagonal line = perfect consistency (trial ≈ neighbors).

    Trials FAR ABOVE the diagonal = suspiciously good — their neighbors
    perform much worse, suggesting overfitting to specific parameter values.
    Trials NEAR the diagonal = robust — similar performance in the
    surrounding parameter neighborhood.
    """
    if not trials:
        return _empty()

    pnames = list(trials[0]["params"].keys()) if trials[0].get("params") else []
    if not pnames:
        return _empty("no parameters")

    n = len(trials)
    if n < k_neighbors + 1:
        return _empty(f"need ≥ {k_neighbors + 1} trials")

    # Build normalized parameter matrix
    param_matrix = np.zeros((n, len(pnames)), dtype=np.float64)
    for j, pn in enumerate(pnames):
        col = []
        for t in trials:
            v = t["params"].get(pn)
            try:
                col.append(float(v) if v is not None else 0.0)
            except Exception:
                col.append(0.0)
        col = np.array(col, dtype=np.float64)
        mn, mx = col.min(), col.max()
        if mx > mn:
            param_matrix[:, j] = (col - mn) / (mx - mn)
        else:
            param_matrix[:, j] = 0.5

    # Metric values
    metric_vals = np.array([get_m(t, metric_key) for t in trials], dtype=np.float64)
    nums = [t["number"] for t in trials]

    # For each trial, find K nearest neighbors by Euclidean distance in param space
    neighbor_means = np.zeros(n, dtype=np.float64)
    for i in range(n):
        dists = np.sqrt(np.sum((param_matrix - param_matrix[i]) ** 2, axis=1))
        dists[i] = np.inf  # exclude self
        nearest = np.argsort(dists)[:k_neighbors]
        neighbor_means[i] = np.mean(metric_vals[nearest])

    # Robustness score = |trial_metric - neighbor_mean| / neighbor_std
    # But for visualization: simple scatter + diagonal
    m_min = min(float(metric_vals.min()), float(neighbor_means.min()))
    m_max = max(float(metric_vals.max()), float(neighbor_means.max()))

    # Color by distance from diagonal (overfitting indicator)
    delta = metric_vals - neighbor_means
    abs_delta = np.abs(delta)

    fig = go.Figure()

    # Diagonal reference
    fig.add_trace(go.Scatter(
        x=[m_min, m_max], y=[m_min, m_max],
        mode="lines",
        line=dict(color=_DIM, width=1, dash="dot"),
        hoverinfo="skip", showlegend=False,
    ))

    # Scatter — colored by absolute deviation
    fig.add_trace(go.Scatter(
        x=metric_vals.tolist(),
        y=neighbor_means.tolist(),
        mode="markers",
        marker=dict(
            color=abs_delta.tolist(),
            colorscale=[
                [0.0, "#15803D"],   # close to diagonal = robust (green)
                [0.5, "#FBBF24"],   # moderate deviation (amber)
                [1.0, "#991B1B"],   # far from diagonal = overfitting risk (red)
            ],
            size=5, opacity=0.75,
            line=dict(width=0),
            colorbar=dict(
                title=dict(text="Δ score", font=dict(family=_FONT, size=9, color=_TEXT2), side="right"),
                thickness=7, len=0.75, x=1.02,
                tickfont=dict(family=_MONO, size=8, color=_DIM),
                tickformat=".2f", outlinewidth=0, bgcolor="rgba(0,0,0,0)",
            ),
        ),
        text=[
            f"#{nums[i]}  score={metric_vals[i]:.3f}  neighbors={neighbor_means[i]:.3f}  Δ={delta[i]:+.3f}"
            for i in range(n)
        ],
        hovertemplate="%{text}<extra></extra>",
    ))

    # Annotations for zones
    fig.add_annotation(
        text="← robust zone", x=m_max, y=m_max,
        xanchor="right", yanchor="bottom",
        showarrow=False,
        font=dict(family=_MONO, size=8, color="#15803D"),
    )
    fig.add_annotation(
        text="overfitting risk →",
        x=m_max, y=m_min + (m_max - m_min) * 0.15,
        xanchor="right", yanchor="bottom",
        showarrow=False,
        font=dict(family=_MONO, size=8, color="#991B1B"),
    )

    label = METRIC_LABEL.get(metric_key, metric_key)
    return _apply(
        fig,
        title=f"robustness · trial {label} vs {k_neighbors}-neighbor mean",
        height=440,
        xaxis=_ax(f"trial {label}"),
        yaxis=_ax(f"neighbor mean {label}"),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SCATTER 3D — 3 PARAMETERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_scatter3d_params(
    trials: list[dict], px: str, py: str, pz: str,
    metric_key: str = "score", bins: int = 40,
) -> go.Figure:
    """
    Réplica exacta de make_surface3d pero con 3 parámetros:
      · X, Y = base del grid (px, py)
      · Z = tercer parámetro (pz) interpolado sobre el grid X×Y
      · Color (surfacecolor) = métrica elegida
    Misma pipeline de interpolación, lighting y cámara que la surface 2D.
    """
    if not trials or not px or not py or not pz:
        return _empty()

    pts = []
    for t in trials:
        xv = t["params"].get(px)
        yv = t["params"].get(py)
        zv = t["params"].get(pz)
        if xv is None or yv is None or zv is None:
            continue
        try:
            pts.append((float(xv), float(yv), float(zv), get_m(t, metric_key)))
        except Exception:
            pass

    if len(pts) < 20:
        return _empty("insufficient data for 3D surface")

    xs, ys, p3s, ms = zip(*pts)

    # Interpolar pz sobre el grid (px, py)
    grid_pz = _build_dense_smooth_grid(xs, ys, p3s, bins=bins, smooth_passes=6)
    if grid_pz is None:
        return _empty("insufficient range")
    xc, yc, z_grid, *_ = grid_pz

    # Interpolar métrica sobre el mismo grid (px, py)
    grid_m = _build_dense_smooth_grid(xs, ys, ms, bins=bins, smooth_passes=6)
    if grid_m is None:
        return _empty("insufficient range for metric")
    _, _, m_grid, *_ = grid_m

    m_flat = [v for row in m_grid for v in row]
    m_min, m_max = min(m_flat), max(m_flat)
    label = METRIC_LABEL.get(metric_key, metric_key)

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=xc, y=yc, z=z_grid,
        surfacecolor=m_grid,
        colorscale=_colorscale(metric_key),
        cmin=m_min, cmax=m_max,
        opacity=0.96,
        lighting=dict(
            ambient=0.70,
            diffuse=0.55,
            roughness=0.85,
            specular=0.05,
            fresnel=0.03,
        ),
        lightposition=dict(x=100, y=100, z=800),
        contours=dict(
            x=dict(show=False),
            y=dict(show=False),
            z=dict(
                show=True,
                usecolormap=False,
                color="rgba(20,20,20,0.55)",
                width=1.2,
                highlightcolor="rgba(20,20,20,0.55)",
                project=dict(z=False),
            ),
        ),
        colorbar={**_cbar(metric_key), "len": 0.70},
        hovertemplate=(
            f"{px}: %{{x:.4g}}<br>"
            f"{py}: %{{y:.4g}}<br>"
            f"{pz}: %{{z:.4g}}<br>"
            f"{label}: %{{surfacecolor:.3f}}"
            "<extra></extra>"
        ),
        showlegend=False,
    ))

    _ax3d = dict(
        gridcolor="#EBEBEF",
        linecolor=_BORDER,
        zerolinecolor=_BORDER,
        backgroundcolor="#FAFBFC",
        tickfont=dict(family=_MONO, size=8, color=_DIM),
        title_font=dict(family=_FONT, size=9, color=_TEXT2),
        showspikes=False,
        showline=True,
        linewidth=0.6,
        ticks="outside",
        ticklen=2,
    )
    fig.update_layout(**_base(
        title=dict(text=f"response surface · 3D · {px} × {py} × {pz}  ·  color={label}"),
        height=500,
        margin=dict(l=0, r=16, t=42, b=8),
        scene=dict(
            xaxis=dict(**_ax3d, title=px),
            yaxis=dict(**_ax3d, title=py),
            zaxis=dict(**_ax3d, title=pz),
            bgcolor="#FAFBFC",
            aspectratio=dict(x=1.4, y=1.2, z=0.70),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=-0.08),
                eye=dict(x=1.50, y=1.50, z=0.70),
            ),
        ),
    ))
    return fig