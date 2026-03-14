"""
MODELOX · charts/pareto.py
Pareto frontier analysis: 2D/3D scatter, hypervolume evolution, tradeoff heatmap.

Design consistency with parameters.py:
  → Same colorscale system (ink→teal→amber sequential, steel↔brick diverging)
  → Same warm accent (#E85D3A) for Pareto-optimal markers
  → Muted dominated points, clean frontier line
  → Minimal chrome, scientific typography
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from visual.optuna_dash.theme import (
    C, PALETTE, METRIC_LABEL, METRIC_INVERT, SCALE_DIV,
    get_m, _lay, _empty, _fix_sub, _colorscale, _MONO, _SANS,
)

# ── Internal constants (same as parameters.py) ───────────────────────────────
_DIM     = "#8C919C"
_TEXT2   = "#5A5F6B"
_ACCENT  = "#E85D3A"
_FONT    = "'Inter','Helvetica Neue',sans-serif"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PARETO EFFICIENCY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def is_pareto_efficient(points: np.ndarray, maximize: np.ndarray) -> np.ndarray:
    """
    Compute Pareto efficiency mask.
    points: shape (n, m) — each row is a solution, each column an objective.
    maximize: bool array of length m — True if objective should be maximized.
    """
    n = len(points)
    if n == 0:
        return np.array([], dtype=bool)
    pts = points.copy().astype(float)
    for j in range(pts.shape[1]):
        if not maximize[j]:
            pts[:, j] = -pts[:, j]
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        dominated = np.all(pts >= pts[i], axis=1)
        dominated[i] = False
        if np.any(dominated & is_efficient):
            is_efficient[i] = False
    return is_efficient


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PARETO 2D
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_pareto2d(
    trials: list[dict],
    mx: str = "drawdown",
    my: str = "score",
    mc: str = "sharpe",
) -> go.Figure:
    """
    Pareto frontier 2D — professional scatter.

    Design:
      → ALL trials colored by metric gradient at visible opacity
      → Dominated trials: smaller, slightly muted
      → Pareto-optimal: larger, dark ring outline, connected by thin line
      → Direct frontier line (not step) — cleaner with few Pareto points
      → T/day shown in all hover tooltips
      → No legend clutter — traces self-document via visual hierarchy
    """
    if not trials:
        return _empty()

    pts = []
    for t in trials:
        xv = get_m(t, mx); yv = get_m(t, my); cv = get_m(t, mc)
        tpd = float(t.get("met", {}).get("trades_por_dia", 0) or 0)
        pts.append((xv, yv, cv, t["number"], tpd))
    xs, ys, cs, nums, tpds = zip(*pts)

    m_max = np.array([mx not in METRIC_INVERT, my not in METRIC_INVERT])
    pareto_mask = is_pareto_efficient(np.column_stack([xs, ys]), m_max)

    # Split dominated vs Pareto
    dom_idx = [i for i in range(len(xs)) if not pareto_mask[i]]
    par_idx = [i for i in range(len(xs)) if pareto_mask[i]]

    label_x = METRIC_LABEL.get(mx, mx)
    label_y = METRIC_LABEL.get(my, my)
    label_c = METRIC_LABEL.get(mc, mc)

    fig = go.Figure()

    # Layer 1: Dominated trials — color-gradient, moderate size
    if dom_idx:
        fig.add_trace(go.Scatter(
            x=[xs[i] for i in dom_idx],
            y=[ys[i] for i in dom_idx],
            mode="markers",
            marker=dict(
                color=[cs[i] for i in dom_idx],
                colorscale=_colorscale(mc),
                size=5,
                opacity=0.55,
                line=dict(width=0),
                colorbar=dict(
                    title=dict(text=label_c,
                               font=dict(size=9, color=_TEXT2, family="Inter,sans-serif")),
                    thickness=7, len=0.75,
                    tickfont=dict(color=_DIM, size=8, family=_MONO),
                    outlinewidth=0, bgcolor="rgba(0,0,0,0)",
                    tickformat=".2f",
                ),
            ),
            text=[
                f"#{nums[i]}  {label_y}={ys[i]:.2f}  {label_x}={xs[i]:.2f}  "
                f"{label_c}={cs[i]:.3f}  T/day={tpds[i]:.2f}"
                for i in dom_idx
            ],
            hovertemplate="%{text}<extra></extra>",
            name="dominated",
            showlegend=False,
        ))

    # Layer 2: Frontier line — direct connection, sorted by X
    if par_idx:
        par_sorted = sorted(par_idx, key=lambda i: xs[i])
        fig.add_trace(go.Scatter(
            x=[xs[i] for i in par_sorted],
            y=[ys[i] for i in par_sorted],
            mode="lines",
            line=dict(color="#18181B", width=1.2, dash="dot"),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Layer 3: Pareto-optimal points — larger, dark outline ring
    if par_idx:
        fig.add_trace(go.Scatter(
            x=[xs[i] for i in par_idx],
            y=[ys[i] for i in par_idx],
            mode="markers",
            marker=dict(
                color=[cs[i] for i in par_idx],
                colorscale=_colorscale(mc),
                size=10,
                opacity=0.90,
                line=dict(color="#18181B", width=1.8),
                showscale=False,
            ),
            text=[
                f"<b>Pareto #{nums[i]}</b>  {label_y}={ys[i]:.2f}  {label_x}={xs[i]:.2f}  "
                f"{label_c}={cs[i]:.3f}  T/day={tpds[i]:.2f}"
                for i in par_idx
            ],
            hovertemplate="%{text}<extra></extra>",
            name=f"Pareto ({len(par_idx)})",
            showlegend=False,
        ))

    return _lay(
        fig,
        title=f"PARETO FRONTIER · {label_y} vs {label_x}  ({len(par_idx)} optimal)",
        xaxis_title=label_x,
        yaxis_title=label_y,
        height=450,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#F0F1F3", linecolor="#E2E4E8", zeroline=False,
                   tickfont=dict(family=_MONO, size=9, color=_DIM)),
        yaxis=dict(gridcolor="#F0F1F3", linecolor="#E2E4E8", zeroline=False,
                   tickfont=dict(family=_MONO, size=9, color=_DIM)),
        margin=dict(l=52, r=16, t=42, b=44),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PARETO 3D
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_pareto3d(
    trials: list[dict],
    mx: str = "drawdown",
    my: str = "roi",
    mz: str = "score",
) -> go.Figure:
    if not trials:
        return _empty()
    pts = []
    for t in trials:
        pts.append((get_m(t, mx), get_m(t, my), get_m(t, mz), t["number"]))
    xs, ys, zs, nums = zip(*pts)

    m_max = np.array([mx not in METRIC_INVERT, my not in METRIC_INVERT, mz not in METRIC_INVERT])
    pareto_mask = is_pareto_efficient(np.column_stack([xs, ys, zs]), m_max)

    dom_x = [xs[i] for i in range(len(xs)) if not pareto_mask[i]]
    dom_y = [ys[i] for i in range(len(ys)) if not pareto_mask[i]]
    dom_z = [zs[i] for i in range(len(zs)) if not pareto_mask[i]]
    dom_n = [nums[i] for i in range(len(nums)) if not pareto_mask[i]]
    par_x = [xs[i] for i in range(len(xs)) if pareto_mask[i]]
    par_y = [ys[i] for i in range(len(ys)) if pareto_mask[i]]
    par_z = [zs[i] for i in range(len(zs)) if pareto_mask[i]]
    par_n = [nums[i] for i in range(len(nums)) if pareto_mask[i]]

    _ax3d = dict(
        gridcolor="#F0F1F3", linecolor="#E2E4E8", backgroundcolor="#FFFFFF",
        tickfont=dict(color=_DIM, family=_MONO, size=8),
        title_font=dict(color=_TEXT2, size=9, family=_FONT),
        showspikes=False, showline=True, linewidth=0.6,
    )

    fig = go.Figure()
    if dom_x:
        fig.add_trace(go.Scatter3d(
            x=dom_x, y=dom_y, z=dom_z, mode="markers",
            marker=dict(size=2.5, color="rgba(140,145,156,0.25)", line=dict(width=0)),
            text=[f"#{n}" for n in dom_n],
            hovertemplate=(f"Trial #%{{text}}<br>{METRIC_LABEL.get(mx,mx)}=%{{x:.2f}}"
                           f"<br>{METRIC_LABEL.get(my,my)}=%{{y:.2f}}"
                           f"<br>{METRIC_LABEL.get(mz,mz)}=%{{z:.2f}}<extra>dominated</extra>"),
            name="dominated",
        ))
    if par_x:
        fig.add_trace(go.Scatter3d(
            x=par_x, y=par_y, z=par_z, mode="markers",
            marker=dict(size=5, color=_ACCENT, symbol="diamond",
                        line=dict(color="#FFFFFF", width=1)),
            text=[f"#{n}" for n in par_n],
            hovertemplate=(f"Pareto #{nums}<br>{METRIC_LABEL.get(mx,mx)}=%{{x:.2f}}"
                           f"<br>{METRIC_LABEL.get(my,my)}=%{{y:.2f}}"
                           f"<br>{METRIC_LABEL.get(mz,mz)}=%{{z:.2f}}<extra>Pareto</extra>"),
            name=f"Pareto optimal ({len(par_x)})",
        ))

    return _lay(fig,
        title=f"PARETO 3D · {METRIC_LABEL.get(mx,mx)} / {METRIC_LABEL.get(my,my)} / {METRIC_LABEL.get(mz,mz)}",
        height=450, margin=dict(l=0, r=10, t=42, b=8),
        paper_bgcolor="#FAFBFC",
        font=dict(family=_FONT, size=11),
        title_font=dict(family=_MONO, size=11, color=_TEXT2),
        scene=dict(
            xaxis=dict(**_ax3d, title=METRIC_LABEL.get(mx, mx)),
            yaxis=dict(**_ax3d, title=METRIC_LABEL.get(my, my)),
            zaxis=dict(**_ax3d, title=METRIC_LABEL.get(mz, mz)),
            bgcolor="#FFFFFF",
            aspectratio=dict(x=1.2, y=1.2, z=0.8),
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.0)),
        ),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HYPERVOLUME EVOLUTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_hypervolume(trials: list[dict]) -> go.Figure:
    if not trials:
        return _empty()
    sorted_t = sorted(trials, key=lambda t: t["number"])
    nums = [t["number"] for t in sorted_t]

    scores = [t["score"] for t in sorted_t]
    rois = [get_m(t, "roi") for t in sorted_t]
    ref_s = min(scores) - 0.1 * (max(scores) - min(scores) + 1)
    ref_r = min(rois) - 0.1 * (max(rois) - min(rois) + 1)

    hv_vals = []
    current_front: list[tuple[float, float]] = []
    for t in sorted_t:
        s = t["score"]; r = get_m(t, "roi")
        current_front.append((s, r))
        pts = np.array(current_front)
        mask = is_pareto_efficient(pts, np.array([True, True]))
        pf = pts[mask]
        pf = pf[pf[:, 0].argsort()]
        hv = 0.0
        prev_s = ref_s
        for i in range(len(pf) - 1, -1, -1):
            hv += (pf[i, 0] - prev_s) * max(0, pf[i, 1] - ref_r)
            prev_s = pf[i, 0]
        hv_vals.append(max(0.0, hv))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nums, y=hv_vals, mode="lines",
        line=dict(color=C["accent"], width=1.8),
        fill="tozeroy",
        fillcolor="rgba(30,58,95,0.06)",
        hovertemplate="Trial %{x} — HV: %{y:.2f}<extra></extra>",
        name="Hypervolume",
    ))

    return _lay(fig,
        title="PARETO HYPERVOLUME EVOLUTION  [Score × ROI]",
        xaxis_title="Trial #", yaxis_title="Hypervolume",
        height=300,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TRADEOFF HEATMAP (Correlation matrix)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_tradeoff_heatmap(trials: list[dict]) -> go.Figure:
    """Pairwise Pearson correlation matrix with diverging colorscale."""
    if not trials:
        return _empty()
    keys = ["score", "roi", "sharpe", "drawdown", "winrate",
            "profit_factor", "trades_por_dia", "expectativa", "pnl_neto"]
    labels = ["SCORE", "ROI", "SHARPE", "DD", "WR", "PF", "T/DAY", "EXP", "PNL"]

    n_k = len(keys)
    vals: dict[str, list[float]] = {}
    for k in keys:
        vals[k] = [get_m(t, k) for t in trials]

    corr = [[0.0] * n_k for _ in range(n_k)]
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            if i == j:
                corr[i][j] = 1.0
                continue
            a, b = vals[ki], vals[kj]
            n = len(a)
            if n < 2:
                continue
            ma = sum(a) / n; mb = sum(b) / n
            cov = sum((a[k] - ma) * (b[k] - mb) for k in range(n))
            va = sum((v - ma) ** 2 for v in a)
            vb = sum((v - mb) ** 2 for v in b)
            corr[i][j] = round(cov / (va ** 0.5 * vb ** 0.5), 3) if va > 1e-9 and vb > 1e-9 else 0.0

    txt = [[f"{corr[i][j]:.2f}" for j in range(n_k)] for i in range(n_k)]
    fig = go.Figure(go.Heatmap(
        x=labels, y=labels, z=corr,
        colorscale=SCALE_DIV,
        zmid=0, zmin=-1, zmax=1,
        text=txt, texttemplate="%{text}",
        textfont=dict(size=9, color=C["text"], family=_MONO),
        colorbar=dict(
            title="r", thickness=7,
            tickfont=dict(color=_DIM, size=8, family=_MONO),
            outlinewidth=0, bgcolor="rgba(0,0,0,0)",
        ),
        hoverongaps=False,
        hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
    ))

    return _lay(fig,
        title="METRIC CORRELATION MATRIX (Pearson)",
        height=440, margin=dict(l=60, r=14, t=38, b=60),
    )