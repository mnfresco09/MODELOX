"""
MODELOX · charts/overview.py
High-level overview charts: convergence, distribution, quality map, ranking.
"""
from __future__ import annotations

import plotly.graph_objects as go

from visual.optuna_dash.theme import (
    C, PALETTE, METRIC_LABEL, METRIC_INVERT,
    get_m, _lay, _empty, _colorscale, _MONO,
)


def make_convergence(trials: list[dict], metric_key: str = "score") -> go.Figure:
    """
    Scatter of all trials + best-so-far line + P25-P75 rolling band.
    QMC phase (first 50%) uses purple markers, TPE phase uses cyan.
    """
    if not trials:
        return _empty()
    x = [t["number"] for t in trials]
    y = [get_m(t, metric_key) for t in trials]
    invert = metric_key in METRIC_INVERT
    cur = float("inf") if invert else float("-inf")
    bsf = []
    for v in y:
        cur = min(cur, v) if invert else max(cur, v)
        bsf.append(cur)

    # Rolling P25/P75 band
    W = max(10, len(y) // 20)
    p25, p75 = [], []
    for i in range(len(y)):
        w = sorted(y[max(0, i - W) : i + 1])
        n = len(w)
        p25.append(w[int(0.25 * (n - 1))])
        p75.append(w[int(0.75 * (n - 1))])

    # Split QMC vs TPE phase
    split = len(trials) // 2
    x_qmc = x[:split]; y_qmc = y[:split]
    x_tpe = x[split:]; y_tpe = y[split:]

    fig = go.Figure()

    # Band
    fig.add_trace(go.Scatter(
        x=x + x[::-1], y=p75 + p25[::-1],
        fill="toself",
        fillcolor="rgba(0,200,255,0.06)",
        line_color="rgba(0,0,0,0)",
        name="P25–P75",
        hoverinfo="skip",
    ))

    # QMC phase markers
    fig.add_trace(go.Scatter(
        x=x_qmc, y=y_qmc, mode="markers",
        name="QMC phase",
        marker=dict(color=C["purple"], size=3, opacity=0.55, line_width=0),
        hovertemplate="Trial %{x} · " + METRIC_LABEL[metric_key] + " %{y:.2f}<extra>QMC</extra>",
    ))

    # TPE phase markers
    fig.add_trace(go.Scatter(
        x=x_tpe, y=y_tpe, mode="markers",
        name="TPE phase",
        marker=dict(color=C["accent"], size=3, opacity=0.55, line_width=0),
        hovertemplate="Trial %{x} · " + METRIC_LABEL[metric_key] + " %{y:.2f}<extra>TPE</extra>",
    ))

    # Best-so-far line
    fig.add_trace(go.Scatter(
        x=x, y=bsf, mode="lines",
        name="Best so far",
        line=dict(color=C["gold"], width=2.2),
        hovertemplate="Trial %{x} — best %{y:.2f}<extra></extra>",
    ))

    # Phase annotations
    mid_qmc = x[split // 2] if x else 0
    mid_tpe = x[split + (len(x) - split) // 2] if len(x) > split else x[-1] if x else 0
    for xp, txt, col in [(mid_qmc, "◈ QMC", C["purple"]), (mid_tpe, "◈ TPE", C["accent"])]:
        fig.add_annotation(
            x=xp, y=1.06, xref="x", yref="paper",
            text=txt, showarrow=False,
            font=dict(color=col, size=9, family=_MONO),
        )

    # Phase divider
    if x:
        fig.add_vline(
            x=x[split - 1] if split > 0 else x[0],
            line=dict(color=C["border2"], width=1, dash="dot"),
        )

    return _lay(
        fig,
        title=f"CONVERGENCE · {METRIC_LABEL[metric_key]}",
        xaxis_title="Trial #",
        yaxis_title=METRIC_LABEL[metric_key],
        height=280,
    )


def make_distribution(trials: list[dict], metric_key: str = "score") -> go.Figure:
    """
    Colored histogram of the metric distribution with Q25/Q50/Q75 markers.
    Bars are colored by quartile zone: red / orange / green / cyan.
    """
    if not trials:
        return _empty()
    vals = [v for v in (get_m(t, metric_key) for t in trials) if v != 0 or metric_key != "score"]
    if not vals:
        return _empty()
    n = len(vals)
    sv = sorted(vals)
    q25 = sv[int(0.25 * (n - 1))]
    q50 = sv[int(0.50 * (n - 1))]
    q75 = sv[int(0.75 * (n - 1))]
    best_v = min(sv) if metric_key in METRIC_INVERT else max(sv)
    mean_v = sum(vals) / n

    n_bins = min(50, max(15, n // 8))
    v_min, v_max = min(vals), max(vals)
    step = (v_max - v_min) / n_bins if v_max > v_min else 1.0
    counts = [0] * n_bins
    for v in vals:
        bi = min(int((v - v_min) / step), n_bins - 1)
        counts[bi] += 1
    xs = [v_min + (i + 0.5) * step for i in range(n_bins)]

    bar_colors = []
    for x in xs:
        if metric_key in METRIC_INVERT:
            # lower is better: color green at low values
            if x > q75:   bar_colors.append(C["red"])
            elif x > q50: bar_colors.append(C["orange"])
            elif x > q25: bar_colors.append(C["green"])
            else:          bar_colors.append(C["accent"])
        else:
            if x < q25:   bar_colors.append(C["red"])
            elif x < q50: bar_colors.append(C["orange"])
            elif x < q75: bar_colors.append(C["green"])
            else:          bar_colors.append(C["accent"])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=xs, y=counts, width=step * 0.9,
        marker=dict(color=bar_colors, line_width=0),
        opacity=0.85,
        name="Trials",
        hovertemplate=f"{METRIC_LABEL[metric_key]}: %{{x:.2f}}<br>Trials: %{{y}}<extra></extra>",
    ))

    for val, label, color in [
        (q25, "Q25", C["red"]),
        (q50, "MED", C["orange"]),
        (q75, "Q75", C["green"]),
        (best_v, "BEST", C["gold"]),
    ]:
        fig.add_vline(
            x=val, line_color=color, line_width=1.2, line_dash="dot",
            annotation_text=f"{label} {val:.1f}",
            annotation_font_color=color, annotation_font_size=8,
            annotation_position="top right",
        )
    fig.add_vline(
        x=mean_v, line_color=C["accent"], line_width=1, line_dash="dash",
        annotation_text=f"μ {mean_v:.1f}",
        annotation_font_color=C["accent"], annotation_font_size=8,
        annotation_position="top left",
    )

    return _lay(
        fig,
        title=f"DISTRIBUTION · {METRIC_LABEL[metric_key]} · n={n:,}",
        xaxis_title=METRIC_LABEL[metric_key],
        yaxis_title="Trials",
        bargap=0.04,
        height=280,
    )


def make_quality_map(
    trials: list[dict],
    mx: str = "drawdown",
    my: str = "roi",
    mc: str = "score",
) -> go.Figure:
    """
    2D scatter risk/return map. Size = sharpe. Highlight best with gold diamond.
    Shows quadrant labels (HIGH RISK/LOW RETURN etc.).
    """
    valid = [t for t in trials if t["met"]]
    if not valid:
        return _empty()

    xs = [get_m(t, mx) for t in valid]
    ys = [get_m(t, my) for t in valid]
    cs = [get_m(t, mc) for t in valid]
    sh = [get_m(t, "sharpe") for t in valid]
    nums = [t["number"] for t in valid]
    max_sh = max(sh) if sh else 1
    sizes = [5 + 14 * (max(0, s) / max_sh) ** 0.5 if max_sh > 0 else 7 for s in sh]
    best_idx = max(range(len(cs)), key=lambda i: cs[i]) if cs else 0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(
            color=cs,
            colorscale=_colorscale(mc),
            size=sizes,
            opacity=0.75,
            line=dict(color="rgba(0,0,0,0)", width=0),
            colorbar=dict(
                title=dict(text=METRIC_LABEL[mc], font=dict(size=9, color=C["dim"])),
                thickness=8,
                tickfont=dict(color=C["dim"], size=9),
                len=0.8,
                outlinewidth=0,
                bgcolor="rgba(0,0,0,0)",
            ),
        ),
        text=[
            f"#{n}  {METRIC_LABEL[my]}: {y:.2f}  {METRIC_LABEL[mx]}: {x:.2f}  Sharpe: {s:.2f}"
            for n, x, y, s in zip(nums, xs, ys, sh)
        ],
        hovertemplate="%{text}<extra></extra>",
        name="Trials",
    ))

    # Best trial
    fig.add_trace(go.Scatter(
        x=[xs[best_idx]], y=[ys[best_idx]], mode="markers+text",
        marker=dict(symbol="diamond", size=14, color=C["gold"],
                    line=dict(color=C["bg"], width=1.5)),
        text=[f" #{nums[best_idx]}"], textposition="middle right",
        textfont=dict(color=C["gold"], size=9, family=_MONO),
        hovertemplate=f"BEST Trial #{nums[best_idx]}<br>{METRIC_LABEL[my]}: {ys[best_idx]:.2f}<extra></extra>",
        name="Best",
        showlegend=False,
    ))

    # Quadrant medians
    med_x = sorted(xs)[len(xs) // 2]
    med_y = sorted(ys)[len(ys) // 2]
    fig.add_hline(y=med_y, line=dict(color=C["border2"], width=0.8, dash="dot"))
    fig.add_vline(x=med_x, line=dict(color=C["border2"], width=0.8, dash="dot"))

    # Quadrant labels
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    xr = x_max - x_min or 1; yr = y_max - y_min or 1
    for lx, ly, txt in [
        (x_min + xr * 0.02, y_max - yr * 0.05, "LOW RISK / HIGH RETURN"),
        (x_max - xr * 0.02, y_max - yr * 0.05, "HIGH RISK / HIGH RETURN"),
        (x_min + xr * 0.02, y_min + yr * 0.02, "LOW RISK / LOW RETURN"),
        (x_max - xr * 0.02, y_min + yr * 0.02, "HIGH RISK / LOW RETURN"),
    ]:
        anchor = "left" if lx < med_x else "right"
        fig.add_annotation(
            x=lx, y=ly, text=txt, showarrow=False,
            font=dict(color=C["dim"], size=8, family=_MONO),
            xanchor=anchor,
        )

    return _lay(
        fig,
        title=f"QUALITY MAP · {METRIC_LABEL[my]} vs {METRIC_LABEL[mx]}  [sz=Sharpe]",
        xaxis_title=f"Risk — {METRIC_LABEL[mx]}",
        yaxis_title=f"Return — {METRIC_LABEL[my]}",
        height=320,
    )


def make_ranking(trials: list[dict], metric_key: str = "score", top_n: int = 20) -> go.Figure:
    """Horizontal bar chart of top N trials with gradient coloring."""
    if not trials:
        return _empty()
    rev = metric_key not in METRIC_INVERT
    top = sorted(trials, key=lambda t: get_m(t, metric_key), reverse=rev)[:top_n]
    labels = [f"#{t['number']}" for t in reversed(top)]
    vals = [get_m(t, metric_key) for t in reversed(top)]
    mn, mx = (min(vals), max(vals)) if vals else (0, 1)

    colors = []
    for v in vals:
        n_norm = (v - mn) / (mx - mn) if mx > mn else 0.5
        if metric_key in METRIC_INVERT:
            n_norm = 1 - n_norm
        # interpolate cyan → gold
        r = int(0 + 255 * n_norm)
        g = int(200 + (215 - 200) * n_norm)
        b = int(255 * (1 - n_norm))
        colors.append(f"rgba({max(0,min(255,r))},{max(0,min(255,g))},{max(0,min(255,b))},0.85)")

    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker=dict(color=colors, line_width=0),
        text=[f"  {v:.2f}" for v in vals],
        textposition="inside",
        textfont=dict(color=C["bg"], size=9, family=_MONO),
        hovertemplate="Trial %{y} — " + METRIC_LABEL[metric_key] + " %{x:.2f}<extra></extra>",
    ))

    return _lay(
        fig,
        title=f"RANKING · {METRIC_LABEL[metric_key]}",
        xaxis_title=METRIC_LABEL[metric_key],
        height=max(280, 26 * min(top_n, len(top)) + 70),
        xaxis_showgrid=True,
        yaxis_showgrid=False,
        bargap=0.22,
    )


def make_stats_summary(trials: list[dict]) -> dict:
    """Return a dict of aggregate statistics over all trials."""
    if not trials:
        return {}
    scores = [t["score"] for t in trials]
    n = len(scores)
    sv = sorted(scores)
    mean_s = sum(scores) / n
    var_s = sum((v - mean_s) ** 2 for v in scores) / n
    std_s = var_s ** 0.5
    best = max(trials, key=lambda t: t["score"])
    return {
        "n_trials": n,
        "best_score": best["score"],
        "best_number": best["number"],
        "mean": mean_s,
        "median": sv[n // 2],
        "std": std_s,
        "q25": sv[int(0.25 * (n - 1))],
        "q75": sv[int(0.75 * (n - 1))],
        "min": sv[0],
        "max": sv[-1],
        "n_cut": sum(1 for t in trials if t.get("cut")),
    }
