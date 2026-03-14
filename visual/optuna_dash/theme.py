"""
MODELOX · theme.py
Professional light-mode design system — clean, institutional, data-analysis grade.

Colorscale philosophy:
  → Perceptually uniform sequential scales (dark→light→accent)
  → No saturated neon — all colors pass WCAG contrast on white
  → Single warm accent (#E85D3A) for optimal-point markers
  → Diverging scale: steel-blue ↔ warm neutral ↔ brick-red
"""
from __future__ import annotations

import plotly.graph_objects as go

# ── Color system ───────────────────────────────────────────────────────────────
# Inspired by FT / Refinitiv / institutional research tools.
# No vivid or neon colors. Single-hue where possible.

C = dict(
    bg       = "#F4F4F5",   # zinc-100 — very light page background
    surface  = "#FFFFFF",   # pure white — panels, cards
    card     = "#FFFFFF",
    border   = "#E4E4E7",   # zinc-200 — subtle dividers
    border2  = "#D4D4D8",   # zinc-300 — slightly stronger
    text     = "#18181B",   # zinc-900 — high-contrast body text
    text2    = "#52525B",   # zinc-600 — secondary
    dim      = "#A1A1AA",   # zinc-400 — muted labels, ticks
    muted    = "#D4D4D8",   # zinc-300 — disabled / placeholder
    # Single accent: deep institutional navy. No cyan, no orange.
    accent   = "#1E3A5F",   # deep navy
    accent2  = "#2D5F9A",   # slightly lighter navy
    accent_lt= "#EFF6FF",   # blue-50 — very light tint
    gold     = "#B0893A",   # muted gold — highlight for best/Pareto markers
    # Semantic: kept dark so they stay professional
    green    = "#15803D",   # green-700
    green_lt = "#F0FDF4",
    red      = "#B91C1C",   # red-700
    red_lt   = "#FEF2F2",
    orange   = "#92400E",   # amber-800
    purple   = "#4C1D95",   # violet-900
    # Marker accents for optimum points
    marker   = "#E85D3A",   # warm accent — single use for optimal markers
    marker2  = "#2563EB",   # cold accent — secondary highlights
)

_MONO = "'IBM Plex Mono','JetBrains Mono','Fira Code',monospace"
_SANS = "'Inter','system-ui',sans-serif"

# ── Metric definitions ────────────────────────────────────────────────────────
METRIC_OPTS = [
    {"label": "Score",        "value": "score"},
    {"label": "ROI (%)",      "value": "roi"},
    {"label": "Sharpe",       "value": "sharpe"},
    {"label": "Drawdown (%)", "value": "drawdown"},
    {"label": "Win Rate (%)", "value": "winrate"},
    {"label": "Pr. Factor",   "value": "profit_factor"},
    {"label": "Trades/Day",   "value": "trades_por_dia"},
    {"label": "Expectancy",   "value": "expectativa"},
    {"label": "Net PnL",      "value": "pnl_neto"},
]
METRIC_LABEL = {o["value"]: o["label"] for o in METRIC_OPTS}
METRIC_INVERT = {"drawdown"}


def get_m(trial: dict, key: str) -> float:
    if key == "score":
        return trial.get("score", 0.0)
    try:
        return float(trial.get("met", {}).get(key) or 0)
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  COLOR SCALES — Escala de pH (rojo → amarillo → verde → azul)
# ══════════════════════════════════════════════════════════════════════════════

# ── Sequential: Red → Orange → Yellow → Green → Blue ─────────────────────────
# Basado en la escala de pH (1 a 14) solicitada.
SCALE_SEQ = [
    [0.000, "#C13525"],   # 1 - Acidic (Red)
    [0.077, "#E36025"],   # 2 - Orange-Red
    [0.154, "#F49321"],   # 3 - Orange
    [0.231, "#FBB714"],   # 4 - Yellow-Orange
    [0.308, "#FFE100"],   # 5 - Yellow
    [0.385, "#B8D432"],   # 6 - Yellow-Green
    [0.462, "#7BBE43"],   # 7 - Neutral (Soft Green)
    [0.538, "#42B649"],   # 8 - Green
    [0.615, "#00A86B"],   # 9 - Green-Teal
    [0.692, "#00A59C"],   # 10 - Blue-Teal
    [0.769, "#008FC2"],   # 11 - Light Blue
    [0.846, "#1F66A8"],   # 12 - Medium Blue
    [0.923, "#1D4592"],   # 13 - Dark Blue
    [1.000, "#162863"],   # 14 - Alkaline (Navy)
]

# ── Sequential inverted: for metrics where LOW = GOOD (drawdown) ─────────────
SCALE_SEQ_R = [[1.0 - s, c] for s, c in reversed(SCALE_SEQ)]

# ── Diverging: steel-blue ↔ warm neutral ↔ brick-red ─────────────────────────
# For correlation matrices, Pareto trade-off axes, any bipolar data.
SCALE_DIV = [
    [0.00, "#1B3A5C"],   # deep steel blue
    [0.15, "#2A6496"],   # medium blue
    [0.30, "#5A9EC9"],   # sky blue
    [0.45, "#C4D9E8"],   # pale blue
    [0.50, "#F0EDE8"],   # warm neutral
    [0.55, "#E8D0B8"],   # pale sand
    [0.70, "#D4946A"],   # terracotta
    [0.85, "#B85842"],   # brick
    [1.00, "#7A2B2B"],   # deep crimson
]

# Backward-compatible alias
SCALE_DIVERGING = SCALE_DIV

# Qualitative — all muted, no vivid
PALETTE = [
    "#1E3A5F",  # navy
    "#2D6EA0",  # medium blue
    "#5B8DB8",  # steel blue
    "#91ADC4",  # pale blue
    "#6B7280",  # slate
    "#374151",  # dark slate
    "#92400E",  # amber-dark
    "#166534",  # green-dark
]

# Muted palette for multi-parameter comparison (violin/box)
PALETTE_MUTED = [
    "#5470C6", "#91CC75", "#EE6666", "#FAC858",
    "#73C0DE", "#3BA272", "#FC8452", "#9A60B4",
]


def _reverse_colorscale(cs: list) -> list:
    return [[1.0 - float(s), c] for s, c in reversed(cs)]


def _colorscale(metric_key: str) -> list:
    return SCALE_SEQ_R if metric_key in METRIC_INVERT else SCALE_SEQ


# ── Plotly base layout ────────────────────────────────────────────────────────
PL = dict(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "#FFFFFF",
    font          = dict(color=C["text2"], family=_MONO, size=10),
    title_font    = dict(color=C["text"], size=11, family=_SANS, weight="bold"),
    legend        = dict(
        bgcolor     = "rgba(255,255,255,0.92)",
        bordercolor = C["border"],
        borderwidth = 1,
        font        = dict(size=9, color=C["text2"], family=_MONO),
        x=0.01, xanchor="left", y=0.99, yanchor="top",
    ),
    margin = dict(l=44, r=16, t=32, b=32),
    hoverlabel = dict(
        bgcolor    = C["surface"],
        bordercolor= C["border2"],
        font       = dict(color=C["text"], size=10, family=_MONO),
    ),
    xaxis = dict(
        gridcolor   = "#F0F0F0",
        linecolor   = C["border2"],
        zerolinecolor = C["border2"],
        tickfont    = dict(color=C["dim"], family=_MONO, size=9),
        showline    = True, linewidth=1,
        gridwidth   = 0.5, zerolinewidth=0.8,
        showgrid    = True,
    ),
    yaxis = dict(
        gridcolor   = "#F0F0F0",
        linecolor   = C["border2"],
        zerolinecolor = C["border2"],
        tickfont    = dict(color=C["dim"], family=_MONO, size=9),
        showline    = True, linewidth=1,
        gridwidth   = 0.5, zerolinewidth=0.8,
        showgrid    = True,
    ),
)

# ── CSS style helpers ─────────────────────────────────────────────────────────
S = dict(
    page     = {"backgroundColor": C["bg"], "minHeight": "100vh", "color": C["text"]},
    card     = {"backgroundColor": C["card"], "border": f"1px solid {C['border']}",
                "borderRadius": "4px", "padding": "16px"},
    card_sm  = {"backgroundColor": C["surface"], "border": f"1px solid {C['border']}",
                "borderRadius": "4px", "padding": "8px 14px"},
    label    = {"color": C["dim"], "fontSize": "9px", "fontWeight": "600",
                "textTransform": "uppercase", "marginBottom": "2px",
                "letterSpacing": "0.10em", "fontFamily": _SANS},
    value    = {"color": C["text"], "fontSize": "19px", "fontWeight": "500",
                "lineHeight": "1", "fontFamily": _MONO},
    sub      = {"color": C["dim"], "fontSize": "9px", "marginTop": "3px",
                "fontFamily": _SANS},
    sep      = {"borderTop": f"1px solid {C['border']}", "margin": "0"},
    tab_sel  = {
        "backgroundColor": "transparent", "color": C["text"],
        "borderBottom": f"2px solid {C['accent']}",
        "borderTop": "none", "borderLeft": "none", "borderRight": "none",
        "padding": "9px 18px", "fontWeight": "600", "fontSize": "11px",
        "fontFamily": _SANS, "textTransform": "uppercase", "letterSpacing": "0.06em",
    },
    tab_unsel = {
        "backgroundColor": "transparent", "color": C["dim"],
        "border": "none", "padding": "9px 18px", "fontSize": "11px",
        "fontFamily": _SANS, "textTransform": "uppercase", "letterSpacing": "0.06em",
    },
)


# ── Chart utility helpers ─────────────────────────────────────────────────────
def _empty(msg: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False,
        font=dict(color=C["dim"], size=12, family=_SANS),
    )
    return _lay(fig)


def _lay(fig: go.Figure, **kw) -> go.Figure:
    d = dict(PL)
    d.update(kw)
    fig.update_layout(**d)
    return fig


def _fix_sub(fig: go.Figure) -> go.Figure:
    """Apply light-mode axis styling to all subplot axes."""
    axis_kw = dict(
        gridcolor="#F0F0F0",
        linecolor=C["border2"],
        zerolinecolor=C["border2"],
        tickfont=dict(color=C["dim"], family=_MONO, size=9),
        gridwidth=0.5,
        zerolinewidth=0.8,
    )
    fig.update_xaxes(**axis_kw)
    fig.update_yaxes(**axis_kw)
    for ann in fig.layout.annotations:
        if hasattr(ann, "font") and ann.font:
            try:
                ann.font.color = C["text2"]
                ann.font.family = _SANS
                ann.font.size = 10
            except Exception:
                pass
    return fig