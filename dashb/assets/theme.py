# ─────────────────────────────────────────────
#  Optuna Dashboard Pro — assets/theme.py
# ─────────────────────────────────────────────

# ── Base palette ───────────────────────────────────────────────────────────────
BG          = "#080c14"       # deepest background
SURFACE     = "#0e1420"       # card / panel background
SURFACE_2   = "#161d2e"       # slightly lighter surface
SURFACE_3   = "#1e2740"       # hover / active surfaces
BORDER      = "#243050"       # subtle borders
BORDER_SOFT = "#1a2238"

TEXT        = "#d4deff"       # primary text
TEXT_DIM    = "#6878a8"       # secondary / muted text
TEXT_FAINT  = "#2e3d5c"       # very muted

ACCENT      = "#00c2ff"       # cyan accent
ACCENT_2    = "#7b6fff"       # purple accent
SUCCESS     = "#00e5a0"       # green
WARNING     = "#ffb84d"       # amber
DANGER      = "#ff4d6d"       # red

# ── Metric gradient: red(bad) → orange → yellow → green → blue(good) ──────────
#  Used for MAXIMIZE direction.  For MINIMIZE, reverse it.
GRADIENT = [
    [0.00, "#c0392b"],   # deep red    — worst
    [0.15, "#e74c3c"],   # red
    [0.30, "#e67e22"],   # orange
    [0.45, "#f1c40f"],   # yellow
    [0.60, "#27ae60"],   # green
    [0.75, "#1abc9c"],   # teal
    [1.00, "#2980b9"],   # blue        — best
]

GRADIENT_MINIMIZE = [[round(1 - s, 4), c] for s, c in reversed(GRADIENT)]


def pick_gradient(direction: str):
    return GRADIENT if direction == "maximize" else GRADIENT_MINIMIZE


# ── Shared Plotly layout defaults ──────────────────────────────────────────────
LAYOUT = dict(
    paper_bgcolor = SURFACE,
    plot_bgcolor  = BG,
    font          = dict(color=TEXT, family="'DM Mono', 'Fira Code', monospace", size=12),
    xaxis = dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER,
                 tickfont=dict(color=TEXT_DIM)),
    yaxis = dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER,
                 tickfont=dict(color=TEXT_DIM)),
    legend = dict(bgcolor=SURFACE_2, bordercolor=BORDER, borderwidth=1,
                  font=dict(color=TEXT_DIM)),
    margin = dict(l=56, r=24, t=48, b=56),
    hoverlabel = dict(
        bgcolor    = SURFACE_3,
        bordercolor= ACCENT,
        font_color = TEXT,
        font_family= "'DM Mono', monospace",
    ),
    coloraxis_colorbar = dict(
        bgcolor    = SURFACE_2,
        bordercolor= BORDER,
        borderwidth= 1,
        tickfont   = dict(color=TEXT_DIM),
        title_font = dict(color=TEXT),
        outlinewidth= 0,
    ),
)