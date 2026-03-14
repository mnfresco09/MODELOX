"""
MODELOX · layout.py
Professional light-mode layout — clean, institutional.
"""
from __future__ import annotations

import os

import dash_bootstrap_components as dbc
from dash import dcc, html

from visual.optuna_dash.theme import C, METRIC_OPTS, S, _MONO, _SANS
from visual.optuna_dash.loader import get_study_options

# ── Global CSS ────────────────────────────────────────────────────────────────
GLOBAL_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

html, body {{
  background: {C['bg']};
  color: {C['text']};
  font-family: {_SANS};
  margin: 0; padding: 0;
  line-height: 1.5;
  font-size: 13px;
  -webkit-font-smoothing: antialiased;
}}

::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {C['border2']}; border-radius: 2px; }}

/* ── Tabs ── */
.dash-tab {{
  border: none !important;
  transition: color .1s, border-color .1s;
}}
.dash-tab--selected {{
  color: {C['text']} !important;
  border-bottom: 2px solid {C['accent']} !important;
  font-weight: 600 !important;
  background: transparent !important;
}}
.tab-content {{ padding: 0 !important; }}

/* ── Dropdowns ── */
.Select-control {{
  background-color: {C['surface']} !important;
  border: 1px solid {C['border2']} !important;
  border-radius: 3px !important;
  box-shadow: none !important;
  min-height: 28px !important;
  height: 28px !important;
}}
.Select-control:hover {{ border-color: {C['text2']} !important; }}
.is-open .Select-control {{ border-color: {C['accent']} !important; box-shadow: 0 0 0 2px {C['accent_lt']} !important; }}
.Select-placeholder, .Select-value-label {{
  color: {C['text']} !important;
  font-size: 11px !important;
  font-family: {_SANS} !important;
  line-height: 28px !important;
}}
.Select-input {{ height: 26px !important; }}
.Select-arrow-zone {{ padding-top: 0 !important; }}
.Select-arrow {{ border-top-color: {C['dim']} !important; border-width: 4px 4px 0 !important; }}
.Select-menu-outer {{
  background-color: {C['surface']} !important;
  border: 1px solid {C['border2']} !important;
  border-radius: 3px !important;
  box-shadow: 0 4px 12px rgba(0,0,0,0.10) !important;
  z-index: 9999 !important;
  margin-top: 2px !important;
}}
.VirtualizedSelectOption {{
  color: {C['text2']} !important;
  font-size: 11px !important;
  padding: 6px 10px !important;
  font-family: {_SANS} !important;
}}
.VirtualizedSelectFocusedOption {{
  background-color: {C['accent_lt']} !important;
  color: {C['text']} !important;
}}

/* ── DataTable ── */
.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner table {{
  font-size: 11px !important;
  font-family: {_MONO} !important;
  border-collapse: collapse !important;
}}
.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th {{
  background-color: {C['bg']} !important;
  color: {C['dim']} !important;
  border-bottom: 1px solid {C['border2']} !important;
  border-top: none !important; border-left: none !important; border-right: none !important;
  font-family: {_SANS} !important;
  font-weight: 600 !important;
  font-size: 10px !important;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  padding: 7px 10px !important;
}}
.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td {{
  background-color: {C['surface']} !important;
  color: {C['text']} !important;
  border-bottom: 1px solid {C['border']} !important;
  border-top: none !important; border-left: none !important; border-right: none !important;
  padding: 5px 10px !important;
}}
.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner tr:hover td {{
  background-color: {C['accent_lt']} !important;
}}
.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner tr.selected td {{
  background-color: {C['accent_lt']} !important;
}}
.dash-filter input {{
  font-size: 10px !important;
  color: {C['text']} !important;
  background: {C['surface']} !important;
  border: 1px solid {C['border2']} !important;
  font-family: {_MONO} !important;
  border-radius: 2px !important;
  padding: 3px 6px !important;
}}

/* ── Plotly charts ── */
.js-plotly-plot {{ border-radius: 3px; overflow: hidden; }}
.modebar {{ display: none !important; }}

/* ── Spinner ── */
.dash-spinner svg circle {{ stroke: {C['accent']} !important; }}
"""


# ── Reusable micro-components ─────────────────────────────────────────────────

def _mdd(ctrl_id: str, default: str = "score", width: str = "140px") -> dcc.Dropdown:
    return dcc.Dropdown(
        id=ctrl_id, options=METRIC_OPTS, value=default,
        clearable=False, searchable=False,
        style={"width": width, "fontSize": "11px"},
    )


def _lbl(txt: str) -> html.Span:
    return html.Span(txt, style={
        "color": C["dim"], "fontSize": "9px", "fontWeight": "600",
        "letterSpacing": "0.10em", "textTransform": "uppercase",
        "fontFamily": _SANS, "whiteSpace": "nowrap",
    })


def _ctrl_row(*items) -> html.Div:
    return html.Div(list(items), style={
        "display": "flex", "alignItems": "center",
        "gap": "8px", "marginBottom": "10px", "flexWrap": "wrap",
        "padding": "7px 12px",
        "backgroundColor": C["bg"],
        "border": f"1px solid {C['border']}",
        "borderRadius": "3px",
    })


def _section(title: str, *children, ctrl=None) -> html.Div:
    title_el = html.Span(title, style={
        "fontSize": "10px", "fontWeight": "600",
        "color": C["text2"], "letterSpacing": "0.08em",
        "textTransform": "uppercase", "fontFamily": _SANS,
    })
    hdr_inner = (
        html.Div([title_el, ctrl], style={
            "display": "flex", "justifyContent": "space-between",
            "alignItems": "center",
        }) if ctrl else title_el
    )
    hdr = html.Div(hdr_inner, style={"marginBottom": "12px",
                                      "paddingBottom": "8px",
                                      "borderBottom": f"1px solid {C['border']}"})
    return html.Div([hdr, *children], style={**S["card"], "marginBottom": "12px"})


def _graph(gid: str, **kw) -> dcc.Graph:
    from visual.optuna_dash.theme import _empty
    return dcc.Graph(
        id=gid, figure=_empty(),
        config={"displayModeBar": False, "responsive": True},
        style={"borderRadius": "3px", "overflow": "hidden"},
        **kw,
    )


def build_layout(db_path: str) -> html.Div:
    db_name = os.path.basename(db_path).replace(".db", "")
    studies  = get_study_options(db_path)
    default  = studies[0]["value"] if studies else None

    # ── TOP BAR ───────────────────────────────────────────────────────────────
    topbar = html.Div([
        html.Div([
            html.Span("MODELOX", style={
                "color": C["text"], "fontSize": "12px", "fontWeight": "700",
                "letterSpacing": "0.12em", "fontFamily": _SANS, "marginRight": "10px",
            }),
            html.Span("Parameter Analysis", style={
                "color": C["dim"], "fontSize": "11px", "fontFamily": _SANS,
            }),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div([
            html.Span(db_name, style={
                "backgroundColor": C["bg"],
                "border": f"1px solid {C['border2']}",
                "color": C["text2"], "fontSize": "10px",
                "fontFamily": _MONO, "padding": "3px 9px",
                "borderRadius": "3px", "marginRight": "12px",
            }),
            _lbl("Study"),
            html.Div(style={"width": "8px"}),
            dcc.Dropdown(
                id="dd-study", options=studies, value=default,
                placeholder="Select study…", clearable=False,
                style={"width": "280px", "fontSize": "11px"},
            ),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={
        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
        "padding": "10px 24px",
        "backgroundColor": C["surface"],
        "borderBottom": f"1px solid {C['border']}",
    })

    # ── KPI ROW ───────────────────────────────────────────────────────────────
    kpi_row = html.Div(id="kpi-row", style={
        "display": "flex", "gap": "6px", "padding": "10px 24px",
        "borderBottom": f"1px solid {C['border']}",
        "overflowX": "auto", "backgroundColor": C["surface"],
    })

    # ── TABS ──────────────────────────────────────────────────────────────────
    tabs = dcc.Tabs(
        id="tabs", value="tab-overview",
        style={
            "borderBottom": f"1px solid {C['border']}",
            "backgroundColor": C["surface"],
            "padding": "0 20px",
        },
        colors={"border": "transparent", "primary": C["accent"], "background": "transparent"},
        children=[
            dcc.Tab(label="Overview",        value="tab-overview", style=S["tab_unsel"], selected_style=S["tab_sel"]),
            dcc.Tab(label="Parameter Space", value="tab-params",   style=S["tab_unsel"], selected_style=S["tab_sel"]),
            dcc.Tab(label="Trial Explorer",  value="tab-trials",   style=S["tab_unsel"], selected_style=S["tab_sel"]),
            dcc.Tab(label="Asset Matrix",    value="tab-assets",   style=S["tab_unsel"], selected_style=S["tab_sel"]),
        ],
    )

    # ── CONTENT AREA ──────────────────────────────────────────────────────────
    content = dbc.Spinner(
        html.Div(id="tab-content", style={"padding": "20px 24px", "minHeight": "78vh"}),
        color=C["accent"],
        spinner_style={"width": "1.6rem", "height": "1.6rem"},
    )

    # ── FOOTER ────────────────────────────────────────────────────────────────
    footer = html.Div([
        html.Span("MODELOX", style={"color": C["text2"], "fontWeight": "600"}),
        html.Span("  ·  DB: ", style={"color": C["dim"]}),
        html.Span(db_name, style={"color": C["dim"], "fontFamily": _MONO}),
        html.Span(id="footer-text", style={"color": C["dim"]}),
    ], style={
        "padding": "5px 24px",
        "borderTop": f"1px solid {C['border']}",
        "fontSize": "9px", "fontFamily": _SANS,
        "letterSpacing": "0.06em",
        "backgroundColor": C["surface"],
        "color": C["dim"],
    })

    return html.Div([
        topbar, kpi_row,
        html.Div(tabs, style={"backgroundColor": C["surface"]}),
        content, footer,
        dcc.Store(id="store-data"),
    ], style=S["page"])
