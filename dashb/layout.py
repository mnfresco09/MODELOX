# ─────────────────────────────────────────────
#  Optuna Dashboard Pro — layout.py
# ─────────────────────────────────────────────
from __future__ import annotations

import sys, os
from dash import dcc, html
import dash_bootstrap_components as dbc

sys.path.insert(0, os.path.dirname(__file__))
from config import REFRESH_INTERVAL_MS
from assets.theme import SURFACE, SURFACE_2, BORDER, TEXT, TEXT_DIM, TEXT_FAINT, ACCENT


def _axis_controls(suffix="") -> html.Div:
    """Reusable x/y/metric selectors + aggregation toggle."""
    return html.Div([
        html.Div("X Axis — Parameter", className="control-label"),
        dcc.Dropdown(id=f"x-param-select{suffix}", clearable=False,
                     style={"fontFamily": "'DM Mono', monospace", "fontSize": "12px"}),
        html.Div("Y Axis — Parameter", className="control-label"),
        dcc.Dropdown(id=f"y-param-select{suffix}", clearable=False,
                     style={"fontFamily": "'DM Mono', monospace", "fontSize": "12px"}),
        html.Div("Depth — Metric", className="control-label"),
        dcc.Dropdown(id=f"metric-select{suffix}", clearable=False,
                     style={"fontFamily": "'DM Mono', monospace", "fontSize": "12px"}),
        html.Div("Aggregation Mode", className="control-label"),
        dcc.RadioItems(
            id=f"agg-mode{suffix}",
            options=[
                {"label": "  Mean per cell", "value": "mean"},
                {"label": "  Individual trials", "value": "individual"},
            ],
            value="mean",
            labelStyle={"display": "block", "color": TEXT_DIM,
                        "fontFamily": "'DM Mono', monospace", "fontSize": "11px",
                        "cursor": "pointer", "marginBottom": "4px"},
            style={"marginTop": "4px"},
        ),
    ], className="controls-panel")


def create_layout() -> html.Div:
    return html.Div([

        # ── Stores ─────────────────────────────────────────────────────────────
        dcc.Store(id="db-path-store",    storage_type="memory"),
        dcc.Store(id="study-data-store", storage_type="memory"),
        dcc.Store(id="study-meta-store", storage_type="memory"),
        dcc.Interval(id="interval-refresh", interval=REFRESH_INTERVAL_MS, disabled=True),

        # ── Header ─────────────────────────────────────────────────────────────
        html.Div([
            html.Div([
                "optuna",
                html.Span("·dash", style={"color": ACCENT,
                                          "textShadow": "0 0 20px rgba(0,194,255,.5)"}),
            ], className="app-logo"),

            dcc.Upload(id="upload-db",
                children=html.Div([html.Span("⬆", className="upload-icon"),
                                   "Drop .db file here or click"]),
                className="upload-zone", accept=".db", max_size=500 * 1024 * 1024),

            html.Div(id="upload-status", className="upload-status"),

            dcc.Dropdown(id="study-select", placeholder="Select study…", clearable=False,
                         style={"width": "240px", "fontFamily": "'DM Mono', monospace",
                                "fontSize": "12px"}),

            html.Div(style={"flex": "1"}),
            html.Div(id="refresh-badge", className="refresh-badge"),
        ], className="dash-header"),

        # ── Info cards ──────────────────────────────────────────────────────────
        html.Div(id="study-info-container"),

        # ── Tabs ───────────────────────────────────────────────────────────────
        html.Div([
            dcc.Tabs(id="main-tabs", value="tab-contour", className="dash-tabs", children=[

                # ── 1. Gravity Cloud ──────────────────────────────────────────
                dcc.Tab(label="⬡  Gravity Cloud", value="tab-contour",
                        className="dash-tab", selected_className="dash-tab--selected",
                        children=html.Div([
                            dbc.Row([
                                dbc.Col(_axis_controls(""), width=3),
                                dbc.Col([
                                    html.Div([
                                        dcc.Graph(id="contour-graph",
                                            config={"displayModeBar": True,
                                                    "modeBarButtonsToRemove": ["lasso2d"],
                                                    "displaylogo": False}),
                                    ], className="chart-panel", style={"padding": "14px 14px 8px"}),
                                ], width=9),
                            ], className="g-3"),
                        ], className="tab-content")),

                # ── 2. Rank Scatter ───────────────────────────────────────────
                dcc.Tab(label="◎  Rank", value="tab-rank",
                        className="dash-tab", selected_className="dash-tab--selected",
                        children=html.Div([
                            dbc.Row([
                                dbc.Col(_axis_controls("-rank"), width=3),
                                dbc.Col([
                                    html.Div([
                                        dcc.Graph(id="rank-graph",
                                            config={"displayModeBar": True,
                                                    "modeBarButtonsToRemove": ["lasso2d"],
                                                    "displaylogo": False}),
                                    ], className="chart-panel", style={"padding": "14px 14px 8px"}),
                                ], width=9),
                            ], className="g-3"),
                        ], className="tab-content")),

                # ── 3. Robustness ─────────────────────────────────────────────
                dcc.Tab(label="⬢  Robustness", value="tab-robustness",
                        className="dash-tab", selected_className="dash-tab--selected",
                        children=html.Div([
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        html.Div("X Axis — Parameter", className="control-label"),
                                        dcc.Dropdown(id="x-param-select-rob", clearable=False,
                                            style={"fontFamily": "'DM Mono', monospace",
                                                   "fontSize": "12px"}),
                                        html.Div("Y Axis — Parameter", className="control-label"),
                                        dcc.Dropdown(id="y-param-select-rob", clearable=False,
                                            style={"fontFamily": "'DM Mono', monospace",
                                                   "fontSize": "12px"}),
                                        html.Div("Metric", className="control-label"),
                                        dcc.Dropdown(id="metric-select-rob", clearable=False,
                                            style={"fontFamily": "'DM Mono', monospace",
                                                   "fontSize": "12px"}),
                                        html.Div("k Neighbors (Plateau)", className="control-label"),
                                        dcc.Slider(id="k-neighbors-slider",
                                            min=3, max=20, step=1, value=8,
                                            marks={i: {"label": str(i),
                                                       "style": {"color": TEXT_DIM,
                                                                 "fontSize": "10px"}}
                                                   for i in [3, 5, 8, 12, 16, 20]},
                                            tooltip={"placement": "bottom",
                                                     "always_visible": False}),

                                        html.Hr(style={"borderColor": BORDER, "margin": "18px 0 10px"}),
                                        html.Div([
                                            html.Div("⬢", style={"fontSize": "18px",
                                                                   "color": "#7b6fff",
                                                                   "opacity": ".6",
                                                                   "marginBottom": "8px"}),
                                            html.Div(
                                                "Volatility: std of metric — "
                                                "low = stable region.\n\n"
                                                "Neighborhood: mean score of k nearest "
                                                "neighbors — high = surrounded by good configs.\n\n"
                                                "Plateau Index: combined score "
                                                "(mean × stability × neighborhood).",
                                                style={"fontSize": "11px", "color": TEXT_FAINT,
                                                       "fontFamily": "'DM Mono', monospace",
                                                       "lineHeight": "1.8", "whiteSpace": "pre-line"},
                                            ),
                                        ]),
                                    ], className="controls-panel"),
                                ], width=3),

                                dbc.Col([
                                    # Sub-tabs for the 3 robustness views
                                    dcc.Tabs(id="rob-tabs", value="tab-rob-plateau",
                                             className="dash-tabs",
                                             children=[
                                        dcc.Tab(label="Plateau Index", value="tab-rob-plateau",
                                                className="dash-tab",
                                                selected_className="dash-tab--selected",
                                                children=html.Div([
                                                    html.Div([
                                                        dcc.Graph(id="plateau-graph",
                                                            config={"displayModeBar": True,
                                                                    "displaylogo": False}),
                                                    ], className="chart-panel",
                                                    style={"padding": "14px 14px 8px"}),
                                                ])),
                                        dcc.Tab(label="Volatility Map", value="tab-rob-std",
                                                className="dash-tab",
                                                selected_className="dash-tab--selected",
                                                children=html.Div([
                                                    html.Div([
                                                        dcc.Graph(id="std-graph",
                                                            config={"displayModeBar": True,
                                                                    "displaylogo": False}),
                                                    ], className="chart-panel",
                                                    style={"padding": "14px 14px 8px"}),
                                                ])),
                                        dcc.Tab(label="Neighborhood Score", value="tab-rob-neigh",
                                                className="dash-tab",
                                                selected_className="dash-tab--selected",
                                                children=html.Div([
                                                    html.Div([
                                                        dcc.Graph(id="neighborhood-graph",
                                                            config={"displayModeBar": True,
                                                                    "displaylogo": False}),
                                                    ], className="chart-panel",
                                                    style={"padding": "14px 14px 8px"}),
                                                ])),
                                    ]),
                                ], width=9),
                            ], className="g-3"),
                        ], className="tab-content")),

                # ── 4. Importance ─────────────────────────────────────────────
                dcc.Tab(label="▦  Importance", value="tab-importance",
                        className="dash-tab", selected_className="dash-tab--selected",
                        children=html.Div([
                            html.Div([
                                dcc.Graph(id="importance-graph",
                                    config={"displayModeBar": False, "displaylogo": False}),
                            ], className="chart-panel"),
                        ], className="tab-content")),

                # ── 5. History ────────────────────────────────────────────────
                dcc.Tab(label="◈  History", value="tab-history",
                        className="dash-tab", selected_className="dash-tab--selected",
                        children=html.Div([
                            html.Div([
                                dcc.Graph(id="history-graph",
                                    config={"displayModeBar": True, "displaylogo": False}),
                            ], className="chart-panel"),
                        ], className="tab-content")),

                # ── 6. Parallel ───────────────────────────────────────────────
                dcc.Tab(label="⋮  Parallel", value="tab-parallel",
                        className="dash-tab", selected_className="dash-tab--selected",
                        children=html.Div([
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        html.Div("Parameters to display", className="control-label"),
                                        dcc.Dropdown(id="parallel-param-select", multi=True,
                                            placeholder="Select parameters…",
                                            style={"fontFamily": "'DM Mono', monospace",
                                                   "fontSize": "12px"}),
                                    ], className="controls-panel"),
                                ], width=12, style={"marginBottom": "12px"}),
                                dbc.Col([
                                    html.Div([
                                        dcc.Graph(id="parallel-graph",
                                            config={"displayModeBar": False,
                                                    "displaylogo": False}),
                                    ], className="chart-panel",
                                    style={"padding": "14px 14px 8px"}),
                                ], width=12),
                            ], className="g-3"),
                        ], className="tab-content")),

                # ── 7. Trials Table ───────────────────────────────────────────
                dcc.Tab(label="≡  Trials", value="tab-trials",
                        className="dash-tab", selected_className="dash-tab--selected",
                        children=html.Div([
                            html.Div([
                                html.Div(id="trials-table-container"),
                            ], className="chart-panel"),
                        ], className="tab-content")),
            ]),
        ], className="dashboard-body"),

    ], style={"minHeight": "100vh", "background": "#080c14"})