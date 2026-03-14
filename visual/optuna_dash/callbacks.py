"""
MODELOX · callbacks.py
Dash callback registrations — professional light mode.
Pareto charts are part of the Parameter Space tab.
"""
from __future__ import annotations

from collections import OrderedDict

import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dash_table, dcc, html

from visual.optuna_dash.theme import (
    C, PALETTE, METRIC_OPTS, METRIC_LABEL, SCALE_DIV,
    get_m, _empty, _lay, _fix_sub, _MONO, _SANS, S,
)
from visual.optuna_dash.loader import load_trials, load_trial_detail, best_t, _calc_importance
from visual.optuna_dash.perf import perf_block
from visual.optuna_dash.charts.overview import (
    make_convergence, make_distribution, make_quality_map,
    make_ranking, make_stats_summary,
)
from visual.optuna_dash.charts.parameters import (
    make_sweep, make_heatmap2d, make_contour_gravity,
    make_surface3d, make_importance_chart,
    make_interaction_matrix, make_stability_heatmap,
    make_parallel_coordinates, make_robustness_scatter,
    make_scatter3d_params,
)
from visual.optuna_dash.charts.pareto import (
    make_pareto2d,
)
from visual.optuna_dash.charts.trials import (
    make_equity_curve, make_waterfall, make_pnl_distribution,
    make_exit_breakdown, make_rolling_sharpe,
)

# ── Shared table styling ──────────────────────────────────────────────────────
_TBL_HDR = {
    "backgroundColor": C["bg"],
    "color": C["dim"],
    "borderBottom": f"1px solid {C['border2']}",
    "borderTop": "none", "borderLeft": "none", "borderRight": "none",
    "fontWeight": "600", "textTransform": "uppercase",
    "fontSize": "9px", "letterSpacing": "0.08em",
    "fontFamily": _SANS, "padding": "7px 10px",
    "position": "sticky", "top": 0,
}
_TBL_CELL = {
    "backgroundColor": C["surface"],
    "color": C["text"],
    "borderBottom": f"1px solid {C['border']}",
    "borderTop": "none", "borderLeft": "none", "borderRight": "none",
    "padding": "5px 10px", "textAlign": "right",
    "fontFamily": _MONO, "fontSize": "11px",
}


# ── Micro UI components ───────────────────────────────────────────────────────

def _graph(gid: str, **kw) -> dcc.Graph:
    return dcc.Graph(
        id=gid, figure=_empty(),
        config={"displayModeBar": False, "responsive": True},
        style={"borderRadius": "3px", "overflow": "hidden"},
        **kw,
    )


def _mdd(ctrl_id: str, default: str = "score", width: str = "140px") -> dcc.Dropdown:
    return dcc.Dropdown(
        id=ctrl_id, options=METRIC_OPTS, value=default,
        clearable=False, searchable=False,
        style={"width": width, "fontSize": "11px"},
    )


def _lbl(txt: str) -> html.Span:
    return html.Span(txt, style={
        "color": C["dim"], "fontSize": "9px", "fontWeight": "600",
        "letterSpacing": "0.09em", "textTransform": "uppercase",
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
        "color": C["text2"], "letterSpacing": "0.07em",
        "textTransform": "uppercase", "fontFamily": _SANS,
    })
    hdr_inner = (
        html.Div([title_el, ctrl], style={
            "display": "flex", "justifyContent": "space-between",
            "alignItems": "center",
        }) if ctrl else title_el
    )
    hdr = html.Div(hdr_inner, style={
        "marginBottom": "10px",
        "paddingBottom": "7px",
        "borderBottom": f"1px solid {C['border']}",
    })
    return html.Div([hdr, *children], style={**S["card"], "marginBottom": "10px"})


def _kpi_card(label: str, value: str, sub: str = "", color: str = "") -> html.Div:
    vc = color or C["text"]
    return html.Div([
        html.Div(label, style={
            "color": C["dim"], "fontSize": "8px", "fontWeight": "600",
            "letterSpacing": "0.10em", "textTransform": "uppercase",
            "fontFamily": _SANS, "marginBottom": "4px",
        }),
        html.Div(value, style={
            "color": vc, "fontSize": "17px", "fontWeight": "500",
            "lineHeight": "1", "fontFamily": _MONO,
        }),
        html.Div(sub, style={
            "color": C["dim"], "fontSize": "8px", "marginTop": "3px",
            "fontFamily": _SANS,
        }) if sub else html.Span(),
    ], style={
        **S["card_sm"],
        "borderBottom": f"2px solid {C['border2']}",
        "minWidth": "96px", "flex": "1",
    })


def _pills(params: dict) -> html.Div:
    if not params:
        return html.Span("—", style={"color": C["dim"], "fontFamily": _MONO})
    items = []
    for k, v in sorted(params.items()):
        val_str = f"{v:.4g}" if isinstance(v, float) else str(v)
        items.append(html.Span([
            html.Span(k, style={
                "color": C["text2"], "fontSize": "9px", "fontWeight": "600",
                "letterSpacing": "0.08em", "textTransform": "uppercase",
                "fontFamily": _SANS,
            }),
            html.Span("  ", style={"margin": "0 2px"}),
            html.Span(val_str, style={
                "color": C["text"], "fontFamily": _MONO,
                "fontSize": "11px", "fontWeight": "500",
            }),
        ], style={
            "backgroundColor": C["bg"],
            "border": f"1px solid {C['border2']}",
            "borderRadius": "3px", "padding": "3px 8px",
            "display": "inline-block", "marginRight": "4px", "marginBottom": "4px",
        }))
    return html.Div(items, style={"lineHeight": "2"})


# ── Main registration ─────────────────────────────────────────────────────────

def register_callbacks(app, db_path: str) -> None:
    fig_cache: OrderedDict[tuple, dict] = OrderedDict()
    FIG_CACHE_MAX = 256

    def _cached_fig(store_data, chart: str, args: tuple, builder):
        """Tiny in-memory figure cache keyed by (study, chart, controls...)."""
        study = (store_data or {}).get("study") if isinstance(store_data, dict) else None
        if not study:
            return builder()

        key = (study, chart, *args)
        hit = fig_cache.get(key)
        if hit is not None:
            fig_cache.move_to_end(key)
            return go.Figure(hit)

        with perf_block(f"chart.{chart}"):
            fig = builder()
        try:
            fig_cache[key] = fig.to_plotly_json()
            fig_cache.move_to_end(key)
            if len(fig_cache) > FIG_CACHE_MAX:
                fig_cache.popitem(last=False)
        except Exception:
            pass
        return fig


    def _get_trials(data) -> list[dict]:
        if not data or not data.get("study"):
            return []
        return load_trials(db_path, data["study"], include_equity=False)

    # ── Store ─────────────────────────────────────────────────────────────────
    @app.callback(Output("store-data", "data"), Input("dd-study", "value"))
    def store_trials(study_name):
        if not study_name:
            return None
        with perf_block("cb.store_trials.preload"):
            load_trials(db_path, study_name, include_equity=False)
        return {"study": study_name}

    # ── KPI row ───────────────────────────────────────────────────────────────
    @app.callback(Output("kpi-row", "children"), Input("store-data", "data"))
    def update_kpis(data):
        trials = _get_trials(data)
        if not trials:
            return html.Div("Select a study to begin.", style={
                "color": C["dim"], "fontSize": "11px", "padding": "6px 0",
                "fontFamily": _SANS,
            })
        bt = best_t(trials)
        scores = [t["score"] for t in trials]
        avg_s = sum(scores) / len(scores) if scores else 0
        m = bt.get("met", {}) if bt else {}
        is_gl = (data or {}).get("study", "").endswith(("-global", "_global"))

        roi = m.get("roi", 0)
        sharpe = m.get("sharpe", 0)
        dd = m.get("drawdown", 0)
        wr = m.get("winrate", 0)
        pf = m.get("profit_factor", 0)

        cards = [
            _kpi_card("Best Score",    f"{bt['score']:.1f}" if bt else "—",  f"Trial #{bt['number']}" if bt else ""),
            _kpi_card("ROI",           f"{roi:.2f}%",   "best trial",  C["green"] if roi > 0 else C["red"]),
            _kpi_card("Sharpe",        f"{sharpe:.3f}", "best trial",  C["green"] if sharpe > 1 else C["text2"]),
            _kpi_card("Max DD",        f"{dd:.2f}%",    "best trial",  C["red"] if dd > 15 else C["text2"]),
            _kpi_card("Win Rate",      f"{wr:.1f}%",    "best trial",  C["green"] if wr > 55 else C["text2"]),
            _kpi_card("Profit Factor", f"{pf:.2f}",     "best trial",  C["green"] if pf > 1.5 else C["text2"]),
            _kpi_card("Trades",        str(int(m.get("n_trades", len(bt.get("equity", []))) if bt else 0)), "best trial"),
            _kpi_card("Trials",        f"{len(trials):,}", f"avg {avg_s:.1f}"),
        ]
        if is_gl:
            acts = sorted(set(t["activo"] for t in trials if t["activo"]))
            cards.append(_kpi_card("Assets", str(len(acts)), "  ".join(acts[:5])))

        return html.Div(cards, style={
            "display": "flex", "gap": "6px", "flexWrap": "nowrap", "overflowX": "auto",
        })

    # ── Footer ────────────────────────────────────────────────────────────────
    @app.callback(Output("footer-text", "children"), Input("store-data", "data"))
    def update_footer(data):
        trials = _get_trials(data)
        if not trials:
            return ""
        bt = best_t(trials)
        study = (data or {}).get("study", "")
        return f"  ·  {len(trials):,} trials  ·  best {bt['score']:.2f}  ·  {study}"

    # ── Main tab renderer ─────────────────────────────────────────────────────
    @app.callback(
        Output("tab-content", "children"),
        Input("tabs", "value"),
        Input("store-data", "data"),
    )
    def render_tab(tab, data):
        no_data = html.Div(
            "Select a study to begin.",
            style={"color": C["dim"], "padding": "60px 0", "textAlign": "center",
                   "fontSize": "13px", "fontFamily": _SANS},
        )
        trials = _get_trials(data)
        if not trials:
            return no_data

        pnames = sorted(set(p for t in trials for p in t["params"]))
        px_opts = [{"label": p, "value": p} for p in pnames]
        p0 = pnames[0] if pnames else None
        p1 = pnames[1] if len(pnames) > 1 else p0
        study = (data or {}).get("study", "")
        is_gl = study.endswith(("-global", "_global"))

        def _pdrop(did, w="160px", idx=0):
            default = pnames[idx] if len(pnames) > idx else (pnames[0] if pnames else None)
            return dcc.Dropdown(
                id=did, options=px_opts, value=default,
                clearable=False, searchable=True,
                style={"width": w, "fontSize": "11px"},
            )

        # ── OVERVIEW ──────────────────────────────────────────────────────────
        if tab == "tab-overview":
            stats = make_stats_summary(trials)
            stat_row = html.Div([
                _kpi_card("Trials",  f"{stats.get('n_trials', 0):,}",  ""),
                _kpi_card("Best",    f"{stats.get('best_score', 0):.2f}", f"#{stats.get('best_number', 0)}"),
                _kpi_card("Mean",    f"{stats.get('mean', 0):.2f}",     ""),
                _kpi_card("Median",  f"{stats.get('median', 0):.2f}",   ""),
                _kpi_card("Std",     f"{stats.get('std', 0):.2f}",      ""),
                _kpi_card("Q25",     f"{stats.get('q25', 0):.2f}",      ""),
                _kpi_card("Q75",     f"{stats.get('q75', 0):.2f}",      ""),
                _kpi_card("Filtered",f"{stats.get('n_cut', 0)}",        "low TPD"),
            ], style={"display": "flex", "gap": "6px", "flexWrap": "wrap"})

            return html.Div([
                dbc.Row([
                    dbc.Col(_section("Convergence",
                        _ctrl_row(_lbl("Metric"), _mdd("ov-m-conv", "score", "150px")),
                        _graph("ov-g-conv"),
                    ), md=7),
                    dbc.Col(_section("Distribution",
                        _ctrl_row(_lbl("Metric"), _mdd("ov-m-dist", "score", "150px")),
                        _graph("ov-g-dist"),
                    ), md=5),
                ], className="g-3 mb-3"),
                dbc.Row([
                    dbc.Col(_section("Quality Map",
                        _ctrl_row(
                            _lbl("Y"), _mdd("ov-m-qm-y", "roi", "120px"),
                            _lbl("X"), _mdd("ov-m-qm-x", "drawdown", "120px"),
                            _lbl("Color"), _mdd("ov-m-qm-c", "score", "120px"),
                        ),
                        _graph("ov-g-qmap"),
                    ), md=8),
                    dbc.Col(_section("Ranking",
                        _ctrl_row(_lbl("Metric"), _mdd("ov-m-rank", "score", "140px")),
                        _graph("ov-g-rank"),
                    ), md=4),
                ], className="g-3 mb-3"),
                _section("Score Statistics", stat_row),
            ])

        # ── PARAMETER SPACE ───────────────────────────────────────────────────
        if tab == "tab-params":
            return html.Div([
                # Row 1 — Contour + Surface 3D
                dbc.Row([
                    dbc.Col(_section("Response Surface — Contour",
                        _ctrl_row(
                            _lbl("X"), _pdrop("ps-ct-px", "160px", 0),
                            _lbl("Y"), _pdrop("ps-ct-py", "160px", 1),
                            _lbl("Metric"), _mdd("ps-m-ct", "score", "140px"),
                        ),
                        _graph("ps-g-contour"),
                    ), md=6),
                    dbc.Col(_section("Response Surface — 3D",
                        _ctrl_row(
                            _lbl("X"), _pdrop("ps-sf-px", "160px", 0),
                            _lbl("Y"), _pdrop("ps-sf-py", "160px", 1),
                            _lbl("Metric"), _mdd("ps-m-sf", "score", "140px"),
                        ),
                        _graph("ps-g-surface"),
                    ), md=6),
                ], className="g-3 mb-3"),
                # Row 2 — Heatmap 2D + Pareto Frontier
                dbc.Row([
                    dbc.Col(_section("Heatmap 2D",
                        _ctrl_row(
                            _lbl("X"), _pdrop("ps-hm-px", "160px", 0),
                            _lbl("Y"), _pdrop("ps-hm-py", "160px", 1),
                            _lbl("Metric"), _mdd("ps-m-hm", "score", "140px"),
                        ),
                        _graph("ps-g-heatmap"),
                    ), md=6),
                    dbc.Col(_section("Pareto Frontier",
                        _ctrl_row(
                            _lbl("Y"), _mdd("ps-pa-my", "score", "130px"),
                            _lbl("X"), _mdd("ps-pa-mx", "drawdown", "130px"),
                            _lbl("Color"), _mdd("ps-pa-mc", "sharpe", "130px"),
                        ),
                        _graph("ps-g-pareto2d"),
                    ), md=6),
                ], className="g-3 mb-3"),
                # Row 3 — Stability Heatmap + Robustness Score
                dbc.Row([
                    dbc.Col(_section("Stability Heatmap",
                        _ctrl_row(
                            _lbl("X"), _pdrop("ps-st-px", "160px", 0),
                            _lbl("Y"), _pdrop("ps-st-py", "160px", 1),
                            _lbl("Metric"), _mdd("ps-m-st", "score", "140px"),
                        ),
                        _graph("ps-g-stability"),
                    ), md=6),
                    dbc.Col(_section("Robustness Score",
                        _ctrl_row(
                            _lbl("Metric"), _mdd("ps-m-rob", "score", "140px"),
                        ),
                        _graph("ps-g-robustness"),
                    ), md=6),
                ], className="g-3 mb-3"),
                # Row 4 — Parallel Coordinates (full width)
                dbc.Row([
                    dbc.Col(_section("Parallel Coordinates",
                        _ctrl_row(
                            _lbl("Color"), _mdd("ps-m-pcoord", "score", "140px"),
                        ),
                        _graph("ps-g-pcoord"),
                    ), md=12),
                ], className="g-3 mb-3"),
                # Row 5 — Sweep + Importance
                dbc.Row([
                    dbc.Col(_section("Sensitivity Sweep",
                        _ctrl_row(
                            _lbl("Parameter"), _pdrop("ps-param", "200px"),
                            _lbl("Metric"), _mdd("ps-m-sweep", "score", "140px"),
                        ),
                        _graph("ps-g-sweep"),
                    ), md=6),
                    dbc.Col(_section("Parameter Importance",
                        _ctrl_row(_lbl("Metric"), _mdd("ps-m-imp", "score", "150px")),
                        _graph("ps-g-imp"),
                    ), md=6),
                ], className="g-3 mb-3"),
                # Row 6 — Interaction Matrix (full width)
                dbc.Row([
                    dbc.Col(_section("Parameter Interaction Matrix",
                        _ctrl_row(
                            _lbl("Metric"), _mdd("ps-m-interact", "score", "140px"),
                        ),
                        _graph("ps-g-interact"),
                    ), md=12),
                ], className="g-3 mb-3"),
                # Row 7 — Scatter 3D (3 variables)
                dbc.Row([
                    dbc.Col(_section("Topology 3D · 3 Parámetros · Iso-superficies",
                        _ctrl_row(
                            _lbl("X"), _pdrop("ps-s3-px", "160px", 0),
                            _lbl("Y"), _pdrop("ps-s3-py", "160px", 1),
                            _lbl("Z"), _pdrop("ps-s3-pz", "160px", 2),
                            _lbl("Color"), _mdd("ps-m-s3", "score", "140px"),
                        ),
                        _graph("ps-g-s3d"),
                    ), md=12),
                ], className="g-3 mb-3"),
            ])

        # ── TRIAL EXPLORER ────────────────────────────────────────────────────
        if tab == "tab-trials":
            rows = []
            for t in sorted(trials, key=lambda t: t["score"], reverse=True):
                row = {"#": t["number"], "Score": round(t["score"], 2)}
                if t.get("activo"):
                    row["Asset"] = t["activo"]
                row.update({
                    "ROI%":   round(t["met"].get("roi", 0), 2),
                    "Sharpe": round(t["met"].get("sharpe", 0), 4),
                    "DD%":    round(t["met"].get("drawdown", 0), 2),
                    "WR%":    round(t["met"].get("winrate", 0), 1),
                    "PF":     round(t["met"].get("profit_factor", 0), 2),
                    "T/Day":  round(float(t["met"].get("trades_por_dia", 0) or 0), 3),
                    "Trades": int(t["met"].get("n_trades", len(t.get("equity", []))) or 0),
                })
                for p in pnames[:8]:
                    v = t["params"].get(p)
                    row[p] = round(v, 4) if isinstance(v, float) else (v or "")
                rows.append(row)
            cols = [{"name": c, "id": c} for c in (rows[0] if rows else {})]

            return html.Div([
                _section(
                    "Trial Universe",
                    html.Div(
                        f"{len(trials):,} trials  ·  click row to inspect  ·  use column headers to filter",
                        style={"color": C["dim"], "fontSize": "9px",
                               "marginBottom": "8px", "fontFamily": _SANS},
                    ),
                    dash_table.DataTable(
                        id="te-table",
                        data=rows, columns=cols,
                        row_selectable="single", selected_rows=[0],
                        style_table={
                            "overflowX": "auto", "maxHeight": "46vh",
                            "overflowY": "auto", "border": f"1px solid {C['border']}",
                        },
                        style_header=_TBL_HDR,
                        style_cell=_TBL_CELL,
                        style_cell_conditional=[
                            {"if": {"column_id": "#"}, "textAlign": "center",
                             "width": "48px", "fontWeight": "600", "color": C["text2"]},
                            {"if": {"column_id": "Asset"}, "textAlign": "center"},
                            {"if": {"column_id": "Score"},
                             "fontWeight": "600", "color": C["accent"]},
                        ],
                        style_data_conditional=[
                            {"if": {"filter_query": "{DD%} > 20", "column_id": "DD%"},
                             "color": C["red"]},
                            {"if": {"filter_query": "{Sharpe} > 1", "column_id": "Sharpe"},
                             "color": C["green"]},
                            {"if": {"filter_query": "{PF} > 2", "column_id": "PF"},
                             "color": C["green"]},
                            {"if": {"filter_query": "{ROI%} < 0", "column_id": "ROI%"},
                             "color": C["red"]},
                            {"if": {"state": "selected"},
                             "backgroundColor": C["accent_lt"]},
                        ],
                        sort_action="native",
                        filter_action="native",
                        page_size=60,
                        fixed_rows={"headers": True},
                    ),
                ),
                html.Div(id="te-detail"),
            ])

        # ── ASSET MATRIX ──────────────────────────────────────────────────────
        if tab == "tab-assets":
            assets_with_data = [t["activo"] for t in trials if t.get("activo")]
            if not assets_with_data:
                return html.Div(
                    "Asset Matrix is available for global studies only.",
                    style={"color": C["dim"], "padding": "60px 0",
                           "textAlign": "center", "fontFamily": _SANS, "fontSize": "13px"},
                )

            def _box_asset(metric_key: str) -> go.Figure:
                by: dict[str, list] = {}
                for t in trials:
                    a = t["activo"] or "—"
                    v = get_m(t, metric_key)
                    try:
                        by.setdefault(a, []).append(float(v))
                    except Exception:
                        pass
                if len(by) < 2:
                    return _empty("Single asset")
                fig = go.Figure()
                for i, (asset, vals) in enumerate(sorted(by.items())):
                    col = PALETTE[i % len(PALETTE)]
                    r2, g2, b2 = int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16)
                    fig.add_trace(go.Box(
                        y=vals, name=asset, boxmean="sd",
                        marker_color=col, line_color=col, line_width=1,
                        fillcolor=f"rgba({r2},{g2},{b2},0.08)",
                    ))
                return _lay(fig, title=METRIC_LABEL[metric_key],
                            yaxis_title=METRIC_LABEL[metric_key],
                            showlegend=False, height=260)

            def _radar_assets() -> go.Figure:
                KEYS = ["sharpe", "roi", "winrate", "profit_factor"]
                LBLS = ["Sharpe", "ROI", "Win Rate", "Profit Factor"]
                by: dict[str, list] = {}
                for t in trials:
                    a = t["activo"] or "—"
                    if not t["met"]:
                        continue
                    by.setdefault(a, []).append(
                        [float(t["met"].get(k, 0) or 0) for k in KEYS]
                    )
                if len(by) < 2:
                    return _empty()
                meds: dict[str, list] = {}
                for asset, rws in by.items():
                    nc = len(KEYS)
                    med = []
                    for ci in range(nc):
                        cv = sorted(r[ci] for r in rws)
                        mid = len(cv) // 2
                        med.append(cv[mid] if len(cv) % 2 else (cv[mid - 1] + cv[mid]) / 2)
                    meds[asset] = med
                nc = len(KEYS)
                col_min = [min(meds[a][c] for a in meds) for c in range(nc)]
                col_max = [max(meds[a][c] for a in meds) for c in range(nc)]

                def _n(v, c):
                    rng = col_max[c] - col_min[c]
                    return (v - col_min[c]) / rng if rng > 1e-9 else 0.5

                cats = LBLS + [LBLS[0]]
                fig = go.Figure()
                for i, (asset, vals) in enumerate(sorted(meds.items())):
                    col = PALETTE[i % len(PALETTE)]
                    r2, g2, b2 = int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16)
                    nr = [_n(vals[ci], ci) for ci in range(nc)] + [_n(vals[0], 0)]
                    fig.add_trace(go.Scatterpolar(
                        r=nr, theta=cats, fill="toself", name=asset,
                        line=dict(color=col, width=1),
                        fillcolor=f"rgba({r2},{g2},{b2},0.07)",
                    ))
                return _lay(fig,
                            title="Asset Comparison (Normalized)",
                            polar=dict(
                                bgcolor=C["surface"],
                                radialaxis=dict(visible=True, range=[0, 1],
                                               color=C["dim"], gridcolor=C["border"],
                                               tickfont=dict(size=8, color=C["dim"])),
                                angularaxis=dict(color=C["text2"], gridcolor=C["border"],
                                                tickfont=dict(size=9, color=C["text2"])),
                            ),
                            height=340)

            def _score_heatmap() -> go.Figure:
                keys = ["score", "roi", "sharpe", "drawdown", "winrate", "profit_factor"]
                lbls = ["Score", "ROI", "Sharpe", "DD%", "WR%", "PF"]
                assets = sorted(set(t["activo"] for t in trials if t["activo"]))
                if len(assets) < 2:
                    return _empty("Single asset")
                z = []
                for a in assets:
                    row_t = [t for t in trials if t["activo"] == a]
                    row = []
                    for k in keys:
                        vals = [get_m(t, k) for t in row_t]
                        row.append(sum(vals) / len(vals) if vals else 0)
                    z.append(row)
                txt = [[f"{v:.2f}" for v in row] for row in z]
                fig = go.Figure(go.Heatmap(
                    x=lbls, y=assets, z=z,
                    colorscale=SCALE_DIV,
                    text=txt, texttemplate="%{text}",
                    textfont=dict(size=9, family=_MONO, color=C["text"]),
                    colorbar=dict(thickness=8, tickfont=dict(color=C["dim"], size=9)),
                    hovertemplate="%{y}  ·  %{x}: %{z:.2f}<extra></extra>",
                ))
                return _lay(fig, title="Asset × Metric",
                            height=max(240, 36 * len(assets) + 80))

            return html.Div([
                dbc.Row([
                    dbc.Col(_section("Score",        _box_asset("score")),        md=4),
                    dbc.Col(_section("Sharpe",       _box_asset("sharpe")),       md=4),
                    dbc.Col(_section("Drawdown",     _box_asset("drawdown")),     md=4),
                ], className="g-3 mb-3"),
                dbc.Row([
                    dbc.Col(_section("Win Rate",     _box_asset("winrate")),      md=4),
                    dbc.Col(_section("Profit Factor",_box_asset("profit_factor")),md=4),
                    dbc.Col(_section("ROI",          _box_asset("roi")),          md=4),
                ], className="g-3 mb-3"),
                dbc.Row([
                    dbc.Col(_section("Radar",         _radar_assets()),           md=5),
                    dbc.Col(_section("Metric Heatmap",_score_heatmap()),          md=7),
                ], className="g-3"),
            ])

        return html.Div("—", style={"color": C["dim"]})

    # ── Overview callbacks ────────────────────────────────────────────────────
    @app.callback(Output("ov-g-conv",  "figure"),
                  Input("ov-m-conv",   "value"), State("store-data", "data"))
    def upd_ov_conv(m, d):
        t = _get_trials(d)
        return _cached_fig(d, "ov-conv", (m,), lambda: make_convergence(t, m)) if t else _empty()

    @app.callback(Output("ov-g-dist",  "figure"),
                  Input("ov-m-dist",   "value"), State("store-data", "data"))
    def upd_ov_dist(m, d):
        t = _get_trials(d)
        return _cached_fig(d, "ov-dist", (m,), lambda: make_distribution(t, m)) if t else _empty()

    @app.callback(Output("ov-g-qmap",  "figure"),
                  Input("ov-m-qm-y",   "value"), Input("ov-m-qm-x", "value"),
                  Input("ov-m-qm-c",   "value"), State("store-data", "data"))
    def upd_ov_qmap(my, mx, mc, d):
        t = _get_trials(d)
        return _cached_fig(d, "ov-qmap", (my, mx, mc), lambda: make_quality_map(t, mx, my, mc)) if t else _empty()

    @app.callback(Output("ov-g-rank",  "figure"),
                  Input("ov-m-rank",   "value"), State("store-data", "data"))
    def upd_ov_rank(m, d):
        t = _get_trials(d)
        return _cached_fig(d, "ov-rank", (m, 20), lambda: make_ranking(t, m, top_n=20)) if t else _empty()

    # ── Parameter Space callbacks ─────────────────────────────────────────────
    @app.callback(Output("ps-g-sweep", "figure"),
                  Input("ps-param",    "value"), Input("ps-m-sweep", "value"),
                  State("store-data",  "data"))
    def upd_ps_sweep(p, m, d):
        t = _get_trials(d)
        return _cached_fig(d, "ps-sweep", (p, m), lambda: make_sweep(t, p, m)) if t and p else _empty()

    @app.callback(Output("ps-g-imp",   "figure"),
                  Input("ps-m-imp",    "value"), State("store-data", "data"))
    def upd_ps_imp(m, d):
        t = _get_trials(d)
        return _cached_fig(d, "ps-imp", (m,), lambda: make_importance_chart(t, m)) if t else _empty()

    @app.callback(Output("ps-g-heatmap", "figure"),
                  Input("ps-hm-px",   "value"), Input("ps-hm-py", "value"),
                  Input("ps-m-hm",    "value"), State("store-data", "data"))
    def upd_ps_heatmap(px, py, m, d):
        t = _get_trials(d)
        return _cached_fig(d, "ps-heatmap", (px, py, m), lambda: make_heatmap2d(t, px, py, m)) if t and px and py else _empty()

    @app.callback(Output("ps-g-contour", "figure"),
                  Input("ps-ct-px",   "value"), Input("ps-ct-py", "value"),
                  Input("ps-m-ct",    "value"), State("store-data", "data"))
    def upd_ps_contour(px, py, m, d):
        t = _get_trials(d)
        return _cached_fig(d, "ps-contour", (px, py, m), lambda: make_contour_gravity(t, px, py, m)) if t and px and py else _empty()

    @app.callback(Output("ps-g-surface", "figure"),
                  Input("ps-sf-px",   "value"), Input("ps-sf-py", "value"),
                  Input("ps-m-sf",    "value"), State("store-data", "data"))
    def upd_ps_surface(px, py, m, d):
        t = _get_trials(d)
        return _cached_fig(d, "ps-surface", (px, py, m), lambda: make_surface3d(t, px, py, m)) if t and px and py else _empty()

    # ── Pareto callbacks (inside Parameter Space tab) ─────────────────────────
    @app.callback(Output("ps-g-pareto2d", "figure"),
                  Input("ps-pa-my",  "value"), Input("ps-pa-mx",  "value"),
                  Input("ps-pa-mc",  "value"), State("store-data", "data"))
    def upd_ps_pareto2d(my, mx, mc, d):
        t = _get_trials(d)
        return _cached_fig(d, "ps-pareto2d", (my, mx, mc), lambda: make_pareto2d(t, mx, my, mc)) if t else _empty()

    # ── Stability Heatmap ─────────────────────────────────────────────────────
    @app.callback(Output("ps-g-stability", "figure"),
                  Input("ps-st-px",  "value"), Input("ps-st-py", "value"),
                  Input("ps-m-st",   "value"), State("store-data", "data"))
    def upd_ps_stability(px, py, m, d):
        t = _get_trials(d)
        return _cached_fig(d, "ps-stability", (px, py, m), lambda: make_stability_heatmap(t, px, py, m)) if t and px and py else _empty()

    # ── Robustness Score ──────────────────────────────────────────────────────
    @app.callback(Output("ps-g-robustness", "figure"),
                  Input("ps-m-rob",  "value"), State("store-data", "data"))
    def upd_ps_robustness(m, d):
        t = _get_trials(d)
        return _cached_fig(d, "ps-robustness", (m,), lambda: make_robustness_scatter(t, m)) if t else _empty()

    # ── Parallel Coordinates ──────────────────────────────────────────────────
    @app.callback(Output("ps-g-pcoord", "figure"),
                  Input("ps-m-pcoord", "value"), State("store-data", "data"))
    def upd_ps_pcoord(m, d):
        t = _get_trials(d)
        return _cached_fig(d, "ps-pcoord", (m,), lambda: make_parallel_coordinates(t, m)) if t else _empty()

    # ── Parameter Interaction Matrix ──────────────────────────────────────────
    @app.callback(Output("ps-g-interact", "figure"),
                  Input("ps-m-interact", "value"), State("store-data", "data"))
    def upd_ps_interact(m, d):
        t = _get_trials(d)
        return _cached_fig(d, "ps-interact", (m,), lambda: make_interaction_matrix(t, m)) if t else _empty()

    # ── Scatter 3D (3 parámetros) ─────────────────────────────────────────────
    @app.callback(Output("ps-g-s3d",   "figure"),
                  Input("ps-s3-px",    "value"), Input("ps-s3-py", "value"),
                  Input("ps-s3-pz",    "value"), Input("ps-m-s3",  "value"),
                  State("store-data",  "data"))
    def upd_ps_s3d(px, py, pz, m, d):
        t = _get_trials(d)
        return _cached_fig(d, "ps-s3d", (px, py, pz, m),
                           lambda: make_scatter3d_params(t, px, py, pz, m)) if t and px and py and pz else _empty()

    # ── Trial Explorer: row-selection detail ──────────────────────────────────
    @app.callback(
        Output("te-detail",     "children"),
        Input("te-table",       "selected_rows"),
        State("te-table",       "data"),
        State("store-data",     "data"),
    )
    def show_trial_detail(selected_rows, table_data, store_data):
        if not selected_rows or not store_data or not table_data:
            return html.Div()
        trial_num = table_data[selected_rows[0]]["#"]
        trials = _get_trials(store_data)
        trial = next((x for x in trials if x["number"] == trial_num), None)
        if not trial:
            return html.Div()

        detail = load_trial_detail(
            db_path,
            store_data.get("study", ""),
            trial.get("id"),
        )
        met = detail.get("met", {}) if isinstance(detail, dict) else {}
        equity = detail.get("equity", []) if isinstance(detail, dict) else []

        trial = {
            **trial,
            "met": met or trial.get("met", {}),
            "equity": equity,
            "activo": detail.get("activo", trial.get("activo", "")) if isinstance(detail, dict) else trial.get("activo", ""),
            "cut": detail.get("cut", trial.get("cut", False)) if isinstance(detail, dict) else trial.get("cut", False),
        }

        m  = trial.get("met", {})
        eq = trial.get("equity", [])

        def _mc(lbl, val, color=None):
            return html.Div([
                html.Div(lbl, style=S["label"]),
                html.Div(val, style={
                    **S["value"], "fontSize": "16px",
                    "color": color or C["text"],
                }),
            ], style={**S["card_sm"]})

        met_row = html.Div([
            _mc("Trial",         str(trial_num)),
            _mc("Score",         f"{trial['score']:.2f}",              C["accent"]),
            _mc("ROI",           f"{m.get('roi',0):.2f}%",            C["green"] if m.get("roi",0) > 0 else C["red"]),
            _mc("Sharpe",        f"{m.get('sharpe',0):.4f}",          C["green"] if m.get("sharpe",0) > 1 else C["text2"]),
            _mc("Max DD",        f"{m.get('drawdown',0):.2f}%",       C["red"] if m.get("drawdown",0) > 15 else C["text2"]),
            _mc("Win Rate",      f"{m.get('winrate',0):.1f}%"),
            _mc("Profit Factor", f"{m.get('profit_factor',0):.2f}",   C["green"] if m.get("profit_factor",0) > 1.5 else C["text2"]),
            _mc("Trades",        str(int(m.get("n_trades", len(eq)) or 0))),
            _mc("Expectancy",    f"${m.get('expectativa',0):.2f}"),
            _mc("Net PnL",       f"${m.get('pnl_neto',0):,.0f}",      C["green"] if m.get("pnl_neto",0) > 0 else C["red"]),
        ], style={"display": "flex", "gap": "6px", "flexWrap": "wrap", "marginBottom": "10px"})

        params_block = html.Div([
            html.Div("Parameters", style={**S["label"], "marginBottom": "7px"}),
            _pills(trial["params"]),
        ], style={**S["card"], "marginBottom": "10px"})

        detail_charts = []
        if eq:
            detail_charts.append(dbc.Row([
                dbc.Col(_section(f"Trial {trial_num}  —  Equity",
                    dcc.Graph(figure=make_equity_curve(trial),
                              config={"displayModeBar": False, "responsive": True}),
                ), md=12),
            ], className="g-3 mb-3"))
            detail_charts.append(dbc.Row([
                dbc.Col(_section("Trade Waterfall",
                    dcc.Graph(figure=make_waterfall(trial),
                              config={"displayModeBar": False, "responsive": True}),
                ), md=6),
                dbc.Col(_section("P&L Distribution",
                    dcc.Graph(figure=make_pnl_distribution(trial),
                              config={"displayModeBar": False, "responsive": True}),
                ), md=3),
                dbc.Col(_section("Exit Breakdown",
                    dcc.Graph(figure=make_exit_breakdown(trial),
                              config={"displayModeBar": False, "responsive": True}),
                ), md=3),
            ], className="g-3 mb-3"))
            detail_charts.append(dbc.Row([
                dbc.Col(_section("Rolling Statistics",
                    dcc.Graph(figure=make_rolling_sharpe(trial),
                              config={"displayModeBar": False, "responsive": True}),
                ), md=12),
            ], className="g-3"))

        return html.Div([
            html.Hr(style={**S["sep"], "margin": "12px 0"}),
            met_row,
            params_block,
            *detail_charts,
        ])