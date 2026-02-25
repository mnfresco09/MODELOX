# ─────────────────────────────────────────────
#  Optuna Dashboard Pro — components/study_info.py
# ─────────────────────────────────────────────
from __future__ import annotations

import sys, os
from dash import html

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from assets.theme import TEXT, TEXT_DIM, TEXT_FAINT, ACCENT, SUCCESS, WARNING, DANGER


def build_study_info(meta: dict) -> html.Div:
    is_running = meta.get("is_running", False)
    direction  = meta.get("direction", "?")
    best_val   = meta.get("best_value")
    best_trial = meta.get("best_trial_number")
    n_complete = meta.get("trial_count", 0)
    study_name = meta.get("study_name", "—")

    best_str = f"{best_val:.6f}" if best_val is not None else "—"
    best_tr  = f"#{best_trial}" if best_trial is not None else "—"

    cards = [
        # Status
        html.Div([
            html.Div("Status", className="info-card-label"),
            html.Div(
                html.Span([
                    html.Span("", className="dot"),
                    " RUNNING" if is_running else " COMPLETE",
                ], className=f"status-pill {'running' if is_running else 'complete'}"),
                style={"marginTop": "6px"},
            ),
        ], className="info-card"),

        # Trials
        html.Div([
            html.Div("Completed Trials", className="info-card-label"),
            html.Div(str(n_complete), className="info-card-value accent"),
        ], className="info-card"),

        # Best value
        html.Div([
            html.Div("Best Value", className="info-card-label"),
            html.Div(best_str, className="info-card-value success",
                     style={"fontSize": "16px"}),
        ], className="info-card"),

        # Best trial
        html.Div([
            html.Div("Best Trial", className="info-card-label"),
            html.Div(best_tr, className="info-card-value accent"),
        ], className="info-card"),

        # Direction
        html.Div([
            html.Div("Direction", className="info-card-label"),
            html.Div(
                ("▼ " if direction == "minimize" else "▲ ") + direction.capitalize(),
                className="info-card-value",
                style={"fontSize": "14px",
                       "color": WARNING if direction == "minimize" else ACCENT},
            ),
        ], className="info-card"),

        # Study name
        html.Div([
            html.Div("Study", className="info-card-label"),
            html.Div(
                study_name,
                style={"fontFamily": "'DM Mono', monospace", "fontSize": "12px",
                       "color": TEXT_DIM, "marginTop": "4px",
                       "overflow": "hidden", "textOverflow": "ellipsis",
                       "whiteSpace": "nowrap", "maxWidth": "180px"},
            ),
        ], className="info-card", style={"minWidth": "180px"}),
    ]

    return html.Div(cards, className="info-bar")


def empty_info() -> html.Div:
    return html.Div(
        html.Span("← Drop a .db file and select a study",
                  style={"fontFamily": "'DM Mono', monospace",
                         "fontSize": "12px", "color": TEXT_FAINT}),
        className="info-bar",
        style={"height": "60px", "alignItems": "center"},
    )