# ─────────────────────────────────────────────
#  Optuna Dashboard Pro — components/trials_table.py
# ─────────────────────────────────────────────
from __future__ import annotations

import sys, os
import pandas as pd
from dash import dash_table, html

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from assets.theme import SURFACE, SURFACE_2, SURFACE_3, BORDER, TEXT, TEXT_DIM, ACCENT, SUCCESS, DANGER


def build_trials_table(df: pd.DataFrame, direction: str = "minimize"):
    if df is None or df.empty:
        return empty_table()

    # ── Format display df ──────────────────────────────────────────────────────
    display_cols_ordered = ["trial_number"]
    metric_col = "value"

    if metric_col in df.columns:
        display_cols_ordered.append(metric_col)

    # param columns
    param_cols = [c for c in df.columns if c.startswith("param_")]
    display_cols_ordered += param_cols

    if "state" in df.columns:
        display_cols_ordered.append("state")
    if "datetime_complete" in df.columns:
        display_cols_ordered.append("datetime_complete")

    display_df = df[[c for c in display_cols_ordered if c in df.columns]].copy()

    # Rename for display
    rename = {"trial_number": "Trial #", "value": "Objective", "state": "State",
               "datetime_complete": "Completed At"}
    rename.update({c: c.replace("param_", "") for c in param_cols})
    display_df = display_df.rename(columns=rename)

    # Format datetime
    if "Completed At" in display_df.columns:
        display_df["Completed At"] = pd.to_datetime(display_df["Completed At"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Round floats
    for col in display_df.select_dtypes(include="float").columns:
        display_df[col] = display_df[col].round(6)

    # Find best row for highlight
    best_idx = None
    if "Objective" in display_df.columns:
        if direction == "minimize":
            best_idx = display_df["Objective"].idxmin()
        else:
            best_idx = display_df["Objective"].idxmax()

    columns = [{"name": c, "id": c, "type": "numeric" if display_df[c].dtype in ["float64", "int64"] else "text"}
               for c in display_df.columns]

    # Style conditions
    style_data_cond = [
        # Zebra rows
        {"if": {"row_index": "odd"},
         "backgroundColor": SURFACE_2},
        # Hover
        {"if": {"state": "active"},
         "backgroundColor": SURFACE_3,
         "border": f"1px solid {ACCENT}"},
    ]

    if best_idx is not None:
        # Highlight best row
        style_data_cond.append({
            "if": {"row_index": int(display_df.index.get_loc(best_idx))},
            "backgroundColor": "rgba(0,229,160,0.08)",
            "borderLeft": f"3px solid {SUCCESS}",
        })
        # Highlight best objective cell
        style_data_cond.append({
            "if": {"row_index": int(display_df.index.get_loc(best_idx)), "column_id": "Objective"},
            "color": SUCCESS,
            "fontWeight": "600",
        })

    table = dash_table.DataTable(
        data=display_df.to_dict("records"),
        columns=columns,
        page_size=20,
        sort_action="native",
        filter_action="native",
        sort_mode="multi",
        style_table={"overflowX": "auto", "overflowY": "auto", "maxHeight": "520px"},
        style_cell={
            "backgroundColor": SURFACE,
            "color": TEXT,
            "borderColor": BORDER,
            "fontFamily": "'DM Mono', monospace",
            "fontSize": "12px",
            "padding": "8px 14px",
            "textAlign": "left",
            "minWidth": "80px",
            "maxWidth": "180px",
            "overflow": "hidden",
            "textOverflow": "ellipsis",
        },
        style_header={
            "backgroundColor": SURFACE_2,
            "color": TEXT_DIM,
            "fontFamily": "'DM Mono', monospace",
            "fontSize": "10px",
            "fontWeight": "500",
            "textTransform": "uppercase",
            "letterSpacing": "0.08em",
            "borderBottom": f"1px solid {BORDER}",
            "padding": "10px 14px",
        },
        style_data_conditional=style_data_cond,
        filter_options={"placeholder_text": "Filter…"},
        style_filter={
            "backgroundColor": SURFACE_2,
            "color": TEXT,
            "borderColor": BORDER,
            "fontFamily": "'DM Mono', monospace",
            "fontSize": "11px",
        },
    )

    return html.Div([
        html.Div(
            f"Showing {len(display_df)} completed trials",
            style={"fontFamily": "'DM Mono', monospace", "fontSize": "11px",
                   "color": TEXT_DIM, "marginBottom": "10px"},
        ),
        table,
    ])


def empty_table():
    return html.Div(
        html.Div([
            html.Div("⊘", style={"fontSize": "32px", "opacity": ".25", "marginBottom": "8px"}),
            html.Div("No trials loaded", style={"fontFamily": "'DM Mono', monospace",
                                                "fontSize": "13px", "color": TEXT_DIM}),
        ], style={"textAlign": "center", "padding": "60px 0"}),
    )