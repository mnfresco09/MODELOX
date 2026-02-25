# ─────────────────────────────────────────────
#  Optuna Dashboard Pro — app.py
# ─────────────────────────────────────────────
from __future__ import annotations

import base64, os, sys, tempfile, traceback
from io import StringIO

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, no_update
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from config  import HOST, PORT, DEBUG
from layout  import create_layout
from data    import loader
from data    import cache as _cache
from components import (contour, rank as rank_comp, robustness,
                         importance, history, parallel,
                         trials_table, study_info)


app = dash.Dash(__name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Optuna Dashboard Pro",
    update_title=None)
app.layout = create_layout()
server = app.server


def _read_df(j: str) -> pd.DataFrame:
    df = pd.read_json(StringIO(j), orient="split")
    if "trial_number" not in df.columns:
        df = df.reset_index().rename(columns={"index": "trial_number"})
    return df


# ── 1. Upload ──────────────────────────────────────────────────────────────────
@app.callback(
    Output("db-path-store", "data"),
    Output("upload-status", "children"),
    Input("upload-db",      "contents"),
    State("upload-db",      "filename"),
    prevent_initial_call=True,
)
def handle_upload(contents, filename):
    if not contents:
        return no_update, no_update
    _, data = contents.split(",", 1)
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.write(base64.b64decode(data)); tmp.close()
    return tmp.name, f"✓  {filename}"


# ── 2. Study dropdown ──────────────────────────────────────────────────────────
@app.callback(
    Output("study-select", "options"),
    Output("study-select", "value"),
    Input("db-path-store", "data"),
)
def update_study_options(db_path):
    if not db_path: return [], None
    try:
        names = loader.get_study_names(db_path)
        return [{"label": n, "value": n} for n in names], (names[0] if names else None)
    except Exception as e:
        print(f"[study options] {e}"); return [], None


# ── 3. Load study ──────────────────────────────────────────────────────────────
@app.callback(
    Output("study-data-store",   "data"),
    Output("study-meta-store",   "data"),
    Output("interval-refresh",   "disabled"),
    Input("study-select",        "value"),
    Input("interval-refresh",    "n_intervals"),
    State("db-path-store",       "data"),
)
def load_study_data(study_name, _n, db_path):
    if not db_path or not study_name:
        return no_update, no_update, True
    try:
        count = loader.get_trial_count(db_path, study_name)
        if not _cache.needs_refresh(db_path, study_name, count):
            entry = _cache.get_cached(db_path, study_name)
            return entry.payload["df_json"], entry.payload["meta"], \
                   not entry.payload["meta"]["is_running"]

        study   = loader.load_study(db_path, study_name)
        df      = loader.get_trials_dataframe(study)
        stats   = loader.get_study_summary_stats(study)
        meta = {
            "study_name":        study_name,
            "param_names":       loader.get_param_names(df),
            "metric_names":      loader.get_metric_names(study, df),
            "direction":         loader.get_study_direction(study),
            "is_running":        loader.is_study_running(study),
            "trial_count":       stats["completed"],
            "best_value":        stats["best_value"],
            "best_trial_number": stats["best_trial"],
            "db_path":           db_path,
        }
        df_json = df.to_json(date_format="iso", orient="split")
        _cache.update(db_path, study_name, count, {"df_json": df_json, "meta": meta})
        return df_json, meta, not meta["is_running"]
    except Exception as e:
        print(f"[load study] {e}"); traceback.print_exc()
        return no_update, no_update, True


# ── 4. Refresh badge ───────────────────────────────────────────────────────────
@app.callback(
    Output("refresh-badge",   "children"),
    Output("refresh-badge",   "className"),
    Input("interval-refresh", "disabled"),
    Input("study-meta-store", "data"),
)
def refresh_badge(disabled, meta):
    if meta and meta.get("is_running"):  return "● LIVE",     "refresh-badge active"
    if meta:                             return "● COMPLETE",  "refresh-badge"
    return "", "refresh-badge"


# ── 5. Populate ALL axis/metric dropdowns ─────────────────────────────────────
# Gravity Cloud + Rank + Robustness share the same param/metric lists
@app.callback(
    # Gravity Cloud
    Output("x-param-select",       "options"), Output("x-param-select",       "value"),
    Output("y-param-select",       "options"), Output("y-param-select",       "value"),
    Output("metric-select",        "options"), Output("metric-select",        "value"),
    # Rank
    Output("x-param-select-rank",  "options"), Output("x-param-select-rank",  "value"),
    Output("y-param-select-rank",  "options"), Output("y-param-select-rank",  "value"),
    Output("metric-select-rank",   "options"), Output("metric-select-rank",   "value"),
    # Robustness
    Output("x-param-select-rob",   "options"), Output("x-param-select-rob",   "value"),
    Output("y-param-select-rob",   "options"), Output("y-param-select-rob",   "value"),
    Output("metric-select-rob",    "options"), Output("metric-select-rob",    "value"),
    # Parallel
    Output("parallel-param-select","options"), Output("parallel-param-select","value"),
    Input("study-meta-store", "data"),
)
def update_all_dropdowns(meta):
    blank = [[], None]
    empty = blank * 9 + [[], []]
    if not meta: return empty

    params  = meta.get("param_names",  [])
    metrics = meta.get("metric_names", [])
    p_opts  = [{"label": p, "value": p} for p in params]
    m_opts  = [{"label": m, "value": m} for m in metrics]
    x0 = params[0] if params else None
    y0 = params[1] if len(params) > 1 else x0
    m0 = metrics[0] if metrics else None

    return (
        p_opts, x0, p_opts, y0, m_opts, m0,   # contour
        p_opts, x0, p_opts, y0, m_opts, m0,   # rank
        p_opts, x0, p_opts, y0, m_opts, m0,   # robustness
        p_opts, params[:8],                    # parallel
    )


# ── 6. Gravity Cloud ───────────────────────────────────────────────────────────
@app.callback(
    Output("contour-graph",   "figure"),
    Input("study-data-store", "data"),
    Input("x-param-select",   "value"),
    Input("y-param-select",   "value"),
    Input("metric-select",    "value"),
    Input("agg-mode",         "value"),
    State("study-meta-store", "data"),
)
def update_contour(df_json, x_p, y_p, metric, agg_mode, meta):
    if not df_json or not x_p or not y_p or not metric:
        return contour.empty_figure("Select study and parameters to display")
    try:
        df  = _read_df(df_json)
        dir = meta.get("direction", "minimize") if meta else "minimize"
        return contour.build_contour_figure(df, x_p, y_p, metric, dir,
                                            aggregate=(agg_mode == "mean"))
    except Exception as e:
        print(f"[contour] {e}"); traceback.print_exc()
        return contour.empty_figure("Render error — see terminal")


# ── 7. Rank Scatter ────────────────────────────────────────────────────────────
@app.callback(
    Output("rank-graph",           "figure"),
    Input("study-data-store",      "data"),
    Input("x-param-select-rank",   "value"),
    Input("y-param-select-rank",   "value"),
    Input("metric-select-rank",    "value"),
    Input("agg-mode-rank",         "value"),
    State("study-meta-store",      "data"),
)
def update_rank(df_json, x_p, y_p, metric, agg_mode, meta):
    if not df_json or not x_p or not y_p or not metric:
        return rank_comp.empty_figure("Select study and parameters to display")
    try:
        df  = _read_df(df_json)
        dir = meta.get("direction", "minimize") if meta else "minimize"
        return rank_comp.build_rank_figure(df, x_p, y_p, metric, dir,
                                           aggregate=(agg_mode == "mean"))
    except Exception as e:
        print(f"[rank] {e}"); traceback.print_exc()
        return rank_comp.empty_figure("Render error — see terminal")


# ── 8. Robustness — Plateau ────────────────────────────────────────────────────
@app.callback(
    Output("plateau-graph",        "figure"),
    Input("study-data-store",      "data"),
    Input("x-param-select-rob",    "value"),
    Input("y-param-select-rob",    "value"),
    Input("metric-select-rob",     "value"),
    Input("k-neighbors-slider",    "value"),
    State("study-meta-store",      "data"),
)
def update_plateau(df_json, x_p, y_p, metric, k, meta):
    if not df_json or not x_p or not y_p or not metric:
        return robustness.empty_figure("Select parameters above")
    try:
        df  = _read_df(df_json)
        dir = meta.get("direction", "minimize") if meta else "minimize"
        return robustness.build_plateau_figure(df, x_p, y_p, metric, dir, k=k or 8)
    except Exception as e:
        print(f"[plateau] {e}"); traceback.print_exc()
        return robustness.empty_figure("Render error — see terminal")


# ── 9. Robustness — Volatility ─────────────────────────────────────────────────
@app.callback(
    Output("std-graph",         "figure"),
    Input("study-data-store",   "data"),
    Input("x-param-select-rob", "value"),
    Input("y-param-select-rob", "value"),
    Input("metric-select-rob",  "value"),
    State("study-meta-store",   "data"),
)
def update_std(df_json, x_p, y_p, metric, meta):
    if not df_json or not x_p or not y_p or not metric:
        return robustness.empty_figure("Select parameters above")
    try:
        df = _read_df(df_json)
        return robustness.build_std_map(df, x_p, y_p, metric)
    except Exception as e:
        print(f"[std map] {e}"); traceback.print_exc()
        return robustness.empty_figure("Render error — see terminal")


# ── 10. Robustness — Neighborhood ─────────────────────────────────────────────
@app.callback(
    Output("neighborhood-graph",   "figure"),
    Input("study-data-store",      "data"),
    Input("x-param-select-rob",    "value"),
    Input("y-param-select-rob",    "value"),
    Input("metric-select-rob",     "value"),
    Input("k-neighbors-slider",    "value"),
    State("study-meta-store",      "data"),
)
def update_neighborhood(df_json, x_p, y_p, metric, k, meta):
    if not df_json or not x_p or not y_p or not metric:
        return robustness.empty_figure("Select parameters above")
    try:
        df  = _read_df(df_json)
        dir = meta.get("direction", "minimize") if meta else "minimize"
        return robustness.build_neighborhood_figure(df, x_p, y_p, metric, dir, k=k or 8)
    except Exception as e:
        print(f"[neighborhood] {e}"); traceback.print_exc()
        return robustness.empty_figure("Render error — see terminal")


# ── 11. Importance ─────────────────────────────────────────────────────────────
@app.callback(
    Output("importance-graph", "figure"),
    Input("study-meta-store",  "data"),
)
def update_importance(meta):
    if not meta: return importance.empty_figure("Load a study")
    try:
        study = loader.load_study(meta["db_path"], meta["study_name"])
        return importance.build_importance_figure(loader.compute_param_importance(study))
    except Exception as e:
        print(f"[importance] {e}")
        return importance.empty_figure(f"Error: {e}")


# ── 12. History ────────────────────────────────────────────────────────────────
@app.callback(
    Output("history-graph",   "figure"),
    Input("study-data-store", "data"),
    State("study-meta-store", "data"),
)
def update_history(df_json, meta):
    if not df_json: return history.empty_figure("Load a study")
    try:
        df  = _read_df(df_json)
        dir = meta.get("direction", "minimize") if meta else "minimize"
        return history.build_history_figure(df, dir)
    except Exception as e:
        print(f"[history] {e}"); traceback.print_exc()
        return history.empty_figure("Render error — see terminal")


# ── 13. Parallel ───────────────────────────────────────────────────────────────
@app.callback(
    Output("parallel-graph",       "figure"),
    Input("study-data-store",      "data"),
    Input("parallel-param-select", "value"),
    State("study-meta-store",      "data"),
)
def update_parallel(df_json, sel, meta):
    if not df_json or not sel:
        return parallel.empty_figure("Select parameters above")
    try:
        df  = _read_df(df_json)
        dir = meta.get("direction", "minimize") if meta else "minimize"
        m   = meta.get("metric_names", ["value"])[0] if meta else "value"
        return parallel.build_parallel_figure(df, sel, m, dir)
    except Exception as e:
        print(f"[parallel] {e}")
        return parallel.empty_figure("Render error — see terminal")


# ── 14. Trials table ───────────────────────────────────────────────────────────
@app.callback(
    Output("trials-table-container", "children"),
    Input("study-data-store",        "data"),
    State("study-meta-store",        "data"),
)
def update_table(df_json, meta):
    if not df_json: return trials_table.empty_table()
    try:
        df  = _read_df(df_json)
        dir = meta.get("direction", "minimize") if meta else "minimize"
        return trials_table.build_trials_table(df, dir)
    except Exception as e:
        print(f"[table] {e}"); traceback.print_exc()
        return trials_table.empty_table()


# ── 15. Study info ─────────────────────────────────────────────────────────────
@app.callback(
    Output("study-info-container", "children"),
    Input("study-meta-store",      "data"),
)
def update_info(meta):
    return study_info.build_study_info(meta) if meta else study_info.empty_info()


if __name__ == "__main__":
    app.run(debug=DEBUG, host=HOST, port=PORT)