"""
MODELOX · app.py
Dash application factory and run entry point.
"""
from __future__ import annotations

import os
import threading
import webbrowser
import atexit

import dash
import dash_bootstrap_components as dbc

from visual.optuna_dash.layout import build_layout, GLOBAL_CSS
from visual.optuna_dash.callbacks import register_callbacks
from visual.optuna_dash.perf import set_perf_debug, print_perf_summary


def create_app(db_path: str, perf_debug: bool = False) -> dash.Dash:
    """
    Create and configure the MODELOX Dash application.

    Parameters
    ----------
    db_path : str
        Absolute path to the Optuna SQLite database file.

    Returns
    -------
    dash.Dash
        Fully configured application instance (layout + callbacks registered).
    """
    set_perf_debug(perf_debug)

    db_name = os.path.basename(db_path).replace(".db", "")

    app = dash.Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.CYBORG,       # Dark Bootstrap base
            dbc.icons.BOOTSTRAP,
        ],
        suppress_callback_exceptions=True,
        title=f"MODELOX · {db_name}",
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    )

    # Inject our custom CSS
    app.index_string = (
        "<!DOCTYPE html><html><head>"
        "{%metas%}{%favicon%}{%css%}"
        "<style>" + GLOBAL_CSS + "</style>"
        "</head><body>"
        "{%app_entry%}"
        "<footer>{%config%}{%scripts%}{%renderer%}</footer>"
        "</body></html>"
    )

    app.layout = build_layout(db_path)
    register_callbacks(app, db_path)

    return app


def run(db_path: str, port: int = 8050, debug: bool = False, perf_debug: bool = False) -> None:
    """
    Launch the MODELOX dashboard and open it in the default browser.

    Parameters
    ----------
    db_path : str
        Absolute path to the Optuna SQLite database file.
    port : int
        HTTP port to listen on (default: 8050).
    debug : bool
        Enable Dash debug mode (default: False).
    perf_debug : bool
        Enable performance timing debug logs and bottleneck summary (default: False).
    """
    db_name = os.path.basename(db_path).replace(".db", "")
    app = create_app(db_path, perf_debug=perf_debug)

    if perf_debug:
        atexit.register(print_perf_summary)

    w = 56
    print(f"\n  ╔{'═'*w}╗")
    print(f"  ║{'MODELOX  AI Parameter Intelligence':^{w}}║")
    print(f"  ╚{'═'*w}╝")
    print(f"  DB     →  {db_name}")
    print(f"  URL    →  http://127.0.0.1:{port}")
    print(f"  Stop   →  Ctrl+C\n")
    if perf_debug:
        print("  PERF   →  ON (timings + bottleneck summary)\n")

    def _open_browser():
        import time
        time.sleep(1.4)
        webbrowser.open(f"http://127.0.0.1:{port}")

    threading.Thread(target=_open_browser, daemon=True).start()
    app.run(debug=debug, port=port, host="127.0.0.1")
