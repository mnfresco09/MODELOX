"""
Dashboard web para trading real MODELOX.

Expone:
- Vista web (Flask)
- Estado en vivo (/api/state)
- Opciones disponibles (/api/options)
- Cola de comandos manuales (/api/command)
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template, request


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(_BASE_DIR, "templates"),
)

_state: Dict[str, Any] = {"status": "starting"}
_state_lock = threading.Lock()

_options: Dict[str, Any] = {
    "assets": [],
    "strategies": [],
    "timeframes": ["1m", "5m", "15m", "30m", "1h", "4h"],
}
_options_lock = threading.Lock()

_commands: List[Dict[str, Any]] = []
_commands_lock = threading.Lock()

_chart_payload: Dict[str, Any] = {"candles": [], "indicators": {}, "signals": []}
_chart_lock = threading.Lock()


def set_options(options: Dict[str, Any]) -> None:
    with _options_lock:
        _options.update(options or {})


def update_state(new_state: Dict[str, Any]) -> None:
    with _state_lock:
        _state.clear()
        _state.update(new_state or {})


def pop_commands() -> List[Dict[str, Any]]:
    with _commands_lock:
        pending = list(_commands)
        _commands.clear()
        return pending


def set_chart_payload(payload: Dict[str, Any]) -> None:
    with _chart_lock:
        _chart_payload.clear()
        _chart_payload.update(payload or {})


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    with _state_lock:
        return jsonify(_state)


@app.route("/api/options")
def api_options():
    with _options_lock:
        return jsonify(_options)


@app.route("/api/chart")
def api_chart():
    with _chart_lock:
        return jsonify(_chart_payload)


@app.route("/api/command", methods=["POST"])
def api_command():
    data = request.get_json(silent=True) or {}
    action = (data.get("action") or "").strip().lower()
    if not action:
        return jsonify({"ok": False, "error": "action requerido"}), 400

    cmd: Dict[str, Any] = {"action": action}
    if "direction" in data:
        cmd["direction"] = str(data["direction"]).upper()
    if "asset" in data:
        cmd["asset"] = str(data["asset"]).upper()
    if "strategy_id" in data:
        try:
            cmd["strategy_id"] = int(data["strategy_id"])
        except Exception:
            return jsonify({"ok": False, "error": "strategy_id inválido"}), 400
    if "timeframe" in data:
        cmd["timeframe"] = str(data["timeframe"]).lower()

    with _commands_lock:
        _commands.append(cmd)

    return jsonify({"ok": True, "queued": cmd})


def start_server(port: int = 8088) -> int:
    def _run() -> None:
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

    thread = threading.Thread(target=_run, daemon=True, name="modelox-web-dashboard")
    thread.start()
    return port
