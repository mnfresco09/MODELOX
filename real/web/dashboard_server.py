from __future__ import annotations

import json
import threading
import time
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import polars as pl
except Exception:  # pragma: no cover
    pl = None

from modelox.strategies.registry import instantiate_strategies, list_available_strategies
from real.trader import MockTrial
from real.config import DISPLAY_NAMES
from visual.grafico import StrictAlignmentMapper, _detect_indicators, _normalize_timestamps_to_unix


STATIC_DIR = Path(__file__).parent / "static"
ALLOWED_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]


def _json(data: Dict[str, Any]) -> bytes:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _safe_float(v: Any, d: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return d


@dataclass
class MarketSnapshot:
    price: float = 0.0
    df_raw: Any = None
    df_signals: Any = None
    updated_at: float = 0.0


class DashboardRuntime:
    def __init__(self, traders: List[Any], host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        self.lock = threading.RLock()
        self.traders: Dict[str, Any] = {
            f"{t.symbol}|{int(getattr(t.config, 'strategy_id', 0))}": t for t in traders
        }
        self.traders_by_symbol: Dict[str, List[Any]] = {}
        for t in traders:
            self.traders_by_symbol.setdefault(t.symbol, []).append(t)
        self.symbol_to_alias: Dict[str, str] = {
            s: DISPLAY_NAMES.get(s, s.replace("-USDT", "")) for s in self.traders_by_symbol.keys()
        }
        self.alias_to_symbol: Dict[str, str] = {v: k for k, v in self.symbol_to_alias.items()}
        preferred = ["BTC", "ETH", "GOLD", "SILVER", "SP500", "NASDAQ"]
        self.assets = [a for a in preferred if a in self.alias_to_symbol] + [
            a for a in sorted(self.alias_to_symbol.keys()) if a not in preferred
        ]
        self.selected_symbol = self.assets[0] if self.assets else ""
        tf_default = traders[0].config.timeframe if traders else "1m"
        self.selected_timeframe = tf_default if tf_default in ALLOWED_TIMEFRAMES else "1m"
        self.price_cache: Dict[str, float] = {s: 0.0 for s in self.traders_by_symbol.keys()}
        self.signal_cache: Dict[tuple[str, int], MarketSnapshot] = {}
        self.tf_cache: Dict[tuple[str, int, str], MarketSnapshot] = {}
        self.last_error: Optional[str] = None
        self._balance_cache: Dict[str, Any] = {"at": 0.0, "value": {}}
        self.strategies = [
            {"id": int(sid), "name": name}
            for sid, name in list_available_strategies()
        ]
        self.selected_strategy_id = int(traders[0].config.strategy_id) if traders else 0
        self.context_version = 1

    def _get_symbol(self, symbol_or_alias: Optional[str]) -> str:
        alias = symbol_or_alias or self.selected_symbol
        return self.alias_to_symbol.get(alias, alias)

    def _get_trader_for_view(self, symbol_or_alias: Optional[str] = None, strategy_id: Optional[int] = None):
        symbol = self._get_symbol(symbol_or_alias)
        sid = int(strategy_id if strategy_id is not None else self.selected_strategy_id)
        for t in self.traders_by_symbol.get(symbol, []):
            if int(getattr(t.config, "strategy_id", 0)) == sid:
                return t
        lst = self.traders_by_symbol.get(symbol, [])
        return lst[0] if lst else None

    def _get_any_trader(self, symbol_or_alias: Optional[str] = None):
        symbol = self._get_symbol(symbol_or_alias)
        lst = self.traders_by_symbol.get(symbol, [])
        return lst[0] if lst else None

    def update_market_data(
        self,
        symbol: str,
        price: float,
        df_raw: Any = None,
        df_signals: Any = None,
        strategy_id: Optional[int] = None,
    ) -> None:
        with self.lock:
            symbol = self._get_symbol(symbol)
            self.price_cache[symbol] = _safe_float(price)
            if df_raw is None and df_signals is None:
                return
            sid = int(strategy_id if strategy_id is not None else self.selected_strategy_id)
            key = (symbol, sid)
            snap = self.signal_cache.setdefault(key, MarketSnapshot())
            snap.price = _safe_float(price)
            if df_raw is not None:
                snap.df_raw = df_raw
            if df_signals is not None:
                snap.df_signals = df_signals
            snap.updated_at = time.time()

    def set_selectors(self, symbol: str, timeframe: str, strategy_id: int) -> Dict[str, Any]:
        with self.lock:
            if symbol not in self.assets:
                return {"ok": False, "error": f"Activo inválido: {symbol}"}
            if timeframe not in ALLOWED_TIMEFRAMES:
                return {"ok": False, "error": f"Timeframe inválido: {timeframe}"}
            if not any(int(s["id"]) == int(strategy_id) for s in self.strategies):
                return {"ok": False, "error": f"Estrategia inválida: {strategy_id}"}

            changed = (
                self.selected_symbol != symbol
                or self.selected_timeframe != timeframe
                or self.selected_strategy_id != int(strategy_id)
            )

            self.selected_symbol = symbol
            self.selected_timeframe = timeframe
            self.selected_strategy_id = int(strategy_id)

            if changed:
                self.context_version += 1
                # Evitar reutilizar datos viejos tras cambios de selector
                self.last_error = None

            return {
                "ok": True,
                "selected": {
                    "asset": self.selected_symbol,
                    "timeframe": self.selected_timeframe,
                    "strategy_id": self.selected_strategy_id,
                },
                "context_version": self.context_version,
            }

    def _generate_signals_for_view(self, symbol: str, strategy_id: int, timeframe: str) -> Any:
        key = (symbol, int(strategy_id), timeframe)
        now = time.time()
        cached = self.tf_cache.get(key)
        if cached and (now - cached.updated_at) < 8.0 and cached.df_signals is not None:
            return cached.df_signals

        # Si tenemos cache directo de feed real (mismo símbolo/estrategia) y mismo TF base
        cached_live = self.signal_cache.get((symbol, int(strategy_id)))
        live_trader = self._get_trader_for_view(symbol, strategy_id)
        if (
            cached_live
            and cached_live.df_signals is not None
            and live_trader is not None
            and timeframe == str(getattr(live_trader.config, "timeframe", ""))
        ):
            return cached_live.df_signals

        base_trader = self._get_any_trader(symbol)
        if base_trader is None:
            return None

        try:
            df_raw = base_trader.fetch_market_data(limit=240, timeframe=timeframe)

            # Si existe trader activo para esa estrategia, usar su engine/params
            if live_trader is not None:
                df_signals = live_trader.generate_signals(df_raw)
            else:
                found = instantiate_strategies(only_id=int(strategy_id))
                if not found:
                    return None
                strategy = found[0]
                try:
                    params = strategy.suggest_params(MockTrial())
                except Exception:
                    params = {}
                df_signals = strategy.generate_signals(df_raw, params)

            snap = MarketSnapshot(
                price=self.price_cache.get(symbol, 0.0),
                df_raw=df_raw,
                df_signals=df_signals,
                updated_at=now,
            )
            self.tf_cache[key] = snap
            return df_signals
        except Exception as e:
            self.last_error = str(e)
            return None

    def _chart_payload(self, df_signals: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        if df_signals is None:
            return {"candles": [], "indicators": {"overlays": [], "sub_panels": []}}

        if pl is not None and hasattr(df_signals, "to_pandas"):
            df_pd = df_signals.to_pandas()
        else:
            df_pd = df_signals

        if df_pd is None or len(df_pd) == 0:
            return {"candles": [], "indicators": {"overlays": [], "sub_panels": []}}

        if "timestamp" not in df_pd.columns:
            return {"candles": [], "indicators": {"overlays": [], "sub_panels": []}}

        timestamps = _normalize_timestamps_to_unix(df_pd["timestamp"].values)
        opens = np.nan_to_num(df_pd["open"].values.astype(np.float64))
        highs = np.nan_to_num(df_pd["high"].values.astype(np.float64))
        lows = np.nan_to_num(df_pd["low"].values.astype(np.float64))
        closes = np.nan_to_num(df_pd["close"].values.astype(np.float64))

        uniq_ts, uniq_idx = np.unique(timestamps, return_index=True)
        timestamps = uniq_ts
        opens = opens[uniq_idx]
        highs = highs[uniq_idx]
        lows = lows[uniq_idx]
        closes = closes[uniq_idx]

        candles = [
            {
                "time": int(timestamps[i]),
                "open": float(opens[i]),
                "high": float(highs[i]),
                "low": float(lows[i]),
                "close": float(closes[i]),
            }
            for i in range(len(timestamps))
        ]

        if len(timestamps) == 0:
            return {"candles": candles, "indicators": {"overlays": [], "sub_panels": []}}

        price_range = (float(np.min(lows)), float(np.max(highs)))
        detected = _detect_indicators(df_pd, price_range, params)
        mapper = StrictAlignmentMapper(timestamps)

        overlays: List[Dict[str, Any]] = []
        for ov in detected.get("overlays", []):
            col = ov.get("col")
            if col not in df_pd.columns:
                continue
            vals = df_pd[col].values.astype(np.float64)
            qvals, factor, _ = mapper.align_quantized(timestamps, vals, precision=int(ov.get("precision", 2)))
            data = [
                {"time": int(timestamps[i]), "value": (None if qvals[i] is None else float(qvals[i]) / factor)}
                for i in range(len(timestamps))
            ]
            overlays.append({
                "name": col,
                "color": ov.get("color", "#4F94CD"),
                "type": ov.get("type", "line"),
                "data": [x for x in data if x["value"] is not None],
            })

        grouped: Dict[str, Dict[str, Any]] = {}
        for panel in detected.get("sub_panels", []):
            col = panel.get("col")
            if col not in df_pd.columns:
                continue
            vals = df_pd[col].values.astype(np.float64)
            qvals, factor, _ = mapper.align_quantized(timestamps, vals, precision=int(panel.get("precision", 4)))
            data = [
                {"time": int(timestamps[i]), "value": (None if qvals[i] is None else float(qvals[i]) / factor)}
                for i in range(len(timestamps))
            ]
            panel_id = str(panel.get("panel_id") or f"auto_{col}")
            if panel_id not in grouped:
                grouped[panel_id] = {"title": panel.get("name", col), "series": []}
            grouped[panel_id]["series"].append({
                "name": panel.get("name", col),
                "color": panel.get("color", "#8A949B"),
                "type": panel.get("type", "line"),
                "bounds": panel.get("bounds"),
                "data": [x for x in data if x["value"] is not None],
            })

        return {
            "candles": candles,
            "indicators": {
                "overlays": overlays,
                "sub_panels": list(grouped.values()),
            },
        }

    def _positions_payload(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for trader in self.traders.values():
            symbol = trader.symbol
            fallback_price = self.price_cache.get(symbol, 0.0)
            strategy_id = int(getattr(trader.config, "strategy_id", 0))
            strategy_name = next((s["name"] for s in self.strategies if int(s["id"]) == strategy_id), f"ID{strategy_id}")
            for t in list(trader.active_trades):
                # Misma referencia que terminal: PnL NO REALIZADO del exchange
                pnl = t.unrealized_pnl if t.unrealized_pnl is not None else 0.0
                # Misma referencia que terminal: mark price si está disponible
                current_price = t.mark_price if getattr(t, "mark_price", None) else fallback_price
                rows.append({
                    "symbol": self.symbol_to_alias.get(symbol, symbol),
                    "side": t.side,
                    "strategy": strategy_name,
                    "entry_price": _safe_float(t.entry_price),
                    "current_price": _safe_float(current_price),
                    "pnl": _safe_float(pnl),
                    "close_key": f"{symbol}|{t.side}",
                })
        return rows

    def _balance_payload(self) -> Dict[str, Any]:
        now = time.time()
        if now - _safe_float(self._balance_cache.get("at"), 0) < 4.0:
            return self._balance_cache.get("value", {})

        trader = self._get_any_trader(self.selected_symbol)
        if not trader:
            return {"available": 0.0, "equity": 0.0, "currency": "USDT"}
        try:
            info = trader.get_balance_info()
            payload = {
                "available": _safe_float(info.get("balance_usdt")),
                "equity": _safe_float(info.get("equity")),
                "currency": trader.config.quote_currency,
            }
        except Exception:
            payload = {
                "available": 0.0,
                "equity": 0.0,
                "currency": trader.config.quote_currency,
            }

        self._balance_cache = {"at": now, "value": payload}
        return payload

    def build_snapshot(self) -> Dict[str, Any]:
        with self.lock:
            symbol = self._get_symbol(self.selected_symbol)
            trader = self._get_trader_for_view(symbol, self.selected_strategy_id)
            if trader is None:
                return {"ok": False, "error": "No trader activo"}

            df_signals = self._generate_signals_for_view(
                symbol=symbol,
                strategy_id=int(self.selected_strategy_id),
                timeframe=self.selected_timeframe,
            )

            chart = self._chart_payload(df_signals, trader.default_params or {})
            last_price = _safe_float(self.price_cache.get(symbol, 0.0))
            if last_price <= 0:
                try:
                    last_price = _safe_float(trader.client.get_current_price(trader.symbol, trader.config.market_type))
                except Exception:
                    pass

            return {
                "ok": True,
                "timestamp": int(time.time()),
                "selected": {
                    "asset": self.selected_symbol,
                    "timeframe": self.selected_timeframe,
                    "strategy_id": int(self.selected_strategy_id),
                },
                "context_version": int(self.context_version),
                "selected_leverage": int(getattr(trader.config, "leverage", 1) or 1),
                "assets": self.assets,
                "timeframes": ALLOWED_TIMEFRAMES,
                "strategies": self.strategies,
                "balance": self._balance_payload(),
                "last_price": last_price,
                "chart": chart,
                "positions": self._positions_payload(),
                "error": self.last_error,
            }

    def open_manual_order(
        self,
        side: str,
        order_type: str,
        margin: float = 0.0,
        quantity: float = 0.0,
    ) -> Dict[str, Any]:
        with self.lock:
            trader = self._get_trader_for_view(self.selected_symbol, self.selected_strategy_id)
            if trader is None:
                trader = self._get_any_trader(self.selected_symbol)
            if trader is None:
                return {"ok": False, "error": "Trader no encontrado"}
            try:
                price = trader.client.get_current_price(trader.symbol, trader.config.market_type)
                leverage = max(1, int(getattr(trader.config, "leverage", 1) or 1))
                qty = _safe_float(quantity, 0.0)
                margin_val = _safe_float(margin, 0.0)

                # Modo recomendado: el usuario elige margen y el tamaño se calcula automático
                if margin_val > 0 and qty <= 0:
                    qty = (margin_val * leverage) / max(price, 1e-12)

                trade = trader.execute_manual_order(side=side, current_price=price, quantity=qty, order_type=order_type)
                if not trade:
                    return {"ok": False, "error": "No se pudo ejecutar la orden"}
                return {
                    "ok": True,
                    "filled": {
                        "margin": margin_val,
                        "quantity": _safe_float(getattr(trade, "quantity", qty)),
                        "price": _safe_float(getattr(trade, "entry_price", price)),
                        "leverage": leverage,
                    },
                }
            except Exception as e:
                return {"ok": False, "error": str(e)}

    def close_position(self, close_key: str) -> Dict[str, Any]:
        with self.lock:
            try:
                symbol, side = close_key.split("|", 1)
            except ValueError:
                return {"ok": False, "error": "close_key inválido"}

            trader = self.traders.get(symbol)
            if trader is None:
                trader = self._get_any_trader(symbol)
            if trader is None:
                return {"ok": False, "error": "Trader no encontrado"}
            try:
                result = trader.client.close_position(symbol, side)
                if not result.success:
                    return {"ok": False, "error": result.error_message or "Cierre rechazado"}
                return {"ok": True}
            except Exception as e:
                return {"ok": False, "error": str(e)}


class DashboardHTTPServer:
    def __init__(self, runtime: DashboardRuntime):
        self.runtime = runtime
        self.httpd: Optional[ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    def start(self) -> str:
        runtime = self.runtime

        class Handler(BaseHTTPRequestHandler):
            def _write_json(self, payload: Dict[str, Any], status: int = 200) -> None:
                raw = _json(payload)
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _read_json(self) -> Dict[str, Any]:
                try:
                    size = int(self.headers.get("Content-Length", "0") or "0")
                except Exception:
                    size = 0
                data = self.rfile.read(size) if size > 0 else b"{}"
                try:
                    return json.loads(data.decode("utf-8"))
                except Exception:
                    return {}

            def log_message(self, format: str, *args):
                return

            def do_GET(self):
                if self.path in {"/", "/index.html"}:
                    p = STATIC_DIR / "index.html"
                    raw = p.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return

                if self.path == "/static/style.css":
                    p = STATIC_DIR / "style.css"
                    raw = p.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/css; charset=utf-8")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return

                if self.path == "/static/app.js":
                    p = STATIC_DIR / "app.js"
                    raw = p.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/javascript; charset=utf-8")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return

                if self.path == "/api/bootstrap":
                    self._write_json(runtime.build_snapshot())
                    return

                if self.path == "/api/stream":
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()
                    try:
                        while True:
                            payload = runtime.build_snapshot()
                            self.wfile.write(b"event: snapshot\n")
                            self.wfile.write(b"data: " + _json(payload) + b"\n\n")
                            self.wfile.flush()
                            time.sleep(1.0)
                    except Exception:
                        return

                self._write_json({"ok": False, "error": "Not found"}, status=404)

            def do_POST(self):
                if self.path == "/api/selectors":
                    data = self._read_json()
                    result = runtime.set_selectors(
                        symbol=str(data.get("asset", "")),
                        timeframe=str(data.get("timeframe", "")),
                        strategy_id=int(data.get("strategy_id", 0) or 0),
                    )
                    self._write_json(result)
                    return

                if self.path == "/api/order/open":
                    data = self._read_json()
                    result = runtime.open_manual_order(
                        side=str(data.get("side", "LONG")).upper(),
                        order_type=str(data.get("order_type", "MARKET")).upper(),
                        margin=_safe_float(data.get("margin", 0.0), 0.0),
                        quantity=_safe_float(data.get("quantity", 0.0), 0.0),
                    )
                    self._write_json(result)
                    return

                if self.path == "/api/order/close":
                    data = self._read_json()
                    result = runtime.close_position(str(data.get("close_key", "")))
                    self._write_json(result)
                    return

                self._write_json({"ok": False, "error": "Not found"}, status=404)

        self.httpd = ThreadingHTTPServer((runtime.host, runtime.port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return f"http://{runtime.host}:{runtime.port}"

    def stop(self) -> None:
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None


class DashboardBridge:
    def __init__(self, runtime: DashboardRuntime, server: DashboardHTTPServer, url: str):
        self.runtime = runtime
        self.server = server
        self.url = url

    def update_market_data(
        self,
        symbol: str,
        price: float,
        df_raw: Any = None,
        df_signals: Any = None,
        strategy_id: Optional[int] = None,
    ) -> None:
        self.runtime.update_market_data(symbol, price, df_raw=df_raw, df_signals=df_signals, strategy_id=strategy_id)

    def stop(self) -> None:
        self.server.stop()


def launch_dashboard(traders: List[Any], host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> DashboardBridge:
    runtime = DashboardRuntime(traders=traders, host=host, port=port)
    server = DashboardHTTPServer(runtime)
    url = server.start()
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    return DashboardBridge(runtime=runtime, server=server, url=url)
