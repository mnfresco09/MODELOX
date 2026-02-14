"""
visual/telegram.py — Notificaciones Telegram minimalistas para MODELOX.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

import urllib.request
import urllib.parse
import json


# ─── CONFIG ──────────────────────────────────────────────────────────────────

import os as _os

BOT_TOKEN = _os.getenv(
    "MODELOX_TELEGRAM_TOKEN",
    "8571553580:AAEOsSs5c1o_AMju3dofdIcUoMqr05Yaj3Y",
)
CHAT_ID: Optional[str] = _os.getenv(
    "MODELOX_TELEGRAM_CHAT_ID",
    "1182462908",
)

_BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"


# ─── INTERNALS ───────────────────────────────────────────────────────────────

def _get_chat_id() -> Optional[str]:
    global CHAT_ID
    if CHAT_ID:
        return CHAT_ID
    try:
        url = f"{_BASE_URL}/getUpdates"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        if data.get("ok") and data.get("result"):
            for update in reversed(data["result"]):
                chat = update.get("message", {}).get("chat", {})
                if chat.get("id"):
                    CHAT_ID = str(chat["id"])
                    return CHAT_ID
    except Exception:
        pass
    return None


def _send(text: str) -> bool:
    chat_id = _get_chat_id()
    if not chat_id:
        return False

    def _do():
        try:
            payload = urllib.parse.urlencode({
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
            }).encode()
            req = urllib.request.Request(
                f"{_BASE_URL}/sendMessage", data=payload, method="POST",
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception:
            pass

    threading.Thread(target=_do, daemon=True).start()
    return True


def _f(v: float, decimals: int = 2) -> str:
    """Formato numero con separador de miles + €."""
    return f"{v:,.{decimals}f} €"


def _fraw(v: float, decimals: int = 2) -> str:
    """Formato numero sin simbolo."""
    return f"{v:,.{decimals}f}"


def _sign(v: float, decimals: int = 2) -> str:
    """Numero con signo explicito + €."""
    return f"{v:+,.{decimals}f} €"


def _signraw(v: float, decimals: int = 2) -> str:
    """Numero con signo sin simbolo."""
    return f"{v:+,.{decimals}f}"


# ═════════════════════════════════════════════════════════════════════════════
# TRADING REAL
# ═════════════════════════════════════════════════════════════════════════════

def notificar_inicio_sesion(
    moneda: str,
    activo: str,
    estrategia: str,
    apalancamiento: int,
    monto: float,
    saldo: float,
    modo: str,
) -> None:
    msg = (
        f"<b>MODELOX LIVE</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{activo}  ·  {estrategia}\n"
        f"{apalancamiento}x  ·  {_f(monto)}/trade  ·  {modo.upper()}\n"
        f"Saldo  {_f(saldo)}"
    )
    _send(msg)


def notificar_trade_abierto(
    activo: str,
    direccion: str,
    precio_entrada: float,
    cantidad: float,
    apalancamiento: int,
    sl_precio: float,
    sl_pct: float,
    tp_precio: float,
    tp_pct: float,
    margen: float,
    valor_posicion: float,
    trailing: bool = False,
    trail_act_pct: float = 0.0,
    trail_dist_pct: float = 0.0,
) -> None:
    tag = "LONG" if direccion.upper() == "LONG" else "SHORT"
    trail_line = f"Trail    +{trail_act_pct:.1f}% act / {trail_dist_pct:.1f}% dist\n" if trailing else ""

    msg = (
        f"<b>OPEN {tag}</b>  {activo}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Entry    {_f(precio_entrada)}\n"
        f"Qty      {cantidad:.6f}  ·  {apalancamiento}x\n"
        f"SL       {_f(sl_precio)}  (−{sl_pct:.1f}%)\n"
        f"TP       {_f(tp_precio)}  (+{tp_pct:.1f}%)\n"
        f"{trail_line}"
        f"Margin   {_f(margen)}  →  {_f(valor_posicion)}"
    )
    _send(msg)


def notificar_trade_cerrado(
    activo: str,
    direccion: str,
    precio_entrada: float,
    precio_salida: float,
    cantidad: float,
    pnl_bruto: float,
    comisiones: float,
    pnl_neto: float,
    roi: float,
    razon: str,
) -> None:
    tag = "WIN" if pnl_neto >= 0 else "LOSS"
    reason_clean = razon.upper().replace("_", " ")

    msg = (
        f"<b>CLOSE · {tag}</b>  {activo} {direccion.upper()}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Entry    {_f(precio_entrada)}\n"
        f"Exit     {_f(precio_salida)}  ({reason_clean})\n"
        f"Qty      {cantidad:.4f}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"PnL      {_sign(pnl_bruto, 4)}\n"
        f"Fees     {_sign(comisiones, 4)}\n"
        f"<b>Net      {_sign(pnl_neto, 4)}  ({_signraw(roi)}%)</b>"
    )
    _send(msg)


# ═════════════════════════════════════════════════════════════════════════════
# BACKTESTING / OPTIMIZACION
# ═════════════════════════════════════════════════════════════════════════════

def notificar_inicio_optimizacion(
    estrategia: str,
    strategy_id: int,
    activo: str,
    n_trials: int,
    timeframe: str = "",
    sampler: str = "",
) -> None:
    tf_part = f"  ·  {timeframe}" if timeframe else ""
    sampler_part = f"  ·  {sampler}" if sampler else ""

    msg = (
        f"<b>MODELOX OPT</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"ID{strategy_id}  {estrategia}\n"
        f"{activo}{tf_part}{sampler_part}\n"
        f"Trials  {n_trials}"
    )
    _send(msg)


def notificar_nuevo_best(
    score: float,
    trial: int,
    roi: float,
    saldo_final: float,
    saldo_inicial: float,
    drawdown: float,
    sqn: float,
    winrate: float,
    trades_dia: float = 0.0,
    n_longs: int = 0,
    n_shorts: int = 0,
) -> None:
    delta = saldo_final - saldo_inicial
    total = n_longs + n_shorts

    msg = (
        f"<b>BEST {score:.2f}</b>  TRIAL {trial}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"ROI    {_signraw(roi)}%\n"
        f"PnL    {_sign(delta)}  ({_f(saldo_inicial)} → {_f(saldo_final)})\n"
        f"DD     {drawdown:.1f}%\n"
        f"WR     {winrate:.1f}%  ·  SQN {sqn:.2f}\n"
        f"T/D    {trades_dia:.3f}  ·  {total} (L:{n_longs} / S:{n_shorts})"
    )
    _send(msg)

