"""
================================================================================
VISUAL/TELEGRAM.PY — NOTIFICACIONES EN TIEMPO REAL
================================================================================

PROPÓSITO:
    Sistema de alertas minimalista y eficiente para Telegram.
    Envía notificaciones de trades, inicios de sesión y resultados de optimización.

CARACTERÍSTICAS:
    - Uso asíncrono (threading) para no bloquear el trading.
    - Formato HTML limpio.
    - Soporte para credenciales vía variables de entorno.
    - Filtro de mensajes redundantes.

================================================================================
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

# ─── PROGRESO ────────────────────────────────────────────────────────────────
# Cada cuántos % enviar notificación de progreso durante la optimización.
# Ejemplos: 10 = cada 10% (10 mensajes), 5 = cada 5% (20 mensajes), 1 = cada 1%
PROGRESO_CADA_PCT = 10

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
    tf_part = f" · {timeframe.upper()}" if timeframe else ""
    sampler_part = f" · {sampler.upper()}" if sampler else ""

    msg = (
        f"🚀 <b>OPTIMIZATION STARTED</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📋 <b>{estrategia}</b>  (ID{strategy_id})\n"
        f"📈 {activo.upper()}{tf_part}{sampler_part}\n"
        f"🔄 Trials: <b>{n_trials:,}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━"
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
    sharpe: float = 0.0,
    profit_factor: float = 0.0,
    trades_dia: float = 0.0,
    n_longs: int = 0,
    n_shorts: int = 0,
    n_trials_total: int = 0,
) -> None:
    delta = saldo_final - saldo_inicial
    total = n_longs + n_shorts
    pct_done = f"  ({100 * trial / n_trials_total:.0f}%)" if n_trials_total > 0 else ""
    pnl_emoji = "🟢" if delta >= 0 else "🔴"

    msg = (
        f"🏆🏆🏆 <b>NEW BEST</b> 🏆🏆🏆\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⭐ SCORE  <b>{score:.1f}</b>   T#{trial}{pct_done}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{pnl_emoji} ROI     <b>{roi:+.1f}%</b>\n"
        f"💰 PnL     {_sign(delta)}  \n"
        f"        ({_f(saldo_inicial)} → {_f(saldo_final)})\n"
        f"📉 DD      {drawdown:.1f}%\n"
        f"🎯 WR      {winrate:.1f}%\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"SHARPE {sharpe:.2f}  │  SQN {sqn:.2f}\n"
        f"PF {profit_factor:.2f}  │  T/D {trades_dia:.3f}\n"
        f"Trades {total}  (L:{n_longs} / S:{n_shorts})\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    _send(msg)


def notificar_progreso_optimizacion(
    pct_completado: int,
    trial_actual: int,
    n_trials_total: int,
    roi_medio: float = 0.0,
    pf_medio: float = 0.0,
    sharpe_medio: float = 0.0,
    sqn_medio: float = 0.0,
    score_medio: float = 0.0,
    best_score: float = 0.0,
    best_trial: int = 0,
) -> None:
    """Notifica progreso cada X% con métricas promedio y best score."""
    bar_filled = pct_completado // 10
    bar_empty = 10 - bar_filled
    progress_bar = "█" * bar_filled + "░" * bar_empty

    msg = (
        f"📊 <b>PROGRESS {pct_completado}%</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"[{progress_bar}]  {trial_actual:,} / {n_trials_total:,}\n"
        f"\n"
        f"<b>AVG METRICS</b> μ({trial_actual:,})\n"
        f"  ROI {roi_medio:+.1f}%  │  PF {pf_medio:.2f}\n"
        f"  SHARPE {sharpe_medio:.2f}  │  SQN {sqn_medio:.2f}\n"
        f"  SCORE {score_medio:.1f}\n"
        f"\n"
        f"⭐ BEST  <b>{best_score:.1f}</b>  (T#{best_trial})\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    _send(msg)


def notificar_fin_optimizacion(
    estrategia: str,
    activo: str,
    n_trials: int,
    best_score: float,
    best_trial: int,
    roi_medio: float = 0.0,
    best_roi: float = 0.0,
    elapsed_min: float = 0.0,
) -> None:
    """Notifica fin de optimización con resumen final."""
    time_str = f"{elapsed_min:.0f}min" if elapsed_min < 60 else f"{elapsed_min / 60:.1f}h"

    msg = (
        f"✅ <b>OPTIMIZATION COMPLETE</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📋 {estrategia}  ·  {activo.upper()}\n"
        f"🔄 {n_trials:,} trials  ·  ⏱ {time_str}\n"
        f"\n"
        f"🏆 BEST SCORE  <b>{best_score:.1f}</b>  (T#{best_trial})\n"
        f"   BEST ROI    <b>{best_roi:+.1f}%</b>\n"
        f"   AVG ROI     {roi_medio:+.1f}%\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    _send(msg)

