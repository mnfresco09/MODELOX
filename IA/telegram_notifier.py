"""
================================================================================
IA/TELEGRAM_NOTIFIER.PY — NOTIFICACIONES TELEGRAM PARA PIPELINE IA
================================================================================
Envía notificaciones en tiempo real sobre el progreso del entrenamiento,
resultados de folds y métricas finales.

Usa las credenciales de visual/telegram.py
================================================================================
"""

from __future__ import annotations

import threading
import urllib.request
import urllib.parse
import json
from typing import Dict, List, Optional
import os

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────
BOT_TOKEN = os.getenv(
    "MODELOX_TELEGRAM_TOKEN",
    "8571553580:AAEOsSs5c1o_AMju3dofdIcUoMqr05Yaj3Y",
)
CHAT_ID = os.getenv(
    "MODELOX_TELEGRAM_CHAT_ID",
    "1182462908",
)

_BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
_ENABLED = True  # Cambiar a False para desactivar notificaciones


# ─── FUNCIONES INTERNAS ───────────────────────────────────────────────────────

def _send(text: str) -> bool:
    """Envía mensaje a Telegram de forma asíncrona."""
    if not _ENABLED or not CHAT_ID:
        return False

    def _do():
        try:
            payload = urllib.parse.urlencode({
                "chat_id": CHAT_ID,
                "text": text,
                "parse_mode": "HTML",
            }).encode()
            req = urllib.request.Request(
                f"{_BASE_URL}/sendMessage", data=payload, method="POST",
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception:
            pass  # Silencioso si falla

    threading.Thread(target=_do, daemon=True).start()
    return True


# ─── NOTIFICACIONES DEL PIPELINE ──────────────────────────────────────────────

def notificar_inicio_pipeline(
    quick_mode: bool,
    max_folds: int,
    optimize: bool,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
) -> None:
    """Notifica el inicio del pipeline IA."""
    modo = "🚀 QUICK" if quick_mode else "🔥 FULL"
    opt_str = " + OPTUNA" if optimize else ""
    dates = f"\n📅 {date_start} → {date_end}" if date_start and date_end else ""
    
    msg = (
        f"<b>🤖 GRU TRADING AI</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{modo}{opt_str}\n"
        f"📊 Folds: {max_folds}{dates}\n"
        f"━━━━━━━━━━━━━━━━━━━━"
    )
    _send(msg)


def notificar_inicio_fold(
    fold_n: int,
    total_folds: int,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    n_train: int,
    n_val: int,
) -> None:
    """Notifica el inicio de un fold."""
    msg = (
        f"<b>📂 FOLD {fold_n}/{total_folds}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🟢 Train: {train_start} → {train_end}\n"
        f"   Secuencias: {n_train:,}\n"
        f"🔵 Val: {val_start} → {val_end}\n"
        f"   Secuencias: {n_val:,}\n"
        f"━━━━━━━━━━━━━━━━━━━━"
    )
    _send(msg)


def notificar_fin_entrenamiento(
    fold_n: int,
    total_folds: int,
    best_epoch: int,
    best_val_loss: float,
    elapsed_sec: float,
) -> None:
    """Notifica el fin del entrenamiento de un fold."""
    time_str = f"{elapsed_sec:.0f}s" if elapsed_sec < 120 else f"{elapsed_sec/60:.1f}min"
    
    msg = (
        f"<b>✅ FOLD {fold_n}/{total_folds} TRAINED</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🏆 Best Epoch: {best_epoch}\n"
        f"📉 Val Loss: {best_val_loss:.5f}\n"
        f"⏱ Time: {time_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━"
    )
    _send(msg)


def notificar_resultados_fold(
    fold_n: int,
    total_folds: int,
    metrics: Dict,
) -> None:
    """Notifica los resultados del backtest de un fold."""
    sqn = metrics.get("sqn", 0)
    roi = metrics.get("roi", 0)
    dd = metrics.get("max_drawdown", 0)
    wr = metrics.get("winrate", 0)
    n_trades = metrics.get("n_trades", 0)
    pnl = metrics.get("pnl_total", 0)
    
    # Emojis según resultados
    roi_emoji = "🟢" if roi > 0 else "🔴" if roi < 0 else "⚪"
    sqn_emoji = "🏆" if sqn >= 3 else "✅" if sqn >= 2 else "⚠️" if sqn >= 1 else "❌"
    
    msg = (
        f"<b>📊 FOLD {fold_n}/{total_folds} RESULTS</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{sqn_emoji} SQN: <b>{sqn:.3f}</b>\n"
        f"{roi_emoji} ROI: <b>{roi:+.2f}%</b>\n"
        f"📉 Drawdown: {dd:.2f}%\n"
        f"🎯 Win Rate: {wr:.1f}%\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 Trades: {n_trades}\n"
        f"💰 PnL: {pnl:+.2f} $\n"
        f"━━━━━━━━━━━━━━━━━━━━"
    )
    _send(msg)


def notificar_sin_trades(
    fold_n: int,
    total_folds: int,
    n_long: int,
    n_short: int,
    n_none: int,
) -> None:
    """Notifica cuando un fold no genera trades."""
    msg = (
        f"<b>⚠️ FOLD {fold_n}/{total_folds}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"❌ <b>SIN TRADES</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Señales generadas:\n"
        f"  🟢 LONG: {n_long}\n"
        f"  🔴 SHORT: {n_short}\n"
        f"  ⚪ None: {n_none}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💡 Ajustar umbrales en config.py"
    )
    _send(msg)


def notificar_resumen_final(
    n_folds: int,
    total_trades: int,
    total_longs: int,
    total_shorts: int,
    avg_roi: float,
    avg_wr: float,
    avg_sqn: float,
    total_pnl: float,
    elapsed_total_min: float,
) -> None:
    """Notifica el resumen final de todos los folds."""
    time_str = f"{elapsed_total_min:.0f}min" if elapsed_total_min < 120 else f"{elapsed_total_min/60:.1f}h"
    roi_emoji = "🟢" if avg_roi > 0 else "🔴" if avg_roi < 0 else "⚪"
    sqn_emoji = "🏆" if avg_sqn >= 3 else "✅" if avg_sqn >= 2 else "⚠️" if avg_sqn >= 1 else "❌"
    
    msg = (
        f"<b>🏁 PIPELINE COMPLETADO</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 Folds: {n_folds}\n"
        f"⏱ Tiempo: {time_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>MÉTRICAS PROMEDIO</b>\n"
        f"{sqn_emoji} SQN: <b>{avg_sqn:.3f}</b>\n"
        f"{roi_emoji} ROI: <b>{avg_roi:+.2f}%</b>\n"
        f"🎯 Win Rate: {avg_wr:.1f}%\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 Total Trades: {total_trades}\n"
        f"   🟢 Longs: {total_longs}\n"
        f"   🔴 Shorts: {total_shorts}\n"
        f"💰 PnL Total: {total_pnl:+.2f} $\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"✅ Resultados guardados en IA/resultados/"
    )
    _send(msg)


def notificar_optuna_inicio(
    n_trials: int,
    timeout_min: int,
) -> None:
    """Notifica el inicio de optimización Optuna."""
    msg = (
        f"<b>🔮 OPTUNA OPTIMIZATION</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🔄 Trials: {n_trials}\n"
        f"⏱ Timeout: {timeout_min}min\n"
        f"━━━━━━━━━━━━━━━━━━━━"
    )
    _send(msg)


def notificar_optuna_resultado(
    best_params: Dict,
    best_val_loss: float,
) -> None:
    """Notifica los mejores hiperparámetros encontrados."""
    params_str = "\n".join([f"  {k}: {v}" for k, v in best_params.items() if k != "best_val_loss"])
    
    msg = (
        f"<b>🏆 OPTUNA BEST PARAMS</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{params_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📉 Val Loss: {best_val_loss:.5f}\n"
        f"━━━━━━━━━━━━━━━━━━━━"
    )
    _send(msg)


def set_enabled(enabled: bool) -> None:
    """Activa o desactiva las notificaciones."""
    global _ENABLED
    _ENABLED = enabled
