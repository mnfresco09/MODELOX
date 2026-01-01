from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import polars as pl


def normalize_timeframe_to_suffix(timeframe: Any) -> str:
    """Normaliza un timeframe a sufijo estilo archivos: 5m/15m/1h.

    Acepta:
    - int/float (minutos): 5, 15, 60
    - str: "5", "5m", "15", "15m", "60", "1h"

    Default: "1h" si no se puede parsear.
    """
    if timeframe is None:
        return "1h"

    if isinstance(timeframe, (int, float)):
        m = int(timeframe)
        return "1h" if m == 60 else f"{m}m"

    s = str(timeframe).strip().lower()
    if not s:
        return "1h"

    if s.endswith("h"):
        return "1h" if s in {"1h", "60m", "60"} else s

    if s.endswith("m"):
        s = s[:-1]

    try:
        m = int(float(s))
    except Exception:
        return "1h"

    return "1h" if m == 60 else f"{m}m"


def suffix_to_minutes(suffix: str) -> int:
    s = str(suffix).strip().lower()
    if s == "1h":
        return 60
    if s.endswith("m"):
        s = s[:-1]
    return int(float(s))


def convert_warmup_bars_to_base(*, warmup_bars: int, from_tf: str, to_tf: str) -> int:
    """Convierte warmup expresado en barras from_tf a barras de to_tf.

    Ej:
      warmup=50 en 15m -> base=5m => 150 barras
    """
    wb = int(warmup_bars)
    if wb <= 0:
        return 0
    f = suffix_to_minutes(from_tf)
    t = suffix_to_minutes(to_tf)
    if t <= 0:
        return wb
    # total minutos cubiertos por warmup
    mins = wb * f
    return int(math.ceil(mins / t))


def align_signals_to_base(
    *,
    df_base: pl.DataFrame,
    df_signals: pl.DataFrame,
) -> pl.DataFrame:
    """Alinea columnas de señales/indicadores de df_signals a df_base sin lookahead.

    - Se hace join_asof usando el último timestamp de df_signals <= timestamp de df_base.
    - Se conservan OHLCV del base timeframe.
    - Se copian columnas no-OHLCV desde df_signals.
    """
    if "timestamp" not in df_base.columns:
        raise ValueError("df_base must contain 'timestamp'")
    if "timestamp" not in df_signals.columns:
        raise ValueError("df_signals must contain 'timestamp'")

    base = df_base.sort("timestamp")
    sig = df_signals.sort("timestamp")

    # No traer OHLCV del dataframe de señales
    skip = {"open", "high", "low", "close", "volume"}
    extra_cols = [c for c in sig.columns if c not in skip and c != "timestamp"]

    sig_small = sig.select(["timestamp", *extra_cols])

    out = base.join_asof(sig_small, on="timestamp", strategy="backward")

    # Señales booleanas: null -> False
    for col in ("signal_long", "signal_short"):
        if col in out.columns:
            out = out.with_columns(pl.col(col).fill_null(False).cast(pl.Boolean))

    return out
