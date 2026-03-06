"""modelox/core/regime.py — Filtro de Régimen de Mercado (EMA 21/200 en 1D).

PROPÓSITO:
    Determina si el mercado está en régimen ALCISTA o BAJISTA usando
    la relación entre EMA(21) y EMA(200) en timeframe diario (1D).
    
    Reglas:
        - EMA(21) > EMA(200) → ALCISTA (bullish)
        - EMA(21) < EMA(200) → BAJISTA (bearish)
    
    Cuando el filtro está activo, se bloquean las señales de entrada
    en los regímenes no permitidos. El periodo completo se mantiene
    intacto para gráficas y reporting.
    
INTEGRACIÓN:
    Se aplica DESPUÉS de generar señales y ANTES del backtest engine.
    Las señales bloqueadas se ponen a False, pero el DataFrame de 
    precios permanece sin cambios (las gráficas muestran todo el periodo).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import polars as pl


# =============================================================================
# 1. CÁLCULO DE EMAs EN 1D
# =============================================================================

def _resample_to_1d(df: pl.DataFrame) -> pl.DataFrame:
    """Resamplea datos OHLCV (cualquier TF) a velas diarias (1D).
    
    Agrupa por fecha y produce open/high/low/close/volume diarios.
    """
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame debe tener columna 'timestamp'")
    
    df_sorted = df.sort("timestamp")
    
    agg_exprs = [
        pl.col("open").first(),
        pl.col("high").max(),
        pl.col("low").min(),
        pl.col("close").last(),
    ]
    if "volume" in df.columns:
        agg_exprs.append(pl.col("volume").sum())
    
    resampled = (
        df_sorted
        .group_by_dynamic("timestamp", every="1d")
        .agg(agg_exprs)
    )
    
    return resampled.sort("timestamp")


def _compute_ema(series: np.ndarray, period: int) -> np.ndarray:
    """Calcula EMA (Exponential Moving Average) con numpy puro.
    
    Usa el método estándar: multiplicador = 2 / (period + 1).
    Los primeros `period` valores usan SMA como seed.
    """
    n = len(series)
    ema = np.empty(n, dtype=np.float64)
    
    if n == 0:
        return ema
    
    # Multiplicador
    alpha = 2.0 / (period + 1)
    
    # SMA como seed para los primeros `period` valores
    if n < period:
        # No hay suficientes datos: usar SMA acumulativa
        ema[0] = series[0]
        for i in range(1, n):
            ema[i] = ema[i - 1] * (1 - alpha) + series[i] * alpha
        return ema
    
    # Seed = SMA de los primeros `period` valores
    sma = np.mean(series[:period])
    ema[:period - 1] = np.nan
    ema[period - 1] = sma
    
    # EMA recursiva a partir de ahí
    for i in range(period, n):
        ema[i] = ema[i - 1] * (1 - alpha) + series[i] * alpha
    
    return ema


# =============================================================================
# 2. DETERMINACIÓN DEL RÉGIMEN
# =============================================================================

def compute_regime_mask_1d(
    df_1m: pl.DataFrame,
    *,
    ema_fast: int = 14,
    ema_slow: int = 100,
) -> pl.DataFrame:
    """Computa el régimen de mercado (ALCISTA/BAJISTA) en timeframe 1D.
    
    Args:
        df_1m:     DataFrame con datos OHLCV (cualquier TF, típicamente 1m).
        ema_fast:  Periodo de EMA rápida (default: 21).
        ema_slow:  Periodo de EMA lenta (default: 200).
    
    Returns:
        DataFrame con columnas:
            - timestamp: fecha de cada día
            - regime: "ALCISTA" o "BAJISTA"
            - regime_bullish: True si EMA_fast > EMA_slow
    """
    # 1. Resamplear a 1D
    df_1d = _resample_to_1d(df_1m)
    
    if df_1d.is_empty() or len(df_1d) < ema_slow:
        # Sin datos suficientes: asumir neutral (permitir todo)
        return df_1d.with_columns([
            pl.lit("NEUTRAL").alias("regime"),
            pl.lit(True).alias("regime_bullish"),
        ])
    
    # 2. Calcular EMAs sobre close diario
    close_arr = df_1d["close"].to_numpy().astype(np.float64)
    ema_fast_arr = _compute_ema(close_arr, ema_fast)
    ema_slow_arr = _compute_ema(close_arr, ema_slow)
    
    # 3. Determinar régimen: ALCISTA si EMA_fast > EMA_slow
    is_bullish = ema_fast_arr > ema_slow_arr
    regime_labels = np.where(is_bullish, "ALCISTA", "BAJISTA")
    
    # Manejar NaN de los warmup periods: marcar como NEUTRAL
    nan_mask = np.isnan(ema_fast_arr) | np.isnan(ema_slow_arr)
    regime_labels[nan_mask] = "NEUTRAL"
    is_bullish_clean = np.where(nan_mask, True, is_bullish)  # Neutral → permitir todo
    
    df_regime = df_1d.select("timestamp").with_columns([
        pl.Series("regime", regime_labels),
        pl.Series("regime_bullish", is_bullish_clean),
        pl.Series("ema_21", ema_fast_arr),
        pl.Series("ema_200", ema_slow_arr),
    ])
    
    return df_regime


# =============================================================================
# 3. APLICAR FILTRO DE RÉGIMEN A SEÑALES
# =============================================================================

def apply_regime_filter(
    df_signals: pl.DataFrame,
    df_data: pl.DataFrame,
    df_1m_raw: pl.DataFrame,
    *,
    regimen_tipo: str = "ALCISTA",
) -> Tuple[pl.DataFrame, int]:
    """Filtra señales de entrada según el régimen de mercado.
    
    NOTA: Esta función recomputa el régimen completo. Para uso en bucles
    de optimización, usar apply_precomputed_regime_filter() en su lugar.
    """
    df_regime = compute_regime_mask_1d(df_1m_raw)
    if df_regime.is_empty():
        return df_signals, 0
    return apply_precomputed_regime_filter(df_signals, df_data, df_regime, regimen_tipo=regimen_tipo)


def apply_precomputed_regime_filter(
    df_signals: pl.DataFrame,
    df_data: pl.DataFrame,
    df_regime: pl.DataFrame,
    *,
    regimen_tipo: str = "ALCISTA",
) -> Tuple[pl.DataFrame, int]:
    """Filtra señales usando un régimen PRE-COMPUTADO (rápido, sin recalcular EMAs).
    
    Diseñado para ser llamado en cada trial de optimización sin overhead.
    Usa un Polars join vectorizado (C-level) — sin bucles Python.
    
    Comportamiento:
    - Bloquea ENTRADAS en períodos de régimen no permitido.
    - Genera señales de EXIT forzado en períodos no permitidos,
      para que trades abiertos se cierren automáticamente al cambiar de régimen.
    
    Args:
        df_signals:   DataFrame con signal_long, signal_short.
        df_data:      DataFrame OHLCV del timeframe de entrada (con timestamp).
        df_regime:    DataFrame PRE-COMPUTADO con columnas timestamp + regime.
        regimen_tipo: "ALCISTA" o "BAJISTA" (el régimen permitido).
    
    Returns:
        Tuple de (df_signals filtrado, días_operables).
    """
    if df_regime.is_empty() or "timestamp" not in df_data.columns:
        return df_signals, 0
    
    regimen_upper = regimen_tipo.upper().strip()
    
    # 1. Preparar tabla de régimen con fecha como clave (solo columnas necesarias)
    regime_lookup = df_regime.select(
        pl.col("timestamp").dt.date().alias("_regime_date"),
        pl.col("regime"),
    )
    
    # 2. Extraer fecha de cada vela del TF de entrada
    trade_dates = df_data.select(
        pl.col("timestamp").dt.date().alias("_regime_date")
    )
    
    # 3. Join vectorizado (Polars C-level, ultra rápido)
    joined = trade_dates.join(regime_lookup, on="_regime_date", how="left")
    
    # 4. Máscara: permitir si régimen coincide o es NEUTRAL/null
    regime_col = joined["regime"].fill_null("NEUTRAL")
    allowed = (regime_col == regimen_upper) | (regime_col == "NEUTRAL")
    not_allowed = ~allowed  # Máscara invertida para exits forzados
    
    # 5. Contar días operables
    dias_operables = regime_lookup.filter(pl.col("regime") == regimen_upper).height
    
    # 6. Aplicar máscara a señales de entrada (bloquear entries en régimen no permitido)
    new_columns = []
    if "signal_long" in df_signals.columns:
        new_columns.append(
            (df_signals["signal_long"].fill_null(False) & allowed).alias("signal_long")
        )
    if "signal_short" in df_signals.columns:
        new_columns.append(
            (df_signals["signal_short"].fill_null(False) & allowed).alias("signal_short")
        )
    
    # 7. Forzar EXIT en períodos de régimen no permitido
    #    → el kernel Numba usa exit_long / exit_short para cerrar trades abiertos
    if "exit_long" in df_signals.columns:
        new_columns.append(
            (df_signals["exit_long"].fill_null(False) | not_allowed).alias("exit_long")
        )
    else:
        new_columns.append(not_allowed.alias("exit_long"))
    
    if "exit_short" in df_signals.columns:
        new_columns.append(
            (df_signals["exit_short"].fill_null(False) | not_allowed).alias("exit_short")
        )
    else:
        new_columns.append(not_allowed.alias("exit_short"))
    
    if new_columns:
        df_signals = df_signals.with_columns(new_columns)
    
    return df_signals, dias_operables


def compute_operable_days(
    df_1m_raw: pl.DataFrame,
    fecha_inicio: str,
    fecha_fin: str,
    regimen_tipo: str = "ALCISTA",
) -> int:
    """Calcula los días operables para un régimen dado en un periodo.
    
    Útil para ajustar trades_por_dia cuando el régimen filtra días.
    
    Args:
        df_1m_raw:     DataFrame OHLCV en 1m.
        fecha_inicio:  Fecha inicio (str YYYY-MM-DD).
        fecha_fin:     Fecha fin (str YYYY-MM-DD).
        regimen_tipo:  "ALCISTA" o "BAJISTA".
    
    Returns:
        Número de días donde el régimen coincide.
    """
    df_regime = compute_regime_mask_1d(df_1m_raw)
    
    if df_regime.is_empty():
        return 0
    
    regimen_upper = regimen_tipo.upper().strip()
    
    # Filtrar por fecha si es posible
    try:
        from .types import filter_by_date
        df_regime = filter_by_date(df_regime, fecha_inicio, fecha_fin)
    except Exception:
        pass
    
    # Contar días con el régimen permitido
    return df_regime.filter(pl.col("regime") == regimen_upper).height
