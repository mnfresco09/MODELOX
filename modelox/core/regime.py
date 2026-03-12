"""modelox/core/regime.py — Filtro de Régimen de Mercado (EMA 21/200 | MACD 12/26/9 en 1D).

PROPÓSITO:
    Determina si el mercado está en régimen ALCISTA o BAJISTA.

    Modo EMA (por defecto):
        - EMA(21) > EMA(200) → ALCISTA (bullish)
        - EMA(21) < EMA(200) → BAJISTA (bearish)

    Modo MACD (12, 26, 9):
        - Histograma > 0 → ALCISTA
        - Histograma ≤ 0 → BAJISTA
        - Buy&Hold sale cuando el histograma cruza de positivo a negativo.

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
# PARÁMETROS DE RÉGIMEN — ajustar aquí para cambiar globalmente
# =============================================================================
EMA_FAST_DEFAULT: int = 7    # EMA rápida (para cruce de régimen)
EMA_SLOW_DEFAULT: int = 21   # EMA lenta (umbral de precios y cruce)


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

def _compute_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula ADX(period) con +DI y -DI usando el suavizado de Wilder.

    Returns:
        (adx, plus_di, minus_di)  — arrays del mismo tamaño que la entrada.
        Los primeros ~2*period valores son NaN (warm-up).
    """
    n = len(close)
    adx      = np.full(n, np.nan)
    plus_di  = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)

    if n < 2 * period + 1:
        return adx, plus_di, minus_di

    # 1. True Range, +DM, -DM (elemento a elemento)
    tr_arr   = np.zeros(n)
    plus_dm  = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        high_diff = high[i] - high[i - 1]
        low_diff  = low[i - 1] - low[i]
        tr_arr[i] = max(high[i] - low[i],
                        abs(high[i] - close[i - 1]),
                        abs(low[i]  - close[i - 1]))
        plus_dm[i]  = high_diff if (high_diff > low_diff and high_diff > 0) else 0.0
        minus_dm[i] = low_diff  if (low_diff > high_diff and low_diff > 0)  else 0.0

    # 2. Wilder smoothing (seed = suma simple de los primeros `period` valores)
    alpha = 1.0 / period

    atr   = np.zeros(n)
    s_pdm = np.zeros(n)
    s_mdm = np.zeros(n)
    dx_arr = np.full(n, np.nan)

    atr[period]   = np.sum(tr_arr[1:period + 1])
    s_pdm[period] = np.sum(plus_dm[1:period + 1])
    s_mdm[period] = np.sum(minus_dm[1:period + 1])

    for i in range(period + 1, n):
        atr[i]   = atr[i - 1]   - atr[i - 1]   * alpha + tr_arr[i]
        s_pdm[i] = s_pdm[i - 1] - s_pdm[i - 1] * alpha + plus_dm[i]
        s_mdm[i] = s_mdm[i - 1] - s_mdm[i - 1] * alpha + minus_dm[i]

    # 3. +DI / -DI
    for i in range(period, n):
        if atr[i] == 0:
            continue
        pdi = 100.0 * s_pdm[i] / atr[i]
        mdi = 100.0 * s_mdm[i] / atr[i]
        plus_di[i]  = pdi
        minus_di[i] = mdi
        denom = pdi + mdi
        dx_arr[i] = 100.0 * abs(pdi - mdi) / denom if denom != 0 else 0.0

    # 4. ADX = Wilder smoothing del DX (desde el segundo periodo completo)
    first_dx = 2 * period  # primer índice donde DX ya tiene `period` valores previos
    if first_dx < n:
        adx[first_dx] = np.nanmean(dx_arr[period:first_dx + 1])
        for i in range(first_dx + 1, n):
            if not np.isnan(dx_arr[i]):
                adx[i] = adx[i - 1] * (1 - alpha) + dx_arr[i] * alpha

    return adx, plus_di, minus_di


def _compute_macd(
    series: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula MACD(fast, slow, signal) y devuelve (macd_line, signal_line, histogram).

    Warm-up: los primeros `slow - 1` valores del MACD serán NaN.
    """
    ema_fast_arr = _compute_ema(series, fast)
    ema_slow_arr = _compute_ema(series, slow)
    macd_line = ema_fast_arr - ema_slow_arr

    # La señal se calcula sobre el MACD (ignorando NaN iniciales)
    signal_line = np.full_like(macd_line, np.nan)
    first_valid = slow - 1  # primer índice con MACD válido
    if len(macd_line) > first_valid + signal:
        signal_line[first_valid:] = _compute_ema(macd_line[first_valid:], signal)

    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_regime_mask_1d(
    df_1m: pl.DataFrame,
    *,
    ema_fast: int = EMA_FAST_DEFAULT,
    ema_slow: int = EMA_SLOW_DEFAULT,
    indicador: str = "EMA",
) -> pl.DataFrame:
    """Computa el régimen de mercado (ALCISTA/BAJISTA) en timeframe 1D.

    Args:
        df_1m:      DataFrame con datos OHLCV (cualquier TF, típicamente 1m).
        ema_fast:   Periodo de EMA rápida (modo EMA, default: 21).
        ema_slow:   Periodo de EMA lenta  (modo EMA, default: 200).
        indicador:  "EMA"  → cruces EMA(21)/EMA(200) en 1D.
                    "MACD" → histograma MACD(12,26,9) en 1D.
                    "ADX"  → ADX(14) + DI en 1D:
                             ALCISTA  si ADX > 25 y +DI > -DI
                             BAJISTA  si ADX > 25 y -DI > +DI
                             NEUTRAL  si ADX ≤ 25 (mercado lateral)

    Returns:
        DataFrame con columnas:
            - timestamp:       fecha de cada día
            - regime:          "ALCISTA" | "BAJISTA" | "NEUTRAL"
            - regime_bullish:  True si régimen es ALCISTA
            (modo EMA)  ema_<fast>, ema_<slow>, ema_slow
            (modo MACD) macd_line, macd_signal, macd_hist
            (modo ADX)  adx, plus_di, minus_di
    """
    ind = indicador.upper().strip()

    # 1. Resamplear a 1D
    df_1d = _resample_to_1d(df_1m)

    if df_1d.is_empty():
        return df_1d.with_columns([
            pl.lit("NEUTRAL").alias("regime"),
            pl.lit(True).alias("regime_bullish"),
        ])

    close_arr = df_1d["close"].to_numpy().astype(np.float64)

    # ------------------------------------------------------------------
    # Modo MACD
    # ------------------------------------------------------------------
    if ind == "MACD":
        macd_line, signal_line, histogram = _compute_macd(close_arr)

        nan_mask = np.isnan(histogram)
        is_bullish = np.where(nan_mask, True, histogram > 0)
        regime_labels = np.where(nan_mask, "NEUTRAL",
                                 np.where(histogram > 0, "ALCISTA", "BAJISTA"))

        return df_1d.select("timestamp").with_columns([
            pl.Series("regime", regime_labels),
            pl.Series("regime_bullish", is_bullish.astype(bool)),
            pl.Series("macd_line", macd_line),
            pl.Series("macd_signal", signal_line),
            pl.Series("macd_hist", histogram),
        ])

    # ------------------------------------------------------------------
    # Modo ADX
    # ------------------------------------------------------------------
    if ind == "ADX":
        high_arr  = df_1d["high"].to_numpy().astype(np.float64)
        low_arr   = df_1d["low"].to_numpy().astype(np.float64)
        adx_arr, plus_di_arr, minus_di_arr = _compute_adx(high_arr, low_arr, close_arr)

        # ALCISTA: ADX > 25 y +DI > -DI
        # BAJISTA: ADX > 25 y -DI > +DI
        # NEUTRAL: ADX ≤ 25 o NaN (mercado lateral)
        nan_mask = np.isnan(adx_arr) | np.isnan(plus_di_arr) | np.isnan(minus_di_arr)
        strong   = (~nan_mask) & (adx_arr > 25)
        bullish  = strong & (plus_di_arr > minus_di_arr)
        bearish  = strong & (minus_di_arr >= plus_di_arr)

        regime_labels = np.where(nan_mask, "NEUTRAL",
                        np.where(bullish,  "ALCISTA",
                        np.where(bearish,  "BAJISTA", "NEUTRAL")))
        is_bullish = bullish  # NEUTRAL → False (bloquear entradas cuando no hay tendencia)

        return df_1d.select("timestamp").with_columns([
            pl.Series("regime", regime_labels),
            pl.Series("regime_bullish", is_bullish.astype(bool)),
            pl.Series("adx", adx_arr),
            pl.Series("plus_di", plus_di_arr),
            pl.Series("minus_di", minus_di_arr),
        ])

    # ------------------------------------------------------------------
    # Modo EMA (por defecto)
    # ------------------------------------------------------------------
    if len(df_1d) < ema_slow:
        return df_1d.with_columns([
            pl.lit("NEUTRAL").alias("regime"),
            pl.lit(True).alias("regime_bullish"),
        ])

    ema_fast_arr = _compute_ema(close_arr, ema_fast)
    ema_slow_arr = _compute_ema(close_arr, ema_slow)

    is_bullish = ema_fast_arr > ema_slow_arr
    regime_labels = np.where(is_bullish, "ALCISTA", "BAJISTA")

    nan_mask = np.isnan(ema_fast_arr) | np.isnan(ema_slow_arr)
    regime_labels[nan_mask] = "NEUTRAL"
    is_bullish_clean = np.where(nan_mask, True, is_bullish)

    return df_1d.select("timestamp").with_columns([
        pl.Series("regime", regime_labels),
        pl.Series("regime_bullish", is_bullish_clean.astype(bool)),
        pl.Series(f"ema_{ema_fast}", ema_fast_arr),
        pl.Series(f"ema_{ema_slow}", ema_slow_arr),
        pl.Series("ema_slow", ema_slow_arr),   # alias canónico para B&H exit
    ])


# =============================================================================
# 3. APLICAR FILTRO DE RÉGIMEN A SEÑALES
# =============================================================================

def apply_regime_filter(
    df_signals: pl.DataFrame,
    df_data: pl.DataFrame,
    df_1m_raw: pl.DataFrame,
    *,
    regimen_tipo: str = "ALCISTA",
    regimen_buy_hold_activo: bool = False,
    regimen_buy_hold_saldo: object = "ALL",
    indicador: str = "EMA",
) -> Tuple[pl.DataFrame, int]:
    """Filtra señales de entrada según el régimen de mercado.

    NOTA: Esta función recomputa el régimen completo. Para uso en bucles
    de optimización, usar apply_precomputed_regime_filter() en su lugar.
    """
    df_regime = compute_regime_mask_1d(df_1m_raw, indicador=indicador)
    if df_regime.is_empty():
        return df_signals, 0
    return apply_precomputed_regime_filter(
        df_signals,
        df_data,
        df_regime,
        regimen_tipo=regimen_tipo,
        regimen_buy_hold_activo=regimen_buy_hold_activo,
        regimen_buy_hold_saldo=regimen_buy_hold_saldo,
    )


def apply_precomputed_regime_filter(
    df_signals: pl.DataFrame,
    df_data: pl.DataFrame,
    df_regime: pl.DataFrame,
    *,
    regimen_tipo: str = "ALCISTA",
    regimen_buy_hold_activo: bool = False,
    regimen_buy_hold_saldo: object = "ALL",
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
    _regime_cols = [
        pl.col("timestamp").dt.date().alias("_regime_date"),
        pl.col("regime"),
    ]
    if "ema_slow" in df_regime.columns:
        _regime_cols.append(pl.col("ema_slow"))
    regime_lookup = df_regime.select(_regime_cols)

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

    # Modo especial: Buy&Hold contracíclico para REGIMEN_TIPO=BAJISTA.
    # Comportamiento:
    # - En BAJISTA/NEUTRAL: opera NORMAL según la estrategia.
    # - En ALCISTA: opera SOLO Buy&Hold LONG.
    # - Se cierra Buy&Hold al pasar de ALCISTA a BAJISTA/NEUTRAL.
    bh_enabled = bool(regimen_buy_hold_activo) and regimen_upper == "BAJISTA"
    if bh_enabled:
        bullish = (regime_col == "ALCISTA")
        bearish_or_neutral = ~bullish

        sig_long_base = df_signals["signal_long"].fill_null(False) if "signal_long" in df_signals.columns else pl.Series(np.zeros(len(bullish), dtype=np.bool_))
        sig_short_base = df_signals["signal_short"].fill_null(False) if "signal_short" in df_signals.columns else pl.Series(np.zeros(len(bullish), dtype=np.bool_))
        exit_long_base = df_signals["exit_long"].fill_null(False) if "exit_long" in df_signals.columns else pl.Series(np.zeros(len(bullish), dtype=np.bool_))
        exit_short_base = df_signals["exit_short"].fill_null(False) if "exit_short" in df_signals.columns else pl.Series(np.zeros(len(bullish), dtype=np.bool_))

        # Cierre Buy&Hold: el low de la vela cae por debajo de la EMA lenta (1D).
        # Condición adicional: solo aplica si llevamos ≥ 2 días en régimen ALCISTA,
        # para evitar salidas inmediatas en el primer día de la posición.
        MIN_DIAS_PARA_EXIT_LOW = 2

        # Contador de días consecutivos en ALCISTA (desde df_regime diario)
        _bull_flags = (df_regime["regime"] == "ALCISTA").to_list()
        _consec: list[int] = []
        _cnt = 0
        for _b in _bull_flags:
            _cnt = _cnt + 1 if _b else 0
            _consec.append(_cnt)
        _consec_lookup = df_regime.select(
            pl.col("timestamp").dt.date().alias("_regime_date")
        ).with_columns(pl.Series("_bullish_days", _consec))
        _joined_days = trade_dates.join(_consec_lookup, on="_regime_date", how="left")
        enough_days = _joined_days["_bullish_days"].fill_null(0) >= MIN_DIAS_PARA_EXIT_LOW

        if "ema_slow" in joined.columns and "low" in df_data.columns:
            ema_slow_per_bar = joined["ema_slow"]
            low_arr = df_data["low"]
            low_below_slow = (low_arr < ema_slow_per_bar).fill_null(False) & enough_days
        else:
            # Fallback: solo detectar cruce diario (comportamiento anterior)
            low_below_slow = pl.Series([False] * len(bullish))

        bullish_prev = bullish.shift(1).fill_null(False)
        close_bh_long = low_below_slow | (bearish_or_neutral & bullish_prev)

        # Reglas:
        # 1) En BAJISTA/NEUTRAL => operar NORMAL según estrategia.
        # 2) En ALCISTA => activar Buy&Hold LONG continuo, sin shorts.
        # 3) Al salir de ALCISTA => cerrar LONG de Buy&Hold.
        signal_long_final = (sig_long_base & bearish_or_neutral) | bullish
        signal_short_final = sig_short_base & bearish_or_neutral
        exit_long_final = (exit_long_base & bearish_or_neutral) | close_bh_long
        exit_short_final = (exit_short_base & bearish_or_neutral) | bullish

        df_signals = df_signals.with_columns([
            signal_long_final.alias("signal_long"),
            signal_short_final.alias("signal_short"),
            exit_long_final.alias("exit_long"),
            exit_short_final.alias("exit_short"),
            bullish.alias("regime_buy_hold_long"),
        ])

        # Exponer metadata para runner/engine/reporting
        try:
            saldo_cfg = str(regimen_buy_hold_saldo).strip().upper()
            buy_hold_saldo_all = saldo_cfg in {"ALL", "TODO", "100%", "100"}
            buy_hold_saldo_val = 0.0 if buy_hold_saldo_all else max(0.0, float(regimen_buy_hold_saldo))
        except Exception:
            buy_hold_saldo_all = True
            buy_hold_saldo_val = 0.0

        df_signals = df_signals.with_columns([
            pl.lit(True).alias("regime_buy_hold_mode"),
            pl.lit(buy_hold_saldo_all).alias("regime_buy_hold_saldo_all"),
            pl.lit(float(buy_hold_saldo_val)).alias("regime_buy_hold_saldo"),
        ])

        dias_operables = regime_lookup.filter(pl.col("regime") == "ALCISTA").height
        return df_signals, dias_operables
    
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
