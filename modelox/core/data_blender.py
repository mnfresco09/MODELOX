"""
================================================================================
DATA BLENDER - Alquimia de Datos Multi-Timeframe
================================================================================

Este módulo implementa la arquitectura "Universal Multi-Timeframe" de MODELOX.

CONCEPTO:
  El sistema SIEMPRE carga datos de máxima resolución ("Átomo", ej: 1m).
  El "Blender" genera timeframes superiores (1h, 4h) en memoria usando Polars.

SEGURIDAD ANTI-LOOKAHEAD:
  Los datos resampleados usan .shift(1) para evitar mirar al futuro.
  Ejemplo: En la vela de las 10:15 (1m), la vela de 1h visible es la de 09:00-10:00,
  NO la de 10:00-11:00 (que contiene información del futuro).

FLUJO:
  1. Estrategia solicita: get_required_timeframes() -> ["1h", "4h"]
  2. Blender genera velas superiores con group_by_dynamic
  3. Aplica .shift(1) para seguridad
  4. Une al DataFrame base con join_asof(strategy="backward")
  5. Columnas resultantes: close (1m), close_1h, close_4h

Autor: Sistema MODELOX
Versión: 1.0 (Enero 2026)
================================================================================
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import polars as pl

from modelox.core.types import normalize_timeframe_to_suffix


# ==============================================================================
# CONSTANTES Y MAPEOS
# ==============================================================================

# Mapeo de sufijos a duración en Polars
_TF_TO_POLARS_DURATION: Dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "1d": "1d",
}

# Columnas OHLCV estándar para resamplear
_OHLCV_RESAMPLE_EXPRS = {
    "open": pl.col("open").first(),
    "high": pl.col("high").max(),
    "low": pl.col("low").min(),
    "close": pl.col("close").last(),
    "volume": pl.col("volume").sum(),
}


# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

def _get_polars_duration(tf_suffix: str) -> str:
    """Convierte sufijo de timeframe a duración Polars."""
    return _TF_TO_POLARS_DURATION.get(tf_suffix, tf_suffix)


def _tf_to_minutes(tf_suffix: str) -> int:
    """Convierte sufijo de timeframe a minutos."""
    s = str(tf_suffix).strip().lower()
    if s.endswith("m"):
        return int(s[:-1])
    elif s.endswith("h"):
        return int(s[:-1]) * 60
    elif s.endswith("d"):
        return int(s[:-1]) * 1440
    try:
        return int(s)
    except ValueError:
        return 1


def _detect_base_timeframe(df: pl.DataFrame) -> str:
    """Detecta el timeframe base del DataFrame analizando intervalos."""
    ts_col = "timestamp" if "timestamp" in df.columns else "datetime"
    if ts_col not in df.columns:
        return "1m"  # Default
    
    try:
        # Tomar diferencias entre timestamps consecutivos
        sample = df.head(100)
        if sample.height < 2:
            return "1m"
        
        diffs = sample.select(
            pl.col(ts_col).diff().drop_nulls().dt.total_minutes()
        ).to_series()
        
        if diffs.is_empty():
            return "1m"
        
        median_diff = int(diffs.median())
        
        # Mapear a timeframe conocido
        if median_diff <= 1:
            return "1m"
        elif median_diff <= 5:
            return "5m"
        elif median_diff <= 15:
            return "15m"
        elif median_diff <= 30:
            return "30m"
        elif median_diff <= 60:
            return "1h"
        elif median_diff <= 120:
            return "2h"
        elif median_diff <= 240:
            return "4h"
        else:
            return "1d"
    except Exception:
        return "1m"


# ==============================================================================
# RESAMPLING CON SEGURIDAD ANTI-LOOKAHEAD
# ==============================================================================

def resample_ohlcv(
    df: pl.DataFrame,
    target_tf: str,
    *,
    ts_col: str = "timestamp",
    shift_bars: int = 1,
) -> pl.DataFrame:
    """
    Resamplea OHLCV a un timeframe superior con protección anti-lookahead.
    
    Args:
        df: DataFrame con datos OHLCV de alta resolución
        target_tf: Timeframe destino (ej: "1h", "4h")
        ts_col: Nombre de la columna de timestamp
        shift_bars: Barras a desplazar para evitar lookahead (default=1)
    
    Returns:
        DataFrame resampleado con columnas sufijadas (ej: close_1h)
    
    CRÍTICO - Seguridad Anti-Lookahead:
        Si estoy en la vela de las 10:15 (base 1m), la vela de 1h (10:00-11:00)
        contiene información del futuro (cierre de las 11:00).
        Aplicamos .shift(1) para ver la vela anterior (09:00-10:00).
    """
    if ts_col not in df.columns:
        ts_col = "datetime" if "datetime" in df.columns else "timestamp"
    
    tf_suffix = normalize_timeframe_to_suffix(target_tf)
    duration = _get_polars_duration(tf_suffix)
    
    # Asegurar que timestamp es datetime
    df_work = df.lazy()
    if df.schema.get(ts_col) != pl.Datetime:
        df_work = df_work.with_columns(pl.col(ts_col).cast(pl.Datetime("us")))
    
    # Resamplear con group_by_dynamic
    resampled = (
        df_work
        .sort(ts_col)
        .group_by_dynamic(ts_col, every=duration)
        .agg([
            _OHLCV_RESAMPLE_EXPRS["open"].alias(f"open_{tf_suffix}"),
            _OHLCV_RESAMPLE_EXPRS["high"].alias(f"high_{tf_suffix}"),
            _OHLCV_RESAMPLE_EXPRS["low"].alias(f"low_{tf_suffix}"),
            _OHLCV_RESAMPLE_EXPRS["close"].alias(f"close_{tf_suffix}"),
            _OHLCV_RESAMPLE_EXPRS["volume"].alias(f"volume_{tf_suffix}"),
        ])
        .collect()
    )
    
    # CRÍTICO: Shift para anti-lookahead
    # La vela actual de TF superior NO está completa hasta que cierre.
    # Debemos ver la vela ANTERIOR (ya cerrada).
    if shift_bars > 0:
        cols_to_shift = [c for c in resampled.columns if c != ts_col]
        resampled = resampled.with_columns([
            pl.col(c).shift(shift_bars).alias(c)
            for c in cols_to_shift
        ])
    
    return resampled


# ==============================================================================
# FUSIÓN AL DATAFRAME BASE
# ==============================================================================

def merge_timeframe_to_base(
    df_base: pl.DataFrame,
    df_resampled: pl.DataFrame,
    *,
    ts_col: str = "timestamp",
) -> pl.DataFrame:
    """
    Une datos resampleados al DataFrame base usando join_asof.
    
    Args:
        df_base: DataFrame de alta resolución (ej: 1m)
        df_resampled: DataFrame resampleado (ej: 1h)
        ts_col: Nombre de la columna de timestamp
    
    Returns:
        DataFrame base enriquecido con columnas del TF superior
    
    Estrategia "backward":
        Para cada fila en base, busca el timestamp <= en resampled.
        Esto asegura que nunca veamos datos del futuro.
    """
    if ts_col not in df_base.columns:
        ts_col = "datetime" if "datetime" in df_base.columns else "timestamp"
    
    # Ordenar ambos DataFrames por timestamp
    df_base_sorted = df_base.sort(ts_col)
    df_resampled_sorted = df_resampled.sort(ts_col)
    
    # Join asof con estrategia backward (nunca mira al futuro)
    result = df_base_sorted.join_asof(
        df_resampled_sorted,
        on=ts_col,
        strategy="backward",
    )
    
    return result


# ==============================================================================
# FUNCIÓN PRINCIPAL: PREPARE MULTITIMEFRAME DATA
# ==============================================================================

def prepare_multitimeframe_data(
    df_base: pl.DataFrame,
    required_timeframes: List[str],
    *,
    base_tf: Optional[str] = None,
    ts_col: str = "timestamp",
    anti_lookahead: bool = True,
) -> pl.DataFrame:
    """
    Prepara DataFrame con múltiples timeframes para estrategias MTF.
    
    Este es el punto de entrada principal del Data Blender.
    
    Args:
        df_base: DataFrame de máxima resolución ("Átomo", ej: 1m)
        required_timeframes: Lista de TFs adicionales (ej: ["1h", "4h"])
        base_tf: Timeframe del df_base (auto-detectado si None)
        ts_col: Nombre de la columna de timestamp
        anti_lookahead: Aplicar shift(1) para seguridad (default=True)
    
    Returns:
        DataFrame base enriquecido con columnas de TFs superiores:
        - close (base), close_1h, close_4h, etc.
        - high, low, open, volume también disponibles
    
    Ejemplo:
        >>> df = load_data("BTC", "1m")
        >>> df_mtf = prepare_multitimeframe_data(df, ["1h", "4h"])
        >>> # Ahora disponibles: close, close_1h, close_4h, etc.
    """
    if not required_timeframes:
        return df_base
    
    # Detectar timeframe base si no se especifica
    if base_tf is None:
        base_tf = _detect_base_timeframe(df_base)
    base_tf = normalize_timeframe_to_suffix(base_tf)
    base_minutes = _tf_to_minutes(base_tf)
    
    # Detectar columna de timestamp
    if ts_col not in df_base.columns:
        ts_col = "datetime" if "datetime" in df_base.columns else "timestamp"
    
    result = df_base
    
    for tf in required_timeframes:
        tf_suffix = normalize_timeframe_to_suffix(tf)
        tf_minutes = _tf_to_minutes(tf_suffix)
        
        # Solo generar TFs superiores al base
        if tf_minutes <= base_minutes:
            continue
        
        # Verificar si ya existe (evitar reprocesar)
        if f"close_{tf_suffix}" in result.columns:
            continue
        
        # Resamplear con protección anti-lookahead
        shift = 1 if anti_lookahead else 0
        df_resampled = resample_ohlcv(
            result,
            tf_suffix,
            ts_col=ts_col,
            shift_bars=shift,
        )
        
        # Fusionar al base
        result = merge_timeframe_to_base(
            result,
            df_resampled,
            ts_col=ts_col,
        )
    
    return result


# ==============================================================================
# UTILIDADES ADICIONALES
# ==============================================================================

def get_available_timeframes(df: pl.DataFrame) -> List[str]:
    """
    Lista los timeframes disponibles en un DataFrame MTF.
    
    Returns:
        Lista de sufijos (ej: ["1m", "1h", "4h"])
    """
    tfs = set()
    
    # El base siempre está presente (columnas sin sufijo)
    if "close" in df.columns:
        base = _detect_base_timeframe(df)
        tfs.add(base)
    
    # Buscar columnas con sufijo _<tf>
    for col in df.columns:
        if col.startswith("close_"):
            tf = col.replace("close_", "")
            if tf in _TF_TO_POLARS_DURATION:
                tfs.add(tf)
    
    # Ordenar por duración
    return sorted(tfs, key=_tf_to_minutes)


def validate_multitimeframe_data(
    df: pl.DataFrame,
    required_timeframes: List[str],
) -> Tuple[bool, List[str]]:
    """
    Valida que el DataFrame contenga todos los TFs requeridos.
    
    Returns:
        (is_valid, missing_tfs)
    """
    get_available_timeframes(df)
    missing = []
    
    for tf in required_timeframes:
        tf_suffix = normalize_timeframe_to_suffix(tf)
        # Verificar columna close_{tf}
        if f"close_{tf_suffix}" not in df.columns:
            missing.append(tf_suffix)
    
    return (len(missing) == 0, missing)
