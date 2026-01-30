"""
================================================================================
DATA MODULE - Carga y Transformación de Datos OHLCV
================================================================================

Este módulo unifica:
1. Carga de datos (Parquet, Feather, CSV)
2. Normalización temporal (UTC, microsegundos)
3. Multi-Timeframe Blending (resampleo con anti-lookahead)
4. Caché de datos MTF para evitar recálculos

SEGURIDAD ANTI-LOOKAHEAD:
  Los datos resampleados usan .shift(1) para evitar mirar al futuro.
  Ejemplo: En la vela de las 10:15 (1m), la vela de 1h visible es la de 09:00-10:00,
  NO la de 10:00-11:00 (que contiene información del futuro).

================================================================================
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import polars as pl

from modelox.core.types import normalize_timeframe_to_suffix


# ==============================================================================
# CACHÉ DE DATOS MTF
# ==============================================================================

# Caché global para DataFrames MTF (key: hash de datos + timeframes)
_MTF_CACHE: Dict[str, pl.DataFrame] = {}
_MTF_CACHE_MAX_SIZE = 8  # Máximo de entradas en caché


def _compute_df_hash(df: pl.DataFrame, timeframes: FrozenSet[str]) -> str:
    """Calcula hash único para un DataFrame y sus timeframes requeridos."""
    # Usar shape + primeras/últimas filas + timeframes para hash rápido
    n_rows = df.height
    if n_rows > 0:
        first_ts = str(df["timestamp"][0]) if "timestamp" in df.columns else ""
        last_ts = str(df["timestamp"][-1]) if "timestamp" in df.columns else ""
        first_close = str(df["close"][0]) if "close" in df.columns else ""
        last_close = str(df["close"][-1]) if "close" in df.columns else ""
    else:
        first_ts = last_ts = first_close = last_close = ""
    
    key_str = f"{n_rows}|{first_ts}|{last_ts}|{first_close}|{last_close}|{sorted(timeframes)}"
    return hashlib.md5(key_str.encode()).hexdigest()[:16]


def _get_cached_mtf(cache_key: str) -> Optional[pl.DataFrame]:
    """Obtiene DataFrame MTF desde caché si existe."""
    return _MTF_CACHE.get(cache_key)


def _set_cached_mtf(cache_key: str, df: pl.DataFrame) -> None:
    """Guarda DataFrame MTF en caché con límite de tamaño."""
    global _MTF_CACHE
    
    # Si caché está llena, eliminar la entrada más antigua
    if len(_MTF_CACHE) >= _MTF_CACHE_MAX_SIZE:
        oldest_key = next(iter(_MTF_CACHE))
        del _MTF_CACHE[oldest_key]
    
    _MTF_CACHE[cache_key] = df


def clear_mtf_cache() -> None:
    """Limpia el caché de datos MTF."""
    global _MTF_CACHE
    _MTF_CACHE.clear()


# ==============================================================================
# CARGA DE DATOS
# ==============================================================================

def load_data(path: str) -> pl.DataFrame:
    """
    Carga datos OHLCV detectando, renombrando y normalizando la precisión temporal.
    Resuelve el conflicto de tipos 'ns' vs 'us' detectado en el log.
    """
    p = Path(path)
    ext = p.suffix.lower()

    # Si el archivo no existe, intenta con el mismo stem en formatos comunes.
    if not p.exists():
        candidates = [
            p,
            p.with_suffix(".feather"),
            p.with_suffix(".fthr"),
            p.with_suffix(".csv"),
            p.with_suffix(".parquet"),
            p.with_suffix(".pq"),
        ]
        for c in candidates:
            if c.exists():
                p = c
                ext = p.suffix.lower()
                break
        else:
            parent = p.parent if p.parent.exists() else Path(".")
            available = sorted(parent.glob(f"{p.stem}.*"))
            hint = ""
            if available:
                hint = f". Encontrados: {[str(a) for a in available]}"
            raise FileNotFoundError(f"No such file: {path}{hint}")

    if ext in {".parquet", ".pq"}:
        q = pl.scan_parquet(str(p))
    elif ext in {".feather", ".fthr"}:
        q = pl.scan_ipc(str(p))
    elif ext in {".csv"}:
        q = pl.scan_csv(str(p))
    else:
        raise ValueError(f"Formato {ext} no soportado. Usa Parquet/Feather/CSV.")

    # Normaliza y materializa los datos en memoria
    return _normalize_pl(q).collect()


def _normalize_pl(q: pl.LazyFrame) -> pl.LazyFrame:
    """
    Normaliza nombres y fuerza precisión de microsegundos para evitar InvalidOperationError.
    """
    schema = q.collect_schema()

    # Busca la columna temporal
    col_time = next(
        (c for c in ["timestamp", "datetime", "date", "time"] if c in schema), None
    )

    if col_time is None:
        raise ValueError(f"Falta columna temporal. Detectadas: {list(schema.keys())}")

    # Estandariza el nombre a 'timestamp'
    if col_time != "timestamp":
        q = q.rename({col_time: "timestamp"})

    # NORMALIZACIÓN DE PRECISIÓN Y ZONA HORARIA (UTC)
    dtype = schema.get("timestamp", schema.get(col_time))
    tz = None
    try:
        tz = getattr(dtype, "time_zone", None)
    except Exception:
        tz = None

    ts_expr = pl.col("timestamp").dt.cast_time_unit("us")
    if tz is None:
        ts_expr = ts_expr.dt.replace_time_zone("UTC")
    elif tz != "UTC":
        ts_expr = ts_expr.dt.convert_time_zone("UTC")

    return q.with_columns(ts_expr).sort("timestamp")


# ==============================================================================
# CONSTANTES MULTI-TIMEFRAME
# ==============================================================================

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

_OHLCV_RESAMPLE_EXPRS = {
    "open": pl.col("open").first(),
    "high": pl.col("high").max(),
    "low": pl.col("low").min(),
    "close": pl.col("close").last(),
    "volume": pl.col("volume").sum(),
}


# ==============================================================================
# FUNCIONES AUXILIARES TIMEFRAME
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
        return "1m"

    try:
        sample = df.head(100)
        if sample.height < 2:
            return "1m"

        diffs = sample.select(
            pl.col(ts_col).diff().drop_nulls().dt.total_minutes()
        ).to_series()

        if diffs.is_empty():
            return "1m"

        median_diff = int(diffs.median())

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

    CRÍTICO - Seguridad Anti-Lookahead:
        Si estoy en la vela de las 10:15 (base 1m), la vela de 1h (10:00-11:00)
        contiene información del futuro (cierre de las 11:00).
        Aplicamos .shift(1) para ver la vela anterior (09:00-10:00).
    """
    if ts_col not in df.columns:
        ts_col = "datetime" if "datetime" in df.columns else "timestamp"

    tf_suffix = normalize_timeframe_to_suffix(target_tf)
    duration = _get_polars_duration(tf_suffix)

    df_work = df.lazy()
    if df.schema.get(ts_col) != pl.Datetime:
        df_work = df_work.with_columns(pl.col(ts_col).cast(pl.Datetime("us")))

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
    Estrategia "backward": nunca ve datos del futuro.
    """
    if ts_col not in df_base.columns:
        ts_col = "datetime" if "datetime" in df_base.columns else "timestamp"

    df_base_sorted = df_base.sort(ts_col)
    df_resampled_sorted = df_resampled.sort(ts_col)

    return df_base_sorted.join_asof(
        df_resampled_sorted,
        on=ts_col,
        strategy="backward",
    )


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
    use_cache: bool = True,
) -> pl.DataFrame:
    """
    Prepara DataFrame con múltiples timeframes para estrategias MTF.

    Args:
        df_base: DataFrame de máxima resolución ("Átomo", ej: 1m)
        required_timeframes: Lista de TFs adicionales (ej: ["1h", "4h"])
        base_tf: Timeframe del df_base (auto-detectado si None)
        ts_col: Nombre de la columna de timestamp
        anti_lookahead: Aplicar shift(1) para seguridad (default=True)
        use_cache: Usar caché para evitar recálculos (default=True)

    Returns:
        DataFrame base enriquecido con columnas de TFs superiores:
        - close (base), close_1h, close_4h, etc.
    """
    if not required_timeframes:
        return df_base

    # =========================================================================
    # CACHÉ: Verificar si ya tenemos este resultado cacheado
    # =========================================================================
    tf_set = frozenset(required_timeframes)
    cache_key = _compute_df_hash(df_base, tf_set) if use_cache else ""
    
    if use_cache:
        cached = _get_cached_mtf(cache_key)
        if cached is not None:
            return cached

    if base_tf is None:
        base_tf = _detect_base_timeframe(df_base)
    base_tf = normalize_timeframe_to_suffix(base_tf)
    base_minutes = _tf_to_minutes(base_tf)

    if ts_col not in df_base.columns:
        ts_col = "datetime" if "datetime" in df_base.columns else "timestamp"

    result = df_base

    for tf in required_timeframes:
        tf_suffix = normalize_timeframe_to_suffix(tf)
        tf_minutes = _tf_to_minutes(tf_suffix)

        # Solo generar TFs superiores al base
        if tf_minutes <= base_minutes:
            continue

        # Verificar si ya existe
        if f"close_{tf_suffix}" in result.columns:
            continue

        shift = 1 if anti_lookahead else 0
        df_resampled = resample_ohlcv(
            result,
            tf_suffix,
            ts_col=ts_col,
            shift_bars=shift,
        )

        result = merge_timeframe_to_base(
            result,
            df_resampled,
            ts_col=ts_col,
        )

    # =========================================================================
    # CACHÉ: Guardar resultado para futuros trials
    # =========================================================================
    if use_cache:
        _set_cached_mtf(cache_key, result)

    return result


# ==============================================================================
# UTILIDADES
# ==============================================================================

def get_available_timeframes(df: pl.DataFrame) -> List[str]:
    """Lista los timeframes disponibles en un DataFrame MTF."""
    tfs = set()

    if "close" in df.columns:
        base = _detect_base_timeframe(df)
        tfs.add(base)

    for col in df.columns:
        if col.startswith("close_"):
            tf = col.replace("close_", "")
            if tf in _TF_TO_POLARS_DURATION:
                tfs.add(tf)

    return sorted(tfs, key=_tf_to_minutes)


def validate_multitimeframe_data(
    df: pl.DataFrame,
    required_timeframes: List[str],
) -> Tuple[bool, List[str]]:
    """Valida que el DataFrame contenga todos los TFs requeridos."""
    missing = []

    for tf in required_timeframes:
        tf_suffix = normalize_timeframe_to_suffix(tf)
        if f"close_{tf_suffix}" not in df.columns:
            missing.append(tf_suffix)

    return (len(missing) == 0, missing)
