"""
# =============================================================================
#
#     ██████╗  █████╗ ████████╗ █████╗    ██████╗ ██╗   ██╗
#     ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗   ██╔══██╗╚██╗ ██╔╝
#     ██║  ██║███████║   ██║   ███████║   ██████╔╝ ╚████╔╝
#     ██║  ██║██╔══██║   ██║   ██╔══██║   ██╔═══╝   ╚██╔╝
#     ██████╔╝██║  ██║   ██║   ██║  ██║   ██║        ██║
#     ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝        ╚═╝
#
#     DATA.PY - CARGA Y TRANSFORMACIÓN DE DATOS OHLCV
#
# =============================================================================
#
#     FUNCIONALIDADES:
#     - Carga de datos (Parquet, Feather, CSV)
#     - Normalización temporal (UTC, microsegundos)
#     - Multi-Timeframe con protección anti-lookahead
#     - Caché para evitar recálculos
#
#     SEGURIDAD ANTI-LOOKAHEAD:
#     Los datos resampleados usan shift(1) para NUNCA ver el futuro.
#     Ejemplo: En la vela 10:15 (1m), la vela 1h visible es 09:00-10:00,
#     NUNCA la de 10:00-11:00 (que contiene datos futuros).
#
# =============================================================================
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import polars as pl

from modelox.core.types import normalize_timeframe_to_suffix


# =============================================================================
# 1. CONSTANTES GLOBALES
# =============================================================================

MTF_CACHE_MAX_SIZE: int = 8

TF_TO_POLARS_DURATION: Dict[str, str] = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "2h": "2h", "3h": "3h", "4h": "4h",
    "6h": "6h", "8h": "8h", "12h": "12h", "1d": "1d",
}

OHLCV_RESAMPLE_EXPRS = {
    "open": pl.col("open").first(),
    "high": pl.col("high").max(),
    "low": pl.col("low").min(),
    "close": pl.col("close").last(),
    "volume": pl.col("volume").sum(),
}


# =============================================================================
# 2. CACHÉ DE DATOS MTF
# =============================================================================

_MTF_CACHE: Dict[str, pl.DataFrame] = {}


def _compute_df_hash(df: pl.DataFrame, timeframes: FrozenSet[str]) -> str:
    """GENERA HASH ÚNICO PARA DATAFRAME + TIMEFRAMES."""
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
    """OBTIENE DATAFRAME DESDE CACHÉ SI EXISTE."""
    return _MTF_CACHE.get(cache_key)


def _set_cached_mtf(cache_key: str, df: pl.DataFrame) -> None:
    """GUARDA DATAFRAME EN CACHÉ CON LÍMITE DE TAMAÑO."""
    global _MTF_CACHE
    if len(_MTF_CACHE) >= MTF_CACHE_MAX_SIZE:
        oldest_key = next(iter(_MTF_CACHE))
        del _MTF_CACHE[oldest_key]
    _MTF_CACHE[cache_key] = df


def clear_mtf_cache() -> None:
    """LIMPIA TODO EL CACHÉ MTF."""
    global _MTF_CACHE
    _MTF_CACHE.clear()


# =============================================================================
# 3. CARGA DE DATOS
# =============================================================================

def load_data(path: str) -> pl.DataFrame:
    """
    CARGA DATOS OHLCV DESDE ARCHIVO
    
    Detecta formato automáticamente (Parquet, Feather, CSV).
    Normaliza timestamp a UTC con precisión microsegundos.
    
    Args:
        path: Ruta al archivo de datos
    
    Returns:
        DataFrame Polars con OHLCV normalizado
    """
    p = Path(path)
    ext = p.suffix.lower()
    
    # Si no existe, buscar en otros formatos
    if not p.exists():
        candidates = [
            p, p.with_suffix(".feather"), p.with_suffix(".fthr"),
            p.with_suffix(".csv"), p.with_suffix(".parquet"), p.with_suffix(".pq"),
        ]
        for c in candidates:
            if c.exists():
                p = c
                ext = p.suffix.lower()
                break
        else:
            parent = p.parent if p.parent.exists() else Path(".")
            available = sorted(parent.glob(f"{p.stem}.*"))
            hint = f". Encontrados: {[str(a) for a in available]}" if available else ""
            raise FileNotFoundError(f"No existe: {path}{hint}")
    
    # Cargar según formato
    if ext in {".parquet", ".pq"}:
        q = pl.scan_parquet(str(p))
    elif ext in {".feather", ".fthr"}:
        q = pl.scan_ipc(str(p))
    elif ext in {".csv"}:
        q = pl.scan_csv(str(p))
    else:
        raise ValueError(f"Formato {ext} no soportado. Usa Parquet/Feather/CSV.")
    
    return _normalize_pl(q).collect()


def _normalize_pl(q: pl.LazyFrame) -> pl.LazyFrame:
    """NORMALIZA COLUMNA TEMPORAL A UTC CON PRECISIÓN MICROSEGUNDOS."""
    schema = q.collect_schema()
    
    col_time = next(
        (c for c in ["timestamp", "datetime", "date", "time"] if c in schema), None
    )
    if col_time is None:
        raise ValueError(f"Falta columna temporal. Detectadas: {list(schema.keys())}")
    
    if col_time != "timestamp":
        q = q.rename({col_time: "timestamp"})
    
    dtype = schema.get("timestamp", schema.get(col_time))
    tz = getattr(dtype, "time_zone", None) if dtype else None
    
    ts_expr = pl.col("timestamp").dt.cast_time_unit("us")
    if tz is None:
        ts_expr = ts_expr.dt.replace_time_zone("UTC")
    elif tz != "UTC":
        ts_expr = ts_expr.dt.convert_time_zone("UTC")
    
    return q.with_columns(ts_expr).sort("timestamp")


# =============================================================================
# 4. UTILIDADES DE TIMEFRAME
# =============================================================================

def _get_polars_duration(tf_suffix: str) -> str:
    """CONVIERTE SUFIJO A DURACIÓN POLARS."""
    return TF_TO_POLARS_DURATION.get(tf_suffix, tf_suffix)


def _tf_to_minutes(tf_suffix: str) -> int:
    """CONVIERTE SUFIJO A MINUTOS: "5m"->5, "1h"->60, "1d"->1440."""
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
    """DETECTA TIMEFRAME BASE ANALIZANDO INTERVALOS ENTRE VELAS."""
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
        
        if median_diff <= 1: return "1m"
        elif median_diff <= 5: return "5m"
        elif median_diff <= 15: return "15m"
        elif median_diff <= 30: return "30m"
        elif median_diff <= 60: return "1h"
        elif median_diff <= 120: return "2h"
        elif median_diff <= 240: return "4h"
        else: return "1d"
    except Exception:
        return "1m"


# =============================================================================
# 5. RESAMPLING CON PROTECCIÓN ANTI-LOOKAHEAD
# =============================================================================

def resample_ohlcv(
    df: pl.DataFrame,
    target_tf: str,
    *,
    ts_col: str = "timestamp",
    shift_bars: int = 1,
) -> pl.DataFrame:
    """
    RESAMPLEA OHLCV A TIMEFRAME SUPERIOR
    
    IMPORTANTE - SEGURIDAD ANTI-LOOKAHEAD:
    Aplica shift(1) para que en la vela actual solo veas
    la vela CERRADA del timeframe superior, nunca la actual.
    
    Args:
        df: DataFrame con OHLCV
        target_tf: Timeframe objetivo (ej: "1h", "4h")
        ts_col: Nombre columna timestamp
        shift_bars: Barras a desplazar (1 = anti-lookahead)
    
    Returns:
        DataFrame resampleado con columnas sufijadas
    """
    if ts_col not in df.columns:
        ts_col = "datetime" if "datetime" in df.columns else "timestamp"
    
    tf_suffix = normalize_timeframe_to_suffix(target_tf)
    duration = _get_polars_duration(tf_suffix)
    
    df_work = df.lazy()
    if df.schema.get(ts_col) != pl.Datetime:
        df_work = df_work.with_columns(pl.col(ts_col).cast(pl.Datetime("us")))
    
    resampled = (
        df_work.sort(ts_col)
        .group_by_dynamic(ts_col, every=duration)
        .agg([
            OHLCV_RESAMPLE_EXPRS["open"].alias(f"open_{tf_suffix}"),
            OHLCV_RESAMPLE_EXPRS["high"].alias(f"high_{tf_suffix}"),
            OHLCV_RESAMPLE_EXPRS["low"].alias(f"low_{tf_suffix}"),
            OHLCV_RESAMPLE_EXPRS["close"].alias(f"close_{tf_suffix}"),
            OHLCV_RESAMPLE_EXPRS["volume"].alias(f"volume_{tf_suffix}"),
        ])
        .collect()
    )
    
    # SHIFT PARA ANTI-LOOKAHEAD
    if shift_bars > 0:
        cols_to_shift = [c for c in resampled.columns if c != ts_col]
        resampled = resampled.with_columns([
            pl.col(c).shift(shift_bars).alias(c) for c in cols_to_shift
        ])
    
    return resampled


def merge_timeframe_to_base(
    df_base: pl.DataFrame,
    df_resampled: pl.DataFrame,
    *,
    ts_col: str = "timestamp",
) -> pl.DataFrame:
    """UNE DATOS RESAMPLEADOS AL BASE CON JOIN_ASOF (BACKWARD)."""
    if ts_col not in df_base.columns:
        ts_col = "datetime" if "datetime" in df_base.columns else "timestamp"
    
    return df_base.sort(ts_col).join_asof(
        df_resampled.sort(ts_col),
        on=ts_col,
        strategy="backward",
    )


# =============================================================================
# 6. FUNCIÓN PRINCIPAL: PREPARAR DATOS MULTI-TIMEFRAME
# =============================================================================

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
    PREPARA DATAFRAME CON MÚLTIPLES TIMEFRAMES
    
    Enriquece el DataFrame base con columnas de timeframes superiores.
    Usa caché para evitar recálculos en trials consecutivos.
    
    Args:
        df_base: DataFrame de máxima resolución (ej: 1m)
        required_timeframes: Lista de TFs adicionales (ej: ["1h", "4h"])
        base_tf: Timeframe del df_base (auto-detectado si None)
        anti_lookahead: Aplicar shift(1) para seguridad
        use_cache: Usar caché para optimizar
    
    Returns:
        DataFrame con columnas: close, close_1h, close_4h, etc.
    """
    if not required_timeframes:
        return df_base
    
    # VERIFICAR CACHÉ
    tf_set = frozenset(required_timeframes)
    cache_key = _compute_df_hash(df_base, tf_set) if use_cache else ""
    
    if use_cache:
        cached = _get_cached_mtf(cache_key)
        if cached is not None:
            return cached
    
    # DETECTAR TIMEFRAME BASE
    if base_tf is None:
        base_tf = _detect_base_timeframe(df_base)
    base_tf = normalize_timeframe_to_suffix(base_tf)
    base_minutes = _tf_to_minutes(base_tf)
    
    if ts_col not in df_base.columns:
        ts_col = "datetime" if "datetime" in df_base.columns else "timestamp"
    
    result = df_base
    
    # GENERAR CADA TIMEFRAME SUPERIOR
    for tf in required_timeframes:
        tf_suffix = normalize_timeframe_to_suffix(tf)
        tf_minutes = _tf_to_minutes(tf_suffix)
        
        if tf_minutes <= base_minutes:
            continue
        if f"close_{tf_suffix}" in result.columns:
            continue
        
        shift = 1 if anti_lookahead else 0
        df_resampled = resample_ohlcv(result, tf_suffix, ts_col=ts_col, shift_bars=shift)
        result = merge_timeframe_to_base(result, df_resampled, ts_col=ts_col)
    
    # GUARDAR EN CACHÉ
    if use_cache:
        _set_cached_mtf(cache_key, result)
    
    return result


# =============================================================================
# 7. UTILIDADES DE VALIDACIÓN
# =============================================================================

def get_available_timeframes(df: pl.DataFrame) -> List[str]:
    """LISTA TIMEFRAMES DISPONIBLES EN UN DATAFRAME MTF."""
    tfs = set()
    
    if "close" in df.columns:
        base = _detect_base_timeframe(df)
        tfs.add(base)
    
    for col in df.columns:
        if col.startswith("close_"):
            tf = col.replace("close_", "")
            if tf in TF_TO_POLARS_DURATION:
                tfs.add(tf)
    
    return sorted(tfs, key=_tf_to_minutes)


def validate_multitimeframe_data(
    df: pl.DataFrame,
    required_timeframes: List[str],
) -> Tuple[bool, List[str]]:
    """VALIDA QUE EL DATAFRAME CONTENGA TODOS LOS TIMEFRAMES REQUERIDOS."""
    missing = []
    for tf in required_timeframes:
        tf_suffix = normalize_timeframe_to_suffix(tf)
        if f"close_{tf_suffix}" not in df.columns:
            missing.append(tf_suffix)
    return (len(missing) == 0, missing)
