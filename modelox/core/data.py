"""modelox.core.data — Carga y transformación de datos OHLCV.

Flujo: load_data() → resample_to_base_timeframe() → prepare_multitimeframe_data().
MTF auxiliares usan shift(1) anti-lookahead. Resampleo base sin shift.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import polars as pl

from modelox.core.types import normalize_timeframe_to_suffix


# =============================================================================
# 1. CACHÉ DE DATOS MTF
# =============================================================================
# Caché global para DataFrames MTF procesados.
# Evita recalcular resampleo entre trials con los mismos datos.

_MTF_CACHE: Dict[str, pl.DataFrame] = {}
_MTF_CACHE_MAX_SIZE = 8


def _compute_df_hash(df: pl.DataFrame, timeframes: FrozenSet[str]) -> str:
    """
    CALCULA HASH ÚNICO PARA UN DATAFRAME Y SUS TIMEFRAMES REQUERIDOS.

    Usa shape + timestamps extremos + precios extremos + TFs para generar
    un hash rápido sin necesidad de hashear todo el DataFrame.
    """
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
    """OBTIENE DATAFRAME MTF DESDE CACHÉ SI EXISTE."""
    return _MTF_CACHE.get(cache_key)


def _set_cached_mtf(cache_key: str, df: pl.DataFrame) -> None:
    """GUARDA DATAFRAME MTF EN CACHÉ CON LÍMITE DE TAMAÑO."""
    global _MTF_CACHE
    if len(_MTF_CACHE) >= _MTF_CACHE_MAX_SIZE:
        oldest_key = next(iter(_MTF_CACHE))
        del _MTF_CACHE[oldest_key]
    _MTF_CACHE[cache_key] = df


def clear_mtf_cache() -> None:
    """LIMPIA EL CACHÉ DE DATOS MTF."""
    global _MTF_CACHE
    _MTF_CACHE.clear()


# =============================================================================
# 2. CARGA DE DATOS
# =============================================================================

def load_data(path: str) -> pl.DataFrame:
    """
    CARGA DATOS OHLCV DESDE DISCO (PARQUET, FEATHER O CSV).

    Detecta formato por extensión, normaliza nombres de columnas,
    fuerza precisión de microsegundos y zona horaria UTC.

    Búsqueda inteligente: si el archivo no existe, prueba extensiones
    alternativas (.feather, .fthr, .csv, .parquet, .pq).

    Args:
        path: Ruta al archivo de datos.

    Returns:
        DataFrame Polars con columnas [timestamp, open, high, low, close, volume].

    Raises:
        FileNotFoundError: Si no se encuentra el archivo.
        ValueError: Si el formato no es soportado o falta columna temporal.
    """
    p = Path(path)
    ext = p.suffix.lower()

    # ─── BÚSQUEDA INTELIGENTE DE ARCHIVO ─────────────────────────────────
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

    # ─── LECTURA SEGÚN FORMATO ───────────────────────────────────────────
    if ext in {".parquet", ".pq"}:
        q = pl.scan_parquet(str(p))
    elif ext in {".feather", ".fthr"}:
        q = pl.scan_ipc(str(p), memory_map=False)
    elif ext in {".csv"}:
        q = pl.scan_csv(str(p))
    else:
        raise ValueError(f"Formato {ext} no soportado. Usa Parquet/Feather/CSV.")

    return _normalize_pl(q).collect()


# =============================================================================
# 3. NORMALIZACIÓN TEMPORAL
# =============================================================================

def _normalize_pl(q: pl.LazyFrame) -> pl.LazyFrame:
    """
    NORMALIZA NOMBRES Y FUERZA PRECISIÓN DE MICROSEGUNDOS UTC.

    Pasos:
        1. Detecta la columna temporal (timestamp, datetime, date, time).
        2. Renombra a "timestamp" si es necesario.
        3. Fuerza precisión a microsegundos ("us").
        4. Convierte zona horaria a UTC.
        5. Ordena por timestamp ascendente.

    Raises:
        ValueError: Si no se encuentra columna temporal.
    """
    # En Polars >=0.20, LazyFrame expone el esquema vía la propiedad `.schema`
    # (collect_schema fue removido). Usamos la propiedad para compatibilidad.
    schema = q.schema

    # ─── DETECTAR COLUMNA TEMPORAL ───────────────────────────────────────
    col_time = next(
        (c for c in ["timestamp", "datetime", "date", "time"] if c in schema), None
    )
    if col_time is None:
        raise ValueError(f"Falta columna temporal. Detectadas: {list(schema.keys())}")

    # ─── ESTANDARIZAR NOMBRE ─────────────────────────────────────────────
    if col_time != "timestamp":
        q = q.rename({col_time: "timestamp"})

    # ─── NORMALIZAR PRECISIÓN Y ZONA HORARIA ─────────────────────────────
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


# =============================================================================
# 4. CONSTANTES MULTI-TIMEFRAME
# =============================================================================

# ─── MAPEO DE SUFIJO TF → DURACIÓN POLARS ───────────────────────────────────
_TF_TO_POLARS_DURATION: Dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "12h": "12h",
    "1d": "1d",
}

# ─── EXPRESIONES DE RESAMPLEO OHLCV ─────────────────────────────────────────
_OHLCV_RESAMPLE_EXPRS = {
    "open": pl.col("open").first(),
    "high": pl.col("high").max(),
    "low": pl.col("low").min(),
    "close": pl.col("close").last(),
    "volume": pl.col("volume").sum(),
}


# =============================================================================
# 5. FUNCIONES AUXILIARES DE TIMEFRAME
# =============================================================================

def _get_polars_duration(tf_suffix: str) -> str:
    """CONVIERTE SUFIJO DE TIMEFRAME A DURACIÓN POLARS."""
    return _TF_TO_POLARS_DURATION.get(tf_suffix, tf_suffix)


def _tf_to_minutes(tf_suffix: str) -> int:
    """CONVIERTE SUFIJO DE TIMEFRAME A MINUTOS."""
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
    """
    DETECTA EL TIMEFRAME BASE DEL DATAFRAME ANALIZANDO INTERVALOS.

    Toma las primeras 100 filas, calcula la mediana de las diferencias
    entre timestamps consecutivos, y mapea a un TF estándar.

    Returns:
        Sufijo del TF detectado ("1m", "5m", "15m", "1h", etc.).
    """
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


# =============================================================================
# 6. RESAMPLEO AL TIMEFRAME BASE (1m → TF ELEGIDO POR EL USUARIO)
# =============================================================================

def resample_to_base_timeframe(
    df_1m: pl.DataFrame,
    target_tf: str,
    *,
    ts_col: str = "timestamp",
) -> pl.DataFrame:
    """
    RESAMPLEA DATOS DE 1m AL TIMEFRAME BASE ELEGIDO POR EL USUARIO.

    A diferencia de resample_ohlcv() (usado para MTF), esta función:
        - NO aplica shift/anti-lookahead (es el TF de operación, no auxiliar).
        - Mantiene nombres OHLCV sin sufijo (open, high, low, close, volume).
        - Si target_tf == "1m", devuelve el DataFrame tal cual (sin resampleo).

    Args:
        df_1m:     DataFrame con datos OHLCV en resolución 1m.
        target_tf: Timeframe destino ("1m", "5m", "15m", "1h", etc.).
        ts_col:    Nombre de la columna temporal.

    Returns:
        DataFrame resampleado al timeframe elegido.

    Ejemplos:
        >>> df_5m = resample_to_base_timeframe(df_1m, "5m")
        >>> df_1h = resample_to_base_timeframe(df_1m, "1h")
        >>> df_1m = resample_to_base_timeframe(df_1m, "1m")  # Sin cambios
    """
    tf_suffix = normalize_timeframe_to_suffix(target_tf)
    target_minutes = _tf_to_minutes(tf_suffix)

    # DETECTAR TF DE LOS DATOS DE ENTRADA
    base_minutes = _tf_to_minutes(_detect_base_timeframe(df_1m))

    # SI YA ESTÁ EN EL TF DESEADO (O SUPERIOR), DEVOLVER TAL CUAL
    if target_minutes <= base_minutes:
        return df_1m

    if ts_col not in df_1m.columns:
        ts_col = "datetime" if "datetime" in df_1m.columns else "timestamp"

    duration = _get_polars_duration(tf_suffix)

    df_work = df_1m.lazy()
    if df_1m.schema.get(ts_col) != pl.Datetime:
        df_work = df_work.with_columns(pl.col(ts_col).cast(pl.Datetime("us")))

    # RESAMPLEAR OHLCV CON NOMBRES SIN SUFIJO (ES EL BASE)
    resampled = (
        df_work
        .sort(ts_col)
        .group_by_dynamic(ts_col, every=duration)
        .agg([
            _OHLCV_RESAMPLE_EXPRS["open"],
            _OHLCV_RESAMPLE_EXPRS["high"],
            _OHLCV_RESAMPLE_EXPRS["low"],
            _OHLCV_RESAMPLE_EXPRS["close"],
            _OHLCV_RESAMPLE_EXPRS["volume"],
        ])
        .collect()
    )

    return resampled


def candles_per_month_for_tf(tf: str) -> int:
    """
    CALCULA CUÁNTAS VELAS POR MES TIENE UN TIMEFRAME DADO.

    Asume 30 días × 24 horas = 43.200 minutos por mes.

    Args:
        tf: Timeframe como string ("1m", "5m", "1h", etc.).

    Returns:
        Número aproximado de velas por mes.
    """
    minutes = _tf_to_minutes(normalize_timeframe_to_suffix(tf))
    if minutes <= 0:
        minutes = 1
    total_minutes_per_month = 30 * 24 * 60  # 43200
    return total_minutes_per_month // minutes


# =============================================================================
# 7. RESAMPLEO MTF CON SEGURIDAD ANTI-LOOKAHEAD
# =============================================================================

def resample_ohlcv(
    df: pl.DataFrame,
    target_tf: str,
    *,
    ts_col: str = "timestamp",
    shift_bars: int = 1,
) -> pl.DataFrame:
    """
    RESAMPLEA OHLCV A UN TIMEFRAME SUPERIOR CON PROTECCIÓN ANTI-LOOKAHEAD.

    CRÍTICO — Seguridad Anti-Lookahead:
        Si estoy en la vela de las 10:15 (base 1m), la vela de 1h (10:00-11:00)
        contiene información del futuro (cierre de las 11:00).
        Aplicamos .shift(1) para ver la vela anterior (09:00-10:00) ya CERRADA.

    Las columnas se nombran con sufijo del TF: open_1h, close_4h, etc.

    Args:
        df:         DataFrame base.
        target_tf:  Timeframe destino.
        ts_col:     Columna temporal.
        shift_bars: Número de barras a desplazar (1 para anti-lookahead).

    Returns:
        DataFrame resampleado con columnas nombradas {col}_{tf_suffix}.
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

    # CRÍTICO: SHIFT PARA ANTI-LOOKAHEAD
    if shift_bars > 0:
        cols_to_shift = [c for c in resampled.columns if c != ts_col]
        resampled = resampled.with_columns([
            pl.col(c).shift(shift_bars).alias(c)
            for c in cols_to_shift
        ])

    return resampled


# =============================================================================
# 8. FUSIÓN AL DATAFRAME BASE
# =============================================================================

def merge_timeframe_to_base(
    df_base: pl.DataFrame,
    df_resampled: pl.DataFrame,
    *,
    ts_col: str = "timestamp",
) -> pl.DataFrame:
    """
    UNE DATOS RESAMPLEADOS AL DATAFRAME BASE USANDO JOIN_ASOF.

    Estrategia "backward": cada fila del base recibe el último valor
    del TF superior cuyo timestamp <= timestamp del base.
    Esto NUNCA ve datos del futuro.

    Args:
        df_base:       DataFrame base (TF de operación).
        df_resampled:  DataFrame resampleado (TF superior).
        ts_col:        Columna temporal.

    Returns:
        DataFrame base enriquecido con columnas del TF superior.
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


# =============================================================================
# 9. FUNCIÓN PRINCIPAL: PREPARE MULTITIMEFRAME DATA
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
    PREPARA DATAFRAME CON MÚLTIPLES TIMEFRAMES PARA ESTRATEGIAS MTF.

    Para cada TF en required_timeframes:
        1. Resamplea OHLCV al TF superior (con anti-lookahead si activado).
        2. Fusiona al DataFrame base con join_asof backward.
        3. Cachea el resultado para evitar recálculos entre trials.

    Args:
        df_base:               DataFrame de resolución base.
        required_timeframes:   Lista de TFs adicionales (ej: ["1h", "4h"]).
        base_tf:               TF del df_base (auto-detectado si None).
        ts_col:                Columna temporal.
        anti_lookahead:        Aplicar shift(1) para seguridad (default=True).
        use_cache:             Usar caché de resultados (default=True).

    Returns:
        DataFrame base enriquecido con columnas de TFs superiores:
        close (base), close_1h, close_4h, etc.
    """
    if not required_timeframes:
        return df_base

    # ─── VERIFICAR CACHÉ ─────────────────────────────────────────────────
    tf_set = frozenset(required_timeframes)
    cache_key = _compute_df_hash(df_base, tf_set) if use_cache else ""

    if use_cache:
        cached = _get_cached_mtf(cache_key)
        if cached is not None:
            return cached

    # ─── DETECTAR TF BASE ────────────────────────────────────────────────
    if base_tf is None:
        base_tf = _detect_base_timeframe(df_base)
    base_tf = normalize_timeframe_to_suffix(base_tf)
    base_minutes = _tf_to_minutes(base_tf)

    if ts_col not in df_base.columns:
        ts_col = "datetime" if "datetime" in df_base.columns else "timestamp"

    # ─── RESAMPLEAR Y FUSIONAR CADA TF ───────────────────────────────────
    result = df_base

    for tf in required_timeframes:
        tf_suffix = normalize_timeframe_to_suffix(tf)
        tf_minutes = _tf_to_minutes(tf_suffix)

        # SOLO GENERAR TFs SUPERIORES AL BASE
        if tf_minutes <= base_minutes:
            continue

        # VERIFICAR SI YA EXISTE
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

    # ─── GUARDAR EN CACHÉ ────────────────────────────────────────────────
    if use_cache:
        _set_cached_mtf(cache_key, result)

    return result


# =============================================================================
# 10. UTILIDADES
# =============================================================================

def get_available_timeframes(df: pl.DataFrame) -> List[str]:
    """
    LISTA LOS TIMEFRAMES DISPONIBLES EN UN DATAFRAME MTF.

    Detecta el TF base + todos los TFs presentes como columnas close_{tf}.

    Returns:
        Lista ordenada de sufijos TF (ej: ["1m", "1h", "4h"]).
    """
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
