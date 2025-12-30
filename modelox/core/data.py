from __future__ import annotations

from pathlib import Path

import polars as pl


def load_data(path: str) -> pl.DataFrame:
    """
    Carga datos OHLCV detectando, renombrando y normalizando la precisión temporal.
    Resuelve el conflicto de tipos 'ns' vs 'us' detectado en el log.
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext in {".parquet", ".pq"}:
        # Inicia escaneo Lazy
        q = pl.scan_parquet(path)
    elif ext in {".feather", ".fthr"}:
        # Feather/Arrow IPC via lazy scan
        q = pl.scan_ipc(path)
    else:
        raise ValueError(f"Formato {ext} no soportado. Usa Parquet.")

    # Normaliza y materializa los datos en memoria
    return _normalize_pl(q).collect()


def _normalize_pl(q: pl.LazyFrame) -> pl.LazyFrame:
    """
    Normaliza nombres y fuerza precisión de microsegundos para evitar InvalidOperationError.
    """
    # collect_schema() evita el PerformanceWarning
    schema = q.collect_schema()

    # Busca la columna temporal independientemente del nombre
    col_time = next(
        (c for c in ["timestamp", "datetime", "date", "time"] if c in schema), None
    )

    if col_time is None:
        raise ValueError(f"Falta columna temporal. Detectadas: {list(schema.keys())}")

    # Estandariza el nombre a 'timestamp' para los indicadores
    if col_time != "timestamp":
        q = q.rename({col_time: "timestamp"})

    # NORMALIZACIÓN DE PRECISIÓN Y ZONA HORARIA (UTC)
    # - Forzamos microsegundos para compatibilidad
    # - Si no tiene tz, la fijamos a UTC
    # - Si tiene tz distinta a UTC, convertimos a UTC
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
