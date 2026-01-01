"""Genera datasets 5m/15m/1h para cada activo.

Objetivo (simple y explícito):
- Usar como fuente el 5m REAL (solo archivos "finales": .parquet / .feather / .csv).
- Escribir 3 formatos por timeframe: parquet, feather (ipc), csv.
- Generar 15m (y opcionalmente 1h) por resample OHLCV.

Convención de archivos (en data/ohlcv):
  <ACTIVO>_ohlcv_5m.{parquet|feather|csv}
  <ACTIVO>_ohlcv_15m.{parquet|feather|csv}
  <ACTIVO>_ohlcv_1h.{parquet|feather|csv}

Uso:
  /Users/manuel/Desktop/MODELOX/.venv311/bin/python x/generar_timeframes.py

Nota:
- Por defecto NO sobreescribe si el archivo destino ya existe.

IMPORTANTE:
- No es posible crear velas 5m/15m REALES a partir de velas 1h (eso sería inventar datos).
- Este script valida que el source "*_5m" sea realmente 5m (mediana delta ~= 300s).
    Si no lo es, lo salta para evitar generar archivos mal etiquetados.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl


@dataclass(frozen=True)
class AssetSpec:
    symbol: str


ASSETS: list[AssetSpec] = [
    AssetSpec("BTC"),
    AssetSpec("GOLD"),
    AssetSpec("SP500"),
    AssetSpec("NASDAQ"),
]


def _resolve_source_dir(root: Path) -> Path:
    """Encuentra carpeta fuente para datos 5m.

    Preferimos (si existe):
        ./data copia/ohlcv
    si no, usamos:
        ./data/ohlcv
    """
    cand = root / "data copia" / "ohlcv"
    if cand.exists() and cand.is_dir():
        return cand
    return root / "data" / "ohlcv"


def _normalize_timestamp_col(df: pl.DataFrame) -> pl.DataFrame:
    # Handle common names
    if "timestamp" not in df.columns:
        for c in ("datetime", "date", "time"):
            if c in df.columns:
                df = df.rename({c: "timestamp"})
                break

    if "timestamp" not in df.columns:
        raise ValueError("Missing timestamp column")

    # Ensure Datetime(us, UTC)
    ts = pl.col("timestamp")
    # If already datetime, cast to us; else parse
    df = df.with_columns(
        ts.cast(pl.Datetime("us")).dt.replace_time_zone("UTC").alias("timestamp")
        if df.schema["timestamp"] != pl.Datetime
        else ts.dt.cast_time_unit("us").dt.replace_time_zone("UTC").alias("timestamp")
    )

    # Ensure required columns exist
    needed = {"open", "high", "low", "close"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}")
    if "volume" not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias("volume"))

    return df.sort("timestamp")


def _infer_median_candle_seconds(df: pl.DataFrame) -> float | None:
    """Devuelve el delta mediano entre velas en segundos.

    Útil para validar que un archivo "*_5m" sea realmente 5m.
    """
    if "timestamp" not in df.columns:
        return None
    if df.height < 3:
        return None
    dsec = (
        df.select((pl.col("timestamp").diff().dt.total_seconds()).alias("dsec"))
        .drop_nulls()
        .get_column("dsec")
    )
    if dsec.len() == 0:
        return None
    return float(dsec.quantile(0.5))


def _resample_ohlcv(df: pl.DataFrame, every: str) -> pl.DataFrame:
    """Resample OHLCV using Polars group_by_dynamic."""
    df = _normalize_timestamp_col(df)
    out = (
        df.group_by_dynamic(
            index_column="timestamp",
            every=every,
            closed="left",
            label="left",
        )
        .agg(
            [
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
            ]
        )
        .drop_nulls(["open", "high", "low", "close"])
        .sort("timestamp")
    )
    return out


def _write_all_formats(df: pl.DataFrame, base_path_no_ext: Path, *, overwrite: bool) -> None:
    base_path_no_ext.parent.mkdir(parents=True, exist_ok=True)

    targets = {
        "parquet": base_path_no_ext.with_suffix(".parquet"),
        "feather": base_path_no_ext.with_suffix(".feather"),
        "csv": base_path_no_ext.with_suffix(".csv"),
    }

    for fmt, path in targets.items():
        if path.exists() and not overwrite:
            continue
        if fmt == "parquet":
            df.write_parquet(path)
        elif fmt == "feather":
            df.write_ipc(path)
        elif fmt == "csv":
            df.write_csv(path)


def _find_source_5m(asset: str, data_dir: Path) -> Path | None:
    # Only accept final sources (no .bak). Keep the repository clean.
    candidates = [
        data_dir / f"{asset}_ohlcv_5m.parquet",
        data_dir / f"{asset}_ohlcv_5m.feather",
        data_dir / f"{asset}_ohlcv_5m.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _read_source(path: Path) -> pl.DataFrame:
    s = str(path).lower()
    if s.endswith(".parquet"):
        return pl.read_parquet(path)
    if s.endswith(".feather"):
        return pl.read_ipc(path)
    if s.endswith(".csv"):
        return pl.read_csv(path, try_parse_dates=True)
    # fallback
    return pl.read_parquet(path)


def generate_for_assets(assets: Iterable[str], *, overwrite: bool = False) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    source_dir = _resolve_source_dir(repo_root)
    data_dir = repo_root / "data" / "ohlcv"

    for a in assets:
        asset = str(a).upper().strip()
        src = _find_source_5m(asset, source_dir)
        if src is None:
            print(f"[SKIP] {asset}: no 5m source found in {source_dir}")
            continue

        print(f"[LOAD] {asset}: {src}")
        df5 = _read_source(src)
        df5 = _normalize_timestamp_col(df5)

        # VALIDACIÓN: la fuente debe ser realmente 5m.
        # Si el source está mal nombrado (p.ej. contiene velas 1h), NO generamos 5m/15m/1h
        # para evitar crear datasets inconsistentes.
        med_sec = _infer_median_candle_seconds(df5)
        if med_sec is None:
            print(f"[WARN] {asset}: cannot infer candle delta; skipping")
            continue
        # tolerancia amplia por posibles huecos/mercado cerrado: sólo miramos el MEDIAN
        if not (240.0 <= med_sec <= 360.0):
            # Ej: 3600s => realmente 1h
            print(
                f"[WARN] {asset}: source '{src.name}' is not 5m (median delta={med_sec:.0f}s). "
                "Skipping generation to avoid wrong *_5m/_15m files."
            )
            continue

        # Write 5m in all formats
        _write_all_formats(df5, data_dir / f"{asset}_ohlcv_5m", overwrite=overwrite)

        # Generate 15m and write
        df15 = _resample_ohlcv(df5, every="15m")
        _write_all_formats(df15, data_dir / f"{asset}_ohlcv_15m", overwrite=overwrite)

        # Generate 1h and write
        df1h = _resample_ohlcv(df5, every="1h")
        _write_all_formats(df1h, data_dir / f"{asset}_ohlcv_1h", overwrite=overwrite)

        print(f"[OK] {asset}: 5m/15m/1h generated")


if __name__ == "__main__":
    generate_for_assets([a.symbol for a in ASSETS], overwrite=False)
