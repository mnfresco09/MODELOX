from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil

import polars as pl


@dataclass(frozen=True)
class ConvertResult:
    asset: str
    source_parquet: Path
    rows_in: int
    rows_out: int
    hours_dropped: int
    action: str  # "converted" | "skipped" | "empty"


def _detect_time_col(df: pl.DataFrame) -> str:
    for c in ("timestamp", "datetime", "date", "time"):
        if c in df.columns:
            return c
    raise ValueError(f"No encuentro columna temporal. Columnas: {df.columns}")


def _ensure_utc(df: pl.DataFrame, time_col: str) -> pl.DataFrame:
    dtype = df.schema[time_col]
    if isinstance(dtype, pl.Datetime):
        tz = dtype.time_zone
        expr = pl.col(time_col).dt.cast_time_unit("us")
        if tz is None:
            expr = expr.dt.replace_time_zone("UTC")
        elif tz != "UTC":
            expr = expr.dt.convert_time_zone("UTC")
        return df.with_columns(expr.alias(time_col))
    return df


def _infer_median_step_minutes(df: pl.DataFrame, time_col: str) -> float | None:
    if df.height < 2:
        return None
    diffs = (
        df.select(pl.col(time_col).sort().diff().drop_nulls().dt.total_minutes())
        .to_series()
        .to_list()
    )
    if not diffs:
        return None
    diffs_sorted = sorted(float(x) for x in diffs)
    mid = len(diffs_sorted) // 2
    if len(diffs_sorted) % 2 == 1:
        return diffs_sorted[mid]
    return (diffs_sorted[mid - 1] + diffs_sorted[mid]) / 2.0


def ohlcv_5m_to_1h_strict(df_5m: pl.DataFrame) -> pl.DataFrame:
    """Convierte OHLCV 5m a 1H alineado al reloj.

    Reglas:
    - Alineación al reloj (horas exactas): label='left', closed='left'
    - OHLCV: open=first, high=max, low=min, close=last, volume=sum
    - Filtra velas incompletas: exige exactamente 12 barras de 5m por hora
    """
    if df_5m.is_empty():
        return df_5m

    time_col = _detect_time_col(df_5m)
    df_5m = _ensure_utc(df_5m, time_col).sort(time_col)

    # Idempotencia: si ya está en 1H, no vuelvas a aplicar el filtro de 12 velas.
    step_min = _infer_median_step_minutes(df_5m, time_col)
    if step_min is not None and 55.0 <= step_min <= 65.0:
        return df_5m

    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols.difference(df_5m.columns)
    if missing:
        raise ValueError(f"Faltan columnas OHLCV requeridas: {sorted(missing)}")

    hour_key = pl.col(time_col).dt.truncate("1h").alias("__hour")

    out = (
        df_5m.with_columns(hour_key)
        .group_by("__hour")
        .agg(
            [
                pl.len().alias("__count"),
                pl.col("open").sort_by(time_col).first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").sort_by(time_col).last().alias("close"),
                pl.col("volume").sum().alias("volume"),
            ]
        )
        .filter(pl.col("__count") == 12)
        .drop("__count")
        .rename({"__hour": time_col})
        .sort(time_col)
    )

    return out


def _asset_from_path(p: Path) -> str:
    name = p.name
    # e.g. GOLD_ohlcv_5m.parquet -> GOLD
    return name.split("_", 1)[0].upper()


def _backup_file(path: Path) -> None:
    if not path.exists():
        return
    bak = path.with_suffix(path.suffix + ".bak")
    # No machaca un backup existente: lo conserva.
    if bak.exists():
        return
    shutil.copy2(path, bak)


def convert_asset(
    base_dir: Path,
    asset: str,
    *,
    overwrite: bool,
    keep_source: bool,
) -> ConvertResult:
    src_parquet = base_dir / f"{asset}_ohlcv_5m.parquet"
    dst_parquet = base_dir / f"{asset}_ohlcv_1h.parquet"

    if not src_parquet.exists():
        # Ya está en nombre 1h, nada que hacer.
        if dst_parquet.exists():
            df_existing = pl.read_parquet(dst_parquet)
            return ConvertResult(
                asset=asset,
                source_parquet=dst_parquet,
                rows_in=df_existing.height,
                rows_out=df_existing.height,
                hours_dropped=0,
                action="skipped",
            )
        raise FileNotFoundError(f"No existe origen 5m ni destino 1h para: {asset}")

    df_5m = pl.read_parquet(src_parquet)
    rows_in = df_5m.height
    if rows_in == 0:
        return ConvertResult(
            asset=asset,
            source_parquet=parquet_path,
            rows_in=0,
            rows_out=0,
            hours_dropped=0,
            action="empty",
        )

    time_col = _detect_time_col(df_5m)
    df_5m = _ensure_utc(df_5m, time_col).sort(time_col)
    step_min = _infer_median_step_minutes(df_5m, time_col)

    if step_min is not None and 55.0 <= step_min <= 65.0:
        # Ya es 1H: dejamos los datos como están.
        df_1h = df_5m
        action = "skipped"
        hours_dropped = 0
    else:
        df_1h = ohlcv_5m_to_1h_strict(df_5m)
        action = "converted"
        rows_out = df_1h.height
        # Horas potenciales (aligned) vs horas válidas (12 velas)
        hour_counts = (
            df_5m.select(pl.col(time_col).dt.truncate("1h").alias("h"))
            .group_by("h")
            .len()
        )
        hours_dropped = int((hour_counts["len"] != 12).sum())

    rows_out = df_1h.height

    if not overwrite:
        raise ValueError("Este script está pensado para sobrescribir (overwrite=True).")

    # Backups para evitar pérdida de datos
    _backup_file(src_parquet)
    if dst_parquet.exists():
        _backup_file(dst_parquet)

    # Escribe destino 1h
    df_1h.write_parquet(dst_parquet)

    # Mantiene consistencia con otros formatos si existen
    src_feather = base_dir / f"{asset}_ohlcv_5m.feather"
    dst_feather = base_dir / f"{asset}_ohlcv_1h.feather"
    if src_feather.exists():
        _backup_file(src_feather)
        if dst_feather.exists():
            _backup_file(dst_feather)
        df_1h.write_ipc(dst_feather)

    src_csv = base_dir / f"{asset}_ohlcv_5m.csv"
    dst_csv = base_dir / f"{asset}_ohlcv_1h.csv"
    if src_csv.exists():
        _backup_file(src_csv)
        if dst_csv.exists():
            _backup_file(dst_csv)
        df_1h.write_csv(dst_csv)

    # Borra el origen 5m (lo pedido) salvo que el usuario pida conservarlo.
    if not keep_source:
        for p in (src_parquet, src_feather, src_csv):
            if p.exists():
                p.unlink()

    return ConvertResult(
        asset=asset,
        source_parquet=dst_parquet,
        rows_in=rows_in,
        rows_out=rows_out,
        hours_dropped=hours_dropped,
        action=action,
    )


def discover_assets(base_dir: Path) -> list[str]:
    assets: list[str] = []
    for p in sorted(base_dir.glob("*_ohlcv_5m.parquet")):
        assets.append(_asset_from_path(p))
    if not assets:
        # Si ya renombraste todo a 1h, igualmente detecta assets.
        for p in sorted(base_dir.glob("*_ohlcv_1h.parquet")):
            assets.append(_asset_from_path(p))
    # unique preserving order
    seen = set()
    out = []
    for a in assets:
        if a not in seen:
            out.append(a)
            seen.add(a)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convierte archivos OHLCV 5m a 1H con reglas estrictas (12 velas de 5m por hora) "
            "y sobrescribe los archivos para mantener los mismos nombres." 
        )
    )
    parser.add_argument(
        "--dir",
        default="data/ohlcv",
        help="Directorio de datos OHLCV (lee *_ohlcv_5m.* y escribe *_ohlcv_1h.*).",
    )
    parser.add_argument(
        "--assets",
        default="all",
        help="Lista separada por comas (BTC,GOLD,SP500,NASDAQ) o 'all'.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescribe los archivos existentes (requerido).",
    )
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="Conserva los archivos *_ohlcv_5m.* (por defecto se eliminan tras crear *_ohlcv_1h.*).",
    )
    args = parser.parse_args()

    base_dir = Path(args.dir)
    if not base_dir.exists():
        raise SystemExit(f"No existe el directorio: {base_dir}")

    if not args.overwrite:
        raise SystemExit("Por seguridad, debes pasar --overwrite para ejecutar.")

    if str(args.assets).lower() == "all":
        assets = discover_assets(base_dir)
    else:
        assets = [a.strip().upper() for a in str(args.assets).split(",") if a.strip()]

    if not assets:
        raise SystemExit("No se detectaron assets para convertir.")

    results: list[ConvertResult] = []
    for asset in assets:
        res = convert_asset(
            base_dir,
            asset,
            overwrite=True,
            keep_source=bool(args.keep_source),
        )
        results.append(res)
        if res.action == "converted":
            print(
                f"[OK] {asset}: {res.rows_in} filas 5m -> {res.rows_out} filas 1H (strict). Horas descartadas={res.hours_dropped}"
            )
        elif res.action == "skipped":
            print(f"[SKIP] {asset}: ya está en 1H. Filas={res.rows_out}")
        else:
            print(f"[WARN] {asset}: archivo vacío, no se convierte")

    total_in = sum(r.rows_in for r in results)
    total_out = sum(r.rows_out for r in results)
    print(f"[TOTAL] {len(results)} assets: {total_in} -> {total_out} filas")


if __name__ == "__main__":
    main()
