from __future__ import annotations

"""Arregla y valida datasets OHLCV multi-timeframe.

Qué hace:
- Audita archivos existentes en data/ohlcv.
- Detecta archivos mal etiquetados (ej: *_ohlcv_5m.* con velas de 1h).
- Mueve los archivos inconsistentes a data/ohlcv/_invalid/<ASSET>/...

NO descarga datos externos.
Si un activo no tiene 5m real, no es posible crear 5m/15m reales.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl


ASSETS = ["BTC", "GOLD", "SP500", "NASDAQ"]
SUFFIXES = ["5m", "15m", "1h"]
FORMATS = ["parquet", "feather", "csv"]


@dataclass(frozen=True)
class Check:
    asset: str
    suffix: str
    fmt: str
    path: Path
    ok: bool
    median_sec: float | None
    reason: str


def _read_any(path: Path) -> pl.DataFrame:
    s = path.name.lower()
    if s.endswith(".parquet"):
        return pl.read_parquet(path)
    if s.endswith(".feather"):
        return pl.read_ipc(path)
    if s.endswith(".csv"):
        return pl.read_csv(path, try_parse_dates=True)
    raise ValueError(f"Unsupported: {path}")


def _normalize_timecol(df: pl.DataFrame) -> pl.DataFrame:
    if "timestamp" not in df.columns:
        for c in ("datetime", "date", "time"):
            if c in df.columns:
                df = df.rename({c: "timestamp"})
                break
    if "timestamp" not in df.columns:
        raise ValueError("Missing time column")
    df = df.with_columns(pl.col("timestamp").dt.cast_time_unit("us").dt.replace_time_zone("UTC"))
    return df.sort("timestamp")


def _median_delta_seconds(df: pl.DataFrame) -> float | None:
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


def _expected_seconds(suffix: str) -> int:
    if suffix == "1h":
        return 3600
    if suffix.endswith("m"):
        return int(float(suffix[:-1])) * 60
    raise ValueError(f"Unknown suffix: {suffix}")


def check_file(path: Path, *, asset: str, suffix: str, fmt: str) -> Check:
    try:
        df = _read_any(path)
        df = _normalize_timecol(df)
        med = _median_delta_seconds(df)
        if med is None:
            return Check(asset, suffix, fmt, path, False, None, "cannot infer median delta")
        expected = _expected_seconds(suffix)
        # tolerancia: mediana dentro de +-20%
        lo = expected * 0.8
        hi = expected * 1.2
        if not (lo <= med <= hi):
            return Check(asset, suffix, fmt, path, False, med, f"median delta {med:.0f}s != expected {expected}s")
        return Check(asset, suffix, fmt, path, True, med, "ok")
    except Exception as e:
        return Check(asset, suffix, fmt, path, False, None, f"error: {e}")


def move_to_invalid(check: Check, *, root: Path) -> None:
    invalid_root = root / "_invalid" / check.asset
    invalid_root.mkdir(parents=True, exist_ok=True)
    target = invalid_root / check.path.name
    # If target exists, add numeric suffix
    if target.exists():
        stem = target.stem
        suf = target.suffix
        i = 2
        while True:
            cand = invalid_root / f"{stem}__{i}{suf}"
            if not cand.exists():
                target = cand
                break
            i += 1
    check.path.rename(target)


def iter_existing(data_dir: Path) -> Iterable[tuple[str, str, str, Path]]:
    for asset in ASSETS:
        for suffix in SUFFIXES:
            for fmt in FORMATS:
                p = data_dir / f"{asset}_ohlcv_{suffix}.{fmt}"
                if p.exists():
                    yield asset, suffix, fmt, p


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data" / "ohlcv"

    checks: list[Check] = []
    for asset, suffix, fmt, path in iter_existing(data_dir):
        checks.append(check_file(path, asset=asset, suffix=suffix, fmt=fmt))

    bad = [c for c in checks if not c.ok]
    good = [c for c in checks if c.ok]

    print(f"[OK] {len(good)} files")
    print(f"[BAD] {len(bad)} files")

    for c in bad:
        print(f"- {c.path.name}: {c.reason}")

    if bad:
        print("\nMoving BAD files to data/ohlcv/_invalid/...\n")
        for c in bad:
            move_to_invalid(c, root=data_dir)
        print("Done.")


if __name__ == "__main__":
    main()
