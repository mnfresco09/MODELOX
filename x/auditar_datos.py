from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import polars as pl

from general.configuracion import resolve_archivo_data_tf
from modelox.core.data import load_data


@dataclass(frozen=True)
class AuditRow:
    asset: str
    tf: int
    rows: int
    dup_ts: int
    dsec_min: float | None
    dsec_p50: float | None
    dsec_p90: float | None
    dsec_max: float | None
    gaps_gt_1_5x: int
    min_ts: str
    max_ts: str
    path: str


def _q(series: pl.Series, probs: Iterable[float]) -> list[float | None]:
    if series.len() == 0:
        return [None for _ in probs]
    return [float(series.quantile(p)) for p in probs]


def audit_one(asset: str, tf: int) -> AuditRow:
    path = resolve_archivo_data_tf(asset, tf, formato="parquet")
    df = load_data(path).select(["timestamp", "open", "high", "low", "close", "volume"]).sort("timestamp")

    n = df.height
    min_ts, max_ts = df.select(
        [
            pl.col("timestamp").min().alias("min_ts"),
            pl.col("timestamp").max().alias("max_ts"),
        ]
    ).row(0)

    uniq = int(df.select(pl.col("timestamp").n_unique().alias("n_unique")).item())
    dup = int(n - uniq)

    dsec = (
        df.select((pl.col("timestamp").diff().dt.total_seconds()).alias("dsec"))
        .drop_nulls()
        .get_column("dsec")
    )

    dmin, d50, d90, dmax = _q(dsec, [0.0, 0.5, 0.9, 1.0])

    expected = 3600 if tf == 60 else int(tf) * 60
    gaps = int((dsec > expected * 1.5).sum()) if dsec.len() else 0

    return AuditRow(
        asset=str(asset),
        tf=int(tf),
        rows=int(n),
        dup_ts=dup,
        dsec_min=dmin,
        dsec_p50=d50,
        dsec_p90=d90,
        dsec_max=dmax,
        gaps_gt_1_5x=gaps,
        min_ts=str(min_ts),
        max_ts=str(max_ts),
        path=str(path),
    )


def main() -> None:
    assets = ["GOLD", "BTC", "SP500", "NASDAQ"]
    tfs = [5, 15, 60]

    rows: list[dict[str, Any]] = []
    for a in assets:
        for tf in tfs:
            r = audit_one(a, tf)
            rows.append(r.__dict__)

    import pandas as pd

    df = pd.DataFrame(rows).sort_values(["asset", "tf"], ascending=[True, True]).reset_index(drop=True)

    # Verify that all three artifact formats exist for each asset+timeframe
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data" / "ohlcv"
    tf_to_suffix = {5: "5m", 15: "15m", 60: "1h"}

    def _artifact_status(asset: str, tf: int) -> tuple[bool, list[str]]:
        suffix = tf_to_suffix.get(tf, str(tf))
        base = data_dir / f"{asset}_ohlcv_{suffix}"
        required = [base.with_suffix(ext) for ext in (".parquet", ".feather", ".csv")]
        missing = [p.name for p in required if not p.exists()]
        return (len(missing) == 0), missing

    print("\n=== DATASET SUMMARY (files + ranges) ===")
    expected_p50 = {5: 300.0, 15: 900.0, 60: 3600.0}
    for asset in assets:
        print(f"\n[{asset}]")
        for tf in tfs:
            row = df[(df["asset"] == asset) & (df["tf"] == tf)]
            if row.empty:
                ok_files, missing = _artifact_status(asset, tf)
                missing_str = ("" if ok_files else f" missing={missing}")
                print(f"  tf={tf}: MISSING PARQUET AUDIT | files_ok={ok_files}{missing_str}")
                continue

            r = row.iloc[0]
            ok_files, missing = _artifact_status(asset, tf)

            # Short timestamps to keep output readable in narrow terminals
            start = str(r["min_ts"]).replace("+00:00", "")[:16]
            end = str(r["max_ts"]).replace("+00:00", "")[:16]
            n = int(r["rows"])
            dup = int(r["dup_ts"])
            p50 = float(r["dsec_p50"]) if pd.notna(r["dsec_p50"]) else None
            gaps = int(r["gaps_gt_1_5x"])

            ok_p50 = (p50 is not None) and (abs(p50 - expected_p50[tf]) < 1e-6)
            ok_dup = dup == 0

            notes: list[str] = []
            if not ok_files:
                notes.append(f"missing_files={missing}")
            if not ok_p50:
                notes.append(f"p50={p50}")
            if not ok_dup:
                notes.append(f"dup={dup}")
            if gaps > 0:
                notes.append(f"gaps>{tf}m={gaps}")

            notes_str = ("" if not notes else "  notes: " + " ; ".join(notes))
            print(f"  tf={tf}  files_ok={ok_files}  rows={n}  p50_sec={p50}")
            print(f"    start: {start}")
            print(f"    end:   {end}")
            if notes_str:
                print(f"    {notes_str}")


if __name__ == "__main__":
    main()
