from __future__ import annotations

import argparse
import calendar
import hashlib
import shutil
import tempfile
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

import polars as pl
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table


BINANCE_BASE_URL = "https://data.binance.vision/data"
DEFAULT_VISION_DIR = Path("vision")

BTC_ASSET = "BTC"
BTC_SYMBOL = "BTCUSDT"
BTC_MARKET = "futures/um"
BTC_INTERVAL = "1m"

SUPPORTED_FORMATS = {"csv", "parquet", "feather"}
DEFAULT_FORMATS = ("csv", "parquet", "feather")

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "num_trades",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]

BTC_FEATURE_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "num_trades",
    "taker_buy_volume",
    "taker_sell_volume",
    "taker_buy_quote_volume",
    "taker_sell_quote_volume",
    "premium_index_close",
    "open_interest_5m",
    "open_interest_value_5m",
    "toptrader_count_long_short_ratio_5m",
    "toptrader_sum_long_short_ratio_5m",
    "global_long_short_ratio_5m",
    "taker_long_short_ratio_5m",
    "predicted_funding_rate_1m",
]

BINANCE_KLINE_TYPES = {"klines", "premiumIndexKlines"}
BINANCE_NON_INTERVAL_TYPES = {"metrics"}

# =============================================================================
# RANGO DE FECHAS
# =============================================================================
FECHA_INICIO = "2020-01-01"              # YYYY-MM-DD   (vacío → lee de general/configuracion o usa 2021-01-01)
FECHA_FIN    = "2025-12-31"              # YYYY-MM-DD   (vacío → hoy)

# =============================================================================
# ARCHIVO DE SALIDA
# =============================================================================
VISION_DIR = Path("datos")   # Carpeta destino (directo al sistema — feather/parquet)
FORMATOS   = "feather"       # feather | parquet | feather,parquet

# =============================================================================
# RENDIMIENTO
# =============================================================================
WORKERS = 6                    # Descargas paralelas simultáneas (subir en conexiones rápidas)
# =============================================================================


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    status: str
    url: str


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _default_date_range() -> tuple[str, str]:
    try:
        import general.configuracion as _cfg

        start = FECHA_INICIO or str(_cfg.FECHA_INICIO)
        end = FECHA_FIN or str(_cfg.FECHA_FIN)
    except Exception:
        start = FECHA_INICIO or "2021-01-01"
        end = FECHA_FIN or date.today().isoformat()
    return start, end


def _parse_formats(raw: str) -> list[str]:
    values: list[str] = []
    for item in str(raw).replace(";", ",").split(","):
        fmt = item.strip().lower().lstrip(".")
        if not fmt:
            continue
        if fmt == "all":
            values.extend(DEFAULT_FORMATS)
            continue
        if fmt in {"ipc", "fthr"}:
            fmt = "feather"
        if fmt not in SUPPORTED_FORMATS:
            allowed = ", ".join(sorted(SUPPORTED_FORMATS | {"all"}))
            raise ValueError(f"Formato no soportado: {item}. Usa: {allowed}")
        values.append(fmt)

    deduped: list[str] = []
    for fmt in values or list(DEFAULT_FORMATS):
        if fmt not in deduped:
            deduped.append(fmt)
    return deduped


def _iter_months(start: date, end: date) -> Iterable[date]:
    cur = date(start.year, start.month, 1)
    last = date(end.year, end.month, 1)
    while cur <= last:
        yield cur
        year = cur.year + (1 if cur.month == 12 else 0)
        month = 1 if cur.month == 12 else cur.month + 1
        cur = date(year, month, 1)


def _iter_days(start: date, end: date) -> Iterable[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _month_bounds(month_start: date, start: date, end: date) -> tuple[date, date]:
    last_day = calendar.monthrange(month_start.year, month_start.month)[1]
    month_end = date(month_start.year, month_start.month, last_day)
    return max(start, month_start), min(end, month_end)


def _start_datetime_utc(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=timezone.utc)


def _end_datetime_utc(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 23, 59, 59, 999000, tzinfo=timezone.utc)


def _epoch_ms_expr(column: str) -> pl.Expr:
    raw = pl.col(column).cast(pl.Int64, strict=False)
    millis = pl.when(raw > 10_000_000_000_000).then(raw // 1000).otherwise(raw)
    return pl.from_epoch(millis, time_unit="ms").dt.cast_time_unit("us").dt.replace_time_zone("UTC")


def _binance_url(
    *,
    period: str,
    data_type: str,
    day_or_month: date,
    symbol: str = BTC_SYMBOL,
    market: str = BTC_MARKET,
    interval: str = BTC_INTERVAL,
) -> tuple[str, str]:
    suffix = f"{day_or_month:%Y-%m}" if period == "monthly" else f"{day_or_month:%Y-%m-%d}"

    if data_type in BINANCE_KLINE_TYPES:
        filename = f"{symbol}-{interval}-{suffix}.zip"
        path = f"{market}/{period}/{data_type}/{symbol}/{interval}/{filename}"
    elif data_type in BINANCE_NON_INTERVAL_TYPES:
        filename = f"{symbol}-{data_type}-{suffix}.zip"
        path = f"{market}/{period}/{data_type}/{symbol}/{filename}"
    else:
        raise ValueError(f"Tipo Binance no soportado: {data_type}")

    return f"{BINANCE_BASE_URL}/{path}", filename


def _download_file(
    url: str,
    dest: Path,
    *,
    progress: Progress,
    overwrite: bool,
    timeout: int = 60,
    retries: int = 3,
) -> DownloadResult | None:
    if dest.exists() and dest.stat().st_size > 0 and not overwrite:
        return DownloadResult(dest, "cache", url)

    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("Falta instalar requests para descargar desde Binance Data Vision") from exc

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                if response.status_code == 404:
                    return None
                response.raise_for_status()

                total_header = response.headers.get("content-length")
                total = int(total_header) if total_header and total_header.isdigit() else None
                task_id = progress.add_task(dest.name, total=total)

                with tmp.open("wb") as fh:
                    for chunk in response.iter_content(chunk_size=2 * 1024 * 1024):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        progress.update(task_id, advance=len(chunk))

                if total is not None:
                    progress.update(task_id, completed=total)
                progress.remove_task(task_id)

            tmp.replace(dest)
            return DownloadResult(dest, "descargado", url)

        except Exception:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            if attempt == retries - 1:
                raise

    return None


def _verify_checksum(zip_path: Path, checksum_url: str, *, timeout: int = 30) -> bool:
    try:
        import requests

        response = requests.get(checksum_url, timeout=timeout)
        if response.status_code == 404:
            return True
        response.raise_for_status()
        expected = response.text.strip().split()[0].lower()
        digest = hashlib.sha256(zip_path.read_bytes()).hexdigest().lower()
        return digest == expected
    except Exception:
        return True


@dataclass
class _ArchiveJob:
    url: str
    dest: Path
    data_type: str
    period: str
    label: str
    checksum_url: str = ""


def _plan_monthly_jobs(
    data_type: str, start: date, end: date, tmp_dir: Path
) -> list[_ArchiveJob]:
    jobs = []
    for month in _iter_months(start, end):
        url, filename = _binance_url(period="monthly", data_type=data_type, day_or_month=month)
        dest = tmp_dir / "monthly" / data_type / filename
        jobs.append(_ArchiveJob(url=url, dest=dest, data_type=data_type, period="monthly",
                                label=f"{month:%Y-%m}", checksum_url=f"{url}.CHECKSUM"))
    return jobs


def _plan_daily_jobs(
    data_type: str, start: date, end: date, tmp_dir: Path
) -> list[_ArchiveJob]:
    jobs = []
    for day in _iter_days(start, end):
        url, filename = _binance_url(period="daily", data_type=data_type, day_or_month=day)
        dest = tmp_dir / "daily" / data_type / filename
        jobs.append(_ArchiveJob(url=url, dest=dest, data_type=data_type, period="daily",
                                label=f"{day:%Y-%m-%d}"))
    return jobs


def _run_jobs_parallel(
    jobs: list[_ArchiveJob],
    *,
    progress: Progress,
    overwrite: bool,
    verify_checksum: bool,
    workers: int,
    console: Optional[Console] = None,
) -> tuple[list[Path], list[_ArchiveJob]]:
    """Ejecuta una lista de jobs en paralelo. Devuelve (archivos_ok, jobs_faltantes)."""
    archives: list[Path] = []
    missing: list[_ArchiveJob] = []
    lock = threading.Lock()

    def _run(job: _ArchiveJob) -> None:
        try:
            result = _download_file(job.url, job.dest, progress=progress, overwrite=overwrite)
        except Exception as exc:
            if console:
                console.print(f"  [red]Error[/] {job.label}: {exc}")
            with lock:
                missing.append(job)
            return

        with lock:
            if result is None:
                missing.append(job)
            else:
                if verify_checksum and job.checksum_url and not _verify_checksum(job.dest, job.checksum_url):
                    if console:
                        console.print(f"  [red]Checksum inválido:[/] {job.label}")
                    missing.append(job)
                else:
                    archives.append(job.dest)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vision") as pool:
        futures = [pool.submit(_run, j) for j in jobs]
        for f in as_completed(futures):
            f.result()  # propaga excepciones no capturadas en _run

    return archives, missing


def _download_binance_archives(
    *,
    data_type: str,
    start: date,
    end: date,
    tmp_dir: Path,
    overwrite: bool,
    verify_checksum: bool,
    daily_fallback: bool,
    console: Console,
    progress: Progress,
    workers: int = 4,
) -> list[Path]:
    archives: list[Path] = []

    if data_type == "metrics":
        jobs = _plan_daily_jobs(data_type, start, end, tmp_dir)
        found, _ = _run_jobs_parallel(jobs, progress=progress, overwrite=overwrite,
                                      verify_checksum=False, workers=workers, console=console)
        archives.extend(found)
        console.print(f"  [dim]metrics:[/] {len(archives)}/{len(jobs)} días")
        return archives

    monthly_jobs = _plan_monthly_jobs(data_type, start, end, tmp_dir)
    ok, missing_months = _run_jobs_parallel(monthly_jobs, progress=progress, overwrite=overwrite,
                                            verify_checksum=verify_checksum, workers=workers,
                                            console=console)
    archives.extend(ok)

    if missing_months and daily_fallback:
        daily_jobs: list[_ArchiveJob] = []
        for job in missing_months:
            month_start = date.fromisoformat(job.label + "-01")
            day_start, day_end = _month_bounds(month_start, start, end)
            daily_jobs.extend(_plan_daily_jobs(data_type, day_start, day_end, tmp_dir))
        if daily_jobs:
            daily_ok, _ = _run_jobs_parallel(daily_jobs, progress=progress, overwrite=overwrite,
                                             verify_checksum=False, workers=workers, console=console)
            archives.extend(daily_ok)

    label = data_type.replace("premiumIndexKlines", "premium").replace("klines", "klines")
    console.print(f"  [dim]{label}:[/] {len(archives)} archivos ({len(monthly_jobs)} meses)")
    return archives


def _zip_csv_member(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.lower().endswith(".csv") and not name.endswith("/"):
                return name
    raise ValueError(f"Zip sin CSV: {zip_path}")


def _read_zip_csv(zip_path: Path, *, has_header: bool = False) -> pl.DataFrame:
    member = _zip_csv_member(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as fh:
            return pl.read_csv(fh, has_header=has_header, infer_schema_length=0)


def _read_kline_zip(zip_path: Path) -> pl.DataFrame:
    df = _read_zip_csv(zip_path)
    if df.width < len(KLINE_COLUMNS):
        raise ValueError(f"Kline con {df.width} columnas, esperado {len(KLINE_COLUMNS)}: {zip_path}")

    rename_map = {old: new for old, new in zip(df.columns[: len(KLINE_COLUMNS)], KLINE_COLUMNS)}
    df = df.rename(rename_map)

    numeric = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_volume",
        "taker_buy_quote_volume",
    ]

    return (
        df.with_columns(
            _epoch_ms_expr("open_time").alias("timestamp"),
            *[pl.col(c).cast(pl.Float64, strict=False).alias(c) for c in numeric],
            pl.col("num_trades").cast(pl.Int64, strict=False).alias("num_trades"),
        )
        .drop_nulls(["timestamp", "open", "high", "low", "close"])
        .select(
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "num_trades",
            "taker_buy_volume",
            (pl.col("volume") - pl.col("taker_buy_volume")).alias("taker_sell_volume"),
            "taker_buy_quote_volume",
            (pl.col("quote_volume") - pl.col("taker_buy_quote_volume")).alias("taker_sell_quote_volume"),
        )
    )


def _read_premium_index_zip(zip_path: Path) -> pl.DataFrame:
    df = _read_zip_csv(zip_path)
    if df.width < len(KLINE_COLUMNS):
        raise ValueError(f"Premium index con {df.width} columnas, esperado {len(KLINE_COLUMNS)}: {zip_path}")

    rename_map = {old: new for old, new in zip(df.columns[: len(KLINE_COLUMNS)], KLINE_COLUMNS)}
    df = df.rename(rename_map)

    return (
        df.with_columns(
            _epoch_ms_expr("open_time").alias("timestamp"),
            pl.col("close").cast(pl.Float64, strict=False).alias("premium_index_close"),
        )
        .drop_nulls(["timestamp", "premium_index_close"])
        .select("timestamp", "premium_index_close")
    )


def _read_metrics_zip(zip_path: Path) -> pl.DataFrame:
    df = _read_zip_csv(zip_path, has_header=True)
    required = {
        "create_time",
        "sum_open_interest",
        "sum_open_interest_value",
        "count_toptrader_long_short_ratio",
        "sum_toptrader_long_short_ratio",
        "count_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metrics sin columnas {sorted(missing)}: {zip_path}")

    return (
        df.with_columns(
            pl.col("create_time")
            .str.strptime(pl.Datetime("us"), format="%Y-%m-%d %H:%M:%S", strict=False)
            .dt.replace_time_zone("UTC")
            .alias("timestamp"),
            pl.col("sum_open_interest").cast(pl.Float64, strict=False).alias("open_interest_5m"),
            pl.col("sum_open_interest_value").cast(pl.Float64, strict=False).alias("open_interest_value_5m"),
            pl.col("count_toptrader_long_short_ratio")
            .cast(pl.Float64, strict=False)
            .alias("toptrader_count_long_short_ratio_5m"),
            pl.col("sum_toptrader_long_short_ratio")
            .cast(pl.Float64, strict=False)
            .alias("toptrader_sum_long_short_ratio_5m"),
            pl.col("count_long_short_ratio").cast(pl.Float64, strict=False).alias("global_long_short_ratio_5m"),
            pl.col("sum_taker_long_short_vol_ratio")
            .cast(pl.Float64, strict=False)
            .alias("taker_long_short_ratio_5m"),
        )
        .drop_nulls(["timestamp"])
        .select(
            "timestamp",
            "open_interest_5m",
            "open_interest_value_5m",
            "toptrader_count_long_short_ratio_5m",
            "toptrader_sum_long_short_ratio_5m",
            "global_long_short_ratio_5m",
            "taker_long_short_ratio_5m",
        )
        .sort("timestamp")
        .unique(subset=["timestamp"], keep="last")
    )


def _concat_or_empty(frames: list[pl.DataFrame]) -> pl.DataFrame:
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed")


def _filter_range(df: pl.DataFrame, start: date, end: date) -> pl.DataFrame:
    if df.height == 0 or "timestamp" not in df.columns:
        return df
    return df.filter(
        (pl.col("timestamp") >= _start_datetime_utc(start))
        & (pl.col("timestamp") <= _end_datetime_utc(end))
    )


def _join_feature_asof(base: pl.DataFrame, feature: pl.DataFrame) -> pl.DataFrame:
    if feature.height == 0:
        return base
    return base.sort("timestamp").join_asof(
        feature.sort("timestamp"),
        on="timestamp",
        strategy="backward",
    )


def _clamp_expr(expr: pl.Expr, lower: float, upper: float) -> pl.Expr:
    return pl.when(expr > upper).then(upper).when(expr < lower).then(lower).otherwise(expr)


def _add_predicted_funding_rate_1m(
    df: pl.DataFrame,
    *,
    premium_col: str = "premium_index_close",
    interest_rate: float = 0.0001,
    clamp_abs: float = 0.0005,
) -> pl.DataFrame:
    if premium_col not in df.columns:
        return df

    avg_expr = (
        (pl.col(premium_col) * pl.col("_funding_weight")).cum_sum().over("_funding_cycle")
        / pl.col("_funding_weight").cum_sum().over("_funding_cycle")
    )

    return (
        df.sort("timestamp")
        .with_columns(pl.col("timestamp").dt.truncate("8h").alias("_funding_cycle"))
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("_funding_cycle").cast(pl.Float64).alias("_funding_weight")
        )
        .with_columns(avg_expr.alias("avg_premium_index_8h_weighted_1m"))
        .with_columns(
            (
                pl.col("avg_premium_index_8h_weighted_1m")
                + _clamp_expr(
                    pl.lit(float(interest_rate)) - pl.col("avg_premium_index_8h_weighted_1m"),
                    -float(clamp_abs),
                    float(clamp_abs),
                )
            ).alias("predicted_funding_rate_1m")
        )
        .drop(["_funding_cycle", "_funding_weight", "avg_premium_index_8h_weighted_1m"])
    )


METRICS_COLUMNS = [
    "open_interest_5m",
    "open_interest_value_5m",
    "toptrader_count_long_short_ratio_5m",
    "toptrader_sum_long_short_ratio_5m",
    "global_long_short_ratio_5m",
    "taker_long_short_ratio_5m",
]
PREMIUM_COLUMNS = ["premium_index_close"]


def _ensure_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Agrega columnas faltantes como null Float64 para mantener esquema consistente."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        df = df.with_columns([pl.lit(None).cast(pl.Float64).alias(c) for c in missing])
    return df


def _select_feature_columns(df: pl.DataFrame, console: Console) -> pl.DataFrame:
    available = [col for col in BTC_FEATURE_COLUMNS if col in df.columns]
    return df.select(available)


def _build_btc_dataset(
    *,
    start: date,
    end: date,
    tmp_dir: Path,
    console: Console,
    progress: Progress,
    verify_checksum: bool,
    daily_fallback: bool,
    workers: int = WORKERS,
) -> pl.DataFrame:
    from rich.rule import Rule

    console.print(Rule("[bold cyan]Descarga[/]", style="dim"))

    results: dict[str, list[Path]] = {}
    errors: list[str] = []
    lock = threading.Lock()

    def _fetch(data_type: str, use_checksum: bool, force_daily: bool) -> None:
        try:
            archives = _download_binance_archives(
                data_type=data_type,
                start=start,
                end=end,
                tmp_dir=tmp_dir,
                overwrite=True,
                verify_checksum=use_checksum,
                daily_fallback=force_daily or daily_fallback,
                console=console,
                progress=progress,
                workers=workers,
            )
        except Exception as exc:
            with lock:
                errors.append(f"{data_type}: {exc}")
            archives = []
        with lock:
            results[data_type] = archives

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="vision_src") as pool:
        futures = [
            pool.submit(_fetch, "klines", verify_checksum, False),
            pool.submit(_fetch, "premiumIndexKlines", verify_checksum, False),
            pool.submit(_fetch, "metrics", False, True),
        ]
        for f in as_completed(futures):
            f.result()

    if errors:
        for err in errors:
            console.print(f"  [red]Error:[/] {err}")

    kline_archives = results.get("klines", [])
    if not kline_archives:
        raise RuntimeError("No se encontraron klines para BTCUSDT en el rango indicado")

    console.print(Rule("[bold cyan]Procesando[/]", style="dim"))

    console.print("  [dim]→[/] klines")
    ohlcv = (
        _concat_or_empty([_read_kline_zip(path) for path in kline_archives])
        .sort("timestamp")
        .unique(subset=["timestamp"], keep="last")
    )
    ohlcv = _filter_range(ohlcv, start, end)

    premium_archives = results.get("premiumIndexKlines", [])
    console.print("  [dim]→[/] premium index")
    if premium_archives:
        premium = (
            _concat_or_empty([_read_premium_index_zip(path) for path in premium_archives])
            .sort("timestamp")
            .unique(subset=["timestamp"], keep="last")
        )
        premium = _filter_range(premium, start, end)
        if premium.height > 0:
            ohlcv = ohlcv.join(premium, on="timestamp", how="left")
        else:
            console.print("  [yellow]⚠[/] sin datos de premium index — columnas en null")
    else:
        console.print("  [yellow]⚠[/] sin archivos de premium index — columnas en null")
    ohlcv = _ensure_columns(ohlcv, PREMIUM_COLUMNS)

    metrics_archives = results.get("metrics", [])
    console.print("  [dim]→[/] metrics 5m")
    if metrics_archives:
        metrics = (
            _concat_or_empty([_read_metrics_zip(path) for path in metrics_archives])
            .sort("timestamp")
            .unique(subset=["timestamp"], keep="last")
            # Forward-fill aplicado sobre todos los archivos juntos:
            # si un día tiene ratios vacíos pero el día anterior tenía valores,
            # los valores del día anterior se propagan correctamente.
            .with_columns([pl.col(c).forward_fill() for c in METRICS_COLUMNS])
        )
        metrics = _filter_range(metrics, start, end)
        if metrics.height > 0:
            n_ratio = metrics["toptrader_count_long_short_ratio_5m"].drop_nulls().len()
            if n_ratio == 0:
                console.print("  [yellow]⚠[/] ratios long/short sin datos para este rango (Binance no los registraba)")
            ohlcv = _join_feature_asof(ohlcv, metrics)
        else:
            console.print("  [yellow]⚠[/] sin datos de metrics 5m — columnas en null")
    else:
        console.print("  [yellow]⚠[/] metrics no disponibles para este rango — columnas en null")
    ohlcv = _ensure_columns(ohlcv, METRICS_COLUMNS)

    ohlcv = _add_predicted_funding_rate_1m(ohlcv)
    ohlcv = _select_feature_columns(ohlcv, console)
    return ohlcv.sort("timestamp")


def _output_path(vision_dir: Path, fmt: str) -> Path:
    return vision_dir / f"{BTC_ASSET}_ohlcv_1m.{fmt}"


def _cleanup_generated_btc_files(vision_dir: Path, selected_formats: list[str], console: Console) -> None:
    expected = {_output_path(vision_dir, fmt).name for fmt in selected_formats}
    generated_prefixes = (
        "BTC_ohlcv_1m",
        "BTC_aggtrades_1m",
        "BTC_premium_index_1m",
        "BTC_metrics_5m",
        "BTC_funding_rate",
    )

    removed: list[str] = []
    if vision_dir.exists():
        for path in vision_dir.iterdir():
            if not path.is_file():
                continue
            if path.name in expected:
                continue
            if any(path.stem.startswith(prefix) for prefix in generated_prefixes):
                path.unlink()
                removed.append(path.name)

    stale_dirs = (
        vision_dir / "raw" / "binance",
        vision_dir / "binance" / BTC_SYMBOL,
    )
    for stale_dir in stale_dirs:
        if stale_dir.exists():
            shutil.rmtree(stale_dir)
            removed.append(str(stale_dir.relative_to(vision_dir)))

    for parent in (vision_dir / "raw", vision_dir / "binance"):
        if parent.exists() and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()

    if removed:
        console.print(f"[dim]Limpieza BTC previa:[/] {', '.join(removed)}")


def _round_sig_decimals(df: pl.DataFrame, sig: int = 4) -> pl.DataFrame:
    """Redondea cada columna Float64 a `sig` decimales significativos,
    ignorando los ceros a la izquierda del primer dígito no-cero.
    Ejemplos: 100.12345678 → 100.1235 | 0.0000234 → 0.0000234 | 9.303344 → 9.3033
    """
    import numpy as np

    float_cols = [c for c, t in df.schema.items() if t == pl.Float64]
    if not float_cols:
        return df

    new_cols = []
    for col in float_cols:
        arr = df[col].to_numpy(allow_copy=True)
        result = arr.copy()
        finite = np.isfinite(arr)
        frac = np.abs(arr - np.floor(arr))
        has_frac = finite & (frac > 1e-15)
        if has_frac.any():
            frac_v = frac[has_frac]
            with np.errstate(divide='ignore', invalid='ignore'):
                places = np.clip(
                    (sig - 1 - np.floor(np.log10(frac_v))).astype(int), 0, 15
                )
            for p in np.unique(places):
                idx = np.where(has_frac)[0][places == p]
                result[idx] = np.round(arr[idx], int(p))
        new_cols.append(pl.Series(col, result, dtype=pl.Float64))

    return df.with_columns(new_cols)


def _write_outputs(df: pl.DataFrame, vision_dir: Path, formats: list[str], console: Console) -> list[Path]:
    vision_dir.mkdir(parents=True, exist_ok=True)
    df = _round_sig_decimals(df)
    paths: list[Path] = []

    for fmt in formats:
        path = _output_path(vision_dir, fmt)
        console.print(f"[cyan]escribiendo[/] {path}")
        if fmt == "csv":
            df.write_csv(path)
        elif fmt == "xlsx":
            # Excel no soporta timestamps con timezone: quitar UTC antes de escribir
            df_xls = df.with_columns(
                pl.col("timestamp").dt.replace_time_zone(None)
            ) if "timestamp" in df.columns else df
            df_xls.write_excel(
                path,
                table_style="Table Style Medium 9",
                autofit=True,
                freeze_panes=(1, 0),
                float_precision=8,
            )
        elif fmt == "parquet":
            df.write_parquet(path)
        elif fmt == "feather":
            df.write_ipc(path)
        else:
            raise ValueError(f"Formato no soportado: {fmt}")
        paths.append(path)

    return paths


def _print_summary(df: pl.DataFrame, paths: list[Path], console: Console) -> None:
    from rich.panel import Panel
    from rich.rule import Rule

    console.print(Rule("[bold cyan]Resultado[/]", style="dim"))

    table = Table(show_header=True, header_style="bold dim", box=None, padding=(0, 2))
    table.add_column("", style="dim", justify="right")
    table.add_column("", style="white")

    start_ts = str(df["timestamp"][0])[:19] if df.height else "-"
    end_ts = str(df["timestamp"][-1])[:19] if df.height else "-"

    days = (df["timestamp"][-1] - df["timestamp"][0]).days if df.height > 1 else 0

    table.add_row("filas",    f"[bold green]{df.height:,}[/]  [dim]({days:,} días)[/]")
    table.add_row("inicio",   start_ts)
    table.add_row("fin",      end_ts)
    table.add_row("columnas", "  ".join(df.columns))

    for path in paths:
        size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0
        table.add_row(path.suffix.lstrip("."), f"[bold]{path}[/]  [dim]{size_mb:.1f} MB[/]")

    console.print(Panel(table, title="[bold]BTCUSDT 1m[/]", border_style="cyan", padding=(1, 2)))
    console.print(Rule(style="dim"))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    default_start, default_end = _default_date_range()
    parser = argparse.ArgumentParser(
        description="Descarga BTCUSDT futures 1m desde Binance Data Vision y genera BTC_ohlcv_1m.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ajuste rápido: edita FECHA_INICIO / FECHA_FIN / VISION_DIR al inicio del archivo.\n"
            "Salida en feather/parquet directo a datos/. Cambia FORMATOS y VISION_DIR al inicio del archivo.\n"
            "Ejemplos:\n"
            "  python descargar_vision.py\n"
            "  python descargar_vision.py --start 2023-01-01 --end 2023-12-31\n"
            "  python descargar_vision.py --workers 12\n"
        ),
    )
    parser.add_argument("--vision-dir", default=str(VISION_DIR), help="Carpeta de salida")
    parser.add_argument("--start", default=default_start, help="Fecha inicial YYYY-MM-DD")
    parser.add_argument("--end", default=default_end, help="Fecha final YYYY-MM-DD")
    parser.add_argument("--workers", type=int, default=WORKERS,
                        help="Descargas paralelas simultáneas (default: %(default)s)")
    parser.add_argument("--no-checksum", action="store_true", help="No valida CHECKSUM de zips mensuales")
    parser.add_argument("--no-daily-fallback", action="store_true", help="No intenta zips diarios si falta un mensual")
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="No borra artefactos BTC antiguos generados por versiones anteriores",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    from rich.panel import Panel
    from rich.rule import Rule

    args = parse_args(argv)
    console = Console()

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    if end < start:
        raise ValueError("--end no puede ser anterior a --start")

    vision_dir = Path(args.vision_dir)

    console.print()
    console.print(Panel(
        f"  Activo  : [bold cyan]BTCUSDT — Futures 1m[/]\n"
        f"  Rango   : [bold]{start:%Y-%m-%d}[/] → [bold]{end:%Y-%m-%d}[/]\n"
        f"  Salida  : [bold]{vision_dir / 'BTC_ohlcv_1m.feather'}[/]\n"
        f"  Workers : [dim]{args.workers} descargas paralelas[/]",
        title="[bold]MODELOX Vision[/]",
        border_style="cyan",
        padding=(0, 2),
    ))
    console.print()

    formats = _parse_formats(FORMATOS)
    if not args.no_clean:
        _cleanup_generated_btc_files(vision_dir, formats, console)

    with tempfile.TemporaryDirectory(prefix="modelox_vision_btc_") as tmp:
        tmp_dir = Path(tmp)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            df = _build_btc_dataset(
                start=start,
                end=end,
                tmp_dir=tmp_dir,
                console=console,
                progress=progress,
                verify_checksum=not args.no_checksum,
                daily_fallback=not args.no_daily_fallback,
                workers=args.workers,
            )

    console.print(Rule("[bold cyan]Guardando[/]", style="dim"))
    paths = _write_outputs(df, vision_dir, formats, console)
    _print_summary(df, paths, console)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
        raise SystemExit(130)
