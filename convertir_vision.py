from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

# =============================================================================
# FORMATO DE SALIDA
# =============================================================================
FORMATOS = "all"   # parquet | feather | all
# =============================================================================

SUPPORTED = {"parquet", "feather"}
SUPPORTED_INPUT = {".xlsx", ".csv"}


def _resolve_formats(raw: str) -> list[str]:
    raw = raw.strip().lower()
    if raw == "all":
        return ["parquet", "feather"]
    fmts = [f.strip() for f in raw.replace(";", ",").split(",") if f.strip()]
    bad = [f for f in fmts if f not in SUPPORTED]
    if bad:
        raise ValueError(f"Formato no soportado: {bad}. Usa: parquet | feather | all")
    seen: list[str] = []
    for f in fmts:
        if f not in seen:
            seen.append(f)
    return seen


def _read_source(src: Path, console: Console) -> pl.DataFrame:
    ext = src.suffix.lower()
    if ext == ".xlsx":
        console.print(Rule("[bold cyan]Leyendo Excel[/]", style="dim"))
        console.print(f"  [dim]{src.name}[/]")
        df = pl.read_excel(src, engine="calamine")
    else:
        console.print(Rule("[bold cyan]Leyendo CSV[/]", style="dim"))
        console.print(f"  [dim]{src.name}[/]")
        with src.open(encoding="utf-8", errors="replace") as _f:
            _first = _f.readline().strip()
        skip = 1 if _first == "sep=," else 0
        df = pl.read_csv(src, try_parse_dates=True, skip_rows=skip)
    console.print(f"  {df.height:,} filas  ·  {len(df.columns)} columnas")
    return df


def _write(df: pl.DataFrame, src: Path, fmt: str, console: Console) -> Path:
    dest = src.with_suffix(f".{fmt}")
    console.print(f"  [dim]→[/] {dest.name}", end="  ")
    if fmt == "parquet":
        df.write_parquet(dest)
    elif fmt == "feather":
        df.write_ipc(dest)
    size_mb = dest.stat().st_size / (1024 * 1024)
    console.print(f"[dim]{size_mb:.1f} MB[/]")
    return dest


def _summary(src: Path, outputs: list[Path], df: pl.DataFrame, console: Console) -> None:
    table = Table(show_header=True, header_style="bold dim", box=None, padding=(0, 2))
    table.add_column("", style="dim", justify="right")
    table.add_column("", style="white")

    src_mb = src.stat().st_size / (1024 * 1024)
    table.add_row("origen",   f"[dim]{src}[/]  [dim]{src_mb:.1f} MB[/]")
    table.add_row("filas",    f"[bold green]{df.height:,}[/]")
    table.add_row("columnas", str(len(df.columns)))

    for path in outputs:
        size_mb = path.stat().st_size / (1024 * 1024)
        ratio = src_mb / size_mb if size_mb > 0 else 0
        table.add_row(
            path.suffix.lstrip("."),
            f"[bold]{path.name}[/]  [dim]{size_mb:.1f} MB  ({ratio:.1f}x compresión)[/]",
        )

    console.print(Panel(table, title="[bold]Conversión completada[/]", border_style="cyan", padding=(1, 2)))


def main() -> int:
    console = Console()

    # Resolver archivo de entrada: argumento CLI o interactivo
    if len(sys.argv) >= 2:
        src = Path(sys.argv[1].strip().strip("'\""))
    else:
        console.print("[bold cyan]MODELOX — Convertir Vision[/]")
        console.print("Arrastra el archivo (.xlsx o .csv) a esta ventana o escribe la ruta:")
        try:
            raw = input("  Archivo: ").strip().strip("'\"")
        except (EOFError, KeyboardInterrupt):
            return 130
        src = Path(raw)

    if not src.exists():
        console.print(f"[red]Error:[/] no existe el archivo: {src}")
        return 1
    if src.suffix.lower() not in SUPPORTED_INPUT:
        allowed = "  |  ".join(sorted(SUPPORTED_INPUT))
        console.print(f"[red]Error:[/] formato de entrada no soportado: {src.suffix}  (soportados: {allowed})")
        return 1

    # Resolver formatos de salida: segundo argumento o FORMATOS del config
    fmt_raw = sys.argv[2] if len(sys.argv) >= 3 else FORMATOS
    try:
        formats = _resolve_formats(fmt_raw)
    except ValueError as exc:
        console.print(f"[red]Error:[/] {exc}")
        return 1

    console.print()
    console.print(Panel(
        f"  Origen  : [bold]{src}[/]\n"
        f"  Formatos: [bold cyan]{', '.join(formats)}[/]",
        title="[bold]MODELOX Vision — Convertir[/]",
        border_style="cyan",
        padding=(0, 2),
    ))
    console.print()

    df = _read_source(src, console)

    console.print(Rule("[bold cyan]Convirtiendo[/]", style="dim"))
    outputs: list[Path] = []
    for fmt in formats:
        outputs.append(_write(df, src, fmt, console))

    console.print(Rule("[bold cyan]Resultado[/]", style="dim"))
    _summary(src, outputs, df, console)
    console.print(Rule(style="dim"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrumpido.")
        raise SystemExit(130)
