#!/usr/bin/env python3
"""
COR.py — Análisis de correlación entre activos (MODELOX)

Escanea automáticamente nuevos_datos/ y data/ohlcv/ buscando archivos
.feather, .parquet y .csv.  Presenta un menú para elegir los dos activos
y ejecuta un análisis profesional de correlación.

Uso:
    python COR.py                          # Menú interactivo
    python COR.py BTC ETH                  # Directo
    python COR.py BTC ETH 1000             # Con ventana rodante
    python COR.py archivo1.feather archivo2.csv  # Rutas manuales
"""

from __future__ import annotations

import sys
import os
import warnings
import itertools
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

warnings.filterwarnings("ignore", category=FutureWarning)

console = Console()

# ─── Carpetas donde buscar datos ─────────────────────────────────────────────
_DATA_DIRS = ["nuevos_datos", "data/ohlcv"]
_SUPPORTED_EXT = {".feather", ".parquet", ".csv"}

# Resolución de análisis: resampleamos datos 1m a 1h para que sea manejable
# y estadísticamente más limpio (menos ruido microestructural).
_RESAMPLE_TF = "1h"

# ─── Activos habilitados (True = incluir, False = excluir) ───────────────────
ACTIVOS_HABILITADOS = {
    "BTC":     True,
    "ETH":     True,
    "GOLD":    True,
    "SILVER":  True,
    "SP500":   True,
    "NASDAQ":  True,
    "BIST100": False,
}


# ═════════════════════════════════════════════════════════════════════════════
# CARGA Y DESCUBRIMIENTO DE DATOS
# ═════════════════════════════════════════════════════════════════════════════

def _scan_data_files() -> Dict[str, Path]:
    """Escanea carpetas de datos y devuelve {ACTIVO: path_mejor_archivo}."""
    found: Dict[str, List[Tuple[Path, int]]] = {}

    for d in _DATA_DIRS:
        dpath = Path(d)
        if not dpath.exists():
            continue
        for f in dpath.iterdir():
            if f.suffix.lower() not in _SUPPORTED_EXT:
                continue
            if "ohlcv" not in f.stem.lower() and "ohlc" not in f.stem.lower():
                continue
            activo = f.stem.split("_")[0].upper()
            # Prioridad: feather > parquet > csv ; 1m > 5m > 1h
            prio = 0
            if f.suffix.lower() == ".feather":
                prio += 100
            elif f.suffix.lower() == ".parquet":
                prio += 50
            if "1m" in f.stem:
                prio += 10
            elif "5m" in f.stem:
                prio += 5
            elif "1h" in f.stem:
                prio += 3
            found.setdefault(activo, []).append((f, prio))

    # Quedarse con el mejor archivo por activo (solo habilitados)
    best: Dict[str, Path] = {}
    for activo, files in found.items():
        if not ACTIVOS_HABILITADOS.get(activo, True):
            continue
        files.sort(key=lambda x: x[1], reverse=True)
        best[activo] = files[0][0]
    return best


def _load(path: Path, resample: bool = True) -> pd.DataFrame:
    """Carga un archivo de datos (feather/parquet/csv) y devuelve DataFrame indexado por timestamp.
    Si resample=False se devuelve en resolución original (1m/5m)."""
    ext = path.suffix.lower()
    if ext == ".feather":
        df = pd.read_feather(path)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        console.print(f"[red]Formato no soportado: {ext}[/red]")
        sys.exit(1)

    # Normalizar timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.set_index("timestamp", inplace=True)
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df.set_index("date", inplace=True)

    df.sort_index(inplace=True)

    if resample:
        df = _resample_if_needed(df)
    return df


def _resample_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """Resamplea a 1h si el DataFrame tiene más de 500k filas."""
    if len(df) > 500_000:
        ohlcv = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
        if "volume" in df.columns:
            ohlcv["volume"] = "sum"
        df = df.resample(_RESAMPLE_TF).agg(ohlcv).dropna(subset=["close"])
    return df


def _extraer_nombre(path: Path) -> str:
    return path.stem.split("_")[0].upper()


# ═════════════════════════════════════════════════════════════════════════════
# MOTOR DE ANÁLISIS
# ═════════════════════════════════════════════════════════════════════════════

def _retornos_log(close: pd.Series) -> pd.Series:
    """Retornos logarítmicos — mejores propiedades estadísticas que pct_change."""
    return np.log(close / close.shift(1))


def analizar(df1: pd.DataFrame, df2: pd.DataFrame,
             n1: str, n2: str, ventana: int = 0) -> Dict:
    """Análisis completo de correlación entre dos activos."""

    # Alinear por timestamps comunes
    idx = df1.index.intersection(df2.index).sort_values()
    if len(idx) < 50:
        console.print("[red]Insuficientes datos comunes (< 50 velas)[/red]")
        sys.exit(1)

    c1 = df1.loc[idx, "close"]
    c2 = df2.loc[idx, "close"]

    r1 = _retornos_log(c1).dropna()
    r2 = _retornos_log(c2).dropna()

    # Realinear después de dropna
    common = r1.index.intersection(r2.index)
    r1, r2 = r1.loc[common], r2.loc[common]
    N = len(r1)

    res: Dict = {
        "n1": n1, "n2": n2, "N": N,
        "start": common[0], "end": common[-1],
    }

    # ── 1. CORRELACIONES (Pearson + Spearman + Kendall) ──────────────────
    res["pearson"] = r1.corr(r2)
    res["spearman"] = r1.corr(r2, method="spearman")
    res["kendall"] = r1.corr(r2, method="kendall")

    # ── 2. BETA GLOBAL y SEPARADO (alcista / bajista) ────────────────────
    var1 = r1.var()
    res["beta"] = r1.cov(r2) / var1 if var1 > 0 else 0.0

    mask_up = r1 > 0
    mask_dn = r1 < 0
    r1u, r2u = r1[mask_up], r2[mask_up]
    r1d, r2d = r1[mask_dn], r2[mask_dn]

    res["beta_up"] = r1u.cov(r2u) / r1u.var() if len(r1u) > 5 and r1u.var() > 0 else 0.0
    res["beta_dn"] = r1d.cov(r2d) / r1d.var() if len(r1d) > 5 and r1d.var() > 0 else 0.0

    # ── 3. COINCIDENCIA DIRECCIONAL ──────────────────────────────────────
    res["pct_same_dir"] = ((r1 > 0) == (r2 > 0)).mean() * 100
    res["pct_2up_when_1up"] = (r2[mask_up] > 0).mean() * 100 if mask_up.sum() > 0 else 0
    res["pct_2dn_when_1dn"] = (r2[mask_dn] < 0).mean() * 100 if mask_dn.sum() > 0 else 0

    # ── 3b. ANÁLISIS BIDIRECCIONAL: los 4 escenarios con % real ───────────
    # Convertir retornos log a % para que sea intuitivo
    p1 = (np.exp(r1) - 1) * 100  # retornos % de activo 1
    p2 = (np.exp(r2) - 1) * 100  # retornos % de activo 2

    m1_up = p1 > 0
    m1_dn = p1 < 0
    m2_up = p2 > 0
    m2_dn = p2 < 0

    def _escenario(mask_ref, mask_dep, ref, dep):
        """Calcula estadísticas de un escenario."""
        sub_ref = ref[mask_ref]
        sub_dep = dep[mask_ref]
        n_total = mask_ref.sum()
        n_dep_same = mask_dep[mask_ref].sum()
        return {
            "n": int(n_total),
            "pct_sigue": (n_dep_same / n_total * 100) if n_total > 0 else 0,
            "ref_mean": sub_ref.mean() if n_total > 0 else 0,
            "dep_mean": sub_dep.mean() if n_total > 0 else 0,
            "dep_median": sub_dep.median() if n_total > 0 else 0,
            "dep_std": sub_dep.std() if n_total > 0 else 0,
            # Cuando ambos se mueven igual: ¿quién se mueve MÁS?
            "dep_more": (abs(sub_dep) > abs(sub_ref)).mean() * 100 if n_total > 0 else 0,
            "ratio_mean": (sub_dep.mean() / sub_ref.mean()) if n_total > 0 and sub_ref.mean() != 0 else 0,
        }

    # A sube → B sube/baja
    res["A_up"] = _escenario(m1_up, m2_up, p1, p2)
    # A baja → B sube/baja
    res["A_dn"] = _escenario(m1_dn, m2_dn, p1, p2)
    # B sube → A sube/baja
    res["B_up"] = _escenario(m2_up, m1_up, p2, p1)
    # B baja → A sube/baja
    res["B_dn"] = _escenario(m2_dn, m1_dn, p2, p1)

    # ── 4. TAIL DEPENDENCE (extremos) ────────────────────────────────────
    # ¿La correlación cambia durante movimientos extremos (>2σ)?
    sigma1 = r1.std()
    mask_crash = r1 < -2 * sigma1
    mask_spike = r1 > 2 * sigma1

    n_crash = mask_crash.sum()
    n_spike = mask_spike.sum()

    if n_crash >= 10:
        res["corr_crash"] = r1[mask_crash].corr(r2[mask_crash])
        res["pct_2dn_in_crash"] = (r2[mask_crash] < 0).mean() * 100
    else:
        res["corr_crash"] = np.nan
        res["pct_2dn_in_crash"] = np.nan

    if n_spike >= 10:
        res["corr_spike"] = r1[mask_spike].corr(r2[mask_spike])
        res["pct_2up_in_spike"] = (r2[mask_spike] > 0).mean() * 100
    else:
        res["corr_spike"] = np.nan
        res["pct_2up_in_spike"] = np.nan

    res["n_crash"] = n_crash
    res["n_spike"] = n_spike

    # ── 5. LAG ANALYSIS (¿quién lidera?) ─────────────────────────────────
    # Correlación cruzada con retardos de -5 a +5
    lags = range(-5, 6)
    lag_corrs = {}
    for lag in lags:
        shifted = r2.shift(-lag)
        valid = r1.notna() & shifted.notna()
        if valid.sum() > 50:
            lag_corrs[lag] = r1[valid].corr(shifted[valid])
        else:
            lag_corrs[lag] = np.nan

    res["lag_corrs"] = lag_corrs
    best_lag = max(lag_corrs, key=lambda k: abs(lag_corrs[k]) if not np.isnan(lag_corrs[k]) else 0)
    res["best_lag"] = best_lag
    res["best_lag_corr"] = lag_corrs[best_lag]

    # ── 6. VOLATILIDAD COMPARATIVA ───────────────────────────────────────
    res["vol1"] = r1.std() * 100
    res["vol2"] = r2.std() * 100
    res["vol_ratio"] = r2.std() / r1.std() if r1.std() > 0 else 0
    res["ret_mean1"] = r1.mean() * 100
    res["ret_mean2"] = r2.mean() * 100

    # ── 7. ROLLING CORRELATION (si se pide ventana) ──────────────────────
    if ventana > 0 and ventana < N:
        rolling_corr = r1.rolling(ventana).corr(r2).dropna()
        res["roll_mean"] = rolling_corr.mean()
        res["roll_std"] = rolling_corr.std()
        res["roll_min"] = rolling_corr.min()
        res["roll_max"] = rolling_corr.max()
        res["roll_current"] = rolling_corr.iloc[-1] if len(rolling_corr) > 0 else np.nan
        # Percentiles
        res["roll_p25"] = rolling_corr.quantile(0.25)
        res["roll_p75"] = rolling_corr.quantile(0.75)
        res["roll_series"] = rolling_corr  # Para detalle
    else:
        res["roll_mean"] = None

    return res


# ═════════════════════════════════════════════════════════════════════════════
# PRESENTACIÓN
# ═════════════════════════════════════════════════════════════════════════════

def _cc(v: float, invert: bool = False) -> str:
    """Colorea un valor de correlación/beta."""
    if np.isnan(v):
        return "[dim]N/A[/dim]"
    if invert:
        v = -v
    if v > 0.7:
        return f"[bold green]{v:+.4f}[/bold green]"
    if v > 0.3:
        return f"[green]{v:+.4f}[/green]"
    if v > -0.3:
        return f"[yellow]{v:+.4f}[/yellow]"
    if v > -0.7:
        return f"[red]{v:+.4f}[/red]"
    return f"[bold red]{v:+.4f}[/bold red]"


def mostrar(r: Dict) -> None:
    n1, n2 = r["n1"], r["n2"]

    console.print()
    console.print(Panel.fit(
        f"[bold]{n1}  vs  {n2}[/bold]\n"
        f"[dim]{r['N']:,} velas  ·  {str(r['start'])[:10]} → {str(r['end'])[:10]}[/dim]",
        title="CORRELACIÓN", border_style="cyan",
    ))

    # ── CORRELACIONES ────────────────────────────────────────────────────
    t = Table(title="Correlaciones", box=box.SIMPLE_HEAVY, show_edge=False)
    t.add_column("Método", style="dim")
    t.add_column("Valor", justify="right")
    t.add_column("Interpretación")

    for name, val in [("Pearson", r["pearson"]),
                      ("Spearman", r["spearman"]),
                      ("Kendall", r["kendall"])]:
        interp = (
            "Fuerte +" if val > 0.7 else
            "Moderada +" if val > 0.3 else
            "Débil" if val > -0.3 else
            "Moderada −" if val > -0.7 else
            "Fuerte −"
        )
        t.add_row(name, _cc(val), interp)
    console.print(t)

    # ── ANÁLISIS BIDIRECCIONAL (la tabla clave) ──────────────────────────
    tb = Table(
        title=f"¿Cuánto se mueve uno cuando el otro sube/baja?",
        box=box.SIMPLE_HEAVY, show_edge=False,
    )
    tb.add_column("Escenario", style="dim")
    tb.add_column("Veces", justify="right")
    tb.add_column("% sigue", justify="right")
    tb.add_column("Ref. medio", justify="right")
    tb.add_column("Dep. medio", justify="right")
    tb.add_column("Ratio", justify="right")
    tb.add_column("Dep. > Ref.", justify="right")

    for label, key, ref_name, dep_name, ref_dir, dep_color in [
        (f"{n1} ↑", "A_up", n1, n2, "green", "green"),
        (f"{n1} ↓", "A_dn", n1, n2, "red", "red"),
        (f"{n2} ↑", "B_up", n2, n1, "green", "green"),
        (f"{n2} ↓", "B_dn", n2, n1, "red", "red"),
    ]:
        e = r[key]
        pct_s = e["pct_sigue"]
        cs = "green" if pct_s > 60 else "yellow" if pct_s > 50 else "red"
        ratio = e["ratio_mean"]
        cr = "green" if abs(ratio) > 1 else "yellow"
        more = e["dep_more"]
        cm = "green" if more > 50 else "red"
        tb.add_row(
            f"[{ref_dir}]{label}[/{ref_dir}]   →  {dep_name}?",
            f"{e['n']:,}",
            f"[{cs}]{pct_s:.1f}%[/{cs}]",
            f"[{ref_dir}]{e['ref_mean']:+.4f}%[/{ref_dir}]",
            f"[{dep_color}]{e['dep_mean']:+.4f}%[/{dep_color}]",
            f"[{cr}]{ratio:.2f}x[/{cr}]",
            f"[{cm}]{more:.1f}%[/{cm}]",
        )

    console.print(tb)
    console.print("  [dim]Ratio = dep_medio / ref_medio · Dep.>Ref. = % veces el dependiente se mueve más[/dim]")

    # ── BETA ─────────────────────────────────────────────────────────────
    t2 = Table(title=f"Beta ({n2} respecto a {n1})", box=box.SIMPLE_HEAVY, show_edge=False)
    t2.add_column("Contexto", style="dim")
    t2.add_column("Beta", justify="right")
    t2.add_column("Significado")

    beta_g = r["beta"]
    beta_u = r["beta_up"]
    beta_d = r["beta_dn"]

    t2.add_row("Global", f"{beta_g:+.4f}",
               f"{n2} se mueve {abs(beta_g):.2f}x por cada 1x de {n1}")
    t2.add_row(f"Cuando {n1} sube", f"[green]{beta_u:+.4f}[/green]",
               f"{n2} sube {abs(beta_u):.2f}x")
    t2.add_row(f"Cuando {n1} baja", f"[red]{beta_d:+.4f}[/red]",
               f"{n2} baja {abs(beta_d):.2f}x")

    # Asimetría
    if beta_u > 0.01 and beta_d > 0.01 and abs(beta_u - beta_d) > 0.1:
        ratio = beta_d / beta_u if beta_u != 0 else 0
        if ratio > 1.3:
            t2.add_row("", "", f"[bold red]Asimétrico: caídas {ratio:.1f}x más reactivas[/bold red]")
        elif ratio < 0.7:
            t2.add_row("", "", f"[bold green]Asimétrico: subidas {1/ratio:.1f}x más reactivas[/bold green]")

    console.print(t2)

    # ── COINCIDENCIA DIRECCIONAL ─────────────────────────────────────────
    t3 = Table(title="Coincidencia direccional", box=box.SIMPLE_HEAVY, show_edge=False)
    t3.add_column("Escenario", style="dim")
    t3.add_column("%", justify="right")

    p_same = r["pct_same_dir"]
    p_2up = r["pct_2up_when_1up"]
    p_2dn = r["pct_2dn_when_1dn"]

    c_same = "green" if p_same > 60 else "yellow" if p_same > 50 else "red"
    c_2up = "green" if p_2up > 60 else "yellow" if p_2up > 50 else "red"
    c_2dn = "green" if p_2dn > 60 else "yellow" if p_2dn > 50 else "red"

    t3.add_row("Misma dirección", f"[{c_same}]{p_same:.1f}%[/{c_same}]")
    t3.add_row(f"{n1} ↑ → {n2} ↑", f"[{c_2up}]{p_2up:.1f}%[/{c_2up}]")
    t3.add_row(f"{n1} ↓ → {n2} ↓", f"[{c_2dn}]{p_2dn:.1f}%[/{c_2dn}]")

    console.print(t3)

    # ── TAIL DEPENDENCE (extremos >2σ) ───────────────────────────────────
    t4 = Table(title="Comportamiento en extremos (> 2σ)", box=box.SIMPLE_HEAVY, show_edge=False)
    t4.add_column("Escenario", style="dim")
    t4.add_column("N", justify="right")
    t4.add_column("Corr", justify="right")
    t4.add_column(f"{n2} sigue", justify="right")

    cc_crash = r["corr_crash"]
    cc_spike = r["corr_spike"]
    pdn = r["pct_2dn_in_crash"]
    pup = r["pct_2up_in_spike"]

    t4.add_row(
        f"Crash de {n1}",
        str(r["n_crash"]),
        _cc(cc_crash) if not np.isnan(cc_crash) else "[dim]—[/dim]",
        f"[red]{pdn:.0f}% baja[/red]" if not np.isnan(pdn) else "[dim]—[/dim]",
    )
    t4.add_row(
        f"Spike de {n1}",
        str(r["n_spike"]),
        _cc(cc_spike) if not np.isnan(cc_spike) else "[dim]—[/dim]",
        f"[green]{pup:.0f}% sube[/green]" if not np.isnan(pup) else "[dim]—[/dim]",
    )
    console.print(t4)

    # ── LAG ANALYSIS ─────────────────────────────────────────────────────
    t5 = Table(title="Análisis de retardo (¿quién lidera?)", box=box.SIMPLE_HEAVY, show_edge=False)
    t5.add_column("Lag", justify="center", style="dim")
    t5.add_column("Corr", justify="right")
    t5.add_column("", style="dim")

    lag_corrs = r["lag_corrs"]
    best_lag = r["best_lag"]

    for lag in sorted(lag_corrs.keys()):
        val = lag_corrs[lag]
        if np.isnan(val):
            continue
        label = ""
        if lag < 0:
            label = f"{n2} lidera {abs(lag)}h"
        elif lag > 0:
            label = f"{n1} lidera {lag}h"
        else:
            label = "simultáneo"

        marker = " ◄" if lag == best_lag and lag != 0 else ""
        style = "bold" if lag == best_lag else ""
        t5.add_row(
            f"[{style}]{lag:+d}[/{style}]" if style else f"{lag:+d}",
            _cc(val),
            f"[{style}]{label}{marker}[/{style}]" if style else label,
        )

    if best_lag != 0:
        if best_lag > 0:
            console.print(t5)
            console.print(f"  [bold]{n1} lidera a {n2} por ~{best_lag}h (corr {lag_corrs[best_lag]:+.4f})[/bold]")
        else:
            console.print(t5)
            console.print(f"  [bold]{n2} lidera a {n1} por ~{abs(best_lag)}h (corr {lag_corrs[best_lag]:+.4f})[/bold]")
    else:
        console.print(t5)
        console.print("  [dim]Movimiento simultáneo (sin retardo significativo)[/dim]")

    # ── VOLATILIDAD ──────────────────────────────────────────────────────
    t6 = Table(title="Volatilidad", box=box.SIMPLE_HEAVY, show_edge=False)
    t6.add_column("", style="dim")
    t6.add_column(n1, justify="right")
    t6.add_column(n2, justify="right")

    t6.add_row("σ (por vela)", f"{r['vol1']:.4f}%", f"{r['vol2']:.4f}%")
    t6.add_row("Retorno medio", f"{r['ret_mean1']:.5f}%", f"{r['ret_mean2']:.5f}%")
    t6.add_row("Ratio σ", "", f"{r['vol_ratio']:.2f}x")
    console.print(t6)

    # ── ROLLING CORRELATION (si hay) ─────────────────────────────────────
    if r["roll_mean"] is not None:
        t7 = Table(title="Correlación rodante", box=box.SIMPLE_HEAVY, show_edge=False)
        t7.add_column("Estadístico", style="dim")
        t7.add_column("Valor", justify="right")

        t7.add_row("Media", _cc(r["roll_mean"]))
        t7.add_row("±Std", f"±{r['roll_std']:.4f}")
        t7.add_row("Rango", f"[{r['roll_min']:+.4f}  →  {r['roll_max']:+.4f}]")
        t7.add_row("P25 / P75", f"{r['roll_p25']:+.4f} / {r['roll_p75']:+.4f}")
        t7.add_row("Actual", f"[bold]{r['roll_current']:+.4f}[/bold]" if not np.isnan(r["roll_current"]) else "[dim]—[/dim]")

        console.print(t7)

    # ── CONCLUSIÓN ───────────────────────────────────────────────────────
    lines = _conclusion(r)
    console.print()
    console.print(Panel("\n".join(lines), title="Conclusión", border_style="cyan"))
    console.print()


def _conclusion(r: Dict) -> List[str]:
    n1, n2 = r["n1"], r["n2"]
    p = r["pearson"]
    s = r["spearman"]
    lines = []

    # Tipo de correlación
    avg = (p + s) / 2
    if avg > 0.7:
        lines.append(f"[bold green]Correlación fuerte positiva ({avg:.2f})[/bold green] — se mueven juntos.")
    elif avg > 0.3:
        lines.append(f"[green]Correlación moderada ({avg:.2f})[/green] — relación parcial.")
    elif avg > -0.3:
        lines.append(f"[yellow]Correlación débil ({avg:.2f})[/yellow] — movimientos independientes.")
    elif avg > -0.7:
        lines.append(f"[red]Correlación negativa ({avg:.2f})[/red] — tendencia opuesta.")
    else:
        lines.append(f"[bold red]Fuerte correlación inversa ({avg:.2f})[/bold red] — hedge natural.")

    # Discrepancia Pearson vs Spearman (no-linealidad)
    if abs(p - s) > 0.1:
        lines.append(f"[yellow]Pearson≠Spearman ({p:.2f} vs {s:.2f})[/yellow] — relación no lineal detectada.")

    # Asimetría beta
    bu, bd = r["beta_up"], r["beta_dn"]
    if bu > 0.01 and bd > 0.01:
        ratio = bd / bu
        if ratio > 1.5:
            lines.append(f"[bold red]Asimétrico:[/bold red] {n2} reacciona {ratio:.1f}x más en caídas que en subidas.")
        elif ratio < 0.67:
            lines.append(f"[bold green]Asimétrico:[/bold green] {n2} reacciona {1/ratio:.1f}x más en subidas que en caídas.")

    # Tail dependence
    if not np.isnan(r["corr_crash"]):
        if r["corr_crash"] > r["pearson"] + 0.1:
            lines.append(f"[red]Contagio en crisis:[/red] correlación sube a {r['corr_crash']:.2f} durante crashes ({r['pct_2dn_in_crash']:.0f}% {n2} baja).")

    # Liderazgo
    bl = r["best_lag"]
    if bl != 0 and abs(r["best_lag_corr"]) > abs(r["pearson"]) * 0.9:
        leader = n1 if bl > 0 else n2
        lines.append(f"[cyan]Liderazgo:[/cyan] {leader} lidera por ~{abs(bl)}h.")

    # Rolling actual vs media
    if r["roll_mean"] is not None and not np.isnan(r.get("roll_current", np.nan)):
        diff = r["roll_current"] - r["roll_mean"]
        if diff > 0.15:
            lines.append(f"Correlación actual ({r['roll_current']:.2f}) SUPERIOR a la media ({r['roll_mean']:.2f}).")
        elif diff < -0.15:
            lines.append(f"Correlación actual ({r['roll_current']:.2f}) INFERIOR a la media ({r['roll_mean']:.2f}).")

    return lines or ["Sin patrones destacables."]


# ═════════════════════════════════════════════════════════════════════════════
# CONCLUSIÓN TEXTO PLANO (para PDF)
# ═════════════════════════════════════════════════════════════════════════════

def _conclusion_plain(r: Dict) -> List[str]:
    """Igual que _conclusion pero sin markup Rich."""
    n1, n2 = r["n1"], r["n2"]
    p, s = r["pearson"], r["spearman"]
    lines = []
    avg = (p + s) / 2
    if avg > 0.7:
        lines.append(f"Correlación fuerte positiva ({avg:.2f}) — se mueven juntos.")
    elif avg > 0.3:
        lines.append(f"Correlación moderada ({avg:.2f}) — relación parcial.")
    elif avg > -0.3:
        lines.append(f"Correlación débil ({avg:.2f}) — movimientos independientes.")
    elif avg > -0.7:
        lines.append(f"Correlación negativa ({avg:.2f}) — tendencia opuesta.")
    else:
        lines.append(f"Fuerte correlación inversa ({avg:.2f}) — hedge natural.")
    if abs(p - s) > 0.1:
        lines.append(f"Pearson≠Spearman ({p:.2f} vs {s:.2f}) — relación no lineal detectada.")
    bu, bd = r["beta_up"], r["beta_dn"]
    if bu > 0.01 and bd > 0.01:
        ratio = bd / bu
        if ratio > 1.5:
            lines.append(f"Asimétrico: {n2} reacciona {ratio:.1f}x más en caídas que en subidas.")
        elif ratio < 0.67:
            lines.append(f"Asimétrico: {n2} reacciona {1/ratio:.1f}x más en subidas que en caídas.")
    if not np.isnan(r["corr_crash"]):
        if r["corr_crash"] > r["pearson"] + 0.1:
            lines.append(f"Contagio en crisis: correlación sube a {r['corr_crash']:.2f} durante crashes ({r['pct_2dn_in_crash']:.0f}% {n2} baja).")
    bl = r["best_lag"]
    if bl != 0 and abs(r["best_lag_corr"]) > abs(r["pearson"]) * 0.9:
        leader = n1 if bl > 0 else n2
        lines.append(f"Liderazgo: {leader} lidera por ~{abs(bl)}h.")
    # Bidireccional
    if "A_up" in r:
        au = r["A_up"]
        ad = r["A_dn"]
        bu2 = r["B_up"]
        bd2 = r["B_dn"]
        lines.append(f"{n1} sube +{au['ref_mean']:.3f}% → {n2} {au['dep_mean']:+.3f}% (ratio {au['ratio_mean']:.2f}x, sigue {au['pct_sigue']:.0f}%)")
        lines.append(f"{n1} baja {ad['ref_mean']:.3f}% → {n2} {ad['dep_mean']:+.3f}% (ratio {ad['ratio_mean']:.2f}x, sigue {ad['pct_sigue']:.0f}%)")
        lines.append(f"{n2} sube +{bu2['ref_mean']:.3f}% → {n1} {bu2['dep_mean']:+.3f}% (ratio {bu2['ratio_mean']:.2f}x, sigue {bu2['pct_sigue']:.0f}%)")
        lines.append(f"{n2} baja {bd2['ref_mean']:.3f}% → {n1} {bd2['dep_mean']:+.3f}% (ratio {bd2['ratio_mean']:.2f}x, sigue {bd2['pct_sigue']:.0f}%)")

    return lines or ["Sin patrones destacables."]


# ═════════════════════════════════════════════════════════════════════════════
# SCAN SOLO nuevos_datos/ (feather)
# ═════════════════════════════════════════════════════════════════════════════

def _scan_nuevos_datos() -> Dict[str, Path]:
    """Escanea solo nuevos_datos/ para feather — usado en análisis completo."""
    result: Dict[str, Path] = {}
    dpath = Path("nuevos_datos")
    if not dpath.exists():
        return result
    for f in sorted(dpath.iterdir()):
        if f.suffix.lower() == ".feather" and "ohlcv" in f.stem.lower():
            activo = f.stem.split("_")[0].upper()
            if not ACTIVOS_HABILITADOS.get(activo, True):
                continue
            result[activo] = f
    return result


# ═════════════════════════════════════════════════════════════════════════════
# GENERADOR PDF PROFESIONAL
# ═════════════════════════════════════════════════════════════════════════════

def _color_corr(v: float) -> str:
    """Devuelve color hex según valor de correlación."""
    if np.isnan(v):
        return "#94a3b8"
    if v > 0.7:  return "#16a34a"
    if v > 0.3:  return "#65a30d"
    if v > -0.3: return "#ca8a04"
    if v > -0.7: return "#dc2626"
    return "#991b1b"


def _safe(v: float, fmt: str = "+.4f") -> str:
    """Formatea un float o devuelve — si es nan."""
    if np.isnan(v):
        return "—"
    return f"{v:{fmt}}"


def _generar_pdf_completo(all_results: List[Dict], output: Path) -> None:
    """Genera un PDF profesional centrado en los 4 escenarios bidireccionales."""
    from weasyprint import HTML

    ahora = datetime.now().strftime("%d/%m/%Y %H:%M")
    n_pairs = len(all_results)
    all_results.sort(key=lambda r: abs(r["pearson"]), reverse=True)

    activos = sorted(set(r["n1"] for r in all_results) | set(r["n2"] for r in all_results))
    n_activos = len(activos)

    # Matrices simétricas
    pearson_mx: Dict = {}
    for r in all_results:
        pearson_mx[(r["n1"], r["n2"])] = r["pearson"]
        pearson_mx[(r["n2"], r["n1"])] = r["pearson"]  # simétrica

    css = """
    @page { size: A4 landscape; margin: 16mm 14mm; }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
           font-size: 10px; color: #1e293b; line-height: 1.4; }

    .cover { page-break-after: always; display: flex; flex-direction: column;
             justify-content: center; align-items: center; height: 100%;
             text-align: center; }
    .cover h1 { font-size: 36px; font-weight: 800; color: #0f172a;
                letter-spacing: 2px; margin-bottom: 6px; }
    .cover .line { width: 80px; height: 3px; background: #0f172a;
                   margin: 0 auto 20px auto; }
    .cover .sub { font-size: 14px; color: #64748b; margin-bottom: 40px; }
    .cover .stats { font-size: 13px; color: #334155; line-height: 2.2; }
    .cover .footer { margin-top: 80px; font-size: 10px; color: #94a3b8; }

    h2 { font-size: 14px; font-weight: 700; color: #0f172a;
         border-bottom: 2px solid #0f172a; padding-bottom: 3px;
         margin: 18px 0 10px 0; }
    h3 { font-size: 11px; font-weight: 600; color: #334155; margin: 8px 0 4px 0; }

    table { width: 100%; border-collapse: collapse; margin-bottom: 8px; font-size: 9.5px; }
    th { background: #0f172a; color: #fff; padding: 4px 7px; text-align: left;
         font-weight: 600; font-size: 8.5px; text-transform: uppercase; letter-spacing: 0.4px; }
    td { padding: 3px 7px; border-bottom: 1px solid #e2e8f0; }
    tr:nth-child(even) td { background: #f8fafc; }

    .pair-block { page-break-inside: avoid; margin-bottom: 12px;
                  border: 1px solid #cbd5e1; border-radius: 6px;
                  padding: 10px 12px; background: #fff; }
    .pair-header { display: flex; justify-content: space-between;
                   align-items: center; margin-bottom: 6px;
                   border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; }
    .pair-title { font-size: 14px; font-weight: 700; color: #0f172a; }
    .pair-badge { font-size: 11px; font-weight: 700; padding: 3px 12px;
                  border-radius: 12px; color: #fff; }

    .scenarios { width: 100%; border-collapse: collapse; margin: 6px 0; }
    .scenarios th { font-size: 8px; padding: 4px 6px; }
    .scenarios td { padding: 4px 6px; font-size: 10px; text-align: center; }
    .scenarios td:first-child { text-align: left; font-weight: 600; }
    .sc-up { color: #16a34a; }
    .sc-dn { color: #dc2626; }
    .sc-bold { font-weight: 700; }

    .grid2 { display: flex; gap: 12px; }
    .grid2 > div { flex: 1; }

    .conclusion-box { background: #f0f9ff; border-left: 3px solid #0284c7;
                      padding: 5px 9px; margin-top: 6px; font-size: 9px;
                      color: #0c4a6e; line-height: 1.5; }

    .matrix-section { page-break-before: always; }
    .matrix-table th, .matrix-table td { text-align: center; padding: 5px; font-size: 10px; }
    .matrix-table td.diag { background: #e2e8f0; color: #94a3b8; }

    .summary-section { page-break-before: always; }
    """

    # ── PORTADA ──────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>{css}</style></head><body>
    <div class="cover">
        <h1>CORRELACIÓN</h1>
        <div class="line"></div>
        <div class="sub">Análisis bidireccional entre activos</div>
        <div class="stats">
            <strong>{n_activos}</strong> activos &nbsp;·&nbsp;
            <strong>{n_pairs}</strong> pares &nbsp;·&nbsp;
            <strong>{n_pairs * 4}</strong> escenarios<br><br>
            {' &nbsp;·&nbsp; '.join(activos)}
        </div>
        <div class="footer">MODELOX — {ahora}</div>
    </div>
    """

    # ── MATRIZ PEARSON ───────────────────────────────────────────────────
    html += '<div class="matrix-section">'
    html += '<h2>Matriz de correlación — Pearson</h2>'
    html += '<table class="matrix-table"><tr><th></th>'
    for a in activos:
        html += f'<th>{a}</th>'
    html += '</tr>'
    for a1 in activos:
        html += f'<tr><th style="text-align:left">{a1}</th>'
        for a2 in activos:
            if a1 == a2:
                html += '<td class="diag">—</td>'
            else:
                v = pearson_mx.get((a1, a2), np.nan)
                c = _color_corr(v)
                html += f'<td style="color:{c};font-weight:700">{_safe(v, "+.3f")}</td>'
        html += '</tr>'
    html += '</table></div>'

    # ── RANKING ──────────────────────────────────────────────────────────
    html += '<div class="summary-section">'
    html += '<h2>Ranking por correlación</h2>'
    html += '<table><tr>'
    html += '<th>#</th><th>Par</th><th>Pearson</th><th>Dir %</th>'
    html += '<th>A↑→B</th><th>A↓→B</th><th>B↑→A</th><th>B↓→A</th></tr>'
    for i, r in enumerate(all_results, 1):
        n1, n2 = r["n1"], r["n2"]
        cp = _color_corr(r["pearson"])
        au, ad = r["A_up"], r["A_dn"]
        bu, bd = r["B_up"], r["B_dn"]
        html += f"""<tr>
            <td style="font-weight:700">{i}</td>
            <td>{n1} ↔ {n2}</td>
            <td style="color:{cp};font-weight:700">{r['pearson']:+.3f}</td>
            <td>{r['pct_same_dir']:.0f}%</td>
            <td class="sc-up">{au['dep_mean']:+.3f}%</td>
            <td class="sc-dn">{ad['dep_mean']:+.3f}%</td>
            <td class="sc-up">{bu['dep_mean']:+.3f}%</td>
            <td class="sc-dn">{bd['dep_mean']:+.3f}%</td>
        </tr>"""
    html += '</table></div>'

    # ── DETALLE POR PAR ──────────────────────────────────────────────────
    html += '<div class="summary-section"><h2>Detalle por par — 4 escenarios</h2></div>'

    for idx, r in enumerate(all_results):
        n1, n2 = r["n1"], r["n2"]
        badge_color = _color_corr(r["pearson"])

        html += f"""<div class="pair-block">
        <div class="pair-header">
            <span class="pair-title">#{idx+1} &nbsp; {n1} ↔ {n2}</span>
            <span class="pair-badge" style="background:{badge_color}">
                Pearson {r['pearson']:+.3f}
            </span>
        </div>
        """

        # Tabla de 4 escenarios (la estrella del PDF)
        html += """<table class="scenarios">
        <tr><th>Escenario</th><th>Veces</th><th>% sigue</th>
            <th>Referencia</th><th>Dependiente</th><th>Ratio</th><th>Dep. &gt; Ref.</th></tr>
        """
        scenarios = [
            (f"{n1} ↑", n2, r["A_up"], "sc-up"),
            (f"{n1} ↓", n2, r["A_dn"], "sc-dn"),
            (f"{n2} ↑", n1, r["B_up"], "sc-up"),
            (f"{n2} ↓", n1, r["B_dn"], "sc-dn"),
        ]
        for label, dep_name, e, cls in scenarios:
            pct_s = e["pct_sigue"]
            cs = "#16a34a" if pct_s > 60 else "#ca8a04" if pct_s > 50 else "#dc2626"
            more = e["dep_more"]
            cm = "#16a34a" if more > 50 else "#dc2626"
            html += f"""<tr>
                <td>{label} → {dep_name}?</td>
                <td>{e['n']:,}</td>
                <td style="color:{cs};font-weight:700">{pct_s:.1f}%</td>
                <td class="{cls}">{e['ref_mean']:+.4f}%</td>
                <td class="{cls} sc-bold">{e['dep_mean']:+.4f}%</td>
                <td style="font-weight:700">{e['ratio_mean']:.2f}x</td>
                <td style="color:{cm}">{more:.1f}%</td>
            </tr>"""
        html += '</table>'

        # Correlaciones + Beta compacto
        html += '<div class="grid2">'
        html += f"""<div>
        <h3>Correlaciones</h3>
        <table>
            <tr><td>Pearson</td><td style="color:{_color_corr(r['pearson'])};font-weight:700">{r['pearson']:+.4f}</td>
                <td>Spearman</td><td>{r['spearman']:+.4f}</td>
                <td>Kendall</td><td>{r['kendall']:+.4f}</td></tr>
        </table>
        </div>
        <div>
        <h3>Beta · Extremos</h3>
        <table>
            <tr><td>Beta global</td><td>{r['beta']:+.3f}</td>
                <td>Crashes ({r['n_crash']})</td><td>{_safe(r['corr_crash'], '+.3f')}</td></tr>
            <tr><td>Beta ↑</td><td style="color:#16a34a">{r['beta_up']:+.3f}</td>
                <td>Spikes ({r['n_spike']})</td><td>{_safe(r['corr_spike'], '+.3f')}</td></tr>
            <tr><td>Beta ↓</td><td style="color:#dc2626">{r['beta_dn']:+.3f}</td>
                <td>Dir. coincidente</td><td>{r['pct_same_dir']:.1f}%</td></tr>
        </table>
        </div></div>
        """

        # Conclusión
        concl = _conclusion_plain(r)
        html += '<div class="conclusion-box">' + "<br>".join(concl) + '</div>'
        html += '</div>'

    html += '</body></html>'
    HTML(string=html).write_pdf(str(output))


# ═════════════════════════════════════════════════════════════════════════════
# ANÁLISIS COMPLETO (TODAS LAS COMBINACIONES)
# ═════════════════════════════════════════════════════════════════════════════

def _analizar_todas_combinaciones() -> None:
    """Analiza todas las combinaciones (no permutaciones) — cada par incluye análisis bidireccional."""
    archivos = _scan_nuevos_datos()
    if len(archivos) < 2:
        console.print("[red]Se necesitan al menos 2 activos en nuevos_datos/[/red]")
        return

    activos = sorted(archivos.keys())
    pares = list(itertools.combinations(activos, 2))  # 6C2 = 15 pares
    total = len(pares)

    console.print(f"  [bold]{len(activos)} activos[/bold]: {', '.join(activos)}")
    console.print(f"  [bold]{total} pares[/bold] (cada uno con 4 escenarios bidireccionales)")
    console.print()

    # Ventana rodante
    raw_v = input("  Ventana rodante (Enter = sin ventana, ej: 500): ").strip()
    ventana = int(raw_v) if raw_v.isdigit() and int(raw_v) > 0 else 0

    usar_1m = (0 < ventana <= 200)
    res_label = "1m" if usar_1m else "1h (resampleado)"
    console.print(f"  Resolución: [bold]{res_label}[/bold]")
    console.print()

    # Precargar todos los DataFrames
    console.print("  Cargando datos...")
    dfs: Dict[str, pd.DataFrame] = {}
    for a in activos:
        p = archivos[a]
        dfs[a] = _load(p, resample=not usar_1m)
        console.print(f"    {a}: {len(dfs[a]):,} velas")
    console.print()

    # Analizar cada par (una sola vez, el análisis ya es bidireccional)
    all_results: List[Dict] = []
    for i, (a1, a2) in enumerate(pares, 1):
        console.print(f"  [{i:2d}/{total}]  {a1} ↔ {a2} ... ", end="")
        try:
            r = analizar(dfs[a1], dfs[a2], a1, a2, ventana=ventana)
            all_results.append(r)
            p = r["pearson"]
            c = "green" if p > 0.3 else "yellow" if p > -0.3 else "red"
            console.print(f"[{c}]Pearson {p:+.4f}[/{c}]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    if not all_results:
        console.print("[red]No se completó ningún análisis[/red]")
        return

    # Generar PDF
    out_dir = Path("resultados")
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = out_dir / f"correlaciones_{ts}.pdf"

    console.print()
    console.print("  Generando PDF...", end=" ")
    _generar_pdf_completo(all_results, pdf_path)
    console.print(f"[bold green]{pdf_path}[/bold green]")
    console.print()


# ═════════════════════════════════════════════════════════════════════════════
# MENÚ INTERACTIVO
# ═════════════════════════════════════════════════════════════════════════════

def _menu_seleccion(disponibles: Dict[str, Path]) -> Optional[Tuple[Path, Path]]:
    """Muestra los activos disponibles y deja elegir dos. Devuelve None si elige opción 1."""
    activos = sorted(disponibles.keys())

    t = Table(title="Activos disponibles", box=box.SIMPLE, show_edge=False)
    t.add_column("#", style="bold", justify="right")
    t.add_column("Activo")
    t.add_column("Archivo", style="dim")
    t.add_column("Formato", style="dim")

    for i, a in enumerate(activos, 1):
        p = disponibles[a]
        t.add_row(str(i), a, str(p), p.suffix.lstrip("."))

    console.print(t)
    console.print()

    def _pick(prompt: str) -> Optional[Path]:
        raw = input(prompt).strip()
        if raw == "0":
            return None  # Señal para análisis completo
        # Por número de activo
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(activos):
                return disponibles[activos[idx]]
        except ValueError:
            pass
        # Por nombre
        raw_up = raw.upper()
        if raw_up in disponibles:
            return disponibles[raw_up]
        # Por ruta directa
        rp = Path(raw.replace("\\ ", " ").strip("'\""))
        if rp.exists():
            return rp
        console.print(f"[red]No encontrado: {raw}[/red]")
        sys.exit(1)

    p1 = _pick("  Opción (0 = TODAS, o # de activo): ")
    if p1 is None:
        return None

    p2 = _pick("  Activo 2 (# o nombre): ")
    if p2 is None:
        return None

    return p1, p2


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def _detectar_rango_comun(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Devuelve (inicio, fin) del rango donde ambos DataFrames tienen datos."""
    start = max(df1.index.min(), df2.index.min())
    end = min(df1.index.max(), df2.index.max())
    return start, end


def _pedir_rango(start: pd.Timestamp, end: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Muestra el rango común y deja al usuario elegir sub-rango o Enter para todo."""
    s_str = str(start)[:10]
    e_str = str(end)[:10]
    console.print(f"  Rango común: [bold]{s_str}[/bold] → [bold]{e_str}[/bold]")
    console.print("  [dim](Enter = rango completo, o escribe fechas)[/dim]")

    raw_s = input(f"  Fecha inicio [{s_str}]: ").strip()
    if raw_s:
        try:
            start = pd.Timestamp(raw_s, tz="UTC")
        except Exception:
            console.print(f"[yellow]Fecha no válida, usando {s_str}[/yellow]")

    raw_e = input(f"  Fecha fin    [{e_str}]: ").strip()
    if raw_e:
        try:
            end = pd.Timestamp(raw_e, tz="UTC")
        except Exception:
            console.print(f"[yellow]Fecha no válida, usando {e_str}[/yellow]")

    return start, end


def main():
    console.print()
    console.print("[bold]CORRELACIÓN ENTRE ACTIVOS[/bold]")
    console.print("━" * 40)
    console.print()

    disponibles = _scan_data_files()
    ventana = 0

    if len(sys.argv) >= 3:
        a1, a2 = sys.argv[1], sys.argv[2]
        ventana = int(sys.argv[3]) if len(sys.argv) >= 4 else 0

        if a1.upper() in disponibles and a2.upper() in disponibles:
            p1, p2 = disponibles[a1.upper()], disponibles[a2.upper()]
        else:
            p1, p2 = Path(a1), Path(a2)
    else:
        if not disponibles:
            console.print("[red]No se encontraron archivos de datos[/red]")
            sys.exit(1)

        # Mostrar opción 1
        nuevos = _scan_nuevos_datos()
        if len(nuevos) >= 2:
            n = len(nuevos)
            n_pares = n * (n - 1) // 2
            console.print(f"  [bold cyan][0][/bold cyan]  TODAS las combinaciones"
                          f" ({n} activos, {n_pares} pares × 4 escenarios) → PDF")
            console.print()

        sel = _menu_seleccion(disponibles)
        if sel is None:
            _analizar_todas_combinaciones()
            return

        p1, p2 = sel
        raw_v = input("  Ventana rodante (Enter = sin ventana, ej: 500): ").strip()
        ventana = int(raw_v) if raw_v.isdigit() and int(raw_v) > 0 else 0

    n1 = _extraer_nombre(p1)
    n2 = _extraer_nombre(p2)

    # ── Decidir resolución según ventana ─────────────────────────────────
    # Ventana ≤ 200 → mantener 1m (máxima resolución)
    # Ventana > 200 o sin ventana → resamplear a 1h
    usar_1m = (0 < ventana <= 200)
    res_label = "1m" if usar_1m else "1h (resampleado)"

    console.print(f"  Resolución: [bold]{res_label}[/bold]")
    console.print()

    console.print(f"  Cargando {n1} ({p1.suffix}) ...", end=" ")
    df1 = _load(p1, resample=not usar_1m)
    console.print(f"[green]{len(df1):,} velas[/green]")

    console.print(f"  Cargando {n2} ({p2.suffix}) ...", end=" ")
    df2 = _load(p2, resample=not usar_1m)
    console.print(f"[green]{len(df2):,} velas[/green]")

    # ── Detectar rango común y dejar elegir ──────────────────────────────
    console.print()
    rango_start, rango_end = _detectar_rango_comun(df1, df2)
    sel_start, sel_end = _pedir_rango(rango_start, rango_end)

    # Filtrar ambos DataFrames al rango seleccionado
    df1 = df1.loc[sel_start:sel_end]
    df2 = df2.loc[sel_start:sel_end]
    console.print(f"  Rango seleccionado: {str(sel_start)[:10]} → {str(sel_end)[:10]}")
    console.print(f"  {n1}: {len(df1):,} velas  ·  {n2}: {len(df2):,} velas")

    if len(df1) < 50 or len(df2) < 50:
        console.print("[red]Insuficientes datos en el rango seleccionado (< 50)[/red]")
        sys.exit(1)

    console.print()
    console.print("  Analizando...")
    console.print()

    r = analizar(df1, df2, n1, n2, ventana)
    mostrar(r)

    # Siempre generar PDF (también para par individual)
    out_dir = Path("resultados")
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = out_dir / f"correlacion_{n1}_{n2}_{ts}.pdf"
    console.print()
    console.print("  Generando PDF...", end=" ")
    _generar_pdf_completo([r], pdf_path)
    console.print(f"[bold green]{pdf_path}[/bold green]")
    console.print()


if __name__ == "__main__":
    main()
