"""
================================================================================
IA/DISPLAY.PY — RICH ULTRA-DETALLADO
================================================================================
Funciones de visualización con Rich:
  • Banner de bienvenida
  • Resumen de configuración y datos
  • Tabla de folds walk-forward
  • Log de trades (colorizado)
  • Panel de métricas por fold
  • Resumen final comparativo
================================================================================
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from IA.backtest import Trade, trades_to_dataframe

console = Console(highlight=False)


# =============================================================================
# HELPERS DE COLOR
# =============================================================================

def _green(v: float, fmt: str = ".2f") -> str:
    return f"[bold green]+{v:{fmt}}[/bold green]" if v > 0 else (
        f"[bold red]{v:{fmt}}[/bold red]" if v < 0 else f"[dim]{v:{fmt}}[/dim]"
    )


def _pct(v: float) -> str:
    return _green(v, ".1f") + "%"


def _usd(v: float) -> str:
    return _green(v, ".2f") + " $"


def _sqn_color(v: float) -> str:
    if v >= 3.0:
        return f"[bold green]{v:.3f}[/bold green] 🏆"
    elif v >= 2.0:
        return f"[green]{v:.3f}[/green] ✓"
    elif v >= 1.0:
        return f"[yellow]{v:.3f}[/yellow] ~"
    elif v >= 0:
        return f"[dim]{v:.3f}[/dim]"
    else:
        return f"[red]{v:.3f}[/red] ✗"


# =============================================================================
# 1. BANNER
# =============================================================================

def print_banner() -> None:
    banner = """
[bold blue]
  ██████╗ ██████╗ ██╗   ██╗    ████████╗██████╗  █████╗ ██████╗ ███████╗██████╗ 
 ██╔════╝ ██╔══██╗██║   ██║    ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔════╝██╔══██╗
 ██║  ███╗██████╔╝██║   ██║       ██║   ██████╔╝███████║██║  ██║█████╗  ██████╔╝
 ██║   ██║██╔══██╗██║   ██║       ██║   ██╔══██╗██╔══██║██║  ██║██╔══╝  ██╔══██╗
 ╚██████╔╝██║  ██║╚██████╔╝       ██║   ██║  ██║██║  ██║██████╔╝███████╗██║  ██║
  ╚═════╝ ╚═╝  ╚═╝ ╚═════╝        ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝[/bold blue]
[bold cyan]                    🤖  BTC GRU TRADING AI  |  MODELOX v1.0[/bold cyan]
[dim]                Pipeline ML Completo: GRU + Walk-Forward + Optuna + Backtest[/dim]
"""
    console.print(banner)
    console.print(Rule("[bold dim]Inicializando pipeline...[/bold dim]"))


# =============================================================================
# 2. CONFIGURACIÓN
# =============================================================================

def print_config(cfg_module) -> None:
    """Muestra panel de configuración con todos los parámetros clave."""
    tbl = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    tbl.add_column("Parámetro",  style="bold cyan",    width=28)
    tbl.add_column("Valor",      style="bright_white", width=20)
    tbl.add_column("Descripción", style="dim",          width=40)

    rows = [
        ("TIMEFRAME",        "1m",                            "Resolución de datos"),
        ("LOOKBACK",         str(cfg_module.LOOKBACK),        "Ventana histórica del modelo"),
        ("GRU_UNITS",        str(cfg_module.GRU_UNITS),       "Unidades por capa GRU"),
        ("N_GRU_LAYERS",     str(cfg_module.N_GRU_LAYERS),    "Capas GRU apiladas"),
        ("DROPOUT",          str(cfg_module.DROPOUT),         "Regularización dropout"),
        ("MAX_EPOCHS",       str(cfg_module.MAX_EPOCHS),      "Épocas máximas"),
        ("PATIENCE",         str(cfg_module.PATIENCE),        "Early stopping patience"),
        ("LEARNING_RATE",    f"{cfg_module.LEARNING_RATE:.0e}", "Adam LR inicial"),
        ("TP_USD",           f"${cfg_module.TP_USD:.0f}",     "Take Profit desde entrada"),
        ("SL_USD",           f"${cfg_module.SL_USD:.0f}",     "Stop Loss desde entrada"),
        ("PROB_THRESHOLD",   f"{cfg_module.PROB_THRESHOLD:.0%}", "Umbral LONG"),
        ("SHORT_THRESHOLD",  f"{cfg_module.SHORT_THRESHOLD:.0%}", "Umbral SHORT"),
        ("TRAIN_YEARS",      str(cfg_module.TRAIN_YEARS),     "Años de entrenamiento"),
        ("EMBARGO_DAYS",     str(cfg_module.EMBARGO_DAYS),    "Días de embargo"),
        ("VAL_YEARS",        str(cfg_module.VAL_YEARS),       "Años de validación"),
        ("STEP_MONTHS",      str(cfg_module.STEP_MONTHS),     "Paso entre folds"),
        ("SALDO_INICIAL",    f"${cfg_module.SALDO_INICIAL:.0f}", "Capital inicial"),
        ("APALANCAMIENTO",   f"{cfg_module.APALANCAMIENTO}x",  "Apalancamiento"),
        ("SALDO_USADO",      f"${cfg_module.SALDO_USADO:.0f}/trade", "Colateral por trade"),
        ("ALPHA_LOSS",       str(cfg_module.ALPHA_LOSS),      "Penalización pérdida asim."),
        ("QUICK_MODE",       str(cfg_module.QUICK_MODE),      "Modo rápido"),
    ]
    for name, val, desc in rows:
        tbl.add_row(name, val, desc)

    console.print(Panel(tbl, title="[bold white]⚙️  CONFIGURACIÓN DEL PIPELINE[/bold white]",
                        border_style="blue"))


# =============================================================================
# 3. RESUMEN DE DATOS
# =============================================================================

def print_data_summary(summary: dict) -> None:
    tbl = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    tbl.add_column("", style="bold cyan",    width=22)
    tbl.add_column("", style="bright_white", width=16)
    tbl.add_column("", style="bold cyan",    width=22)
    tbl.add_column("", style="bright_white", width=16)

    tbl.add_row(
        "Rango datos",     f"{summary['date_start']} → {summary['date_end']}",
        "Total velas",     f"{summary['n_total']:,}",
    )
    tbl.add_row(
        "Muestras válidas", f"{summary['n_valid']:,}",
        "Sin etiqueta",     f"{summary['n_skip']:,}",
    )
    tbl.add_row(
        "LONG (TP first)",  f"{summary['n_long']:,}  ({summary['pct_long']:.1f}%)",
        "SHORT (SL first)", f"{summary['n_short']:,}  ({summary['pct_short']:.1f}%)",
    )
    tbl.add_row(
        "Features",         str(summary["n_features"]),
        "Lookback",         f"{summary['lookback']} timesteps",
    )

    console.print(Panel(tbl, title="[bold white]📊 RESUMEN DE DATOS BTC 1M[/bold white]",
                        border_style="cyan"))


# =============================================================================
# 4. TABLA DE FOLDS
# =============================================================================

def print_folds_table(fold_summaries: list) -> None:
    tbl = Table(
        show_header=True, header_style="bold yellow",
        box=box.ROUNDED, border_style="yellow",
    )
    tbl.add_column("Fold",        width=5,  justify="center")
    tbl.add_column("Train Start", width=12)
    tbl.add_column("Train End",   width=12)
    tbl.add_column("Val Start",   width=12)
    tbl.add_column("Val End",     width=12)
    tbl.add_column("N Train",     width=9,  justify="right")
    tbl.add_column("N Val",       width=9,  justify="right")
    tbl.add_column("% LONG",      width=8,  justify="right")
    tbl.add_column("% SHORT",     width=8,  justify="right")

    for fs in fold_summaries:
        tbl.add_row(
            str(fs["fold_n"]),
            fs["train_start"], fs["train_end"],
            fs["val_start"],   fs["val_end"],
            f"{fs['n_train']:,}", f"{fs['n_val']:,}",
            f"{fs['vl_long_pct']:.1f}%",
            f"{fs['vl_short_pct']:.1f}%",
        )

    console.print(Panel(tbl, title="[bold white]📅 FOLDS WALK-FORWARD[/bold white]",
                        border_style="yellow"))


# =============================================================================
# 5. PANEL DE MÉTRICAS POR FOLD
# =============================================================================

def print_fold_result(fold_n: int, metrics: dict, equity_curve: np.ndarray) -> None:
    """Panel completo de resultados de un fold con backtest."""

    # ── Tabla de métricas principales ────────────────────────────────
    mt = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    mt.add_column("", style="bold cyan",    width=24)
    mt.add_column("", style="bright_white", width=16)
    mt.add_column("", style="bold cyan",    width=24)
    mt.add_column("", style="bright_white", width=16)

    roi_str    = _pct(metrics.get("roi", 0))
    dd_str     = f"[red]-{abs(metrics.get('max_drawdown',0)):.1f}%[/red]"
    sqn_str    = _sqn_color(metrics.get("sqn", 0))
    wr_str     = f"[{'green' if metrics.get('winrate',0)>52 else 'yellow'}]{metrics.get('winrate',0):.1f}%[/]"

    mt.add_row("💰 ROI",         roi_str,  "📉 Max Drawdown",    dd_str)
    mt.add_row("📈 SQN",         sqn_str,  "🎯 Win Rate",         wr_str)
    mt.add_row("📊 Trades Total", str(metrics.get("n_trades", 0)),
               "💵 PnL Total",    _usd(metrics.get("pnl_total", 0)))
    mt.add_row("🟢 LONG",
               f"{metrics.get('n_long',0)} (WR {metrics.get('wr_long',0):.1f}%)",
               "🔴 SHORT",
               f"{metrics.get('n_short',0)} (WR {metrics.get('wr_short',0):.1f}%)")
    mt.add_row("✅ Wins",         str(metrics.get("n_wins",  0)),
               "❌ Losses",       str(metrics.get("n_losses", 0)))
    mt.add_row("📤 TP hit",       str(metrics.get("n_tp",      0)),
               "🛑 SL hit",       str(metrics.get("n_sl",      0)))
    mt.add_row("⏱  Timeout",      str(metrics.get("n_timeout",  0)),
               "⏳ Dur. media",   f"{metrics.get('dur_mean_velas',0):.0f} velas")
    mt.add_row("💰 Saldo Inicial", f"${metrics.get('saldo_inicial',0):.2f}",
               "💰 Saldo Final",  f"${metrics.get('saldo_final',0):.2f}")
    mt.add_row("📈 Mejor Trade",   _usd(metrics.get("best_trade",  0)),
               "📉 Peor Trade",    _usd(metrics.get("worst_trade", 0)))
    mt.add_row("🏆 Racha Gana.",   str(metrics.get("max_win_streak",  0)),
               "😰 Racha Perd.",   str(metrics.get("max_loss_streak", 0)))
    mt.add_row("📊 Profit Factor",
               f"{metrics.get('profit_factor', float('nan')):.3f}" if not np.isnan(metrics.get('profit_factor', float('nan'))) else "N/A",
               "🎲 Payoff Ratio",
               f"{metrics.get('payoff_ratio', float('nan')):.3f}" if not np.isnan(metrics.get('payoff_ratio', float('nan'))) else "N/A")
    mt.add_row("📐 Expectancy",    f"${metrics.get('expectancy', 0):.3f}/trade",
               "⚡ Sharpe",        f"{metrics.get('sharpe', 0):.3f}")

    # ── Mini equity curve (ASCII) ─────────────────────────────────────
    eq_mini = _ascii_equity(equity_curve)

    console.print(Panel(
        mt,
        title=f"[bold white]📊 RESULTADOS FOLD {fold_n}[/bold white]",
        subtitle=f"[dim]{eq_mini}[/dim]",
        border_style="green" if metrics.get("roi", 0) > 0 else "red",
    ))


def _ascii_equity(eq: np.ndarray, width: int = 60) -> str:
    """Genera representación ASCII de la curva de equity."""
    if len(eq) < 2:
        return "—"
    mn, mx = eq.min(), eq.max()
    if mx == mn:
        return "─" * width
    norm   = (eq - mn) / (mx - mn)
    # Samplear puntos
    step   = max(1, len(norm) // width)
    pts    = norm[::step][:width]
    chars  = " ▁▂▃▄▅▆▇█"
    result = ""
    for p in pts:
        idx    = int(round(p * (len(chars) - 1)))
        result += chars[idx]
    return result


# =============================================================================
# 6. LOG DE TRADES
# =============================================================================

def print_trades_table(trades: List[Trade], max_rows: int = 30) -> None:
    """Tabla con el log de trades (últimos max_rows)."""
    if not trades:
        console.print("[dim]  Sin trades en este período.[/dim]")
        return

    tbl = Table(
        show_header=True, header_style="bold white",
        box=box.SIMPLE_HEAD, border_style="bright_black",
        row_styles=["", "dim"],
    )
    tbl.add_column("#",          width=4,  justify="right", style="dim")
    tbl.add_column("Tipo",       width=7,  justify="center")
    tbl.add_column("Entry Time", width=18)
    tbl.add_column("Entry $",    width=10, justify="right", style="cyan")
    tbl.add_column("Exit $",     width=10, justify="right", style="cyan")
    tbl.add_column("Exit",       width=9,  justify="center")
    tbl.add_column("PnL Bruto",  width=11, justify="right")
    tbl.add_column("Comisión",   width=9,  justify="right", style="dim")
    tbl.add_column("PnL Neto",   width=11, justify="right")
    tbl.add_column("Saldo",      width=10, justify="right", style="bright_white")
    tbl.add_column("Dur.",       width=6,  justify="right", style="dim")

    # Mostrar últimos max_rows
    display_trades = trades[-max_rows:]
    start_i = len(trades) - len(display_trades)

    for i, t in enumerate(display_trades, start=start_i + 1):
        tipo_str   = "[green]LONG[/green]"  if t.tipo == "LONG" else "[red]SHORT[/red]"
        exit_str   = (
            "[green]TP[/green]"      if t.exit_reason == "TP"      else
            "[red]SL[/red]"          if t.exit_reason == "SL"      else
            "[yellow]TIMEOUT[/yellow]"
        )
        pnl_br_str  = _usd(t.pnl_bruto)
        pnl_net_str = _usd(t.pnl_neto)
        com_str     = f"[dim]-{t.comision:.2f} $[/dim]"

        tbl.add_row(
            str(i),
            tipo_str,
            t.entry_time.strftime("%Y-%m-%d %H:%M") if t.entry_time else "—",
            f"{t.entry_price:,.2f}",
            f"{t.exit_price:,.2f}",
            exit_str,
            pnl_br_str, com_str, pnl_net_str,
            f"${t.saldo_despues:,.2f}",
            str(t.duracion_velas),
        )

    total = len(trades)
    title = f"[bold white]📋 LOG DE TRADES[/bold white]"
    subtitle = f"[dim]Mostrando {len(display_trades)} de {total} trades[/dim]"
    console.print(Panel(tbl, title=title, subtitle=subtitle, border_style="bright_black"))


# =============================================================================
# 7. RESUMEN WALK-FORWARD FINAL
# =============================================================================

def print_walkforward_summary(
    fold_metrics:  List[dict],
    fold_summaries: List[dict],
    all_trades:    List[Trade],
    equity_curves: List[np.ndarray],
    saldo_inicial: float,
) -> None:
    """Tabla comparativa de todos los folds + métricas consolidadas."""

    console.print()
    console.print(Rule("[bold yellow]🏁  RESUMEN WALK-FORWARD COMPLETO[/bold yellow]"))
    console.print()

    # ── Tabla por fold ───────────────────────────────────────────────
    tbl = Table(
        show_header=True, header_style="bold white",
        box=box.HEAVY_HEAD, border_style="yellow",
    )
    tbl.add_column("Fold",      width=5,  justify="center")
    tbl.add_column("Período Val.", width=24, justify="center")
    tbl.add_column("Trades",    width=7,  justify="right")
    tbl.add_column("Long",      width=6,  justify="right")
    tbl.add_column("Short",     width=6,  justify="right")
    tbl.add_column("WR %",      width=8,  justify="right")
    tbl.add_column("ROI %",     width=9,  justify="right")
    tbl.add_column("Drawdown",  width=10, justify="right")
    tbl.add_column("SQN",       width=8,  justify="right")
    tbl.add_column("PnL $",     width=10, justify="right")
    tbl.add_column("Sharpe",    width=8,  justify="right")

    all_pnl    = 0.0
    all_n      = 0
    all_n_long = 0
    all_n_short= 0

    for i, (m, fs) in enumerate(zip(fold_metrics, fold_summaries), 1):
        roi_v  = m.get("roi", 0)
        dd_v   = m.get("max_drawdown", 0)
        sqn_v  = m.get("sqn", 0)
        wr_v   = m.get("winrate", 0)
        pnl_v  = m.get("pnl_total", 0)
        sh_v   = m.get("sharpe", 0)

        roi_style = "[green]" if roi_v > 0 else "[red]"
        wr_style  = "[green]" if wr_v  > 52 else "[yellow]"

        tbl.add_row(
            str(i),
            f"{fs['val_start']} → {fs['val_end']}",
            str(m.get("n_trades", 0)),
            str(m.get("n_long",   0)),
            str(m.get("n_short",  0)),
            f"{wr_style}{wr_v:.1f}%[/]",
            f"{roi_style}{roi_v:+.1f}%[/]",
            f"[red]-{abs(dd_v):.1f}%[/red]",
            _sqn_color(sqn_v),
            _usd(pnl_v),
            f"{sh_v:.3f}",
        )
        all_pnl    += pnl_v
        all_n      += m.get("n_trades", 0)
        all_n_long += m.get("n_long",   0)
        all_n_short+= m.get("n_short",  0)

    # ── Fila de totales ───────────────────────────────────────────────
    all_roi = 100.0 * all_pnl / saldo_inicial
    all_wr  = 100.0 * sum(m.get("n_wins",0) for m in fold_metrics) / max(all_n, 1)

    tbl.add_section()
    tbl.add_row(
        "[bold]TOTAL[/bold]", "—",
        f"[bold]{all_n}[/bold]",
        f"[bold green]{all_n_long}[/bold green]",
        f"[bold red]{all_n_short}[/bold red]",
        f"[bold]{'[green]' if all_wr>52 else '[yellow]'}{all_wr:.1f}%[/][/bold]",
        f"[bold]{'+' if all_roi>0 else ''}{all_roi:.1f}%[/bold]",
        "—",
        "—",
        f"[bold]{_usd(all_pnl)}[/bold]",
        "—",
    )

    console.print(Panel(tbl,
        title="[bold white]🏆 COMPARATIVA WALK-FORWARD — TODOS LOS FOLDS[/bold white]",
        border_style="yellow"))

    # ── Equidad consolidada ───────────────────────────────────────────
    if equity_curves:
        all_eq = np.concatenate([eq[1:] for eq in equity_curves if len(eq) > 1])
        if len(all_eq) > 0:
            full_eq  = np.concatenate([[saldo_inicial], all_eq])
            eq_ascii = _ascii_equity(full_eq, width=80)
            console.print(Panel(
                f"[cyan]{eq_ascii}[/cyan]",
                title="[bold white]📈 EQUITY CURVE CONSOLIDADA[/bold white]",
                subtitle=f"[dim]${saldo_inicial:.0f} → ${full_eq[-1]:.2f} | Max: ${full_eq.max():.2f} | Min: ${full_eq.min():.2f}[/dim]",
                border_style="cyan",
            ))


# =============================================================================
# 8. RESUMEN DE OPTUNA
# =============================================================================

def print_optuna_result(best_params: dict, best_value: float) -> None:
    tbl = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    tbl.add_column("Parámetro", style="bold cyan",    width=25)
    tbl.add_column("Valor",     style="bright_white", width=20)

    for k, v in best_params.items():
        tbl.add_row(k, str(v))
    tbl.add_row("[bold]Val Loss Óptimo[/bold]", f"[green]{best_value:.6f}[/green]")

    console.print(Panel(tbl,
        title="[bold white]🔮 MEJORES HIPERPARÁMETROS (OPTUNA)[/bold white]",
        border_style="magenta"))


# =============================================================================
# 9. MODELO SUMMARY
# =============================================================================

def print_model_summary(summary: dict) -> None:
    tbl = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    tbl.add_column("", style="bold cyan",    width=22)
    tbl.add_column("", style="bright_white", width=18)
    tbl.add_column("", style="bold cyan",    width=22)
    tbl.add_column("", style="bright_white", width=18)

    tbl.add_row(
        "Arquitectura",   summary["arquitectura"],
        "Dispositivo",    summary["dispositivo"],
    )
    tbl.add_row(
        "Capas GRU",      str(summary["n_layers"]),
        "Unidades GRU",   str(summary["gru_units"]),
    )
    tbl.add_row(
        "Features",       str(summary["n_features"]),
        "FC Units",       str(summary["fc_units"]),
    )
    tbl.add_row(
        "Dropout",        str(summary["dropout"]),
        "Parámetros",     f"{summary['parametros']:,}",
    )

    console.print(Panel(tbl,
        title="[bold white]🧠 ARQUITECTURA DEL MODELO GRU[/bold white]",
        border_style="magenta"))
