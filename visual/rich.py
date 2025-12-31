"""
MODELOX Institutional Terminal Interface (Bloomberg/Reuters Style)
===================================================================
Professional-grade CLI visualization with Rich library.
Minimalist, clean, robust data architecture.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich.style import Style
from rich.columns import Columns


# ============================================================================
# THEME SYSTEM (Centralized Institutional Styling)
# ============================================================================

@dataclass(frozen=True)
class Theme:
    """
    Institutional color palette - Bloomberg/Reuters inspired.
    Minimalist dark mode with professional aesthetics.
    """
    # === Primary Colors ===
    ACCENT: str = "slate_blue1"           # Primary accent (headers, highlights)
    SUCCESS: str = "spring_green3"        # Profits, wins, positive metrics
    DANGER: str = "bright_red"            # Losses, alerts, negative metrics
    WARNING: str = "dark_orange"          # Caution, neutral-negative
    
    # === Grayscale Hierarchy ===
    TEXT_PRIMARY: str = "grey85"          # Main text
    TEXT_SECONDARY: str = "grey62"        # Labels, descriptions
    TEXT_MUTED: str = "grey46"            # Dimmed, less important
    TEXT_DIM: str = "grey30"              # Borders, separators
    
    # === Panel/Border Colors ===
    BORDER_LIGHT: str = "grey42"          # Panel borders
    BORDER_DARK: str = "grey27"           # Table borders
    BACKGROUND: str = "grey11"            # Background hint
    
    # === Semantic Aliases ===
    PROFIT: str = "spring_green3"
    LOSS: str = "bright_red"
    NEUTRAL: str = "grey62"
    BEST_MARKER: str = "gold1"
    
    # === Box Styles ===
    BOX_PANEL: box.Box = box.ROUNDED
    BOX_TABLE: box.Box = box.MINIMAL
    BOX_GRID: box.Box = box.SIMPLE


# Global theme instance
THEME = Theme()


# ============================================================================
# FLEXIBLE METRIC MAPPER (Fixes 0.0 bug)
# ============================================================================

class MetricMapper:
    """
    Aggressive metric extraction with flexible key mapping.
    Searches in multiple locations: metricas dict, user_attrs directly, and nested structures.
    Handles variations like: winrate, win_rate, wr, WinRate, etc.
    Handles string-to-number conversions.
    """
    
    # Define all possible key variations for each metric
    MAPPINGS: Dict[str, Tuple[str, ...]] = {
        "winrate": ("winrate", "win_rate", "wr", "winRate", "WinRate", "win_pct", "win_percent"),
        "sharpe": ("sharpe", "sharpe_ratio", "sharpeRatio", "sr"),
        "sortino": ("sortino", "sortino_ratio", "sortinoRatio"),
        "profit_factor": ("profit_factor", "profitFactor", "pf", "profit_f"),
        "drawdown": ("drawdown", "max_drawdown", "maxDrawdown", "dd", "max_dd", "mdd"),
        "roi": ("roi", "ROI", "return_pct", "return_percent", "retorno", "pnl_percent"),
        "saldo_actual": ("saldo_actual", "saldo_final", "final_balance", "balance", "saldo"),
        "comisiones_total": ("comisiones_total", "comisiones", "commissions", "total_comm", "fees"),
        "total_trades": ("total_trades", "num_trades", "trades", "n_trades", "trade_count"),
        "count_longs": ("count_longs", "num_longs", "longs", "long_count", "n_longs"),
        "count_shorts": ("count_shorts", "num_shorts", "shorts", "short_count", "n_shorts"),
        "pnl_neto": ("pnl_neto", "pnl", "net_pnl", "profit", "net_profit"),
    }
    
    @classmethod
    def _extract_value(cls, obj: Any, default: Any = 0.0) -> Any:
        """
        Aggressively extract value from various data types.
        Handles strings, nested dicts, etc.
        """
        if obj is None:
            return default
        
        # Handle direct values
        if isinstance(obj, (int, float, bool)):
            return obj
        
        # Handle string numbers
        if isinstance(obj, str):
            try:
                # Try int first
                if '.' not in obj:
                    return int(obj)
                # Try float
                return float(obj)
            except (ValueError, TypeError):
                return default
        
        # Handle nested dict (take first numeric value)
        if isinstance(obj, dict):
            for v in obj.values():
                result = cls._extract_value(v, None)
                if result is not None:
                    return result
        
        return default
    
    @classmethod
    def get(cls, metrics: Dict[str, Any], key: str, default: Any = 0.0) -> Any:
        """
        Get metric value with aggressive flexible key matching.
        Tries all known variations of the key and handles type conversions.
        """
        # Direct match first
        if key in metrics and metrics[key] is not None:
            return cls._extract_value(metrics[key], default)
        
        # Try all variations
        variations = cls.MAPPINGS.get(key, (key,))
        for var in variations:
            if var in metrics and metrics[var] is not None:
                return cls._extract_value(metrics[var], default)
        
        # Case-insensitive fallback
        key_lower = key.lower()
        for k, v in metrics.items():
            if k.lower() == key_lower and v is not None:
                return cls._extract_value(v, default)
        
        return default
    
    @classmethod
    def get_float(cls, metrics: Dict[str, Any], key: str, default: float = 0.0) -> float:
        """Get metric as float with safe conversion."""
        val = cls.get(metrics, key, default)
        try:
            return float(val) if val is not None else default
        except (ValueError, TypeError):
            return default
    
    @classmethod
    def get_int(cls, metrics: Dict[str, Any], key: str, default: int = 0) -> int:
        """Get metric as int with safe conversion."""
        val = cls.get(metrics, key, default)
        try:
            return int(val) if val is not None else default
        except (ValueError, TypeError):
            return default
    
    @classmethod
    def extract_from_trial(cls, trial, key: str, default: Any = 0.0) -> Any:
        """
        Aggressively extract metric from Optuna trial object.
        Searches in multiple locations:
        1. trial.user_attrs['metricas'][key]
        2. trial.user_attrs[key] (direct)
        3. All nested dicts in user_attrs
        """
        # Try metricas dict first
        if "metricas" in trial.user_attrs:
            met = trial.user_attrs["metricas"]
            if isinstance(met, dict):
                result = cls.get(met, key, None)
                if result is not None and result != default:
                    return result
        
        # Try direct user_attrs
        result = cls.get(trial.user_attrs, key, None)
        if result is not None and result != default:
            return result
        
        # Search all nested dicts in user_attrs
        for attr_key, attr_val in trial.user_attrs.items():
            if isinstance(attr_val, dict) and attr_key != "metricas":
                result = cls.get(attr_val, key, None)
                if result is not None and result != default:
                    return result
        
        return default


# ============================================================================
# FORMATTING UTILITIES
# ============================================================================

def fmt_number(val: float, decimals: int = 2, suffix: str = "") -> str:
    """Format number with consistent decimal places."""
    try:
        return f"{float(val):,.{decimals}f}{suffix}"
    except (ValueError, TypeError):
        return f"0.{'0' * decimals}{suffix}"


def fmt_styled(val: float, decimals: int = 2, suffix: str = "", color: str = "") -> str:
    """Format number with Rich color styling."""
    formatted = fmt_number(val, decimals, suffix)
    if color:
        return f"[{color}]{formatted}[/{color}]"
    return formatted


def get_pnl_color(value: float) -> str:
    """Get color for PnL values (green positive, red negative)."""
    return THEME.SUCCESS if value >= 0 else THEME.DANGER


def get_metric_color(value: float, good: float, warn: float, higher_is_better: bool = True) -> str:
    """Get color based on metric thresholds."""
    if higher_is_better:
        if value >= good:
            return THEME.SUCCESS
        elif value >= warn:
            return THEME.WARNING
        return THEME.DANGER
    else:
        if value <= good:
            return THEME.SUCCESS
        elif value <= warn:
            return THEME.WARNING
        return THEME.DANGER


# ============================================================================
# GRID BUILDERS FOR PANEL COLUMNS
# ============================================================================

def _build_performance_grid(metrics: Dict[str, Any]) -> Table:
    """
    Build PERFORMANCE column grid.
    Contains: Win Rate, Profit Factor, Sharpe, Max Drawdown, Total Trades
    """
    M = MetricMapper
    
    winrate = M.get_float(metrics, "winrate")
    profit_factor = M.get_float(metrics, "profit_factor")
    sharpe = M.get_float(metrics, "sharpe")
    drawdown = M.get_float(metrics, "drawdown")
    total_trades = M.get_int(metrics, "total_trades")
    expectancy = M.get_float(metrics, "expectativa")
    trades_per_day = M.get_float(metrics, "trades_por_dia")
    longs = M.get_int(metrics, "count_longs")
    shorts = M.get_int(metrics, "count_shorts")
    
    grid = Table.grid(padding=(0, 2), expand=True)
    grid.add_column("label", style=THEME.TEXT_SECONDARY, width=14)
    grid.add_column("value", justify="right")
    
    # Winrate y resto neutros (sin colores agresivos)
    grid.add_row(
        "Win Rate",
        fmt_number(winrate, 1, "%")
    )

    # Expectativa: única métrica de performance en rojo/verde
    exp_color = get_pnl_color(expectancy)
    grid.add_row(
        "Expectativa",
        fmt_styled(expectancy, 2, "", exp_color)
    )

    grid.add_row(
        "Trades/Día",
        fmt_number(trades_per_day, 2, "")
    )
    grid.add_row(
        "Sharpe",
        fmt_number(sharpe, 2, "")
    )
    grid.add_row(
        "Profit Factor",
        fmt_number(profit_factor, 2, "")
    )
    grid.add_row(
        "Max Drawdown",
        fmt_number(drawdown, 1, "%")
    )
    grid.add_row(
        f"[{THEME.TEXT_DIM}]───────────────────[/]", ""
    )
    grid.add_row(
        "Total Trades",
        f"[{THEME.ACCENT}]{total_trades}[/]"
    )
    grid.add_row(
        "Long / Short",
        f"[{THEME.TEXT_PRIMARY}]{longs}[/]  /  [{THEME.TEXT_PRIMARY}]{shorts}[/]"
    )
    
    return grid


def _build_financials_grid(metrics: Dict[str, Any], saldo_inicial: float) -> Table:
    """
    Build FINANCIALS column grid.
    Contains: PnL Neto, ROI %, Comisiones, Saldo Final
    """
    M = MetricMapper
    
    saldo_final = M.get_float(metrics, "saldo_actual", saldo_inicial)
    saldo_min = M.get_float(metrics, "saldo_min", saldo_inicial)
    saldo_max = M.get_float(metrics, "saldo_max", saldo_inicial)
    saldo_mean = M.get_float(metrics, "saldo_mean", saldo_inicial)
    comisiones = M.get_float(metrics, "comisiones_total")
    roi = M.get_float(metrics, "roi") if saldo_inicial > 0 else 0.0
    pnl_neto = saldo_final - saldo_inicial
    
    grid = Table.grid(padding=(0, 2), expand=True)
    grid.add_column("label", style=THEME.TEXT_SECONDARY, width=14)
    grid.add_column("value", justify="right")
    
    # PnL Neto (principal) - único campo financiero con rojo/verde fuerte
    pnl_sign = "+" if pnl_neto >= 0 else ""
    pnl_color = get_pnl_color(pnl_neto)
    grid.add_row(
        "[bold]PnL Neto[/]",
        f"[bold {pnl_color}]{pnl_sign}${pnl_neto:,.2f}[/]"
    )

    # ROI (neutro)
    grid.add_row(
        "ROI",
        fmt_number(roi, 1, "%")
    )

    grid.add_row(
        f"[{THEME.TEXT_DIM}]───────────────────[/]", ""
    )

    # Balances
    grid.add_row("Saldo Inicial", f"[{THEME.TEXT_PRIMARY}]${saldo_inicial:,.2f}[/]")
    grid.add_row("Saldo Mínimo", f"[{THEME.TEXT_PRIMARY}]${saldo_min:,.2f}[/]")
    grid.add_row("Saldo Medio", f"[{THEME.TEXT_PRIMARY}]${saldo_mean:,.2f}[/]")
    grid.add_row("Saldo Máximo", f"[{THEME.TEXT_PRIMARY}]${saldo_max:,.2f}[/]")
    grid.add_row("Saldo Final", f"[{THEME.TEXT_PRIMARY}]${saldo_final:,.2f}[/]")

    grid.add_row(
        f"[{THEME.TEXT_DIM}]───────────────────[/]", ""
    )

    # Comisiones
    grid.add_row(
        "Comisiones",
        f"[{THEME.WARNING}]${comisiones:,.2f}[/]"
    )
    
    return grid


def _build_params_grid(params: Dict[str, Any], max_params: int = 0) -> Table:
    """Build PARAMS column grid showing all Optuna parameters for this trial."""

    grid = Table.grid(padding=(0, 1), expand=True)
    grid.add_column("param", style=THEME.TEXT_MUTED, width=18)
    grid.add_column("value", style=THEME.TEXT_PRIMARY, justify="left")
    
    # Filter params (exclude internal __ prefixed)
    clean_params = {
        k: v for k, v in params.items()
        if not str(k).startswith("__") and k not in {"NOMBRE_COMBO"}
    }
    
    # Sort keys alphabetically; show all (max_params==0 => no limit)
    sorted_keys_all = sorted(clean_params.keys())
    sorted_keys = sorted_keys_all if max_params == 0 else sorted_keys_all[:max_params]
    
    for key in sorted_keys:
        value = clean_params[key]
        
        # Format param name
        pname = str(key).replace("_", " ").title()
        if len(pname) > 11:
            pname = pname[:10] + "…"
        
        # Format value
        if isinstance(value, float):
            if abs(value) < 0.01:
                val_str = f"{value:.4f}"
            elif abs(value) < 10:
                val_str = f"{value:.2f}"
            else:
                val_str = f"{int(value)}"
        elif isinstance(value, bool):
            val_str = f"[{THEME.SUCCESS}]ON[/]" if value else f"[{THEME.TEXT_DIM}]OFF[/]"
        else:
            val_str = str(value)[:10]
        
        # Render as "Nombre Param = valor" en dos columnas claras
        grid.add_row(pname, f"= {val_str}")
    
    # Show "+N more" if truncated
    if max_params and len(clean_params) > max_params:
        remaining = len(clean_params) - max_params
        grid.add_row(f"[{THEME.TEXT_DIM}]+{remaining} more[/]", "")
    
    return grid


# ============================================================================
# MAIN PANEL DISPLAY (mostrar_panel_elegante)
# ============================================================================

def mostrar_panel_elegante(
    metrics: Dict[str, Any],
    params: Dict[str, Any],
    score: float,
    trial_num: int,
    saldo_inicial: float,
    indicadores_activos: Optional[List[str]] = None,
    combo_str: str = "",
    activo: str = "",
    best_so_far: Optional[float] = None,
) -> None:
    """
    Display institutional 3-column trial results panel.
    
    Layout:
    ┌─────────────────────────────────────────────────────────────────┐
    │ ASSET │ STRATEGY │ TRIAL # │ SCORE ★                           │
    ├───────────────────┬───────────────────┬─────────────────────────┤
    │ PERFORMANCE       │ FINANCIALS        │ PARAMETERS              │
    │ Win Rate    55.0% │ PnL Neto  +$45.20 │ rsi_period = 14         │
    │ Profit F     1.32 │ ROI        15.1%  │ macd_fast  = 12         │
    │ Sharpe       1.45 │ ─────────────     │ stop_loss  = 2.5        │
    │ Max DD      12.3% │ Comisiones $8.40  │ take_prof  = 4.0        │
    │ ─────────────     │ Saldo Fin $345.20 │ threshold  = 0.02       │
    │ Total Trades  42  │                   │                         │
    └───────────────────┴───────────────────┴─────────────────────────┘
    """
    console = Console()
    
    # Determine if this is the best trial
    is_best = best_so_far is not None and score >= float(best_so_far)
    
    # ================== HEADER LINE ==================
    header_parts = []
    
    # Asset
    asset_display = activo.upper() if activo else "ASSET"
    header_parts.append(f"[bold {THEME.TEXT_PRIMARY}]{asset_display}[/]")
    
    # Strategy
    if combo_str:
        header_parts.append(f"[{THEME.ACCENT}]{combo_str}[/]")
    
    # Trial number
    header_parts.append(f"[{THEME.TEXT_SECONDARY}]TRIAL[/] [{THEME.TEXT_PRIMARY}]{trial_num}[/]")
    
    # Score with best indicator
    score_color = THEME.BEST_MARKER if is_best else THEME.SUCCESS
    best_star = " ★" if is_best else ""
    header_parts.append(f"[bold {score_color}]SCORE {score:.2f}{best_star}[/]")
    
    # Best so far reference
    if best_so_far is not None and not is_best:
        header_parts.append(f"[{THEME.TEXT_DIM}]BEST {best_so_far:.2f}[/]")
    
    header_line = f" [{THEME.TEXT_DIM}]│[/] ".join(header_parts)
    
    # ================== BUILD 3-COLUMN TABLE ==================
    main_table = Table(
        box=THEME.BOX_TABLE,
        show_header=True,
        header_style=f"{THEME.TEXT_SECONDARY}",
        border_style=THEME.BORDER_DARK,
        padding=(0, 1),
        expand=False,
        width=100
    )
    
    main_table.add_column("PERFORMANCE", justify="left", width=30)
    main_table.add_column("FINANCIALS", justify="left", width=30)
    main_table.add_column("PARAMETERS", justify="left", width=34)
    
    # Build each column grid
    perf_grid = _build_performance_grid(metrics)
    fin_grid = _build_financials_grid(metrics, saldo_inicial)
    params_grid = _build_params_grid(params)
    
    main_table.add_row(perf_grid, fin_grid, params_grid)
    
    # ================== RENDER ==================
    console.print()
    console.print(f"  {header_line}")
    console.print(main_table)


# ============================================================================
# TOP TRIALS RANKING TABLE (mostrar_top_trials)
# ============================================================================

def mostrar_top_trials(study, n: int = 5) -> None:
    """
    Display TOP N TRIALS ranking table.
    Uses aggressive metric extraction from Optuna trial objects.
    Searches in: metricas dict, user_attrs directly, and nested structures.
    
    Columns: Trial #, Score, ROI%, Winrate%, Drawdown%, Balance, Trades
    """
    console = Console()
    M = MetricMapper
    
    # Get completed trials with valid scores
    valid_trials = [
        t for t in study.trials 
        if t.value is not None and t.value != -9999 and t.state.name == "COMPLETE"
    ]
    
    if not valid_trials:
        console.print(f"\n  [{THEME.TEXT_MUTED}]No valid trials to display.[/]\n")
        return
    
    # Sort by score descending
    top_trials = sorted(valid_trials, key=lambda t: t.value or 0, reverse=True)[:n]
    
    # Build table
    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style=f"bold {THEME.TEXT_SECONDARY}",
        border_style=THEME.BORDER_DARK,
        padding=(0, 2),
        title=f"[bold {THEME.ACCENT}]═══ TOP {n} TRIALS ═══[/]",
        title_justify="center"
    )
    
    table.add_column("#", justify="center", style=THEME.TEXT_MUTED, width=6)
    table.add_column("SCORE", justify="right", style=THEME.SUCCESS, width=10)
    table.add_column("ROI %", justify="right", width=10)
    table.add_column("WIN %", justify="right", width=10)
    table.add_column("DD %", justify="right", width=10)
    table.add_column("BALANCE", justify="right", width=12)
    table.add_column("TRADES", justify="center", style=THEME.TEXT_PRIMARY, width=8)
    
    for i, trial in enumerate(top_trials):
        # AGGRESSIVE EXTRACTION: Try multiple sources
        score = trial.value or 0.0
        
        # Extract metrics using aggressive search
        roi = M.extract_from_trial(trial, "roi", 0.0)
        winrate = M.extract_from_trial(trial, "winrate", 0.0)
        drawdown = M.extract_from_trial(trial, "drawdown", 0.0)
        saldo_final = M.extract_from_trial(trial, "saldo_actual", 0.0)
        trades = M.extract_from_trial(trial, "total_trades", 0)
        
        # Convert to proper types
        roi = float(roi) if roi != 0.0 else 0.0
        winrate = float(winrate) if winrate != 0.0 else 0.0
        drawdown = float(drawdown) if drawdown != 0.0 else 0.0
        saldo_final = float(saldo_final) if saldo_final != 0.0 else 0.0
        trades = int(trades) if trades != 0 else 0
        
        # Format with colors
        roi_color = get_pnl_color(roi)
        win_color = get_metric_color(winrate, 55, 45)
        dd_color = get_metric_color(drawdown, 15, 30, higher_is_better=False)
        balance_color = THEME.TEXT_PRIMARY
        
        # First row gets gold styling
        trial_style = THEME.BEST_MARKER if i == 0 else THEME.TEXT_MUTED
        score_style = f"bold {THEME.BEST_MARKER}" if i == 0 else THEME.SUCCESS
        
        table.add_row(
            f"[{trial_style}]{trial.number}[/]",
            f"[{score_style}]{score:.2f}[/]",
            f"[{roi_color}]{roi:+.1f}[/]",
            f"[{win_color}]{winrate:.1f}[/]",
            f"[{dd_color}]{drawdown:.1f}[/]",
            f"[{balance_color}]${saldo_final:,.2f}[/]",
            str(trades)
        )
    
    console.print()
    console.print(table)
    console.print()


# ============================================================================
# OPTIMIZATION COMPLETION PANEL (mostrar_fin_optimizacion)
# ============================================================================

def mostrar_fin_optimizacion(
    total_trials: int,
    best_score: float,
    best_trial: int,
    estrategia: str = "",
) -> None:
    """
    Display elegant optimization completion panel.
    Centered with checkmark and summary.
    """
    console = Console()
    
    # Build content
    content_lines = []
    
    # Main title with checkmark
    content_lines.append(f"[bold {THEME.SUCCESS}]✔[/]  [bold {THEME.TEXT_PRIMARY}]OPTIMIZATION COMPLETE[/]")
    content_lines.append("")
    
    # Stats grid
    stats = Table.grid(padding=(0, 2))
    stats.add_column("label", style=THEME.TEXT_SECONDARY, justify="right")
    stats.add_column("value", style=THEME.TEXT_PRIMARY, justify="left")
    
    stats.add_row("Trials Executed", f"[bold]{total_trials}[/]")
    stats.add_row("Best Score", f"[bold {THEME.BEST_MARKER}]{best_score:.2f}[/]")
    stats.add_row("Best Trial", f"[{THEME.ACCENT}]#{best_trial}[/]")
    
    if estrategia:
        stats.add_row("Strategy", f"[{THEME.TEXT_MUTED}]{estrategia}[/]")
    
    # Compose panel content
    panel_content = Group(
        Align.center(Text.from_markup("\n".join(content_lines))),
        Align.center(stats)
    )
    
    panel = Panel(
        panel_content,
        box=THEME.BOX_PANEL,
        border_style=THEME.BORDER_LIGHT,
        padding=(1, 4),
        width=50
    )
    
    console.print()
    console.print(Align.center(panel))
    console.print()


# ============================================================================
# STARTUP HEADER (mostrar_cabecera_inicio)
# ============================================================================

def mostrar_cabecera_inicio(
    activo: str,
    combo_nombre: str,
    indicadores: List[str],
    n_trials: int,
    archivo_data: str = "",
) -> None:
    """
    Display minimalist startup header (institutional style).
    """
    console = Console()
    
    # Clear screen
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Build content grid
    grid = Table.grid(padding=(0, 2), expand=False)
    grid.add_column("label", style=THEME.TEXT_SECONDARY, width=12, justify="right")
    grid.add_column("value")
    
    # Asset with icon
    asset_icons = {
        "BTC": "₿", "GOLD": "●", "SP500": "◆", "SP": "◆", "NASDAQ": "■", "NDX": "■"
    }
    asset_icon = asset_icons.get(activo.upper(), "○")
    grid.add_row("ASSET", f"[bold {THEME.ACCENT}]{asset_icon} {activo.upper()}[/]")
    
    # Strategy
    grid.add_row("STRATEGY", f"[{THEME.TEXT_PRIMARY}]{combo_nombre}[/]")
    
    # Indicators
    if indicadores:
        inds_str = " · ".join([f"[{THEME.TEXT_MUTED}]{ind.upper()}[/]" for ind in indicadores])
        grid.add_row("INDICATORS", inds_str)
    
    # Trials
    grid.add_row("TRIALS", f"[bold {THEME.TEXT_PRIMARY}]{n_trials}[/]")
    
    # Data file
    if archivo_data:
        grid.add_row("DATA", f"[{THEME.TEXT_DIM}]{archivo_data}[/]")
    
    # Title
    title_text = Text()
    title_text.append("═══ ", style=THEME.BORDER_LIGHT)
    title_text.append("MODELOX", style=f"bold {THEME.TEXT_PRIMARY}")
    title_text.append(" ═══", style=THEME.BORDER_LIGHT)
    
    # Panel
    panel = Panel(
        Align.center(grid),
        title=title_text,
        title_align="center",
        subtitle=f"[{THEME.TEXT_DIM}]Optuna Backtesting Engine[/]",
        subtitle_align="center",
        box=THEME.BOX_PANEL,
        border_style=THEME.BORDER_LIGHT,
        padding=(1, 3),
        width=60
    )
    
    console.print()
    console.print(Align.center(panel))
    console.print()


# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================

def mostrar_panel_rich(
    metrics: dict,
    params: dict,
    score: float,
    trial_num: int,
    saldo_inicial: float,
    indicadores_activos: list[str] | None = None,
    combo_idx: int = 1,
    n_combos: int = 1,
    combo_str: str = "",
    activo: str = "",
) -> None:
    """Legacy wrapper - maps to institutional panel."""
    mostrar_panel_elegante(
        metrics=metrics,
        params=params,
        score=score,
        trial_num=trial_num,
        saldo_inicial=saldo_inicial,
        indicadores_activos=indicadores_activos,
        combo_str=combo_str,
        activo=activo,
        best_so_far=params.get("__best_score_so_far"),
    )


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "THEME",
    "MetricMapper",
    "mostrar_panel_elegante",
    "mostrar_top_trials",
    "mostrar_fin_optimizacion",
    "mostrar_cabecera_inicio",
    "mostrar_panel_rich",
]
