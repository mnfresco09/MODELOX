"""
MODELOX Terminal Interface v3.0 - Classic Institutional Design
===============================================================
Elegant, clean, professional terminal display.
Neutral tones with selective color highlighting.
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
from rich.columns import Columns


# ============================================================================
# THEME - Classic Institutional Palette (Neutral & Elegant)
# ============================================================================

@dataclass(frozen=True)
class Theme:
    """Classic institutional color palette - Neutral & Professional."""
    # Primary accent (for score, headers) - darker blue
    BLUE: str = "steel_blue3"
    CYAN: str = "cyan"                 # For phase names in plateau mode
    # Performance colors (only for PnL based on ROI)
    GREEN: str = "green3"              # ROI > 100% (vibrant)
    GREEN_SOFT: str = "pale_green3"    # ROI 0% to 100% (clear but not flashy)
    ORANGE: str = "orange3"            # ROI -50% to 0% (soft orange)
    RED: str = "red3"                  # ROI -50% to -70% (medium bright red)
    # Best marker
    GOLD: str = "gold3"
    BEST_BOX: str = "dodger_blue2"     # Bright blue for best trial box
    # Neutral tones
    TEXT: str = "grey78"
    MUTED: str = "grey50"
    DIM: str = "grey35"
    BORDER: str = "grey42"
    WHITE: str = "white"


THEME = Theme()
console = Console()


# ============================================================================
# METRIC MAPPER - Flexible Key Extraction
# ============================================================================

class M:
    """Ultra-compact metric mapper."""
    
    KEYS = {
        "winrate": ("winrate", "win_rate", "wr"),
        "sharpe": ("sharpe", "sharpe_ratio", "sr"),
        "profit_factor": ("profit_factor", "pf"),
        "drawdown": ("drawdown", "max_drawdown", "dd", "mdd"),
        "roi": ("roi", "ROI", "return_pct"),
        "saldo_actual": ("saldo_actual", "saldo_final", "balance"),
        "comisiones_total": ("comisiones_total", "comisiones", "fees"),
        "total_trades": ("total_trades", "num_trades", "trades", "n_trades"),
        "count_longs": ("count_longs", "longs", "n_longs"),
        "count_shorts": ("count_shorts", "shorts", "n_shorts"),
        "pnl_neto": ("pnl_neto", "pnl", "net_pnl"),
        "trades_por_dia": ("trades_por_dia", "trades_dia", "tpd"),
        "sqn": ("sqn", "SQN"),
        "expectativa": ("expectativa", "expectancy", "exp"),
        "saldo_mean": ("saldo_mean", "avg_balance"),
    }

    @classmethod
    def get(cls, d: Dict, key: str, default: float = 0.0) -> float:
        if not d:
            return default
        if key in d and d[key] is not None:
            try:
                return float(d[key])
            except:
                return default
        for var in cls.KEYS.get(key, (key,)):
            if var in d and d[var] is not None:
                try:
                    return float(d[var])
                except:
                    pass
        return default

    @classmethod
    def get_int(cls, d: Dict, key: str, default: int = 0) -> int:
        return int(cls.get(d, key, default))

    @classmethod
    def from_trial(cls, trial, key: str, default: float = 0.0) -> float:
        if "metricas" in trial.user_attrs:
            v = cls.get(trial.user_attrs["metricas"], key, None)
            if v is not None:
                return v
        return cls.get(trial.user_attrs, key, default)


# ============================================================================
# COLOR HELPERS (Only for PnL based on ROI)
# ============================================================================

def _pnl_color(roi: float) -> str:
    """
    Get PnL color based on ROI thresholds:
    - GREEN:      ROI > 100% (vibrant green)
    - GREEN_SOFT: ROI 0% to 100% (clear but not flashy)
    - ORANGE:     ROI -50% to 0%
    - RED:        ROI -50% to -70% (and below)
    """
    if roi > 100:
        return THEME.GREEN
    elif roi > 0:
        return THEME.GREEN_SOFT  # Clear green for 0-100%
    elif roi >= -50:
        return THEME.ORANGE  # -50% to 0%
    else:
        return THEME.RED  # -50% to -70% and worse


def _score_color(is_best: bool) -> str:
    """Score color: gold if best, blue otherwise."""
    return THEME.GOLD if is_best else THEME.BLUE


def _format_score(score: float) -> str:
    """
    Formatea el score para display.
    
    Scoring v2.0 rango [1, 1000]:
        - score >= 100: mostrar como entero
        - score >= 10: mostrar con 1 decimal
        - score < 10: mostrar con 2 decimales
    """
    if score >= 100:
        return f"{score:.0f}"
    elif score >= 10:
        return f"{score:.1f}"
    else:
        return f"{score:.2f}"


# ============================================================================
# MAIN TRIAL DISPLAY - Box Layout
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
    timeframe_entry: str = "",
    timeframe_exit: str = "",
    neighborhood_result: Optional[Dict[str, Any]] = None,  # Deprecated, ignorado
    # Nuevos parámetros para Topógrafo de Mesetas
    phase_name: str = "",  # Ej: "EXPLORACIÓN", "REFINAMIENTO"
    phase_progress: str = "",  # Ej: "42/2000"
) -> None:
    """
    Classic elegant trial display with organized boxes.
    All text uppercase, neutral tones, selective colors.
    """
    is_best = best_so_far is not None and score >= float(best_so_far)
    
    # Extract metrics
    wr = M.get(metrics, "winrate")
    exp = M.get(metrics, "expectativa")
    shp = M.get(metrics, "sharpe")
    sqn = M.get(metrics, "sqn")
    pf = M.get(metrics, "profit_factor")
    dd = M.get(metrics, "drawdown")
    trades = M.get_int(metrics, "total_trades")
    tpd = M.get(metrics, "trades_por_dia")
    longs = M.get_int(metrics, "count_longs")
    shorts = M.get_int(metrics, "count_shorts")
    
    pnl = M.get(metrics, "pnl_neto")
    roi = M.get(metrics, "roi")
    saldo_final = M.get(metrics, "saldo_actual", saldo_inicial)
    saldo_avg = M.get(metrics, "saldo_mean", saldo_inicial)
    fees = M.get(metrics, "comisiones_total")
    
    # Exit params
    exit_type = str(params.get("__exit_type", params.get("exit_type", "FIXED"))).upper()
    sl = params.get("__exit_sl_pct", params.get("exit_sl_pct", 0.0))
    tp = params.get("__exit_tp_pct", params.get("exit_tp_pct", 0.0))
    trail_act = params.get("__exit_trail_act_pct", params.get("exit_trail_act_pct", 0.0))
    trail_dist = params.get("__exit_trail_dist_pct", params.get("exit_trail_dist_pct", 0.0))
    
    # Header info
    asset = activo.upper()[:6] if activo else "ASSET"
    strat = combo_str.upper()[:12] if combo_str else "STRATEGY"
    tf = timeframe_entry.upper() if timeframe_entry else "1M"
    
    # ═══════════════════════════════════════════════════════════════════════
    # HEADER LINE
    # ═══════════════════════════════════════════════════════════════════════
    star = " ★" if is_best else ""
    score_c = _score_color(is_best)
    
    # Get best score for header (only show if NOT current best)
    s = _stats.obtener_promedios()
    best_str = ""
    if not is_best and s['mejor_score'] > float('-inf') and s['n_trials'] > 0:
        best_str = f"  │  [{THEME.GOLD}]BEST {_format_score(s['mejor_score'])} TRIAL {s['mejor_trial']}[/]"
    
    # Get cantidad for asset display
    cantidad = params.get("cantidad", params.get("__cantidad", None))
    asset_str = f"[bold {THEME.BLUE}]{asset} {cantidad}[/]" if cantidad is not None else f"[bold {THEME.BLUE}]{asset}[/]"
    
    # Construir info de fase si está disponible
    phase_str = ""
    if phase_name:
        phase_str = f"[bold {THEME.BLUE}]{phase_name}[/]  │  "
    
    # Mostrar progreso de fase en lugar del número de trial global
    trial_display = f"[{THEME.WHITE}]{trial_num}[/]"
    if phase_progress:
        trial_display = f"[bold {THEME.WHITE}]{phase_progress}[/]"
    
    # Header normal (con SCORE)
    header = (f"{asset_str}  │  {strat}  │  "
              f"TF {tf}  │  {phase_str}TRIAL {trial_display}  │  "
              f"[bold {score_c}]SCORE {_format_score(score)}{star}[/]{best_str}")
    
    # Header para best trial (sin SCORE, se muestra aparte)
    header_no_score = (f"{asset_str}  │  {strat}  │  "
                       f"TF {tf}  │  TRIAL [{THEME.WHITE}]{trial_num}[/]")
    
    # ═══════════════════════════════════════════════════════════════════════
    # BOX 1: PERFORMANCE
    # ═══════════════════════════════════════════════════════════════════════
    perf_grid = Table.grid(padding=(0, 2))
    perf_grid.add_column(style=THEME.MUTED, width=14)
    perf_grid.add_column(style=THEME.TEXT, justify="right", width=10)
    
    perf_grid.add_row("WIN RATE", f"{wr:.1f}%")
    perf_grid.add_row("EXPECTANCY", f"{exp:.2f}")
    perf_grid.add_row("SHARPE", f"{shp:.2f}")
    perf_grid.add_row("SQN", f"{sqn:.3f}")
    perf_grid.add_row("PROFIT F", f"{pf:.2f}")
    perf_grid.add_row("MAX DD", f"{dd:.1f}%")
    perf_grid.add_row("TRADES/DAY", f"{tpd:.3f}")
    perf_grid.add_row("L / S", f"{longs} / {shorts}")
    
    perf_box = Panel(
        perf_grid,
        title=f"[{THEME.MUTED}]PERFORMANCE[/]",
        border_style=THEME.BORDER,
        box=box.ROUNDED,
        padding=(0, 1),
        width=30
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # BOX 2: FINANCIALS
    # ═══════════════════════════════════════════════════════════════════════
    fin_grid = Table.grid(padding=(0, 2))
    fin_grid.add_column(style=THEME.MUTED, width=10)
    fin_grid.add_column(style=THEME.TEXT, justify="right", width=14)
    
    pnl_c = _pnl_color(roi)
    pnl_sign = "+" if pnl >= 0 else ""
    
    fin_grid.add_row("PNL", f"[{pnl_c}]{pnl_sign}${pnl:,.2f}[/]")
    fin_grid.add_row("ROI", f"[{pnl_c}]{roi:.1f}%[/]")
    fin_grid.add_row("", f"[{THEME.DIM}]{'─' * 12}[/]")
    fin_grid.add_row("INITIAL", f"${saldo_inicial:,.0f}")
    fin_grid.add_row("FINAL", f"${saldo_final:,.2f}")
    fin_grid.add_row("AVG", f"${saldo_avg:,.2f}")
    fin_grid.add_row("", "")
    fin_grid.add_row("FEES", f"${fees:,.2f}")
    
    fin_box = Panel(
        fin_grid,
        title=f"[{THEME.MUTED}]FINANCIALS[/]",
        border_style=THEME.BORDER,
        box=box.ROUNDED,
        padding=(0, 1),
        width=28
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # AVERAGES LINE (horizontal, below boxes)
    # ═══════════════════════════════════════════════════════════════════════
    avg_line = ""
    if s['n_trials'] > 0:
        avg_line = (f"[{THEME.MUTED}]μ({s['n_trials']})[/]  "
                    f"ROI [{THEME.TEXT}]{s['roi_medio']:.1f}%[/]  │  "
                    f"PF [{THEME.TEXT}]{s['pf_medio']:.2f}[/]  │  "
                    f"SHARPE [{THEME.TEXT}]{s['sharpe_medio']:.2f}[/]  │  "
                    f"SQN [{THEME.TEXT}]{s['sqn_medio']:.2f}[/]  │  "
                    f"SCORE [{THEME.TEXT}]{_format_score(s['score_medio'])}[/]")
    
    # ═══════════════════════════════════════════════════════════════════════
    # BOX 3: PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════
    param_grid = Table.grid(padding=(0, 2))
    param_grid.add_column(style=THEME.MUTED, width=16)
    param_grid.add_column(style=THEME.TEXT, justify="right", width=12)
    
    is_trailing = "TRAIL" in exit_type
    
    # Exit section
    param_grid.add_row(f"[bold {THEME.TEXT}]EXIT[/]", f"[bold {THEME.TEXT}]{'TRAILING' if is_trailing else ''}[/]")
    param_grid.add_row("  SL%", f"= {sl:.1f}%")
    
    if is_trailing:
        param_grid.add_row("  TRAIL ACT%", f"= {trail_act:.1f}%")
        param_grid.add_row("  TRAIL DIST%", f"= {trail_dist:.1f}%")
    else:
        param_grid.add_row("  TP%", f"= {tp:.1f}%")
    
    # Strategy params
    skip_keys = {"exit_type", "exit_sl_pct", "exit_tp_pct", "exit_trail_act_pct", 
                 "exit_trail_dist_pct", "NOMBRE_COMBO", "cantidad"}
    strategy_params = {k: v for k, v in params.items() 
                       if not str(k).startswith("__") and k not in skip_keys}
    
    if strategy_params:
        for k, v in list(strategy_params.items())[:6]:
            name = k.replace("_", " ").upper()[:14]
            if isinstance(v, float):
                val = f"= {v:.2f}" if abs(v) < 100 else f"= {v:.0f}"
            else:
                val = f"= {v}"
            param_grid.add_row(f"  {name}", val)
    
    param_box = Panel(
        param_grid,
        title=f"[{THEME.MUTED}]PARAMETERS[/]",
        border_style=THEME.BORDER,
        box=box.ROUNDED,
        padding=(0, 1),
        width=34
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # RENDER
    # ═══════════════════════════════════════════════════════════════════════
    
    # Row 1: PERFORMANCE | FINANCIALS | PARAMETERS
    row1 = Table.grid(padding=(0, 1))
    row1.add_column()
    row1.add_column()
    row1.add_column()
    row1.add_row(perf_box, fin_box, param_box)
    
    # Build content list for potential grouping
    content_parts = [header, "", row1]
    
    if avg_line:
        content_parts.append(avg_line)
    
    console.print()
    
    if is_best:
        # Para best trial: SCORE grande y prominente debajo del título
        score_line = Text()
        score_line.append(f"SCORE ", style=f"bold {THEME.GOLD}")
        score_line.append(f"{_format_score(score)}", style=f"bold {THEME.GOLD}")
        
        # Contenido con header sin score + score prominente
        best_content_parts = [score_line, "", header_no_score, "", row1]
        if avg_line:
            best_content_parts.append(avg_line)
        
        best_content = Group(*best_content_parts)
        best_panel = Panel(
            best_content,
            border_style=THEME.BEST_BOX,
            box=box.DOUBLE,
            padding=(1, 2),
            title=f"[bold {THEME.GOLD}]★ NEW BEST ★[/]",
            title_align="center",
        )
        console.print(best_panel, justify="center")
    else:
        # Normal render without wrapper
        console.print(header, justify="center")
        console.print()
        console.print(row1, justify="center")
        
        if avg_line:
            console.print(avg_line, justify="center")


# ============================================================================
# EVOLUTION TRACKER
# ============================================================================

class EstadisticasOptimizacion:
    """Optimization statistics tracker."""
    
    def __init__(self):
        self.valores = {"roi": [], "sqn": [], "exp": [], "sharpe": [], "score": [], "pf": []}
        self.mejor_score = float("-inf")
        self.mejor_trial = 0
        self.n = 0

    def actualizar(self, metricas: Dict[str, Any], score: float, trial: int) -> None:
        self.valores["roi"].append(M.get(metricas, "roi"))
        self.valores["sqn"].append(M.get(metricas, "sqn"))
        self.valores["exp"].append(M.get(metricas, "expectativa"))
        self.valores["sharpe"].append(M.get(metricas, "sharpe"))
        self.valores["score"].append(score)
        self.valores["pf"].append(M.get(metricas, "profit_factor"))
        self.n += 1
        if score > self.mejor_score:
            self.mejor_score = score
            self.mejor_trial = trial

    def promedio(self, key: str) -> float:
        v = self.valores.get(key, [])
        # Filtrar NaN e infinitos
        valid = [x for x in v if x == x and abs(x) != float('inf')]  # x != x es True solo para NaN
        return sum(valid) / len(valid) if valid else 0.0

    def obtener_promedios(self) -> Dict[str, float]:
        return {
            "roi_medio": self.promedio("roi"),
            "sqn_medio": self.promedio("sqn"),
            "expectativa_media": self.promedio("exp"),
            "sharpe_medio": self.promedio("sharpe"),
            "score_medio": self.promedio("score"),
            "pf_medio": self.promedio("pf"),
            "n_trials": self.n,
            "mejor_score": self.mejor_score,
            "mejor_trial": self.mejor_trial,
        }


_stats = EstadisticasOptimizacion()


def resetear_estadisticas() -> None:
    global _stats
    _stats = EstadisticasOptimizacion()


def actualizar_estadisticas(metricas: Dict[str, Any], score: float, trial: int) -> None:
    _stats.actualizar(metricas, score, trial)


def mostrar_evolucion_metricas(mostrar_cada_n: int = 25, forzar: bool = False) -> None:
    """Show evolution box every N trials - two columns, centered."""
    if not forzar and (_stats.n == 0 or _stats.n % mostrar_cada_n != 0):
        return
    
    s = _stats.obtener_promedios()
    
    # Two-column grid for compact display
    grid = Table.grid(padding=(0, 3))
    grid.add_column(style=THEME.MUTED, width=8)
    grid.add_column(style=THEME.TEXT, justify="right", width=10)
    grid.add_column(style=THEME.MUTED, width=8)
    grid.add_column(style=THEME.TEXT, justify="right", width=10)
    
    grid.add_row("ROI", f"{s['roi_medio']:.2f}%", "SQN", f"{s['sqn_medio']:.3f}")
    grid.add_row("EXP", f"{s['expectativa_media']:.2f}", "SHARPE", f"{s['sharpe_medio']:.2f}")
    grid.add_row("PF", f"{s['pf_medio']:.2f}", "SCORE", f"[{THEME.WHITE}]{_format_score(s['score_medio'])}[/]")
    
    evo_box = Panel(
        grid,
        title=f"[{THEME.MUTED}]SUMMARY ({s['n_trials']} TRIALS)[/]",
        border_style=THEME.BORDER,
        box=box.ROUNDED,
        padding=(0, 1),
        width=48
    )
    console.print(evo_box, justify="center")


def mostrar_evolucion_inline(metrics: Dict[str, Any], score: float, trial_num: int) -> str:
    """Returns inline evolution string."""
    roi = M.get(metrics, "roi")
    sharpe = M.get(metrics, "sharpe")
    sqn = M.get(metrics, "sqn")
    exp = M.get(metrics, "expectativa")
    
    return (f"[{THEME.MUTED}]│[/] ROI {roi:.1f}% "
            f"[{THEME.MUTED}]│[/] SHP {sharpe:.2f} "
            f"[{THEME.MUTED}]│[/] SQN {sqn:.2f} "
            f"[{THEME.MUTED}]│[/] EXP {exp:.2f}")


# ============================================================================
# TOP TRIALS TABLE
# ============================================================================

def mostrar_top_trials(study, n: int = 5) -> None:
    """Display top N trials ranking - elegant table."""
    valid = [t for t in study.trials 
             if t.value is not None and t.value != -9999 and t.state.name == "COMPLETE"]
    
    if not valid:
        console.print(f"[{THEME.MUTED}]NO VALID TRIALS[/]")
        return
    
    top = sorted(valid, key=lambda t: t.value or 0, reverse=True)[:n]
    
    table = Table(
        box=box.ROUNDED,
        border_style=THEME.BORDER,
        header_style=f"bold {THEME.MUTED}",
        title=f"[{THEME.BLUE}]TOP {n} TRIALS[/]",
        padding=(0, 1),
        collapse_padding=True
    )
    
    table.add_column("#", style=THEME.MUTED, justify="center", width=4)
    table.add_column("SCORE", style=THEME.TEXT, justify="right", width=8)
    table.add_column("ROI", style=THEME.TEXT, justify="right", width=8)
    table.add_column("WIN", style=THEME.TEXT, justify="right", width=6)
    table.add_column("DD", style=THEME.TEXT, justify="right", width=6)
    table.add_column("BALANCE", style=THEME.TEXT, justify="right", width=12)
    table.add_column("T", style=THEME.MUTED, justify="center", width=4)
    
    for i, t in enumerate(top):
        score = t.value or 0.0
        roi = M.from_trial(t, "roi")
        wr = M.from_trial(t, "winrate")
        dd = M.from_trial(t, "drawdown")
        bal = M.from_trial(t, "saldo_actual")
        trades = int(M.from_trial(t, "total_trades"))
        
        mark = "★" if i == 0 else str(t.number)
        score_style = f"bold {THEME.GOLD}" if i == 0 else THEME.BLUE
        
        table.add_row(
            mark,
            f"[{score_style}]{score:.1f}[/]",
            f"{roi:+.0f}%",
            f"{wr:.0f}%",
            f"{dd:.0f}%",
            f"${bal:,.0f}",
            str(trades)
        )
    
    console.print()
    console.print(table, justify="center")


# ============================================================================
# STARTUP HEADER - Elegant & Informative
# ============================================================================

def mostrar_cabecera_inicio(
    activo: str,
    combo_nombre: str,
    indicadores: List[str],
    n_trials: int,
    archivo_data: str = "",
    timeframe: str = "",
    periodo: str = "",
    exit_type: str = "FIXED",
    strategy_exit_enabled: bool = False,
    perturbacion_activar: bool = False,
    sampler_type: str = "CMA",
) -> None:
    """Display elegant startup header with comprehensive info - centered, white text."""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    icons = {"BTC": "₿", "GOLD": "◆", "SP500": "◇", "NASDAQ": "□"}
    icon = icons.get(activo.upper(), "○")
    
    exit_d = "CUSTOM" if strategy_exit_enabled else exit_type.upper()
    perturb = "ON" if perturbacion_activar else "OFF"
    
    # Main info grid - centered alignment
    info_grid = Table.grid(padding=(0, 3))
    info_grid.add_column(style=THEME.MUTED, width=14, justify="right")
    info_grid.add_column(style=THEME.WHITE, width=28, justify="left")
    
    info_grid.add_row("ASSET", f"[bold]{icon} {activo.upper()}[/]")
    info_grid.add_row("STRATEGY", f"{combo_nombre.upper()[:28]}")
    if timeframe:
        info_grid.add_row("TIMEFRAME", f"{timeframe.upper()}")
    if periodo:
        p = periodo.replace(" UTC", "").strip().upper()[:28]
        info_grid.add_row("PERIOD", f"{p}")
    info_grid.add_row("", "")
    info_grid.add_row("EXIT MODE", f"{exit_d}")
    info_grid.add_row("TRIALS", f"[bold]{n_trials}[/]")
    info_grid.add_row("PERTURBATION", f"{perturb}")
    
    if indicadores:
        inds = " ".join([i[:6].upper() for i in indicadores[:4]])
        if len(indicadores) > 4:
            inds += f" +{len(indicadores)-4}"
        info_grid.add_row("PARAMS", f"{inds}")
    
    if archivo_data:
        info_grid.add_row("DATA", f"{archivo_data[-28:].upper()}")
    
    # Main panel - white title with sampler type, centered
    sampler_label = sampler_type.upper() if sampler_type else "CMA"
    header_panel = Panel(
        info_grid,
        title=f"[bold {THEME.WHITE}]═══ MODELOX OPTIMIZATION {sampler_label} ═══[/]",
        subtitle=f"[{THEME.DIM}]V3.0[/]",
        border_style=THEME.BORDER,
        box=box.DOUBLE,
        padding=(1, 2),
        width=52
    )
    
    console.print()
    console.print(header_panel, justify="center")
    console.print()


def mostrar_fin_optimizacion(
    total_trials: int,
    best_score: float,
    best_trial: int,
    estrategia: str = "",
) -> None:
    """Display optimization completion - disabled."""
    pass


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def mostrar_fila_boxes(*boxes: Optional[Panel], spacing: int = 1) -> None:
    """Show multiple panels in a row."""
    valid = [b for b in boxes if b is not None]
    if not valid:
        return
    
    row = Table.grid(padding=(0, spacing))
    for _ in valid:
        row.add_column()
    row.add_row(*valid)
    console.print(row)


def crear_mini_box(title: str, rows: List[Tuple[str, str]], width: int = 16) -> Panel:
    """Create compact mini box."""
    grid = Table.grid(padding=(0, 1))
    grid.add_column(style=THEME.MUTED, width=6)
    grid.add_column(style=THEME.TEXT, justify="right", width=width - 10)
    
    for label, value in rows:
        grid.add_row(label.upper()[:6], str(value).upper()[:10])
    
    return Panel(
        grid, 
        title=f"[{THEME.MUTED}]{title.upper()}[/]",
        border_style=THEME.BORDER,
        box=box.ROUNDED,
        width=width,
        padding=(0, 0)
    )


def mostrar_trial_compacto(
    metrics: Dict[str, Any],
    params: Dict[str, Any],
    score: float,
    trial_num: int,
    saldo_inicial: float,
    best_so_far: Optional[float] = None,
    neighborhood_result: Optional[Dict[str, Any]] = None,
) -> None:
    """Show trial in compact 4-box format."""
    is_best = best_so_far is not None and score >= float(best_so_far)
    star = "★" if is_best else ""
    score_c = _score_color(is_best)
    
    # Info
    info_rows = [("T#", str(trial_num)), ("SCORE", f"[{score_c}]{_format_score(score)}{star}[/]")]
    if best_so_far and not is_best:
        info_rows.append(("BEST", f"{_format_score(best_so_far)}"))
    
    # Perf
    perf_rows = [
        ("WR", f"{M.get(metrics, 'winrate'):.1f}%"),
        ("SHP", f"{M.get(metrics, 'sharpe'):.2f}"),
        ("SQN", f"{M.get(metrics, 'sqn'):.2f}"),
        ("PF", f"{M.get(metrics, 'profit_factor'):.2f}"),
        ("DD", f"{M.get(metrics, 'drawdown'):.1f}%"),
    ]
    
    # Fin
    saldo_final = M.get(metrics, "saldo_actual", saldo_inicial)
    pnl = saldo_final - saldo_inicial
    roi = M.get(metrics, "roi")
    pnl_c = _pnl_color(roi)
    
    fin_rows = [
        ("PNL", f"[{pnl_c}]{'+' if pnl >= 0 else ''}${pnl:,.0f}[/]"),
        ("ROI", f"[{pnl_c}]{roi:.1f}%[/]"),
        ("FIN", f"${saldo_final:,.0f}"),
        ("TRD", str(M.get_int(metrics, "total_trades"))),
    ]
    
    # Box 4
    if neighborhood_result:
        if hasattr(neighborhood_result, "to_dict"):
            neighborhood_result = neighborhood_result.to_dict()
        agg = neighborhood_result.get("aggregated_score", 0.0)
        n_ok = neighborhood_result.get("n_neighbors_successful", 0)
        n_t = neighborhood_result.get("n_neighbors_tested", 0)
        disp = neighborhood_result.get("avg_dispersion", 0.0) * 100
        approved = neighborhood_result.get("robustness_approved", False)
        status = "●" if approved else "○"
        nbh_rows = [("AGG", f"{agg:.2f}"), ("NEI", f"{n_ok}/{n_t}"), 
                    ("DSP", f"{disp:.0f}%"), ("", status)]
        box4 = crear_mini_box("NBH", nbh_rows, 14)
    else:
        exit_t = str(params.get("__exit_type", params.get("exit_type", "FIX")))[:3].upper()
        sl = params.get("__exit_sl_pct", params.get("exit_sl_pct", 0.0))
        tp = params.get("__exit_tp_pct", params.get("exit_tp_pct", 0.0))
        box4 = crear_mini_box("EXIT", [("TYPE", exit_t), ("SL", f"{sl:.1f}%"), ("TP", f"{tp:.1f}%")], 14)
    
    mostrar_fila_boxes(
        crear_mini_box("INFO", info_rows, 16),
        crear_mini_box("PERF", perf_rows, 16),
        crear_mini_box("FIN", fin_rows, 18),
        box4
    )


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
    """Legacy wrapper."""
    mostrar_panel_elegante(
        metrics=metrics, params=params, score=score, trial_num=trial_num,
        saldo_inicial=saldo_inicial, indicadores_activos=indicadores_activos,
        combo_str=combo_str, activo=activo,
        best_so_far=params.get("__best_score_so_far"),
    )


# Alias for MetricMapper
MetricMapper = M


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "THEME", "MetricMapper", "M",
    "mostrar_panel_elegante", "mostrar_panel_rich",
    "mostrar_resultado_vecindario",
    "mostrar_top_trials",
    "mostrar_fin_optimizacion", 
    "mostrar_cabecera_inicio",
    "EstadisticasOptimizacion",
    "resetear_estadisticas",
    "actualizar_estadisticas",
    "mostrar_evolucion_metricas",
    "mostrar_evolucion_inline",
    "mostrar_fila_boxes",
    "crear_mini_box",
    "mostrar_trial_compacto",
]
