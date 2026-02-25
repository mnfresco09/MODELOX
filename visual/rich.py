"""
================================================================================
VISUAL/RICH.PY — INTERFAZ DE TERMINAL INSTITUCIONAL
================================================================================

PROPÓSITO:
    Renderizado de métricas y tablas en terminal con diseño limpio y profesional.
    Usa la librería `rich` para paneles, tablas y colores semánticos.

ESTILO:
    - Neutral y elegante (Gris/Azul/Dorado).
    - Evita el "arcoíris" excesivo.
    - Resalta solo lo importante (ROI positivo, Best Trial).

COMPONENTES:
    - `mostrar_panel_elegante`: Dashboard principal por trial.
    - `mostrar_cabecera_inicio`: Resumen de configuración al arrancar.
    - `mostrar_evolucion_inline`: Barra de progreso minimalista.

================================================================================
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
        "duration_mean_min": ("duration_mean_min", "duracion_media_min", "avg_duration_min"),
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


def _format_duration_min(duration_min: float) -> str:
    """Formatea duración media: <60 => XMIN, >=60 => XH:YMIN."""
    mins = max(0, int(round(float(duration_min))))
    if mins < 60:
        return f"{mins}MIN"
    h, m = divmod(mins, 60)
    return f"{h}H:{m:02d}MIN"


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
    # Rango de meses para datos sintéticos
    rango_meses: Optional[tuple] = None,  # Ej: (60, 66) para "MES 60 A 66"
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
    duration_mean_min = M.get(metrics, "duration_mean_min")
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
    
    # Construir string de rango de meses para datos sintéticos
    rango_str = ""
    if rango_meses is not None:
        mes_ini, mes_fin = rango_meses
        rango_str = f"[{THEME.GOLD}]MES {mes_ini}-{mes_fin}[/]  │  "
    
    # Construir info de fase si está disponible
    phase_str = ""
    if phase_name:
        phase_str = f"[bold {THEME.BLUE}]{phase_name}[/]  │  "
    
    # Mostrar progreso de fase en lugar del número de trial global
    trial_display = f"[{THEME.WHITE}]{trial_num}[/]"
    if phase_progress:
        trial_display = f"[bold {THEME.WHITE}]{phase_progress}[/]"
    
    # Header normal (con SCORE)
    header = (f"{rango_str}{asset_str}  │  {strat}  │  "
              f"TF {tf}  │  {phase_str}TRIAL {trial_display}  │  "
              f"[bold {score_c}]SCORE {_format_score(score)}{star}[/]{best_str}")
    
    # Header para best trial (sin SCORE, se muestra aparte)
    header_no_score = (f"{rango_str}{asset_str}  │  {strat}  │  "
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
    perf_grid.add_row("DUR. MEDIA", _format_duration_min(duration_mean_min))
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
        for k, v in list(strategy_params.items())[:12]:
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
        self.n_total = 0

    def actualizar(self, metricas: Dict[str, Any], score: float, trial: int) -> None:
        self.n_total += 1
        if metricas:
            self.valores["roi"].append(M.get(metricas, "roi"))
            self.valores["sqn"].append(M.get(metricas, "sqn"))
            self.valores["exp"].append(M.get(metricas, "expectativa"))
            self.valores["sharpe"].append(M.get(metricas, "sharpe"))
            self.valores["score"].append(score)
            self.valores["pf"].append(M.get(metricas, "profit_factor"))
            self.n += 1  # Only increment valid trials for averages
            if score > self.mejor_score:
                self.mejor_score = score
                self.mejor_trial = trial

    def promedio(self, key: str) -> float:
        v = self.valores.get(key, [])
        valid = [x for x in v if x == x and abs(x) != float('inf')]
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
            "n_total": self.n_total,
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
    if not forzar and (_stats.n_total == 0 or _stats.n_total % mostrar_cada_n != 0):
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
    
    failed = s['n_total'] - s['n_trials']
    title_extra = f" | {failed} FAIL" if failed > 0 else ""
    
    evo_box = Panel(
        grid,
        title=f"[{THEME.MUTED}]SUMMARY ({s['n_trials']} OK{title_extra})[/]",
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
    
    table.add_column("#", style=THEME.MUTED, justify="center")
    table.add_column("SCORE", style=THEME.TEXT, justify="right", width=8)
    table.add_column("ROI", style=THEME.TEXT, justify="right", width=8)
    table.add_column("WIN", style=THEME.TEXT, justify="right", width=6)
    table.add_column("DD", style=THEME.TEXT, justify="right", width=6)
    table.add_column("BALANCE", style=THEME.TEXT, justify="right", width=12)
    table.add_column("T", style=THEME.MUTED, justify="center")
    
    for i, t in enumerate(top):
        score = t.value or 0.0
        roi = M.from_trial(t, "roi")
        wr = M.from_trial(t, "winrate")
        dd = M.from_trial(t, "drawdown")
        bal = M.from_trial(t, "saldo_actual")
        trades = int(M.from_trial(t, "total_trades"))
        
        mark = str(t.number)
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
    strategy_id: Optional[int] = None,
    archivo_data: str = "",
    timeframe: str = "",
    timeframe_exit: str = "",
    periodo: str = "",
    exit_type: str = "FIXED",
    strategy_exit_enabled: bool = False,

    sampler_type: str = "CMA",
    synthetic_mode: bool = False,
    synthetic_years: int = 0,
    perturbacion: bool = False,
) -> None:
    """Display elegant startup header with comprehensive info - centered, white text."""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    icons = {"BTC": "₿", "GOLD": "◆", "SP500": "◇", "NASDAQ": "□"}
    icon = icons.get(activo.upper(), "○")
    
    exit_d = "CUSTOM" if strategy_exit_enabled else exit_type.upper()
    
    # Main info grid - centered alignment
    info_grid = Table.grid(padding=(0, 3))
    info_grid.add_column(style=THEME.MUTED, width=14, justify="right")
    info_grid.add_column(style=THEME.WHITE, width=32, justify="left")
    
    info_grid.add_row("ASSET", f"[bold]{icon} {activo.upper()}[/]")

    # Mostrar nombre de estrategia + ID (si existe)
    sid = strategy_id
    if sid is None:
        try:
            import re as _re
            _m = _re.search(r"ID(\d+)", str(combo_nombre).upper())
            sid = int(_m.group(1)) if _m else None
        except Exception:
            sid = None

    strategy_display = combo_nombre.upper()
    if sid is not None:
        strategy_display = f"{strategy_display} [ID{sid}]"
    info_grid.add_row("STRATEGY", f"{strategy_display[:32]}")
    # Mostrar Timeframe: ENTRY X | EXIT X
    tf_entry_str = timeframe.upper() if timeframe else "1M"
    tf_exit_str = timeframe_exit.upper() if timeframe_exit else tf_entry_str
    tf_str = f"ENTRY {tf_entry_str} | EXIT {tf_exit_str}"
    info_grid.add_row("TIMEFRAME", tf_str)
        
    if periodo:
        p = periodo.replace(" UTC", "").strip().upper()[:32]
        info_grid.add_row("PERIOD", f"{p}")
    info_grid.add_row("", "")
    info_grid.add_row("EXIT MODE", f"{exit_d}")
    info_grid.add_row("TRIALS", f"[bold]{n_trials}[/]")
    
    # Sin color para ON/OFF de perturbación (solicitado)
    pert_status = "ON" if perturbacion else "OFF"
    info_grid.add_row("PERTURBACION", pert_status)

    
    if indicadores:
        inds = " ".join([i[:6].upper() for i in indicadores[:4]])
        if len(indicadores) > 4:
            inds += f" +{len(indicadores)-4}"
        info_grid.add_row("PARAMS", f"{inds}")
    
    # DATA source - show generated (Quant-GAN) or real
    if synthetic_mode:
        tf_upper = timeframe.upper() if timeframe else "1M"
        futuro_label = f"[bold {THEME.WHITE}]FUTURO/{activo.upper()}_{tf_upper}[/]"
        info_grid.add_row("DATA", futuro_label)
    elif archivo_data:
        info_grid.add_row("DATA", f"{archivo_data[-32:].upper()}")
    
    # Main panel - white title with sampler type, centered
    sampler_label = sampler_type.upper() if sampler_type else "CMA"
    header_panel = Panel(
        info_grid,
        title=f"[bold {THEME.WHITE}]═══ MODELOX OPTIMIZATION {sampler_label} ═══[/]",
        subtitle=f"[{THEME.DIM}]V3.0[/]",
        border_style=THEME.BORDER,
        box=box.DOUBLE,
        padding=(1, 2),
        width=56
    )
    
    console.print()
    console.print(header_panel, justify="center")
    console.print()
    
    # Notificar Telegram
    try:
        from visual.telegram import notificar_inicio_optimizacion
        # Extraer strategy_id del nombre si es posible
        import re as _re
        _id_match = _re.search(r'ID(\d+)', combo_nombre.upper())
        _sid = int(_id_match.group(1)) if _id_match else 0
        notificar_inicio_optimizacion(
            estrategia=combo_nombre,
            strategy_id=_sid,
            activo=activo,
            n_trials=n_trials,
            timeframe=timeframe,
            sampler=sampler_type,
        )
    except Exception:
        pass

def mostrar_fin_optimizacion(
    total_trials: int,
    best_score: float,
    best_trial: int,
    estrategia: str = "",
    activo: str = "",
    roi_medio: float = 0.0,
    best_roi: float = 0.0,
    excel_report: Optional[str] = None,
    **kwargs,
) -> None:
    """Display optimization completion and notify telegram."""
    try:
        from visual.telegram import notificar_fin_optimizacion
        notificar_fin_optimizacion(
            estrategia=estrategia,
            activo=activo,
            n_trials=total_trials,
            best_score=best_score,
            best_trial=best_trial,
            roi_medio=roi_medio,
            best_roi=best_roi,
            excel_report=excel_report,
        )
    except Exception:
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



# Alias for MetricMapper
MetricMapper = M



# ============================================================================
# ELEGANT RICH REPORTER - Reporter para integración con OptimizationRunner
# ============================================================================
from dataclasses import dataclass, field
from typing import Protocol, Any


class ReporterProtocol(Protocol):
    """Protocolo base para reporters."""
    def needs_dataframe(self, score: float) -> bool: ...
    def on_trial_end(self, artifacts: Any) -> None: ...
    def on_strategy_end(self, strategy_name: str, study: Any) -> None: ...


@dataclass
class ElegantRichReporter:
    """
    Bloomberg/TradingView-style Rich console reporter.
    
    Muestra paneles elegantes con métricas de cada trial y resumen al final.
    """

    saldo_inicial: float = 300.0
    activo: str = ""
    n_trials_total: int = 0
    mostrar_evolucion_cada: int = 50
    mostrar_evolucion_inline: bool = False
    _best_score: float = field(default=float("-inf"), init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)
    _notified_pcts: set = field(default_factory=set, init=False, repr=False)

    def needs_dataframe(self, score: float) -> bool:
        return False

    def on_trial_end(self, artifacts) -> None:
        if not self._initialized:
            resetear_estadisticas()
            self._initialized = True

        current_score = artifacts.score or 0
        best_so_far = None

        _was_best = False
        if current_score > self._best_score:
            self._best_score = current_score
            _was_best = True
        best_so_far = self._best_score if self._best_score > float("-inf") else None

        indicadores = list(getattr(artifacts, "indicators_used", []))
        if not indicadores and artifacts.params:
            raw = artifacts.params.get("__indicators_used", [])
            if isinstance(raw, (list, tuple)):
                indicadores = [str(x) for x in raw if x]

        metrics = dict(artifacts.metrics or {})
        try:
            eq = getattr(artifacts, "equity_curve", None)
            if isinstance(eq, (list, tuple)) and len(eq) > 0:
                if "saldo_max" not in metrics or metrics.get("saldo_max") in (None, 0, 0.0):
                    metrics["saldo_max"] = float(max(eq))
        except Exception:
            pass

        actualizar_estadisticas(metrics, current_score, artifacts.trial_number)

        tf_entry = ""
        tf_exit = ""
        if artifacts.params:
            tf_entry = str(artifacts.params.get("__timeframe_entry", ""))
            tf_exit = str(artifacts.params.get("__timeframe_exit", ""))

        params_for_reporting = artifacts.params.copy() if artifacts.params else {}
        
        # Extraer rango de meses para datos sintéticos
        rango_meses = params_for_reporting.get('__rango_meses', None)
        sampler_used = params_for_reporting.get('__sampler_used', "")

        mostrar_panel_elegante(
            metrics=metrics,
            params=params_for_reporting,
            score=artifacts.score or 0,
            trial_num=artifacts.trial_number,
            saldo_inicial=self.saldo_inicial,
            indicadores_activos=indicadores,
            combo_str=artifacts.strategy_name or "",
            activo=self.activo,
            best_so_far=best_so_far,
            timeframe_entry=tf_entry,
            timeframe_exit=tf_exit,
            rango_meses=rango_meses,
            phase_name=sampler_used,
        )
        
        # Notificar Telegram si es nuevo best
        if _was_best:
            try:
                from visual.telegram import notificar_nuevo_best
                notificar_nuevo_best(
                    score=current_score,
                    trial=artifacts.trial_number,
                    roi=M.get(metrics, "roi"),
                    saldo_final=M.get(metrics, "saldo_actual", self.saldo_inicial),
                    saldo_inicial=self.saldo_inicial,
                    drawdown=M.get(metrics, "drawdown"),
                    sqn=M.get(metrics, "sqn"),
                    winrate=M.get(metrics, "winrate"),
                    sharpe=M.get(metrics, "sharpe"),
                    profit_factor=M.get(metrics, "profit_factor"),
                    trades_dia=M.get(metrics, "trades_por_dia"),
                    n_longs=M.get_int(metrics, "n_trades_long"),
                    n_shorts=M.get_int(metrics, "n_trades_short"),
                    n_trials_total=self.n_trials_total,
                )
            except Exception:
                pass
        
        # Notificar Telegram progreso cada X%
        if self.n_trials_total > 0:
            pct = int(100 * _stats.n / self.n_trials_total)
            # Usar intervalo configurable desde telegram.py
            try:
                from visual.telegram import PROGRESO_CADA_PCT
                step = PROGRESO_CADA_PCT
            except Exception:
                step = 10
            for milestone in range(step, 100, step):
                if pct >= milestone and milestone not in self._notified_pcts:
                    self._notified_pcts.add(milestone)
                    try:
                        from visual.telegram import notificar_progreso_optimizacion
                        s = _stats.obtener_promedios()
                        notificar_progreso_optimizacion(
                            pct_completado=milestone,
                            trial_actual=_stats.n,
                            n_trials_total=self.n_trials_total,
                            roi_medio=s["roi_medio"],
                            pf_medio=s["pf_medio"],
                            sharpe_medio=s["sharpe_medio"],
                            sqn_medio=s["sqn_medio"],
                            score_medio=s["score_medio"],
                            best_score=s["mejor_score"],
                            best_trial=s["mejor_trial"],
                        )
                    except Exception:
                        pass

    def on_strategy_end(self, strategy_name: str, study) -> None:
        self._initialized = False
        if study and hasattr(study, 'trials') and study.trials:
            mostrar_evolucion_metricas(forzar=True)
            mostrar_top_trials(study, n=5)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "THEME", "MetricMapper", "M",
    "mostrar_panel_elegante",
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
    # Reporter
    "ElegantRichReporter",
    # TimeGAN Training
    "TimeGANTrainingDisplay",
    "mostrar_timegan_header",
]


# ============================================================================
# TIMEGAN TRAINING DISPLAY
# ============================================================================

class TimeGANTrainingDisplay:
    """
    Display elegante para el entrenamiento de TimeGAN.
    Muestra progreso de las 3 fases de entrenamiento con información ultra-detallada.
    Incluye: tiempos, velocidad, ETA, progreso 100%.
    """
    
    def __init__(
        self, 
        activo: str, 
        timeframe: str, 
        total_epochs: int,
        source_file: str = "",
        output_file: str = "",
        years_to_generate: int = 100,
        real_data_rows: int = 0,
        last_timestamp: str = "",
        last_close: float = 0.0,
        device: str = "cpu",
        use_mixed_precision: bool = False,
        use_compile: bool = False,
    ):
        import time as _time
        self.activo = activo.upper()
        self.timeframe = timeframe.upper()
        self.total_epochs = total_epochs
        self.source_file = source_file
        self.output_file = output_file
        self.years_to_generate = years_to_generate
        self.real_data_rows = real_data_rows
        self.last_timestamp = last_timestamp
        self.last_close = last_close
        self.device = device.upper()
        self.use_mixed_precision = use_mixed_precision
        self.use_compile = use_compile
        self.current_phase = 0
        self.phases = ["EMBEDDING", "SUPERVISOR", "ADVERSARIAL"]
        self.phase_epochs = total_epochs // 3
        self.phase_losses = {1: [], 2: [], 3: []}
        
        # Timing
        self._time = _time
        self.training_start_time: float = 0.0
        self.phase_start_times: Dict[int, float] = {}
        self.phase_end_times: Dict[int, float] = {}
        self.generation_start_time: float = 0.0
        
    def show_header(self) -> None:
        """Muestra cabecera de inicio de entrenamiento con toda la info."""
        import os
        from datetime import datetime
        os.system('cls' if os.name == 'nt' else 'clear')
        
        self.training_start_time = self._time.time()
        
        icons = {"BTC": "₿", "GOLD": "◆", "SP500": "◇", "NASDAQ": "□"}
        icon = icons.get(self.activo, "○")
        
        # Grid principal
        info_grid = Table.grid(padding=(0, 2))
        info_grid.add_column(style=THEME.MUTED, width=16, justify="right")
        info_grid.add_column(style=THEME.WHITE, width=42, justify="left")
        
        info_grid.add_row("ASSET", f"[bold cyan]{icon} {self.activo}[/]")
        info_grid.add_row("TIMEFRAME", f"[bold]{self.timeframe}[/]")
        info_grid.add_row("STARTED", f"[white]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/]")
        info_grid.add_row("", "")
        
        # Hardware/Aceleración
        info_grid.add_row("[dim]─── HARDWARE ───[/]", "")
        info_grid.add_row("DEVICE", f"[bold yellow]{self.device}[/]")
        
        accel_parts = []
        if self.use_mixed_precision:
            accel_parts.append("[green]FP16[/]")
        if self.use_compile:
            accel_parts.append("[green]COMPILE[/]")
        accel_str = " + ".join(accel_parts) if accel_parts else "[dim]NONE[/]"
        info_grid.add_row("ACCELERATION", accel_str)
        
        info_grid.add_row("", "")
        
        # Archivos
        info_grid.add_row("[dim]─── SOURCE ───[/]", "")
        source_short = self.source_file[-45:] if len(self.source_file) > 45 else self.source_file
        info_grid.add_row("INPUT FILE", f"[white]{source_short}[/]")
        info_grid.add_row("ROWS", f"[white]{self.real_data_rows:,}[/]")
        if self.last_timestamp:
            info_grid.add_row("LAST TIMESTAMP", f"[white]{str(self.last_timestamp)[:19]}[/]")
        if self.last_close > 0:
            info_grid.add_row("LAST CLOSE", f"[bold green]${self.last_close:,.2f}[/]")
        
        info_grid.add_row("", "")
        info_grid.add_row("[dim]─── OUTPUT ───[/]", "")
        output_short = self.output_file[-45:] if len(self.output_file) > 45 else self.output_file
        info_grid.add_row("OUTPUT FILE", f"[cyan]{output_short}[/]")
        info_grid.add_row("YEARS TO GEN", f"[bold cyan]{self.years_to_generate}[/]")
        
        info_grid.add_row("", "")
        info_grid.add_row("[dim]─── TRAINING ───[/]", "")
        info_grid.add_row("TOTAL EPOCHS", f"[bold]{self.total_epochs}[/]")
        info_grid.add_row("EPOCHS/PHASE", f"{self.phase_epochs}")
        info_grid.add_row("", "")
        info_grid.add_row("PHASE 1", f"[cyan]EMBEDDING[/]     [dim](Autoencoder)[/]")
        info_grid.add_row("PHASE 2", f"[cyan]SUPERVISOR[/]    [dim](Temporal)[/]")
        info_grid.add_row("PHASE 3", f"[cyan]ADVERSARIAL[/]   [dim](GAN D:G 2:1)[/]")
        
        header_panel = Panel(
            info_grid,
            title=f"[bold cyan]═══ TimeGAN TRAINING ═══[/]",
            subtitle=f"[{THEME.DIM}]SYNTHETIC DATA GENERATOR[/]",
            border_style="cyan",
            box=box.DOUBLE,
            padding=(1, 2),
            width=64
        )
        
        console.print()
        console.print(header_panel, justify="center")
        console.print()
    
    def show_phase_start(self, phase_num: int, phase_name: str) -> None:
        """Muestra inicio de una fase con separador y timestamp."""
        from datetime import datetime
        self.current_phase = phase_num
        self.phase_start_times[phase_num] = self._time.time()
        
        console.print()
        phase_text = Text()
        phase_text.append(f"  {'─' * 20} ", style=THEME.DIM)
        phase_text.append(f"PHASE {phase_num}/3: ", style="bold white")
        phase_text.append(phase_name.upper(), style="bold cyan")
        phase_text.append(f" {'─' * 20}", style=THEME.DIM)
        console.print(phase_text)
        console.print()
        
        # Descripción de la fase
        descriptions = {
            1: "Learning latent space representation of OHLCV data",
            2: "Training temporal dynamics supervisor",
            3: "Adversarial training (Generator vs Discriminator, ratio 2:1)",
        }
        desc = descriptions.get(phase_num, "")
        now = datetime.now().strftime('%H:%M:%S')
        console.print(f"    [dim italic]{desc}[/]")
        console.print(f"    [dim]Started: {now} | Target: {self.phase_epochs} epochs[/]")
        console.print()
    
    def show_phase_progress(self, epoch: int, loss: float, extra: str = "") -> None:
        """Muestra progreso dentro de una fase con barra, métricas, velocidad y ETA."""
        self.phase_losses[self.current_phase].append(loss)
        
        bar_width = 18
        progress = epoch / self.phase_epochs
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)
        pct = progress * 100
        
        # Calcular tendencia de loss
        losses = self.phase_losses[self.current_phase]
        if len(losses) >= 2:
            trend = losses[-1] - losses[-2]
            if trend < 0:
                trend_str = f"[green]↓{abs(trend):.4f}[/]"
            else:
                trend_str = f"[red]↑{trend:.4f}[/]"
        else:
            trend_str = "[dim]--[/]"
        
        # Mejor loss de la fase
        best_loss = min(losses) if losses else loss
        
        # Calcular velocidad y ETA
        elapsed = self._time.time() - self.phase_start_times.get(self.current_phase, self._time.time())
        speed = epoch / elapsed if elapsed > 0 else 0
        remaining_epochs = self.phase_epochs - epoch
        eta_seconds = remaining_epochs / speed if speed > 0 else 0
        
        # Formatear ETA
        if eta_seconds < 60:
            eta_str = f"{eta_seconds:.0f}s"
        elif eta_seconds < 3600:
            eta_str = f"{eta_seconds/60:.1f}m"
        else:
            eta_str = f"{eta_seconds/3600:.1f}h"
        
        # Construir línea completa con Rich markup
        extra_part = f" │ {extra}" if extra else ""
        
        line = (
            f"    [cyan][{bar}][/] "
            f"[white]{epoch:4d}/{self.phase_epochs}[/] "
            f"[dim]({pct:5.1f}%)[/] "
            f"│ [dim]L:[/][cyan]{loss:.4f}[/] "
            f"│ [dim]B:[/][green]{best_loss:.4f}[/] "
            f"│ {trend_str} "
            f"│ [dim]{speed:.1f}ep/s[/] "
            f"│ [yellow]ETA {eta_str}[/]"
            f"{extra_part}"
        )
        
        console.print(line)
    
    def show_phase_complete(self, phase_name: str, final_loss: float) -> None:
        """Muestra finalización de una fase con resumen detallado y tiempos."""
        self.phase_end_times[self.current_phase] = self._time.time()
        
        console.print()
        console.print()
        
        losses = self.phase_losses[self.current_phase]
        initial_loss = losses[0] if losses else final_loss
        best_loss = min(losses) if losses else final_loss
        improvement = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0
        
        # Calcular duración de la fase
        phase_duration = self.phase_end_times[self.current_phase] - self.phase_start_times.get(self.current_phase, self.phase_end_times[self.current_phase])
        if phase_duration < 60:
            duration_str = f"{phase_duration:.1f}s"
        elif phase_duration < 3600:
            duration_str = f"{phase_duration/60:.1f}m"
        else:
            duration_str = f"{phase_duration/3600:.1f}h"
        
        # Velocidad promedio
        avg_speed = self.phase_epochs / phase_duration if phase_duration > 0 else 0
        
        # Box de resumen
        summary_grid = Table.grid(padding=(0, 2))
        summary_grid.add_column(style=THEME.MUTED, width=14, justify="right")
        summary_grid.add_column(style=THEME.WHITE, width=20, justify="left")
        
        summary_grid.add_row("STATUS", "[bold green]✓ COMPLETE[/]")
        summary_grid.add_row("DURATION", f"[bold]{duration_str}[/]")
        summary_grid.add_row("AVG SPEED", f"[bold]{avg_speed:.1f} ep/s[/]")
        summary_grid.add_row("", "")
        summary_grid.add_row("INITIAL LOSS", f"{initial_loss:.6f}")
        summary_grid.add_row("FINAL LOSS", f"[cyan]{final_loss:.6f}[/]")
        summary_grid.add_row("BEST LOSS", f"[green]{best_loss:.6f}[/]")
        summary_grid.add_row("IMPROVEMENT", f"[bold green]{improvement:+.1f}%[/]")
        
        summary_panel = Panel(
            summary_grid,
            title=f"[green]{phase_name.upper()} COMPLETE[/]",
            border_style="green",
            box=box.ROUNDED,
            padding=(0, 1),
            width=42
        )
        
        console.print(summary_panel, justify="center")
    
    def show_generation_start(self, total_candles: int, candles_per_year: int) -> None:
        """Muestra inicio de generación de datos con tiempo."""
        from datetime import datetime
        self.generation_start_time = self._time.time()
        
        console.print()
        console.print(f"  [bold cyan]{'─' * 20} GENERATING DATA {'─' * 20}[/]")
        console.print()
        console.print(f"    [dim]Started:[/]                   [white]{datetime.now().strftime('%H:%M:%S')}[/]")
        console.print(f"    [dim]Total candles to generate:[/] [bold]{total_candles:,}[/]")
        console.print(f"    [dim]Candles per year:[/]          [bold]{candles_per_year:,}[/]")
        console.print()
    
    def show_generation_progress(self, years_done: int, total_years: int) -> None:
        """Muestra progreso de generación de datos."""
        bar_width = 30
        progress = years_done / total_years
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)
        pct = progress * 100
        
        line = f"    [cyan][{bar}][/] [white]{years_done}/{total_years}[/] years [dim]({pct:.0f}%)[/]"
        console.print(line, end="\r")
    
    def show_complete(self, output_path: str, file_size_mb: float, 
                      start_ts: str = "", end_ts: str = "", total_candles: int = 0) -> None:
        """Muestra finalización del entrenamiento y generación con resumen total."""
        from datetime import datetime
        console.print()
        console.print()
        
        # Calcular tiempos totales
        total_time = self._time.time() - self.training_start_time
        gen_time = self._time.time() - self.generation_start_time if self.generation_start_time > 0 else 0
        training_time = total_time - gen_time
        
        # Formatear tiempos
        def fmt_time(seconds: float) -> str:
            if seconds < 60:
                return f"{seconds:.1f}s"
            elif seconds < 3600:
                return f"{seconds/60:.1f}m"
            else:
                return f"{seconds/3600:.1f}h"
        
        result_grid = Table.grid(padding=(0, 2))
        result_grid.add_column(style=THEME.MUTED, width=16, justify="right")
        result_grid.add_column(style=THEME.WHITE, width=40, justify="left")
        
        result_grid.add_row("STATUS", "[bold green]✓ SYNTHETIC DATA READY[/]")
        result_grid.add_row("COMPLETED", f"[white]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/]")
        result_grid.add_row("", "")
        
        # Tiempos
        result_grid.add_row("[dim]─── TIMING ───[/]", "")
        result_grid.add_row("TRAINING TIME", f"[bold yellow]{fmt_time(training_time)}[/]")
        result_grid.add_row("GENERATION TIME", f"[bold yellow]{fmt_time(gen_time)}[/]")
        result_grid.add_row("TOTAL TIME", f"[bold green]{fmt_time(total_time)}[/]")
        result_grid.add_row("", "")
        
        # Output
        result_grid.add_row("[dim]─── OUTPUT ───[/]", "")
        result_grid.add_row("FILE", f"[cyan]{output_path[-40:]}[/]")
        result_grid.add_row("SIZE", f"[bold]{file_size_mb:.1f} MB[/]")
        if total_candles:
            result_grid.add_row("CANDLES", f"[bold]{total_candles:,}[/]")
            # Calcular velocidad de generación
            gen_speed = total_candles / gen_time if gen_time > 0 else 0
            result_grid.add_row("GEN SPEED", f"[dim]{gen_speed:,.0f} candles/s[/]")
        
        result_grid.add_row("", "")
        result_grid.add_row("[dim]─── DATA RANGE ───[/]", "")
        if start_ts:
            result_grid.add_row("START", f"{str(start_ts)[:19]}")
        if end_ts:
            result_grid.add_row("END", f"{str(end_ts)[:19]}")
        
        result_panel = Panel(
            result_grid,
            title=f"[bold green]═══ GENERATION COMPLETE ═══[/]",
            border_style="green",
            box=box.DOUBLE,
            padding=(1, 2),
            width=64
        )
        
        console.print(result_panel, justify="center")
        console.print()


def mostrar_timegan_header(
    activo: str,
    timeframe: str,
    years: int,
    real_data_path: str,
) -> None:
    """Muestra cabecera simple para inicio de generación TimeGAN."""
    icons = {"BTC": "₿", "GOLD": "◆", "SP500": "◇", "NASDAQ": "□"}
    icon = icons.get(activo.upper(), "○")
    
    console.print()
    console.print(f"  [bold cyan]{'═' * 50}[/]", justify="center")
    console.print(f"  [bold cyan]   🧠 TimeGAN - SYNTHETIC DATA GENERATOR[/]", justify="center")
    console.print(f"  [bold cyan]{'═' * 50}[/]", justify="center")
    console.print()
    console.print(f"  [grey50]ASSET:[/]      [white]{icon} {activo.upper()}[/]")
    console.print(f"  [grey50]TIMEFRAME:[/]  [white]{timeframe.upper()}[/]")
    console.print(f"  [grey50]YEARS:[/]      [white]{years}[/]")
    console.print(f"  [grey50]SOURCE:[/]     [white]{real_data_path[-40:]}[/]")
    console.print()
