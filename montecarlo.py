#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  MODELOX MONTE CARLO ROBUSTNESS VALIDATOR                    ║
║                                                                              ║
║  Validates strategy robustness by testing on thousands of synthetic markets  ║
║  generated through price perturbation and block bootstrapping.               ║
║                                                                              ║
║  Methods:                                                                    ║
║    • NOISE INJECTION: Adds gaussian noise to OHLC prices                     ║
║    • BLOCK BOOTSTRAP: Shuffles price blocks preserving trend structure       ║
║                                                                              ║
║  If your strategy survives in >70% of synthetic markets = ROBUST             ║
║  If it fails with tiny price changes = OVERFITTED                            ║
║                                                                              ║
║  Usage: python montecarlo.py                                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")

# =============================================================================
# RICH IMPORTS
# =============================================================================
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.text import Text
    from rich import box
    from rich.columns import Columns
    from rich.align import Align
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console(force_terminal=True, color_system="truecolor") if RICH_AVAILABLE else None

# =============================================================================
# MATPLOTLIB IMPORTS
# =============================================================================
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.patheffects as path_effects
    from scipy import stats as scipy_stats
    from scipy.ndimage import gaussian_filter
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# =============================================================================
# MODELOX IMPORTS
# =============================================================================
try:
    from modelox.core.types import BacktestConfig, Strategy, normalize_timeframe_to_suffix
    from modelox.core.engine import BacktestParams, calculate_performance_vectorized_numba
    from modelox.core.metrics import resumen_metricas
    from modelox.core.exits import ExitSettings, _normalize_exit_values
    from modelox.strategies.registry import discover_strategies
except ImportError as e:
    print(f"❌ Error importando módulos ModeloX: {e}")
    sys.exit(1)

# =============================================================================
# VISUAL IMPORTS
# =============================================================================
try:
    from visual.grafico import plot_trades
    GRAFICO_AVAILABLE = True
except ImportError:
    GRAFICO_AVAILABLE = False


# =============================================================================
# THEME
# =============================================================================
class Theme:
    # Background
    BG_PRIMARY = "#0d1117"
    BG_SECONDARY = "#161b22"
    BG_CARD = "#1c2128"
    
    # Text
    TEXT_PRIMARY = "#e6edf3"
    TEXT_SECONDARY = "#8b949e"
    TEXT_MUTED = "#6e7681"
    
    # Accents
    ACCENT = "#58a6ff"
    GREEN = "#3fb950"
    RED = "#f85149"
    ORANGE = "#d29922"
    PURPLE = "#a371f7"
    CYAN = "#39c5cf"
    GOLD = "#f0c000"
    
    # Border
    BORDER = "#30363d"


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class MonteCarloConfig:
    """Configuración para Monte Carlo Robustness Validator."""
    
    # Strategy
    combinacion_id: int = 15
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    
    # Exit Configuration
    exit_type: str = "pnl_fixed"
    exit_sl_pct: float = 20.0
    exit_tp_pct: float = 30.0
    exit_trail_act_pct: float = 15.0
    exit_trail_dist_pct: float = 3.0
    
    # Backtest Configuration
    saldo_inicial: float = 1000.0
    saldo_operativo_max: float = 1000000.0
    saldo_minimo_operativo: float = 300.0
    comision_pct: float = 0.00045
    comision_sides: int = 1
    saldo_usado: float = 75.0
    apalancamiento_max: float = 60.0
    qty_max_activo: float = float("inf")
    
    # Data
    activo: str = "BTC"
    timeframe: str = "1m"
    fecha_inicio: str = "2020-01-01"
    fecha_fin: str = "2024-09-18"
    data_dir: str = "data/ohlcv"
    
    # Monte Carlo Settings
    n_simulations: int = 100
    noise_pct: float = 0.1  # 0.1% noise range (-0.1% to +0.1%)
    block_size: int = 1440  # 1 day in 1m bars (24*60)
    methods: List[str] = field(default_factory=lambda: ["noise", "block_bootstrap"])
    
    # Output
    output_dir: str = "resultados"
    export_csv: bool = True
    export_excel: bool = True
    export_pdf: bool = True
    show_plots: bool = True
    verbose: bool = True


# =============================================================================
# SYNTHETIC MARKET GENERATORS
# =============================================================================
class SyntheticMarketGenerator:
    """Generates synthetic market data for Monte Carlo testing."""
    
    def __init__(self, df: pl.DataFrame, rng: np.random.Generator):
        self.df = df
        self.rng = rng
        self.n_rows = df.height
        
        # Pre-extract arrays for speed
        self.open_orig = df["open"].to_numpy()
        self.high_orig = df["high"].to_numpy()
        self.low_orig = df["low"].to_numpy()
        self.close_orig = df["close"].to_numpy()
    
    def generate_noise(self, noise_pct: float) -> pl.DataFrame:
        """
        Inyecta ruido gaussiano en los precios OHLC.
        
        Cada precio se multiplica por (1 + noise) donde noise ~ U(-noise_pct, +noise_pct)
        Mantiene la coherencia OHLC: high >= max(open,close), low <= min(open,close)
        """
        noise_range = noise_pct / 100.0
        
        # Generar ruido uniforme para cada vela (mismo ruido para toda la vela = desplazamiento)
        candle_noise = self.rng.uniform(-noise_range, noise_range, self.n_rows)
        
        # Ruido adicional pequeño para variación intra-vela
        open_noise = candle_noise + self.rng.uniform(-noise_range * 0.2, noise_range * 0.2, self.n_rows)
        high_noise = candle_noise + self.rng.uniform(0, noise_range * 0.3, self.n_rows)  # Solo positivo
        low_noise = candle_noise - self.rng.uniform(0, noise_range * 0.3, self.n_rows)   # Solo negativo
        close_noise = candle_noise + self.rng.uniform(-noise_range * 0.2, noise_range * 0.2, self.n_rows)
        
        # Aplicar ruido
        open_new = self.open_orig * (1 + open_noise)
        high_new = self.high_orig * (1 + high_noise)
        low_new = self.low_orig * (1 + low_noise)
        close_new = self.close_orig * (1 + close_noise)
        
        # Asegurar coherencia OHLC
        high_new = np.maximum(high_new, np.maximum(open_new, close_new))
        low_new = np.minimum(low_new, np.minimum(open_new, close_new))
        
        return self.df.with_columns([
            pl.Series("open", open_new),
            pl.Series("high", high_new),
            pl.Series("low", low_new),
            pl.Series("close", close_new),
        ])
    
    def generate_block_bootstrap(self, block_size: int) -> pl.DataFrame:
        """
        Block Bootstrap: Reordena bloques de velas preservando estructura de tendencias.
        
        1. Divide el histórico en bloques de N velas
        2. Mezcla los bloques aleatoriamente
        3. Ajusta los precios para que conecten suavemente entre bloques
        """
        n_blocks = self.n_rows // block_size
        if n_blocks < 3:
            # Si hay muy pocos bloques, usar noise en su lugar
            return self.generate_noise(0.1)
        
        # Crear índices de bloques y mezclar
        block_indices = list(range(n_blocks))
        self.rng.shuffle(block_indices)
        
        # Extraer bloques y reordenar
        open_new = np.empty(n_blocks * block_size)
        high_new = np.empty(n_blocks * block_size)
        low_new = np.empty(n_blocks * block_size)
        close_new = np.empty(n_blocks * block_size)
        
        for new_idx, orig_idx in enumerate(block_indices):
            start_orig = orig_idx * block_size
            end_orig = start_orig + block_size
            start_new = new_idx * block_size
            end_new = start_new + block_size
            
            open_new[start_new:end_new] = self.open_orig[start_orig:end_orig]
            high_new[start_new:end_new] = self.high_orig[start_orig:end_orig]
            low_new[start_new:end_new] = self.low_orig[start_orig:end_orig]
            close_new[start_new:end_new] = self.close_orig[start_orig:end_orig]
        
        # Ajustar precios para continuidad entre bloques
        for i in range(1, n_blocks):
            block_start = i * block_size
            prev_close = close_new[block_start - 1]
            curr_open = open_new[block_start]
            
            # Calcular ratio de ajuste
            if curr_open > 0:
                ratio = prev_close / curr_open
                
                # Aplicar ratio a todo el bloque
                open_new[block_start:block_start + block_size] *= ratio
                high_new[block_start:block_start + block_size] *= ratio
                low_new[block_start:block_start + block_size] *= ratio
                close_new[block_start:block_start + block_size] *= ratio
        
        # Crear nuevo DataFrame (truncado al tamaño de bloques completos)
        df_truncated = self.df.head(n_blocks * block_size)
        
        return df_truncated.with_columns([
            pl.Series("open", open_new),
            pl.Series("high", high_new),
            pl.Series("low", low_new),
            pl.Series("close", close_new),
        ])


# =============================================================================
# MONTE CARLO VALIDATOR
# =============================================================================
class MonteCarloValidator:
    """
    Monte Carlo Robustness Validator.
    
    Tests strategy on thousands of synthetic markets to determine
    if it's truly robust or just overfitted to historical data.
    """
    
    def __init__(self, config: MonteCarloConfig):
        self.config = config
        self.rng = np.random.default_rng(42)  # Seed for reproducibility
        
        # State
        self.strategy: Optional[Strategy] = None
        self.df: Optional[pl.DataFrame] = None
        self.generator: Optional[SyntheticMarketGenerator] = None
        self.simulation_results: List[Dict[str, Any]] = []
        self.analysis: Dict[str, Any] = {}
        
        # Configure exits
        sl, tp, trail_act, trail_dist = _normalize_exit_values(
            config.exit_type,
            config.exit_sl_pct,
            config.exit_tp_pct,
            config.exit_trail_act_pct,
            config.exit_trail_dist_pct,
        )
        self.exit_settings = ExitSettings(
            exit_type=config.exit_type,
            sl_pct=sl,
            tp_pct=tp if config.exit_type == "pnl_fixed" else 0.0,
            trail_act_pct=trail_act,
            trail_dist_pct=trail_dist,
        )
    
    # =========================================================================
    # DATA & STRATEGY LOADING
    # =========================================================================
    def load_strategy(self) -> None:
        """Load strategy by combinacion_id."""
        if RICH_AVAILABLE:
            console.print(f"\n[cyan]🔍 Buscando estrategia ID={self.config.combinacion_id}...[/cyan]")
        
        strategies = discover_strategies()
        
        # strategies is Dict[str, Type[Strategy]]
        for strat_name, strat_cls in strategies.items():
            try:
                instance = strat_cls()
                if instance.combinacion_id == self.config.combinacion_id:
                    self.strategy = instance
                    break
            except:
                continue
        
        if self.strategy is None:
            print(f"❌ No se encontró estrategia con ID {self.config.combinacion_id}")
            print(f"   IDs disponibles: {[strat_cls().combinacion_id for strat_cls in strategies.values()]}")
            sys.exit(1)
        
        if RICH_AVAILABLE:
            console.print(f"[green]✅ Estrategia: {self.strategy.name} (ID: {self.strategy.combinacion_id})[/green]")
    
    def load_data(self) -> None:
        """Load OHLC data."""
        if RICH_AVAILABLE:
            console.print(f"\n[cyan]📊 Cargando datos de {self.config.activo}...[/cyan]")
        
        suffix = normalize_timeframe_to_suffix(self.config.timeframe)
        csv_path = Path(self.config.data_dir) / f"{self.config.activo}_ohlcv_{suffix}.csv"
        feather_path = Path(self.config.data_dir) / f"{self.config.activo}_ohlcv_{suffix}.feather"
        
        if feather_path.exists():
            self.df = pl.read_ipc(feather_path)
        elif csv_path.exists():
            self.df = pl.read_csv(csv_path)
        else:
            print(f"❌ No se encontró archivo de datos")
            sys.exit(1)
        
        # Normalize datetime column name - keep both timestamp and datetime
        if "timestamp" in self.df.columns:
            if "datetime" not in self.df.columns:
                self.df = self.df.with_columns(
                    pl.col("timestamp").alias("datetime")
                )
        elif "datetime" in self.df.columns:
            self.df = self.df.with_columns(
                pl.col("datetime").alias("timestamp")
            )
        
        # Parse datetime if needed
        if self.df["datetime"].dtype == pl.Utf8:
            self.df = self.df.with_columns(
                pl.col("datetime").str.to_datetime().alias("datetime"),
                pl.col("timestamp").str.to_datetime().alias("timestamp"),
            )
        
        # Remove timezone if present
        dt_col = self.df["datetime"]
        if hasattr(dt_col.dtype, 'time_zone') and dt_col.dtype.time_zone is not None:
            self.df = self.df.with_columns(
                pl.col("datetime").dt.replace_time_zone(None).alias("datetime")
            )
        
        # Filter by date range
        fecha_inicio = datetime.fromisoformat(self.config.fecha_inicio)
        fecha_fin = datetime.fromisoformat(self.config.fecha_fin)
        
        self.df = self.df.filter(
            (pl.col("datetime") >= fecha_inicio) &
            (pl.col("datetime") <= fecha_fin)
        )
        
        # Create generator
        self.generator = SyntheticMarketGenerator(self.df, self.rng)
        
        if RICH_AVAILABLE:
            console.print(f"[green]✅ Datos cargados: {self.df.height:,} barras[/green]")
            console.print(f"   [dim]Rango: {self.df['datetime'].min()} → {self.df['datetime'].max()}[/dim]")
    
    # =========================================================================
    # BACKTEST EXECUTION
    # =========================================================================
    def _build_params(self) -> Dict[str, Any]:
        """Build strategy parameters."""
        # Use strategy's suggest_params with a mock trial that returns midpoints
        class MidpointTrial:
            def __init__(self, overrides):
                self.overrides = overrides
            def suggest_int(self, name, low, high, **kwargs):
                return self.overrides.get(name, (low + high) // 2)
            def suggest_float(self, name, low, high, **kwargs):
                return self.overrides.get(name, (low + high) / 2)
            def suggest_categorical(self, name, choices):
                return self.overrides.get(name, choices[0])
        
        mock_trial = MidpointTrial(self.config.strategy_params)
        params = self.strategy.suggest_params(mock_trial)
        
        # Add system params
        params["__exit_type"] = self.exit_settings.exit_type
        params["__exit_sl_pct"] = self.exit_settings.sl_pct
        params["__exit_tp_pct"] = self.exit_settings.tp_pct
        params["__exit_trail_act_pct"] = self.exit_settings.trail_act_pct
        params["__exit_trail_dist_pct"] = self.exit_settings.trail_dist_pct
        params["__saldo_usado"] = self.config.saldo_usado
        params["__apalancamiento_max"] = self.config.apalancamiento_max
        params["__timeframe_base"] = self.config.timeframe
        
        return params
    
    def _run_backtest(self, df: pl.DataFrame, params: Dict[str, Any], return_full: bool = False) -> Dict[str, Any]:
        """Run a single backtest and return metrics.
        
        Args:
            df: OHLC data
            params: Strategy parameters
            return_full: If True, also return trades_df, equity_curve, signals_df for charting
        """
        try:
            # Generate signals
            signals_df = self.strategy.generate_signals(df, params)
            
            # Create backtest config
            bt_config = BacktestConfig(
                saldo_inicial=self.config.saldo_inicial,
                saldo_operativo_max=self.config.saldo_operativo_max,
                saldo_minimo_operativo=self.config.saldo_minimo_operativo,
                comision_pct=self.config.comision_pct,
                comision_sides=self.config.comision_sides,
                saldo_usado=self.config.saldo_usado,
                apalancamiento_max=self.config.apalancamiento_max,
                qty_max_activo=self.config.qty_max_activo,
                exit_type=self.exit_settings.exit_type,
                exit_sl_pct=self.exit_settings.sl_pct,
                exit_tp_pct=self.exit_settings.tp_pct,
                exit_trail_act_pct=self.exit_settings.trail_act_pct,
                exit_trail_dist_pct=self.exit_settings.trail_dist_pct,
            )
            
            bt_params = BacktestParams.from_config_and_params(bt_config, params)
            
            # Execute backtest
            trades_df, equity_curve = calculate_performance_vectorized_numba(
                df=df,
                signals=signals_df,
                params=bt_params,
                strategy=self.strategy,
            )
            
            if trades_df.is_empty():
                result = self._empty_metrics()
                if return_full:
                    result["_trades_df"] = None
                    result["_equity_curve"] = None
                    result["_signals_df"] = None
                    result["_df"] = None
                return result
            
            # Calculate metrics
            metrics = resumen_metricas(
                trades_df,
                saldo_inicial=self.config.saldo_inicial,
                equity_curve=equity_curve,
            )
            
            result = {
                "roi": float(metrics.get("roi", 0)),
                "winrate": float(metrics.get("winrate", 0)),
                "drawdown": float(metrics.get("drawdown", 0)),
                "sharpe": float(metrics.get("sharpe", 0)),
                "sortino": float(metrics.get("sortino", 0)),
                "profit_factor": float(metrics.get("profit_factor", 0)),
                "n_trades": int(metrics.get("total_trades", 0)),
                "pnl_neto": float(metrics.get("pnl_neto", 0)),
                "profitable": float(metrics.get("roi", 0)) > 0,
            }
            
            if return_full:
                result["_trades_df"] = trades_df
                result["_equity_curve"] = equity_curve
                result["_signals_df"] = signals_df
                result["_df"] = df
            
            return result
            
        except Exception as e:
            result = self._empty_metrics()
            if return_full:
                result["_trades_df"] = None
                result["_equity_curve"] = None
                result["_signals_df"] = None
                result["_df"] = None
            return result
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics for failed backtests."""
        return {
            "roi": 0.0,
            "winrate": 0.0,
            "drawdown": 100.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "profit_factor": 0.0,
            "n_trades": 0,
            "pnl_neto": 0.0,
            "profitable": False,
        }
    
    # =========================================================================
    # MONTE CARLO SIMULATION
    # =========================================================================
    def run_monte_carlo(self) -> None:
        """Run Monte Carlo simulations with live reporting."""
        if RICH_AVAILABLE:
            console.print(f"\n[bold cyan]🎲 Monte Carlo Robustness Test[/bold cyan]")
            console.print(f"   [dim]Simulaciones: {self.config.n_simulations:,} | Métodos: {', '.join(self.config.methods)} | Ruido: ±{self.config.noise_pct}%[/dim]\n")
        
        params = self._build_params()
        self.simulation_results = []
        
        start_time = time.time()
        n_per_method = self.config.n_simulations // len(self.config.methods)
        
        # Running statistics
        running_stats = {
            "profitable": 0, "total": 0, "roi_sum": 0.0, "roi_sq_sum": 0.0,
            "roi_max": float("-inf"), "roi_min": float("inf"), "roi_values": [],
            "winrate_sum": 0.0, "drawdown_sum": 0.0, "sharpe_sum": 0.0, "trades_sum": 0,
            "best_sim": None, "worst_sim": None,
            "by_method": {m: {"count": 0, "profitable": 0, "roi_sum": 0.0} for m in self.config.methods},
            "streak_profit": 0, "streak_loss": 0, "max_streak_profit": 0, "max_streak_loss": 0,
        }
        
        # Store best simulation data for charting
        self.best_sim_data = None
        self.best_sim_method = None
        self.best_sim_df = None
        
        if RICH_AVAILABLE:
            from rich.live import Live
            
            METHOD_NAMES = {"noise": "Ruido Gaussiano", "block_bootstrap": "Block Bootstrap"}
            
            def create_display(sim_idx: int, method: str, last: Dict[str, Any], s: Dict) -> Table:
                """Create compact live display."""
                pct = (sim_idx + 1) / self.config.n_simulations * 100
                elapsed = time.time() - start_time
                eta = elapsed / (sim_idx + 1) * (self.config.n_simulations - sim_idx - 1) if sim_idx > 0 else 0
                speed = (sim_idx + 1) / elapsed if elapsed > 0 else 0
                
                # Single compact table
                t = Table(box=box.ROUNDED, border_style="cyan", padding=(0, 1), expand=False, 
                         title=f"[bold cyan]🎲 Monte Carlo[/] [dim]| {METHOD_NAMES.get(method, method)}[/]")
                t.add_column("", style="white", width=18)
                t.add_column("", justify="right", width=12)
                t.add_column("", style="white", width=18)
                t.add_column("", justify="right", width=12)
                
                # Progress bar
                filled = min(20, int(pct / 5))
                bar = "█" * filled + "░" * (20 - filled)
                t.add_row(
                    f"[cyan]{bar}[/]", f"[bold]{pct:.1f}%[/]",
                    "Simulación", f"[bold]{sim_idx+1}[/]/{self.config.n_simulations}"
                )
                t.add_row(
                    "[dim]Tiempo[/]", f"{elapsed:.0f}s",
                    "[dim]Restante[/]", f"{eta:.0f}s"
                )
                t.add_row("─" * 18, "─" * 12, "─" * 18, "─" * 12)
                
                # Last simulation
                if last:
                    roi = last.get("roi", 0)
                    rc = "green" if roi > 0 else "red"
                    t.add_row(
                        "[bold]Último ROI[/]", f"[{rc} bold]{roi:+.2f}%[/]",
                        "Operaciones", f"{last.get('n_trades', 0)}"
                    )
                    t.add_row(
                        "Tasa Acierto", f"{last.get('winrate', 0):.1f}%",
                        "Drawdown Máx", f"{last.get('drawdown', 0):.1f}%"
                    )
                
                t.add_row("─" * 18, "─" * 12, "─" * 18, "─" * 12)
                
                # Global stats
                if s["total"] > 0:
                    rob = s["profitable"] / s["total"] * 100
                    rc = "green" if rob >= 70 else ("yellow" if rob >= 50 else "red")
                    avg_roi = s["roi_sum"] / s["total"]
                    ac = "green" if avg_roi > 0 else "red"
                    
                    t.add_row(
                        "[bold]ROBUSTEZ[/]", f"[{rc} bold]{rob:.1f}%[/]",
                        "Rentables", f"[{rc}]{s['profitable']}/{s['total']}[/]"
                    )
                    t.add_row(
                        "ROI Promedio", f"[{ac}]{avg_roi:+.2f}%[/]",
                        "Rango ROI", f"{s['roi_min']:.0f}% → {s['roi_max']:.0f}%"
                    )
                    t.add_row(
                        "Winrate Prom.", f"{s['winrate_sum']/s['total']:.1f}%",
                        "Drawdown Prom.", f"{s['drawdown_sum']/s['total']:.1f}%"
                    )
                    
                    # Best/Worst
                    if s['best_sim']:
                        t.add_row("─" * 18, "─" * 12, "─" * 18, "─" * 12)
                        t.add_row(
                            "[green]🏆 Mejor[/]", f"#{s['best_sim']['idx']+1} ({s['best_sim']['roi']:+.1f}%)",
                            "[red]💀 Peor[/]", f"#{s['worst_sim']['idx']+1} ({s['worst_sim']['roi']:+.1f}%)" if s['worst_sim'] else "-"
                        )
                
                return t
            
            with Live(console=console, refresh_per_second=4, transient=True) as live:
                sim_idx = 0
                for method in self.config.methods:
                    for i in range(n_per_method):
                        if method == "noise":
                            df_synthetic = self.generator.generate_noise(self.config.noise_pct)
                        else:
                            df_synthetic = self.generator.generate_block_bootstrap(self.config.block_size)
                        
                        metrics = self._run_backtest(df_synthetic, params)
                        metrics["method"] = method
                        metrics["iteration"] = sim_idx
                        self.simulation_results.append(metrics)
                        
                        # Update stats
                        s = running_stats
                        s["total"] += 1
                        roi = metrics["roi"]
                        s["roi_sum"] += roi
                        s["roi_values"].append(roi)
                        s["winrate_sum"] += metrics["winrate"]
                        s["drawdown_sum"] += metrics["drawdown"]
                        s["sharpe_sum"] += metrics["sharpe"]
                        s["trades_sum"] += metrics["n_trades"]
                        s["by_method"][method]["count"] += 1
                        s["by_method"][method]["roi_sum"] += roi
                        
                        if roi > 0:
                            s["profitable"] += 1
                            s["by_method"][method]["profitable"] += 1
                            s["streak_profit"] += 1
                            s["streak_loss"] = 0
                            s["max_streak_profit"] = max(s["max_streak_profit"], s["streak_profit"])
                        else:
                            s["streak_loss"] += 1
                            s["streak_profit"] = 0
                            s["max_streak_loss"] = max(s["max_streak_loss"], s["streak_loss"])
                        
                        if roi > s["roi_max"]:
                            s["roi_max"] = roi
                            s["best_sim"] = {"idx": sim_idx, "roi": roi}
                            # Re-run with full data to save for charting
                            self.best_sim_data = self._run_backtest(df_synthetic, params, return_full=True)
                            self.best_sim_method = method
                            self.best_sim_df = df_synthetic.clone()
                        if roi < s["roi_min"]:
                            s["roi_min"] = roi
                            s["worst_sim"] = {"idx": sim_idx, "roi": roi}
                        
                        live.update(create_display(sim_idx, method, metrics, s))
                        sim_idx += 1
        else:
            sim_idx = 0
            for method in self.config.methods:
                for i in range(n_per_method):
                    if method == "noise":
                        df_synthetic = self.generator.generate_noise(self.config.noise_pct)
                    else:
                        df_synthetic = self.generator.generate_block_bootstrap(self.config.block_size)
                    
                    metrics = self._run_backtest(df_synthetic, params)
                    metrics["method"] = method
                    metrics["iteration"] = sim_idx
                    self.simulation_results.append(metrics)
                    
                    sim_idx += 1
                    if (sim_idx) % 100 == 0:
                        print(f"   Progreso: {sim_idx}/{self.config.n_simulations}")
        
        elapsed = time.time() - start_time
        if RICH_AVAILABLE:
            console.print(f"\n[green]✅ Simulación completada en {elapsed:.1f}s[/green]")
    
    # =========================================================================
    # ANALYSIS
    # =========================================================================
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze Monte Carlo results."""
        if RICH_AVAILABLE:
            console.print(f"\n[cyan]📈 Analizando resultados...[/cyan]")
        
        results_df = pl.DataFrame(self.simulation_results)
        
        # Calculate key statistics
        roi_values = results_df["roi"].to_numpy()
        profitable_count = np.sum(roi_values > 0)
        total_count = len(roi_values)
        
        self.analysis = {
            "total_simulations": total_count,
            "profitable_simulations": int(profitable_count),
            "robustness_score": float(profitable_count / total_count * 100),
            
            # ROI Statistics
            "roi_mean": float(np.mean(roi_values)),
            "roi_std": float(np.std(roi_values)),
            "roi_median": float(np.median(roi_values)),
            "roi_min": float(np.min(roi_values)),
            "roi_max": float(np.max(roi_values)),
            "roi_p5": float(np.percentile(roi_values, 5)),
            "roi_p25": float(np.percentile(roi_values, 25)),
            "roi_p75": float(np.percentile(roi_values, 75)),
            "roi_p95": float(np.percentile(roi_values, 95)),
            
            # Other metrics
            "winrate_mean": float(results_df["winrate"].mean()),
            "drawdown_mean": float(results_df["drawdown"].mean()),
            "sharpe_mean": float(results_df["sharpe"].mean()),
            "trades_mean": float(results_df["n_trades"].mean()),
            
            # By method
            "by_method": {},
        }
        
        for method in self.config.methods:
            method_df = results_df.filter(pl.col("method") == method)
            if method_df.height > 0:
                method_roi = method_df["roi"].to_numpy()
                self.analysis["by_method"][method] = {
                    "count": method_df.height,
                    "profitable": int(np.sum(method_roi > 0)),
                    "robustness": float(np.sum(method_roi > 0) / method_df.height * 100),
                    "roi_mean": float(np.mean(method_roi)),
                    "roi_std": float(np.std(method_roi)),
                }
        
        # Determine verdict
        score = self.analysis["robustness_score"]
        if score >= 70:
            self.analysis["verdict"] = "ROBUST"
            self.analysis["verdict_detail"] = "Strategy survives in >70% of synthetic markets"
        elif score >= 50:
            self.analysis["verdict"] = "MODERATE"
            self.analysis["verdict_detail"] = "Strategy has moderate robustness (50-70%)"
        else:
            self.analysis["verdict"] = "OVERFITTED"
            self.analysis["verdict_detail"] = "Strategy likely overfitted to historical data"
        
        return self.analysis
    
    # =========================================================================
    # REPORTING
    # =========================================================================
    def print_report(self) -> None:
        """Print Monte Carlo report."""
        if not RICH_AVAILABLE:
            self._print_simple_report()
            return
        
        console.print()
        
        # Header
        header = Panel(
            f"[bold]MONTE CARLO ROBUSTNESS REPORT[/bold]\n"
            f"[dim]{self.strategy.name} (ID: {self.strategy.combinacion_id})[/dim]",
            border_style="cyan",
            padding=(0, 2)
        )
        console.print(Align.center(header))
        
        # Main Statistics Table
        stats_table = Table(
            title="[bold]ROBUSTNESS ANALYSIS[/bold]",
            box=box.ROUNDED,
            border_style="dim",
            show_header=True,
            header_style="bold cyan"
        )
        
        stats_table.add_column("Metric", style="white")
        stats_table.add_column("Value", justify="right")
        stats_table.add_column("Interpretation", style="dim")
        
        score = self.analysis["robustness_score"]
        score_color = "green" if score >= 70 else ("yellow" if score >= 50 else "red")
        
        stats_table.add_row(
            "Total Simulations",
            f"{self.analysis['total_simulations']:,}",
            ""
        )
        stats_table.add_row(
            "Profitable Simulations",
            f"[{score_color}]{self.analysis['profitable_simulations']:,}[/]",
            f"of {self.analysis['total_simulations']:,}"
        )
        stats_table.add_row(
            "ROBUSTNESS SCORE",
            f"[bold {score_color}]{score:.1f}%[/]",
            "≥70% = Robust"
        )
        stats_table.add_row("", "", "")
        stats_table.add_row(
            "ROI Mean",
            f"{self.analysis['roi_mean']:+.2f}%",
            ""
        )
        stats_table.add_row(
            "ROI Std Dev",
            f"{self.analysis['roi_std']:.2f}%",
            "Lower = More stable"
        )
        stats_table.add_row(
            "ROI Range",
            f"[red]{self.analysis['roi_min']:.1f}%[/] → [green]{self.analysis['roi_max']:.1f}%[/]",
            ""
        )
        stats_table.add_row(
            "ROI 5th Percentile",
            f"{self.analysis['roi_p5']:.2f}%",
            "Worst case (95% confidence)"
        )
        stats_table.add_row("", "", "")
        stats_table.add_row(
            "Avg Trades/Simulation",
            f"{self.analysis['trades_mean']:.0f}",
            ""
        )
        stats_table.add_row(
            "Avg Winrate",
            f"{self.analysis['winrate_mean']:.1f}%",
            ""
        )
        stats_table.add_row(
            "Avg Max Drawdown",
            f"{self.analysis['drawdown_mean']:.1f}%",
            ""
        )
        
        console.print(Align.center(stats_table))
        
        # Method Breakdown
        if self.analysis["by_method"]:
            method_table = Table(
                title="[bold]RESULTS BY METHOD[/bold]",
                box=box.ROUNDED,
                border_style="dim"
            )
            
            method_table.add_column("Method", style="white")
            method_table.add_column("Simulations", justify="right")
            method_table.add_column("Profitable", justify="right")
            method_table.add_column("Robustness", justify="right")
            method_table.add_column("ROI Mean", justify="right")
            
            for method, data in self.analysis["by_method"].items():
                rob = data["robustness"]
                rob_color = "green" if rob >= 70 else ("yellow" if rob >= 50 else "red")
                roi_color = "green" if data["roi_mean"] > 0 else "red"
                
                method_table.add_row(
                    method.upper(),
                    str(data["count"]),
                    str(data["profitable"]),
                    f"[{rob_color}]{rob:.1f}%[/]",
                    f"[{roi_color}]{data['roi_mean']:+.2f}%[/]"
                )
            
            console.print()
            console.print(Align.center(method_table))
        
        # Verdict Panel
        verdict = self.analysis["verdict"]
        verdict_color = {"ROBUST": "green", "MODERATE": "yellow", "OVERFITTED": "red"}[verdict]
        verdict_emoji = {"ROBUST": "✓", "MODERATE": "⚠", "OVERFITTED": "✗"}[verdict]
        
        verdict_panel = Panel(
            f"[bold {verdict_color}]{verdict_emoji} {verdict}[/]\n"
            f"[dim]{self.analysis['verdict_detail']}[/]",
            title="[bold]VERDICT[/bold]",
            border_style=verdict_color,
            padding=(1, 4)
        )
        console.print()
        console.print(Align.center(verdict_panel))
    
    def _print_simple_report(self) -> None:
        """Print simple text report."""
        print("\n" + "=" * 60)
        print("MONTE CARLO ROBUSTNESS REPORT")
        print("=" * 60)
        print(f"Strategy: {self.strategy.name}")
        print(f"Simulations: {self.analysis['total_simulations']:,}")
        print(f"Profitable: {self.analysis['profitable_simulations']:,}")
        print(f"ROBUSTNESS SCORE: {self.analysis['robustness_score']:.1f}%")
        print(f"Verdict: {self.analysis['verdict']}")
        print("=" * 60)
    
    # =========================================================================
    # EXPORTS
    # =========================================================================
    def export_results(self) -> None:
        """Export results to files."""
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.config.output_dir) / f"ID{self.config.combinacion_id}_{self.strategy.name}" / "montecarlo" / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = f"MC_{self.config.combinacion_id}_{self.config.activo}"
        
        if RICH_AVAILABLE:
            console.print(f"\n[cyan]💾 Exportando resultados...[/cyan]")
        
        # CSV
        if self.config.export_csv:
            csv_path = output_dir / f"{base_name}_simulations.csv"
            pl.DataFrame(self.simulation_results).write_csv(str(csv_path))
            if RICH_AVAILABLE:
                console.print(f"   [dim]CSV: {csv_path.name}[/dim]")
        
        # Excel
        if self.config.export_excel:
            self._export_excel(output_dir, base_name)
        
        # PDF
        if self.config.export_pdf and MATPLOTLIB_AVAILABLE:
            self._export_pdf(output_dir, base_name)
        
        # HTML Chart for best trial
        self._export_best_trial_chart(output_dir, base_name)
        
        if RICH_AVAILABLE:
            console.print(f"[green]✅ Resultados guardados en: {output_dir}[/green]")
    
    def _export_excel(self, output_dir: Path, base_name: str):
        """Export to professional Excel workbook with multiple sheets."""
        try:
            import xlsxwriter
        except ImportError:
            return
        
        excel_path = output_dir / f"{base_name}.xlsx"
        results_df = pl.DataFrame(self.simulation_results)
        
        with xlsxwriter.Workbook(str(excel_path)) as wb:
            # =================================================================
            # FORMATS
            # =================================================================
            # Headers
            fmt_title = wb.add_format({
                'bold': True, 'font_size': 18, 'font_color': '#58a6ff',
                'bg_color': '#0d1117', 'align': 'center', 'valign': 'vcenter'
            })
            fmt_header = wb.add_format({
                'bold': True, 'font_size': 11, 'font_color': '#e6edf3',
                'bg_color': '#1c2128', 'border': 1, 'border_color': '#30363d',
                'align': 'center', 'valign': 'vcenter'
            })
            fmt_section = wb.add_format({
                'bold': True, 'font_size': 12, 'font_color': '#58a6ff',
                'bg_color': '#161b22', 'border': 1
            })
            
            # Numbers
            fmt_number = wb.add_format({'num_format': '#,##0.00', 'align': 'right'})
            fmt_pct = wb.add_format({'num_format': '0.00%', 'align': 'right'})
            fmt_pct_color = wb.add_format({'num_format': '+0.00%;-0.00%', 'align': 'right'})
            fmt_int = wb.add_format({'num_format': '#,##0', 'align': 'right'})
            
            # Conditional
            fmt_good = wb.add_format({
                'font_color': '#3fb950', 'bold': True, 'num_format': '+0.00%'
            })
            fmt_bad = wb.add_format({
                'font_color': '#f85149', 'bold': True, 'num_format': '+0.00%'
            })
            fmt_neutral = wb.add_format({
                'font_color': '#d29922', 'bold': True, 'num_format': '0.00%'
            })
            
            # Verdict
            fmt_robust = wb.add_format({
                'bold': True, 'font_size': 14, 'font_color': '#ffffff',
                'bg_color': '#238636', 'align': 'center', 'border': 2
            })
            fmt_moderate = wb.add_format({
                'bold': True, 'font_size': 14, 'font_color': '#000000',
                'bg_color': '#d29922', 'align': 'center', 'border': 2
            })
            fmt_overfitted = wb.add_format({
                'bold': True, 'font_size': 14, 'font_color': '#ffffff',
                'bg_color': '#da3633', 'align': 'center', 'border': 2
            })
            
            # =================================================================
            # SHEET 1: EXECUTIVE SUMMARY
            # =================================================================
            ws = wb.add_worksheet("📊 Executive Summary")
            ws.set_column('A:A', 35)
            ws.set_column('B:B', 20)
            ws.set_column('C:C', 30)
            ws.set_column('D:D', 15)
            ws.set_column('E:E', 15)
            
            # Title
            ws.merge_range('A1:E1', '🎲 MONTE CARLO ROBUSTNESS REPORT', fmt_title)
            ws.set_row(0, 40)
            
            # Strategy Info
            ws.write(2, 0, 'Strategy', fmt_section)
            ws.merge_range('B3:C3', self.strategy.name, fmt_header)
            ws.write(3, 0, 'Combinacion ID')
            ws.write(3, 1, self.config.combinacion_id, fmt_int)
            ws.write(4, 0, 'Asset')
            ws.write(4, 1, self.config.activo)
            ws.write(5, 0, 'Timeframe')
            ws.write(5, 1, self.config.timeframe)
            ws.write(6, 0, 'Date Range')
            ws.write(6, 1, f"{self.config.fecha_inicio} → {self.config.fecha_fin}")
            
            # Verdict
            ws.write(8, 0, 'VERDICT', fmt_section)
            verdict = self.analysis["verdict"]
            verdict_fmt = fmt_robust if verdict == "ROBUST" else (fmt_moderate if verdict == "MODERATE" else fmt_overfitted)
            ws.merge_range('B9:C9', f"  {verdict}  ", verdict_fmt)
            ws.write(9, 3, self.analysis["verdict_detail"])
            
            # Key Metrics
            ws.write(11, 0, 'KEY METRICS', fmt_section)
            ws.write(12, 0, 'Total Simulations')
            ws.write(12, 1, self.analysis["total_simulations"], fmt_int)
            ws.write(13, 0, 'Profitable Simulations')
            ws.write(13, 1, self.analysis["profitable_simulations"], fmt_int)
            ws.write(13, 2, f"({self.analysis['profitable_simulations']/self.analysis['total_simulations']*100:.1f}%)")
            
            ws.write(14, 0, 'ROBUSTNESS SCORE')
            score = self.analysis["robustness_score"]
            score_fmt = fmt_good if score >= 70 else (fmt_neutral if score >= 50 else fmt_bad)
            ws.write(14, 1, score / 100, score_fmt)
            
            # ROI Statistics
            ws.write(16, 0, 'ROI STATISTICS', fmt_section)
            ws.write(17, 0, 'ROI Mean')
            ws.write(17, 1, self.analysis["roi_mean"] / 100, fmt_pct_color)
            ws.write(18, 0, 'ROI Std Dev')
            ws.write(18, 1, self.analysis["roi_std"] / 100, fmt_pct)
            ws.write(19, 0, 'ROI Median')
            ws.write(19, 1, self.analysis["roi_median"] / 100, fmt_pct_color)
            ws.write(20, 0, 'ROI Min (Worst)')
            ws.write(20, 1, self.analysis["roi_min"] / 100, fmt_bad)
            ws.write(21, 0, 'ROI Max (Best)')
            ws.write(21, 1, self.analysis["roi_max"] / 100, fmt_good)
            
            # Percentiles
            ws.write(23, 0, 'PERCENTILES', fmt_section)
            ws.write(24, 0, '5th Percentile (95% VaR)')
            ws.write(24, 1, self.analysis["roi_p5"] / 100, fmt_pct_color)
            ws.write(24, 2, "← Worst case with 95% confidence")
            ws.write(25, 0, '25th Percentile')
            ws.write(25, 1, self.analysis["roi_p25"] / 100, fmt_pct_color)
            ws.write(26, 0, '75th Percentile')
            ws.write(26, 1, self.analysis["roi_p75"] / 100, fmt_pct_color)
            ws.write(27, 0, '95th Percentile')
            ws.write(27, 1, self.analysis["roi_p95"] / 100, fmt_pct_color)
            ws.write(27, 2, "← Best case with 95% confidence")
            
            # Other Metrics
            ws.write(29, 0, 'PERFORMANCE AVERAGES', fmt_section)
            ws.write(30, 0, 'Avg Trades per Simulation')
            ws.write(30, 1, self.analysis["trades_mean"], fmt_number)
            ws.write(31, 0, 'Avg Winrate')
            ws.write(31, 1, self.analysis["winrate_mean"] / 100, fmt_pct)
            ws.write(32, 0, 'Avg Max Drawdown')
            ws.write(32, 1, self.analysis["drawdown_mean"] / 100, fmt_pct)
            ws.write(33, 0, 'Avg Sharpe Ratio')
            ws.write(33, 1, self.analysis["sharpe_mean"], fmt_number)
            
            # =================================================================
            # SHEET 2: METHOD COMPARISON
            # =================================================================
            ws2 = wb.add_worksheet("📈 Method Analysis")
            ws2.set_column('A:A', 20)
            ws2.set_column('B:G', 15)
            
            ws2.merge_range('A1:G1', 'ANALYSIS BY PERTURBATION METHOD', fmt_title)
            ws2.set_row(0, 35)
            
            headers = ['Method', 'Simulations', 'Profitable', 'Robustness', 'ROI Mean', 'ROI Std', 'Status']
            for col, h in enumerate(headers):
                ws2.write(2, col, h, fmt_header)
            
            row = 3
            for method, data in self.analysis["by_method"].items():
                rob = data["robustness"]
                ws2.write(row, 0, method.upper())
                ws2.write(row, 1, data["count"], fmt_int)
                ws2.write(row, 2, data["profitable"], fmt_int)
                ws2.write(row, 3, rob / 100, fmt_pct)
                ws2.write(row, 4, data["roi_mean"] / 100, fmt_pct_color)
                ws2.write(row, 5, data["roi_std"] / 100, fmt_pct)
                
                status = "✓ PASS" if rob >= 70 else ("⚠ WARN" if rob >= 50 else "✗ FAIL")
                status_fmt = fmt_good if rob >= 70 else (fmt_neutral if rob >= 50 else fmt_bad)
                ws2.write(row, 6, status, status_fmt)
                row += 1
            
            # =================================================================
            # SHEET 3: ALL SIMULATIONS
            # =================================================================
            ws3 = wb.add_worksheet("📋 All Simulations")
            
            # Headers
            columns = results_df.columns
            for col, name in enumerate(columns):
                ws3.write(0, col, name.upper(), fmt_header)
                ws3.set_column(col, col, 12)
            
            # Data
            for row_idx, row_data in enumerate(results_df.iter_rows()):
                for col_idx, val in enumerate(row_data):
                    if isinstance(val, bool):
                        ws3.write(row_idx + 1, col_idx, "✓" if val else "✗")
                    elif isinstance(val, float):
                        if columns[col_idx] in ['roi', 'winrate', 'drawdown']:
                            ws3.write(row_idx + 1, col_idx, val / 100, fmt_pct_color)
                        else:
                            ws3.write(row_idx + 1, col_idx, val, fmt_number)
                    else:
                        ws3.write(row_idx + 1, col_idx, val)
            
            # Conditional formatting for ROI column
            roi_col = columns.index('roi') if 'roi' in columns else -1
            if roi_col >= 0:
                ws3.conditional_format(1, roi_col, len(results_df), roi_col, {
                    'type': '3_color_scale',
                    'min_color': '#f85149',
                    'mid_color': '#d29922',
                    'max_color': '#3fb950'
                })
            
            # =================================================================
            # SHEET 4: DISTRIBUTION STATS
            # =================================================================
            ws4 = wb.add_worksheet("📊 Distribution Stats")
            ws4.set_column('A:A', 25)
            ws4.set_column('B:F', 15)
            
            ws4.merge_range('A1:F1', 'STATISTICAL DISTRIBUTION ANALYSIS', fmt_title)
            
            # ROI Distribution Bins
            roi_values = results_df["roi"].to_numpy()
            bins = np.linspace(roi_values.min(), roi_values.max(), 21)
            hist, bin_edges = np.histogram(roi_values, bins=bins)
            
            ws4.write(3, 0, 'ROI HISTOGRAM', fmt_section)
            ws4.write(4, 0, 'Bin Range', fmt_header)
            ws4.write(4, 1, 'Count', fmt_header)
            ws4.write(4, 2, 'Frequency %', fmt_header)
            ws4.write(4, 3, 'Cumulative %', fmt_header)
            
            cumsum = 0
            for i, (count, edge) in enumerate(zip(hist, bin_edges[:-1])):
                cumsum += count
                ws4.write(5 + i, 0, f"{edge:.1f}% to {bin_edges[i+1]:.1f}%")
                ws4.write(5 + i, 1, count, fmt_int)
                ws4.write(5 + i, 2, count / len(roi_values), fmt_pct)
                ws4.write(5 + i, 3, cumsum / len(roi_values), fmt_pct)
            
            # Add sparkline-style bar chart
            ws4.add_table(4, 0, 4 + len(hist), 3, {
                'style': 'Table Style Medium 2',
                'columns': [
                    {'header': 'Bin Range'},
                    {'header': 'Count'},
                    {'header': 'Frequency %'},
                    {'header': 'Cumulative %'}
                ]
            })
            
            # =================================================================
            # SHEET 5: CONFIGURATION
            # =================================================================
            ws5 = wb.add_worksheet("⚙️ Configuration")
            ws5.set_column('A:A', 25)
            ws5.set_column('B:B', 30)
            
            ws5.merge_range('A1:B1', 'MONTE CARLO CONFIGURATION', fmt_title)
            
            config_items = [
                ('Strategy ID', self.config.combinacion_id),
                ('Asset', self.config.activo),
                ('Timeframe', self.config.timeframe),
                ('Date Start', self.config.fecha_inicio),
                ('Date End', self.config.fecha_fin),
                ('', ''),
                ('MONTE CARLO SETTINGS', ''),
                ('Total Simulations', self.config.n_simulations),
                ('Methods', ', '.join(self.config.methods)),
                ('Noise Percentage', f"±{self.config.noise_pct}%"),
                ('Block Size', f"{self.config.block_size} bars"),
                ('', ''),
                ('EXIT SETTINGS', ''),
                ('Exit Type', self.config.exit_type),
                ('Stop Loss', f"{self.config.exit_sl_pct}%"),
                ('Take Profit', f"{self.config.exit_tp_pct}%"),
                ('', ''),
                ('BACKTEST SETTINGS', ''),
                ('Initial Balance', f"${self.config.saldo_inicial:,.2f}"),
                ('Commission', f"{self.config.comision_pct * 100:.4f}%"),
                ('Position Size', f"{self.config.saldo_usado}%"),
                ('Max Leverage', f"{self.config.apalancamiento_max}x"),
            ]
            
            row = 3
            for key, val in config_items:
                if key == '':
                    row += 1
                elif val == '':
                    ws5.write(row, 0, key, fmt_section)
                    row += 1
                else:
                    ws5.write(row, 0, key)
                    ws5.write(row, 1, val)
                    row += 1
        
        if RICH_AVAILABLE:
            console.print(f"   [dim]Excel: {excel_path.name}[/dim]")
    
    def _export_pdf(self, output_dir: Path, base_name: str):
        """Export to professional PDF with multiple charts including 3D analysis."""
        pdf_path = output_dir / f"{base_name}.pdf"
        
        results_df = pl.DataFrame(self.simulation_results)
        roi_values = results_df["roi"].to_numpy()
        winrate_values = results_df["winrate"].to_numpy()
        dd_values = results_df["drawdown"].to_numpy()
        sharpe_values = results_df["sharpe"].to_numpy()
        trades_values = results_df["n_trades"].to_numpy()
        
        with PdfPages(str(pdf_path)) as pdf:
            # =================================================================
            # PAGE 1: EXECUTIVE SUMMARY
            # =================================================================
            fig = plt.figure(figsize=(11, 8.5))
            fig.patch.set_facecolor(Theme.BG_PRIMARY)
            
            # Title
            fig.text(0.5, 0.92, '🎲 MONTE CARLO ROBUSTNESS REPORT', fontsize=24, ha='center',
                    fontweight='bold', color=Theme.ACCENT)
            fig.text(0.5, 0.86, f'{self.strategy.name} (ID: {self.config.combinacion_id})', 
                    fontsize=12, ha='center', color=Theme.TEXT_SECONDARY)
            
            # Big Robustness Score
            score = self.analysis["robustness_score"]
            score_color = Theme.GREEN if score >= 70 else (Theme.ORANGE if score >= 50 else Theme.RED)
            
            fig.text(0.5, 0.62, f'{score:.1f}%', fontsize=96, ha='center',
                    fontweight='bold', color=score_color)
            fig.text(0.5, 0.48, 'ROBUSTNESS SCORE', fontsize=16, ha='center',
                    color=Theme.TEXT_SECONDARY, fontweight='bold')
            
            # Verdict
            verdict = self.analysis["verdict"]
            verdict_emoji = {"ROBUST": "✓", "MODERATE": "⚠", "OVERFITTED": "✗"}[verdict]
            fig.text(0.5, 0.36, f'{verdict_emoji} {verdict}', fontsize=28, ha='center',
                    fontweight='bold', color=score_color)
            fig.text(0.5, 0.29, self.analysis["verdict_detail"], fontsize=11, ha='center',
                    color=Theme.TEXT_MUTED)
            
            # Key Stats Row
            stats_y = 0.16
            stats = [
                (f"{self.analysis['total_simulations']:,}", "Simulaciones"),
                (f"{self.analysis['profitable_simulations']:,}", "Rentables"),
                (f"{self.analysis['roi_mean']:+.2f}%", "ROI Promedio"),
                (f"{self.analysis['roi_std']:.2f}%", "Desv. Estándar"),
                (f"{self.analysis['roi_p5']:.1f}%", "Percentil 5"),
            ]
            for i, (val, label) in enumerate(stats):
                x = 0.1 + i * 0.2
                fig.text(x, stats_y, val, fontsize=14, ha='center', fontweight='bold', color=Theme.TEXT_PRIMARY)
                fig.text(x, stats_y - 0.04, label, fontsize=9, ha='center', color=Theme.TEXT_MUTED)
            
            # Footer
            fig.text(0.5, 0.03, f'{self.config.activo} | {self.config.timeframe} | {self.config.fecha_inicio} → {self.config.fecha_fin}',
                    fontsize=8, ha='center', color=Theme.TEXT_MUTED)
            
            pdf.savefig(fig, facecolor=Theme.BG_PRIMARY)
            plt.close(fig)
            
            # =================================================================
            # PAGE 2: DISTRIBUTION ANALYSIS (4 charts)
            # =================================================================
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            fig.patch.set_facecolor(Theme.BG_PRIMARY)
            fig.suptitle('📊 DISTRIBUTION ANALYSIS', fontsize=16, fontweight='bold',
                        color=Theme.TEXT_PRIMARY, y=0.98)
            
            for ax in axes.flat:
                ax.set_facecolor(Theme.BG_SECONDARY)
                for spine in ax.spines.values():
                    spine.set_color(Theme.BORDER)
            
            # ROI Histogram with KDE
            ax1 = axes[0, 0]
            n, bins, patches = ax1.hist(roi_values, bins=50, color=Theme.ACCENT, alpha=0.7, edgecolor='none', density=True)
            # Color bars based on value
            for i, patch in enumerate(patches):
                if bins[i] < 0:
                    patch.set_facecolor(Theme.RED)
                    patch.set_alpha(0.6)
            # Add KDE
            if len(roi_values) > 10:
                kde = scipy_stats.gaussian_kde(roi_values)
                x_kde = np.linspace(roi_values.min(), roi_values.max(), 100)
                ax1.plot(x_kde, kde(x_kde), color=Theme.CYAN, linewidth=2, label='KDE')
            ax1.axvline(0, color='white', linestyle='--', linewidth=2, alpha=0.8, label='Break-even')
            ax1.axvline(np.mean(roi_values), color=Theme.GREEN, linestyle='-', linewidth=2, label=f'Media: {np.mean(roi_values):.1f}%')
            ax1.axvline(np.median(roi_values), color=Theme.ORANGE, linestyle=':', linewidth=2, label=f'Mediana: {np.median(roi_values):.1f}%')
            ax1.set_title('Distribución ROI', color=Theme.TEXT_PRIMARY, fontweight='bold')
            ax1.set_xlabel('ROI (%)', color=Theme.TEXT_MUTED)
            ax1.set_ylabel('Densidad', color=Theme.TEXT_MUTED)
            ax1.legend(fontsize=7, loc='upper right')
            ax1.tick_params(colors=Theme.TEXT_MUTED)
            
            # Winrate vs Drawdown Scatter
            ax2 = axes[0, 1]
            colors = [Theme.GREEN if r > 0 else Theme.RED for r in roi_values]
            scatter = ax2.scatter(winrate_values, dd_values, c=roi_values, cmap='RdYlGn', 
                                 alpha=0.6, s=20, edgecolors='none')
            ax2.axhline(25, color=Theme.ORANGE, linestyle='--', alpha=0.5, label='DD 25%')
            ax2.axvline(50, color=Theme.CYAN, linestyle='--', alpha=0.5, label='WR 50%')
            ax2.set_title('Winrate vs Drawdown', color=Theme.TEXT_PRIMARY, fontweight='bold')
            ax2.set_xlabel('Winrate (%)', color=Theme.TEXT_MUTED)
            ax2.set_ylabel('Max Drawdown (%)', color=Theme.TEXT_MUTED)
            ax2.legend(fontsize=7)
            ax2.tick_params(colors=Theme.TEXT_MUTED)
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.ax.tick_params(colors=Theme.TEXT_MUTED)
            cbar.set_label('ROI %', color=Theme.TEXT_MUTED)
            
            # Cumulative Distribution (CDF)
            ax3 = axes[1, 0]
            sorted_roi = np.sort(roi_values)
            cdf = np.arange(1, len(sorted_roi) + 1) / len(sorted_roi) * 100
            ax3.fill_between(sorted_roi, cdf, alpha=0.3, color=Theme.ACCENT)
            ax3.plot(sorted_roi, cdf, color=Theme.ACCENT, linewidth=2)
            ax3.axhline(50, color=Theme.ORANGE, linestyle='--', alpha=0.7, label='Mediana')
            ax3.axvline(0, color=Theme.RED, linestyle='--', alpha=0.7, label='Break-even')
            # Mark percentiles
            p5 = np.percentile(roi_values, 5)
            p95 = np.percentile(roi_values, 95)
            ax3.axvline(p5, color=Theme.RED, linestyle=':', alpha=0.8, label=f'P5: {p5:.1f}%')
            ax3.axvline(p95, color=Theme.GREEN, linestyle=':', alpha=0.8, label=f'P95: {p95:.1f}%')
            ax3.set_title('Distribución Acumulada (CDF)', color=Theme.TEXT_PRIMARY, fontweight='bold')
            ax3.set_xlabel('ROI (%)', color=Theme.TEXT_MUTED)
            ax3.set_ylabel('Percentil (%)', color=Theme.TEXT_MUTED)
            ax3.legend(fontsize=7, loc='lower right')
            ax3.tick_params(colors=Theme.TEXT_MUTED)
            
            # Box Plot by Method
            ax4 = axes[1, 1]
            method_data = []
            method_labels = []
            for m in self.config.methods:
                m_roi = results_df.filter(pl.col("method") == m)["roi"].to_numpy()
                if len(m_roi) > 0:
                    method_data.append(m_roi)
                    method_labels.append(m.upper().replace("_", "\n"))
            
            bp = ax4.boxplot(method_data, labels=method_labels, patch_artist=True, 
                            widths=0.6, notch=True)
            colors_bp = [Theme.ACCENT, Theme.PURPLE, Theme.CYAN, Theme.ORANGE]
            for i, (patch, median) in enumerate(zip(bp['boxes'], bp['medians'])):
                patch.set_facecolor(colors_bp[i % len(colors_bp)])
                patch.set_alpha(0.7)
                median.set_color('white')
                median.set_linewidth(2)
            for whisker in bp['whiskers']:
                whisker.set_color(Theme.TEXT_MUTED)
            for cap in bp['caps']:
                cap.set_color(Theme.TEXT_MUTED)
            ax4.axhline(0, color=Theme.RED, linestyle='--', alpha=0.7)
            ax4.set_title('ROI por Método', color=Theme.TEXT_PRIMARY, fontweight='bold')
            ax4.set_ylabel('ROI (%)', color=Theme.TEXT_MUTED)
            ax4.tick_params(colors=Theme.TEXT_MUTED)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, facecolor=Theme.BG_PRIMARY)
            plt.close(fig)
            
            # =================================================================
            # PAGE 3: 3D SURFACE & ADVANCED ANALYSIS
            # =================================================================
            fig = plt.figure(figsize=(11, 8.5))
            fig.patch.set_facecolor(Theme.BG_PRIMARY)
            fig.suptitle('🏔️ 3D ROBUSTNESS LANDSCAPE', fontsize=16, fontweight='bold',
                        color=Theme.TEXT_PRIMARY, y=0.98)
            
            # 3D Surface: ROI vs Winrate vs Drawdown
            ax_3d = fig.add_subplot(221, projection='3d')
            ax_3d.set_facecolor(Theme.BG_SECONDARY)
            
            # Create 2D histogram for surface
            H, xedges, yedges = np.histogram2d(winrate_values, dd_values, bins=20)
            X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
            H = gaussian_filter(H.T, sigma=1)  # Smooth
            
            surf = ax_3d.plot_surface(X, Y, H, cmap='viridis', alpha=0.8, 
                                      edgecolor='none', antialiased=True)
            ax_3d.set_xlabel('Winrate (%)', color=Theme.TEXT_MUTED, fontsize=8)
            ax_3d.set_ylabel('Drawdown (%)', color=Theme.TEXT_MUTED, fontsize=8)
            ax_3d.set_zlabel('Frecuencia', color=Theme.TEXT_MUTED, fontsize=8)
            ax_3d.set_title('Densidad WR/DD', color=Theme.TEXT_PRIMARY, fontsize=10)
            ax_3d.tick_params(colors=Theme.TEXT_MUTED, labelsize=7)
            
            # 3D Scatter: ROI colored by profitability
            ax_3d2 = fig.add_subplot(222, projection='3d')
            ax_3d2.set_facecolor(Theme.BG_SECONDARY)
            
            colors_3d = [Theme.GREEN if r > 0 else Theme.RED for r in roi_values]
            ax_3d2.scatter(winrate_values, dd_values, roi_values, 
                          c=roi_values, cmap='RdYlGn', alpha=0.6, s=15)
            ax_3d2.set_xlabel('Winrate (%)', color=Theme.TEXT_MUTED, fontsize=8)
            ax_3d2.set_ylabel('Drawdown (%)', color=Theme.TEXT_MUTED, fontsize=8)
            ax_3d2.set_zlabel('ROI (%)', color=Theme.TEXT_MUTED, fontsize=8)
            ax_3d2.set_title('ROI en Espacio WR/DD', color=Theme.TEXT_PRIMARY, fontsize=10)
            ax_3d2.tick_params(colors=Theme.TEXT_MUTED, labelsize=7)
            
            # Heatmap: ROI by Winrate/Drawdown bins
            ax_heat = fig.add_subplot(223)
            ax_heat.set_facecolor(Theme.BG_SECONDARY)
            
            # Create binned heatmap
            wr_bins = np.linspace(winrate_values.min(), winrate_values.max(), 15)
            dd_bins = np.linspace(dd_values.min(), dd_values.max(), 15)
            
            heatmap_data = np.zeros((len(dd_bins)-1, len(wr_bins)-1))
            for i in range(len(dd_bins)-1):
                for j in range(len(wr_bins)-1):
                    mask = ((winrate_values >= wr_bins[j]) & (winrate_values < wr_bins[j+1]) &
                            (dd_values >= dd_bins[i]) & (dd_values < dd_bins[i+1]))
                    if np.sum(mask) > 0:
                        heatmap_data[i, j] = np.mean(roi_values[mask])
            
            heatmap_data = gaussian_filter(heatmap_data, sigma=0.8)
            im = ax_heat.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', origin='lower',
                               extent=[wr_bins[0], wr_bins[-1], dd_bins[0], dd_bins[-1]])
            ax_heat.set_xlabel('Winrate (%)', color=Theme.TEXT_MUTED)
            ax_heat.set_ylabel('Drawdown (%)', color=Theme.TEXT_MUTED)
            ax_heat.set_title('ROI Heatmap (WR/DD)', color=Theme.TEXT_PRIMARY, fontsize=10)
            ax_heat.tick_params(colors=Theme.TEXT_MUTED)
            cbar = plt.colorbar(im, ax=ax_heat)
            cbar.ax.tick_params(colors=Theme.TEXT_MUTED)
            cbar.set_label('ROI Promedio %', color=Theme.TEXT_MUTED)
            
            # Violin Plot
            ax_violin = fig.add_subplot(224)
            ax_violin.set_facecolor(Theme.BG_SECONDARY)
            
            parts = ax_violin.violinplot(method_data, positions=range(len(method_labels)),
                                         showmeans=True, showmedians=True)
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors_bp[i % len(colors_bp)])
                pc.set_alpha(0.7)
            parts['cmeans'].set_color(Theme.GREEN)
            parts['cmedians'].set_color('white')
            ax_violin.axhline(0, color=Theme.RED, linestyle='--', alpha=0.7)
            ax_violin.set_xticks(range(len(method_labels)))
            ax_violin.set_xticklabels(method_labels)
            ax_violin.set_ylabel('ROI (%)', color=Theme.TEXT_MUTED)
            ax_violin.set_title('Distribución por Método (Violin)', color=Theme.TEXT_PRIMARY, fontsize=10)
            ax_violin.tick_params(colors=Theme.TEXT_MUTED)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, facecolor=Theme.BG_PRIMARY)
            plt.close(fig)
            
            # =================================================================
            # PAGE 4: TIME SERIES & CONVERGENCE
            # =================================================================
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            fig.patch.set_facecolor(Theme.BG_PRIMARY)
            fig.suptitle('📈 CONVERGENCE & EVOLUTION', fontsize=16, fontweight='bold',
                        color=Theme.TEXT_PRIMARY, y=0.98)
            
            for ax in axes.flat:
                ax.set_facecolor(Theme.BG_SECONDARY)
                for spine in ax.spines.values():
                    spine.set_color(Theme.BORDER)
            
            # Running average ROI
            ax1 = axes[0, 0]
            running_avg = np.cumsum(roi_values) / np.arange(1, len(roi_values) + 1)
            running_std = np.array([np.std(roi_values[:i+1]) for i in range(len(roi_values))])
            x = np.arange(len(roi_values))
            ax1.fill_between(x, running_avg - running_std, running_avg + running_std, 
                            alpha=0.3, color=Theme.ACCENT)
            ax1.plot(x, running_avg, color=Theme.ACCENT, linewidth=2, label='Media Móvil')
            ax1.axhline(0, color=Theme.RED, linestyle='--', alpha=0.7)
            ax1.axhline(running_avg[-1], color=Theme.GREEN, linestyle=':', alpha=0.8,
                       label=f'Final: {running_avg[-1]:.2f}%')
            ax1.set_title('Convergencia ROI Promedio', color=Theme.TEXT_PRIMARY, fontweight='bold')
            ax1.set_xlabel('Simulación #', color=Theme.TEXT_MUTED)
            ax1.set_ylabel('ROI Promedio (%)', color=Theme.TEXT_MUTED)
            ax1.legend(fontsize=8)
            ax1.tick_params(colors=Theme.TEXT_MUTED)
            
            # Running robustness %
            ax2 = axes[0, 1]
            profitable_cumsum = np.cumsum(roi_values > 0)
            running_robustness = profitable_cumsum / np.arange(1, len(roi_values) + 1) * 100
            ax2.plot(x, running_robustness, color=Theme.CYAN, linewidth=2)
            ax2.axhline(70, color=Theme.GREEN, linestyle='--', alpha=0.7, label='70% Robusto')
            ax2.axhline(50, color=Theme.ORANGE, linestyle='--', alpha=0.7, label='50% Moderado')
            ax2.fill_between(x, running_robustness, 70, where=(running_robustness >= 70),
                            alpha=0.3, color=Theme.GREEN)
            ax2.fill_between(x, running_robustness, 50, where=((running_robustness >= 50) & (running_robustness < 70)),
                            alpha=0.3, color=Theme.ORANGE)
            ax2.fill_between(x, running_robustness, 0, where=(running_robustness < 50),
                            alpha=0.3, color=Theme.RED)
            ax2.set_title('Convergencia Robustez', color=Theme.TEXT_PRIMARY, fontweight='bold')
            ax2.set_xlabel('Simulación #', color=Theme.TEXT_MUTED)
            ax2.set_ylabel('Robustez (%)', color=Theme.TEXT_MUTED)
            ax2.legend(fontsize=8)
            ax2.tick_params(colors=Theme.TEXT_MUTED)
            ax2.set_ylim(0, 100)
            
            # ROI scatter by simulation order
            ax3 = axes[1, 0]
            colors_sim = [Theme.GREEN if r > 0 else Theme.RED for r in roi_values]
            ax3.scatter(x, roi_values, c=colors_sim, alpha=0.5, s=10)
            ax3.axhline(0, color='white', linestyle='--', alpha=0.7)
            ax3.axhline(np.mean(roi_values), color=Theme.CYAN, linestyle='-', alpha=0.8,
                       label=f'Media: {np.mean(roi_values):.1f}%')
            ax3.set_title('ROI por Simulación', color=Theme.TEXT_PRIMARY, fontweight='bold')
            ax3.set_xlabel('Simulación #', color=Theme.TEXT_MUTED)
            ax3.set_ylabel('ROI (%)', color=Theme.TEXT_MUTED)
            ax3.legend(fontsize=8)
            ax3.tick_params(colors=Theme.TEXT_MUTED)
            
            # Best/Worst tracking
            ax4 = axes[1, 1]
            running_best = np.maximum.accumulate(roi_values)
            running_worst = np.minimum.accumulate(roi_values)
            ax4.fill_between(x, running_worst, running_best, alpha=0.2, color=Theme.ACCENT)
            ax4.plot(x, running_best, color=Theme.GREEN, linewidth=2, label=f'Mejor: {running_best[-1]:.1f}%')
            ax4.plot(x, running_worst, color=Theme.RED, linewidth=2, label=f'Peor: {running_worst[-1]:.1f}%')
            ax4.axhline(0, color='white', linestyle='--', alpha=0.5)
            ax4.set_title('Rango Mejor/Peor Acumulado', color=Theme.TEXT_PRIMARY, fontweight='bold')
            ax4.set_xlabel('Simulación #', color=Theme.TEXT_MUTED)
            ax4.set_ylabel('ROI (%)', color=Theme.TEXT_MUTED)
            ax4.legend(fontsize=8)
            ax4.tick_params(colors=Theme.TEXT_MUTED)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, facecolor=Theme.BG_PRIMARY)
            plt.close(fig)
            
            # =================================================================
            # PAGE 5: METHOD COMPARISON
            # =================================================================
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            fig.patch.set_facecolor(Theme.BG_PRIMARY)
            fig.suptitle('📊 METHOD COMPARISON', fontsize=16, fontweight='bold',
                        color=Theme.TEXT_PRIMARY, y=0.98)
            
            for ax in axes.flat:
                ax.set_facecolor(Theme.BG_SECONDARY)
                for spine in ax.spines.values():
                    spine.set_color(Theme.BORDER)
            
            methods = list(self.analysis["by_method"].keys())
            
            # Robustness comparison bar
            ax1 = axes[0, 0]
            robustness = [self.analysis["by_method"][m]["robustness"] for m in methods]
            colors_bar = [Theme.GREEN if r >= 70 else (Theme.ORANGE if r >= 50 else Theme.RED) for r in robustness]
            bars = ax1.bar([m.upper() for m in methods], robustness, color=colors_bar, alpha=0.8)
            ax1.axhline(70, color=Theme.GREEN, linestyle='--', linewidth=2, label='70%')
            ax1.axhline(50, color=Theme.ORANGE, linestyle='--', linewidth=2, label='50%')
            for bar, val in zip(bars, robustness):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%',
                        ha='center', fontsize=10, fontweight='bold', color=Theme.TEXT_PRIMARY)
            ax1.set_title('Robustez por Método', color=Theme.TEXT_PRIMARY, fontweight='bold')
            ax1.set_ylabel('Robustez (%)', color=Theme.TEXT_MUTED)
            ax1.legend(fontsize=8)
            ax1.tick_params(colors=Theme.TEXT_MUTED)
            ax1.set_ylim(0, 105)
            
            # ROI Mean comparison
            ax2 = axes[0, 1]
            roi_means = [self.analysis["by_method"][m]["roi_mean"] for m in methods]
            colors_roi = [Theme.GREEN if r > 0 else Theme.RED for r in roi_means]
            bars = ax2.bar([m.upper() for m in methods], roi_means, color=colors_roi, alpha=0.8)
            ax2.axhline(0, color='white', linestyle='-', linewidth=1)
            for bar, val in zip(bars, roi_means):
                y_pos = bar.get_height() + 0.5 if val >= 0 else bar.get_height() - 1.5
                ax2.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:+.1f}%',
                        ha='center', fontsize=10, fontweight='bold', color=Theme.TEXT_PRIMARY)
            ax2.set_title('ROI Promedio por Método', color=Theme.TEXT_PRIMARY, fontweight='bold')
            ax2.set_ylabel('ROI (%)', color=Theme.TEXT_MUTED)
            ax2.tick_params(colors=Theme.TEXT_MUTED)
            
            # Overlapping histograms
            ax3 = axes[1, 0]
            for i, m in enumerate(methods):
                m_roi = results_df.filter(pl.col("method") == m)["roi"].to_numpy()
                ax3.hist(m_roi, bins=30, alpha=0.5, label=m.upper(), color=colors_bp[i % len(colors_bp)])
            ax3.axvline(0, color='white', linestyle='--', alpha=0.7)
            ax3.set_title('Distribución ROI Comparada', color=Theme.TEXT_PRIMARY, fontweight='bold')
            ax3.set_xlabel('ROI (%)', color=Theme.TEXT_MUTED)
            ax3.legend(fontsize=8)
            ax3.tick_params(colors=Theme.TEXT_MUTED)
            
            # Summary stats table as text
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            # Create table data
            table_data = [['Método', 'Sims', 'Rentables', 'Robust%', 'ROI μ', 'ROI σ']]
            for m in methods:
                data = self.analysis["by_method"][m]
                table_data.append([
                    m.upper(),
                    str(data["count"]),
                    str(data["profitable"]),
                    f'{data["robustness"]:.1f}%',
                    f'{data["roi_mean"]:+.2f}%',
                    f'{data["roi_std"]:.2f}%'
                ])
            
            table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                             colWidths=[0.2, 0.12, 0.14, 0.14, 0.14, 0.14])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.8)
            
            # Style header
            for i in range(len(table_data[0])):
                table[(0, i)].set_facecolor(Theme.BG_CARD)
                table[(0, i)].set_text_props(color=Theme.ACCENT, fontweight='bold')
            # Style data rows
            for row in range(1, len(table_data)):
                for col in range(len(table_data[0])):
                    table[(row, col)].set_facecolor(Theme.BG_SECONDARY)
                    table[(row, col)].set_text_props(color=Theme.TEXT_PRIMARY)
            
            ax4.set_title('Resumen Estadístico', color=Theme.TEXT_PRIMARY, fontweight='bold', pad=20)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, facecolor=Theme.BG_PRIMARY)
            plt.close(fig)
        
        if RICH_AVAILABLE:
            console.print(f"   [dim]PDF: {pdf_path.name} (5 páginas)[/dim]")
    
    def _export_best_trial_chart(self, output_dir: Path, base_name: str):
        """Generate interactive HTML chart for the best Monte Carlo trial."""
        if not GRAFICO_AVAILABLE:
            if RICH_AVAILABLE:
                console.print("   [dim yellow]⚠ visual.grafico no disponible, saltando gráfico HTML[/dim]")
            return
        
        if self.best_sim_data is None or self.best_sim_df is None:
            if RICH_AVAILABLE:
                console.print("   [dim yellow]⚠ No hay datos del mejor trial para generar gráfico[/dim]")
            return
        
        trades_df = self.best_sim_data.get("_trades_df")
        equity_curve = self.best_sim_data.get("_equity_curve")
        signals_df = self.best_sim_data.get("_signals_df")
        
        if trades_df is None or signals_df is None:
            return
        
        # Merge signals with the original OHLCV data
        # signals_df may not have OHLC columns, so we use best_sim_df which has the synthetic market data
        # and merge the signals onto it
        if "open" not in signals_df.columns and "open" in self.best_sim_df.columns:
            # signals_df should have timestamp, use that to merge with OHLC data
            if "timestamp" in signals_df.columns and "timestamp" in self.best_sim_df.columns:
                df_for_chart = self.best_sim_df.join(
                    signals_df.select([c for c in signals_df.columns if c != "timestamp" or c == "timestamp"]),
                    on="timestamp",
                    how="left"
                )
            else:
                # Fallback: use best_sim_df directly if it has signals
                df_for_chart = self.best_sim_df
        else:
            df_for_chart = signals_df
        
        # Build params for chart
        params = self._build_params()
        
        # Get metrics
        best_roi = self.best_sim_data.get("roi", 0)
        best_idx = 0
        for i, r in enumerate(self.simulation_results):
            if r.get("roi") == best_roi:
                best_idx = i
                break
        
        metrics = {
            "roi": self.best_sim_data.get("roi", 0),
            "winrate": self.best_sim_data.get("winrate", 0),
            "drawdown": self.best_sim_data.get("drawdown", 0),
            "sharpe": self.best_sim_data.get("sharpe", 0),
            "profit_factor": self.best_sim_data.get("profit_factor", 0),
            "total_trades": self.best_sim_data.get("n_trades", 0),
        }
        
        chart_dir = output_dir / "charts"
        chart_dir.mkdir(parents=True, exist_ok=True)
        
        # Select random 2-month window within the backtest range
        import random
        from datetime import timedelta
        
        # Get the full date range from the data
        if "timestamp" in df_for_chart.columns:
            timestamps = df_for_chart["timestamp"].to_list()
            data_start = min(timestamps)
            data_end = max(timestamps)
        else:
            data_start = self.config.fecha_inicio
            data_end = self.config.fecha_fin
        
        # Calculate 2-month window (60 days)
        two_months = timedelta(days=60)
        total_range = data_end - data_start
        
        if total_range > two_months:
            # Random start within valid range
            max_start = data_end - two_months
            random_offset = random.random() * (max_start - data_start).total_seconds()
            chart_start = data_start + timedelta(seconds=random_offset)
            chart_end = chart_start + two_months
        else:
            # Data range is less than 2 months, use full range
            chart_start = data_start
            chart_end = data_end
        
        try:
            plot_trades(
                df=df_for_chart,
                df_trades=trades_df,
                plot_base=str(chart_dir),
                fecha_inicio_plot=chart_start,
                fecha_fin_plot=chart_end,
                trial_number=best_idx,
                params=params,
                score=best_roi,
                combo=f"MC_BEST_{self.best_sim_method.upper()}",
                metrics=metrics,
                equity_curve=equity_curve,
                saldo_inicial=self.config.saldo_inicial,
                max_archivos=1,
                activo=self.config.activo,
            )
            
            if RICH_AVAILABLE:
                chart_range = f"{chart_start.strftime('%Y-%m-%d')} → {chart_end.strftime('%Y-%m-%d')}"
                console.print(f"   [dim]HTML Chart: charts/TRIAL-{best_idx}_*.html ({chart_range})[/dim]")
        except Exception as e:
            if RICH_AVAILABLE:
                import traceback
                from rich.text import Text
                error_text = Text(f"   ⚠ Error generando gráfico: {e}", style="dim red")
                console.print(error_text)
                traceback.print_exc()
    
    def plot_distributions(self) -> None:
        """Show interactive plots."""
        if not self.config.show_plots or not MATPLOTLIB_AVAILABLE:
            return
        
        results_df = pl.DataFrame(self.simulation_results)
        roi_values = results_df["roi"].to_numpy()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.patch.set_facecolor(Theme.BG_PRIMARY)
        fig.suptitle(f'Monte Carlo Results: {self.strategy.name}', fontsize=14,
                    fontweight='bold', color=Theme.TEXT_PRIMARY)
        
        for ax in axes.flat:
            ax.set_facecolor(Theme.BG_SECONDARY)
        
        # ROI Histogram
        axes[0, 0].hist(roi_values, bins=50, color=Theme.ACCENT, alpha=0.7)
        axes[0, 0].axvline(0, color=Theme.RED, linestyle='--', linewidth=2)
        axes[0, 0].set_title('ROI Distribution', color=Theme.TEXT_PRIMARY)
        axes[0, 0].tick_params(colors=Theme.TEXT_MUTED)
        
        # Cumulative distribution
        sorted_roi = np.sort(roi_values)
        cdf = np.arange(1, len(sorted_roi) + 1) / len(sorted_roi)
        axes[0, 1].plot(sorted_roi, cdf * 100, color=Theme.CYAN, linewidth=2)
        axes[0, 1].axhline(50, color=Theme.ORANGE, linestyle='--', alpha=0.5)
        axes[0, 1].axvline(0, color=Theme.RED, linestyle='--', alpha=0.5)
        axes[0, 1].set_title('Cumulative Distribution', color=Theme.TEXT_PRIMARY)
        axes[0, 1].set_ylabel('Percentile (%)', color=Theme.TEXT_MUTED)
        axes[0, 1].tick_params(colors=Theme.TEXT_MUTED)
        
        # Box plot by method
        method_data = [results_df.filter(pl.col("method") == m)["roi"].to_numpy() 
                      for m in self.config.methods]
        bp = axes[1, 0].boxplot(method_data, labels=[m.upper() for m in self.config.methods],
                               patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(Theme.ACCENT)
            patch.set_alpha(0.7)
        axes[1, 0].axhline(0, color=Theme.RED, linestyle='--')
        axes[1, 0].set_title('ROI by Method', color=Theme.TEXT_PRIMARY)
        axes[1, 0].tick_params(colors=Theme.TEXT_MUTED)
        
        # Scatter: ROI vs Trades
        axes[1, 1].scatter(results_df["n_trades"].to_numpy(), roi_values, 
                          c=roi_values, cmap='RdYlGn', alpha=0.5, s=10)
        axes[1, 1].axhline(0, color=Theme.RED, linestyle='--')
        axes[1, 1].set_title('ROI vs Number of Trades', color=Theme.TEXT_PRIMARY)
        axes[1, 1].set_xlabel('Trades', color=Theme.TEXT_MUTED)
        axes[1, 1].set_ylabel('ROI (%)', color=Theme.TEXT_MUTED)
        axes[1, 1].tick_params(colors=Theme.TEXT_MUTED)
        
        plt.tight_layout()
        plt.show()
    
    # =========================================================================
    # MAIN EXECUTION
    # =========================================================================
    def run(self) -> Dict[str, Any]:
        """Run complete Monte Carlo validation."""
        if RICH_AVAILABLE:
            console.clear()
            header = Panel(
                "[bold cyan]MODELOX MONTE CARLO ROBUSTNESS VALIDATOR[/bold cyan]\n"
                "[dim]Testing strategy survival across synthetic markets[/dim]",
                border_style="cyan",
                padding=(1, 2)
            )
            console.print(Align.center(header))
        
        self.load_strategy()
        self.load_data()
        self.run_monte_carlo()
        self.analyze_results()
        self.print_report()
        self.export_results()
        self.plot_distributions()
        
        return self.analysis


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    defaults = MonteCarloConfig()
    
    parser = argparse.ArgumentParser(
        description="Monte Carlo Robustness Validator for ModeloX strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python montecarlo.py                          # Use defaults from config
  python montecarlo.py -id 15 -n 500            # 500 simulations for strategy 15
  python montecarlo.py --noise_pct 0.2          # Higher noise level
  python montecarlo.py --methods noise          # Only noise injection
        """
    )
    
    parser.add_argument("--combinacion_id", "-id", type=int, default=defaults.combinacion_id)
    parser.add_argument("--n_simulations", "-n", type=int, default=defaults.n_simulations)
    parser.add_argument("--noise_pct", type=float, default=defaults.noise_pct,
                       help="Noise percentage for price perturbation (default: 0.1)")
    parser.add_argument("--block_size", type=int, default=defaults.block_size,
                       help="Block size for bootstrap (default: 1440 = 1 day)")
    parser.add_argument("--methods", type=str, nargs="+", 
                       choices=["noise", "block_bootstrap"],
                       default=defaults.methods)
    
    parser.add_argument("--exit_type", type=str, choices=["pnl_fixed", "pnl_trailing"],
                       default=defaults.exit_type)
    parser.add_argument("--sl", type=float, default=defaults.exit_sl_pct)
    parser.add_argument("--tp", type=float, default=defaults.exit_tp_pct)
    
    parser.add_argument("--activo", "-a", type=str, default=defaults.activo)
    parser.add_argument("--timeframe", "-tf", type=str, default=defaults.timeframe)
    parser.add_argument("--fecha_inicio", type=str, default=defaults.fecha_inicio)
    parser.add_argument("--fecha_fin", type=str, default=defaults.fecha_fin)
    
    parser.add_argument("--no_csv", action="store_true")
    parser.add_argument("--no_excel", action="store_true")
    parser.add_argument("--no_pdf", action="store_true")
    parser.add_argument("--no_plots", action="store_true")
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    config = MonteCarloConfig(
        combinacion_id=args.combinacion_id,
        n_simulations=args.n_simulations,
        noise_pct=args.noise_pct,
        block_size=args.block_size,
        methods=args.methods,
        exit_type=args.exit_type,
        exit_sl_pct=args.sl,
        exit_tp_pct=args.tp,
        activo=args.activo,
        timeframe=args.timeframe,
        fecha_inicio=args.fecha_inicio,
        fecha_fin=args.fecha_fin,
        export_csv=not args.no_csv,
        export_excel=not args.no_excel,
        export_pdf=not args.no_pdf,
        show_plots=not args.no_plots,
    )
    
    validator = MonteCarloValidator(config)
    validator.run()


if __name__ == "__main__":
    main()
