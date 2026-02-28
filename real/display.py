"""
VISUALIZACIÓN RICH PARA TRADING REAL
=====================================
Interfaz de terminal elegante para el trading en vivo.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.layout import Layout


console = Console()


# =============================================================================
# TEMA VISUAL
# =============================================================================

class Colors:
    """Paleta de colores para trading."""
    PROFIT = "green3"
    LOSS = "red3"
    NEUTRAL = "grey70"
    ACCENT = "cyan"
    HEADER = "bold white"
    MUTED = "grey50"
    GOLD = "gold3"
    BLUE = "steel_blue3"
    WARNING = "orange3"


# =============================================================================
# COMPONENTES DE UI
# =============================================================================

def create_header(strategy_name: str, symbol: str, market_type: str) -> Panel:
    """Crea el header principal."""
    title = Text()
    title.append("REAL TRADING\n", style="bold white")
    title.append(f"Estrategia: ", style="dim white")
    title.append(f"{strategy_name}\n", style="white")
    title.append(f"Simbolo: ", style="dim white")
    title.append(f"{symbol} ", style="white")
    title.append(f"({market_type.upper()})", style="dim")
    
    return Panel(
        Align.center(title),
        box=box.DOUBLE,
        border_style="dim white",
        padding=(1, 2),
    )


def create_balance_panel(balance_info: Dict[str, Any]) -> Panel:
    """Panel de balance de cuenta con información detallada del exchange."""
    balance = balance_info.get("balance_usdt", 0)
    equity = balance_info.get("equity", balance)
    used_margin = balance_info.get("used_margin", 0)
    unrealized_pnl = balance_info.get("unrealized_pnl", 0)
    realized_pnl = balance_info.get("realized_pnl_today", 0)
    currency = balance_info.get("quote_currency", "USDT")
    
    content = Text()
    content.append("CUENTA EN VIVO\n\n", style="bold")
    
    # Balance disponible
    content.append(f"Disponible: ", style="dim white")
    content.append(f"${balance:,.2f} {currency}\n", style="white")
    
    # Equity (balance + PnL no realizado)
    if equity != balance:
        content.append(f"Equity: ", style="dim white")
        content.append(f"${equity:,.2f}\n", style="white")
    
    # Margen usado
    if used_margin > 0:
        content.append(f"Margen usado: ", style="dim white")
        content.append(f"${used_margin:,.2f}\n", style="white")
    
    # PnL no realizado (posiciones abiertas)
    if unrealized_pnl != 0:
        content.append(f"PnL abierto: ", style="dim white")
        pnl_color = "#5daa68" if unrealized_pnl >= 0 else "#c0392b"
        pnl_sign = "+" if unrealized_pnl >= 0 else ""
        content.append(f"{pnl_sign}${unrealized_pnl:,.2f}\n", style=pnl_color)
    
    # PnL realizado hoy
    if realized_pnl != 0:
        content.append(f"PnL hoy: ", style="dim white")
        pnl_color2 = "#5daa68" if realized_pnl >= 0 else "#c0392b"
        pnl_sign = "+" if realized_pnl >= 0 else ""
        content.append(f"{pnl_sign}${realized_pnl:,.2f}\n", style=pnl_color2)
    
    content.append(f"\nTrades activos: ", style="dim white")
    content.append(f"{balance_info.get('active_trades', 0)}", style="white")
    
    return Panel(
        content,
        title="[bold]CUENTA[/bold]",
        box=box.ROUNDED,
        border_style="dim white",
    )


def create_config_panel(config: Any) -> Panel:
    """Panel de configuracion."""
    content = Text()
    content.append("CONFIGURACION\n\n", style="bold")
    content.append(f"APALANCAMIENTO: ", style="dim white")
    content.append(f"{config.leverage}x\n", style="white")
    content.append(f"POR TRADE: ", style="dim white")
    content.append(f"${config.amount_per_trade:,.2f}\n", style="white")
    content.append(f"STOP LOSS: ", style="dim white")
    content.append(f"-{config.stop_loss_pct}%\n", style="white")
    content.append(f"TAKE PROFIT: ", style="dim white")
    content.append(f"+{config.take_profit_pct}%\n", style="white")
    content.append(f"MODO: ", style="dim white")
    
    exec_mode = getattr(config, 'execution_mode', 'auto')
    content.append(exec_mode.upper(), style="white")
    
    return Panel(
        content,
        title="[bold]CONFIG[/bold]",
        box=box.ROUNDED,
        border_style="dim white",
    )


def create_positions_table(trades: List[Any]) -> Panel:
    """Tabla de posiciones activas."""
    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold",
        expand=True,
    )
    
    table.add_column("Simbolo", style="white")
    table.add_column("Lado", justify="center")
    table.add_column("Entrada", justify="right")
    table.add_column("SL", justify="right", style="white")
    table.add_column("TP", justify="right", style="white")
    table.add_column("Cantidad", justify="right")
    table.add_column("Tiempo", justify="right", style="dim")
    
    for trade in trades:
        side_style = "white"
        side_text = "LONG" if trade.side == "LONG" else "SHORT"
        
        # Tiempo desde la entrada
        elapsed = datetime.now() - trade.timestamp
        elapsed_str = f"{int(elapsed.total_seconds() // 60)}m"
        
        table.add_row(
            trade.symbol,
            Text(side_text, style=side_style),
            f"${trade.entry_price:,.2f}",
            f"${trade.sl_price:,.2f}",
            f"${trade.tp_price:,.2f}",
            f"{trade.quantity:.4f}",
            elapsed_str,
        )
    
    if not trades:
        table.add_row(
            "[dim]Sin posiciones abiertas[/dim]",
            "", "", "", "", "", ""
        )
    
    return Panel(
        table,
        title="[bold]POSICIONES ACTIVAS[/bold]",
        box=box.ROUNDED,
        border_style="dim white",
    )


def create_history_table(trades: List[Any], limit: int = 10) -> Panel:
    """Tabla de historial de trades."""
    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold",
        expand=True,
    )
    
    table.add_column("Hora", style="dim")
    table.add_column("Simbolo", style="white")
    table.add_column("Lado", justify="center")
    table.add_column("Entrada", justify="right")
    table.add_column("Salida", justify="right")
    table.add_column("PnL", justify="right")
    table.add_column("Razón", justify="center")
    
    for trade in trades[-limit:]:
        side_style = "white"
        side_text = "LONG" if trade.side == "LONG" else "SHORT"
        
        pnl = trade.pnl or 0
        pnl_color = "#5daa68" if pnl > 0 else "#c0392b"
        pnl_text = f"+${pnl:.2f}" if pnl > 0 else f"-${abs(pnl):.2f}"
        
        reason_style = "white"
        
        table.add_row(
            trade.exit_time.strftime("%H:%M") if trade.exit_time else "-",
            trade.symbol,
            Text(side_text, style=side_style),
            f"${trade.entry_price:,.2f}",
            f"${trade.exit_price:,.2f}" if trade.exit_price else "-",
            Text(pnl_text, style=pnl_color),
            Text(trade.reason or "-", style=reason_style),
        )
    
    if not trades:
        table.add_row("[dim]Sin historial[/dim]", "", "", "", "", "", "")
    
    return Panel(
        table,
        title="[bold]HISTORIAL[/bold]",
        box=box.ROUNDED,
        border_style="dim white",
    )


def create_stats_panel(stats: Dict[str, Any]) -> Panel:
    """Panel de estadísticas."""
    content = Text()
    content.append("ESTADISTICAS\n\n", style="bold")
    
    content.append(f"Total trades: ", style="dim white")
    content.append(f"{stats.get('total_trades', 0)}\n", style="white")
    
    content.append(f"Ganados: ", style="dim white")
    content.append(f"{stats.get('wins', 0)} ", style="white")
    content.append(f"/ Perdidos: ", style="dim white")
    content.append(f"{stats.get('losses', 0)}\n", style="white")
    
    winrate = stats.get('winrate', 0)
    content.append(f"Winrate: ", style="dim white")
    content.append(f"{winrate:.1f}%\n", style="white")
    
    pnl = stats.get('total_pnl', 0)
    pnl_color = "#5daa68" if pnl >= 0 else "#c0392b"
    pnl_sign = "+" if pnl >= 0 else ""
    content.append(f"PnL Total: ", style="dim white")
    content.append(f"{pnl_sign}${pnl:,.2f}", style=pnl_color)
    
    return Panel(
        content,
        title="[bold]STATS[/bold]",
        box=box.ROUNDED,
        border_style="dim white",
    )


def create_market_panel(symbol: str, price: float, change_pct: float = 0) -> Panel:
    """Panel de precio de mercado."""
    content = Text()
    content.append("MERCADO\n\n", style="bold")
    content.append(f"{symbol}\n", style="white")
    content.append(f"${price:,.2f}", style="bold white")
    
    if change_pct != 0:
        change_color = "#5daa68" if change_pct > 0 else "#c0392b"
        change_sign = "+" if change_pct > 0 else ""
        content.append(f"\n{change_sign}{change_pct:.2f}%", style=change_color)
    
    return Panel(
        Align.center(content),
        box=box.ROUNDED,
        border_style="dim white",
    )


# =============================================================================
# PANTALLAS PRINCIPALES
# =============================================================================

def show_startup_screen(
    strategy_name: str,
    strategy_id: int,
    config: Any,
    balance: float,
) -> None:
    """Muestra la pantalla de inicio."""
    console.clear()
    
    from real.config import DISPLAY_NAMES
    display_symbol = DISPLAY_NAMES.get(config.symbol, config.symbol)
    
    # Info de configuracion
    info_table = Table(box=box.ROUNDED, show_header=False, expand=True, border_style="dim white")
    info_table.add_column("Campo", style="dim white")
    info_table.add_column("Valor", style="white")
    
    info_table.add_row("ESTRATEGIA", f"{strategy_name} (ID: {strategy_id})")
    info_table.add_row("ACTIVO", display_symbol)
    info_table.add_row("APALANCAMIENTO", f"{config.leverage}x")
    info_table.add_row("SALDO DISPONIBLE", f"${balance:,.2f} {config.quote_currency}")
    info_table.add_row("MONTO POR TRADE", f"${config.amount_per_trade:,.2f}")
    info_table.add_row("STOP LOSS", f"-{config.stop_loss_pct}%")
    info_table.add_row("TAKE PROFIT", f"+{config.take_profit_pct}%")
    info_table.add_row("TIMEFRAME", f"{config.timeframe}")
    
    exec_mode = getattr(config, 'execution_mode', 'auto')
    info_table.add_row("MODO", exec_mode.upper())
    
    console.print(Panel(info_table, title="[bold]CONFIGURACION[/bold]", border_style="dim white"))
    console.print()


def show_strategies_list(strategies: List[tuple]) -> None:
    """Muestra la lista de estrategias disponibles."""
    console.print("\n[bold]ESTRATEGIAS DISPONIBLES[/bold]\n")
    
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold", border_style="dim white")
    table.add_column("ID", justify="center", style="bold white")
    table.add_column("Nombre", style="white")
    
    for strategy_id, name in strategies:
        table.add_row(str(strategy_id), name)
    
    console.print(table)
    console.print()


def show_connection_status(success: bool, balance: float = 0, currency: str = "USDT") -> None:
    """Muestra el estado de la conexión."""
    if success:
        console.print(f"\nConexion exitosa a BingX")
        console.print(f"[dim]Balance: ${balance:,.2f} {currency}[/]\n")
    else:
        console.print(f"\nError de conexion a BingX")
        console.print(f"[dim]Verifica tus API keys[/]\n")


def show_trade_executed(trade: Any, is_open: bool = True, config: Any = None) -> None:
    """Muestra un trade ejecutado con un panel profesional."""
    if is_open:
        # Crear tabla con detalles
        table = Table(show_header=False, box=None, padding=(0, 3))
        table.add_column("Label", style="dim white", justify="right", width=22)
        table.add_column("Value", style="bold white", justify="left", width=30)
        
        # Información principal
        from real.config import DISPLAY_NAMES
        display_symbol = DISPLAY_NAMES.get(trade.symbol, trade.symbol)
        
        table.add_row("DIRECCION", trade.side)
        table.add_row("SIMBOLO", display_symbol)
        table.add_row("PRECIO ENTRADA", f"${trade.entry_price:,.2f}")
        table.add_row("CANTIDAD", f"{trade.quantity:.6f}")
        table.add_row("APALANCAMIENTO", f"{trade.leverage}x")
        table.add_row("", "")
        
        # Stop Loss y Take Profit
        sl_pct = abs((trade.sl_price - trade.entry_price) / trade.entry_price * 100)
        tp_pct = abs((trade.tp_price - trade.entry_price) / trade.entry_price * 100)
        
        table.add_row("STOP LOSS", f"${trade.sl_price:,.2f} (-{sl_pct:.1f}%)")
        table.add_row("TAKE PROFIT", f"${trade.tp_price:,.2f} (+{tp_pct:.1f}%)")
        
        # Trailing Stop si está configurado
        if config and hasattr(config, 'trailing_stop_enabled') and config.trailing_stop_enabled:
            table.add_row("", "")
            table.add_row("TRAILING STOP", "ACTIVADO")
            table.add_row("   ACTIVACION", f"+{config.trailing_stop_activation_pct:.1f}%")
            table.add_row("   DISTANCIA", f"{config.trailing_stop_distance_pct:.1f}%")
        
        # Valor de la posición
        position_value = trade.entry_price * trade.quantity
        margin_used = position_value / trade.leverage
        table.add_row("", "")
        table.add_row("VALOR POSICION", f"${position_value:,.2f}")
        table.add_row("MARGEN USADO", f"${margin_used:,.2f}")
        
        # Panel
        panel = Panel(
            Align.center(table),
            title=f"[bold]{trade.side} POSICION ABIERTA[/bold]",
            title_align="center",
            border_style="dim white",
            padding=(1, 2),
        )
        
        console.print()
        console.print(panel)
        console.print()
        
    else:
        pnl = trade.pnl or 0
        reason = trade.reason or "Manual"
        
        # Crear tabla con detalles
        table = Table(show_header=False, box=None, padding=(0, 3))
        table.add_column("Label", style="dim white", justify="right", width=22)
        table.add_column("Value", style="bold white", justify="left", width=30)
        
        from real.config import DISPLAY_NAMES
        display_symbol = DISPLAY_NAMES.get(trade.symbol, trade.symbol)
        
        table.add_row("ACTIVO", display_symbol)
        table.add_row("DIRECCION", trade.side)
        table.add_row("PRECIO ENTRADA", f"${trade.entry_price:,.2f}")
        table.add_row("PRECIO SALIDA", f"${trade.exit_price:,.2f}")
        table.add_row("RAZON CIERRE", reason)
        table.add_row("", "")
        
        pnl_color = "#5daa68" if pnl >= 0 else "#c0392b"
        table.add_row("PNL", f"[{pnl_color}]${pnl:+,.2f}[/]")
        
        if trade.entry_price > 0:
            pnl_pct = (pnl / (trade.entry_price * trade.quantity)) * 100
            table.add_row("ROI", f"[{pnl_color}]{pnl_pct:+.2f}%[/]")
        
        panel = Panel(
            Align.center(table),
            title="[bold]POSICION CERRADA[/bold]",
            title_align="center",
            border_style="dim white",
            padding=(1, 2),
        )
        
        console.print()
        console.print(panel)
        console.print()


def show_error(message: str) -> None:
    """Muestra un mensaje de error."""
    # Escapar corchetes para evitar problemas con Rich markup
    safe_message = str(message).replace("[", "\\[").replace("]", "\\]")
    console.print(f"\n[{Colors.LOSS}]Error: {safe_message}[/]\n")


def show_warning(message: str) -> None:
    """Muestra un mensaje de advertencia."""
    safe_message = str(message).replace("[", "\\[").replace("]", "\\]")
    console.print(f"\n[{Colors.WARNING}]{safe_message}[/]\n")


def show_info(message: str) -> None:
    """Muestra un mensaje informativo."""
    safe_message = str(message).replace("[", "\\[").replace("]", "\\]")
    console.print(f"\n[{Colors.ACCENT}]{safe_message}[/]\n")


def show_trade_closed_externally(trade: Any) -> None:
    """
    Muestra cuando un trade fue cerrado (manual, SL o TP).
    TODOS los datos son REALES obtenidos del exchange.
    """
    from real.config import DISPLAY_NAMES
    
    # Datos del exchange
    pnl = getattr(trade, 'pnl', 0) or 0
    trading_fees = getattr(trade, 'trading_fees', 0) or 0
    funding_fees = getattr(trade, 'funding_fees', 0) or 0
    total_fees = trading_fees + funding_fees
    total_pnl = getattr(trade, 'total_pnl', None) or (pnl + total_fees)
    quantity = getattr(trade, 'quantity', 0) or 0
    reason = getattr(trade, 'reason', 'EXCHANGE_CLOSE') or 'EXCHANGE_CLOSE'
    
    # Titulo segun motivo
    reason_map = {
        "SL": "STOP LOSS EJECUTADO",
        "TP": "TAKE PROFIT EJECUTADO",
        "EXCHANGE_CLOSE": "POSICION CERRADA",
    }
    title = reason_map.get(reason, "POSICION CERRADA")
    
    # ROI
    leverage = getattr(trade, 'leverage', 1) or 1
    margin = (quantity * trade.entry_price) / leverage if trade.entry_price > 0 else 0
    roi = (total_pnl / margin) * 100 if margin > 0 else 0
    
    display_symbol = DISPLAY_NAMES.get(trade.symbol, trade.symbol)
    
    # Tabla centrada, colores neutros
    table = Table(
        show_header=False,
        box=None,
        padding=(0, 3),
    )
    table.add_column("Label", style="dim white", justify="right", width=22)
    table.add_column("Value", style="bold white", justify="left", width=30)
    
    table.add_row("ACTIVO", display_symbol)
    table.add_row("DIRECCION", trade.side)
    table.add_row("VOLUMEN", f"{quantity:.4f}")
    table.add_row("PRECIO ENTRADA", f"${trade.entry_price:,.2f}")
    table.add_row("PRECIO SALIDA", f"${trade.exit_price:,.2f}" if trade.exit_price else "N/A")
    table.add_row("", "")
    table.add_row("PNL BRUTO", f"${pnl:+,.4f}")
    table.add_row("COMISIONES", f"${total_fees:+,.4f}")
    
    pnl_color = "#5daa68" if total_pnl >= 0 else "#c0392b"
    table.add_row("PNL NETO", f"[{pnl_color}]${total_pnl:+,.4f}[/]")
    table.add_row("ROI", f"[{pnl_color}]{roi:+.2f}%[/]")
    
    panel = Panel(
        Align.center(table),
        title=f"[bold]{title}[/]",
        title_align="center",
        border_style="dim white",
        padding=(1, 2),
    )
    
    console.print()
    console.print(panel)
    console.print()


# =============================================================================
# LIVE DISPLAY - ACTUALIZACIÓN EN TIEMPO REAL
# =============================================================================

def create_live_trading_panel(
    symbol: str,
    current_price: float,
    active_trades: List[Any],
    stats: Dict[str, Any],
    leverage: int = 1,
    last_update: Optional[datetime] = None,
) -> Panel:
    """
    Crea un panel de trading en vivo que muestra el estado actual.
    Diseñado para usarse con Rich Live para actualización cada segundo.
    """
    from rich.columns import Columns
    
    now = last_update or datetime.now()
    
    # Layout principal
    layout = Layout()
    
    # ═══════════════════════════════════════════════════════════════════
    # SECCION SUPERIOR: Precio y hora
    # ═══════════════════════════════════════════════════════════════════
    from real.config import DISPLAY_NAMES
    display_sym = DISPLAY_NAMES.get(symbol, symbol)
    
    header_text = Text()
    header_text.append(f"{display_sym}", style="bold white")
    header_text.append("  │  ", style="dim")
    header_text.append(f"${current_price:,.2f}", style="bold white")
    header_text.append("  │  ", style="dim")
    header_text.append(f"{now.strftime('%H:%M:%S')}", style="dim")
    
    # ═══════════════════════════════════════════════════════════════════
    # SECCIÓN CENTRAL: Posiciones activas con PnL en tiempo real
    # ═══════════════════════════════════════════════════════════════════
    if active_trades:
        positions_table = Table(
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold white",
            expand=True,
            padding=(0, 1),
        )

        positions_table.add_column("ACTIVO", justify="left", width=8)
        positions_table.add_column("LADO", justify="left", width=7)
        positions_table.add_column("VOLUMEN", justify="right")
        positions_table.add_column("ENTRADA", justify="right")
        positions_table.add_column("ACTUAL", justify="right")
        positions_table.add_column("PNL", justify="right")

        for trade in active_trades:
            # PnL REAL del exchange (unrealized_pnl)
            pnl_usd = getattr(trade, 'unrealized_pnl', None) or 0

            entry_price = trade.entry_price
            quantity = trade.quantity

            # Precio de mercado: usar mark_price del exchange si está disponible
            mark = getattr(trade, 'mark_price', None) or 0
            display_price = mark if mark > 0 else current_price

            # Colores según PnL
            pnl_style = "bold green" if pnl_usd > 5 else ("green" if pnl_usd >= 0 else ("bold red" if pnl_usd < -5 else "red"))

            side_icon = "LONG" if trade.side == "LONG" else "SHORT"
            asset_name = display_sym if len(active_trades) <= 1 else DISPLAY_NAMES.get(trade.symbol, trade.symbol)

            positions_table.add_row(
                asset_name,
                side_icon,
                f"{quantity:.4f}",
                f"${entry_price:,.2f}",
                f"${display_price:,.2f}",
                f"[{pnl_style}]${pnl_usd:+.2f}[/]",
            )

        positions_content = positions_table
    else:
        positions_content = Text("Sin posiciones abiertas", style="dim italic", justify="center")
    
    # ═══════════════════════════════════════════════════════════════════
    # SECCIÓN INFERIOR: Estadísticas de sesión
    # ═══════════════════════════════════════════════════════════════════
    stats_text = Text()
    
    total_trades = stats.get('total_trades', 0)
    winrate = stats.get('winrate', 0)
    total_pnl = stats.get('total_pnl', 0)
    
    if total_trades > 0:
        wr_style = "green" if winrate >= 50 else "red"
        pnl_style = "green" if total_pnl >= 0 else "red"
        
        stats_text.append(f"TRADES OPERADOS: ", style="dim")
        stats_text.append(f"{total_trades}", style="bold")
        stats_text.append(f"  │  ", style="dim")
        stats_text.append(f"WIN RATE: ", style="dim")
        stats_text.append(f"{winrate:.1f}%", style=wr_style)
        stats_text.append(f"  │  ", style="dim")
        stats_text.append(f"PNL SESION: ", style="dim")
        stats_text.append(f"${total_pnl:+,.2f}", style=pnl_style)
    else:
        stats_text.append("Esperando primer trade...", style="dim italic")
    
    # ═══════════════════════════════════════════════════════════════════
    # ENSAMBLAR PANEL FINAL
    # ═══════════════════════════════════════════════════════════════════
    content = Group(
        Align.center(header_text),
        Text("─" * 60, style="dim", justify="center"),
        positions_content,
        Text("─" * 60, style="dim", justify="center"),
        Align.center(stats_text),
    )
    
    return Panel(
        content,
        title="[bold]TRADING EN VIVO[/]",
        subtitle="[dim]Ctrl+C para detener[/]",
        box=box.DOUBLE,
        border_style="dim white",
        padding=(1, 2),
    )


class LiveTradingDisplay:
    """
    Display en vivo para trading que se actualiza cada segundo.
    Usa Rich Live para actualización fluida sin parpadeos.
    """
    
    def __init__(self, symbol: str, leverage: int = 1):
        self.symbol = symbol
        self.leverage = leverage
        self.active_trades: List[Any] = []
        self.stats: Dict[str, Any] = {}
        self.current_price: float = 0
        self.last_signal: Optional[str] = None
        self.messages: List[str] = []
        self._live: Optional[Live] = None
    
    def update(
        self,
        current_price: float,
        active_trades: List[Any],
        stats: Dict[str, Any],
    ) -> None:
        """Actualiza los datos del display."""
        self.current_price = current_price
        self.active_trades = active_trades
        self.stats = stats
    
    def add_message(self, message: str) -> None:
        """Añade un mensaje al log (máximo 5 mensajes)."""
        self.messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        if len(self.messages) > 5:
            self.messages.pop(0)
    
    def render(self) -> Panel:
        """Genera el panel de display actualizado."""
        return create_live_trading_panel(
            symbol=self.symbol,
            current_price=self.current_price,
            active_trades=self.active_trades,
            stats=self.stats,
            leverage=self.leverage,
        )
    
    def start(self) -> Live:
        """Inicia el Live display y retorna el contexto."""
        self._live = Live(
            self.render(),
            console=console,
            refresh_per_second=1,
            transient=False,
        )
        return self._live
    
    def refresh(self) -> None:
        """Actualiza el display."""
        if self._live:
            self._live.update(self.render())