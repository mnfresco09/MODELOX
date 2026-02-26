#!/usr/bin/env python3
"""
MODELOX REAL TRADING - INTERFAZ PRINCIPAL
==========================================
Script principal para ejecutar trading en vivo con BingX.

Uso:
    python real/main.py

El script te guiará para configurar:
- Estrategia a usar (por ID)
- Activos a operar (uno o varios)
- Tipo de mercado (perpetuo/spot)
- Apalancamiento
- Saldo y gestión de riesgo
- Stop Loss y Take Profit
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional

# Añadir el directorio raíz al path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from rich.console import Console
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.panel import Panel
from rich.text import Text
from rich import box

from real.bingx_client import BingXClient
from real.config import TradingConfig, get_bingx_symbol
from real.trader import RealTrader
from real.display import (
    console,
    Colors,
    show_startup_screen,
    show_strategies_list,
    show_connection_status,
    show_trade_executed,
    show_trade_closed_externally,
    show_error,
    show_warning,
    show_info,
    create_header,
    create_balance_panel,
    create_config_panel,
    create_positions_table,
    create_history_table,
    create_stats_panel,
    create_market_panel,
    LiveTradingDisplay,
)

from modelox.strategies.registry import (
    list_available_strategies,
    instantiate_strategies,
)


# =============================================================================
# SELECCIÓN DE ESTRATEGIA
# =============================================================================

def select_strategy() -> tuple[int, str, type]:
    """Permite al usuario seleccionar una estrategia."""
    available = list_available_strategies()

    if not available:
        show_error("No se encontraron estrategias disponibles.")
        sys.exit(1)

    show_strategies_list(available)

    strat_dict = {str(sid): (sid, name) for sid, name in available}

    while True:
        strategy_id = Prompt.ask(
            "Ingresa el ID de la estrategia",
            default=str(available[0][0])
        )

        if strategy_id in strat_dict:
            sid, name = strat_dict[strategy_id]
            break
        else:
            show_error(f"ID '{strategy_id}' no válido. Elige uno de la lista.")

    strategies = instantiate_strategies(only_id=sid)
    strategy_class = type(strategies[0])

    console.print(f"\n[dim white]>[/dim white] Estrategia seleccionada: [bold]{name}[/bold] (ID: {sid})\n")

    return sid, name, strategy_class


# =============================================================================
# CONFIGURACIÓN DE ACTIVOS
# =============================================================================

def configure_trading() -> tuple[List[TradingConfig], List[Optional[str]]]:
    """
    Configuración del trading.
    Permite seleccionar uno o varios activos (separados por coma).
    Devuelve una lista de TradingConfig y una lista de initial_sides.
    """
    from real.config import (
        ACTIVOS_DISPONIBLES, MONEDA, APALANCAMIENTO, MONTO_POR_TRADE,
        STOP_LOSS_PCT, TAKE_PROFIT_PCT, TRAILING_STOP_ACTIVADO,
        TRAILING_STOP_ACTIVACION_PCT, TRAILING_STOP_DISTANCIA_PCT,
        TIMEFRAME, MODO_EJECUCION, DIRECCION,
    )

    console.print("\n[bold]SELECCIONAR ACTIVOS[/bold] (puedes elegir varios separados por coma)\n")

    for num, info in ACTIVOS_DISPONIBLES.items():
        console.print(f"  [bold]{num}[/bold] = {info['nombre']}")

    console.print()
    raw = Prompt.ask("Activos (ej: 1  o  1,2,3)", default="1")

    # Parsear números
    selected_nums: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        try:
            n = int(part)
            if n in ACTIVOS_DISPONIBLES and n not in selected_nums:
                selected_nums.append(n)
        except ValueError:
            pass

    if not selected_nums:
        show_warning("Selección inválida. Usando BTC por defecto.")
        selected_nums = [1]

    configs: List[TradingConfig] = []
    initial_sides: List[Optional[str]] = []

    temp_client = BingXClient()

    for activo_num in selected_nums:
        activo_info = ACTIVOS_DISPONIBLES[activo_num]
        symbol      = activo_info["symbol"]
        activo_nombre = activo_info["nombre"]

        console.print(f"[dim white]>[/dim white] {activo_nombre}")

        # Obtener leverage máximo para este símbolo
        max_leverage = temp_client.get_max_leverage(symbol)
        if max_leverage == 1:
            max_leverage = 150

        leverage = min(APALANCAMIENTO, max_leverage)

        # Dirección para modo manual
        initial_side: Optional[str] = None
        if MODO_EJECUCION == "manual":
            initial_side = DIRECCION

        config = TradingConfig(
            market_type="perpetual",
            symbol=symbol,
            quote_currency=MONEDA,
            leverage=leverage,
            total_balance_to_use=MONTO_POR_TRADE,
            amount_per_trade=MONTO_POR_TRADE,
            stop_loss_pct=STOP_LOSS_PCT,
            take_profit_pct=TAKE_PROFIT_PCT,
            trailing_stop_enabled=TRAILING_STOP_ACTIVADO,
            trailing_stop_activation_pct=TRAILING_STOP_ACTIVACION_PCT,
            trailing_stop_distance_pct=TRAILING_STOP_DISTANCIA_PCT,
            timeframe=TIMEFRAME,
            execution_mode=MODO_EJECUCION,
            max_leverage=max_leverage,
        )

        configs.append(config)
        initial_sides.append(initial_side)

    return configs, initial_sides


# =============================================================================
# CONEXIÓN
# =============================================================================

def test_connection(config: TradingConfig) -> tuple[bool, float]:
    """Prueba la conexión a BingX."""
    console.print("\n[dim]Conectando a BingX...[/dim]")

    client  = BingXClient()
    success = client.test_connection()
    balance = 0.0

    if success:
        balance = client.get_currency_balance(
            currency=config.quote_currency,
            market_type=config.market_type,
        )

    show_connection_status(success, balance, config.quote_currency)
    return success, balance


# =============================================================================
# STATS AGREGADAS (multi-trader)
# =============================================================================

def _aggregate_stats(traders: List[RealTrader]) -> dict:
    """Agrega estadísticas de todos los traders activos."""
    all_closed = [t for tr in traders for t in tr.closed_trades]
    total = len(all_closed)

    if total == 0:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "winrate": 0.0,
            "total_pnl": 0.0,
            "active_trades": sum(len(tr.active_trades) for tr in traders),
        }

    def _pnl(t):
        if t.total_pnl is not None:
            return t.total_pnl
        return t.pnl or 0

    wins      = sum(1 for t in all_closed if _pnl(t) > 0)
    total_pnl = sum(_pnl(t) for t in all_closed)

    return {
        "total_trades": total,
        "wins": wins,
        "losses": total - wins,
        "winrate": wins / total * 100 if total > 0 else 0.0,
        "total_pnl": total_pnl,
        "active_trades": sum(len(tr.active_trades) for tr in traders),
    }


# =============================================================================
# LOOP PRINCIPAL DE TRADING
# =============================================================================

def run_trading_loop(
    traders: List[RealTrader],
    strategy_name: str,
    initial_sides: List[Optional[str]],
):
    """
    Ejecuta el loop principal de trading con soporte multi-activo.
    Cada trader opera su propio símbolo de forma independiente.
    """
    from real.config import DISPLAY_NAMES

    # ─── MODO MANUAL: abrir trades inmediatamente ─────────────────────────────
    for trader, initial_side in zip(traders, initial_sides):
        cfg = trader.config
        if cfg.execution_mode == "manual" and initial_side:
            current_price = trader.client.get_current_price(trader.symbol, cfg.market_type)
            console.print(f"\n[bold]MODO MANUAL: Abriendo {initial_side} en {DISPLAY_NAMES.get(trader.symbol, trader.symbol)} inmediatamente...[/bold]")
            trade = trader.execute_signal(initial_side, current_price)
            if trade:
                show_trade_executed(trade, is_open=True)

                from visual.telegram import notificar_trade_abierto
                sl_pct = abs((trade.sl_price - trade.entry_price) / trade.entry_price * 100)
                tp_pct = abs((trade.tp_price - trade.entry_price) / trade.entry_price * 100)
                position_value = trade.entry_price * trade.quantity
                margin_used    = position_value / trade.leverage
                notificar_trade_abierto(
                    activo=DISPLAY_NAMES.get(trade.symbol, trade.symbol),
                    direccion=trade.side,
                    precio_entrada=trade.entry_price,
                    cantidad=trade.quantity,
                    apalancamiento=trade.leverage,
                    sl_precio=trade.sl_price, sl_pct=sl_pct,
                    tp_precio=trade.tp_price, tp_pct=tp_pct,
                    margen=margin_used, valor_posicion=position_value,
                    trailing=cfg.trailing_stop_enabled,
                    trail_act_pct=cfg.trailing_stop_activation_pct,
                    trail_dist_pct=cfg.trailing_stop_distance_pct,
                )
                time.sleep(2)
            else:
                console.print("\nRevisa el error arriba. Ajusta los parámetros e intenta de nuevo.")
                return
    else:
        if all(t.config.execution_mode != "manual" for t in traders):
            activos = ", ".join(DISPLAY_NAMES.get(t.symbol, t.symbol) for t in traders)
            console.print(f"\n[bold]MODO AUTO: Esperando señales ({activos})...[/bold]")
            time.sleep(1)

    # ─── Crear display ────────────────────────────────────────────────────────
    # Símbolo principal: si hay varios, mostrar "MULTI"
    if len(traders) == 1:
        display_symbol = traders[0].symbol
    else:
        all_names = "/".join(DISPLAY_NAMES.get(t.symbol, t.symbol) for t in traders)
        display_symbol = all_names

    live_display = LiveTradingDisplay(
        symbol=display_symbol,
        leverage=traders[0].config.leverage,
    )

    # Contadores por trader para fetch de datos/señales
    last_data_fetch   = {t.symbol: 0.0 for t in traders}
    data_fetch_interval = 10  # segundos
    dfs = {t.symbol: None for t in traders}

    # Precio actual por trader (para el display)
    current_prices = {t.symbol: 0.0 for t in traders}

    try:
        with live_display.start() as live:
            while True:
                loop_start = time.time()

                # ═══════════════════════════════════════════════════════════
                # POR CADA TRADER: actualizar PnL y verificar cierres
                # ═══════════════════════════════════════════════════════════
                all_closed: List[tuple] = []  # (trade, config)

                for trader in traders:
                    # Actualizar precio actual
                    try:
                        cp = trader.client.get_current_price(trader.symbol, trader.config.market_type)
                        current_prices[trader.symbol] = cp
                    except Exception:
                        pass

                    # Actualizar PnL desde exchange si hay trades abiertos
                    if trader.active_trades:
                        trader.update_positions_pnl_from_exchange()

                    # Verificar cierres (externos, SL, TP)
                    closed = trader.check_and_update_positions()
                    for t in closed:
                        all_closed.append((t, trader.config))

                # ═══════════════════════════════════════════════════════════
                # MOSTRAR PANELES DE CIERRE
                # ═══════════════════════════════════════════════════════════
                if all_closed:
                    live.stop()

                    import os
                    os.system('clear')

                    for trade, cfg in all_closed:
                        console.print()
                        show_trade_closed_externally(trade)

                        # Guardar en Excel
                        from real.trade_logger import log_trade
                        log_trade(trade, cfg, strategy_name)

                        # Notificar Telegram
                        from visual.telegram import notificar_trade_cerrado
                        _pnl  = getattr(trade, 'pnl', 0) or 0
                        _fees = (getattr(trade, 'trading_fees', 0) or 0) + (getattr(trade, 'funding_fees', 0) or 0)
                        _total_pnl = getattr(trade, 'total_pnl', None) or (_pnl + _fees)
                        _qty    = getattr(trade, 'quantity', 0) or 0
                        _lev    = getattr(trade, 'leverage', 1) or 1
                        _margin = (_qty * trade.entry_price) / _lev if trade.entry_price > 0 else 0
                        _roi    = (_total_pnl / _margin) * 100 if _margin > 0 else 0
                        notificar_trade_cerrado(
                            activo=DISPLAY_NAMES.get(trade.symbol, trade.symbol),
                            direccion=trade.side,
                            precio_entrada=trade.entry_price,
                            precio_salida=trade.exit_price or 0,
                            cantidad=_qty,
                            pnl_bruto=_pnl,
                            comisiones=_fees,
                            pnl_neto=_total_pnl,
                            roi=_roi,
                            razon=getattr(trade, 'reason', 'CIERRE') or 'CIERRE',
                        )
                        console.print()

                    console.print("\n[dim]Presiona ENTER para continuar[/]")
                    try:
                        input()
                    except Exception:
                        time.sleep(5)

                    os.system('clear')
                    live.start()

                # ═══════════════════════════════════════════════════════════
                # SEÑALES (cada N segundos, por trader)
                # ═══════════════════════════════════════════════════════════
                now = time.time()
                for trader in traders:
                    if now - last_data_fetch.get(trader.symbol, 0) < data_fetch_interval:
                        continue
                    last_data_fetch[trader.symbol] = now

                    # Obtener datos de mercado
                    try:
                        dfs[trader.symbol] = trader.fetch_market_data(limit=300)
                    except Exception as e:
                        live_display.add_message(f"Error datos {DISPLAY_NAMES.get(trader.symbol, trader.symbol)}: {e}")

                    # Generar señales en modo AUTO
                    cfg = trader.config
                    df  = dfs.get(trader.symbol)
                    if cfg.execution_mode != "auto" or df is None or df.is_empty():
                        continue

                    try:
                        signals_df  = trader.generate_signals(df)
                        last_row    = signals_df.tail(1)
                        signal_long  = last_row["signal_long"][0]  if "signal_long"  in last_row.columns else False
                        signal_short = last_row["signal_short"][0] if "signal_short" in last_row.columns else False

                        cp = current_prices.get(trader.symbol, 0)

                        if signal_long and len(trader.active_trades) == 0:
                            trade = trader.execute_signal("LONG", cp)
                            if trade:
                                live_display.add_message(f"LONG {DISPLAY_NAMES.get(trader.symbol, trader.symbol)} @ ${cp:,.2f}")
                                from visual.telegram import notificar_trade_abierto
                                _sl_p = abs((trade.sl_price - trade.entry_price) / trade.entry_price * 100)
                                _tp_p = abs((trade.tp_price - trade.entry_price) / trade.entry_price * 100)
                                _pv = trade.entry_price * trade.quantity
                                _mu = _pv / trade.leverage
                                notificar_trade_abierto(
                                    activo=DISPLAY_NAMES.get(trade.symbol, trade.symbol),
                                    direccion="LONG", precio_entrada=trade.entry_price,
                                    cantidad=trade.quantity, apalancamiento=trade.leverage,
                                    sl_precio=trade.sl_price, sl_pct=_sl_p,
                                    tp_precio=trade.tp_price, tp_pct=_tp_p,
                                    margen=_mu, valor_posicion=_pv,
                                    trailing=cfg.trailing_stop_enabled,
                                    trail_act_pct=cfg.trailing_stop_activation_pct,
                                    trail_dist_pct=cfg.trailing_stop_distance_pct,
                                )

                        elif signal_short and len(trader.active_trades) == 0:
                            trade = trader.execute_signal("SHORT", cp)
                            if trade:
                                live_display.add_message(f"SHORT {DISPLAY_NAMES.get(trader.symbol, trader.symbol)} @ ${cp:,.2f}")
                                from visual.telegram import notificar_trade_abierto
                                _sl_p = abs((trade.sl_price - trade.entry_price) / trade.entry_price * 100)
                                _tp_p = abs((trade.tp_price - trade.entry_price) / trade.entry_price * 100)
                                _pv = trade.entry_price * trade.quantity
                                _mu = _pv / trade.leverage
                                notificar_trade_abierto(
                                    activo=DISPLAY_NAMES.get(trade.symbol, trade.symbol),
                                    direccion="SHORT", precio_entrada=trade.entry_price,
                                    cantidad=trade.quantity, apalancamiento=trade.leverage,
                                    sl_precio=trade.sl_price, sl_pct=_sl_p,
                                    tp_precio=trade.tp_price, tp_pct=_tp_p,
                                    margen=_mu, valor_posicion=_pv,
                                    trailing=cfg.trailing_stop_enabled,
                                    trail_act_pct=cfg.trailing_stop_activation_pct,
                                    trail_dist_pct=cfg.trailing_stop_distance_pct,
                                )

                    except Exception as e:
                        live_display.add_message(f"Error señales {DISPLAY_NAMES.get(trader.symbol, trader.symbol)}: {e}")

                # ═══════════════════════════════════════════════════════════
                # ACTUALIZAR DISPLAY
                # ═══════════════════════════════════════════════════════════
                all_active = [t for tr in traders for t in tr.active_trades]
                stats = _aggregate_stats(traders)

                # Precio de referencia para el header: primer trader
                primary_price = current_prices.get(traders[0].symbol, 0)

                live_display.update(
                    current_price=primary_price,
                    active_trades=all_active,
                    stats=stats,
                )
                live_display.refresh()

                elapsed    = time.time() - loop_start
                sleep_time = max(0.1, 1.0 - elapsed)
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass

    # ─── CIERRE Y RESUMEN ─────────────────────────────────────────────────────
    console.print("\n\nTrading detenido por el usuario")

    for trader in traders:
        if len(trader.active_trades) > 0:
            asset = DISPLAY_NAMES.get(trader.symbol, trader.symbol)
            if Confirm.ask(f"¿Cerrar posiciones abiertas en {asset}?", default=True):
                for trade in trader.active_trades[:]:
                    result = trader.client.close_position(trader.symbol, trade.side)
                    if result.success:
                        console.print(f"Posición {trade.side} en {asset} cerrada")

    stats = _aggregate_stats(traders)
    console.print("\n[bold]RESUMEN FINAL[/bold]")
    console.print(f"Total trades: {stats['total_trades']}")
    console.print(f"Ganados: {stats['wins']} | Perdidos: {stats['losses']}")
    console.print(f"Winrate: {stats['winrate']:.1f}%")

    pnl = stats['total_pnl']
    pnl_color = "#5daa68" if pnl >= 0 else "#c0392b"
    console.print(f"PnL Total: [{pnl_color}]${pnl:+,.2f}[/{pnl_color}]")


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

def main():
    """Punto de entrada principal."""
    console.clear()

    try:
        # 1. Seleccionar estrategia
        strategy_id, strategy_name, strategy_class = select_strategy()

        # 2. Configurar activos (uno o varios)
        configs, initial_sides = configure_trading()
        for config in configs:
            config.strategy_id = strategy_id

        # 3. Probar conexión (usando el primero)
        success, balance = test_connection(configs[0])

        if not success:
            show_error("No se pudo conectar a BingX. Verifica tus API keys.")
            sys.exit(1)

        # 4. Verificar saldo
        min_amount = configs[0].amount_per_trade
        if balance < min_amount:
            show_warning(f"Saldo insuficiente (${balance:.2f}) para el monto por trade (${min_amount:.2f})")
            if not Confirm.ask("¿Continuar de todos modos?", default=False):
                sys.exit(0)

        # 5. Mostrar configuración
        show_startup_screen(strategy_name, strategy_id, configs[0], balance)

        # Si hay más de un activo, mostrar los adicionales
        if len(configs) > 1:
            from real.config import DISPLAY_NAMES
            extra = ", ".join(DISPLAY_NAMES.get(c.symbol, c.symbol) for c in configs[1:])
            console.print(f"[dim]Activos adicionales: {extra}[/dim]\n")

        # 5.1. Inicializar Excel de sesión (crea / sobreescribe)
        from real.trade_logger import initialize_session
        initialize_session(
            initial_balance=balance,
            leverage=configs[0].leverage,
            amount_per_trade=configs[0].amount_per_trade,
            quote_currency=configs[0].quote_currency,
        )

        # 5.2. Notificar Telegram inicio de sesión
        from visual.telegram import notificar_inicio_sesion
        from real.config import DISPLAY_NAMES
        all_assets = ", ".join(DISPLAY_NAMES.get(c.symbol, c.symbol) for c in configs)
        notificar_inicio_sesion(
            moneda=configs[0].quote_currency,
            activo=all_assets,
            estrategia=f"{strategy_name} (ID: {strategy_id})",
            apalancamiento=configs[0].leverage,
            monto=configs[0].amount_per_trade,
            saldo=balance,
            modo=configs[0].execution_mode,
        )

        # 6. Crear traders (uno por activo)
        traders = [RealTrader(config, strategy_class) for config in configs]

        # 7. Ejecutar loop
        run_trading_loop(traders, strategy_name, initial_sides)

    except Exception as e:
        show_error(f"Error inesperado: {e}")
        raise


if __name__ == "__main__":
    main()
