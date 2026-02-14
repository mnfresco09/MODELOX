#!/usr/bin/env python3
"""
MODELOX REAL TRADING - INTERFAZ PRINCIPAL
==========================================
Script principal para ejecutar trading en vivo con BingX.

Uso:
    python real/main.py

El script te guiará para configurar:
- Estrategia a usar (por ID)
- Activo a operar
- Tipo de mercado (perpetuo/spot)
- Apalancamiento
- Saldo y gestión de riesgo
- Stop Loss y Take Profit
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

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


def select_strategy() -> tuple[int, str, type]:
    """Permite al usuario seleccionar una estrategia."""
    # Obtener estrategias disponibles
    available = list_available_strategies()
    
    if not available:
        show_error("No se encontraron estrategias disponibles.")
        sys.exit(1)
    
    show_strategies_list(available)
    
    # Crear dict para búsqueda
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
    
    # Instanciar la estrategia
    strategies = instantiate_strategies(only_id=sid)
    strategy_class = type(strategies[0])
    
    console.print(f"\n[dim white]>[/dim white] Estrategia seleccionada: [bold]{name}[/bold] (ID: {sid})\n")
    
    return sid, name, strategy_class


def configure_trading() -> TradingConfig:
    """Configuracion del trading. Solo pide el activo (por numero)."""
    from real.config import (
        ACTIVOS_DISPONIBLES, MONEDA, APALANCAMIENTO, MONTO_POR_TRADE,
        STOP_LOSS_PCT, TAKE_PROFIT_PCT, TRAILING_STOP_ACTIVADO,
        TRAILING_STOP_ACTIVACION_PCT, TRAILING_STOP_DISTANCIA_PCT,
        TIMEFRAME, MODO_EJECUCION, DIRECCION,
    )
    
    console.print("\n[bold]SELECCIONAR ACTIVO[/bold]\n")
    
    for num, info in ACTIVOS_DISPONIBLES.items():
        console.print(f"  [bold]{num}[/bold] = {info['nombre']}")
    
    console.print()
    activo_num = IntPrompt.ask("Activo", default=1)
    
    if activo_num not in ACTIVOS_DISPONIBLES:
        show_error(f"Numero {activo_num} no valido. Usando BTC.")
        activo_num = 1
    
    activo_info = ACTIVOS_DISPONIBLES[activo_num]
    symbol = activo_info["symbol"]
    activo_nombre = activo_info["nombre"]
    
    console.print(f"[dim white]>[/dim white] {activo_nombre}")
    
    # Obtener leverage maximo
    temp_client = BingXClient()
    max_leverage = temp_client.get_max_leverage(symbol)
    if max_leverage == 1:
        max_leverage = 150
    
    # Ajustar leverage si excede el maximo
    leverage = min(APALANCAMIENTO, max_leverage)
    
    # Direccion para modo manual
    initial_side = None
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
    
    return config, initial_side


def test_connection(config: TradingConfig) -> tuple[bool, float]:
    """Prueba la conexión a BingX."""
    console.print("\n[dim]Conectando a BingX...[/dim]")
    
    client = BingXClient()
    success = client.test_connection()
    balance = 0.0
    
    if success:
        # Usar el nuevo método que soporta VST y USDT
        balance = client.get_currency_balance(
            currency=config.quote_currency,
            market_type=config.market_type
        )
    
    show_connection_status(success, balance, config.quote_currency)
    return success, balance


def run_trading_loop(
    trader: RealTrader,
    strategy_name: str,
    config: TradingConfig,
    initial_side: Optional[str] = None,
):
    """
    Ejecuta el loop principal de trading.
    Usa Rich Live para actualizar el display cada 1 segundo.
    """
    
    # Obtener precio actual
    current_price = trader.client.get_current_price(trader.symbol, config.market_type)
    
    # MODO MANUAL: Abrir trade inmediatamente
    if config.execution_mode == "manual" and initial_side:
        console.print(f"\n[bold]MODO MANUAL: Abriendo {initial_side} inmediatamente...[/bold]")
        trade = trader.execute_signal(initial_side, current_price)
        if trade:
            show_trade_executed(trade, is_open=True)
            
            # Notificar Telegram
            from visual.telegram import notificar_trade_abierto
            from real.config import DISPLAY_NAMES
            sl_pct = abs((trade.sl_price - trade.entry_price) / trade.entry_price * 100)
            tp_pct = abs((trade.tp_price - trade.entry_price) / trade.entry_price * 100)
            position_value = trade.entry_price * trade.quantity
            margin_used = position_value / trade.leverage
            notificar_trade_abierto(
                activo=DISPLAY_NAMES.get(trade.symbol, trade.symbol),
                direccion=trade.side,
                precio_entrada=trade.entry_price,
                cantidad=trade.quantity,
                apalancamiento=trade.leverage,
                sl_precio=trade.sl_price,
                sl_pct=sl_pct,
                tp_precio=trade.tp_price,
                tp_pct=tp_pct,
                margen=margin_used,
                valor_posicion=position_value,
                trailing=config.trailing_stop_enabled,
                trail_act_pct=config.trailing_stop_activation_pct,
                trail_dist_pct=config.trailing_stop_distance_pct,
            )
            
            time.sleep(2)  # Dar tiempo a ver el mensaje
        else:
            # El error específico ya se muestra en execute_signal
            console.print("\nRevisa el error arriba. Ajusta los parametros e intenta de nuevo.")
            return
    else:
        console.print("\n[bold]MODO AUTO: Esperando senales de la estrategia...[/bold]")
        time.sleep(1)
    
    # Crear display en vivo
    live_display = LiveTradingDisplay(
        symbol=config.symbol,
        leverage=config.leverage,
    )
    
    last_data_fetch = 0
    data_fetch_interval = 10  # Segundos entre actualizaciones de datos/señales
    df = None
    last_signal = None
    
    try:
        with live_display.start() as live:
            while True:
                loop_start = time.time()
                
                # ═══════════════════════════════════════════════════════════
                # ACTUALIZAR POSICIONES Y PnL DEL EXCHANGE (llamada única)
                # ═══════════════════════════════════════════════════════════
                if trader.active_trades:
                    # Esta llamada obtiene precio, PnL y fees en una sola operación
                    trader.update_positions_pnl_from_exchange()
                
                # Obtener precio actual (puede venir de la posición o del ticker)
                try:
                    current_price = trader.client.get_current_price(trader.symbol, config.market_type)
                except Exception:
                    pass  # Mantener precio anterior si falla
                
                # ═══════════════════════════════════════════════════════════
                # VERIFICAR CIERRES EXTERNOS Y SL/TP
                # ═══════════════════════════════════════════════════════════
                closed = trader.check_and_update_positions()
                
                # Mostrar paneles de cierre (pausando el Live display)
                if closed:
                    # Detener el live display completamente
                    live.stop()
                    
                    # Limpiar la consola completamente
                    import os
                    os.system('clear')
                    
                    for trade in closed:
                        console.print()
                        
                        # Panel de cierre con datos REALES
                        show_trade_closed_externally(trade)
                        
                        # Guardar en Excel
                        from real.trade_logger import log_trade
                        log_trade(trade, config, strategy_name)
                        
                        # Notificar Telegram
                        from visual.telegram import notificar_trade_cerrado
                        from real.config import DISPLAY_NAMES
                        _pnl = getattr(trade, 'pnl', 0) or 0
                        _fees = (getattr(trade, 'trading_fees', 0) or 0) + (getattr(trade, 'funding_fees', 0) or 0)
                        _total_pnl = getattr(trade, 'total_pnl', None) or (_pnl + _fees)
                        _qty = getattr(trade, 'quantity', 0) or 0
                        _leverage = getattr(trade, 'leverage', 1) or 1
                        _margin = (_qty * trade.entry_price) / _leverage if trade.entry_price > 0 else 0
                        _roi = (_total_pnl / _margin) * 100 if _margin > 0 else 0
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
                    
                    # Esperar para que el usuario pueda ver
                    console.print("\n[dim]Presiona ENTER para continuar[/]")
                    try:
                        input()
                    except:
                        time.sleep(5)
                    
                    # Limpiar de nuevo antes de reiniciar
                    os.system('clear')
                    live.start()
                
                # ═══════════════════════════════════════════════════════════
                # OBTENER DATOS Y SEÑALES (cada N segundos - menos frecuente)
                # ═══════════════════════════════════════════════════════════
                now = time.time()
                if now - last_data_fetch >= data_fetch_interval:
                    last_data_fetch = now
                    
                    # Obtener datos de mercado
                    try:
                        df = trader.fetch_market_data(limit=300)
                    except Exception as e:
                        live_display.add_message(f"Error datos: {e}")
                    
                    # Generar señales en modo AUTO
                    if config.execution_mode == "auto" and df is not None and not df.is_empty():
                        try:
                            signals_df = trader.generate_signals(df)
                            last_row = signals_df.tail(1)
                            signal_long = last_row["signal_long"][0] if "signal_long" in last_row.columns else False
                            signal_short = last_row["signal_short"][0] if "signal_short" in last_row.columns else False
                            
                            # Ejecutar señales si no hay posición abierta
                            if signal_long and len(trader.active_trades) == 0:
                                trade = trader.execute_signal("LONG", current_price)
                                if trade:
                                    live_display.add_message(f"LONG abierto @ ${current_price:,.2f}")
                                    last_signal = "LONG"
                                    # Notificar Telegram
                                    from visual.telegram import notificar_trade_abierto
                                    from real.config import DISPLAY_NAMES as _DN
                                    _sl_p = abs((trade.sl_price - trade.entry_price) / trade.entry_price * 100)
                                    _tp_p = abs((trade.tp_price - trade.entry_price) / trade.entry_price * 100)
                                    _pv = trade.entry_price * trade.quantity
                                    _mu = _pv / trade.leverage
                                    notificar_trade_abierto(
                                        activo=_DN.get(trade.symbol, trade.symbol),
                                        direccion="LONG", precio_entrada=trade.entry_price,
                                        cantidad=trade.quantity, apalancamiento=trade.leverage,
                                        sl_precio=trade.sl_price, sl_pct=_sl_p,
                                        tp_precio=trade.tp_price, tp_pct=_tp_p,
                                        margen=_mu, valor_posicion=_pv,
                                        trailing=config.trailing_stop_enabled,
                                        trail_act_pct=config.trailing_stop_activation_pct,
                                        trail_dist_pct=config.trailing_stop_distance_pct,
                                    )
                            
                            elif signal_short and len(trader.active_trades) == 0:
                                trade = trader.execute_signal("SHORT", current_price)
                                if trade:
                                    live_display.add_message(f"SHORT abierto @ ${current_price:,.2f}")
                                    last_signal = "SHORT"
                                    # Notificar Telegram
                                    from visual.telegram import notificar_trade_abierto
                                    from real.config import DISPLAY_NAMES as _DN2
                                    _sl_p2 = abs((trade.sl_price - trade.entry_price) / trade.entry_price * 100)
                                    _tp_p2 = abs((trade.tp_price - trade.entry_price) / trade.entry_price * 100)
                                    _pv2 = trade.entry_price * trade.quantity
                                    _mu2 = _pv2 / trade.leverage
                                    notificar_trade_abierto(
                                        activo=_DN2.get(trade.symbol, trade.symbol),
                                        direccion="SHORT", precio_entrada=trade.entry_price,
                                        cantidad=trade.quantity, apalancamiento=trade.leverage,
                                        sl_precio=trade.sl_price, sl_pct=_sl_p2,
                                        tp_precio=trade.tp_price, tp_pct=_tp_p2,
                                        margen=_mu2, valor_posicion=_pv2,
                                        trailing=config.trailing_stop_enabled,
                                        trail_act_pct=config.trailing_stop_activation_pct,
                                        trail_dist_pct=config.trailing_stop_distance_pct,
                                    )
                        
                        except Exception as e:
                            live_display.add_message(f"Error senales: {e}")
                
                # ═══════════════════════════════════════════════════════════
                # ACTUALIZAR DISPLAY
                # ═══════════════════════════════════════════════════════════
                stats = trader.get_stats()
                live_display.update(
                    current_price=current_price,
                    active_trades=trader.active_trades,
                    stats=stats,
                )
                live_display.refresh()
                
                # Calcular tiempo restante para completar 1 segundo
                elapsed = time.time() - loop_start
                sleep_time = max(0.1, 1.0 - elapsed)  # Mínimo 0.1s, target 1s
                time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        pass  # Salir del Live context limpiamente
    
    # ═══════════════════════════════════════════════════════════════════════
    # CIERRE Y RESUMEN
    # ═══════════════════════════════════════════════════════════════════════
    console.print("\n\nTrading detenido por el usuario")
    
    # Si hay posiciones abiertas, preguntar si cerrar
    if len(trader.active_trades) > 0:
        if Confirm.ask("Cerrar posiciones abiertas?", default=True):
            for trade in trader.active_trades[:]:
                result = trader.client.close_position(trader.symbol, trade.side)
                if result.success:
                    console.print(f"Posicion {trade.side} cerrada")
    
    # Mostrar resumen final
    stats = trader.get_stats()
    console.print("\n[bold]RESUMEN FINAL[/bold]")
    console.print(f"Total trades: {stats['total_trades']}")
    console.print(f"Ganados: {stats['wins']} | Perdidos: {stats['losses']}")
    console.print(f"Winrate: {stats['winrate']:.1f}%")
    
    pnl = stats['total_pnl']
    pnl_color = "#5daa68" if pnl >= 0 else "#c0392b"
    console.print(f"PnL Total: [{pnl_color}]${pnl:+,.2f}[/{pnl_color}]")


def main():
    """Punto de entrada principal."""
    console.clear()
    
    try:
        # 1. Seleccionar estrategia
        strategy_id, strategy_name, strategy_class = select_strategy()
        
        # 2. Configurar trading (ahora retorna config e initial_side)
        config, initial_side = configure_trading()
        config.strategy_id = strategy_id
        
        # 3. Probar conexión
        success, balance = test_connection(config)
        
        if not success:
            show_error("No se pudo conectar a BingX. Verifica tus API keys.")
            sys.exit(1)
        
        # 4. Verificar saldo
        if balance < config.amount_per_trade:
            show_warning(f"Saldo insuficiente (${balance:.2f}) para el monto por trade (${config.amount_per_trade:.2f})")
            if not Confirm.ask("¿Continuar de todos modos?", default=False):
                sys.exit(0)
        
        # 5. Mostrar configuración final
        show_startup_screen(strategy_name, strategy_id, config, balance)
        
        # 5.1 Notificar Telegram
        from visual.telegram import notificar_inicio_sesion
        from real.config import DISPLAY_NAMES
        notificar_inicio_sesion(
            moneda=config.quote_currency,
            activo=DISPLAY_NAMES.get(config.symbol, config.symbol),
            estrategia=f"{strategy_name} (ID: {strategy_id})",
            apalancamiento=config.leverage,
            monto=config.amount_per_trade,
            saldo=balance,
            modo=config.execution_mode,
        )
        
        # 6. Crear trader y ejecutar
        trader = RealTrader(config, strategy_class)
        run_trading_loop(trader, strategy_name, config, initial_side)
        
    except Exception as e:
        show_error(f"Error inesperado: {e}")
        raise


if __name__ == "__main__":
    main()
