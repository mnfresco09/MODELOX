"""
TRADER EN VIVO - MODELOX REAL
==============================
Ejecuta estrategias en tiempo real usando BingX.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

import polars as pl

from .bingx_client import BingXClient, OrderResult
from .config import TradingConfig, get_bingx_symbol


class MockTrial:
    """
    Mock de un trial de Optuna para generar parámetros por defecto.
    Usa el valor medio de cada rango.
    """
    
    def suggest_int(self, name: str, low: int, high: int, step: int = 1, log: bool = False) -> int:
        """Retorna el valor medio del rango."""
        return (low + high) // 2
    
    def suggest_float(self, name: str, low: float, high: float, step: float = None, log: bool = False) -> float:
        """Retorna el valor medio del rango."""
        return (low + high) / 2
    
    def suggest_categorical(self, name: str, choices: list) -> Any:
        """Retorna el primer elemento de las opciones."""
        return choices[0] if choices else None


@dataclass
class TradeRecord:
    """Registro de un trade ejecutado."""
    timestamp: datetime
    symbol: str
    side: str  # "LONG" o "SHORT"
    entry_price: float
    quantity: float
    sl_price: float
    tp_price: float
    leverage: int
    order_id: str
    status: str = "OPEN"
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
    reason: Optional[str] = None
    # PnL en tiempo real desde el exchange (NO estimado)
    unrealized_pnl: Optional[float] = None
    position_id: Optional[str] = None
    # Desglose de fees (del exchange)
    trading_fees: Optional[float] = None  # Comisiones de trading
    funding_fees: Optional[float] = None  # Fees de financiación
    total_pnl: Optional[float] = None     # PnL + fees = total neto
    # Fee de apertura (para estimar durante posición abierta)
    opening_fee: Optional[float] = None
    # Timestamp de apertura en ms
    open_timestamp_ms: Optional[int] = None
    # Order ID de cierre (para consultar datos reales)
    close_order_id: Optional[str] = None
    # Precio de mercado actual (mark price del exchange)
    mark_price: Optional[float] = None


class RealTrader:
    """
    Ejecutor de trading en vivo.
    
    Usa las estrategias de MODELOX para generar señales
    y ejecuta órdenes reales en BingX.
    """
    
    def __init__(
        self,
        config: TradingConfig,
        strategy_class: Optional[Type] = None,
    ):
        self.config = config
        self.client = BingXClient()
        self.strategy_class = strategy_class
        self.strategy_instance = strategy_class() if strategy_class else None
        
        self.active_trades: List[TradeRecord] = []
        self.closed_trades: List[TradeRecord] = []
        self.is_running = False
        
        # Símbolo en formato BingX
        self.symbol = get_bingx_symbol(config.symbol)
        
        # Parámetros por defecto de la estrategia
        self.default_params = self._generate_default_params()
    
    def _generate_default_params(self) -> Dict[str, Any]:
        """
        Genera parámetros por defecto para la estrategia usando MockTrial.
        Usa los valores medios de los rangos definidos en suggest_params.
        """
        if self.strategy_instance is None:
            return {}
        
        try:
            mock_trial = MockTrial()
            params = self.strategy_instance.suggest_params(mock_trial)
            return params
        except Exception as e:
            print(f"[Warning] No se pudieron generar parámetros por defecto: {e}")
            return {}
    
    def set_strategy(self, strategy_class: Type) -> None:
        """Establece la estrategia a usar."""
        self.strategy_class = strategy_class
        self.strategy_instance = strategy_class()
        self.default_params = self._generate_default_params()
    
    def get_balance_info(self) -> Dict[str, Any]:
        """Obtiene información completa del balance actual desde el exchange."""
        market = self.config.market_type
        
        # Usar el nuevo método que obtiene info completa
        account_info = self.client.get_account_info()
        positions = self.client.get_positions(self.symbol) if market == "perpetual" else []
        
        return {
            "balance_usdt": account_info.get("available_margin", 0),
            "equity": account_info.get("equity", 0),
            "used_margin": account_info.get("used_margin", 0),
            "unrealized_pnl": account_info.get("unrealized_pnl", 0),
            "realized_pnl_today": account_info.get("realized_pnl_today", 0),
            "quote_currency": self.config.quote_currency,
            "positions": positions,
            "active_trades": len(self.active_trades),
        }
    
    def fetch_market_data(self, limit: int = 200, timeframe: Optional[str] = None) -> pl.DataFrame:
        """
        Obtiene datos de mercado actuales para generar señales.
        
        Returns:
            DataFrame de Polars con columnas OHLCV
        """
        tf = timeframe or self.config.timeframe

        klines = self.client.get_klines(
            symbol=self.symbol,
            interval=tf,
            limit=limit,
            market_type=self.config.market_type,
        )
        
        if not klines:
            return pl.DataFrame()
        
        # Convertir a DataFrame
        rows = []
        for k in klines:
            rows.append({
                "timestamp": datetime.fromtimestamp(k.get("time", 0) / 1000),
                "open": float(k.get("open", 0)),
                "high": float(k.get("high", 0)),
                "low": float(k.get("low", 0)),
                "close": float(k.get("close", 0)),
                "volume": float(k.get("volume", 0)),
            })
        
        return pl.DataFrame(rows)
    
    def generate_signals(self, df: pl.DataFrame, params: Optional[Dict] = None) -> pl.DataFrame:
        """
        Genera señales usando la estrategia configurada.
        
        Returns:
            DataFrame con columnas signal_long y signal_short
        """
        if self.strategy_instance is None:
            raise ValueError("No hay estrategia configurada. Usa set_strategy() primero.")
        
        # Usar parámetros por defecto de la estrategia
        if params is None or len(params) == 0:
            params = self.default_params.copy()
        
        return self.strategy_instance.generate_signals(df, params)
    
    def calculate_quantity(self, price: float) -> float:
        """Calcula la cantidad a operar basado en la configuración."""
        # Monto a usar por trade
        amount = self.config.amount_per_trade
        leverage = self.config.leverage if self.config.market_type == "perpetual" else 1
        
        # Cantidad = (monto * apalancamiento) / precio
        # NOTA: El monto es el MARGEN, no el valor nocional
        # Valor nocional = monto * leverage
        quantity = (amount * leverage) / price
        
        # Redondear a precisión adecuada
        if price > 10000:  # BTC
            quantity = round(quantity, 4)
        elif price > 100:  # ETH, etc
            quantity = round(quantity, 3)
        else:
            quantity = round(quantity, 2)
        
        return quantity
    
    def validate_order(self, quantity: float, price: float) -> tuple[bool, str]:
        """
        Valida si la orden puede ejecutarse.
        
        Returns:
            (is_valid, error_message)
        """
        leverage = self.config.leverage
        
        # Calcular margen requerido
        notional_value = quantity * price
        margin_required = notional_value / leverage
        
        # Obtener balance disponible usando el nuevo método que parsea correctamente
        try:
            account_info = self.client.get_account_info()
            available = account_info.get("available_margin", 0)
        except:
            available = 0
        
        if margin_required > available:
            return False, f"Margen insuficiente: necesitas ${margin_required:,.2f} pero solo tienes ${available:,.2f}"
        
        # Validar cantidad mínima (BingX requiere mínimo ~$5-10 de valor nocional)
        if notional_value < 5:
            return False, f"Valor nocional muy bajo: ${notional_value:.2f} (mínimo ~$5)"
        
        # BingX tiene límites de posición máxima según leverage
        # Ejemplo: 150x → max $300,000 nocional
        # Esta validación se hace mejor dejando que la API responda
        
        return True, ""
    
    def execute_signal(
        self,
        side: str,  # "LONG" o "SHORT"
        current_price: float,
    ) -> Optional[TradeRecord]:
        """
        Ejecuta una señal de trading.
        
        Args:
            side: "LONG" o "SHORT"
            current_price: Precio actual del activo
            
        Returns:
            TradeRecord si se ejecutó correctamente
        """
        from real.display import show_error, show_info
        
        quantity = self.calculate_quantity(current_price)
        order_side = "BUY" if side == "LONG" else "SELL"
        
        # Validar orden antes de enviar
        is_valid, error_msg = self.validate_order(quantity, current_price)
        if not is_valid:
            show_error(error_msg)
            return None
        
        # Mostrar info de la orden
        leverage = self.config.leverage if self.config.market_type == "perpetual" else 1
        notional = quantity * current_price
        margin = notional / leverage
        show_info(f"Abriendo {side}: {quantity} @ ${current_price:,.2f} (Margen: ${margin:,.2f})")
        
        # Calcular SL y TP (% sobre MARGEN → convertir a % sobre precio)
        # Formula: sl_precio = sl_margen / leverage
        sl_price_pct = self.config.stop_loss_pct / leverage  # % sobre precio
        tp_price_pct = self.config.take_profit_pct / leverage  # % sobre precio
        
        if side == "LONG":
            sl_price = current_price * (1 - sl_price_pct / 100)
            tp_price = current_price * (1 + tp_price_pct / 100)
        else:
            sl_price = current_price * (1 + sl_price_pct / 100)
            tp_price = current_price * (1 - tp_price_pct / 100)
        
        # Ejecutar orden REAL en BingX (tanto USDT como VST)
        result = self.client.place_order_with_sl_tp(
            symbol=self.symbol,
            side=order_side,
            quantity=quantity,
            entry_price=current_price,
            sl_pct=self.config.stop_loss_pct,
            tp_pct=self.config.take_profit_pct,
            leverage=self.config.leverage,
        )
        
        main_order: OrderResult = result["main_order"]
        success = main_order.success
        order_id = main_order.order_id or ""
        
        if not success:
            # Mostrar error específico
            error_msg = main_order.error_message or "Error desconocido"
            from real.display import show_error
            show_error(f"Orden rechazada: {error_msg}")
            return None
        
        if success:
            now = datetime.now()
            
            # Esperar a que se registre la orden y obtener datos REALES del exchange
            import time
            time.sleep(0.8)  # Esperar a que se registren los fills
            
            # Obtener precio de entrada REAL y fee del exchange usando el Order ID
            real_entry_price = current_price  # Fallback
            real_quantity = quantity  # Fallback
            opening_fee = 0.0
            
            if order_id:
                # Intentar obtener de fills (más preciso)
                fill_details = self.client.get_order_fill_details(self.symbol, order_id)
                if fill_details and fill_details.get("avgPrice", 0) > 0:
                    real_entry_price = fill_details["avgPrice"]
                    real_quantity = fill_details.get("filledQty", quantity)
                    opening_fee = fill_details.get("commission", 0)
                else:
                    # Fallback: obtener de order details
                    order_details = self.client.get_order_details(self.symbol, order_id)
                    if order_details:
                        avg_price = float(order_details.get("avgPrice", 0) or 0)
                        if avg_price > 0:
                            real_entry_price = avg_price
                        opening_fee = float(order_details.get("commission", 0) or 0)
                        filled_qty = float(order_details.get("executedQty", 0) or 0)
                        if filled_qty > 0:
                            real_quantity = filled_qty
            
            # Recalcular SL y TP con el precio de entrada REAL
            sl_price_pct = self.config.stop_loss_pct / self.config.leverage
            tp_price_pct = self.config.take_profit_pct / self.config.leverage
            
            if side == "LONG":
                sl_price = real_entry_price * (1 - sl_price_pct / 100)
                tp_price = real_entry_price * (1 + tp_price_pct / 100)
            else:
                sl_price = real_entry_price * (1 + sl_price_pct / 100)
                tp_price = real_entry_price * (1 - tp_price_pct / 100)
            
            trade = TradeRecord(
                timestamp=now,
                symbol=self.symbol,
                side=side,
                entry_price=real_entry_price,  # Precio REAL del exchange
                quantity=real_quantity,  # Cantidad REAL ejecutada
                sl_price=sl_price,
                tp_price=tp_price,
                leverage=self.config.leverage,
                order_id=order_id,
                opening_fee=opening_fee,  # Fee real de apertura
                open_timestamp_ms=int(now.timestamp() * 1000),
            )
            self.active_trades.append(trade)
            return trade
        
        return None

    def execute_manual_order(
        self,
        side: str,
        current_price: float,
        quantity: float,
        order_type: str = "MARKET",
    ) -> Optional[TradeRecord]:
        """
        Ejecuta una orden manual (Order Entry del dashboard).

        Actualmente soporta MARKET. Mantiene el flujo de validación y
        construcción de TradeRecord idéntico a execute_signal, pero con
        cantidad explícita enviada por usuario.
        """
        from real.display import show_error, show_info

        side = str(side).upper().strip()
        order_type = str(order_type).upper().strip()
        if side not in {"LONG", "SHORT"}:
            show_error("Side inválido. Usa LONG o SHORT.")
            return None
        if order_type != "MARKET":
            show_error("Solo se soporta order_type MARKET en esta versión.")
            return None
        if quantity <= 0:
            show_error("Cantidad inválida. Debe ser mayor que 0.")
            return None

        # Normalizar precisión por activo
        if current_price > 10000:
            quantity = round(quantity, 4)
        elif current_price > 100:
            quantity = round(quantity, 3)
        else:
            quantity = round(quantity, 2)

        order_side = "BUY" if side == "LONG" else "SELL"
        is_valid, error_msg = self.validate_order(quantity, current_price)
        if not is_valid:
            show_error(error_msg)
            return None

        leverage = self.config.leverage if self.config.market_type == "perpetual" else 1
        notional = quantity * current_price
        margin = notional / leverage
        show_info(f"Manual {side}: {quantity} @ ${current_price:,.2f} (Margen: ${margin:,.2f})")

        result = self.client.place_order_with_sl_tp(
            symbol=self.symbol,
            side=order_side,
            quantity=quantity,
            entry_price=current_price,
            sl_pct=self.config.stop_loss_pct,
            tp_pct=self.config.take_profit_pct,
            leverage=self.config.leverage,
        )

        main_order: OrderResult = result["main_order"]
        if not main_order.success:
            show_error(f"Orden manual rechazada: {main_order.error_message or 'Error desconocido'}")
            return None

        now = datetime.now()
        time.sleep(0.8)

        order_id = main_order.order_id or ""
        real_entry_price = current_price
        real_quantity = quantity
        opening_fee = 0.0

        if order_id:
            fill_details = self.client.get_order_fill_details(self.symbol, order_id)
            if fill_details and fill_details.get("avgPrice", 0) > 0:
                real_entry_price = fill_details["avgPrice"]
                real_quantity = fill_details.get("filledQty", quantity)
                opening_fee = fill_details.get("commission", 0)
            else:
                order_details = self.client.get_order_details(self.symbol, order_id)
                if order_details:
                    avg_price = float(order_details.get("avgPrice", 0) or 0)
                    if avg_price > 0:
                        real_entry_price = avg_price
                    opening_fee = float(order_details.get("commission", 0) or 0)
                    filled_qty = float(order_details.get("executedQty", 0) or 0)
                    if filled_qty > 0:
                        real_quantity = filled_qty

        sl_price_pct = self.config.stop_loss_pct / self.config.leverage
        tp_price_pct = self.config.take_profit_pct / self.config.leverage

        if side == "LONG":
            sl_price = real_entry_price * (1 - sl_price_pct / 100)
            tp_price = real_entry_price * (1 + tp_price_pct / 100)
        else:
            sl_price = real_entry_price * (1 + sl_price_pct / 100)
            tp_price = real_entry_price * (1 - tp_price_pct / 100)

        trade = TradeRecord(
            timestamp=now,
            symbol=self.symbol,
            side=side,
            entry_price=real_entry_price,
            quantity=real_quantity,
            sl_price=sl_price,
            tp_price=tp_price,
            leverage=self.config.leverage,
            order_id=order_id,
            opening_fee=opening_fee,
            open_timestamp_ms=int(now.timestamp() * 1000),
        )
        self.active_trades.append(trade)
        return trade
    
    def update_positions_pnl_from_exchange(self) -> None:
        """
        Actualiza el PnL REAL y datos de las posiciones activas desde el exchange.
        Obtiene: unrealizedProfit REAL, precio de entrada REAL, cantidad REAL.
        Estima fees basado en el fee de apertura x2.
        """
        if not self.active_trades:
            return
        
        # Obtener posiciones reales del exchange con todos los detalles
        exchange_positions = self.client.get_positions(self.symbol)
        
        # Crear mapa de datos reales por lado
        position_by_side = {}
        for pos in exchange_positions:
            pos_amt = float(pos.get("positionAmt", 0) or pos.get("availableAmt", 0) or 0)
            pos_side = pos.get("positionSide", "")
            
            if abs(pos_amt) > 0:
                position_by_side[pos_side] = {
                    "unrealized_pnl": float(pos.get("unrealizedProfit", 0) or 0),
                    "position_id": str(pos.get("positionId", "")),
                    "avg_price": float(pos.get("avgPrice", 0) or 0),
                    "mark_price": float(pos.get("markPrice", 0) or pos.get("lastPrice", 0) or 0),
                    "position_amt": abs(pos_amt),
                    "margin": float(pos.get("margin", 0) or pos.get("initialMargin", 0) or 0),
                    "liquidation_price": float(pos.get("liquidationPrice", 0) or 0),
                }
        
        # Actualizar trades activos con datos REALES del exchange
        for trade in self.active_trades:
            if trade.side in position_by_side:
                pos_data = position_by_side[trade.side]
                
                # Actualizar PnL no realizado REAL
                trade.unrealized_pnl = pos_data["unrealized_pnl"]
                trade.mark_price = pos_data.get("mark_price", 0) or 0
                trade.position_id = pos_data["position_id"]
                
                # ACTUALIZAR precio de entrada con el REAL del exchange si difiere
                real_entry = pos_data["avg_price"]
                if real_entry > 0 and abs(real_entry - trade.entry_price) > 0.01:
                    trade.entry_price = real_entry
                
                # ACTUALIZAR cantidad con la REAL del exchange
                real_qty = pos_data["position_amt"]
                if real_qty > 0 and abs(real_qty - trade.quantity) > 0.0001:
                    trade.quantity = real_qty
                
                # Estimar fees (apertura x2 para incluir cierre futuro)
                if trade.opening_fee:
                    trade.trading_fees = trade.opening_fee * 2
                else:
                    # Fallback: ~0.1% del nocional (0.05% x2)
                    notional = trade.quantity * trade.entry_price
                    trade.trading_fees = -notional * 0.001
                
                # Total ESTIMADO = PnL no realizado + fees estimados
                trade.total_pnl = trade.unrealized_pnl + trade.trading_fees
    
    def sync_positions_with_exchange(self) -> List[TradeRecord]:
        """
        Sincroniza las posiciones locales con el estado real del exchange.
        Detecta posiciones cerradas manualmente en el exchange.
        Obtiene PnL REAL y fees del exchange.
        
        Returns:
            Lista de trades que fueron cerrados en el exchange
        """
        closed_externally = []
        
        # Obtener posiciones reales del exchange
        exchange_positions = self.client.get_positions(self.symbol)
        
        # Crear un diccionario de posiciones activas en el exchange
        # Clave: positionSide (LONG/SHORT), Valor: datos
        active_positions = {}
        for pos in exchange_positions:
            pos_amt = float(pos.get("positionAmt", 0) or pos.get("availableAmt", 0) or 0)
            pos_side = pos.get("positionSide", "")
            pos_id = str(pos.get("positionId", ""))
            
            if abs(pos_amt) > 0:
                active_positions[pos_side] = {
                    "amount": pos_amt,
                    "position_id": pos_id,
                    "unrealized_pnl": float(pos.get("unrealizedProfit", 0) or 0),
                    "avg_price": float(pos.get("avgPrice", 0) or 0),
                }
        
        # Verificar cada trade local
        for trade in self.active_trades[:]:
            trade_side = trade.side  # "LONG" o "SHORT"
            
            # Si la posición ya no existe en el exchange, fue cerrada
            position_still_open = trade_side in active_positions
            
            if not position_still_open:
                # La posición fue cerrada en el exchange (manual, SL o TP)
                trade.status = "CLOSED"
                trade.exit_time = datetime.now()
                trade.reason = "EXCHANGE_CLOSE"
                
                # Esperar a que el exchange registre el cierre en positionHistory
                import time
                time.sleep(2.0)
                
                # Consultar datos REALES con reintentos
                # A veces el exchange tarda en registrar positionHistory
                close_data = None
                for attempt in range(3):
                    close_data = self.client.get_real_close_data(
                        symbol=self.symbol,
                        position_side=trade.side,
                        open_order_id=trade.order_id,
                        open_timestamp_ms=trade.open_timestamp_ms,
                    )
                    # Si obtuvimos datos válidos (al menos entry_price), salir
                    if close_data.get("entry_price", 0) > 0:
                        break
                    time.sleep(1.5)  # Esperar más e intentar de nuevo
                
                # Datos REALES del exchange
                trade.pnl = close_data.get("realized_pnl", 0)
                trade.trading_fees = close_data.get("trading_fees", 0)
                trade.funding_fees = close_data.get("funding_fees", 0)
                trade.total_pnl = close_data.get("total_pnl", 0)
                trade.close_order_id = close_data.get("close_order_id")
                
                # Precios REALES (del order fill, incluye slippage)
                if close_data.get("entry_price", 0) > 0:
                    trade.entry_price = close_data["entry_price"]
                if close_data.get("exit_price", 0) > 0:
                    trade.exit_price = close_data["exit_price"]
                else:
                    trade.exit_price = self.client.get_current_price(self.symbol, self.config.market_type)
                if close_data.get("volume", 0) > 0:
                    trade.quantity = close_data["volume"]
                
                self.active_trades.remove(trade)
                self.closed_trades.append(trade)
                closed_externally.append(trade)
        
        return closed_externally
    
    def check_and_update_positions(self) -> List[TradeRecord]:
        """
        Verifica el estado de las posiciones activas.
        Primero sincroniza con el exchange para detectar cierres manuales,
        luego verifica SL/TP localmente.
        
        Returns:
            Lista de trades que se cerraron
        """
        closed = []
        
        # PRIMERO: Sincronizar con el exchange para detectar cierres manuales
        manual_closed = self.sync_positions_with_exchange()
        closed.extend(manual_closed)
        
        # SEGUNDO: Verificar SL/TP para trades que siguen activos
        # NOTA: Las órdenes SL/TP ya están en el exchange (se envían con la orden principal).
        # Si BingX ejecuta un SL/TP, la posición desaparece y sync_positions_with_exchange
        # ya lo detecta arriba. Esta verificación local es un FALLBACK de seguridad.
        # Cuando se detecta aquí, también consultamos datos REALES del exchange.
        current_price = self.client.get_current_price(self.symbol, self.config.market_type)
        
        for trade in self.active_trades[:]:  # Copiar lista para modificar
            # Verificar SL/TP
            hit_sl = False
            hit_tp = False
            
            if trade.side == "LONG":
                hit_sl = current_price <= trade.sl_price
                hit_tp = current_price >= trade.tp_price
            else:
                hit_sl = current_price >= trade.sl_price
                hit_tp = current_price <= trade.tp_price
            
            if hit_sl or hit_tp:
                trade.status = "CLOSED"
                trade.exit_time = datetime.now()
                trade.reason = "SL" if hit_sl else "TP"
                
                # Esperar a que el exchange registre la ejecución
                import time
                time.sleep(2.0)
                
                # Consultar datos REALES del exchange con reintentos
                close_data = None
                for attempt in range(3):
                    close_data = self.client.get_real_close_data(
                        symbol=self.symbol,
                        position_side=trade.side,
                        open_order_id=trade.order_id,
                        open_timestamp_ms=trade.open_timestamp_ms,
                    )
                    if close_data.get("entry_price", 0) > 0:
                        break
                    time.sleep(1.5)
                
                # Datos REALES del exchange
                trade.pnl = close_data.get("realized_pnl", 0)
                trade.trading_fees = close_data.get("trading_fees", 0)
                trade.funding_fees = close_data.get("funding_fees", 0)
                trade.total_pnl = close_data.get("total_pnl", 0)
                trade.close_order_id = close_data.get("close_order_id")
                
                # Precios REALES (del order fill, incluye slippage)
                if close_data.get("exit_price", 0) > 0:
                    trade.exit_price = close_data["exit_price"]
                else:
                    trade.exit_price = current_price  # Fallback
                if close_data.get("entry_price", 0) > 0:
                    trade.entry_price = close_data["entry_price"]
                if close_data.get("volume", 0) > 0:
                    trade.quantity = close_data["volume"]
                
                self.active_trades.remove(trade)
                self.closed_trades.append(trade)
                closed.append(trade)
        
        return closed
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de la sesión de trading."""
        total_trades = len(self.closed_trades)
        if total_trades == 0:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "winrate": 0.0,
                "total_pnl": 0.0,
                "active_trades": len(self.active_trades),
            }
        
        # Usar total_pnl (neto con fees) si está disponible, sino pnl
        def get_pnl(t):
            if t.total_pnl is not None:
                return t.total_pnl
            return t.pnl or 0
        
        wins = sum(1 for t in self.closed_trades if get_pnl(t) > 0)
        losses = total_trades - wins
        total_pnl = sum(get_pnl(t) for t in self.closed_trades)
        
        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "winrate": wins / total_trades * 100 if total_trades > 0 else 0.0,
            "total_pnl": total_pnl,
            "active_trades": len(self.active_trades),
        }
