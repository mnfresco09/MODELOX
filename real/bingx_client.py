"""
CLIENTE API BINGX - TRADING EN VIVO
====================================
Cliente completo para interactuar con la API de BingX.
Soporta futuros perpetuos (USDT-M) y futuros estándar.

IMPORTANTE: Para trading demo (VST) usar dominio: https://open-api-vst.bingx.com
            Para trading real (USDT) usar dominio: https://open-api.bingx.com
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlencode

import requests

from .config import (
    BINGX_API_KEY,
    BINGX_SECRET_KEY,
)

# Dominios de BingX
BINGX_BASE_URL_LIVE = "https://open-api.bingx.com"
BINGX_BASE_URL_DEMO = "https://open-api-vst.bingx.com"


@dataclass
class OrderResult:
    """Resultado de una orden ejecutada."""
    success: bool
    order_id: Optional[str] = None
    symbol: str = ""
    side: str = ""
    quantity: float = 0.0
    price: float = 0.0
    status: str = ""
    error_message: Optional[str] = None
    raw_response: Optional[Dict] = None


class BingXClient:
    """
    Cliente para la API de BingX.
    
    Soporta:
    - Futuros Perpetuos (Swap USDT-M) - Sin vencimiento
    - Futuros Estándar (Delivery) - Con vencimiento
    
    IMPORTANTE: 
    - Para trading demo (VST): use_demo=True
    - Para trading real (USDT): use_demo=False
    
    Ejemplo de uso:
        client = BingXClient(use_demo=True)  # Para VST
        balance = client.get_balance()
        order = client.place_order("BTC-USDT", "BUY", 0.001, leverage=10)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        use_demo: bool = True,  # Por defecto usar demo/VST
    ):
        self.api_key = api_key or BINGX_API_KEY
        self.secret_key = secret_key or BINGX_SECRET_KEY
        self.use_demo = use_demo
        self.base_url = BINGX_BASE_URL_DEMO if use_demo else BINGX_BASE_URL_LIVE
        self.session = requests.Session()
        self.session.headers.update({
            "X-BX-APIKEY": self.api_key,
        })
    
    # =========================================================================
    # UTILIDADES DE FIRMA - FORMATO OFICIAL BINGX
    # =========================================================================
    
    def _generate_signature(self, params_str: str) -> str:
        """Genera la firma HMAC-SHA256 según formato oficial de BingX."""
        signature = hmac.new(
            self.secret_key.encode("utf-8"),
            params_str.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _parse_params(self, params: Dict[str, Any]) -> str:
        """Convierte parámetros a string ordenado + timestamp (formato oficial)."""
        sorted_keys = sorted(params.keys())
        params_str = "&".join([f"{k}={params[k]}" for k in sorted_keys])
        timestamp = str(int(time.time() * 1000))
        if params_str:
            return f"{params_str}&timestamp={timestamp}"
        return f"timestamp={timestamp}"
    
    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        signed: bool = True,
    ) -> Dict[str, Any]:
        """
        Ejecuta una request a la API siguiendo el formato oficial de BingX.
        
        Los parámetros se envían en la URL query string (no en el body),
        con la firma añadida al final.
        """
        params = params or {}
        
        if signed:
            # Formato oficial: params ordenados + timestamp + signature
            params_str = self._parse_params(params)
            signature = self._generate_signature(params_str)
            url = f"{self.base_url}{endpoint}?{params_str}&signature={signature}"
        else:
            params_str = self._parse_params(params) if params else ""
            url = f"{self.base_url}{endpoint}"
            if params_str:
                url = f"{url}?{params_str}"
        
        headers = {"X-BX-APIKEY": self.api_key}
        
        try:
            response = requests.request(method.upper(), url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            return {"code": -1, "msg": str(e), "data": None}
    
    # =========================================================================
    # INFORMACIÓN DE CUENTA
    # =========================================================================
    
    def get_balance(
        self, 
        market_type: Literal["perpetual", "standard"] = "perpetual",
        currency: Literal["USDT", "VST"] = "USDT"
    ) -> Dict[str, Any]:
        """
        Obtiene el balance de la cuenta.
        
        Args:
            market_type: "perpetual" o "standard" (ambos son futuros)
            currency: "USDT" para dinero real, "VST" para cuenta demo
            
        Returns:
            Dict con información del balance incluyendo:
            - balance: Balance total
            - equity: Capital (balance + PnL no realizado)
            - availableMargin: Margen disponible
            - usedMargin: Margen usado
            - unrealizedProfit: PnL no realizado
        """
        # Detectar si es cuenta demo basado en el dominio configurado
        if self.use_demo or market_type == "standard":
            # Futuros estándar / cuenta demo usa endpoint contract
            endpoint = "/openApi/contract/v1/balance"
        else:
            # Usar v3 para obtener información más completa (USDT y USDC)
            endpoint = "/openApi/swap/v3/user/balance"
        return self._request("GET", endpoint)
    
    def get_currency_balance(
        self, 
        currency: Literal["USDT", "VST"] = "USDT",
        market_type: Literal["perpetual", "standard"] = "perpetual"
    ) -> float:
        """
        Obtiene el balance disponible de una moneda específica.
        
        Args:
            currency: "USDT" para dinero real, "VST" para cuenta demo
            market_type: "perpetual" o "standard"
            
        Returns:
            Balance disponible como float
        """
        result = self.get_balance(market_type, currency)
        
        if result.get("code") != 0:
            return 0.0
        
        data = result.get("data", {})
        
        # Para VST o futuros estándar, el endpoint /contract/v1/balance devuelve lista
        if isinstance(data, list):
            for asset_info in data:
                if asset_info.get("asset") == currency:
                    # crossWalletBalance es el saldo disponible real
                    val = asset_info.get("crossWalletBalance") or asset_info.get("availableBalance") or asset_info.get("balance")
                    try:
                        return float(val) if val else 0.0
                    except (ValueError, TypeError):
                        return 0.0
            return 0.0
        
        # Para swap perpetuo
        balance_info = data.get("balance", {})
        if isinstance(balance_info, dict):
            for key in ["availableMargin", "equity", "balance"]:
                val = balance_info.get(key)
                if val is not None:
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        continue
        elif isinstance(balance_info, list):
            for b in balance_info:
                if b.get("asset") == currency:
                    return float(b.get("availableMargin", 0))
        
        return 0.0
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Obtiene información detallada de la cuenta incluyendo:
        - Balance total
        - Equity (capital con PnL)
        - Margen disponible/usado
        - PnL no realizado total
        
        Returns:
            Dict con toda la información de la cuenta
        """
        result = self.get_balance()
        
        default_result = {
            "balance": 0,
            "equity": 0,
            "available_margin": 0,
            "used_margin": 0,
            "unrealized_pnl": 0,
            "realized_pnl_today": 0,
        }
        
        if result.get("code") != 0:
            return default_result
        
        data = result.get("data", {})
        
        # Para demo (VST), data es una lista de balances por moneda
        if isinstance(data, list):
            # Buscar VST primero (demo), luego USDT
            for target_asset in ["VST", "USDT"]:
                for asset_info in data:
                    if asset_info.get("asset") == target_asset:
                        # El endpoint contract/v1/balance tiene campos diferentes
                        # Priorizar availableBalance sobre availableMargin para demo
                        available = (
                            float(asset_info.get("availableBalance", 0) or 0) or
                            float(asset_info.get("availableMargin", 0) or 0) or
                            float(asset_info.get("crossWalletBalance", 0) or 0)
                        )
                        return {
                            "balance": float(asset_info.get("balance", 0) or asset_info.get("crossWalletBalance", 0) or 0),
                            "equity": float(asset_info.get("equity", 0) or available or 0),
                            "available_margin": available,
                            "used_margin": float(asset_info.get("usedMargin", 0) or asset_info.get("initialMargin", 0) or 0),
                            "unrealized_pnl": float(asset_info.get("unrealizedProfit", 0) or asset_info.get("crossUnPnl", 0) or 0),
                            "realized_pnl_today": float(asset_info.get("realisedProfit", 0) or 0),
                        }
            return default_result
        
        # Para v3 (trading real), data tiene estructura diferente
        balance_info = data.get("balance", data)
        if isinstance(balance_info, dict):
            return {
                "balance": float(balance_info.get("balance", 0) or 0),
                "equity": float(balance_info.get("equity", 0) or 0),
                "available_margin": float(balance_info.get("availableMargin", 0) or 0),
                "used_margin": float(balance_info.get("usedMargin", 0) or 0),
                "unrealized_pnl": float(balance_info.get("unrealizedProfit", 0) or 0),
                "realized_pnl_today": float(balance_info.get("realisedProfit", 0) or 0),
            }
        
        # Para v3 que devuelve lista de assets
        if isinstance(balance_info, list):
            for asset in balance_info:
                if asset.get("asset") in ["USDT", "VST"]:
                    return {
                        "balance": float(asset.get("balance", 0) or 0),
                        "equity": float(asset.get("equity", 0) or 0),
                        "available_margin": float(asset.get("availableMargin", 0) or 0),
                        "used_margin": float(asset.get("usedMargin", 0) or 0),
                        "unrealized_pnl": float(asset.get("unrealizedProfit", 0) or 0),
                        "realized_pnl_today": float(asset.get("realisedProfit", 0) or 0),
                    }
        
        return default_result
    
    def get_leverage_info(self, symbol: str) -> Dict[str, Any]:
        """
        Obtiene información de leverage para un símbolo.
        
        Args:
            symbol: Símbolo del activo (ej: "BTC-USDT")
            
        Returns:
            Dict con: maxLongLeverage, maxShortLeverage, longLeverage, shortLeverage
        """
        endpoint = "/openApi/swap/v2/trade/leverage"
        result = self._request("GET", endpoint, {"symbol": symbol})
        
        if result.get("code") != 0:
            return {"error": result.get("msg", "Unknown error")}
        
        return result.get("data", {})
    
    def get_max_leverage(self, symbol: str) -> int:
        """
        Obtiene el leverage máximo permitido para un símbolo.
        
        Args:
            symbol: Símbolo del activo (ej: "BTC-USDT")
            
        Returns:
            Leverage máximo como int (default 1 si hay error)
        """
        info = self.get_leverage_info(symbol)
        
        if "error" in info:
            return 1
        
        # Tomar el menor entre long y short para ser conservador
        max_long = info.get("maxLongLeverage", 1)
        max_short = info.get("maxShortLeverage", 1)
        
        return min(max_long, max_short)
    
    def get_current_leverage(self, symbol: str) -> tuple[int, int]:
        """
        Obtiene el leverage actualmente configurado para un símbolo.
        
        Args:
            symbol: Símbolo del activo
            
        Returns:
            Tuple (long_leverage, short_leverage)
        """
        info = self.get_leverage_info(symbol)
        
        if "error" in info:
            return (1, 1)
        
        return (
            info.get("longLeverage", 1),
            info.get("shortLeverage", 1)
        )
    
    def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Obtiene las posiciones abiertas (solo perpetuos).
        
        Args:
            symbol: Símbolo específico o None para todas
            
        Returns:
            Lista de posiciones abiertas
        """
        endpoint = "/openApi/swap/v2/user/positions"
        params = {}
        if symbol:
            params["symbol"] = symbol
        
        result = self._request("GET", endpoint, params)
        
        if result.get("code") != 0:
            return []
        
        return result.get("data") or []
    
    # =========================================================================
    # INFORMACIÓN DE MERCADO
    # =========================================================================
    
    def get_ticker(self, symbol: str, market_type: Literal["perpetual", "standard"] = "perpetual") -> Dict:
        """Obtiene el precio actual de un símbolo."""
        # Ambos tipos de futuros usan el mismo endpoint
        endpoint = "/openApi/swap/v2/quote/ticker"
        result = self._request("GET", endpoint, {"symbol": symbol}, signed=False)
        return result.get("data", {})
    
    def get_current_price(self, symbol: str, market_type: Literal["perpetual", "standard"] = "perpetual") -> float:
        """Obtiene solo el precio actual."""
        ticker = self.get_ticker(symbol, market_type)
        return float(ticker.get("lastPrice", 0) or ticker.get("price", 0) or 0)
    
    def get_klines(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 100,
        market_type: Literal["perpetual", "standard"] = "perpetual",
    ) -> List[Dict]:
        """
        Obtiene datos de velas (OHLCV).
        
        Args:
            symbol: Símbolo (ej: "BTC-USDT")
            interval: Timeframe ("1m", "5m", "15m", "1h", "4h", "1d")
            limit: Número de velas (max 1000)
            market_type: Tipo de futuros (perpetual o standard)
            
        Returns:
            Lista de velas con open, high, low, close, volume
        """
        # Ambos tipos de futuros usan el mismo endpoint
        endpoint = "/openApi/swap/v2/quote/klines"
        
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        
        result = self._request("GET", endpoint, params, signed=False)
        return result.get("data") or []
    
    # =========================================================================
    # CONFIGURACIÓN DE APALANCAMIENTO
    # =========================================================================
    
    def set_leverage(self, symbol: str, leverage: int, side: str = "BOTH") -> Dict:
        """
        Configura el apalancamiento para un símbolo (solo perpetuos).
        
        Args:
            symbol: Símbolo (ej: "BTC-USDT")
            leverage: Nivel de apalancamiento (1-125)
            side: "LONG", "SHORT" o "BOTH"
        """
        endpoint = "/openApi/swap/v2/trade/leverage"
        params = {
            "symbol": symbol,
            "leverage": leverage,
            "side": side,
        }
        return self._request("POST", endpoint, params)
    
    def set_margin_mode(self, symbol: str, margin_mode: Literal["ISOLATED", "CROSSED"] = "ISOLATED") -> Dict:
        """Configura el modo de margen (aislado o cruzado)."""
        endpoint = "/openApi/swap/v2/trade/marginType"
        params = {
            "symbol": symbol,
            "marginType": margin_mode,
        }
        return self._request("POST", endpoint, params)
    
    # =========================================================================
    # ÓRDENES
    # =========================================================================
    
    def place_order(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: float,
        order_type: Literal["MARKET", "LIMIT"] = "MARKET",
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        leverage: Optional[int] = None,
        market_type: Literal["perpetual", "standard"] = "perpetual",
        position_side: Literal["LONG", "SHORT"] = "LONG",
    ) -> OrderResult:
        """
        Coloca una orden de mercado o límite.
        
        Args:
            symbol: Símbolo (ej: "BTC-USDT")
            side: "BUY" o "SELL"
            quantity: Cantidad a operar
            order_type: "MARKET" o "LIMIT"
            price: Precio límite (solo para órdenes límite)
            stop_loss: Precio de stop loss
            take_profit: Precio de take profit
            leverage: Apalancamiento
            market_type: "perpetual" o "standard"
            position_side: "LONG" o "SHORT"
            
        Returns:
            OrderResult con el resultado de la orden
        """
        # Configurar apalancamiento para ambos lados
        if leverage:
            self.set_leverage(symbol, leverage, "LONG")
            self.set_leverage(symbol, leverage, "SHORT")
        
        # Futuros perpetuos
        endpoint = "/openApi/swap/v2/trade/order"
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
            "positionSide": position_side,
        }
        
        if order_type == "LIMIT" and price:
            params["price"] = price
        
        if stop_loss:
            params["stopLoss"] = json.dumps({"type": "STOP_MARKET", "stopPrice": stop_loss})
        
        if take_profit:
            params["takeProfit"] = json.dumps({"type": "TAKE_PROFIT_MARKET", "stopPrice": take_profit})
        
        result = self._request("POST", endpoint, params)
        
        if result.get("code") == 0:
            data = result.get("data", {})
            order_data = data.get("order", data)  # La respuesta tiene "order" anidado
            return OrderResult(
                success=True,
                order_id=str(order_data.get("orderId", order_data.get("orderID", ""))),
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=float(order_data.get("avgPrice", 0) or order_data.get("price", 0) or 0),
                status=order_data.get("status", "NEW"),
                raw_response=result,
            )
        else:
            return OrderResult(
                success=False,
                error_message=result.get("msg", "Error desconocido"),
                raw_response=result,
            )
    
    def place_market_order(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: float,
        **kwargs,
    ) -> OrderResult:
        """Shortcut para orden de mercado."""
        return self.place_order(symbol, side, quantity, order_type="MARKET", **kwargs)
    
    def close_position(
        self,
        symbol: str,
        position_side: Literal["LONG", "SHORT"] = "LONG",
    ) -> OrderResult:
        """
        Cierra una posición abierta.
        
        Args:
            symbol: Símbolo
            position_side: "LONG" o "SHORT"
        """
        # Obtener la posición actual
        positions = self.get_positions(symbol)
        
        for pos in positions:
            if pos.get("positionSide") == position_side:
                qty = abs(float(pos.get("positionAmt", 0)))
                if qty > 0:
                    # Cerrar = orden contraria
                    close_side = "SELL" if position_side == "LONG" else "BUY"
                    return self.place_market_order(
                        symbol, close_side, qty, position_side=position_side
                    )
        
        return OrderResult(success=False, error_message="No hay posición abierta")
    
    def cancel_order(self, symbol: str, order_id: str, market_type: Literal["perpetual", "standard"] = "perpetual") -> Dict:
        """Cancela una orden pendiente."""
        endpoint = "/openApi/swap/v2/trade/order"
        params = {
            "symbol": symbol,
            "orderId": order_id,
        }
        return self._request("DELETE", endpoint, params)
    
    def get_open_orders(self, symbol: Optional[str] = None, market_type: Literal["perpetual", "standard"] = "perpetual") -> List[Dict]:
        """Obtiene las órdenes abiertas."""
        endpoint = "/openApi/swap/v2/trade/openOrders"
        
        params = {}
        if symbol:
            params["symbol"] = symbol
        
        result = self._request("GET", endpoint, params)
        return result.get("data", {}).get("orders", [])
    
    # =========================================================================
    # ÓRDENES CON SL/TP
    # =========================================================================
    
    def place_order_with_sl_tp(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: float,
        entry_price: float,
        sl_pct: float,
        tp_pct: float,
        leverage: int = 10,
    ) -> Dict[str, Any]:
        """
        Coloca una orden con Stop Loss y Take Profit calculados automáticamente.
        
        Args:
            symbol: Símbolo
            side: "BUY" (long) o "SELL" (short)
            quantity: Cantidad
            entry_price: Precio de entrada estimado
            sl_pct: Stop Loss en porcentaje (ej: 2.0 para -2%)
            tp_pct: Take Profit en porcentaje (ej: 4.0 para +4%)
            leverage: Apalancamiento
            
        Returns:
            Dict con la orden principal y las órdenes de SL/TP
        """
        # Determinar position_side basado en el side
        position_side = "LONG" if side == "BUY" else "SHORT"
        
        # Calcular precios de SL y TP
        if side == "BUY":
            # Long: SL abajo, TP arriba
            sl_price = entry_price * (1 - sl_pct / 100)
            tp_price = entry_price * (1 + tp_pct / 100)
        else:
            # Short: SL arriba, TP abajo
            sl_price = entry_price * (1 + sl_pct / 100)
            tp_price = entry_price * (1 - tp_pct / 100)
        
        # Ejecutar orden principal
        main_order = self.place_market_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            leverage=leverage,
            stop_loss=sl_price,
            take_profit=tp_price,
            position_side=position_side,
        )
        
        return {
            "main_order": main_order,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "entry_price": entry_price,
        }
    
    # =========================================================================
    # VERIFICACIÓN DE CONEXIÓN
    # =========================================================================
    
    def test_connection(self) -> bool:
        """Verifica si la conexión a la API funciona."""
        try:
            result = self.get_balance()
            return result.get("code") == 0
        except Exception:
            return False
    
    def get_server_time(self) -> int:
        """Obtiene el tiempo del servidor."""
        result = self._request("GET", "/openApi/swap/v2/server/time", signed=False)
        return result.get("data", {}).get("serverTime", 0)    
    # =========================================================================
    # HISTORIAL DE TRADES (para PnL real)
    # =========================================================================
    
    def get_position_history(
        self,
        symbol: str,
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
        limit: int = 20,
    ) -> List[Dict]:
        """
        Obtiene el historial de posiciones cerradas con PnL real.
        
        Args:
            symbol: Símbolo del par
            start_time_ms: Timestamp de inicio en ms
            end_time_ms: Timestamp de fin en ms
            limit: Número de registros
            
        Returns:
            Lista de posiciones históricas con:
            - positionId
            - symbol
            - positionSide (LONG/SHORT)
            - openAvgPrice
            - closeAvgPrice
            - closedVolume
            - pnl (PnL realizado)
            - tradeFee (comisiones)
            - fundingFee
            - openTime
            - closeTime
        """
        import time
        
        endpoint = "/openApi/swap/v1/trade/positionHistory"
        
        # Si no se pasan tiempos, usar últimos 7 días
        if end_time_ms is None:
            end_time_ms = int(time.time() * 1000)
        if start_time_ms is None:
            start_time_ms = end_time_ms - (7 * 24 * 60 * 60 * 1000)  # 7 días atrás
        
        params = {
            "symbol": symbol,
            "startTs": start_time_ms,
            "endTs": end_time_ms,
            "pageSize": limit,
            "pageIndex": 1,
        }
        
        result = self._request("GET", endpoint, params)
        
        if result.get("code") != 0:
            return []
        
        data = result.get("data") or {}
        return data.get("positionHistory") or data.get("list") or []
    
    def get_trade_income(
        self,
        symbol: str,
        income_type: Optional[str] = None,
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """
        Obtiene el flujo de fondos/income de la cuenta.
        
        Args:
            symbol: Símbolo del par
            income_type: Tipo de income (None=todos, o específico):
                - REALIZED_PNL: PnL realizado al cerrar posición
                - TRADING_FEE: Comisiones de trading
                - FUNDING_FEE: Fees de financiación
                - INSURANCE_CLEAR: Liquidación
                - TRIAL_FUND: Fondo de prueba
            start_time_ms: Timestamp de inicio
            end_time_ms: Timestamp de fin
            limit: Número máximo de registros
            
        Returns:
            Lista de income entries con:
            - symbol
            - incomeType
            - income (cantidad, negativo para fees)
            - asset (USDT/USDC)
            - time (timestamp ms)
            - info (detalles adicionales)
        """
        endpoint = "/openApi/swap/v2/user/income"
        params = {
            "symbol": symbol,
            "limit": limit,
        }
        
        if income_type:
            params["incomeType"] = income_type
        if start_time_ms:
            params["startTime"] = start_time_ms
        if end_time_ms:
            params["endTime"] = end_time_ms
        
        result = self._request("GET", endpoint, params)
        
        if result.get("code") != 0:
            return []
        
        return result.get("data") or []
    
    def get_last_closed_trade_pnl(self, symbol: str) -> Dict[str, float]:
        """
        Obtiene los datos del ÚLTIMO trade cerrado buscando el REALIZED_PNL más reciente.
        Suma los TRADING_FEE cercanos al cierre.
        También obtiene precios del historial de posiciones.
        
        Returns:
            Dict con:
            - realized_pnl: PnL del cierre (realisedProfit)
            - trading_fees: Fees de apertura + cierre
            - funding_fees: Funding fees
            - total: Suma de todo (netProfit)
            - close_timestamp: Timestamp del cierre
            - entry_price: Precio de entrada (del historial)
            - exit_price: Precio de cierre (del historial)
            - volume: Volumen operado
        """
        # Obtener los últimos 50 income entries para capturar fees
        entries = self.get_trade_income(symbol=symbol, limit=50)
        
        if not entries:
            return {"realized_pnl": 0, "trading_fees": 0, "funding_fees": 0, "total": 0, "close_timestamp": 0}
        
        # Buscar el REALIZED_PNL más reciente (indica cierre de posición)
        close_timestamp = None
        realized_pnl = 0.0
        
        for entry in entries:
            if entry.get("incomeType") == "REALIZED_PNL":
                close_timestamp = entry.get("time", 0)
                realized_pnl = float(entry.get("income", 0) or 0)
                break  # El primero es el más reciente
        
        if not close_timestamp:
            return {"realized_pnl": 0, "trading_fees": 0, "funding_fees": 0, "total": 0, "close_timestamp": 0}
        
        # Sumar TODOS los TRADING_FEE desde el cierre hacia atrás
        # El fee de cierre tiene el MISMO timestamp que REALIZED_PNL
        # El fee de apertura tiene un timestamp anterior (segundos a minutos antes)
        trading_fees = 0.0
        funding_fees = 0.0
        fees_found = 0
        
        for entry in entries:
            entry_time = entry.get("time", 0)
            entry_type = entry.get("incomeType", "")
            income = float(entry.get("income", 0) or 0)
            
            if entry_type == "TRADING_FEE":
                # Capturar los 2 fees más cercanos al cierre (apertura y cierre)
                if fees_found < 2:
                    trading_fees += income
                    fees_found += 1
            elif entry_type == "FUNDING_FEE":
                # Funding fees dentro de 5 minutos del cierre
                if entry_time >= close_timestamp - 300000 and entry_time <= close_timestamp:
                    funding_fees += income
        
        # También obtener precios del historial de posiciones
        # Buscar la posición cuyo updateTime coincida con el close_timestamp
        entry_price = 0.0
        exit_price = 0.0
        volume = 0.0
        
        import time
        end_time = int(time.time() * 1000)
        start_time = end_time - (24 * 60 * 60 * 1000)
        
        history = self.get_position_history(
            symbol=symbol,
            start_time_ms=start_time,
            end_time_ms=end_time,
            limit=10,
        )
        
        if history:
            # Buscar la posición cuyo updateTime esté cerca del close_timestamp
            best_match = None
            min_diff = float('inf')
            
            for pos in history:
                pos_update_time = int(pos.get("updateTime", 0) or 0)
                diff = abs(pos_update_time - close_timestamp)
                if diff < min_diff:
                    min_diff = diff
                    best_match = pos
            
            # Si encontramos una posición con updateTime cercano (< 5 segundos)
            if best_match and min_diff < 5000:
                entry_price = float(best_match.get("avgPrice", 0) or 0)
                exit_price = float(best_match.get("avgClosePrice", 0) or 0)
                volume = float(best_match.get("closePositionAmt", 0) or best_match.get("positionAmt", 0) or 0)
            elif history:
                # Fallback: usar la más reciente
                pos = history[0]
                entry_price = float(pos.get("avgPrice", 0) or 0)
                exit_price = float(pos.get("avgClosePrice", 0) or 0)
                volume = float(pos.get("closePositionAmt", 0) or pos.get("positionAmt", 0) or 0)
        
        return {
            "realized_pnl": realized_pnl,
            "trading_fees": trading_fees,
            "funding_fees": funding_fees,
            "total": realized_pnl + trading_fees + funding_fees,
            "close_timestamp": close_timestamp,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "volume": volume,
        }
    
    def get_order_details(self, symbol: str, order_id: str) -> Optional[Dict]:
        """
        Obtiene los detalles de una orden específica por su ID.
        
        Returns:
            Dict con profit, commission, avgPrice, etc. o None si no se encuentra
        """
        result = self._request('GET', '/openApi/swap/v2/trade/order', {
            'symbol': symbol,
            'orderId': order_id,
        })
        
        if result.get("code") != 0:
            return None
        
        return result.get("data", {}).get("order")
    
    def get_trade_fees_since(
        self,
        symbol: str,
        since_timestamp_ms: int,
    ) -> Dict[str, float]:
        """
        Obtiene TODOS los datos de un trade desde su apertura hasta ahora.
        Usa el timestamp de apertura para filtrar income entries.
        
        Args:
            symbol: Símbolo del par
            since_timestamp_ms: Timestamp de apertura del trade
            
        Returns:
            Dict con datos REALES del exchange:
            - realized_pnl: PnL cerrado (del REALIZED_PNL entry)
            - trading_fees: Comisiones totales (suma de TRADING_FEE)
            - funding_fees: Fees de financiación
            - total: PnL neto (pnl + fees)
        """
        import time
        
        # Obtener todos los income entries desde la apertura
        entries = self.get_trade_income(
            symbol=symbol,
            start_time_ms=since_timestamp_ms,
            end_time_ms=int(time.time() * 1000),
            limit=100,
        )
        
        trading_fees = 0.0
        funding_fees = 0.0
        realized_pnl = 0.0
        
        for entry in (entries or []):
            income_type = entry.get("incomeType", "")
            income = float(entry.get("income", 0) or 0)
            
            if income_type == "TRADING_FEE":
                trading_fees += income
            elif income_type == "FUNDING_FEE":
                funding_fees += income
            elif income_type == "REALIZED_PNL":
                realized_pnl += income
        
        return {
            "realized_pnl": realized_pnl,
            "trading_fees": trading_fees,
            "funding_fees": funding_fees,
            "total": realized_pnl + trading_fees + funding_fees,
        }
    
    def query_order_until_filled(
        self,
        symbol: str,
        order_id: str,
        max_attempts: int = 10,
        wait_seconds: float = 0.5,
    ) -> Optional[Dict[str, Any]]:
        """
        Consulta una orden por ID en bucle hasta que status=FILLED.
        
        Endpoint: GET /openApi/swap/v2/trade/order
        
        Args:
            symbol: Símbolo del par
            order_id: ID de la orden
            max_attempts: Intentos máximos
            wait_seconds: Segundos entre intentos
            
        Returns:
            Dict con datos REALES de la orden ejecutada:
            - orderId: ID de la orden
            - status: "FILLED"
            - avgPrice: Precio promedio REAL de ejecución (incluye slippage)
            - executedQty: Cantidad real ejecutada
            - cumFee: Comisión REAL cobrada (negativo)
            - cumQuote: Valor real en USDT operado
            - profit: PnL realizado (solo para órdenes de cierre)
            - side: BUY/SELL
            - positionSide: LONG/SHORT
        """
        for attempt in range(max_attempts):
            order = self.get_order_details(symbol, order_id)
            if order and order.get("status") == "FILLED":
                return order
            time.sleep(wait_seconds)
        
        # Último intento - devolver lo que haya
        return self.get_order_details(symbol, order_id)
    
    def find_close_order_id(
        self,
        symbol: str,
        position_side: str,
        after_timestamp_ms: Optional[int] = None,
    ) -> Optional[str]:
        """
        Busca el orderId de la orden de cierre más reciente para un lado.
        Útil cuando la posición fue cerrada por SL/TP o manualmente.
        
        Busca en allOrders la orden FILLED más reciente del lado contrario.
        - Para cerrar LONG → busca SELL+LONG con status FILLED
        - Para cerrar SHORT → busca BUY+SHORT con status FILLED
        
        Args:
            symbol: Símbolo del par
            position_side: "LONG" o "SHORT"
            after_timestamp_ms: Solo buscar órdenes después de este timestamp
            
        Returns:
            orderId de la orden de cierre, o None
        """
        result = self._request('GET', '/openApi/swap/v2/trade/allOrders', {
            'symbol': symbol,
            'limit': 50,
        })
        
        orders = result.get('data', {}).get('orders', [])
        close_side = "SELL" if position_side == "LONG" else "BUY"
        
        for order in orders:
            if (order.get("positionSide") == position_side and
                order.get("side") == close_side and
                order.get("status") == "FILLED"):
                # Si hay filtro de timestamp, verificar
                if after_timestamp_ms:
                    order_time = int(order.get("updateTime", 0) or order.get("time", 0) or 0)
                    if order_time < after_timestamp_ms:
                        continue
                return str(order.get("orderId", ""))
        
        return None
    
    def get_real_close_data(
        self,
        symbol: str,
        position_side: str,
        open_order_id: Optional[str] = None,
        open_timestamp_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        MÉTODO PRINCIPAL: Obtiene datos REALES y EXACTOS de cierre de un trade.
        
        Estrategia simplificada y robusta:
        1. Obtener positionID de la orden de apertura (open_order_id)
        2. Buscar en allOrders TODAS las órdenes con ese positionID
           → orden de apertura (BUY): avgPrice, commission
           → orden de cierre (SELL FILLED): avgPrice, commission, profit
        3. Complementar con positionHistory (match por positionId)
           → netProfit, totalFunding, positionCommission (verdad absoluta)
        
        Si no encontramos por positionID, usamos positionHistory[0] como fallback.
        """
        result = {
            "entry_price": 0.0, "exit_price": 0.0, "volume": 0.0,
            "opening_fee": 0.0, "closing_fee": 0.0, "trading_fees": 0.0,
            "funding_fees": 0.0, "realized_pnl": 0.0, "total_pnl": 0.0,
            "close_order_id": None,
        }
        
        # ══════════════════════════════════════════════════════════════
        # PASO 1: Obtener positionID de la orden de apertura
        # ══════════════════════════════════════════════════════════════
        position_id = None
        
        if open_order_id:
            open_order = self.get_order_details(symbol, open_order_id)
            if open_order:
                position_id = str(open_order.get("positionID", "") or "")
                result["entry_price"] = float(open_order.get("avgPrice", 0) or 0)
                result["opening_fee"] = float(open_order.get("commission", 0) or 0)
        
        # ══════════════════════════════════════════════════════════════
        # PASO 2: Buscar órdenes por positionID en allOrders
        # ══════════════════════════════════════════════════════════════
        if position_id:
            # Buscar con limit alto para cubrir activos con muchas ordenes (BTC)
            all_orders_result = self._request('GET', '/openApi/swap/v2/trade/allOrders', {
                'symbol': symbol,
                'limit': 500,
            })
            all_orders = all_orders_result.get('data', {}).get('orders', [])
            
            for order in all_orders:
                if str(order.get("positionID", "")) != position_id:
                    continue
                
                status = order.get("status")
                side = order.get("side")
                close_side = "SELL" if position_side == "LONG" else "BUY"
                
                # Orden de cierre FILLED
                if side == close_side and status == "FILLED":
                    result["exit_price"] = float(order.get("avgPrice", 0) or 0)
                    result["volume"] = abs(float(order.get("executedQty", 0) or 0))
                    result["closing_fee"] = float(order.get("commission", 0) or 0)
                    result["realized_pnl"] = float(order.get("profit", 0) or 0)
                    result["close_order_id"] = str(order.get("orderId", ""))
                
                # Orden de apertura FILLED (refinar datos)
                open_side = "BUY" if position_side == "LONG" else "SELL"
                if side == open_side and status == "FILLED":
                    entry_p = float(order.get("avgPrice", 0) or 0)
                    if entry_p > 0:
                        result["entry_price"] = entry_p
                    opening_f = float(order.get("commission", 0) or 0)
                    if opening_f != 0:
                        result["opening_fee"] = opening_f
        
        # Trading fees = apertura + cierre (de las órdenes individuales)
        order_fees = result["opening_fee"] + result["closing_fee"]
        
        # ══════════════════════════════════════════════════════════════
        # PASO 3: Complementar con positionHistory
        # Estrategia: positionHistory con match exacto es verdad absoluta.
        # Si no hay match exacto, usar fallback para COMPLEMENTAR
        # lo que falte, pero NUNCA sobreescribir datos validos de ordenes.
        # ══════════════════════════════════════════════════════════════
        history = self.get_position_history(symbol=symbol, limit=20)
        history_match = None
        exact_match = False
        
        if history:
            # Buscar por positionId exacto (verdad absoluta)
            if position_id:
                for pos in history:
                    if str(pos.get("positionId", "")) == position_id:
                        history_match = pos
                        exact_match = True
                        break
            
            # Fallback: posicion mas reciente del mismo lado
            if not history_match:
                for pos in history:
                    if pos.get("positionSide") == position_side:
                        history_match = pos
                        break
                # Ultimo fallback
                if not history_match:
                    history_match = history[0]
        
        if history_match:
            h_entry = float(history_match.get("avgPrice", 0) or 0)
            h_exit = float(history_match.get("avgClosePrice", 0) or 0)
            h_volume = abs(float(history_match.get("closePositionAmt", 0) or history_match.get("positionAmt", 0) or 0))
            h_pnl = float(history_match.get("realisedProfit", 0) or 0)
            h_fees = float(history_match.get("positionCommission", 0) or 0)
            h_funding = float(history_match.get("totalFunding", 0) or 0)
            h_net = float(history_match.get("netProfit", 0) or 0)
            
            if exact_match:
                # ── Match exacto: positionHistory es la VERDAD ABSOLUTA ──
                # Usar todos sus datos, sobreescribir todo
                if h_entry > 0:
                    result["entry_price"] = h_entry
                if h_exit > 0:
                    result["exit_price"] = h_exit
                if h_volume > 0:
                    result["volume"] = h_volume
                result["realized_pnl"] = h_pnl
                result["trading_fees"] = h_fees
                result["funding_fees"] = h_funding
                result["total_pnl"] = h_net
            else:
                # ── Fallback (no exacto): solo COMPLEMENTAR lo que falte ──
                # NUNCA sobreescribir datos que ya vinieron de allOrders
                if result["entry_price"] == 0 and h_entry > 0:
                    result["entry_price"] = h_entry
                if result["exit_price"] == 0 and h_exit > 0:
                    result["exit_price"] = h_exit
                if result["volume"] == 0 and h_volume > 0:
                    result["volume"] = h_volume
                if result["realized_pnl"] == 0:
                    result["realized_pnl"] = h_pnl
                
                # Fees: usar datos de ordenes si los tenemos, si no del fallback
                if order_fees != 0:
                    result["trading_fees"] = order_fees
                else:
                    result["trading_fees"] = h_fees
                
                result["funding_fees"] = h_funding
                result["total_pnl"] = result["realized_pnl"] + result["trading_fees"] + result["funding_fees"]
        else:
            # Sin positionHistory: calcular con datos de ordenes
            result["trading_fees"] = order_fees
            if result["total_pnl"] == 0:
                result["total_pnl"] = result["realized_pnl"] + order_fees
        
        return result
    
    def get_fill_history(
        self,
        symbol: str,
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
        order_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """
        Obtiene el historial de trades ejecutados (fills) con detalle de fees.
        
        Args:
            symbol: Símbolo del par
            start_time_ms: Timestamp de inicio
            end_time_ms: Timestamp de fin
            order_id: ID de orden específica
            limit: Número máximo de registros
            
        Returns:
            Lista de fills con:
            - symbol
            - orderId
            - tradeId
            - side (BUY/SELL)
            - positionSide (LONG/SHORT)
            - price
            - quantity
            - realizedPnl
            - commission (fee del trade)
            - filledTime
        """
        import time
        
        endpoint = "/openApi/swap/v2/trade/fillHistory"
        
        # Si no se pasan tiempos, usar últimos 7 días
        if end_time_ms is None:
            end_time_ms = int(time.time() * 1000)
        if start_time_ms is None:
            start_time_ms = end_time_ms - (7 * 24 * 60 * 60 * 1000)
        
        params = {
            "symbol": symbol,
            "startTs": start_time_ms,
            "endTs": end_time_ms,
            "pageSize": limit,
        }
        
        if order_id:
            params["orderId"] = order_id
        
        result = self._request("GET", endpoint, params)
        
        if result.get("code") != 0:
            return []
        
        return result.get("data", {}).get("list", [])
    
    def get_commission_rate(self) -> Dict[str, float]:
        """
        Obtiene las tasas de comisión del usuario.
        
        Returns:
            Dict con:
            - takerCommissionRate: Tasa taker (market orders)
            - makerCommissionRate: Tasa maker (limit orders)
        """
        endpoint = "/openApi/swap/v2/user/commissionRate"
        result = self._request("GET", endpoint)
        
        if result.get("code") != 0:
            return {"takerCommissionRate": 0.0005, "makerCommissionRate": 0.0002}  # Defaults
        
        data = result.get("data", {})
        return {
            "takerCommissionRate": float(data.get("takerCommissionRate", 0.0005) or 0.0005),
            "makerCommissionRate": float(data.get("makerCommissionRate", 0.0002) or 0.0002),
        }
    
    def get_position_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene detalles completos de una posición abierta.
        
        Args:
            symbol: Símbolo del par
            
        Returns:
            Dict con detalles de la posición o None si no hay posición:
            - positionSide: LONG o SHORT
            - positionAmt: Cantidad
            - avgPrice: Precio promedio de entrada
            - markPrice: Precio de marca actual
            - unrealizedProfit: PnL no realizado
            - liquidationPrice: Precio de liquidación
            - leverage: Apalancamiento
            - margin: Margen usado
            - positionId: ID de la posición
        """
        positions = self.get_positions(symbol)
        
        for pos in positions:
            pos_amt = float(pos.get("positionAmt", 0) or pos.get("availableAmt", 0) or 0)
            if abs(pos_amt) > 0:
                return {
                    "positionSide": pos.get("positionSide"),
                    "positionAmt": pos_amt,
                    "avgPrice": float(pos.get("avgPrice", 0) or 0),
                    "markPrice": float(pos.get("markPrice", 0) or pos.get("lastPrice", 0) or 0),
                    "unrealizedProfit": float(pos.get("unrealizedProfit", 0) or 0),
                    "liquidationPrice": float(pos.get("liquidationPrice", 0) or 0),
                    "leverage": int(pos.get("leverage", 1) or 1),
                    "margin": float(pos.get("margin", 0) or pos.get("initialMargin", 0) or 0),
                    "positionId": str(pos.get("positionId", "")),
                    "riskRate": float(pos.get("riskRate", 0) or 0),
                }
        
        return None

    def get_order_fill_details(self, symbol: str, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene los detalles de fills de una orden específica.
        Esto incluye el precio REAL de ejecución, cantidad y fees.
        
        Args:
            symbol: Símbolo del par
            order_id: ID de la orden
            
        Returns:
            Dict con:
            - avgPrice: Precio promedio de ejecución REAL
            - filledQty: Cantidad ejecutada
            - commission: Comisión total
            - fills: Lista de fills individuales
        """
        fills = self.get_fill_history(symbol=symbol, order_id=order_id, limit=10)
        
        if not fills:
            # Fallback: intentar obtener de order details
            order = self.get_order_details(symbol, order_id)
            if order:
                return {
                    "avgPrice": float(order.get("avgPrice", 0) or order.get("price", 0) or 0),
                    "filledQty": float(order.get("executedQty", 0) or order.get("quantity", 0) or 0),
                    "commission": float(order.get("commission", 0) or 0),
                    "fills": [],
                }
            return None
        
        # Calcular promedios ponderados de los fills
        total_qty = 0.0
        total_value = 0.0
        total_commission = 0.0
        
        for fill in fills:
            qty = float(fill.get("quantity", 0) or fill.get("qty", 0) or 0)
            price = float(fill.get("price", 0) or 0)
            commission = float(fill.get("commission", 0) or fill.get("fee", 0) or 0)
            
            total_qty += qty
            total_value += qty * price
            total_commission += abs(commission)  # Comisión siempre positiva para suma
        
        avg_price = total_value / total_qty if total_qty > 0 else 0
        
        return {
            "avgPrice": avg_price,
            "filledQty": total_qty,
            "commission": -total_commission,  # Negativo porque es un costo
            "fills": fills,
        }

    def get_closed_position_details(
        self,
        symbol: str,
        position_side: str,
        open_time_ms: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Obtiene los detalles de una posición CERRADA del historial.
        Usa el endpoint de historial de posiciones que tiene TODOS los datos reales.
        
        Args:
            symbol: Símbolo del par
            position_side: "LONG" o "SHORT"
            open_time_ms: Timestamp de apertura para identificar la posición
            
        Returns:
            Dict con datos REALES del exchange:
            - openAvgPrice: Precio de entrada real
            - closeAvgPrice: Precio de cierre real
            - closedVolume: Volumen operado
            - pnl: PnL realizado (PnL cerrado)
            - tradeFee: Comisiones de trading (negativo)
            - fundingFee: Funding fees
            - totalPnl: PnL neto (netProfit del exchange)
        """
        import time
        
        # Buscar en historial reciente (últimas 24h)
        end_time = int(time.time() * 1000)
        start_time = end_time - (24 * 60 * 60 * 1000)  # 24 horas
        
        history = self.get_position_history(
            symbol=symbol,
            start_time_ms=start_time,
            end_time_ms=end_time,
            limit=20,
        )
        
        if not history:
            return None
        
        # Buscar la posición más reciente que coincida con el lado
        for pos in history:
            if pos.get("positionSide") == position_side:
                # Verificar que el tiempo de apertura coincida (si se proporciona)
                if open_time_ms:
                    pos_open_time = int(pos.get("openTime", 0) or 0)
                    # Tolerancia de 5 minutos
                    if abs(pos_open_time - open_time_ms) > 300000:
                        continue
                
                # Mapear los campos REALES del API de BingX:
                # - avgPrice = precio de entrada
                # - avgClosePrice = precio de cierre
                # - realisedProfit = PnL cerrado (ganancia/pérdida por movimiento)
                # - positionCommission = comisiones totales (negativo)
                # - totalFunding = funding fees
                # - netProfit = realisedProfit + positionCommission + totalFunding
                # - positionAmt o closePositionAmt = volumen
                
                pnl = float(pos.get("realisedProfit", 0) or 0)
                trade_fee = float(pos.get("positionCommission", 0) or 0)
                funding_fee = float(pos.get("totalFunding", 0) or 0)
                net_profit = float(pos.get("netProfit", 0) or 0)
                
                # Si netProfit existe, usarlo directamente; si no, calcularlo
                total_pnl = net_profit if net_profit != 0 else (pnl + trade_fee + funding_fee)
                
                return {
                    "positionId": str(pos.get("positionId", "")),
                    "openAvgPrice": float(pos.get("avgPrice", 0) or 0),
                    "closeAvgPrice": float(pos.get("avgClosePrice", 0) or 0),
                    "closedVolume": float(pos.get("closePositionAmt", 0) or pos.get("positionAmt", 0) or 0),
                    "pnl": pnl,  # realisedProfit = PnL cerrado
                    "tradeFee": trade_fee,  # positionCommission (ya negativo)
                    "fundingFee": funding_fee,  # totalFunding
                    "totalPnl": total_pnl,  # netProfit del exchange
                    "openTime": pos.get("openTime"),
                    "closeTime": pos.get("updateTime"),
                    "raw": pos,  # Para debug
                }
        
        return None