"""
CONFIGURACION DE TRADING REAL - BINGX
======================================
TODA la configuracion de trading se define aqui.
Solo el ACTIVO y la ESTRATEGIA se eligen al ejecutar main.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, Optional

# =============================================================================
# CREDENCIALES API - BINGX
# =============================================================================

BINGX_API_KEY = os.getenv(
    "BINGX_API_KEY",
    "Hgcv3nBcgFbVveapMzJO7MLrwjN9Flaeki7r0BjEZcbOPo8xNSwh3iGlu6LIfmNXC5v43gFBuhR9Znkh6Wbg"
)

BINGX_SECRET_KEY = os.getenv(
    "BINGX_SECRET_KEY",
    "CNUL0N7YK0har921FK6yWQTXXcNTDkc9TBJIkP2wk84NTLAmidlg6pXwoJa6VoZ9bAVD62dqNhhE9GL0RyQhA"
)

# =============================================================================
# ENDPOINTS DE BINGX
# =============================================================================

BINGX_BASE_URL_DEMO = "https://open-api-vst.bingx.com"
BINGX_BASE_URL_LIVE = "https://open-api.bingx.com"


# =============================================================================
# CONFIGURACION DE TRADING
# =============================================================================
# TODOS los parametros se configuran aqui.
# Al ejecutar main.py solo se elige ACTIVO y ESTRATEGIA.
# =============================================================================

# MONEDA: "USDT" = DINERO REAL  |  "VST" = DEMO (DINERO VIRTUAL)
MONEDA = "VST"

# APALANCAMIENTO: De 1x a 150x segun el activo
APALANCAMIENTO = 150

# MONTO POR TRADE: Saldo que se usa para abrir cada posicion (en USD)
MONTO_POR_TRADE = 1000.0

# STOP LOSS: Porcentaje sobre el MARGEN. Ej: 10% con 150x = cierra si pierde 10% del margen
STOP_LOSS_PCT = 10.0

# TAKE PROFIT: Porcentaje sobre el MARGEN. Ej: 20% con 150x = cierra si gana 20% del margen
TAKE_PROFIT_PCT = 20.0

# TRAILING STOP: Sigue el precio para maximizar ganancias
TRAILING_STOP_ACTIVADO = False
TRAILING_STOP_ACTIVACION_PCT = 2.0   # Se activa cuando hay +2% de ganancia
TRAILING_STOP_DISTANCIA_PCT = 1.0    # Sigue a 1% de distancia del maximo

# TIMEFRAME: Temporalidad de las velas para señales (1m, 5m, 15m, 1h, 4h)
TIMEFRAME = "1m"

# MODO DE EJECUCION:
#   "manual" = ABRE TRADE INMEDIATAMENTE AL INICIAR
#   "auto"   = ESPERA SEÑALES DE LA ESTRATEGIA PARA ABRIR
MODO_EJECUCION = "manual"

# DIRECCION DEL TRADE (SOLO PARA MODO MANUAL):
#   "LONG"  = COMPRA (apuesta a que sube)
#   "SHORT" = VENTA (apuesta a que baja)
DIRECCION = "LONG"


# =============================================================================
# ACTIVOS DISPONIBLES
# =============================================================================
# Solo estos 6 activos estan disponibles para operar.
# Se seleccionan por numero al ejecutar main.py.
# =============================================================================

ACTIVOS_DISPONIBLES = {
    1: {"nombre": "BTC",    "symbol": "BTC-USDT"},
    2: {"nombre": "ETH",    "symbol": "ETH-USDT"},
    3: {"nombre": "GOLD",   "symbol": "NCCOGOLD2USD-USDT"},
    4: {"nombre": "NASDAQ", "symbol": "NCSINASDAQ1002USD-USDT"},
    5: {"nombre": "SP500",  "symbol": "NCSISP5002USD-USDT"},
    6: {"nombre": "SILVER", "symbol": "NCCOSILVER2USD-USDT"},
}


@dataclass
class TradingConfig:
    """Configuracion para una sesion de trading en vivo."""
    
    market_type: Literal["perpetual"] = "perpetual"
    symbol: str = "BTC-USDT"
    quote_currency: Literal["USDT", "VST"] = "VST"
    leverage: int = APALANCAMIENTO
    max_leverage: int = 150
    total_balance_to_use: float = MONTO_POR_TRADE
    amount_per_trade: float = MONTO_POR_TRADE
    stop_loss_pct: float = STOP_LOSS_PCT
    take_profit_pct: float = TAKE_PROFIT_PCT
    trailing_stop_enabled: bool = TRAILING_STOP_ACTIVADO
    trailing_stop_activation_pct: float = TRAILING_STOP_ACTIVACION_PCT
    trailing_stop_distance_pct: float = TRAILING_STOP_DISTANCIA_PCT
    strategy_id: int = 9
    timeframe: str = TIMEFRAME
    execution_mode: Literal["manual", "auto"] = MODO_EJECUCION

# Mapeo de nombres a simbolos BingX
SYMBOL_MAP = {
    "BTC": "BTC-USDT",
    "ETH": "ETH-USDT",
    "GOLD": "NCCOGOLD2USD-USDT",
    "NASDAQ": "NCSINASDAQ1002USD-USDT",
    "SP500": "NCSISP5002USD-USDT",
    "SILVER": "NCCOSILVER2USD-USDT",
}

# Nombres para mostrar al usuario
DISPLAY_NAMES = {
    "BTC-USDT": "BTC",
    "ETH-USDT": "ETH",
    "NCCOGOLD2USD-USDT": "GOLD",
    "NCSINASDAQ1002USD-USDT": "NASDAQ",
    "NCSISP5002USD-USDT": "SP500",
    "NCCOSILVER2USD-USDT": "SILVER",
}


def get_bingx_symbol(activo: str) -> str:
    """Convierte nombre de activo a simbolo BingX."""
    activo_upper = activo.upper().strip()
    if activo_upper in SYMBOL_MAP:
        return SYMBOL_MAP[activo_upper]
    if "-USDT" in activo_upper:
        return activo_upper
    return f"{activo_upper}-USDT"
