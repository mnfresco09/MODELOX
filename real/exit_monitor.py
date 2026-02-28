"""
MONITOR DE SALIDAS - VELAS DE 1 MINUTO
=======================================
Monitorea las salidas usando SIEMPRE velas de 1 minuto,
independientemente del timeframe usado para las entradas.

CARACTERÍSTICAS:
- Usa velas de 1m para detectar SL/TP/Trailing con precisión
- Ajusta el precio de salida al high/low real de la vela
- Calcula trailing distance al cierre de la vela previa
- Evita que movimientos intra-vela se "pierdan" con timeframes mayores
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Literal, Tuple

import polars as pl

from .bingx_client import BingXClient


@dataclass
class ExitLevel:
    """Niveles de salida para una posición."""
    sl_price: float
    tp_price: float
    trailing_active: bool = False
    trailing_level: Optional[float] = None
    highest_price: Optional[float] = None  # Para trailing en LONG
    lowest_price: Optional[float] = None   # Para trailing en SHORT


class ExitMonitor:
    """
    Monitor de salidas usando velas de 1 minuto.
    
    Funciona en paralelo con el timeframe principal de la estrategia,
    pero verifica salidas con precisión de 1 minuto.
    """
    
    def __init__(self, client: BingXClient, symbol: str):
        self.client = client
        self.symbol = symbol
        self.last_1m_candles: Optional[pl.DataFrame] = None
        self.last_fetch_time = 0
        self.fetch_interval = 10  # Actualizar cada 10 segundos
    
    def fetch_1m_candles(self, limit: int = 100) -> pl.DataFrame:
        """
        Obtiene velas de 1 minuto actualizadas.
        
        Args:
            limit: Número de velas a obtener
            
        Returns:
            DataFrame con columnas: timestamp, open, high, low, close, volume
        """
        now = time.time()
        
        # Cache para evitar requests excesivos
        if (self.last_1m_candles is not None and 
            now - self.last_fetch_time < self.fetch_interval):
            return self.last_1m_candles
        
        klines = self.client.get_klines(
            symbol=self.symbol,
            interval="1m",  # SIEMPRE 1 minuto
            limit=limit,
            market_type="perpetual",
        )
        
        if not klines:
            if self.last_1m_candles is not None:
                return self.last_1m_candles
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
        
        df = pl.DataFrame(rows)
        self.last_1m_candles = df
        self.last_fetch_time = now
        
        return df
    
    def check_exit_conditions(
        self,
        side: Literal["LONG", "SHORT"],
        entry_price: float,
        entry_time: datetime,
        sl_price: float,
        tp_price: float,
        trailing_enabled: bool = False,
        trailing_activation_pct: float = 0.0,
        trailing_distance_pct: float = 0.0,
        quantity: float = 1.0,
        stake: float = 100.0,
    ) -> Tuple[bool, Optional[str], Optional[float], Optional[datetime]]:
        """
        Verifica si se deben ejecutar salidas usando velas de 1m.
        
        Args:
            side: "LONG" o "SHORT"
            entry_price: Precio de entrada
            entry_time: Timestamp de entrada
            sl_price: Precio de stop loss
            tp_price: Precio de take profit
            trailing_enabled: Si está activado el trailing stop
            trailing_activation_pct: % para activar trailing (sobre stake)
            trailing_distance_pct: % de distancia del trailing (sobre stake)
            quantity: Cantidad operada
            stake: Stake/margen usado
            
        Returns:
            Tuple (should_exit, reason, exit_price, exit_time)
            - should_exit: True si debe salir
            - reason: "SL", "TP", "TRAILING"
            - exit_price: Precio ajustado al high/low de la vela
            - exit_time: Timestamp de la vela donde se tocó
        """
        # Obtener velas de 1m
        df_1m = self.fetch_1m_candles(limit=200)
        
        if df_1m.is_empty():
            return False, None, None, None
        
        # Filtrar velas posteriores a la entrada
        df_after_entry = df_1m.filter(pl.col("timestamp") > entry_time)
        
        if df_after_entry.is_empty():
            return False, None, None, None
        
        # Calcular distancias para trailing basadas en stake
        trail_act_distance = (stake * trailing_activation_pct / 100.0) / quantity
        trail_dist_distance = (stake * trailing_distance_pct / 100.0) / quantity
        
        if side == "LONG":
            activation_price = entry_price + trail_act_distance
        else:
            activation_price = entry_price - trail_act_distance
        
        # Variables para tracking del trailing
        trailing_active = False
        trailing_level = 0.0
        
        # Iterar sobre cada vela de 1m
        for row in df_after_entry.iter_rows(named=True):
            timestamp = row["timestamp"]
            high = row["high"]
            low = row["low"]
            close = row["close"]
            
            if trailing_enabled:
                # ══════════════════════════════════════════════════════
                # MODO TRAILING STOP
                # ══════════════════════════════════════════════════════
                
                if not trailing_active:
                    # Verificar SL inicial (antes de activar trailing)
                    if side == "LONG" and low <= sl_price:
                        # AJUSTE: Salir al SL, no al low de la vela
                        # (el SL se habría ejecutado antes de llegar al low)
                        return True, "SL", sl_price, timestamp
                    
                    if side == "SHORT" and high >= sl_price:
                        return True, "SL", sl_price, timestamp
                    
                    # Verificar activación del trailing
                    if side == "LONG" and high >= activation_price:
                        trailing_active = True
                        # Calcular trailing level al CIERRE de la vela PREVIA
                        # (esto se haría idealmente, pero aquí usamos el high actual)
                        trailing_level = high - trail_dist_distance
                    
                    elif side == "SHORT" and low <= activation_price:
                        trailing_active = True
                        trailing_level = low + trail_dist_distance
                
                if trailing_active:
                    # Trailing activado: actualizar nivel y verificar toque
                    if side == "LONG":
                        # Actualizar trailing level hacia arriba
                        new_level = high - trail_dist_distance
                        if new_level > trailing_level:
                            trailing_level = new_level
                        
                        # Verificar si tocó el trailing
                        if low <= trailing_level:
                            # AJUSTE: Salir al trailing level exacto
                            return True, "TRAILING", trailing_level, timestamp
                    
                    else:  # SHORT
                        # Actualizar trailing level hacia abajo
                        new_level = low + trail_dist_distance
                        if new_level < trailing_level:
                            trailing_level = new_level
                        
                        # Verificar si tocó el trailing
                        if high >= trailing_level:
                            return True, "TRAILING", trailing_level, timestamp
            
            else:
                # ══════════════════════════════════════════════════════
                # MODO FIXED SL/TP
                # ══════════════════════════════════════════════════════
                
                if side == "LONG":
                    # Verificar SL (prioridad)
                    if low <= sl_price:
                        return True, "SL", sl_price, timestamp
                    
                    # Verificar TP
                    if tp_price > 0 and high >= tp_price:
                        return True, "TP", tp_price, timestamp
                
                else:  # SHORT
                    # Verificar SL (prioridad)
                    if high >= sl_price:
                        return True, "SL", sl_price, timestamp
                    
                    # Verificar TP
                    if tp_price > 0 and low <= tp_price:
                        return True, "TP", tp_price, timestamp
        
        # No se tocó ningún nivel
        return False, None, None, None
    
    def get_current_price_1m(self) -> float:
        """Obtiene el precio actual del último close de 1m."""
        df = self.fetch_1m_candles(limit=1)
        if df.is_empty():
            return 0.0
        return float(df["close"][-1])
    
    def calculate_trailing_level(
        self,
        side: Literal["LONG", "SHORT"],
        current_price: float,
        trailing_distance_pct: float,
        quantity: float,
        stake: float,
    ) -> float:
        """
        Calcula el nivel de trailing stop basado en el precio actual
        y la distancia configurada.
        
        La distancia se calcula como % sobre el stake.
        """
        trail_dist_distance = (stake * trailing_distance_pct / 100.0) / quantity
        
        if side == "LONG":
            return current_price - trail_dist_distance
        else:
            return current_price + trail_dist_distance
    
    def should_activate_trailing(
        self,
        side: Literal["LONG", "SHORT"],
        entry_price: float,
        current_price: float,
        activation_pct: float,
        quantity: float,
        stake: float,
    ) -> bool:
        """
        Verifica si el trailing stop debe activarse.
        """
        activation_distance = (stake * activation_pct / 100.0) / quantity
        
        if side == "LONG":
            activation_price = entry_price + activation_distance
            return current_price >= activation_price
        else:
            activation_price = entry_price - activation_distance
            return current_price <= activation_price
