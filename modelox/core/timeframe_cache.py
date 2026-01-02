"""
Global Timeframe Cache

Cache global de DataFrames por timeframe y activo.
Evita recargas innecesarias cuando múltiples estrategias usan los mismos timeframes.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple, Optional

import polars as pl

from modelox.core.data import load_data

logger = logging.getLogger(__name__)


class GlobalTimeframeCache:
    """
    Cache global de DataFrames por (activo, timeframe).
    
    Features:
    - Thread-safe (single-threaded execution assumed)
    - Lazy loading (solo carga cuando se solicita)
    - Clear manual o automático
    
    Usage:
        cache = GlobalTimeframeCache.get_instance()
        df = cache.get_or_load("BTC", "1h", filepath)
    """
    
    _instance: Optional[GlobalTimeframeCache] = None
    _cache: Dict[Tuple[str, str], pl.DataFrame] = {}
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._cache = {}
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> GlobalTimeframeCache:
        """Retorna la instancia singleton."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get_or_load(
        self,
        activo: str,
        timeframe: str,
        filepath: str,
    ) -> pl.DataFrame:
        """
        Obtiene DataFrame del cache o lo carga si no existe.
        
        Args:
            activo: Nombre del activo (BTC, GOLD, etc.)
            timeframe: Timeframe (5m, 15m, 1h, etc.)
            filepath: Ruta al archivo de datos
        
        Returns:
            DataFrame de Polars
        """
        key = (activo.upper(), timeframe)
        
        if key in self._cache:
            logger.debug(f"Cache HIT: {activo} {timeframe}")
            return self._cache[key]
        
        logger.info(f"Cache MISS: Cargando {activo} {timeframe} desde {filepath}")
        df = load_data(filepath)
        self._cache[key] = df
        
        return df
    
    def get(self, activo: str, timeframe: str) -> Optional[pl.DataFrame]:
        """
        Obtiene DataFrame del cache sin cargar.
        
        Returns:
            DataFrame si existe, None si no está en cache
        """
        key = (activo.upper(), timeframe)
        return self._cache.get(key)
    
    def put(self, activo: str, timeframe: str, df: pl.DataFrame) -> None:
        """
        Añade DataFrame al cache manualmente.
        """
        key = (activo.upper(), timeframe)
        self._cache[key] = df
        logger.debug(f"Cache PUT: {activo} {timeframe}")
    
    def clear(self) -> None:
        """Limpia todo el cache."""
        n_entries = len(self._cache)
        self._cache.clear()
        logger.info(f"Cache limpiado: {n_entries} entries eliminadas")
    
    def clear_activo(self, activo: str) -> None:
        """Limpia todas las entradas de un activo."""
        activo_upper = activo.upper()
        keys_to_remove = [
            key for key in self._cache.keys()
            if key[0] == activo_upper
        ]
        
        for key in keys_to_remove:
            del self._cache[key]
        
        logger.debug(f"Cache limpiado para {activo}: {len(keys_to_remove)} entries")
    
    def get_stats(self) -> Dict[str, any]:
        """Retorna estadísticas del cache."""
        return {
            "entries": len(self._cache),
            "activos": len(set(k[0] for k in self._cache.keys())),
            "timeframes": len(set(k[1] for k in self._cache.keys())),
            "keys": list(self._cache.keys()),
        }


# Helper para acceso rápido
def get_timeframe_cache() -> GlobalTimeframeCache:
    """Retorna la instancia del cache global."""
    return GlobalTimeframeCache.get_instance()
