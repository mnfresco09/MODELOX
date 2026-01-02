"""
Health Monitoring para MODELOX

Sistema de monitoreo de recursos (RAM, CPU) optimizado para Mac.
Evita sobre-threading y memory leaks durante optimizaciones largas.
"""

from __future__ import annotations

import gc
import logging
import os
import platform
import sys
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class HealthMonitor:
    """
    Monitoreo de salud del sistema durante ejecución de backtests.
    
    Features:
    - Límite configurable de RAM
    - Detección de sobre-threading en Mac
    - Cleanup automático (gc.collect)
    - Logging estructurado
    """
    
    enabled: bool = True
    ram_threshold_pct: float = 80.0  # Porcentaje máximo de RAM
    _system_info: Optional[str] = None
    
    def __post_init__(self):
        """Inicializa información del sistema."""
        if self.enabled:
            self._system_info = f"{platform.system()} {platform.release()}"
            logger.info(f"HealthMonitor iniciado en {self._system_info}")
            
            # Configurar límites de threads para Mac
            if platform.system() == "Darwin":  # macOS
                self._setup_mac_optimizations()
    
    def _setup_mac_optimizations(self) -> None:
        """
        Configuración específica para Mac.
        
        Razón: Python + Optuna + NumPy puede sobre-threadear en Mac,
        causando slowdowns por context switching y memory contention.
        """
        # Forzar single-threading en librerías numéricas
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        
        logger.info("Mac optimizations aplicadas: single-thread mode")
    
    def check_system_health(self) -> None:
        """
        Revisa estado del sistema y alerta si hay problemas.
        """
        if not self.enabled:
            return
        
        try:
            import psutil
        except ImportError:
            logger.warning("psutil no disponible, health check deshabilitado")
            self.enabled = False
            return
        
        # Chequear RAM
        memory = psutil.virtual_memory()
        ram_used_pct = memory.percent
        
        if ram_used_pct > self.ram_threshold_pct:
            logger.warning(
                f"⚠️  Uso de RAM alto: {ram_used_pct:.1f}% (límite: {self.ram_threshold_pct}%)"
            )
            logger.warning("Considera reducir N_TRIALS o activos simultáneos")
        else:
            logger.debug(f"RAM: {ram_used_pct:.1f}%")
        
        # Chequear CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        logger.debug(f"CPU: {cpu_percent:.1f}%")
    
    def force_cleanup(self, deep: bool = False) -> None:
        """
        Fuerza limpieza de memoria.
        
        Args:
            deep: Si True, ejecuta múltiples rondas de gc.collect
        """
        if not self.enabled:
            return
        
        if deep:
            # Limpieza profunda (3 rondas)
            for _ in range(3):
                gc.collect()
            logger.debug("Deep cleanup ejecutado")
        else:
            gc.collect()
            logger.debug("Cleanup ejecutado")
    
    def get_system_summary(self) -> dict[str, any]:
        """Retorna resumen del estado del sistema."""
        if not self.enabled:
            return {"enabled": False}
        
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()
            
            return {
                "enabled": True,
                "system": self._system_info,
                "ram_used_pct": memory.percent,
                "ram_available_gb": memory.available / (1024**3),
                "cpu_count": cpu_count,
                "cpu_percent": psutil.cpu_percent(interval=0.1),
            }
        except Exception as e:
            logger.error(f"Error obteniendo system summary: {e}")
            return {"enabled": False, "error": str(e)}


# Singleton global para fácil acceso
_global_monitor: Optional[HealthMonitor] = None


def get_health_monitor(
    enabled: bool = True,
    ram_threshold_pct: float = 80.0,
) -> HealthMonitor:
    """
    Retorna el monitor global (singleton).
    Si no existe, lo crea.
    """
    global _global_monitor
    
    if _global_monitor is None:
        _global_monitor = HealthMonitor(
            enabled=enabled,
            ram_threshold_pct=ram_threshold_pct,
        )
    
    return _global_monitor


# Alias para compatibilidad con código existente
class HealthGuard:
    """Clase de compatibilidad con el código existente."""
    
    @staticmethod
    def check_system_health(ram_threshold: float = 80.0) -> None:
        monitor = get_health_monitor(ram_threshold_pct=ram_threshold)
        monitor.check_system_health()
    
    @staticmethod
    def force_cleanup(deep: bool = False) -> None:
        monitor = get_health_monitor()
        monitor.force_cleanup(deep=deep)
    
    @staticmethod
    def setup_mac_limits() -> None:
        """Setup de límites de Mac (ahora automático en __post_init__)."""
        monitor = get_health_monitor()
        if platform.system() == "Darwin":
            monitor._setup_mac_optimizations()
