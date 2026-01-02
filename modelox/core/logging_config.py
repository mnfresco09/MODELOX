"""
Logging Configuration for MODELOX

Configuración centralizada de logging con niveles apropiados.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    enable_optuna: bool = False,
) -> None:
    """
    Configura el sistema de logging de MODELOX.
    
    Args:
        level: Nivel de logging (logging.DEBUG, INFO, WARNING, ERROR)
        format_string: Formato personalizado (None usa el default)
        enable_optuna: Si True, habilita logs de Optuna (por defecto silenciado)
    """
    
    # Formato default
    if format_string is None:
        format_string = "[%(levelname)s] %(name)s: %(message)s"
    
    # Configurar root logger
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # Sobrescribe configuraciones previas
    )
    
    # Silenciar módulos ruidosos por defecto
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    
    # Optuna: por defecto silenciado
    if not enable_optuna:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            logging.getLogger("optuna").setLevel(logging.WARNING)
        except ImportError:
            pass
    
    # Reporters: por defecto WARNING
    logging.getLogger("modelox.reporting.excel_reporter").setLevel(logging.WARNING)
    logging.getLogger("modelox.reporting.plot_reporter").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Obtiene un logger configurado para un módulo.
    
    Usage:
        logger = get_logger(__name__)
        logger.info("Mensaje")
    """
    return logging.getLogger(name)
