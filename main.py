#!/usr/bin/env python3
"""
MODELOX - PUNTO DE ENTRADA PRINCIPAL
=====================================
Ejecuta con: python main.py

Opciones:
- Trading Real: Conecta con BingX para operar en vivo
- Backtesting: Ejecuta optimización de estrategias (ejecutar.py)
"""

import sys
import warnings
from pathlib import Path

# Suprimir warnings de SSL de urllib3
warnings.filterwarnings("ignore", category=Warning, module="urllib3")

# Asegurar que el directorio raíz está en el path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))


def main():
    """Punto de entrada principal."""
    # Importar aquí para evitar problemas de importación circular
    from real.main import main as real_main
    
    # Ejecutar el sistema de trading real
    real_main()


if __name__ == "__main__":
    main()
