#!/usr/bin/env python
"""
Wrapper para ejecutar gexcel desde la raíz del proyecto.
Convierte archivos CSV de resultados a Excel con formato Dashboard MODELOX.

USO:
    python gexcel.py
    (Luego arrastrar el CSV cuando se solicite)
"""

import sys
from pathlib import Path

# Asegurar que el directorio raíz esté en el path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from visual.gexcel import main

if __name__ == "__main__":
    main()
