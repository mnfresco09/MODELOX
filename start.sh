#!/bin/bash
# Buscar la librería libstdc++.so.6 y añadirla al LD_LIBRARY_PATH
LIB_PATH=$(dirname $(find /nix/store -name "libstdc++.so.6" -print -quit))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIB_PATH

# Activar entorno y ejecutar
source venv/bin/activate
python ejecutar.py