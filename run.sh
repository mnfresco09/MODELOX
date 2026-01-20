#!/bin/bash
# Este script ejecuta cualquier comando de python dentro de un entorno Nix puro y funcional
# Uso: ./run.sh script.py

nix-shell -p \
  python311 \
  python311Packages.virtualenv \
  python311Packages.pip \
  stdenv.cc.cc.lib \
  zlib \
  --run "
    # Configurar librerías
    export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib
    
    # Activar entorno virtual
    if [ ! -d 'venv' ]; then
       python3 -m venv venv
       source venv/bin/activate
       pip install -r app/backend/requirements.txt
       pip install matplotlib seaborn
    else
       source venv/bin/activate
    fi
    
    # Ejecutar el script pasado como argumento
    python $1
  "