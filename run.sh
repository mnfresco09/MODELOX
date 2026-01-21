#!/bin/bash
# Obtener rutas reales de las librerías usando Nix
echo "Configurando entorno Nix..."
LIB_STDC=$(nix-build --no-out-link "<nixpkgs>" -A stdenv.cc.cc.lib)/lib
LIB_Z=$(nix-build --no-out-link "<nixpkgs>" -A zlib)/lib

# Configurar LD_LIBRARY_PATH para esta ejecución solamente
export LD_LIBRARY_PATH="$LIB_STDC:$LIB_Z:$LD_LIBRARY_PATH"

# Activar entorno virtual
source venv/bin/activate

# Ejecutar el script solicitado
echo "Ejecutando: $1"
python "$1"