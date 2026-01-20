#!/bin/bash
echo "Configurando entorno..."

# Buscar librerías críticas
STDC_PATH=$(find /nix/store -name libstdc++.so.6 -print -quit 2>/dev/null | xargs dirname)
ZLIB_PATH=$(find /nix/store -name libz.so.1 -print -quit 2>/dev/null | xargs dirname)

if [ -z "$STDC_PATH" ] || [ -z "$ZLIB_PATH" ]; then
    echo "⚠️  Advertencia: No se pudieron encontrar algunas librerías automáticamente."
else
    export LD_LIBRARY_PATH="$STDC_PATH:$ZLIB_PATH:$LD_LIBRARY_PATH"
    echo "✅ Librerías configuradas correctamente."
fi

# Activar venv si existe
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Ejecutar el comando que el usuario quiera, o abrir una shell
if [ "$#" -gt 0 ]; then
    exec "$@"
else
    echo "Entorno listo. Puedes ejecutar 'python analisis.py' ahora."
fi