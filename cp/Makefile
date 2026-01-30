# =============================================================================
# MODELOX Nuclear Engine - Makefile
# =============================================================================
# 
# Comandos disponibles:
#   make build    - Compila las extensiones C
#   make clean    - Limpia archivos de compilación
#   make rebuild  - Limpia y recompila
#   make test     - Ejecuta tests de rendimiento
#   make install  - Instala dependencias y compila
#
# =============================================================================

PYTHON := python3
PIP := pip3

.PHONY: build clean rebuild test install help

# Compilar extensiones
build:
	@echo "🔨 Compilando extensiones C..."
	$(PYTHON) setup.py build_ext --inplace
	@echo "✅ Compilación completada"

# Limpiar archivos de compilación
clean:
	@echo "🧹 Limpiando archivos de compilación..."
	rm -rf build/
	rm -f nuclear_engine.c
	rm -f *.so
	rm -f *.pyd
	rm -rf __pycache__/
	rm -rf *.egg-info/
	@echo "✅ Limpieza completada"

# Recompilar desde cero
rebuild: clean build

# Instalar dependencias
install:
	@echo "📦 Instalando dependencias..."
	$(PIP) install cython numpy
	@echo "🔨 Compilando..."
	$(PYTHON) setup.py build_ext --inplace
	@echo "✅ Instalación completada"

# Test de rendimiento
test:
	@echo "🧪 Ejecutando test de rendimiento..."
	cd .. && $(PYTHON) cp/benchmark.py

# Ayuda
help:
	@echo "MODELOX Nuclear Engine - Comandos disponibles:"
	@echo ""
	@echo "  make build    - Compila las extensiones C"
	@echo "  make clean    - Limpia archivos de compilación"
	@echo "  make rebuild  - Limpia y recompila"
	@echo "  make test     - Ejecuta tests de rendimiento"
	@echo "  make install  - Instala dependencias y compila"
	@echo ""

# Default
all: build
